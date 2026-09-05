"""Import Polymarket outcome price histories into the event-first SQLite database.

This script combines data collection and loading. It reads outcomes from the
SQLite database, uses each outcome's CLOB token ID to request historical prices
from the Polymarket CLOB API, then writes the returned observations into the
normalised price_history table.

The CLOB token ID is used only for API retrieval. It is not stored in
price_history because price observations are linked to outcomes through the
composite key (market_id, outcome_index).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from api.polymarket_clob_api import fetch_clob_price_history  # noqa: E402
from api.http_session import create_session  # noqa: E402

DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "markets.db"

PRICE_HISTORY_INTERVAL = "max"
PRICE_HISTORY_FIDELITY = 720
DEFAULT_SLEEP_SECONDS = 0.02
DEFAULT_BATCH_SIZE = 100


@dataclass(frozen=True)
class OutcomeTarget:
    """Outcome that has enough information to request price history."""

    market_id: str
    outcome_index: int
    clob_token_id: str


@dataclass(frozen=True)
class PricePoint:
    """Single historical probability observation returned by the CLOB API."""

    timestamp: int
    price: float


def utc_now_text() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()


def timestamp_to_utc_text(timestamp: int) -> str:
    """Convert a Unix timestamp to an ISO-8601 UTC string."""

    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def connect_database(database_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection and enable foreign key enforcement."""

    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    ensure_price_attempt_schema(connection)
    return connection


def ensure_price_attempt_schema(connection: sqlite3.Connection) -> None:
    """Upgrade older databases so attempt rows can store CLOB token IDs."""

    columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(price_history_collection_attempts)"
        ).fetchall()
    }
    if "clob_token_id" not in columns:
        connection.execute(
            """
            ALTER TABLE price_history_collection_attempts
            ADD COLUMN clob_token_id TEXT
            """
        )
        connection.commit()


def fetch_price_history(
    *,
    clob_token_id: str,
    interval: str,
    fidelity: int,
    session: requests.Session,
) -> list[PricePoint]:
    """Fetch price history for one Polymarket CLOB token."""

    payload = fetch_clob_price_history(
        clob_token_id=clob_token_id,
        interval=interval,
        fidelity=fidelity,
        session=session,
    )
    history = payload["history"]

    price_points: list[PricePoint] = []
    for point in history:
        timestamp = point.get("t")
        price = point.get("p")

        if timestamp is None or price is None:
            continue

        price_points.append(
            PricePoint(
                timestamp=int(timestamp),
                price=float(price),
            )
        )

    return price_points


def get_outcome_targets(
    connection: sqlite3.Connection,
    limit: int | None,
    retry_failed: bool,
    market_ids: list[str] | None = None,
) -> list[OutcomeTarget]:
    """Return outcomes that should be attempted for price-history collection."""

    already_attempted_filter = ""
    if not retry_failed:
        already_attempted_filter = """
            AND NOT EXISTS (
                SELECT 1
                FROM price_history_collection_attempts phca
                WHERE phca.market_id = o.market_id
                  AND phca.outcome_index = o.outcome_index
            )
        """

    market_filter = ""
    params: list[Any] = []
    if market_ids:
        placeholders = ", ".join("?" for _ in market_ids)
        market_filter = f"AND o.market_id IN ({placeholders})"
        params.extend(market_ids)

    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT ?"
        params.append(limit)

    query = f"""
        SELECT
            o.market_id,
            o.outcome_index,
            o.clob_token_id
        FROM outcomes o
        JOIN markets m
            ON o.market_id = m.market_id
        WHERE o.clob_token_id IS NOT NULL
          AND o.clob_token_id != ''
          {market_filter}
          {already_attempted_filter}
        ORDER BY
            COALESCE(m.volume, 0) DESC,
            o.market_id,
            o.outcome_index
        {limit_clause}
    """

    rows = connection.execute(query, params).fetchall()
    return [
        OutcomeTarget(
            market_id=str(row[0]),
            outcome_index=int(row[1]),
            clob_token_id=str(row[2]),
        )
        for row in rows
    ]


def insert_price_points(
    connection: sqlite3.Connection,
    target: OutcomeTarget,
    price_points: list[PricePoint],
    collection_time_utc: str,
) -> int:
    """Insert price-history rows for one outcome and return inserted/updated count."""

    rows = [
        (
            target.market_id,
            target.outcome_index,
            point.timestamp,
            timestamp_to_utc_text(point.timestamp),
            point.price,
            collection_time_utc,
        )
        for point in price_points
    ]

    if not rows:
        return 0

    before_changes = connection.total_changes
    connection.executemany(
        """
        INSERT OR IGNORE INTO price_history (
            market_id,
            outcome_index,
            timestamp,
            datetime_utc,
            price,
            collection_time_utc
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return connection.total_changes - before_changes


def record_collection_attempt(
    connection: sqlite3.Connection,
    target: OutcomeTarget,
    attempted_at_utc: str,
    status: str,
    points_collected: int = 0,
    error_message: str | None = None,
) -> None:
    """Record the outcome-level collection attempt."""

    connection.execute(
        """
        INSERT INTO price_history_collection_attempts (
            market_id,
            outcome_index,
            clob_token_id,
            attempted_at_utc,
            status,
            points_collected,
            error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            target.market_id,
            target.outcome_index,
            target.clob_token_id,
            attempted_at_utc,
            status,
            points_collected,
            error_message,
        ),
    )


def import_price_history(
    database_path: Path,
    limit: int | None,
    sleep_seconds: float,
    batch_size: int,
    retry_failed: bool,
    interval: str,
    fidelity: int,
    market_ids: list[str] | None = None,
) -> None:
    """Collect and load price histories for selected outcomes."""

    connection = connect_database(database_path)
    api_session = create_session()

    try:
        targets = get_outcome_targets(
            connection=connection,
            limit=limit,
            retry_failed=retry_failed,
            market_ids=market_ids,
        )

        print(f"Database: {database_path}")
        print(f"Outcome targets selected: {len(targets):,}")
        print(f"Price-history interval: {interval}")
        print(f"Price-history fidelity: {fidelity} minutes")
        if market_ids:
            print(f"Market ID filter: {', '.join(market_ids)}")

        successes = 0
        empty_histories = 0
        failures = 0
        inserted_points = 0

        for index, target in enumerate(targets, start=1):
            attempted_at_utc = utc_now_text()

            try:
                history = fetch_price_history(
                    clob_token_id=target.clob_token_id,
                    interval=interval,
                    fidelity=fidelity,
                    session=api_session,
                )
                collection_time_utc = utc_now_text()
                inserted = insert_price_points(
                    connection=connection,
                    target=target,
                    price_points=history,
                    collection_time_utc=collection_time_utc,
                )

                status = "success" if history else "empty_history"
                record_collection_attempt(
                    connection=connection,
                    target=target,
                    attempted_at_utc=attempted_at_utc,
                    status=status,
                    points_collected=len(history),
                )

                if history:
                    successes += 1
                    inserted_points += inserted
                else:
                    empty_histories += 1

            except Exception as exc:  # noqa: BLE001 - collection should continue after failures
                failures += 1
                record_collection_attempt(
                    connection=connection,
                    target=target,
                    attempted_at_utc=attempted_at_utc,
                    status="error",
                    points_collected=0,
                    error_message=str(exc),
                )

            if index % batch_size == 0:
                connection.commit()
                print(
                    f"Processed {index:,}/{len(targets):,} outcomes | "
                    f"success={successes:,}, empty={empty_histories:,}, "
                    f"errors={failures:,}, inserted_points={inserted_points:,}"
                )

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        connection.commit()

        print("\nPrice-history import complete")
        print(f"Successful histories: {successes:,}")
        print(f"Empty histories: {empty_histories:,}")
        print(f"Failures: {failures:,}")
        print(f"Inserted price points: {inserted_points:,}")

    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Import Polymarket CLOB price histories into SQLite."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
        help="Path to the SQLite database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of outcomes to process. Useful for testing.",
    )
    parser.add_argument(
        "--market-id",
        action="append",
        default=None,
        help="Restrict collection to a specific market_id. Can be used multiple times.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Delay between API requests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of outcomes to process before committing progress.",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default=PRICE_HISTORY_INTERVAL,
        help="CLOB price-history interval parameter. Default is all.",
    )
    parser.add_argument(
        "--fidelity",
        type=int,
        default=PRICE_HISTORY_FIDELITY,
        help="Price-history sampling interval in minutes. Default is hourly.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry outcomes even if they already have a collection attempt recorded.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    import_price_history(
        database_path=args.database,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
        batch_size=args.batch_size,
        retry_failed=args.retry_failed,
        interval=args.interval,
        fidelity=args.fidelity,
        market_ids=args.market_id,
    )


if __name__ == "__main__":
    main()
