
"""
Load collected Polymarket event data into the event-first SQLite database.

The raw event collection file contains events from the Polymarket Gamma API.
Each event may contain many related binary markets, and each market normally
contains two outcomes: Yes and No.

This loader inserts:
- events into events
- event-level tags into tags
- event/tag relationships into event_tags
- nested binary markets into markets
- market outcomes into outcomes

Run from the project root:

    python3 scripts/stage_02_database/load_events_snapshot.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import json
import sqlite3
from typing import Any, Iterator

try:
    import ijson
except ModuleNotFoundError:  # pragma: no cover - exercised only without dependencies
    ijson = None


DATABASE_PATH = Path("data/database/markets.db")
RAW_EVENTS_DIR = Path("data/raw/polymarket/events")
RAW_EVENT_FILE = RAW_EVENTS_DIR / "polymarket_events_keyset_20260614_143725.json"
BATCH_SIZE = 5_000


JsonDict = dict[str, Any]


def utc_now() -> str:
    """Return the current UTC time as an ISO formatted string."""
    return datetime.now(timezone.utc).isoformat()


def load_json_string(value: Any) -> Any:
    """
    Parse API fields that are sometimes returned as JSON-encoded strings.

    Polymarket fields such as outcomes, outcomePrices and clobTokenIds often
    look like this in the raw JSON:

        "[\"Yes\", \"No\"]"

    This function converts them into normal Python lists. If parsing fails, the
    original value is returned so the loader does not crash on one unexpected
    field format.
    """
    if value is None:
        return None

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    return value


def to_int_bool(value: Any) -> int | None:
    """Convert boolean-like API values into SQLite-friendly integers."""
    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return 1
        if lowered in {"false", "0", "no"}:
            return 0

    return None


def to_float(value: Any) -> float | None:
    """Convert numeric API values into floats where possible."""
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_text(value: Any) -> str | None:
    """Convert a value to text, preserving None."""
    if value is None:
        return None
    return str(value)



def json_default(value: Any) -> Any:
    """Convert non-standard JSON values produced during streaming."""
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def raw_json(value: Any) -> str:
    """Serialise raw API content for auditability."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_default)


def event_raw_json_without_nested_markets(event: JsonDict) -> str:
    """
    Serialise event-level raw JSON without duplicating nested markets.

    The full raw event file is very large and each event can contain many nested
    markets. Markets are stored separately in the markets table, so removing the
    nested market list from the event audit JSON avoids duplicating a large
    amount of data inside SQLite.
    """
    event_copy = dict(event)
    event_copy.pop("markets", None)
    return raw_json(event_copy)



def iter_events_from_file(path: Path) -> Iterator[JsonDict]:
    """Stream events from a large raw event collection file one at a time."""
    if ijson is None:
        raise ModuleNotFoundError(
            "The ijson package is required to stream raw event files. "
            "Install project dependencies with `pip install -r requirements.txt`."
        )

    with path.open("rb") as file:
        for event in ijson.items(file, "events.item"):
            if isinstance(event, dict):
                yield event


def find_latest_raw_event_file(raw_events_dir: Path = RAW_EVENTS_DIR) -> Path:
    """Return the newest raw Polymarket event snapshot with event records."""
    if ijson is None:
        raise ModuleNotFoundError(
            "The ijson package is required to inspect raw event snapshots. "
            "Install project dependencies with `pip install -r requirements.txt`."
        )

    event_files = sorted(
        raw_events_dir.glob("polymarket_events_keyset_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for event_file in event_files:
        try:
            with event_file.open("rb") as file:
                first_event = next(ijson.items(file, "events.item"), None)
        except (OSError, ValueError):
            continue

        if isinstance(first_event, dict):
            return event_file

    raise FileNotFoundError(f"No non-empty raw event snapshots found in {raw_events_dir}")


def infer_resolved_winners(market: JsonDict, outcome_prices: list[Any]) -> tuple[list[int | None], str]:
    """
    Infer winning outcomes for resolved binary markets.

    For resolved Polymarket binary markets, final outcome prices are often 1/0.
    This function marks outcomes priced at 1 as winners and outcomes priced at 0
    as non-winners. If the market is not resolved, or the prices are ambiguous,
    it leaves the winner field as None.
    """
    outcome_count = len(outcome_prices)
    winners: list[int | None] = [None] * outcome_count

    resolution_status = market.get("umaResolutionStatus") or market.get("umaResolutionStatuses")
    closed = to_int_bool(market.get("closed"))

    numeric_prices = [to_float(price) for price in outcome_prices]

    if resolution_status == "resolved" or closed == 1:
        if all(price is not None for price in numeric_prices):
            if any(price == 1.0 for price in numeric_prices):
                winners = [1 if price == 1.0 else 0 for price in numeric_prices]
                return winners, "inferred_from_final_prices"

    return winners, "not_inferred"


def insert_event(cursor: sqlite3.Cursor, event: JsonDict, collection_time: str) -> None:
    """Insert one event row."""
    cursor.execute(
        """
        INSERT OR REPLACE INTO events (
            event_id,
            title,
            slug,
            description,
            active,
            closed,
            archived,
            volume,
            volume_24hr,
            volume_1wk,
            volume_1mo,
            volume_1yr,
            liquidity,
            liquidity_clob,
            open_interest,
            created_at,
            updated_at,
            start_date,
            end_date,
            closed_time,
            resolution_source,
            series_slug,
            collection_time_utc,
            raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            to_text(event.get("id")),
            event.get("title"),
            event.get("slug"),
            event.get("description"),
            to_int_bool(event.get("active")),
            to_int_bool(event.get("closed")),
            to_int_bool(event.get("archived")),
            to_float(event.get("volume")),
            to_float(event.get("volume24hr")),
            to_float(event.get("volume1wk")),
            to_float(event.get("volume1mo")),
            to_float(event.get("volume1yr")),
            to_float(event.get("liquidity")),
            to_float(event.get("liquidityClob")),
            to_float(event.get("openInterest")),
            event.get("createdAt"),
            event.get("updatedAt"),
            event.get("startDate"),
            event.get("endDate"),
            event.get("closedTime"),
            event.get("resolutionSource"),
            event.get("seriesSlug"),
            collection_time,
            event_raw_json_without_nested_markets(event),
        ),
    )


def insert_event_tags(cursor: sqlite3.Cursor, event: JsonDict) -> int:
    """Insert event tags and event-tag bridge rows."""
    event_id = to_text(event.get("id"))
    tags = event.get("tags") or []
    inserted_links = 0

    if not isinstance(tags, list):
        return inserted_links

    for tag in tags:
        if not isinstance(tag, dict):
            continue

        tag_id = to_text(tag.get("id"))
        if tag_id is None:
            continue

        cursor.execute(
            """
            INSERT OR REPLACE INTO tags (
                tag_id,
                label,
                slug,
                force_show,
                force_hide,
                is_carousel,
                requires_translation,
                created_at,
                updated_at,
                published_at,
                raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tag_id,
                tag.get("label"),
                tag.get("slug"),
                to_int_bool(tag.get("forceShow")),
                to_int_bool(tag.get("forceHide")),
                to_int_bool(tag.get("isCarousel")),
                to_int_bool(tag.get("requiresTranslation")),
                tag.get("createdAt"),
                tag.get("updatedAt"),
                tag.get("publishedAt"),
                raw_json(tag),
            ),
        )

        cursor.execute(
            """
            INSERT OR IGNORE INTO event_tags (event_id, tag_id)
            VALUES (?, ?)
            """,
            (event_id, tag_id),
        )
        inserted_links += 1

    return inserted_links


def insert_market(cursor: sqlite3.Cursor, event_id: str, market: JsonDict, collection_time: str) -> None:
    """Insert one market row."""
    cursor.execute(
        """
        INSERT OR REPLACE INTO markets (
            market_id,
            event_id,
            condition_id,
            question_id,
            question,
            slug,
            description,
            active,
            closed,
            archived,
            accepting_orders,
            approved,
            restricted,
            volume,
            volume_num,
            volume_clob,
            volume_24hr,
            volume_1wk,
            volume_1mo,
            volume_1yr,
            liquidity,
            liquidity_num,
            liquidity_clob,
            best_bid,
            best_ask,
            spread,
            last_trade_price,
            market_type,
            group_item_title,
            group_item_threshold,
            created_at,
            updated_at,
            start_date,
            end_date,
            closed_time,
            resolution_source,
            uma_resolution_status,
            collection_time_utc,
            raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            to_text(market.get("id")),
            event_id,
            market.get("conditionId"),
            market.get("questionID") or market.get("questionId"),
            market.get("question"),
            market.get("slug"),
            market.get("description"),
            to_int_bool(market.get("active")),
            to_int_bool(market.get("closed")),
            to_int_bool(market.get("archived")),
            to_int_bool(market.get("acceptingOrders")),
            to_int_bool(market.get("approved")),
            to_int_bool(market.get("restricted")),
            to_float(market.get("volume")),
            to_float(market.get("volumeNum")),
            to_float(market.get("volumeClob")),
            to_float(market.get("volume24hr")),
            to_float(market.get("volume1wk")),
            to_float(market.get("volume1mo")),
            to_float(market.get("volume1yr")),
            to_float(market.get("liquidity")),
            to_float(market.get("liquidityNum")),
            to_float(market.get("liquidityClob")),
            to_float(market.get("bestBid")),
            to_float(market.get("bestAsk")),
            to_float(market.get("spread")),
            to_float(market.get("lastTradePrice")),
            market.get("marketType"),
            market.get("groupItemTitle"),
            market.get("groupItemThreshold"),
            market.get("createdAt"),
            market.get("updatedAt"),
            market.get("startDate"),
            market.get("endDate"),
            market.get("closedTime"),
            market.get("resolutionSource"),
            market.get("umaResolutionStatus"),
            collection_time,
            raw_json(market),
        ),
    )


def insert_outcomes(cursor: sqlite3.Cursor, market: JsonDict) -> int:
    """Insert the outcomes for one binary market."""
    market_id = to_text(market.get("id"))
    if market_id is None:
        return 0

    outcome_names = load_json_string(market.get("outcomes")) or []
    outcome_prices = load_json_string(market.get("outcomePrices")) or []
    clob_token_ids = load_json_string(market.get("clobTokenIds")) or []

    if not isinstance(outcome_names, list):
        outcome_names = []
    if not isinstance(outcome_prices, list):
        outcome_prices = []
    if not isinstance(clob_token_ids, list):
        clob_token_ids = []

    outcome_count = max(len(outcome_names), len(outcome_prices), len(clob_token_ids))
    winners, resolution_status = infer_resolved_winners(market, outcome_prices)

    inserted = 0

    for outcome_index in range(outcome_count):
        outcome_name = outcome_names[outcome_index] if outcome_index < len(outcome_names) else None
        current_price = outcome_prices[outcome_index] if outcome_index < len(outcome_prices) else None
        clob_token_id = clob_token_ids[outcome_index] if outcome_index < len(clob_token_ids) else None
        resolved_winner = winners[outcome_index] if outcome_index < len(winners) else None

        cursor.execute(
            """
            INSERT OR REPLACE INTO outcomes (
                market_id,
                outcome_index,
                outcome_name,
                current_price,
                clob_token_id,
                resolved_winner,
                resolution_inference_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                market_id,
                outcome_index,
                outcome_name,
                to_float(current_price),
                to_text(clob_token_id),
                resolved_winner,
                resolution_status,
            ),
        )
        inserted += 1

    return inserted


def load_events(connection: sqlite3.Connection, events: Iterator[JsonDict]) -> None:
    """Load events, tags, markets and outcomes into the database."""
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("PRAGMA journal_mode = WAL")
    cursor.execute("PRAGMA synchronous = NORMAL")

    collection_time = utc_now()

    event_count = 0
    market_count = 0
    outcome_count = 0
    event_tag_count = 0
    skipped_markets = 0

    for event in events:
        if not isinstance(event, dict):
            continue

        event_id = to_text(event.get("id"))
        if event_id is None:
            continue

        insert_event(cursor, event, collection_time)
        event_count += 1

        event_tag_count += insert_event_tags(cursor, event)

        markets = event.get("markets") or []
        if not isinstance(markets, list):
            continue

        for market in markets:
            if not isinstance(market, dict):
                continue

            if market.get("id") is None:
                skipped_markets += 1
                continue

            insert_market(cursor, event_id, market, collection_time)
            market_count += 1
            outcome_count += insert_outcomes(cursor, market)

        if event_count % BATCH_SIZE == 0:
            connection.commit()
            print(
                f"Loaded {event_count:,} events, "
                f"{market_count:,} markets, "
                f"{outcome_count:,} outcomes..."
            )

    cursor.execute(
        """
        INSERT INTO event_collection_attempts (
            attempted_at_utc,
            endpoint,
            limit_value,
            offset_value,
            status,
            events_collected,
            markets_collected,
            error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            collection_time,
            "raw_event_json_file",
            None,
            None,
            "success",
            event_count,
            market_count,
            None if skipped_markets == 0 else f"Skipped {skipped_markets} markets without ids.",
        ),
    )

    connection.commit()

    print("Load complete")
    print("-" * 80)
    print(f"Events loaded: {event_count:,}")
    print(f"Markets loaded: {market_count:,}")
    print(f"Outcomes loaded: {outcome_count:,}")
    print(f"Event-tag links loaded: {event_tag_count:,}")
    print(f"Skipped markets without ids: {skipped_markets:,}")


def print_database_counts(connection: sqlite3.Connection) -> None:
    """Print quick validation counts after loading."""
    cursor = connection.cursor()

    tables = [
        "events",
        "markets",
        "tags",
        "event_tags",
        "outcomes",
        "event_collection_attempts",
    ]

    print("\nDatabase counts")
    print("-" * 80)

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table}: {count:,}")

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT market_id
            FROM outcomes
            GROUP BY market_id
            HAVING COUNT(*) != 2
        )
        """
    )
    non_binary_markets = cursor.fetchone()[0]
    print(f"Markets with outcome count not equal to 2: {non_binary_markets:,}")

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM outcomes
        WHERE resolved_winner = 1
        """
    )
    resolved_winning_outcomes = cursor.fetchone()[0]
    print(f"Resolved winning outcomes inferred: {resolved_winning_outcomes:,}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Load a raw Polymarket event snapshot into SQLite."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
        help="Path to the event-first SQLite database.",
    )
    parser.add_argument(
        "--event-file",
        type=Path,
        default=None,
        help="Raw event snapshot to load. Defaults to the latest snapshot.",
    )
    parser.add_argument(
        "--default-legacy-file",
        action="store_true",
        help="Use the original hard-coded raw event file instead of the latest snapshot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.event_file is not None:
        event_file = args.event_file
    elif args.default_legacy_file:
        event_file = RAW_EVENT_FILE
    else:
        event_file = find_latest_raw_event_file()

    print(f"Streaming raw event file: {event_file}")

    if not event_file.exists():
        raise FileNotFoundError(f"Raw event file not found: {event_file}")

    if not args.database.exists():
        raise FileNotFoundError(
            f"Database not found at {args.database}. "
            "Run scripts/stage_02_database/create_event_database.py first."
        )

    events = iter_events_from_file(event_file)
    connection = sqlite3.connect(args.database)

    try:
        load_events(connection, events)
        print_database_counts(connection)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
