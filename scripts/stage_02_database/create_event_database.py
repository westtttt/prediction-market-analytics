"""
Create the event-first SQLite database used by the dissertation project.

This database is separate from the original market-first database. It reflects
an event-first data model based on the Polymarket Gamma API, where events
contain collections of related binary markets. Multi-outcome forecasting
contexts are therefore represented as events containing multiple binary markets,
rather than as single markets with many outcomes.

Database created:

    data/database/markets.db

Tables:
- events
- markets
- tags
- event_tags
- outcomes
- price_history
- event_collection_attempts
- price_history_collection_attempts

Run from the project root:

    python3 scripts/stage_02_database/create_event_database.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3


DATABASE_DIR = Path("data/database")
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = DATABASE_DIR / "markets.db"


def create_tables(connection: sqlite3.Connection) -> None:
    """
    Create all event-first database tables and indexes.
    """

    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")

    cursor.execute(
        """
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            title TEXT,
            slug TEXT,
            description TEXT,
            active INTEGER,
            closed INTEGER,
            archived INTEGER,
            volume REAL,
            volume_24hr REAL,
            volume_1wk REAL,
            volume_1mo REAL,
            volume_1yr REAL,
            liquidity REAL,
            liquidity_clob REAL,
            open_interest REAL,
            created_at TEXT,
            updated_at TEXT,
            start_date TEXT,
            end_date TEXT,
            closed_time TEXT,
            resolution_source TEXT,
            series_slug TEXT,
            collection_time_utc TEXT,
            raw_json TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE markets (
            market_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            condition_id TEXT,
            question_id TEXT,
            question TEXT,
            slug TEXT,
            description TEXT,
            active INTEGER,
            closed INTEGER,
            archived INTEGER,
            accepting_orders INTEGER,
            approved INTEGER,
            restricted INTEGER,
            volume REAL,
            volume_num REAL,
            volume_clob REAL,
            volume_24hr REAL,
            volume_1wk REAL,
            volume_1mo REAL,
            volume_1yr REAL,
            liquidity REAL,
            liquidity_num REAL,
            liquidity_clob REAL,
            best_bid REAL,
            best_ask REAL,
            spread REAL,
            last_trade_price REAL,
            market_type TEXT,
            group_item_title TEXT,
            group_item_threshold TEXT,
            created_at TEXT,
            updated_at TEXT,
            start_date TEXT,
            end_date TEXT,
            closed_time TEXT,
            resolution_source TEXT,
            uma_resolution_status TEXT,
            collection_time_utc TEXT,
            raw_json TEXT,
            FOREIGN KEY (event_id)
                REFERENCES events(event_id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE tags (
            tag_id TEXT PRIMARY KEY,
            label TEXT,
            slug TEXT,
            force_show INTEGER,
            force_hide INTEGER,
            is_carousel INTEGER,
            requires_translation INTEGER,
            created_at TEXT,
            updated_at TEXT,
            published_at TEXT,
            raw_json TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE event_tags (
            event_id TEXT NOT NULL,
            tag_id TEXT NOT NULL,
            PRIMARY KEY (event_id, tag_id),
            FOREIGN KEY (event_id)
                REFERENCES events(event_id),
            FOREIGN KEY (tag_id)
                REFERENCES tags(tag_id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE outcomes (
            market_id TEXT NOT NULL,
            outcome_index INTEGER NOT NULL,
            outcome_name TEXT,
            current_price REAL,
            clob_token_id TEXT,
            resolved_winner INTEGER,
            resolution_inference_status TEXT,
            PRIMARY KEY (market_id, outcome_index),
            FOREIGN KEY (market_id)
                REFERENCES markets(market_id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL,
            outcome_index INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            datetime_utc TEXT,
            price REAL,
            collection_time_utc TEXT,
            FOREIGN KEY (market_id, outcome_index)
                REFERENCES outcomes(market_id, outcome_index)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE event_collection_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempted_at_utc TEXT NOT NULL,
            endpoint TEXT,
            limit_value INTEGER,
            offset_value INTEGER,
            status TEXT NOT NULL,
            events_collected INTEGER DEFAULT 0,
            markets_collected INTEGER DEFAULT 0,
            error_message TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE price_history_collection_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL,
            outcome_index INTEGER,
            clob_token_id TEXT NOT NULL,
            attempted_at_utc TEXT NOT NULL,
            status TEXT NOT NULL,
            points_collected INTEGER DEFAULT 0,
            error_message TEXT,
            FOREIGN KEY (market_id)
                REFERENCES markets(market_id)
        )
        """
    )

    cursor.execute(
        """
        CREATE UNIQUE INDEX idx_price_history_unique
        ON price_history (market_id, outcome_index, timestamp)
        """
    )

    cursor.execute("CREATE INDEX idx_events_slug ON events (slug)")
    cursor.execute("CREATE INDEX idx_events_title ON events (title)")
    cursor.execute("CREATE INDEX idx_events_active_closed ON events (active, closed)")
    cursor.execute("CREATE INDEX idx_events_volume ON events (volume)")

    cursor.execute("CREATE INDEX idx_markets_event ON markets (event_id)")
    cursor.execute("CREATE INDEX idx_markets_question ON markets (question)")
    cursor.execute("CREATE INDEX idx_markets_active_closed ON markets (active, closed)")
    cursor.execute("CREATE INDEX idx_markets_group_item_title ON markets (group_item_title)")

    cursor.execute("CREATE INDEX idx_tags_slug ON tags (slug)")
    cursor.execute("CREATE INDEX idx_tags_label ON tags (label)")

    cursor.execute("CREATE INDEX idx_event_tags_event ON event_tags (event_id)")
    cursor.execute("CREATE INDEX idx_event_tags_tag ON event_tags (tag_id)")

    cursor.execute("CREATE INDEX idx_outcomes_market ON outcomes (market_id)")
    cursor.execute("CREATE INDEX idx_outcomes_clob_token ON outcomes (clob_token_id)")
    cursor.execute("CREATE INDEX idx_outcomes_resolved_winner ON outcomes (resolved_winner)")
    cursor.execute(
        "CREATE INDEX idx_outcomes_resolution_status ON outcomes (resolution_inference_status)"
    )

    cursor.execute("CREATE INDEX idx_price_history_market ON price_history (market_id)")
    cursor.execute(
        "CREATE INDEX idx_price_history_market_timestamp ON price_history (market_id, timestamp)"
    )
    cursor.execute(
        "CREATE INDEX idx_price_history_market_outcome ON price_history (market_id, outcome_index)"
    )

    cursor.execute(
        "CREATE INDEX idx_event_collection_attempts_status ON event_collection_attempts (status)"
    )
    cursor.execute(
        "CREATE INDEX idx_price_history_attempts_market ON price_history_collection_attempts (market_id)"
    )
    cursor.execute(
        "CREATE INDEX idx_price_history_attempts_status ON price_history_collection_attempts (status)"
    )

    connection.commit()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Create an empty event-first Polymarket SQLite database."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
        help="Database path to create.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database_path = args.database
    print(f"Creating event-first database: {database_path}")

    if database_path.exists():
        raise FileExistsError(
            f"Database already exists at {database_path}. "
            "Choose a new --database path when creating a fresh snapshot."
        )

    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)

    create_tables(connection)

    connection.close()

    print("Event-first database created successfully.")
    print(f"Location: {database_path}")


if __name__ == "__main__":
    main()
