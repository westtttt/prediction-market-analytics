"""Build the market-level analysis dataset from the local SQLite snapshot."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pandas as pd
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as error:
    raise RuntimeError("pyarrow is required to write markets_analysis.parquet") from error


ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / "data" / "database" / "markets.db"
OUTPUT_DIR = ROOT / "data" / "outputs" / "analysis"
PARQUET_PATH = OUTPUT_DIR / "markets_analysis.parquet"
SAMPLE_PATH = OUTPUT_DIR / "markets_analysis_sample.csv"

CHUNK_SIZE = 100_000
SAMPLE_ROWS = 1_000

DATE_COLUMNS = [
    "event_created_at",
    "event_updated_at",
    "event_start_date",
    "event_end_date",
    "event_closed_time",
    "market_created_at",
    "market_updated_at",
    "market_start_date",
    "market_end_date",
    "market_closed_time",
    "first_price_time",
    "last_price_time",
]

NUMERIC_COLUMNS = [
    "event_volume",
    "event_volume_24hr",
    "event_volume_1wk",
    "event_volume_1mo",
    "event_volume_1yr",
    "event_liquidity",
    "event_liquidity_clob",
    "event_open_interest",
    "market_volume",
    "market_volume_num",
    "market_volume_clob",
    "market_volume_24hr",
    "market_volume_1wk",
    "market_volume_1mo",
    "market_volume_1yr",
    "market_liquidity",
    "market_liquidity_num",
    "market_liquidity_clob",
    "best_bid",
    "best_ask",
    "spread",
    "last_trade_price",
    "price_min",
    "price_max",
    "price_mean",
]

INTEGER_COLUMNS = [
    "event_active",
    "event_closed",
    "event_archived",
    "market_active",
    "market_closed",
    "market_archived",
    "accepting_orders",
    "approved",
    "restricted",
    "tag_count",
    "outcome_count",
    "has_resolved_winner",
    "price_points",
    "first_price_timestamp",
    "last_price_timestamp",
]

TEXT_COLUMNS = [
    "market_id",
    "event_id",
    "event_title",
    "market_question",
    "event_description",
    "market_description",
    "event_slug",
    "market_slug",
    "tag_labels",
    "tag_slugs",
    "market_type",
    "group_item_title",
    "group_item_threshold",
    "event_resolution_source",
    "market_resolution_source",
    "uma_resolution_status",
    "series_slug",
    "outcome_names",
    "current_prices",
    "winner_outcome_names",
    "resolution_inference_statuses",
]


ROLLUP_QUERIES = [
    (
        "tag rollup",
        """
        CREATE TEMP TABLE tag_rollup AS
        SELECT
            event_id,
            group_concat(label, '; ') AS tag_labels,
            group_concat(slug, '; ') AS tag_slugs,
            COUNT(*) AS tag_count
        FROM (
            SELECT DISTINCT
                et.event_id,
                t.label,
                t.slug
            FROM event_tags et
            JOIN tags t ON t.tag_id = et.tag_id
            WHERE t.label IS NOT NULL
            ORDER BY et.event_id, lower(t.label)
        )
        GROUP BY event_id
        """,
    ),
    (
        "outcome rollup",
        """
        CREATE TEMP TABLE outcome_rollup AS
        SELECT
            market_id,
            COUNT(*) AS outcome_count,
            group_concat(outcome_name, '; ') AS outcome_names,
            group_concat(current_price, '; ') AS current_prices,
            MAX(CASE WHEN resolved_winner = 1 THEN 1 ELSE 0 END) AS has_resolved_winner,
            group_concat(CASE WHEN resolved_winner = 1 THEN outcome_name END, '; ') AS winner_outcome_names,
            group_concat(DISTINCT resolution_inference_status) AS resolution_inference_statuses
        FROM outcomes
        GROUP BY market_id
        """,
    ),
    (
        "price-history coverage rollup",
        """
        CREATE TEMP TABLE price_rollup AS
        SELECT
            market_id,
            COUNT(*) AS price_points,
            MIN(timestamp) AS first_price_timestamp,
            MAX(timestamp) AS last_price_timestamp,
            MIN(datetime_utc) AS first_price_time,
            MAX(datetime_utc) AS last_price_time,
            MIN(price) AS price_min,
            MAX(price) AS price_max,
            AVG(price) AS price_mean
        FROM price_history
        GROUP BY market_id
        """,
    ),
    ("tag rollup index", "CREATE INDEX idx_temp_tag_rollup_event ON tag_rollup(event_id)"),
    ("outcome rollup index", "CREATE INDEX idx_temp_outcome_rollup_market ON outcome_rollup(market_id)"),
    ("price rollup index", "CREATE INDEX idx_temp_price_rollup_market ON price_rollup(market_id)"),
]


MARKET_ANALYSIS_QUERY = """
SELECT
    m.market_id,
    m.event_id,
    e.title AS event_title,
    m.question AS market_question,
    e.description AS event_description,
    m.description AS market_description,
    e.slug AS event_slug,
    m.slug AS market_slug,
    tr.tag_labels,
    tr.tag_slugs,
    tr.tag_count,
    e.active AS event_active,
    e.closed AS event_closed,
    e.archived AS event_archived,
    m.active AS market_active,
    m.closed AS market_closed,
    m.archived AS market_archived,
    m.accepting_orders,
    m.approved,
    m.restricted,
    e.volume AS event_volume,
    e.volume_24hr AS event_volume_24hr,
    e.volume_1wk AS event_volume_1wk,
    e.volume_1mo AS event_volume_1mo,
    e.volume_1yr AS event_volume_1yr,
    e.liquidity AS event_liquidity,
    e.liquidity_clob AS event_liquidity_clob,
    e.open_interest AS event_open_interest,
    m.volume AS market_volume,
    m.volume_num AS market_volume_num,
    m.volume_clob AS market_volume_clob,
    m.volume_24hr AS market_volume_24hr,
    m.volume_1wk AS market_volume_1wk,
    m.volume_1mo AS market_volume_1mo,
    m.volume_1yr AS market_volume_1yr,
    m.liquidity AS market_liquidity,
    m.liquidity_num AS market_liquidity_num,
    m.liquidity_clob AS market_liquidity_clob,
    m.best_bid,
    m.best_ask,
    m.spread,
    m.last_trade_price,
    m.market_type,
    m.group_item_title,
    m.group_item_threshold,
    e.created_at AS event_created_at,
    e.updated_at AS event_updated_at,
    e.start_date AS event_start_date,
    e.end_date AS event_end_date,
    e.closed_time AS event_closed_time,
    m.created_at AS market_created_at,
    m.updated_at AS market_updated_at,
    m.start_date AS market_start_date,
    m.end_date AS market_end_date,
    m.closed_time AS market_closed_time,
    e.resolution_source AS event_resolution_source,
    m.resolution_source AS market_resolution_source,
    m.uma_resolution_status,
    e.series_slug,
    o.outcome_count,
    o.outcome_names,
    o.current_prices,
    o.has_resolved_winner,
    o.winner_outcome_names,
    o.resolution_inference_statuses,
    p.price_points,
    p.first_price_timestamp,
    p.last_price_timestamp,
    p.first_price_time,
    p.last_price_time,
    p.price_min,
    p.price_max,
    p.price_mean
FROM markets m
LEFT JOIN events e ON e.event_id = m.event_id
LEFT JOIN tag_rollup tr ON tr.event_id = m.event_id
LEFT JOIN outcome_rollup o ON o.market_id = m.market_id
LEFT JOIN price_rollup p ON p.market_id = m.market_id
"""


def connect_read_only() -> sqlite3.Connection:
    """Open the source database without allowing writes."""
    if not DATABASE.exists():
        raise FileNotFoundError(f"Database not found: {DATABASE}")

    connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    connection.execute("PRAGMA temp_store = FILE;")
    return connection


def check_parquet_dependency() -> None:
    """Fail early if pyarrow is not available."""
    if importlib.util.find_spec("pyarrow") is None:
        raise RuntimeError("pyarrow is required to write markets_analysis.parquet")


def log(message: str = "") -> None:
    """Print progress immediately during long local builds."""
    print(message, flush=True)


def parse_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse date fields and add invalid-date flags."""
    for column in DATE_COLUMNS:
        original = frame[column]
        parsed = pd.to_datetime(original, errors="coerce", utc=True)
        frame[f"{column}_invalid"] = original.notna() & parsed.isna()
        frame[column] = parsed
    return frame


def clean_chunk(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean types and add basic derived analysis fields."""
    for column in TEXT_COLUMNS:
        frame[column] = frame[column].astype("string")

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in INTEGER_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")

    frame = parse_dates(frame)

    close_or_end = frame["market_closed_time"].fillna(frame["market_end_date"])
    frame["days_open"] = (
        (close_or_end - frame["market_created_at"]).dt.total_seconds() / 86_400
    ).round(2)
    frame.loc[frame["days_open"] < 0, "days_open"] = pd.NA

    frame["has_volume"] = frame["market_volume"].fillna(0) > 0
    frame["has_liquidity"] = frame["market_liquidity"].fillna(0) > 0
    frame["has_price_history"] = frame["price_points"].fillna(0) > 0
    no_current_prices = frame["current_prices"].fillna("").str.strip() == ""
    unusable_market = no_current_prices & ~frame["has_price_history"] & ~frame["has_volume"]
    frame["is_market_active"] = frame["market_active"] == 1
    frame["is_market_closed"] = frame["market_closed"] == 1
    frame["is_resolved"] = frame["has_resolved_winner"].fillna(0) == 1
    frame["market_created_year"] = frame["market_created_at"].dt.year.astype("Int64")
    frame["market_created_month"] = frame["market_created_at"].dt.strftime("%Y-%m").astype("string")
    frame["primary_tag"] = frame["tag_labels"].str.split("; ").str[0]

    return frame.loc[~unusable_market].copy()


def append_chunk(writer: pq.ParquetWriter | None, frame: pd.DataFrame) -> pq.ParquetWriter:
    """Append a dataframe chunk to the Parquet file."""
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(PARQUET_PATH, table.schema, compression="snappy")
    writer.write_table(table)
    return writer


def build_rollups(connection: sqlite3.Connection) -> None:
    """Build temporary summary tables used by the final market-level export."""
    for label, query in ROLLUP_QUERIES:
        log(f"Building {label}...")
        connection.execute(query)


def main() -> None:
    """Build the analysis-ready market table."""
    check_parquet_dependency()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if PARQUET_PATH.exists():
        PARQUET_PATH.unlink()
    if SAMPLE_PATH.exists():
        SAMPLE_PATH.unlink()

    writer: pq.ParquetWriter | None = None
    sample_parts: list[pd.DataFrame] = []
    total_rows = 0
    source_rows = 0
    column_count = 0
    has_price_history = 0
    has_volume = 0
    has_liquidity = 0
    invalid_date_counts = {column: 0 for column in DATE_COLUMNS}

    try:
        with connect_read_only() as connection:
            build_rollups(connection)
            log("Writing market-level Parquet dataset...")
            chunks = pd.read_sql_query(
                MARKET_ANALYSIS_QUERY,
                connection,
                chunksize=CHUNK_SIZE,
            )
            for chunk_number, chunk in enumerate(chunks, start=1):
                source_rows += len(chunk)
                clean = clean_chunk(chunk)
                column_count = len(clean.columns)
                writer = append_chunk(writer, clean)

                total_rows += len(clean)
                has_price_history += int(clean["has_price_history"].sum())
                has_volume += int(clean["has_volume"].sum())
                has_liquidity += int(clean["has_liquidity"].sum())
                for column in DATE_COLUMNS:
                    invalid_date_counts[column] += int(clean[f"{column}_invalid"].sum())

                if sum(len(part) for part in sample_parts) < SAMPLE_ROWS:
                    remaining = SAMPLE_ROWS - sum(len(part) for part in sample_parts)
                    sample_parts.append(clean.head(remaining))

                log(f"Processed chunk {chunk_number}: {total_rows:,} markets")
    finally:
        if writer is not None:
            writer.close()

    if sample_parts:
        sample = pd.concat(sample_parts, ignore_index=True)
        sample.to_csv(SAMPLE_PATH, index=False)

    parquet_mb = PARQUET_PATH.stat().st_size / 1024 / 1024
    log()
    log("Analysis dataset built")
    log(f"Source rows checked: {source_rows:,}")
    log(f"Rows: {total_rows:,}")
    log(f"Unusable no-probability markets removed: {source_rows - total_rows:,}")
    log(f"Columns: {column_count}")
    log(f"Markets with price history: {has_price_history:,}")
    log(f"Markets with volume: {has_volume:,}")
    log(f"Markets with liquidity: {has_liquidity:,}")
    log(f"Output: {PARQUET_PATH} ({parquet_mb:.1f} MB)")
    log(f"Sample: {SAMPLE_PATH}")

    dirty_dates = {key: value for key, value in invalid_date_counts.items() if value}
    if dirty_dates:
        log("Invalid date values coerced to missing:")
        for column, count in dirty_dates.items():
            log(f"- {column}: {count:,}")


if __name__ == "__main__":
    main()
