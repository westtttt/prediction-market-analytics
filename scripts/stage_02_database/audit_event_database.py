"""Run read-only audit checks against the local market database."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / "data" / "database" / "markets.db"
OUTPUT_DIR = ROOT / "data" / "outputs" / "audit"
COMPLETENESS_SAMPLE_LIMIT = 100_000

KEY_FIELDS = {
    "events": [
        "event_id",
        "title",
        "description",
        "active",
        "closed",
        "volume",
        "liquidity",
        "created_at",
        "updated_at",
        "end_date",
        "closed_time",
    ],
    "markets": [
        "market_id",
        "event_id",
        "question",
        "description",
        "active",
        "closed",
        "volume",
        "liquidity",
        "best_bid",
        "best_ask",
        "last_trade_price",
        "created_at",
        "updated_at",
        "end_date",
        "closed_time",
    ],
    "outcomes": [
        "market_id",
        "outcome_index",
        "outcome_name",
        "current_price",
        "clob_token_id",
        "resolved_winner",
        "resolution_inference_status",
    ],
    "price_history": [
        "market_id",
        "outcome_index",
        "timestamp",
        "datetime_utc",
        "price",
    ],
    "tags": ["tag_id", "label", "slug"],
    "event_tags": ["event_id", "tag_id"],
}


def audit_row(
    area: str,
    table: str,
    metric: str,
    value: Any,
    checked_rows: int | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Return one standard audit summary row."""
    return {
        "area": area,
        "table": table,
        "metric": metric,
        "value": value,
        "checked_rows": checked_rows,
        "notes": notes,
    }


def connect_read_only() -> sqlite3.Connection:
    """Open the project database in read-only mode."""
    if not DATABASE.exists():
        raise FileNotFoundError(f"Database not found: {DATABASE}")

    connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON;")
    return connection


def fetch_table_counts(connection: sqlite3.Connection) -> list[dict[str, int | str]]:
    """Return row counts for each user table."""
    tables = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()

    rows: list[dict[str, int | str]] = []
    for table in tables:
        table_name = table["name"]
        count = connection.execute(f'SELECT COUNT(*) AS count FROM "{table_name}"').fetchone()
        rows.append({"table": table_name, "row_count": int(count["count"])})
    return rows


def fetch_schema(
    connection: sqlite3.Connection, table_counts: list[dict[str, int | str]]
) -> list[dict[str, Any]]:
    """Return column metadata for each user table."""
    rows: list[dict[str, Any]] = []
    for table in table_counts:
        table_name = str(table["table"])
        columns = connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        for column in columns:
            rows.append(
                {
                    "table": table_name,
                    "column": column["name"],
                    "type": column["type"],
                    "not_null": int(column["notnull"]),
                    "default_value": column["dflt_value"],
                    "primary_key_position": int(column["pk"]),
                }
            )
    return rows


def fetch_audit_summary(
    connection: sqlite3.Connection, table_counts: list[dict[str, int | str]]
) -> list[dict[str, Any]]:
    """Return a compact set of audit metrics."""
    rows: list[dict[str, Any]] = []
    row_counts = {str(row["table"]): int(row["row_count"]) for row in table_counts}

    for table_name, fields in KEY_FIELDS.items():
        total_rows = row_counts[table_name]
        available_columns = {
            row["name"]
            for row in connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        }
        present_fields = [field for field in fields if field in available_columns]

        for field in fields:
            if field not in available_columns:
                rows.append(
                    audit_row(
                        "completeness",
                        table_name,
                        f"{field}_non_null_percent",
                        None,
                        notes="field not present",
                    )
                )
        if not present_fields:
            continue

        expressions = ["COUNT(*) AS checked_rows"]
        for field in present_fields:
            expressions.append(
                f'SUM(CASE WHEN "{field}" IS NOT NULL THEN 1 ELSE 0 END) '
                f'AS "{field}__non_null_count"'
            )
            expressions.append(
                f'SUM(CASE WHEN TRIM(CAST("{field}" AS TEXT)) = \'\' THEN 1 ELSE 0 END) '
                f'AS "{field}__blank_count"'
            )
        selected_fields = ", ".join(f'"{field}"' for field in present_fields)
        result = connection.execute(
            f"""
            SELECT {", ".join(expressions)}
            FROM (
                SELECT {selected_fields}
                FROM "{table_name}"
                LIMIT {COMPLETENESS_SAMPLE_LIMIT}
            )
            """
        ).fetchone()
        checked_rows = int(result["checked_rows"] or 0)
        notes = (
            f"sampled first {checked_rows:,} rows from {total_rows:,}"
            if checked_rows < total_rows
            else ""
        )

        for field in present_fields:
            non_null_count = int(result[f"{field}__non_null_count"] or 0)
            blank_count = int(result[f"{field}__blank_count"] or 0)
            non_null_percent = (
                round(non_null_count / checked_rows * 100, 2) if checked_rows else 0
            )
            blank_percent = round(blank_count / checked_rows * 100, 2) if checked_rows else 0
            rows.extend(
                [
                    audit_row(
                        "completeness",
                        table_name,
                        f"{field}_non_null_percent",
                        non_null_percent,
                        checked_rows,
                        notes,
                    ),
                    audit_row(
                        "completeness",
                        table_name,
                        f"{field}_blank_percent",
                        blank_percent,
                        checked_rows,
                        notes,
                    ),
                ]
            )
    rows.extend(fetch_status_summary(connection, row_counts))
    rows.extend(fetch_date_ranges(connection))
    rows.extend(fetch_coverage_summary(connection, row_counts))
    return rows


def fetch_status_summary(
    connection: sqlite3.Connection, row_counts: dict[str, int]
) -> list[dict[str, Any]]:
    """Return sampled status counts for events, markets and outcomes."""
    rows: list[dict[str, Any]] = []

    for table_name, fields in {
        "events": ["active", "closed", "archived"],
        "markets": ["active", "closed", "archived", "accepting_orders"],
    }.items():
        select_fields = ", ".join(f'"{field}"' for field in fields)
        grouped = connection.execute(
            f"""
            SELECT {select_fields}, COUNT(*) AS count
            FROM (
                SELECT {select_fields}
                FROM "{table_name}"
                LIMIT {COMPLETENESS_SAMPLE_LIMIT}
            )
            GROUP BY {select_fields}
            ORDER BY count DESC
            """
        ).fetchall()
        for group in grouped:
            status = "_".join(f"{field}_{group[field]}" for field in fields)
            checked_rows = min(row_counts[table_name], COMPLETENESS_SAMPLE_LIMIT)
            rows.append(
                audit_row(
                    "status",
                    table_name,
                    status,
                    int(group["count"]),
                    checked_rows,
                    f"sampled first {checked_rows:,} rows",
                )
            )

    for field in ["resolved_winner", "resolution_inference_status"]:
        grouped = connection.execute(
            f"""
            SELECT "{field}" AS status_value, COUNT(*) AS count
            FROM (
                SELECT "{field}"
                FROM outcomes
                LIMIT {COMPLETENESS_SAMPLE_LIMIT}
            )
            GROUP BY "{field}"
            ORDER BY count DESC
            """
        ).fetchall()
        for group in grouped:
            value = group["status_value"]
            label = "null" if value is None else str(value)
            checked_rows = min(row_counts["outcomes"], COMPLETENESS_SAMPLE_LIMIT)
            rows.append(
                audit_row(
                    "status",
                    "outcomes",
                    f"{field}_{label}",
                    int(group["count"]),
                    checked_rows,
                    f"sampled first {checked_rows:,} rows",
                )
            )
    return rows


def fetch_date_ranges(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return sampled min/max date values for key temporal fields."""
    rows: list[dict[str, Any]] = []
    date_fields = {
        "events": ["created_at", "updated_at", "start_date", "end_date", "closed_time"],
        "markets": ["created_at", "updated_at", "start_date", "end_date", "closed_time"],
    }

    for table_name, fields in date_fields.items():
        expressions = []
        for field in fields:
            expressions.append(f'MIN("{field}") AS "{field}__min"')
            expressions.append(f'MAX("{field}") AS "{field}__max"')
        expressions.append("COUNT(*) AS checked_rows")
        selected_fields = ", ".join(f'"{field}"' for field in fields)
        result = connection.execute(
            f"""
            SELECT {", ".join(expressions)}
            FROM (
                SELECT {selected_fields}
                FROM "{table_name}"
                LIMIT {COMPLETENESS_SAMPLE_LIMIT}
            )
            """
        ).fetchone()
        checked_rows = int(result["checked_rows"] or 0)
        notes = f"sampled first {checked_rows:,} rows"
        for field in fields:
            rows.append(
                audit_row(
                    "date_range",
                    table_name,
                    f"{field}_min",
                    result[f"{field}__min"],
                    checked_rows,
                    notes,
                )
            )
            rows.append(
                audit_row(
                    "date_range",
                    table_name,
                    f"{field}_max",
                    result[f"{field}__max"],
                    checked_rows,
                    notes,
                )
            )

    price_history_range = connection.execute(
        f"""
        SELECT
            MIN(timestamp) AS min_timestamp,
            MAX(timestamp) AS max_timestamp,
            MIN(datetime_utc) AS min_datetime,
            MAX(datetime_utc) AS max_datetime,
            COUNT(*) AS checked_rows
        FROM (
            SELECT timestamp, datetime_utc
            FROM price_history
            LIMIT {COMPLETENESS_SAMPLE_LIMIT}
        )
        """
    ).fetchone()
    checked_rows = int(price_history_range["checked_rows"] or 0)
    notes = f"sampled first {checked_rows:,} rows"
    rows.append(
        audit_row(
            "date_range",
            "price_history",
            "datetime_utc_min",
            price_history_range["min_datetime"],
            checked_rows,
            notes,
        )
    )
    rows.append(
        audit_row(
            "date_range",
            "price_history",
            "datetime_utc_max",
            price_history_range["max_datetime"],
            checked_rows,
            notes,
        )
    )
    rows.append(
        audit_row(
            "date_range",
            "price_history",
            "timestamp_min",
            price_history_range["min_timestamp"],
            checked_rows,
            notes,
        )
    )
    rows.append(
        audit_row(
            "date_range",
            "price_history",
            "timestamp_max",
            price_history_range["max_timestamp"],
            checked_rows,
            notes,
        )
    )
    return rows


def fetch_coverage_summary(
    connection: sqlite3.Connection, row_counts: dict[str, int]
) -> list[dict[str, Any]]:
    """Return compact relationship/coverage metrics."""
    rows: list[dict[str, Any]] = []

    event_tag_coverage = connection.execute(
        """
        SELECT COUNT(DISTINCT event_id) AS tagged_events
        FROM event_tags
        """
    ).fetchone()
    tagged_events = int(event_tag_coverage["tagged_events"] or 0)
    rows.append(
        audit_row(
            "coverage",
            "event_tags",
            "events_with_tags_percent",
            round(tagged_events / row_counts["events"] * 100, 2),
            row_counts["events"],
            f"{tagged_events:,} events with at least one tag",
        )
    )

    price_sample = connection.execute(
        f"""
        SELECT
            COUNT(DISTINCT market_id) AS sample_markets_with_history,
            COUNT(DISTINCT market_id || ':' || outcome_index) AS sample_outcomes_with_history,
            COUNT(*) AS checked_rows
        FROM (
            SELECT market_id, outcome_index
            FROM price_history
            LIMIT {COMPLETENESS_SAMPLE_LIMIT}
        )
        """
    ).fetchone()
    checked_rows = int(price_sample["checked_rows"] or 0)
    notes = f"sampled first {checked_rows:,} price-history rows"
    rows.extend(
        [
            audit_row(
                "coverage",
                "price_history",
                "sample_distinct_markets_with_history",
                int(price_sample["sample_markets_with_history"] or 0),
                checked_rows,
                notes,
            ),
            audit_row(
                "coverage",
                "price_history",
                "sample_distinct_outcomes_with_history",
                int(price_sample["sample_outcomes_with_history"] or 0),
                checked_rows,
                notes,
            ),
            audit_row(
                "coverage",
                "price_history",
                "price_points",
                row_counts["price_history"],
            ),
        ]
    )

    rows.append(
        audit_row(
            "coverage",
            "markets",
            "avg_markets_per_event",
            round(row_counts["markets"] / row_counts["events"], 2),
            notes="calculated from table counts",
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write audit rows to CSV."""
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the first audit pass."""
    with connect_read_only() as connection:
        table_counts = fetch_table_counts(connection)
        schema = fetch_schema(connection, table_counts)
        audit_summary = fetch_audit_summary(connection, table_counts)

    write_csv(OUTPUT_DIR / "table_counts.csv", table_counts)
    write_csv(OUTPUT_DIR / "schema.csv", schema)
    write_csv(OUTPUT_DIR / "audit_summary.csv", audit_summary)
    print(f"Wrote audit outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
