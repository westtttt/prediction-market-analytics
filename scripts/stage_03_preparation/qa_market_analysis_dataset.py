"""Run quality checks against the generated market-level analysis dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "outputs" / "analysis" / "markets_analysis.parquet"
OUTPUT_PATH = ROOT / "data" / "outputs" / "analysis" / "markets_analysis_qa.csv"

KEY_COLUMNS = [
    "market_id",
    "event_id",
    "event_title",
    "market_question",
    "tag_labels",
    "tag_slugs",
    "tag_count",
    "market_volume",
    "market_liquidity",
    "outcome_count",
    "has_resolved_winner",
    "price_points",
    "days_open",
    "primary_tag",
]

BOOLEAN_COLUMNS = [
    "has_volume",
    "has_liquidity",
    "has_price_history",
    "is_market_active",
    "is_market_closed",
    "is_resolved",
]

INVALID_DATE_COLUMNS = [
    "event_created_at_invalid",
    "event_updated_at_invalid",
    "event_start_date_invalid",
    "event_end_date_invalid",
    "event_closed_time_invalid",
    "market_created_at_invalid",
    "market_updated_at_invalid",
    "market_start_date_invalid",
    "market_end_date_invalid",
    "market_closed_time_invalid",
    "first_price_time_invalid",
    "last_price_time_invalid",
]

NUMERIC_PROFILE_COLUMNS = [
    "market_volume",
    "market_liquidity",
    "price_points",
    "days_open",
    "tag_count",
    "outcome_count",
    "last_trade_price",
    "price_mean",
]


def qa_row(area: str, metric: str, value: Any, notes: str = "") -> dict[str, Any]:
    """Return one standard QA output row."""
    return {
        "area": area,
        "metric": metric,
        "value": value,
        "notes": notes,
    }


def require_input() -> None:
    """Fail clearly if the generated analysis dataset is missing."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Analysis dataset not found: {INPUT_PATH}. "
            "Run scripts/stage_03_preparation/build_market_analysis_dataset.py first."
        )


def load_dataset() -> pd.DataFrame:
    """Load the generated market-level analysis dataset."""
    require_input()
    return pd.read_parquet(INPUT_PATH)


def add_dataset_shape(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add row, column, and file-size checks."""
    rows.extend(
        [
            qa_row("shape", "rows", len(frame)),
            qa_row("shape", "columns", len(frame.columns)),
            qa_row(
                "shape",
                "file_size_mb",
                round(INPUT_PATH.stat().st_size / 1024 / 1024, 2),
            ),
        ]
    )


def add_uniqueness_checks(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add key uniqueness checks."""
    market_count = int(frame["market_id"].nunique(dropna=True))
    event_count = int(frame["event_id"].nunique(dropna=True))
    rows.extend(
        [
            qa_row("uniqueness", "unique_market_ids", market_count),
            qa_row("uniqueness", "duplicate_market_rows", len(frame) - market_count),
            qa_row("uniqueness", "unique_event_ids", event_count),
        ]
    )


def add_missingness_checks(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add missingness percentages for the key analysis columns."""
    for column in KEY_COLUMNS:
        missing = int(frame[column].isna().sum())
        rows.append(
            qa_row(
                "missingness",
                f"{column}_missing_percent",
                round(missing / len(frame) * 100, 2),
                f"{missing:,} missing",
            )
        )


def add_boolean_coverage(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add percentages for core boolean flags."""
    for column in BOOLEAN_COLUMNS:
        count = int(frame[column].fillna(False).sum())
        rows.append(
            qa_row(
                "coverage",
                f"{column}_percent",
                round(count / len(frame) * 100, 2),
                f"{count:,} markets",
            )
        )


def add_invalid_date_checks(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add invalid date counts produced during dataset construction."""
    for column in INVALID_DATE_COLUMNS:
        count = int(frame[column].fillna(False).sum())
        rows.append(
            qa_row(
                "date_quality",
                f"{column}_count",
                count,
                f"{round(count / len(frame) * 100, 2)}% of markets",
            )
        )


def add_numeric_profiles(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add compact distribution summaries for core numeric fields."""
    for column in NUMERIC_PROFILE_COLUMNS:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            rows.append(qa_row("numeric_profile", f"{column}_non_null_count", 0))
            continue

        rows.extend(
            [
                qa_row("numeric_profile", f"{column}_non_null_count", int(len(series))),
                qa_row("numeric_profile", f"{column}_min", round(float(series.min()), 4)),
                qa_row("numeric_profile", f"{column}_median", round(float(series.median()), 4)),
                qa_row("numeric_profile", f"{column}_p95", round(float(series.quantile(0.95)), 4)),
                qa_row("numeric_profile", f"{column}_max", round(float(series.max()), 4)),
            ]
        )


def add_top_tags(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add the most common primary tags."""
    tag_counts = frame["primary_tag"].dropna().value_counts().head(20)
    for tag, count in tag_counts.items():
        rows.append(
            qa_row(
                "top_primary_tags",
                str(tag),
                int(count),
                f"{round(count / len(frame) * 100, 2)}% of markets",
            )
        )


def add_metadata_checks(rows: list[dict[str, Any]]) -> None:
    """Add low-cost Parquet metadata checks."""
    metadata = pq.read_metadata(INPUT_PATH)
    rows.extend(
        [
            qa_row("parquet", "row_groups", metadata.num_row_groups),
            qa_row("parquet", "metadata_rows", metadata.num_rows),
            qa_row("parquet", "metadata_columns", metadata.num_columns),
        ]
    )


def write_summary(rows: list[dict[str, Any]]) -> None:
    """Write QA rows to the output CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the analysis dataset QA pass."""
    frame = load_dataset()
    rows: list[dict[str, Any]] = []

    add_metadata_checks(rows)
    add_dataset_shape(rows, frame)
    add_uniqueness_checks(rows, frame)
    add_missingness_checks(rows, frame)
    add_boolean_coverage(rows, frame)
    add_invalid_date_checks(rows, frame)
    add_numeric_profiles(rows, frame)
    add_top_tags(rows, frame)
    write_summary(rows)

    print(f"Wrote QA summary to {OUTPUT_PATH}", flush=True)
    print(f"Rows checked: {len(frame):,}", flush=True)
    print(f"Duplicate market rows: {len(frame) - frame['market_id'].nunique(dropna=True):,}", flush=True)


if __name__ == "__main__":
    main()
