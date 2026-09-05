"""Build dissertation-ready descriptive statistics tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "outputs" / "analysis" / "markets_analysis.parquet"
CATEGORY_PATH = ROOT / "data" / "outputs" / "analysis" / "market_categories.parquet"
OUTPUT_PATH = ROOT / "data" / "outputs" / "analysis" / "descriptive_statistics.csv"

NUMERIC_COLUMNS = [
    "market_volume",
    "market_liquidity",
    "price_points",
    "days_open",
    "tag_count",
    "outcome_count",
    "last_trade_price",
    "price_mean",
    "spread",
]

BINARY_COLUMNS = [
    "has_volume",
    "has_liquidity",
    "has_price_history",
    "is_market_active",
    "is_market_closed",
    "is_resolved",
]

DATE_FLAG_COLUMNS = [
    "event_created_at_invalid",
    "event_start_date_invalid",
    "event_end_date_invalid",
    "event_closed_time_invalid",
    "market_created_at_invalid",
    "market_start_date_invalid",
    "market_end_date_invalid",
    "market_closed_time_invalid",
    "first_price_time_invalid",
    "last_price_time_invalid",
]

LOAD_COLUMNS = (
    ["market_id", "event_id", "primary_tag", "market_created_month", "market_created_year"]
    + NUMERIC_COLUMNS
    + BINARY_COLUMNS
    + DATE_FLAG_COLUMNS
)


def stat_row(section: str, variable: str, metric: str, value: Any, notes: str = "") -> dict[str, Any]:
    """Return one standard descriptive-statistics row."""
    return {
        "section": section,
        "variable": variable,
        "metric": metric,
        "value": value,
        "notes": notes,
    }


def require_inputs() -> None:
    """Fail clearly if required generated datasets are missing."""
    missing = [path for path in [INPUT_PATH, CATEGORY_PATH] if not path.exists()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing input dataset(s): {paths}")


def load_data() -> pd.DataFrame:
    """Load market-level statistics fields plus cleaned categories."""
    require_inputs()
    frame = pd.read_parquet(INPUT_PATH, columns=LOAD_COLUMNS)
    categories = pd.read_parquet(CATEGORY_PATH, columns=["market_id", "clean_category"])
    frame = frame.merge(categories, on="market_id", how="left", validate="one_to_one")
    frame["clean_category"] = frame["clean_category"].fillna("Other")
    return frame


def add_dataset_overview(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add high-level dataset counts."""
    rows.extend(
        [
            stat_row("overview", "dataset", "markets", len(frame)),
            stat_row("overview", "dataset", "events", int(frame["event_id"].nunique())),
            stat_row("overview", "dataset", "raw_primary_tags", int(frame["primary_tag"].nunique(dropna=True))),
            stat_row("overview", "dataset", "clean_categories", int(frame["clean_category"].nunique(dropna=True))),
            stat_row(
                "overview",
                "dataset",
                "months_covered",
                int(frame["market_created_month"].nunique(dropna=True)),
            ),
        ]
    )


def add_numeric_statistics(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add count, missingness, and distribution stats for numeric variables."""
    total = len(frame)
    for column in NUMERIC_COLUMNS:
        series = pd.to_numeric(frame[column], errors="coerce")
        non_missing = series.dropna()
        missing_count = total - len(non_missing)
        rows.extend(
            [
                stat_row("numeric", column, "non_missing_count", int(len(non_missing))),
                stat_row("numeric", column, "missing_percent", pct(missing_count, total)),
            ]
        )
        if non_missing.empty:
            continue
        for metric, value in [
            ("mean", non_missing.mean()),
            ("std", non_missing.std()),
            ("min", non_missing.min()),
            ("p25", non_missing.quantile(0.25)),
            ("median", non_missing.median()),
            ("p75", non_missing.quantile(0.75)),
            ("p90", non_missing.quantile(0.90)),
            ("p95", non_missing.quantile(0.95)),
            ("p99", non_missing.quantile(0.99)),
            ("max", non_missing.max()),
        ]:
            rows.append(stat_row("numeric", column, metric, round(float(value), 4)))

        positive_count = int((non_missing > 0).sum())
        rows.append(stat_row("numeric", column, "positive_percent", pct(positive_count, total)))


def add_binary_statistics(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add true/false counts for key indicators."""
    total = len(frame)
    for column in BINARY_COLUMNS:
        values = frame[column].fillna(False).astype(bool)
        true_count = int(values.sum())
        rows.extend(
            [
                stat_row("binary", column, "true_count", true_count),
                stat_row("binary", column, "true_percent", pct(true_count, total)),
                stat_row("binary", column, "false_count", total - true_count),
            ]
        )


def add_category_statistics(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add compact descriptive summaries for categorical variables."""
    total = len(frame)
    for column in ["clean_category", "primary_tag", "market_created_year"]:
        counts = frame[column].astype("string").fillna("(missing)").value_counts()
        rows.extend(
            [
                stat_row("categorical", column, "unique_values", int(counts.size)),
                stat_row("categorical", column, "top_value", str(counts.index[0])),
                stat_row("categorical", column, "top_value_count", int(counts.iloc[0])),
                stat_row("categorical", column, "top_value_percent", pct(counts.iloc[0], total)),
            ]
        )


def add_date_quality_statistics(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add invalid-date counts and percentages."""
    total = len(frame)
    for column in DATE_FLAG_COLUMNS:
        values = frame[column].fillna(False).astype(bool)
        invalid_count = int(values.sum())
        rows.extend(
            [
                stat_row("date_quality", column, "invalid_count", invalid_count),
                stat_row("date_quality", column, "invalid_percent", pct(invalid_count, total)),
            ]
        )


def pct(part: Any, whole: Any) -> float:
    """Return a rounded percentage while handling empty denominators."""
    denominator = float(whole or 0)
    if denominator == 0:
        return 0.0
    return round(float(part or 0) / denominator * 100, 2)


def write_rows(rows: list[dict[str, Any]]) -> None:
    """Write descriptive statistics to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build the descriptive statistics table."""
    frame = load_data()
    rows: list[dict[str, Any]] = []
    add_dataset_overview(rows, frame)
    add_numeric_statistics(rows, frame)
    add_binary_statistics(rows, frame)
    add_category_statistics(rows, frame)
    add_date_quality_statistics(rows, frame)
    write_rows(rows)
    print(f"Wrote descriptive statistics to {OUTPUT_PATH}", flush=True)
    print(f"Rows checked: {len(frame):,}", flush=True)
    print(f"Statistic rows: {len(rows):,}", flush=True)


if __name__ == "__main__":
    main()
