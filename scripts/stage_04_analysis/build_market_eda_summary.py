"""Create compact EDA summaries from the market-level analysis dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "outputs" / "analysis" / "markets_analysis.parquet"
CATEGORY_PATH = ROOT / "data" / "outputs" / "analysis" / "market_categories.parquet"
OUTPUT_PATH = ROOT / "data" / "outputs" / "analysis" / "market_eda_summary.csv"

TOP_N = 25
VOLUME_SHARE_LEVELS = [0.01, 0.05, 0.10]
NOISY_TAGS = {
    "1H",
    "4H",
    "5M",
    "15M",
    "Daily Temperature",
    "Games",
    "All",
}

LOAD_COLUMNS = [
    "market_id",
    "event_id",
    "primary_tag",
    "tag_count",
    "tag_labels",
    "market_created_month",
    "market_created_year",
    "market_type",
    "outcome_count",
    "market_volume",
    "market_liquidity",
    "price_points",
    "days_open",
    "has_volume",
    "has_liquidity",
    "has_price_history",
    "is_market_active",
    "is_market_closed",
    "is_resolved",
    "last_trade_price",
    "price_mean",
    "market_start_date_invalid",
    "market_end_date_invalid",
    "market_closed_time_invalid",
]


def row(area: str, group: str, metric: str, value: Any, notes: str = "") -> dict[str, Any]:
    """Return one standard EDA summary row."""
    return {
        "area": area,
        "group": group,
        "metric": metric,
        "value": value,
        "notes": notes,
    }


def require_input() -> None:
    """Fail clearly if the analysis dataset is missing."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Analysis dataset not found: {INPUT_PATH}. "
            "Run scripts/stage_03_preparation/build_market_analysis_dataset.py first."
        )
    if not CATEGORY_PATH.exists():
        raise FileNotFoundError(
            f"Category dataset not found: {CATEGORY_PATH}. "
            "Run scripts/stage_03_preparation/build_market_category_layer.py first."
        )


def load_dataset() -> pd.DataFrame:
    """Load the columns needed for the first EDA pass."""
    require_input()
    frame = pd.read_parquet(INPUT_PATH, columns=LOAD_COLUMNS)
    categories = pd.read_parquet(
        CATEGORY_PATH,
        columns=["market_id", "clean_category", "category_rule", "category_priority"],
    )
    return frame.merge(categories, on="market_id", how="left", validate="one_to_one")


def add_overview(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add overall dataset composition and coverage metrics."""
    rows.extend(
        [
            row("overview", "all", "markets", len(frame)),
            row("overview", "all", "events", int(frame["event_id"].nunique())),
            row("overview", "all", "primary_tags", int(frame["primary_tag"].nunique(dropna=True))),
            row("overview", "all", "clean_categories", int(frame["clean_category"].nunique(dropna=True))),
            row("overview", "all", "market_types", int(frame["market_type"].nunique(dropna=True))),
            row("overview", "all", "resolved_percent", pct(frame["is_resolved"].sum(), len(frame))),
            row("overview", "all", "price_history_percent", pct(frame["has_price_history"].sum(), len(frame))),
            row("overview", "all", "positive_volume_percent", pct(frame["has_volume"].sum(), len(frame))),
            row("overview", "all", "positive_liquidity_percent", pct(frame["has_liquidity"].sum(), len(frame))),
        ]
    )


def add_top_primary_tags(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add top primary tags by market count and volume."""
    grouped = (
        frame.groupby("primary_tag", dropna=False)
        .agg(
            markets=("market_id", "count"),
            events=("event_id", "nunique"),
            volume=("market_volume", "sum"),
            median_volume=("market_volume", "median"),
            price_history_markets=("has_price_history", "sum"),
            resolved_markets=("is_resolved", "sum"),
            liquidity_markets=("has_liquidity", "sum"),
        )
        .reset_index()
    )
    grouped["primary_tag"] = grouped["primary_tag"].fillna("(missing)")

    for sort_column, area in [("markets", "top_tags_by_count"), ("volume", "top_tags_by_volume")]:
        top = grouped.sort_values(sort_column, ascending=False).head(TOP_N)
        for _, item in top.iterrows():
            tag = str(item["primary_tag"])
            notes = "possible noisy/operational tag" if tag in NOISY_TAGS else ""
            rows.extend(
                [
                    row(area, tag, "markets", int(item["markets"]), notes),
                    row(area, tag, "events", int(item["events"]), notes),
                    row(area, tag, "volume", round(float(item["volume"] or 0), 2), notes),
                    row(
                        area,
                        tag,
                        "price_history_percent",
                        pct(item["price_history_markets"], item["markets"]),
                        notes,
                    ),
                    row(
                        area,
                        tag,
                        "resolved_percent",
                        pct(item["resolved_markets"], item["markets"]),
                        notes,
                    ),
                    row(
                        area,
                        tag,
                        "positive_liquidity_percent",
                        pct(item["liquidity_markets"], item["markets"]),
                        notes,
                    ),
                ]
            )


def add_clean_categories(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add category summaries using the cleaned category layer."""
    grouped = (
        frame.groupby("clean_category", dropna=False)
        .agg(
            markets=("market_id", "count"),
            events=("event_id", "nunique"),
            volume=("market_volume", "sum"),
            median_volume=("market_volume", "median"),
            price_history_markets=("has_price_history", "sum"),
            resolved_markets=("is_resolved", "sum"),
            liquidity_markets=("has_liquidity", "sum"),
            median_price_points=("price_points", "median"),
        )
        .reset_index()
    )
    grouped["clean_category"] = grouped["clean_category"].fillna("(missing)")

    for sort_column, area in [
        ("markets", "clean_categories_by_count"),
        ("volume", "clean_categories_by_volume"),
    ]:
        top = grouped.sort_values(sort_column, ascending=False)
        for _, item in top.iterrows():
            category = str(item["clean_category"])
            rows.extend(
                [
                    row(area, category, "markets", int(item["markets"])),
                    row(area, category, "events", int(item["events"])),
                    row(area, category, "volume", round(float(item["volume"] or 0), 2)),
                    row(area, category, "median_volume", round(float(item["median_volume"] or 0), 2)),
                    row(
                        area,
                        category,
                        "price_history_percent",
                        pct(item["price_history_markets"], item["markets"]),
                    ),
                    row(
                        area,
                        category,
                        "resolved_percent",
                        pct(item["resolved_markets"], item["markets"]),
                    ),
                    row(
                        area,
                        category,
                        "positive_liquidity_percent",
                        pct(item["liquidity_markets"], item["markets"]),
                    ),
                    row(
                        area,
                        category,
                        "median_price_points",
                        round(float(item["median_price_points"] or 0), 2),
                    ),
                ]
            )


def add_monthly_category_trends(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add monthly market counts for the largest cleaned categories."""
    top_categories = (
        frame["clean_category"]
        .fillna("(missing)")
        .value_counts()
        .head(8)
        .index
        .tolist()
    )
    trend_frame = frame[
        frame["clean_category"].isin(top_categories)
        & frame["market_created_month"].notna()
    ]
    monthly = (
        trend_frame.groupby(["market_created_month", "clean_category"])
        .agg(markets=("market_id", "count"), volume=("market_volume", "sum"))
        .reset_index()
        .sort_values(["market_created_month", "clean_category"])
    )
    for _, item in monthly.iterrows():
        group = f"{item['market_created_month']}|{item['clean_category']}"
        rows.append(row("monthly_category_trend", group, "markets", int(item["markets"])))
        rows.append(row("monthly_category_trend", group, "volume", round(float(item["volume"] or 0), 2)))


def add_monthly_trends(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add market creation trends by month."""
    monthly = (
        frame.dropna(subset=["market_created_month"])
        .groupby("market_created_month")
        .agg(
            markets=("market_id", "count"),
            volume=("market_volume", "sum"),
            price_history_markets=("has_price_history", "sum"),
            resolved_markets=("is_resolved", "sum"),
        )
        .reset_index()
        .sort_values("market_created_month")
    )
    for _, item in monthly.iterrows():
        group = str(item["market_created_month"])
        rows.extend(
            [
                row("monthly_trend", group, "markets", int(item["markets"])),
                row("monthly_trend", group, "volume", round(float(item["volume"] or 0), 2)),
                row(
                    "monthly_trend",
                    group,
                    "price_history_percent",
                    pct(item["price_history_markets"], item["markets"]),
                ),
                row(
                    "monthly_trend",
                    group,
                    "resolved_percent",
                    pct(item["resolved_markets"], item["markets"]),
                ),
            ]
        )


def add_volume_concentration(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add concentration metrics showing whether activity is dominated by outliers."""
    volume = pd.to_numeric(frame["market_volume"], errors="coerce").fillna(0)
    positive = volume[volume > 0].sort_values(ascending=False)
    total = float(positive.sum())
    rows.append(row("volume_concentration", "positive_volume_markets", "count", int(len(positive))))
    rows.append(row("volume_concentration", "all", "total_volume", round(total, 2)))
    if total <= 0 or positive.empty:
        return

    for level in VOLUME_SHARE_LEVELS:
        count = max(1, int(len(positive) * level))
        share = float(positive.head(count).sum() / total * 100)
        rows.append(
            row(
                "volume_concentration",
                f"top_{int(level * 100)}_percent",
                "volume_share_percent",
                round(share, 2),
                f"{count:,} markets",
            )
        )


def add_numeric_profiles(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add numeric summaries for variables likely to become dashboard measures."""
    for column in ["market_volume", "market_liquidity", "price_points", "days_open", "tag_count"]:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        rows.extend(
            [
                row("numeric_profile", column, "non_null", int(len(series))),
                row("numeric_profile", column, "median", round(float(series.median()), 4)),
                row("numeric_profile", column, "p90", round(float(series.quantile(0.90)), 4)),
                row("numeric_profile", column, "p95", round(float(series.quantile(0.95)), 4)),
                row("numeric_profile", column, "p99", round(float(series.quantile(0.99)), 4)),
                row("numeric_profile", column, "max", round(float(series.max()), 4)),
            ]
        )


def add_cleaning_signals(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add EDA-driven cleaning signals that should shape later analysis choices."""
    total = len(frame)
    for column in [
        "market_start_date_invalid",
        "market_end_date_invalid",
        "market_closed_time_invalid",
    ]:
        count = int(frame[column].fillna(False).sum())
        rows.append(row("cleaning_signal", column, "invalid_percent", pct(count, total), f"{count:,} markets"))

    noisy_count = int(frame["primary_tag"].isin(NOISY_TAGS).sum())
    rows.append(
        row(
            "cleaning_signal",
            "noisy_primary_tags",
            "market_percent",
            pct(noisy_count, total),
            "tags that may need grouping or filtering for category-level EDA",
        )
    )

    other_count = int(frame["clean_category"].fillna("Other").eq("Other").sum())
    rows.append(
        row(
            "cleaning_signal",
            "clean_category_other",
            "market_percent",
            pct(other_count, total),
            f"{other_count:,} markets",
        )
    )

    zero_or_missing_volume = int((pd.to_numeric(frame["market_volume"], errors="coerce").fillna(0) <= 0).sum())
    rows.append(
        row(
            "cleaning_signal",
            "zero_or_missing_volume",
            "market_percent",
            pct(zero_or_missing_volume, total),
            f"{zero_or_missing_volume:,} markets",
        )
    )


def add_market_type_and_outcomes(rows: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    """Add simple composition checks for market type and outcome counts."""
    for value, count in frame["market_type"].fillna("(missing)").value_counts().head(TOP_N).items():
        rows.append(row("market_type", str(value), "markets", int(count), f"{pct(count, len(frame))}% of markets"))

    for value, count in frame["outcome_count"].fillna("(missing)").value_counts().sort_index().items():
        rows.append(row("outcome_count", str(value), "markets", int(count), f"{pct(count, len(frame))}% of markets"))


def pct(part: Any, whole: Any) -> float:
    """Return a rounded percentage while handling empty denominators."""
    denominator = float(whole or 0)
    if denominator == 0:
        return 0.0
    return round(float(part or 0) / denominator * 100, 2)


def write_summary(rows: list[dict[str, Any]]) -> None:
    """Write the EDA summary CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the first compact EDA pass."""
    frame = load_dataset()
    rows: list[dict[str, Any]] = []

    add_overview(rows, frame)
    add_top_primary_tags(rows, frame)
    add_clean_categories(rows, frame)
    add_monthly_trends(rows, frame)
    add_monthly_category_trends(rows, frame)
    add_volume_concentration(rows, frame)
    add_numeric_profiles(rows, frame)
    add_cleaning_signals(rows, frame)
    add_market_type_and_outcomes(rows, frame)
    write_summary(rows)

    print(f"Wrote EDA summary to {OUTPUT_PATH}", flush=True)
    print(f"Rows checked: {len(frame):,}", flush=True)
    print(f"Summary rows: {len(rows):,}", flush=True)


if __name__ == "__main__":
    main()
