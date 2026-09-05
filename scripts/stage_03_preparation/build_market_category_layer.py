"""Build a cleaned market category layer from raw Polymarket tags."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "outputs" / "analysis" / "markets_analysis.parquet"
RULES_PATH = ROOT / "config" / "category_rules.csv"
OUTPUT_PATH = ROOT / "data" / "outputs" / "analysis" / "market_categories.parquet"
SUMMARY_PATH = ROOT / "data" / "outputs" / "analysis" / "market_category_summary.csv"

LOAD_COLUMNS = [
    "market_id",
    "event_id",
    "market_question",
    "event_title",
    "primary_tag",
    "tag_labels",
    "tag_slugs",
    "market_volume",
    "has_price_history",
    "is_resolved",
]


@dataclass(frozen=True)
class CategoryRule:
    """One ordered category assignment rule."""

    priority: int
    category: str
    keywords: tuple[str, ...]
    notes: str


def output_row(area: str, group: str, metric: str, value: Any, notes: str = "") -> dict[str, Any]:
    """Return one standard summary row."""
    return {
        "area": area,
        "group": group,
        "metric": metric,
        "value": value,
        "notes": notes,
    }


def require_inputs() -> None:
    """Fail clearly if required inputs are missing."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Analysis dataset not found: {INPUT_PATH}. "
            "Run scripts/stage_03_preparation/build_market_analysis_dataset.py first."
        )
    if not RULES_PATH.exists():
        raise FileNotFoundError(f"Category rules not found: {RULES_PATH}")


def load_rules() -> list[CategoryRule]:
    """Load ordered category rules from the tracked CSV config."""
    rules: list[CategoryRule] = []
    with RULES_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            keywords = tuple(
                keyword.strip().lower()
                for keyword in record["keywords"].split(";")
                if keyword.strip()
            )
            rules.append(
                CategoryRule(
                    priority=int(record["priority"]),
                    category=record["category"].strip(),
                    keywords=keywords,
                    notes=record.get("notes", "").strip(),
                )
            )
    return sorted(rules, key=lambda rule: rule.priority)


def normalise_text(frame: pd.DataFrame) -> pd.Series:
    """Combine useful text fields into one lower-case matching string."""
    text_columns = ["primary_tag", "tag_labels", "tag_slugs", "event_title", "market_question"]
    combined = frame[text_columns].fillna("").astype("string").agg(" | ".join, axis=1)
    return combined.str.lower()


def keyword_pattern(keywords: tuple[str, ...]) -> str:
    """Build one escaped regex pattern for a category's keywords."""
    return "|".join(re.escape(keyword) for keyword in keywords)


def assign_categories(frame: pd.DataFrame, rules: list[CategoryRule]) -> pd.DataFrame:
    """Assign the first matching category rule to each market."""
    result = frame[
        [
            "market_id",
            "event_id",
            "primary_tag",
            "tag_labels",
            "tag_slugs",
            "market_volume",
            "has_price_history",
            "is_resolved",
        ]
    ].copy()
    result["clean_category"] = "Other"
    result["category_rule"] = "no_rule_matched"
    result["category_priority"] = pd.NA

    text = normalise_text(frame)
    unmatched = pd.Series(True, index=frame.index)

    for rule in rules:
        pattern = keyword_pattern(rule.keywords)
        matches = unmatched & text.str.contains(pattern, regex=True, na=False)
        result.loc[matches, "clean_category"] = rule.category
        result.loc[matches, "category_rule"] = "; ".join(rule.keywords)
        result.loc[matches, "category_priority"] = rule.priority
        unmatched &= ~matches

    result["category_priority"] = pd.to_numeric(result["category_priority"], errors="coerce").astype("Int64")
    return result


def build_summary(categories: pd.DataFrame) -> list[dict[str, Any]]:
    """Build compact category coverage and activity summaries."""
    rows: list[dict[str, Any]] = []
    total_markets = len(categories)
    total_volume = float(pd.to_numeric(categories["market_volume"], errors="coerce").fillna(0).sum())

    grouped = (
        categories.groupby("clean_category", dropna=False)
        .agg(
            markets=("market_id", "count"),
            events=("event_id", "nunique"),
            volume=("market_volume", "sum"),
            price_history_markets=("has_price_history", "sum"),
            resolved_markets=("is_resolved", "sum"),
        )
        .reset_index()
        .sort_values("markets", ascending=False)
    )

    for _, item in grouped.iterrows():
        category = str(item["clean_category"])
        markets = int(item["markets"])
        volume = float(item["volume"] or 0)
        rows.extend(
            [
                output_row("category", category, "markets", markets, pct(markets, total_markets)),
                output_row("category", category, "events", int(item["events"])),
                output_row("category", category, "volume", round(volume, 2), pct(volume, total_volume)),
                output_row(
                    "category",
                    category,
                    "price_history_percent",
                    pct(item["price_history_markets"], markets),
                ),
                output_row(
                    "category",
                    category,
                    "resolved_percent",
                    pct(item["resolved_markets"], markets),
                ),
            ]
        )

    other_count = int((categories["clean_category"] == "Other").sum())
    rows.append(output_row("coverage", "Other", "market_percent", pct(other_count, total_markets)))
    return rows


def pct(part: Any, whole: Any) -> float:
    """Return a rounded percentage while handling empty denominators."""
    denominator = float(whole or 0)
    if denominator == 0:
        return 0.0
    return round(float(part or 0) / denominator * 100, 2)


def write_summary(rows: list[dict[str, Any]]) -> None:
    """Write category summary rows to CSV."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build the cleaned category layer and summary."""
    require_inputs()
    rules = load_rules()
    frame = pd.read_parquet(INPUT_PATH, columns=LOAD_COLUMNS)
    categories = assign_categories(frame, rules)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    categories.to_parquet(OUTPUT_PATH, index=False)
    summary_rows = build_summary(categories)
    write_summary(summary_rows)

    other_count = int((categories["clean_category"] == "Other").sum())
    print(f"Wrote category layer to {OUTPUT_PATH}", flush=True)
    print(f"Wrote category summary to {SUMMARY_PATH}", flush=True)
    print(f"Rows categorised: {len(categories):,}", flush=True)
    print(f"Unmatched/Other: {other_count:,} ({pct(other_count, len(categories))}%)", flush=True)


if __name__ == "__main__":
    main()
