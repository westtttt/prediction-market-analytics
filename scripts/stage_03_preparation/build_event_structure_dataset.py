"""Build an event-level structure dataset for aggregation and normalisation."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "data" / "outputs" / "analysis"
MARKETS_PATH = ANALYSIS_DIR / "markets_analysis.parquet"
CATEGORIES_PATH = ANALYSIS_DIR / "market_categories.parquet"
OUTPUT_PATH = ANALYSIS_DIR / "event_structure.parquet"
SUMMARY_PATH = ANALYSIS_DIR / "event_structure_summary.csv"

SECONDS_PER_DAY = 86_400
STALE_PRICE_DAYS = 7

MARKET_COLUMNS = [
    "event_id",
    "event_title",
    "market_id",
    "market_question",
    "market_type",
    "group_item_title",
    "group_item_threshold",
    "outcome_count",
    "outcome_names",
    "current_prices",
    "has_resolved_winner",
    "winner_outcome_names",
    "market_volume",
    "has_price_history",
    "price_points",
    "last_price_timestamp",
    "market_closed_time",
    "market_end_date",
]


def require_inputs() -> None:
    """Fail clearly if required generated inputs are missing."""
    missing = [path for path in [MARKETS_PATH, CATEGORIES_PATH] if not path.exists()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing input file(s): {paths}")


def split_semicolon(value: Any) -> list[str]:
    """Split a semicolon-separated artifact field."""
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def parse_float_list(value: Any) -> list[float | None]:
    """Parse semicolon-separated probability text into numeric values."""
    values: list[float | None] = []
    for part in split_semicolon(value):
        try:
            parsed = float(part)
        except ValueError:
            values.append(None)
            continue
        values.append(parsed if math.isfinite(parsed) else None)
    return values


def yes_probability(outcome_names: Any, current_prices: Any) -> float | None:
    """Return the current Yes probability when a market has a Yes outcome."""
    outcomes = [part.lower() for part in split_semicolon(outcome_names)]
    prices = parse_float_list(current_prices)
    if "yes" not in outcomes:
        return None
    index = outcomes.index("yes")
    if index >= len(prices):
        return None
    probability = prices[index]
    if probability is None or probability < 0 or probability > 1:
        return None
    return probability


def has_yes_no_outcomes(outcome_names: Any, outcome_count: Any) -> bool:
    """Return whether the market is a binary Yes/No market."""
    outcomes = {part.lower() for part in split_semicolon(outcome_names)}
    try:
        count = int(outcome_count)
    except (TypeError, ValueError):
        count = 0
    return count == 2 and {"yes", "no"}.issubset(outcomes)


def yes_resolved_winner(winner_outcome_names: Any) -> int:
    """Return 1 when Yes is listed as a resolved winner."""
    winners = {part.lower() for part in split_semicolon(winner_outcome_names)}
    return int("yes" in winners)


def infer_event_type(title: Any) -> str:
    """Assign a coarse event type from transparent title keywords."""
    lower = "" if pd.isna(title) else str(title).lower()
    if "player props" in lower or lower.endswith(" props") or " - props" in lower:
        return "player_props_related"
    if "exact score" in lower:
        return "exact_score"
    if "# of tweets" in lower or "# tweets" in lower or "number of tweets" in lower:
        return "count_range"
    if "price on" in lower or "price of" in lower or "bitcoin price" in lower or "ethereum price" in lower:
        return "price_range"
    if "winner" in lower or "nominee" in lower or "champion" in lower:
        return "winner_market"
    if "election" in lower or "president" in lower or "senate" in lower:
        return "politics_or_election"
    if " vs. " in lower or " vs " in lower:
        return "match_or_head_to_head"
    return "other"


def classification_reason(
    aggregation_label: str,
    event_type: str,
    yes_no_market_count: int,
    yes_winner_count: int,
    raw_probability_sum: float | None,
    absolute_distance_from_one: float | None,
) -> str:
    """Return a short reason for the event-structure label."""
    if yes_no_market_count < 2:
        return "Fewer than two binary Yes/No sibling markets."
    if aggregation_label == "uncertain_needs_review":
        if raw_probability_sum is not None and raw_probability_sum >= 3:
            return "Raw Yes probability sum is far above one, but resolution structure is ambiguous."
        return "Event structure is ambiguous and should be reviewed before normalisation."
    if event_type == "player_props_related":
        return "Player-prop events can contain multiple true propositions."
    if yes_winner_count > 1:
        return "Resolved event has multiple Yes winners."
    if raw_probability_sum is not None and raw_probability_sum >= 3:
        return "Raw Yes probability sum is far above one."
    if aggregation_label == "mutually_exclusive_distribution":
        if absolute_distance_from_one is not None and absolute_distance_from_one <= 0.10:
            return "Binary sibling probabilities sum close to one."
        return "Resolved event has exactly one Yes winner and a compatible event type."
    if aggregation_label == "related_non_exclusive":
        return "Event appears to group related but non-exclusive propositions."
    return "Event is not suitable for event-family normalisation."


def classify_event(row: pd.Series) -> tuple[str, str, str, bool]:
    """Classify one event's suitability for probability normalisation."""
    event_type = str(row["inferred_event_type"])
    yes_no_market_count = int(row["yes_no_market_count"])
    yes_priced_market_count = int(row["yes_priced_market_count"])
    yes_winner_count = int(row["yes_winner_count"])
    resolved_yes_no_market_count = int(row["resolved_yes_no_market_count"])
    raw_probability_sum = row["raw_yes_probability_sum"]
    absolute_distance = row["absolute_distance_from_one"]
    near_one = bool(row["probability_sum_within_10pct"])
    single_winner = resolved_yes_no_market_count > 0 and yes_winner_count == 1
    stale_warning = bool(row["stale_price_warning"])

    if yes_no_market_count < 2 or yes_priced_market_count < 2:
        label = "unsuitable_for_normalisation"
        confidence = "high"
    elif event_type == "player_props_related":
        label = "related_non_exclusive"
        confidence = "high"
    elif yes_winner_count > 1:
        label = "related_non_exclusive"
        confidence = "high"
    elif raw_probability_sum is not None and raw_probability_sum >= 3 and not single_winner:
        label = "related_non_exclusive"
        confidence = "medium"
    elif event_type in {"exact_score", "price_range", "count_range", "winner_market"} and (near_one or single_winner):
        label = "mutually_exclusive_distribution"
        confidence = "medium" if stale_warning else "high"
    elif single_winner and absolute_distance is not None and absolute_distance <= 0.25:
        label = "mutually_exclusive_distribution"
        confidence = "medium" if stale_warning else "high"
    elif near_one and event_type not in {"match_or_head_to_head", "player_props_related"}:
        label = "uncertain_needs_review"
        confidence = "medium" if not stale_warning else "low"
    else:
        label = "uncertain_needs_review"
        confidence = "low" if event_type == "other" else "medium"

    can_normalise = label == "mutually_exclusive_distribution" and yes_priced_market_count >= 2
    reason = classification_reason(
        label,
        event_type,
        yes_no_market_count,
        yes_winner_count,
        raw_probability_sum,
        absolute_distance,
    )
    return label, confidence, reason, can_normalise


def pct(part: Any, whole: Any) -> float:
    """Return a rounded percentage while handling empty denominators."""
    denominator = float(whole or 0)
    if denominator == 0:
        return 0.0
    return round(float(part or 0) / denominator * 100, 2)


def mode_or_other(series: pd.Series) -> str:
    """Return the most common non-null category value."""
    values = series.dropna().astype(str)
    if values.empty:
        return "Other"
    return str(values.mode().iloc[0])


def build_event_structure(markets: pd.DataFrame) -> pd.DataFrame:
    """Build one event-level structure row per event."""
    frame = markets.copy()
    frame["event_id"] = frame["event_id"].astype("string")
    frame["yes_no_market"] = frame.apply(
        lambda row: has_yes_no_outcomes(row["outcome_names"], row["outcome_count"]),
        axis=1,
    )
    frame["yes_probability"] = frame.apply(
        lambda row: yes_probability(row["outcome_names"], row["current_prices"]),
        axis=1,
    )
    frame["yes_priced_market"] = frame["yes_no_market"] & frame["yes_probability"].notna()
    frame["yes_resolved_winner"] = frame["winner_outcome_names"].map(yes_resolved_winner)
    frame["resolved_yes_no_market"] = frame["yes_no_market"] & frame["has_resolved_winner"].fillna(False).astype(bool)
    frame["last_price_timestamp"] = pd.to_numeric(frame["last_price_timestamp"], errors="coerce")
    frame["market_volume"] = pd.to_numeric(frame["market_volume"], errors="coerce").fillna(0)
    frame["price_points"] = pd.to_numeric(frame["price_points"], errors="coerce").fillna(0)
    frame["market_reference_time"] = pd.to_datetime(
        frame["market_closed_time"].fillna(frame["market_end_date"]),
        errors="coerce",
        utc=True,
    )
    frame["market_reference_timestamp"] = frame["market_reference_time"].astype("int64") // 10**9
    frame.loc[frame["market_reference_time"].isna(), "market_reference_timestamp"] = pd.NA

    grouped = frame.groupby("event_id", dropna=False)
    events = grouped.agg(
        event_title=("event_title", "first"),
        primary_category=("clean_category", mode_or_other),
        market_count=("market_id", "count"),
        yes_no_market_count=("yes_no_market", "sum"),
        yes_priced_market_count=("yes_priced_market", "sum"),
        resolved_yes_no_market_count=("resolved_yes_no_market", "sum"),
        yes_winner_count=("yes_resolved_winner", "sum"),
        markets_with_price_history=("has_price_history", "sum"),
        total_price_points=("price_points", "sum"),
        total_market_volume=("market_volume", "sum"),
        raw_yes_probability_sum=("yes_probability", "sum"),
        favourite_yes_probability=("yes_probability", "max"),
        earliest_latest_price_timestamp=("last_price_timestamp", "min"),
        latest_latest_price_timestamp=("last_price_timestamp", "max"),
        event_reference_timestamp=("market_reference_timestamp", "max"),
    ).reset_index()

    no_priced = events["yes_priced_market_count"].eq(0)
    events.loc[no_priced, ["raw_yes_probability_sum", "favourite_yes_probability"]] = pd.NA
    events["distance_from_one"] = events["raw_yes_probability_sum"] - 1
    events["absolute_distance_from_one"] = events["distance_from_one"].abs()
    events["probability_sum_within_5pct"] = events["absolute_distance_from_one"].le(0.05)
    events["probability_sum_within_10pct"] = events["absolute_distance_from_one"].le(0.10)
    events.loc[no_priced, ["probability_sum_within_5pct", "probability_sum_within_10pct"]] = False
    events["latest_timestamp_spread_hours"] = (
        (events["latest_latest_price_timestamp"] - events["earliest_latest_price_timestamp"]) / 3_600
    ).round(2)
    events.loc[events["yes_priced_market_count"].lt(2), "latest_timestamp_spread_hours"] = pd.NA

    stale_cutoff = events["event_reference_timestamp"] - (STALE_PRICE_DAYS * SECONDS_PER_DAY)
    events["stale_price_warning"] = (
        events["event_reference_timestamp"].notna()
        & events["latest_latest_price_timestamp"].notna()
        & events["latest_latest_price_timestamp"].lt(stale_cutoff)
    )
    events["price_history_coverage_percent"] = events.apply(
        lambda row: pct(row["markets_with_price_history"], row["market_count"]),
        axis=1,
    )
    events["yes_price_coverage_percent"] = events.apply(
        lambda row: pct(row["yes_priced_market_count"], row["yes_no_market_count"]),
        axis=1,
    )
    events["is_resolved_single_winner_event"] = (
        events["resolved_yes_no_market_count"].gt(0) & events["yes_winner_count"].eq(1)
    )
    events["is_potential_normalisation_candidate"] = (
        events["yes_priced_market_count"].ge(2)
        & (
            events["probability_sum_within_10pct"]
            | events["is_resolved_single_winner_event"]
        )
    )
    events["inferred_event_type"] = events["event_title"].map(infer_event_type)

    classifications = events.apply(classify_event, axis=1, result_type="expand")
    classifications.columns = [
        "aggregation_label",
        "classification_confidence",
        "classification_reason",
        "can_normalise",
    ]
    events = pd.concat([events, classifications], axis=1)

    ordered_columns = [
        "event_id",
        "event_title",
        "primary_category",
        "inferred_event_type",
        "aggregation_label",
        "classification_confidence",
        "classification_reason",
        "can_normalise",
        "market_count",
        "yes_no_market_count",
        "yes_priced_market_count",
        "yes_price_coverage_percent",
        "resolved_yes_no_market_count",
        "yes_winner_count",
        "is_resolved_single_winner_event",
        "is_potential_normalisation_candidate",
        "raw_yes_probability_sum",
        "distance_from_one",
        "absolute_distance_from_one",
        "probability_sum_within_5pct",
        "probability_sum_within_10pct",
        "favourite_yes_probability",
        "markets_with_price_history",
        "price_history_coverage_percent",
        "total_price_points",
        "total_market_volume",
        "earliest_latest_price_timestamp",
        "latest_latest_price_timestamp",
        "latest_timestamp_spread_hours",
        "event_reference_timestamp",
        "stale_price_warning",
    ]
    return events.loc[:, ordered_columns].sort_values(
        ["can_normalise", "market_count", "total_market_volume"],
        ascending=[False, False, False],
    )


def summary_row(section: str, group: str, metric: str, value: Any, notes: str = "") -> dict[str, Any]:
    """Return one standard summary row."""
    return {
        "section": section,
        "group": group,
        "metric": metric,
        "value": value,
        "notes": notes,
    }


def build_summary(events: pd.DataFrame) -> list[dict[str, Any]]:
    """Build compact event-structure summary rows."""
    rows = [
        summary_row("coverage", "all", "events", len(events)),
        summary_row("coverage", "all", "events_with_multiple_markets", int(events["market_count"].ge(2).sum())),
        summary_row("coverage", "all", "events_with_at_least_five_markets", int(events["market_count"].ge(5).sum())),
        summary_row("coverage", "all", "events_with_yes_no_siblings", int(events["yes_no_market_count"].ge(2).sum())),
        summary_row("coverage", "all", "normalisable_events", int(events["can_normalise"].sum())),
        summary_row(
            "coverage",
            "all",
            "normalisable_event_percent",
            pct(events["can_normalise"].sum(), len(events)),
        ),
    ]

    for label, frame in events.groupby("aggregation_label", dropna=False, sort=True):
        rows.extend(
            [
                summary_row("aggregation_label", str(label), "events", len(frame)),
                summary_row("aggregation_label", str(label), "market_count_median", round(float(frame["market_count"].median()), 2)),
                summary_row("aggregation_label", str(label), "yes_no_market_count_median", round(float(frame["yes_no_market_count"].median()), 2)),
            ]
        )

    for event_type, frame in events.groupby("inferred_event_type", dropna=False, sort=True):
        rows.append(summary_row("event_type", str(event_type), "events", len(frame)))

    return rows


def write_summary(rows: list[dict[str, Any]]) -> None:
    """Write summary rows to CSV."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_markets() -> pd.DataFrame:
    """Load market-level analysis rows and cleaned categories."""
    markets = pd.read_parquet(MARKETS_PATH, columns=MARKET_COLUMNS)
    categories = pd.read_parquet(CATEGORIES_PATH, columns=["market_id", "clean_category"])
    markets["market_id"] = markets["market_id"].astype("string")
    categories["market_id"] = categories["market_id"].astype("string")
    markets = markets.merge(categories, on="market_id", how="left", validate="one_to_one")
    markets["clean_category"] = markets["clean_category"].fillna("Other")
    return markets


def main() -> None:
    """Build event-level structure outputs."""
    require_inputs()
    markets = load_markets()
    events = build_event_structure(markets)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(OUTPUT_PATH, index=False)
    write_summary(build_summary(events))

    print(f"Wrote event structure dataset to {OUTPUT_PATH}", flush=True)
    print(f"Wrote event structure summary to {SUMMARY_PATH}", flush=True)
    print(f"Events: {len(events):,}", flush=True)
    print(f"Normalisable events: {int(events['can_normalise'].sum()):,}", flush=True)


if __name__ == "__main__":
    main()
