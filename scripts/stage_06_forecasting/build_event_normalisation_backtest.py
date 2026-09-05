"""Backtest raw versus event-normalised probabilities for coherent event families."""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = ROOT / "data" / "database" / "markets.db"
ANALYSIS_DIR = ROOT / "data" / "outputs" / "analysis"
MARKETS_PATH = ANALYSIS_DIR / "markets_analysis.parquet"
EVENT_STRUCTURE_PATH = ANALYSIS_DIR / "event_structure.parquet"
PREDICTION_OUTPUT = ANALYSIS_DIR / "event_normalisation_backtest_predictions.parquet"
SUMMARY_OUTPUT = ANALYSIS_DIR / "event_normalisation_backtest_summary.csv"

HORIZON_DAYS = [30, 7, 3, 1]
MAX_STALENESS_DAYS = 7
CHUNK_SIZE = 50_000
EPSILON = 1e-15

MARKET_COLUMNS = [
    "event_id",
    "event_title",
    "market_id",
    "market_question",
    "outcome_count",
    "outcome_names",
    "winner_outcome_names",
    "has_resolved_winner",
    "has_price_history",
    "market_created_at",
    "market_volume",
    "price_points",
]

EVENT_COLUMNS = [
    "event_id",
    "primary_category",
    "inferred_event_type",
    "aggregation_label",
    "classification_confidence",
    "can_normalise",
    "event_reference_timestamp",
]


def clean_id(value: Any) -> str:
    """Normalise ids loaded from CSV/Parquet."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def split_semicolon(value: Any) -> list[str]:
    """Split a semicolon-separated artifact field."""
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def yes_outcome_index(outcome_names: Any) -> int | None:
    """Return the index of a Yes outcome, if present."""
    outcomes = [part.lower() for part in split_semicolon(outcome_names)]
    if "yes" not in outcomes:
        return None
    return outcomes.index("yes")


def yes_actual(winner_outcome_names: Any) -> int | None:
    """Return 1 if Yes won, 0 if another listed outcome won."""
    winners = [part.lower() for part in split_semicolon(winner_outcome_names)]
    if not winners:
        return None
    return int("yes" in winners)


def timestamp_seconds(series: pd.Series) -> pd.Series:
    """Convert datetime-like values to Unix seconds while preserving missing values."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    timestamps = parsed.astype("int64") // 10**9
    return timestamps.where(parsed.notna(), pd.NA)


def clip_probability(value: float) -> float:
    """Return a finite probability in the closed interval [0, 1]."""
    if not math.isfinite(float(value)):
        return math.nan
    return min(1.0, max(0.0, float(value)))


def brier_score(probability: float, actual: int) -> float:
    """Return squared probability error for one forecast."""
    p = clip_probability(probability)
    return (p - int(actual)) ** 2


def log_loss(probability: float, actual: int) -> float:
    """Return clipped binary log loss for one forecast."""
    p = min(1.0 - EPSILON, max(EPSILON, clip_probability(probability)))
    y = int(actual)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def require_inputs() -> None:
    """Fail clearly if required inputs are missing."""
    missing = [path for path in [DATABASE_PATH, MARKETS_PATH, EVENT_STRUCTURE_PATH] if not path.exists()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing input file(s): {paths}")


def load_targets(max_events: int | None = None) -> pd.DataFrame:
    """Load resolved Yes/No markets from normalisable event families."""
    events = pd.read_parquet(EVENT_STRUCTURE_PATH, columns=EVENT_COLUMNS)
    events["event_id"] = events["event_id"].map(clean_id)
    events = events[events["can_normalise"].astype(bool)].copy()
    events = events[events["event_reference_timestamp"].notna()].copy()
    if max_events is not None:
        events = events.sort_values(["classification_confidence", "event_reference_timestamp"]).head(max_events)

    markets = pd.read_parquet(MARKETS_PATH, columns=MARKET_COLUMNS)
    markets["event_id"] = markets["event_id"].map(clean_id)
    markets["market_id"] = markets["market_id"].map(clean_id)
    markets = markets.merge(events, on="event_id", how="inner", validate="many_to_one")
    markets["yes_index"] = markets["outcome_names"].map(yes_outcome_index)
    markets["actual"] = markets["winner_outcome_names"].map(yes_actual)
    markets["created_timestamp"] = timestamp_seconds(markets["market_created_at"])
    markets["market_volume"] = pd.to_numeric(markets["market_volume"], errors="coerce").fillna(0)
    markets["price_points"] = pd.to_numeric(markets["price_points"], errors="coerce").fillna(0)

    targets = markets[
        markets["yes_index"].notna()
        & markets["actual"].notna()
        & markets["has_resolved_winner"].fillna(False).astype(bool)
        & markets["has_price_history"].fillna(False).astype(bool)
        & markets["created_timestamp"].notna()
    ].copy()
    targets["yes_index"] = targets["yes_index"].astype("int64")
    targets["actual"] = targets["actual"].astype("int8")
    targets["event_reference_timestamp"] = pd.to_numeric(
        targets["event_reference_timestamp"], errors="coerce"
    ).astype("int64")
    targets["created_timestamp"] = pd.to_numeric(targets["created_timestamp"], errors="coerce").astype("int64")
    return targets


def build_cutoff_targets(targets: pd.DataFrame) -> pd.DataFrame:
    """Create one lookup target per market and horizon."""
    frames: list[pd.DataFrame] = []
    base_columns = [
        "event_id",
        "event_title",
        "market_id",
        "market_question",
        "primary_category",
        "inferred_event_type",
        "classification_confidence",
        "aggregation_label",
        "yes_index",
        "actual",
        "market_volume",
        "price_points",
        "created_timestamp",
        "event_reference_timestamp",
    ]
    for horizon in HORIZON_DAYS:
        frame = targets.loc[:, base_columns].copy()
        frame["horizon_days"] = horizon
        frame["as_of_timestamp"] = frame["event_reference_timestamp"] - horizon * 86_400
        frame = frame[frame["as_of_timestamp"].gt(frame["created_timestamp"])].copy()
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=base_columns + ["horizon_days", "as_of_timestamp", "target_id"])
    cutoff_targets = pd.concat(frames, ignore_index=True)
    cutoff_targets["target_id"] = range(len(cutoff_targets))
    return cutoff_targets


def lookup_prices_as_of(cutoff_targets: pd.DataFrame) -> pd.DataFrame:
    """Look up latest Yes prices observed at or before each cutoff."""
    lookup_columns = ["target_id", "market_id", "yes_index", "as_of_timestamp"]
    result_frames: list[pd.DataFrame] = []
    with sqlite3.connect(DATABASE_PATH) as connection:
        for start in range(0, len(cutoff_targets), CHUNK_SIZE):
            chunk = cutoff_targets.iloc[start : start + CHUNK_SIZE]
            rows = [
                (int(row.target_id), clean_id(row.market_id), int(row.yes_index), int(row.as_of_timestamp))
                for row in chunk.loc[:, lookup_columns].itertuples(index=False)
            ]
            cursor = connection.cursor()
            cursor.execute("drop table if exists targets")
            cursor.execute("create temp table targets (target_id integer, market_id text, outcome_index integer, as_of integer)")
            cursor.executemany("insert into targets values (?, ?, ?, ?)", rows)
            cursor.execute("create index idx_temp_targets on targets(market_id, outcome_index, as_of)")
            prices = pd.read_sql_query(
                """
                select
                    t.target_id,
                    ph.timestamp as forecast_timestamp,
                    ph.datetime_utc as forecast_time,
                    ph.price as raw_probability
                from targets t
                join price_history ph
                  on ph.market_id = t.market_id
                 and ph.outcome_index = t.outcome_index
                 and ph.timestamp = (
                    select max(p2.timestamp)
                    from price_history p2
                    where p2.market_id = t.market_id
                      and p2.outcome_index = t.outcome_index
                      and p2.timestamp <= t.as_of
                 )
                """,
                connection,
            )
            result_frames.append(prices)
            print(f"Looked up prices for {min(start + CHUNK_SIZE, len(cutoff_targets)):,}/{len(cutoff_targets):,} rows", flush=True)
    if not result_frames:
        return pd.DataFrame(columns=["target_id", "forecast_timestamp", "forecast_time", "raw_probability"])
    return pd.concat(result_frames, ignore_index=True)


def build_predictions(cutoff_targets: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Build raw and normalised prediction rows."""
    predictions = cutoff_targets.merge(prices, on="target_id", how="inner", validate="one_to_one")
    if predictions.empty:
        return predictions
    predictions["raw_probability"] = pd.to_numeric(predictions["raw_probability"], errors="coerce").map(clip_probability)
    predictions = predictions[predictions["raw_probability"].notna()].copy()
    predictions["price_age_days"] = (
        (predictions["as_of_timestamp"] - predictions["forecast_timestamp"]) / 86_400
    ).round(4)
    predictions = predictions[predictions["price_age_days"].le(MAX_STALENESS_DAYS)].copy()
    predictions["event_probability_sum_as_of"] = predictions.groupby(
        ["event_id", "horizon_days"], sort=False
    )["raw_probability"].transform("sum")
    predictions["event_priced_markets_as_of"] = predictions.groupby(
        ["event_id", "horizon_days"], sort=False
    )["market_id"].transform("count")
    predictions = predictions[
        predictions["event_priced_markets_as_of"].ge(2)
        & predictions["event_probability_sum_as_of"].gt(0)
    ].copy()
    predictions["normalised_probability"] = (
        predictions["raw_probability"] / predictions["event_probability_sum_as_of"]
    ).map(clip_probability)
    predictions["raw_brier_score"] = predictions.apply(
        lambda row: brier_score(row["raw_probability"], row["actual"]),
        axis=1,
    )
    predictions["normalised_brier_score"] = predictions.apply(
        lambda row: brier_score(row["normalised_probability"], row["actual"]),
        axis=1,
    )
    predictions["raw_log_loss"] = predictions.apply(
        lambda row: log_loss(row["raw_probability"], row["actual"]),
        axis=1,
    )
    predictions["normalised_log_loss"] = predictions.apply(
        lambda row: log_loss(row["normalised_probability"], row["actual"]),
        axis=1,
    )
    predictions["brier_delta"] = predictions["raw_brier_score"] - predictions["normalised_brier_score"]
    predictions["log_loss_delta"] = predictions["raw_log_loss"] - predictions["normalised_log_loss"]
    return predictions.drop(columns=["target_id"]).sort_values(["horizon_days", "event_id", "market_id"])


def pct_change(raw: float, normalised: float) -> float:
    """Return percentage improvement from raw to normalised."""
    if raw == 0:
        return 0.0
    return round((raw - normalised) / raw * 100, 2)


def summarise_group(frame: pd.DataFrame, section: str, group: str) -> dict[str, Any]:
    """Return one summary row for a prediction subset."""
    raw_brier = float(frame["raw_brier_score"].mean())
    normalised_brier = float(frame["normalised_brier_score"].mean())
    raw_loss = float(frame["raw_log_loss"].mean())
    normalised_loss = float(frame["normalised_log_loss"].mean())
    return {
        "section": section,
        "group": group,
        "prediction_rows": len(frame),
        "events": int(frame["event_id"].nunique()),
        "markets": int(frame["market_id"].nunique()),
        "actual_yes_rate": round(float(frame["actual"].mean()), 6),
        "mean_raw_probability": round(float(frame["raw_probability"].mean()), 6),
        "mean_normalised_probability": round(float(frame["normalised_probability"].mean()), 6),
        "mean_event_probability_sum": round(float(frame["event_probability_sum_as_of"].mean()), 6),
        "raw_brier_score": round(raw_brier, 6),
        "normalised_brier_score": round(normalised_brier, 6),
        "brier_improvement_percent": pct_change(raw_brier, normalised_brier),
        "raw_log_loss": round(raw_loss, 6),
        "normalised_log_loss": round(normalised_loss, 6),
        "log_loss_improvement_percent": pct_change(raw_loss, normalised_loss),
        "mean_price_age_days": round(float(frame["price_age_days"].mean()), 4),
    }


def build_summary(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """Build overall, horizon, category, and event-type summaries."""
    rows = [summarise_group(predictions, "overall", "all")]
    for horizon, horizon_frame in predictions.groupby("horizon_days", sort=True):
        rows.append(summarise_group(horizon_frame, "horizon", f"{horizon}_days"))
    for category, category_frame in predictions.groupby("primary_category", sort=True):
        if len(category_frame) >= 1_000:
            rows.append(summarise_group(category_frame, "category", str(category)))
    for event_type, type_frame in predictions.groupby("inferred_event_type", sort=True):
        if len(type_frame) >= 1_000:
            rows.append(summarise_group(type_frame, "event_type", str(event_type)))
    return rows


def write_summary(rows: list[dict[str, Any]]) -> None:
    """Write summary rows to CSV."""
    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_backtest(max_events: int | None = None) -> pd.DataFrame:
    """Run the event-normalisation backtest."""
    require_inputs()
    targets = load_targets(max_events=max_events)
    cutoff_targets = build_cutoff_targets(targets)
    print(f"Eligible resolved normalisable markets: {len(targets):,}", flush=True)
    print(f"Cutoff lookup rows: {len(cutoff_targets):,}", flush=True)
    prices = lookup_prices_as_of(cutoff_targets)
    predictions = build_predictions(cutoff_targets, prices)
    return predictions


def main() -> None:
    """Run and persist event-normalisation backtest outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-events", type=int, default=None, help="Optional event limit for quick local checks.")
    args = parser.parse_args()

    predictions = run_backtest(max_events=args.max_events)
    if predictions.empty:
        raise RuntimeError("No event-normalisation predictions were generated.")
    predictions.to_parquet(PREDICTION_OUTPUT, index=False)
    summary = build_summary(predictions)
    write_summary(summary)
    overall = summary[0]
    print(f"Wrote predictions to {PREDICTION_OUTPUT}", flush=True)
    print(f"Wrote summary to {SUMMARY_OUTPUT}", flush=True)
    print(f"Prediction rows: {len(predictions):,}", flush=True)
    print(
        "Overall Brier: "
        f"raw={overall['raw_brier_score']} "
        f"normalised={overall['normalised_brier_score']} "
        f"improvement={overall['brier_improvement_percent']}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
