"""Validate current and historical forecasting output invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "data" / "outputs" / "analysis"
CURRENT_PATH = ANALYSIS_DIR / "current_forecast_examples.json"
FORECAST_PATH = ANALYSIS_DIR / "aggregation_backtest_forecasts.parquet"
EVALUATION_PATH = ANALYSIS_DIR / "aggregation_backtest_evaluation.parquet"
EVENT_NORMALISATION_PATH = ANALYSIS_DIR / "event_normalisation_backtest_predictions.parquet"

FORBIDDEN_FORECAST_COLUMNS = {"actual", "brier_score", "log_loss", "winner_outcome_names"}


def require(condition: bool, message: str) -> None:
    """Raise an assertion error with a clear validation message."""
    if not condition:
        raise AssertionError(message)


def validate_current_examples() -> dict[str, int]:
    """Check that current forecast examples are not evaluated."""
    payload = json.loads(CURRENT_PATH.read_text(encoding="utf-8"))
    examples = payload.get("examples", [])
    require(bool(examples), "current forecast examples are empty")
    for example in examples:
        summary = example["summary"]
        require(summary["forecastMode"] == "current", "current example has wrong forecastMode")
        require(summary["asOf"] is None, "current example should not have an asOf cutoff")
        require(summary["accuracyEvaluated"] is False, "current example should not be accuracy evaluated")
        require(example.get("aggregationMethods"), "current example is missing aggregation methods")
    return {"current_examples": len(examples)}


def validate_historical_outputs() -> dict[str, int]:
    """Check historical no-leakage and post-forecast evaluation invariants."""
    forecasts = pd.read_parquet(FORECAST_PATH)
    evaluated = pd.read_parquet(EVALUATION_PATH)
    require(not forecasts.empty, "historical forecast output is empty")
    require(not evaluated.empty, "historical evaluated output is empty")
    forbidden = FORBIDDEN_FORECAST_COLUMNS.intersection(forecasts.columns)
    require(not forbidden, f"forecast output contains eventual outcome columns: {sorted(forbidden)}")
    require({"actual", "brier_score", "log_loss"}.issubset(evaluated.columns), "evaluated output is missing scoring columns")
    require(
        int((forecasts["max_forecast_timestamp"] > forecasts["as_of_timestamp"]).sum()) == 0,
        "historical forecasts contain price observations after as_of",
    )
    require(forecasts["forecast_mode"].eq("historical").all(), "historical forecast output has wrong mode")
    require(evaluated["outcome_joined"].eq(True).all(), "evaluated output should mark outcomes as joined")
    require(forecasts["forecast_probability"].between(0, 1).all(), "forecast probabilities must be between 0 and 1")
    return {
        "historical_forecast_rows": len(forecasts),
        "historical_evaluated_rows": len(evaluated),
        "historical_targets": int(evaluated["target_market_id"].nunique()),
    }


def validate_event_normalisation_outputs() -> dict[str, int]:
    """Check event-normalisation no-leakage and probability invariants."""
    predictions = pd.read_parquet(EVENT_NORMALISATION_PATH)
    require(not predictions.empty, "event-normalisation prediction output is empty")
    require(
        predictions["raw_probability"].between(0, 1).all(),
        "raw event-normalisation probabilities must be between 0 and 1",
    )
    require(
        predictions["normalised_probability"].between(0, 1).all(),
        "normalised event probabilities must be between 0 and 1",
    )
    require(
        int((predictions["forecast_timestamp"] > predictions["as_of_timestamp"]).sum()) == 0,
        "event-normalisation predictions contain prices after as_of",
    )
    require(
        predictions["event_priced_markets_as_of"].ge(2).all(),
        "event-normalisation rows must have at least two priced sibling markets",
    )
    return {
        "event_normalisation_rows": len(predictions),
        "event_normalisation_events": int(predictions["event_id"].nunique()),
        "event_normalisation_markets": int(predictions["market_id"].nunique()),
    }


def main() -> None:
    """Run all validation checks."""
    result = {}
    result.update(validate_current_examples())
    result.update(validate_historical_outputs())
    result.update(validate_event_normalisation_outputs())
    for key, value in result.items():
        print(f"{key}: {value}", flush=True)


if __name__ == "__main__":
    main()
