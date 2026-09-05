"""Summarise historical aggregation backtest metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "data" / "outputs" / "analysis"
INPUT_PATH = ANALYSIS_DIR / "aggregation_backtest_evaluation.parquet"
SUMMARY_PATH = ANALYSIS_DIR / "aggregation_backtest_summary.csv"
CALIBRATION_PATH = ANALYSIS_DIR / "aggregation_backtest_calibration.csv"


def summarise_group(frame: pd.DataFrame, section: str, group: str) -> dict[str, object]:
    """Return one aggregate metric row."""
    return {
        "section": section,
        "group": group,
        "forecast_rows": len(frame),
        "target_markets": int(frame["target_market_id"].nunique()),
        "mean_brier_score": round(float(frame["brier_score"].mean()), 6),
        "mean_log_loss": round(float(frame["log_loss"].mean()), 6),
        "mean_forecast_probability": round(float(frame["forecast_probability"].mean()), 6),
        "actual_yes_rate": round(float(frame["actual"].mean()), 6),
        "mean_markets_used": round(float(frame["markets_used"].mean()), 2),
        "mean_events_used": round(float(frame["events_used"].mean()), 2),
    }


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Build method, horizon, category, and method-by-horizon summaries."""
    rows = [summarise_group(frame, "overall", "all")]
    for method, method_frame in frame.groupby("method", sort=True):
        rows.append(summarise_group(method_frame, "method", str(method)))
    for horizon, horizon_frame in frame.groupby("horizon_days", sort=True):
        rows.append(summarise_group(horizon_frame, "horizon", f"{horizon}_days"))
    for (method, horizon), group_frame in frame.groupby(["method", "horizon_days"], sort=True):
        rows.append(summarise_group(group_frame, "method_horizon", f"{method}:{horizon}_days"))
    for (method, category), group_frame in frame.groupby(["method", "target_category"], sort=True):
        if len(group_frame) >= 5:
            rows.append(summarise_group(group_frame, "method_category", f"{method}:{category}"))
    return pd.DataFrame(rows)


def build_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    """Build probability-bin calibration rows."""
    data = frame.copy()
    data["probability_bin"] = pd.cut(data["forecast_probability"], bins=[i / 10 for i in range(11)], include_lowest=True)
    calibration = (
        data.groupby(["method", "horizon_days", "probability_bin"], observed=True)
        .agg(
            forecast_rows=("actual", "size"),
            mean_forecast_probability=("forecast_probability", "mean"),
            actual_yes_rate=("actual", "mean"),
            mean_brier_score=("brier_score", "mean"),
        )
        .reset_index()
    )
    for column in ["mean_forecast_probability", "actual_yes_rate", "mean_brier_score"]:
        calibration[column] = calibration[column].round(6)
    calibration["probability_bin"] = calibration["probability_bin"].astype(str)
    return calibration


def main() -> None:
    """Write summary tables."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing evaluated backtest file: {INPUT_PATH}")
    frame = pd.read_parquet(INPUT_PATH)
    summary = build_summary(frame)
    calibration = build_calibration(frame)
    summary.to_csv(SUMMARY_PATH, index=False)
    calibration.to_csv(CALIBRATION_PATH, index=False)
    best = summary[summary["section"].eq("method")].sort_values("mean_brier_score").head(1)
    print(f"Wrote aggregation backtest summary to {SUMMARY_PATH}", flush=True)
    print(f"Wrote aggregation calibration summary to {CALIBRATION_PATH}", flush=True)
    if not best.empty:
        row = best.iloc[0]
        print(f"Best method by mean Brier: {row['group']} ({row['mean_brier_score']})", flush=True)


if __name__ == "__main__":
    main()
