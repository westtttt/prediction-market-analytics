"""Build dissertation-ready reliability summaries and figures."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "outputs" / "analysis" / "reliability_cutoffs.parquet"
OUTPUT_PATH = ROOT / "data" / "outputs" / "analysis" / "reliability_analysis_summary.csv"
FIGURE_DIR = ROOT / "data" / "outputs" / "figures"

PALETTE = ["#2f6f73", "#8f5f2a", "#4f6fae", "#b35c44", "#6a7f3f", "#8a5f91", "#c08a2d"]

FIGURE_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
}


def require_input() -> None:
    """Fail clearly if the reliability cutoff dataset is missing."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input dataset: {INPUT_PATH}")


def load_data() -> pd.DataFrame:
    """Load the cutoff reliability dataset and add analysis bands."""
    require_input()
    frame = pd.read_parquet(INPUT_PATH)
    frame["forecast_probability"] = pd.to_numeric(frame["forecast_probability"], errors="coerce")
    frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
    frame["squared_error"] = (frame["forecast_probability"] - frame["actual"]) ** 2
    frame["volume_band"] = make_band(frame["market_volume"], "volume")
    frame["history_depth_band"] = make_band(frame["price_points"], "price history")
    return frame


def make_band(series: pd.Series, label: str) -> pd.Series:
    """Create stable low-to-high bands for skewed market measures."""
    values = pd.to_numeric(series, errors="coerce")
    positive = values[values > 0]
    result = pd.Series("Missing/zero", index=series.index, dtype="object")
    if positive.nunique() < 4:
        return result

    binned = pd.qcut(positive, q=4, duplicates="drop")
    labels = [f"{label} Q{i + 1}" for i in range(len(binned.cat.categories))]
    result.loc[positive.index] = pd.qcut(positive, q=len(labels), labels=labels, duplicates="drop").astype("object")
    return result


def metric_row(section: str, group: str, metric: str, value: Any, notes: str = "") -> dict[str, Any]:
    """Return one standard summary row."""
    return {"section": section, "group": group, "metric": metric, "value": value, "notes": notes}


def brier(frame: pd.DataFrame) -> float:
    """Return mean squared probability error."""
    return round(float(frame["squared_error"].mean()), 6)


def build_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Build compact reliability summary rows."""
    rows: list[dict[str, Any]] = []
    for horizon, horizon_frame in frame.groupby("horizon_days", sort=True):
        rows.extend(
            [
                metric_row("horizon", f"{horizon}_days", "contract_rows", len(horizon_frame)),
                metric_row("horizon", f"{horizon}_days", "markets", int(horizon_frame["market_id"].nunique())),
                metric_row("horizon", f"{horizon}_days", "brier_score", brier(horizon_frame)),
                metric_row(
                    "horizon",
                    f"{horizon}_days",
                    "mean_probability",
                    round(float(horizon_frame["forecast_probability"].mean()), 6),
                ),
                metric_row(
                    "horizon",
                    f"{horizon}_days",
                    "actual_win_rate",
                    round(float(horizon_frame["actual"].mean()), 6),
                ),
            ]
        )

    for group_name, column in [("category", "clean_category"), ("volume_band", "volume_band"), ("history_depth", "history_depth_band")]:
        grouped = frame.groupby(["horizon_days", column], sort=True)
        for (horizon, group), group_frame in grouped:
            if len(group_frame) < 1_000:
                continue
            rows.extend(
                [
                    metric_row(group_name, f"{horizon}_days:{group}", "contract_rows", len(group_frame)),
                    metric_row(group_name, f"{horizon}_days:{group}", "markets", int(group_frame["market_id"].nunique())),
                    metric_row(group_name, f"{horizon}_days:{group}", "brier_score", brier(group_frame)),
                ]
            )
    return rows


def save_summary(rows: list[dict[str, Any]]) -> None:
    """Write the summary CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save a figure with consistent settings."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def palette_for(count: int) -> list[str]:
    """Return a palette with exactly the requested number of colours."""
    return sns.color_palette(PALETTE, n_colors=count)


def figure_brier_by_horizon(frame: pd.DataFrame) -> Path:
    """Plot Brier score across forecast horizons."""
    data = frame.groupby("horizon_days", as_index=False).agg(brier_score=("squared_error", "mean"), markets=("market_id", "nunique"))
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    sns.barplot(
        data=data,
        x="horizon_days",
        y="brier_score",
        hue="horizon_days",
        palette=palette_for(len(data)),
        legend=False,
        ax=ax,
    )
    ax.set_title("Forecast Reliability by Time Before Close")
    ax.set_xlabel("Days before market close")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_ylim(0, max(0.2, data["brier_score"].max() * 1.25))
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)
    return save_figure(fig, "09_brier_score_by_horizon.png")


def figure_calibration_by_horizon(frame: pd.DataFrame) -> Path:
    """Plot calibration curves for each horizon."""
    data = frame.copy()
    data["probability_bin"] = pd.cut(data["forecast_probability"], bins=[i / 10 for i in range(11)], include_lowest=True)
    calibration = (
        data.groupby(["horizon_days", "probability_bin"], observed=True)
        .agg(mean_probability=("forecast_probability", "mean"), actual_win_rate=("actual", "mean"), rows=("actual", "size"))
        .reset_index()
    )
    calibration = calibration[calibration["rows"] >= 1_000]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.lineplot(
        data=calibration,
        x="mean_probability",
        y="actual_win_rate",
        hue="horizon_days",
        marker="o",
        palette=palette_for(calibration["horizon_days"].nunique()),
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1.0)
    ax.set_title("Calibration by Forecast Horizon")
    ax.set_xlabel("Mean forecast probability")
    ax.set_ylabel("Actual win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.legend(title="Days before close")
    return save_figure(fig, "10_calibration_curve_by_horizon.png")


def figure_brier_by_category(frame: pd.DataFrame) -> Path:
    """Plot one-day Brier score for major categories."""
    one_day = frame[frame["horizon_days"] == 1].copy()
    data = (
        one_day.groupby("clean_category")
        .agg(brier_score=("squared_error", "mean"), markets=("market_id", "nunique"))
        .reset_index()
    )
    data = data[data["markets"] >= 5_000].sort_values("brier_score", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(
        data=data,
        y="clean_category",
        x="brier_score",
        hue="clean_category",
        palette=palette_for(len(data)),
        legend=False,
        ax=ax,
    )
    ax.set_title("One-Day Reliability by Category")
    ax.set_xlabel("Brier score (lower is better)")
    ax.set_ylabel("")
    ax.set_xlim(0, data["brier_score"].max() * 1.15)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)
    return save_figure(fig, "11_brier_score_by_category_1d.png")


def figure_brier_by_band(frame: pd.DataFrame, column: str, title: str, filename: str) -> Path:
    """Plot Brier score by horizon and an analysis band."""
    data = (
        frame.groupby(["horizon_days", column], observed=True)
        .agg(brier_score=("squared_error", "mean"), markets=("market_id", "nunique"))
        .reset_index()
    )
    data = data[data["markets"] >= 5_000]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(
        data=data,
        x=column,
        y="brier_score",
        hue="horizon_days",
        marker="o",
        palette=palette_for(data["horizon_days"].nunique()),
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Brier score (lower is better)")
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Days before close")
    return save_figure(fig, filename)


def main() -> None:
    """Build reliability summaries and figures."""
    plt.rcParams.update(FIGURE_STYLE)
    frame = load_data()
    rows = build_summary(frame)
    save_summary(rows)
    figures = [
        figure_brier_by_horizon(frame),
        figure_calibration_by_horizon(frame),
        figure_brier_by_category(frame),
        figure_brier_by_band(frame, "volume_band", "Reliability by Market Volume Band", "12_brier_score_by_volume_band.png"),
        figure_brier_by_band(
            frame,
            "history_depth_band",
            "Reliability by Price-History Depth",
            "13_brier_score_by_price_history_depth.png",
        ),
    ]
    print(f"Wrote reliability summary to {OUTPUT_PATH}", flush=True)
    for figure in figures:
        print(f"Wrote figure {figure}", flush=True)


if __name__ == "__main__":
    main()
