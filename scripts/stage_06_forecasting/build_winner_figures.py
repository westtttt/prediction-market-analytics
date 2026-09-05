"""Build dissertation-ready winner analysis summaries and figures."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "outputs" / "analysis" / "winner_analysis.parquet"
SUMMARY_PATH = ROOT / "data" / "outputs" / "analysis" / "winner_results_summary.csv"
UPSET_EXAMPLES_PATH = ROOT / "data" / "outputs" / "analysis" / "winner_upset_examples.csv"
FIGURE_DIR = ROOT / "data" / "outputs" / "figures"

MIN_CATEGORY_MARKETS = 5_000
MIN_BAND_MARKETS = 5_000

FIGURE_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
}

PALETTE = ["#2f6f73", "#8f5f2a", "#4f6fae", "#b35c44", "#6a7f3f", "#8a5f91", "#c08a2d", "#5f6f7f"]
HORIZON_PALETTE = {
    1: "#0072B2",
    3: "#009E73",
    7: "#D55E00",
    14: "#CC79A7",
    30: "#E69F00",
}


def require_input() -> None:
    """Fail clearly if the winner dataset is missing."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input dataset: {INPUT_PATH}")


def load_market_horizons() -> pd.DataFrame:
    """Load one row per market and cutoff horizon."""
    require_input()
    frame = pd.read_parquet(INPUT_PATH)
    market_horizons = frame.drop_duplicates(["market_id", "horizon_days"]).copy()
    market_horizons["horizon_label"] = market_horizons["horizon_days"].map(lambda value: f"{value} days")
    return market_horizons


def result_row(section: str, group: str, metric: str, value: Any, notes: str = "") -> dict[str, Any]:
    """Return one compact result row."""
    return {"section": section, "group": group, "metric": metric, "value": value, "notes": notes}


def build_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Build compact winner result summary rows."""
    rows: list[dict[str, Any]] = []
    for horizon, horizon_frame in frame.groupby("horizon_days", sort=True):
        no_ties = horizon_frame[~horizon_frame["favourite_tie"]]
        rows.extend(
            [
                result_row("horizon", f"{horizon}_days", "markets", len(horizon_frame)),
                result_row("horizon", f"{horizon}_days", "mean_winner_probability", round(float(horizon_frame["winner_probability_at_cutoff"].mean()), 6)),
                result_row("horizon", f"{horizon}_days", "median_winner_probability", round(float(horizon_frame["winner_probability_at_cutoff"].median()), 6)),
                result_row("horizon", f"{horizon}_days", "single_favourite_markets", len(no_ties)),
                result_row("horizon", f"{horizon}_days", "favourite_accuracy_excluding_ties", round(float(no_ties["market_favourite_won"].mean()), 6)),
                result_row("horizon", f"{horizon}_days", "upset_rate_winner_below_50", round(float(horizon_frame["winner_below_50"].mean()), 6)),
                result_row("horizon", f"{horizon}_days", "major_upset_rate_winner_below_25", round(float(horizon_frame["winner_below_25"].mean()), 6)),
            ]
        )

    for group_name, column in [("category", "clean_category"), ("volume_band", "volume_band"), ("liquidity_band", "liquidity_band")]:
        for (horizon, group), group_frame in frame.groupby(["horizon_days", column], sort=True):
            if len(group_frame) < MIN_BAND_MARKETS:
                continue
            no_ties = group_frame[~group_frame["favourite_tie"]]
            if no_ties.empty:
                continue
            rows.extend(
                [
                    result_row(group_name, f"{horizon}_days:{group}", "markets", len(group_frame)),
                    result_row(group_name, f"{horizon}_days:{group}", "favourite_accuracy_excluding_ties", round(float(no_ties["market_favourite_won"].mean()), 6)),
                    result_row(group_name, f"{horizon}_days:{group}", "upset_rate_winner_below_50", round(float(group_frame["winner_below_50"].mean()), 6)),
                ]
            )
    return rows


def save_summary(rows: list[dict[str, Any]]) -> None:
    """Write the compact result summary."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_upset_examples(frame: pd.DataFrame) -> None:
    """Write a small table of high-volume surprise markets."""
    one_day = frame[(frame["horizon_days"] == 1) & frame["winner_below_50"]].copy()
    examples = (
        one_day.sort_values(["market_volume", "winner_probability_at_cutoff"], ascending=[False, True])
        .loc[
            :,
            [
                "market_id",
                "event_id",
                "event_title",
                "market_question",
                "clean_category",
                "winner_probability_at_cutoff",
                "market_volume",
                "volume_band",
                "market_closed_time",
            ],
        ]
        .head(25)
    )
    examples.to_csv(UPSET_EXAMPLES_PATH, index=False)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save one figure with consistent settings."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def palette_for(count: int) -> list[str]:
    """Return a palette with exactly the requested number of colours."""
    return sns.color_palette(PALETTE, n_colors=count)


def horizon_palette_for(values: pd.Series) -> dict[int, str]:
    """Return distinct colours for cutoff horizons, with a fallback for unexpected values."""
    ordered_values = sorted(int(value) for value in values.dropna().unique())
    fallback = sns.color_palette("colorblind", n_colors=len(ordered_values)).as_hex()
    return {value: HORIZON_PALETTE.get(value, fallback[index]) for index, value in enumerate(ordered_values)}


def figure_winner_probability(frame: pd.DataFrame) -> Path:
    """Plot eventual winner probability by cutoff horizon."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sns.boxplot(
        data=frame,
        x="horizon_days",
        y="winner_probability_at_cutoff",
        hue="horizon_days",
        palette=palette_for(frame["horizon_days"].nunique()),
        showfliers=False,
        legend=False,
        ax=ax,
    )
    ax.set_title("Eventual Winner Probability Before Market Close")
    ax.set_xlabel("Days before close")
    ax.set_ylabel("Winner probability at cutoff")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    return save_figure(fig, "14_winner_probability_by_cutoff.png")


def figure_favourite_accuracy_by_category(frame: pd.DataFrame) -> Path:
    """Plot single-favourite accuracy by category and horizon."""
    no_ties = frame[~frame["favourite_tie"]]
    data = (
        no_ties.groupby(["horizon_days", "clean_category"])
        .agg(accuracy=("market_favourite_won", "mean"), markets=("market_id", "nunique"))
        .reset_index()
    )
    data = data[data["markets"] >= MIN_CATEGORY_MARKETS]
    category_order = (
        data.groupby("clean_category")["markets"].sum().sort_values(ascending=False).head(10).index.tolist()
    )
    data = data[data["clean_category"].isin(category_order)]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    sns.barplot(
        data=data,
        y="clean_category",
        x="accuracy",
        hue="horizon_days",
        palette=horizon_palette_for(data["horizon_days"]),
        order=category_order,
        ax=ax,
    )
    ax.set_title("Favourite Accuracy by Category")
    ax.set_xlabel("Accuracy where one favourite exists")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.legend(title="Days before close", loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
    return save_figure(fig, "15_favourite_accuracy_by_category.png")


def figure_accuracy_by_volume(frame: pd.DataFrame) -> Path:
    """Plot favourite accuracy by volume band and horizon."""
    no_ties = frame[~frame["favourite_tie"]]
    data = (
        no_ties.groupby(["horizon_days", "volume_band"], observed=True)
        .agg(accuracy=("market_favourite_won", "mean"), markets=("market_id", "nunique"))
        .reset_index()
    )
    data = data[data["markets"] >= MIN_BAND_MARKETS]
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    sns.lineplot(
        data=data,
        x="volume_band",
        y="accuracy",
        hue="horizon_days",
        marker="o",
        palette=palette_for(data["horizon_days"].nunique()),
        ax=ax,
    )
    ax.set_title("Favourite Accuracy by Market Volume Band")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy where one favourite exists")
    ax.set_ylim(0.5, 0.9)
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Days before close", loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
    return save_figure(fig, "16_favourite_accuracy_by_volume_band.png")


def figure_upset_rate_by_category(frame: pd.DataFrame) -> Path:
    """Plot one-day upset rate by category."""
    one_day = frame[frame["horizon_days"] == 1]
    data = (
        one_day.groupby("clean_category")
        .agg(upset_rate=("winner_below_50", "mean"), markets=("market_id", "nunique"))
        .reset_index()
    )
    data = data[data["markets"] >= MIN_CATEGORY_MARKETS].sort_values("upset_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(
        data=data,
        y="clean_category",
        x="upset_rate",
        hue="clean_category",
        palette=palette_for(len(data)),
        legend=False,
        ax=ax,
    )
    ax.set_title("One-Day Upset Rate by Category")
    ax.set_xlabel("Share of markets where winner was below 50%")
    ax.set_ylabel("")
    ax.set_xlim(0, max(0.3, data["upset_rate"].max() * 1.15))
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3)
    return save_figure(fig, "17_upset_rate_by_category.png")


def main() -> None:
    """Build winner analysis summaries and figures."""
    plt.rcParams.update(FIGURE_STYLE)
    frame = load_market_horizons()
    rows = build_summary(frame)
    save_summary(rows)
    save_upset_examples(frame)
    figures = [
        figure_winner_probability(frame),
        figure_favourite_accuracy_by_category(frame),
        figure_accuracy_by_volume(frame),
        figure_upset_rate_by_category(frame),
    ]
    print(f"Wrote winner summary to {SUMMARY_PATH}")
    print(f"Wrote upset examples to {UPSET_EXAMPLES_PATH}")
    for path in figures:
        print(f"Wrote figure {path}")


if __name__ == "__main__":
    main()
