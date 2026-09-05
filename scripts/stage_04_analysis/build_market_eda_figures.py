"""Build dissertation-ready EDA figures from the analysis datasets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
MARKETS_PATH = ROOT / "data" / "outputs" / "analysis" / "markets_analysis.parquet"
CATEGORIES_PATH = ROOT / "data" / "outputs" / "analysis" / "market_categories.parquet"
FIGURE_DIR = ROOT / "data" / "outputs" / "figures"
FIGURE_INDEX = FIGURE_DIR / "figure_index.csv"

TOP_CATEGORY_COUNT = 10
TOP_MONTHLY_CATEGORIES = 6

MARKET_COLUMNS = [
    "market_id",
    "event_id",
    "market_created_month",
    "market_volume",
    "price_points",
    "days_open",
    "has_price_history",
    "is_resolved",
]
CATEGORY_COLUMNS = ["market_id", "clean_category"]

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

PALETTE = [
    "#2f6f73",
    "#8f5f2a",
    "#4f6fae",
    "#b35c44",
    "#6a7f3f",
    "#8a5f91",
    "#c08a2d",
    "#4e7d55",
    "#7c6b5a",
    "#5f6f7f",
    "#8c6d31",
    "#4c6f91",
]


def require_inputs() -> None:
    """Fail clearly if required generated datasets are missing."""
    missing = [path for path in [MARKETS_PATH, CATEGORIES_PATH] if not path.exists()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing input dataset(s): {paths}")


def load_data() -> pd.DataFrame:
    """Load and join market-level fields with cleaned categories."""
    require_inputs()
    markets = pd.read_parquet(MARKETS_PATH, columns=MARKET_COLUMNS)
    categories = pd.read_parquet(CATEGORIES_PATH, columns=CATEGORY_COLUMNS)
    frame = markets.merge(categories, on="market_id", how="left", validate="one_to_one")
    frame["clean_category"] = frame["clean_category"].fillna("Other")
    frame["market_volume"] = pd.to_numeric(frame["market_volume"], errors="coerce").fillna(0)
    return frame


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save a figure with consistent dissertation-friendly settings."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def format_large_number(value: float) -> str:
    """Format large axis labels compactly."""
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}bn"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def positive_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return positive numeric values from a dataframe column."""
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    return series[series > 0]


def category_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return market and volume summaries by cleaned category."""
    return (
        frame.groupby("clean_category")
        .agg(
            markets=("market_id", "count"),
            events=("event_id", "nunique"),
            volume=("market_volume", "sum"),
            price_history_percent=("has_price_history", lambda x: x.mean() * 100),
            resolved_percent=("is_resolved", lambda x: x.mean() * 100),
        )
        .reset_index()
    )


def figure_category_counts(frame: pd.DataFrame) -> tuple[Path, str]:
    """Plot market counts by cleaned category."""
    data = category_summary(frame).sort_values("markets", ascending=False).head(TOP_CATEGORY_COUNT)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(
        data=data,
        y="clean_category",
        x="markets",
        hue="clean_category",
        ax=ax,
        palette=PALETTE[: len(data)],
        legend=False,
    )
    ax.set_title("Market Count by Cleaned Category")
    ax.set_xlabel("Markets")
    ax.set_ylabel("")
    ax.set_xlim(0, data["markets"].max() * 1.14)
    ax.xaxis.set_major_formatter(lambda value, _: format_large_number(value))
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    for container in ax.containers:
        ax.bar_label(container, labels=[format_large_number(v) for v in container.datavalues], padding=3)
    path = save_figure(fig, "01_market_count_by_category.png")
    return path, "Shows the cleaned composition of the dataset and confirms Sports/Crypto dominance by count."


def figure_category_volume(frame: pd.DataFrame) -> tuple[Path, str]:
    """Plot total market volume by cleaned category."""
    data = category_summary(frame).sort_values("volume", ascending=False).head(TOP_CATEGORY_COUNT)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(
        data=data,
        y="clean_category",
        x="volume",
        hue="clean_category",
        ax=ax,
        palette=PALETTE[: len(data)],
        legend=False,
    )
    ax.set_title("Total Market Volume by Cleaned Category")
    ax.set_xlabel("Total volume")
    ax.set_ylabel("")
    ax.set_xlim(0, data["volume"].max() * 1.14)
    ax.xaxis.set_major_formatter(lambda value, _: format_large_number(value))
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    for container in ax.containers:
        ax.bar_label(container, labels=[format_large_number(v) for v in container.datavalues], padding=3)
    path = save_figure(fig, "02_total_volume_by_category.png")
    return path, "Shows that volume tells a different story from count, with Elections much larger by volume than by frequency."


def figure_price_history_coverage(frame: pd.DataFrame) -> tuple[Path, str]:
    """Plot price-history coverage by cleaned category."""
    data = category_summary(frame)
    data = data[data["markets"] >= 1_000].sort_values("price_history_percent", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(
        data=data,
        y="clean_category",
        x="price_history_percent",
        hue="clean_category",
        ax=ax,
        palette=PALETTE[: len(data)],
        legend=False,
    )
    ax.set_title("Price-History Coverage by Cleaned Category")
    ax.set_xlabel("Markets with price history (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3)
    path = save_figure(fig, "03_price_history_coverage_by_category.png")
    return path, "Identifies which categories are strongest or weakest for time-series, calibration, and forecasting analysis."


def figure_volume_concentration(frame: pd.DataFrame) -> tuple[Path, str]:
    """Plot cumulative volume share by market percentile."""
    volume = frame["market_volume"].fillna(0)
    positive = volume[volume > 0].sort_values(ascending=False).reset_index(drop=True)
    cumulative_share = positive.cumsum() / positive.sum() * 100
    market_percentile = (pd.Series(range(1, len(positive) + 1)) / len(positive)) * 100

    sample = pd.DataFrame(
        {
            "market_percentile": market_percentile.iloc[::500].tolist() + [100],
            "volume_share": cumulative_share.iloc[::500].tolist() + [100],
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(sample["market_percentile"], sample["volume_share"], color="#2f6f73", linewidth=2.2)
    annotation_positions = {
        1: (7, 68),
        5: (10, 84),
        10: (15, 94),
    }
    for percentile in [1, 5, 10]:
        idx = max(0, int(len(positive) * percentile / 100) - 1)
        share = cumulative_share.iloc[idx]
        ax.scatter([percentile], [share], color="#b35c44", zorder=3)
        ax.annotate(
            f"Top {percentile}% = {share:.1f}%",
            xy=(percentile, share),
            xytext=annotation_positions[percentile],
            arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.8},
            fontsize=9,
        )
    ax.set_title("Concentration of Market Volume")
    ax.set_xlabel("Markets ranked by volume (%)")
    ax.set_ylabel("Cumulative share of total volume (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(color="#dddddd", linewidth=0.8)
    path = save_figure(fig, "04_volume_concentration.png")
    return path, "Shows the extreme skew in market activity and justifies robust volume summaries."


def figure_monthly_category_trends(frame: pd.DataFrame) -> tuple[Path, str]:
    """Plot monthly market creation trends for major categories."""
    top_categories = (
        frame["clean_category"].value_counts().head(TOP_MONTHLY_CATEGORIES).index.tolist()
    )
    data = frame[
        frame["clean_category"].isin(top_categories)
        & frame["market_created_month"].notna()
    ]
    monthly = (
        data.groupby(["market_created_month", "clean_category"])
        .size()
        .reset_index(name="markets")
        .sort_values("market_created_month")
    )
    monthly["market_created_month"] = pd.to_datetime(monthly["market_created_month"], errors="coerce")
    monthly = monthly.dropna(subset=["market_created_month"])

    fig, ax = plt.subplots(figsize=(10, 5.8))
    sns.lineplot(
        data=monthly,
        x="market_created_month",
        y="markets",
        hue="clean_category",
        ax=ax,
        linewidth=1.8,
        palette=PALETTE[: len(top_categories)],
    )
    ax.set_title("Monthly Market Creation by Major Category")
    ax.set_xlabel("Month created")
    ax.set_ylabel("Markets created")
    ax.yaxis.set_major_formatter(lambda value, _: format_large_number(value))
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.legend(title="", ncol=2, frameon=False, loc="upper left")
    path = save_figure(fig, "05_monthly_market_creation_by_category.png")
    return path, "Shows category-level market creation trends over time using the cleaned category layer."


def figure_volume_distribution(frame: pd.DataFrame) -> tuple[Path, str]:
    """Plot the distribution of positive market volume on a log scale."""
    volume = positive_series(frame, "market_volume")
    upper = volume.quantile(0.99)
    plotted = volume[volume <= upper]
    bins = np.logspace(np.log10(plotted.min()), np.log10(plotted.max()), 80)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(plotted, bins=bins, color="#2f6f73", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_title("Distribution of Positive Market Volume")
    ax.set_xlabel("Market volume (log scale, values above 99th percentile excluded)")
    ax.set_ylabel("Markets")
    ax.yaxis.set_major_formatter(lambda value, _: format_large_number(value))
    ax.grid(color="#dddddd", linewidth=0.8)

    median = volume.median()
    p95 = volume.quantile(0.95)
    ax.axvline(median, color="#b35c44", linewidth=1.5)
    ax.axvline(p95, color="#8f5f2a", linewidth=1.5)
    ax.text(median, ax.get_ylim()[1] * 0.88, f"Median\n{format_large_number(median)}", ha="center", fontsize=9)
    ax.text(p95, ax.get_ylim()[1] * 0.72, f"95th pct\n{format_large_number(p95)}", ha="center", fontsize=9)

    path = save_figure(fig, "06_positive_volume_distribution.png")
    return path, "Shows market-volume skew without relying on category grouping."


def figure_price_history_depth(frame: pd.DataFrame) -> tuple[Path, str]:
    """Plot the distribution of price-history points per covered market."""
    points = positive_series(frame, "price_points")
    plotted = points[points <= points.quantile(0.99)]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(plotted, bins=70, color="#4f6fae", edgecolor="white", linewidth=0.3)
    ax.set_title("Distribution of Price-History Depth")
    ax.set_xlabel("Price-history observations per market (values above 99th percentile excluded)")
    ax.set_ylabel("Markets")
    ax.yaxis.set_major_formatter(lambda value, _: format_large_number(value))
    ax.grid(color="#dddddd", linewidth=0.8)

    median = points.median()
    p95 = points.quantile(0.95)
    ax.axvline(median, color="#b35c44", linewidth=1.5)
    ax.axvline(p95, color="#8f5f2a", linewidth=1.5)
    ax.text(median + 4, ax.get_ylim()[1] * 0.88, f"Median: {median:.0f}", fontsize=9)
    ax.text(p95 + 6, ax.get_ylim()[1] * 0.72, f"95th pct: {p95:.0f}", fontsize=9)

    path = save_figure(fig, "07_price_history_depth_distribution.png")
    return path, "Shows how much time-series evidence is available per covered market."


def figure_days_open_distribution(frame: pd.DataFrame) -> tuple[Path, str]:
    """Plot the distribution of market duration in days."""
    days_open = pd.to_numeric(frame["days_open"], errors="coerce").dropna()
    days_open = days_open[days_open >= 0]
    plotted = days_open[days_open <= days_open.quantile(0.99)]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(plotted, bins=70, color="#8f5f2a", edgecolor="white", linewidth=0.3)
    ax.set_title("Distribution of Market Duration")
    ax.set_xlabel("Days open (values above 99th percentile excluded)")
    ax.set_ylabel("Markets")
    ax.yaxis.set_major_formatter(lambda value, _: format_large_number(value))
    ax.grid(color="#dddddd", linewidth=0.8)

    median = days_open.median()
    p95 = days_open.quantile(0.95)
    ax.axvline(median, color="#b35c44", linewidth=1.5)
    ax.axvline(p95, color="#2f6f73", linewidth=1.5)
    ax.text(median + 3, ax.get_ylim()[1] * 0.88, f"Median: {median:.1f} days", fontsize=9)
    ax.text(p95 + 3, ax.get_ylim()[1] * 0.72, f"95th pct: {p95:.1f} days", fontsize=9)

    path = save_figure(fig, "08_days_open_distribution.png")
    return path, "Shows market-duration skew and helps frame later forecasting cut-off choices."


def write_figure_index(entries: list[dict[str, Any]]) -> None:
    """Write a lightweight index describing the generated figures."""
    with FIGURE_INDEX.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "title", "purpose"])
        writer.writeheader()
        writer.writerows(entries)


def main() -> None:
    """Build the first dissertation-ready EDA figure set."""
    plt.rcParams.update(FIGURE_STYLE)
    sns.set_theme(style="whitegrid", rc=FIGURE_STYLE)
    frame = load_data()

    figure_functions = [
        ("Market Count by Cleaned Category", figure_category_counts),
        ("Total Market Volume by Cleaned Category", figure_category_volume),
        ("Price-History Coverage by Cleaned Category", figure_price_history_coverage),
        ("Concentration of Market Volume", figure_volume_concentration),
        ("Monthly Market Creation by Major Category", figure_monthly_category_trends),
        ("Distribution of Positive Market Volume", figure_volume_distribution),
        ("Distribution of Price-History Depth", figure_price_history_depth),
        ("Distribution of Market Duration", figure_days_open_distribution),
    ]

    entries: list[dict[str, Any]] = []
    for title, function in figure_functions:
        path, purpose = function(frame)
        entries.append({"filename": path.name, "title": title, "purpose": purpose})
        print(f"Wrote {path}", flush=True)

    write_figure_index(entries)
    print(f"Wrote {FIGURE_INDEX}", flush=True)


if __name__ == "__main__":
    main()
