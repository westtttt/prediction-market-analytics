"""Build a reusable resolved-market winner analysis dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CUTOFF_PATH = ROOT / "data" / "outputs" / "analysis" / "reliability_cutoffs.parquet"
CONTRACT_PATH = ROOT / "data" / "outputs" / "analysis" / "reliability_contracts.parquet"
MARKET_PATH = ROOT / "data" / "outputs" / "analysis" / "markets_analysis.parquet"
OUTPUT_PATH = ROOT / "data" / "outputs" / "analysis" / "winner_analysis.parquet"
SUMMARY_PATH = ROOT / "data" / "outputs" / "analysis" / "winner_analysis_summary.csv"


def require_inputs() -> None:
    """Fail clearly if required analysis files are missing."""
    missing = [path for path in [CUTOFF_PATH, CONTRACT_PATH, MARKET_PATH] if not path.exists()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing input file(s): {paths}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load compact reliability and market metadata inputs."""
    cutoffs = pd.read_parquet(CUTOFF_PATH)
    contracts = pd.read_parquet(
        CONTRACT_PATH,
        columns=["market_id", "outcome_index", "current_price", "resolved_winner", "resolution_inference_status"],
    )
    markets = pd.read_parquet(
        MARKET_PATH,
        columns=["market_id", "event_title", "market_question", "market_end_date"],
    )
    return cutoffs, contracts, markets


def make_band(series: pd.Series, label: str) -> pd.Series:
    """Create quartile bands while keeping missing and zero values explicit."""
    values = pd.to_numeric(series, errors="coerce")
    positive = values[values > 0]
    result = pd.Series("Missing/zero", index=series.index, dtype="object")
    if positive.nunique() < 4:
        return result

    binned = pd.qcut(positive, q=4, duplicates="drop")
    labels = [f"{label} Q{i + 1}" for i in range(len(binned.cat.categories))]
    result.loc[positive.index] = pd.qcut(positive, q=len(labels), labels=labels, duplicates="drop").astype("object")
    return result


def valid_market_horizons(frame: pd.DataFrame) -> pd.MultiIndex:
    """Return market/horizon pairs with two outcomes, two probabilities, and one winner."""
    grouped = frame.groupby(["market_id", "horizon_days"], sort=False).agg(
        rows=("outcome_index", "size"),
        outcomes=("outcome_index", "nunique"),
        winners=("actual", "sum"),
        forecast_rows=("forecast_probability", lambda series: series.notna().sum()),
    )
    return grouped[
        (grouped["rows"] == 2)
        & (grouped["outcomes"] == 2)
        & (grouped["winners"] == 1)
        & (grouped["forecast_rows"] == 2)
    ].index


def build_market_horizon_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Build one summary row per valid market and horizon."""
    keys = ["market_id", "horizon_days"]
    grouped = frame.groupby(keys, sort=False)
    summary = grouped.agg(
        max_probability_at_cutoff=("forecast_probability", "max"),
        favourite_count=("forecast_probability", lambda series: int(series.eq(series.max()).sum())),
    ).reset_index()
    winners = frame[frame["actual"].eq(1)].loc[:, keys + ["forecast_probability"]].rename(
        columns={"forecast_probability": "winner_probability_at_cutoff"}
    )
    favourites = frame.merge(summary[keys + ["max_probability_at_cutoff"]], on=keys, how="left", validate="many_to_one")
    favourites = favourites[favourites["forecast_probability"].eq(favourites["max_probability_at_cutoff"])]
    favourite_wins = favourites.groupby(keys, sort=False)["actual"].max().reset_index(name="market_favourite_won")
    summary = summary.merge(winners, on=keys, how="left", validate="one_to_one")
    summary = summary.merge(favourite_wins, on=keys, how="left", validate="one_to_one")
    summary["favourite_tie"] = summary["favourite_count"] > 1
    summary["market_favourite_won"] = summary["market_favourite_won"].eq(1) & ~summary["favourite_tie"]
    return summary.drop(columns=["favourite_count"])


def build_dataset(cutoffs: pd.DataFrame, contracts: pd.DataFrame, markets: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one row per market outcome and cutoff horizon."""
    frame = cutoffs.copy()
    frame["forecast_probability"] = pd.to_numeric(frame["forecast_probability"], errors="coerce")
    frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
    frame = frame[
        frame["forecast_probability"].between(0, 1)
        & frame["actual"].isin([0, 1])
    ].copy()

    frame = frame.merge(
        contracts.rename(columns={"current_price": "final_probability"}),
        on=["market_id", "outcome_index"],
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(markets, on="market_id", how="left", validate="many_to_one")

    start_pairs = frame.groupby(["market_id", "horizon_days"]).ngroups
    eligible_pairs = valid_market_horizons(frame)
    valid_keys = eligible_pairs.to_frame(index=False)
    frame = frame.merge(valid_keys, on=["market_id", "horizon_days"], how="inner", validate="many_to_one")
    market_summary = build_market_horizon_summary(frame)
    frame = frame.merge(market_summary, on=["market_id", "horizon_days"], how="left", validate="many_to_one")
    frame["is_winner"] = frame["actual"].astype("int8")
    frame["is_favourite_at_cutoff"] = frame["forecast_probability"].eq(frame["max_probability_at_cutoff"]) & ~frame["favourite_tie"]
    frame["winner_below_50"] = frame["winner_probability_at_cutoff"] < 0.5
    frame["winner_below_25"] = frame["winner_probability_at_cutoff"] < 0.25
    frame["market_volume"] = pd.to_numeric(frame["market_volume"], errors="coerce")
    frame["market_liquidity"] = pd.to_numeric(frame["market_liquidity"], errors="coerce")
    frame["volume_band"] = make_band(frame["market_volume"], "volume")
    frame["liquidity_band"] = make_band(frame["market_liquidity"], "liquidity")

    columns = [
        "market_id",
        "event_id",
        "event_title",
        "market_question",
        "clean_category",
        "horizon_days",
        "cutoff_time",
        "forecast_time",
        "outcome_index",
        "outcome_name",
        "forecast_probability",
        "final_probability",
        "is_winner",
        "is_favourite_at_cutoff",
        "favourite_tie",
        "market_favourite_won",
        "winner_probability_at_cutoff",
        "winner_below_50",
        "winner_below_25",
        "market_volume",
        "market_liquidity",
        "volume_band",
        "liquidity_band",
        "days_open",
        "price_points",
        "market_end_date",
        "market_closed_time",
    ]
    coverage = {
        "input_contract_rows": len(cutoffs),
        "candidate_market_horizons": start_pairs,
        "eligible_market_horizons": frame.groupby(["market_id", "horizon_days"]).ngroups,
        "eligible_markets": int(frame["market_id"].nunique()),
        "output_rows": len(frame),
        "favourite_tie_market_horizons": int(frame[frame["favourite_tie"]].groupby(["market_id", "horizon_days"]).ngroups),
    }
    return frame.loc[:, columns], coverage


def summary_row(section: str, group: str, metric: str, value: Any, notes: str = "") -> dict[str, Any]:
    """Return one compact summary row."""
    return {"section": section, "group": group, "metric": metric, "value": value, "notes": notes}


def build_summary(frame: pd.DataFrame, coverage: dict[str, Any]) -> list[dict[str, Any]]:
    """Build compact winner-analysis summary rows."""
    rows = [summary_row("coverage", "all", key, value) for key, value in coverage.items()]
    market_horizons = frame.drop_duplicates(["market_id", "horizon_days"])
    for horizon, horizon_frame in market_horizons.groupby("horizon_days", sort=True):
        group = f"{horizon}_days"
        no_ties = horizon_frame[~horizon_frame["favourite_tie"]]
        rows.extend(
            [
                summary_row("horizon", group, "markets", len(horizon_frame)),
                summary_row("horizon", group, "mean_winner_probability", round(float(horizon_frame["winner_probability_at_cutoff"].mean()), 6)),
                summary_row("horizon", group, "median_winner_probability", round(float(horizon_frame["winner_probability_at_cutoff"].median()), 6)),
                summary_row("horizon", group, "favourite_tie_rate", round(float(horizon_frame["favourite_tie"].mean()), 6)),
                summary_row("horizon", group, "favourite_accuracy_excluding_ties", round(float(no_ties["market_favourite_won"].mean()), 6)),
                summary_row("horizon", group, "upset_rate_winner_below_50", round(float(horizon_frame["winner_below_50"].mean()), 6)),
                summary_row("horizon", group, "major_upset_rate_winner_below_25", round(float(horizon_frame["winner_below_25"].mean()), 6)),
            ]
        )
    return rows


def write_outputs(frame: pd.DataFrame, rows: list[dict[str, Any]]) -> None:
    """Write the winner dataset and compact summary."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT_PATH, index=False)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build winner analysis outputs."""
    require_inputs()
    cutoffs, contracts, markets = load_inputs()
    frame, coverage = build_dataset(cutoffs, contracts, markets)
    rows = build_summary(frame, coverage)
    write_outputs(frame, rows)
    print(f"Wrote winner dataset to {OUTPUT_PATH}")
    print(f"Wrote winner summary to {SUMMARY_PATH}")
    print(f"Eligible markets: {coverage['eligible_markets']:,}")


if __name__ == "__main__":
    main()
