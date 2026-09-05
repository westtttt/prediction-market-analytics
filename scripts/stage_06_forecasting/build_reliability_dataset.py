"""Build a contract-level reliability dataset from resolved binary markets."""

from __future__ import annotations

import csv
import math
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = ROOT / "data" / "database" / "markets.db"
MARKET_PATH = ROOT / "data" / "outputs" / "analysis" / "markets_analysis.parquet"
CATEGORY_PATH = ROOT / "data" / "outputs" / "analysis" / "market_categories.parquet"
OUTPUT_PATH = ROOT / "data" / "outputs" / "analysis" / "reliability_contracts.parquet"
CUTOFF_OUTPUT_PATH = ROOT / "data" / "outputs" / "analysis" / "reliability_cutoffs.parquet"
SUMMARY_PATH = ROOT / "data" / "outputs" / "analysis" / "reliability_summary.csv"
HORIZON_DAYS = [1, 3, 7]

MARKET_COLUMNS = [
    "market_id",
    "event_id",
    "market_question",
    "outcome_count",
    "is_resolved",
    "has_price_history",
    "price_points",
    "market_volume",
    "market_liquidity",
    "days_open",
    "market_closed_time",
    "market_end_date",
]


def require_inputs() -> None:
    """Fail clearly if required source files are missing."""
    missing = [path for path in [DATABASE_PATH, MARKET_PATH, CATEGORY_PATH] if not path.exists()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing input file(s): {paths}")


def load_markets() -> pd.DataFrame:
    """Load resolved binary markets and cleaned categories."""
    markets = pd.read_parquet(MARKET_PATH, columns=MARKET_COLUMNS)
    categories = pd.read_parquet(CATEGORY_PATH, columns=["market_id", "clean_category"])
    markets = markets.merge(categories, on="market_id", how="left", validate="one_to_one")
    markets["clean_category"] = markets["clean_category"].fillna("Other")
    return markets[(markets["outcome_count"] == 2) & markets["is_resolved"]].copy()


def load_outcomes() -> pd.DataFrame:
    """Load outcome-level probabilities and resolution labels from SQLite."""
    query = """
        select
            market_id,
            outcome_index,
            outcome_name,
            current_price,
            resolved_winner,
            resolution_inference_status
        from outcomes
    """
    with sqlite3.connect(DATABASE_PATH) as connection:
        return pd.read_sql_query(query, connection)


def build_contracts(markets: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create one reliability row per outcome contract."""
    start_markets = len(markets)
    outcomes = outcomes.merge(markets, on="market_id", how="inner", validate="many_to_one")

    grouped = outcomes.groupby("market_id", sort=False).agg(
        outcome_rows=("outcome_index", "count"),
        winner_rows=("resolved_winner", "sum"),
        priced_rows=("current_price", lambda series: series.notna().sum()),
    )
    eligible_ids = grouped[
        (grouped["outcome_rows"] == 2) & (grouped["winner_rows"] == 1) & (grouped["priced_rows"] == 2)
    ].index

    contracts = outcomes[outcomes["market_id"].isin(eligible_ids)].copy()
    contracts["probability"] = pd.to_numeric(contracts["current_price"], errors="coerce")
    contracts["actual"] = contracts["resolved_winner"].astype("int8")
    contracts = contracts[(contracts["probability"] >= 0) & (contracts["probability"] <= 1)].copy()
    contracts["error"] = contracts["probability"] - contracts["actual"]
    contracts["squared_error"] = contracts["error"] ** 2
    contracts["probability_source"] = "outcomes.current_price"
    contracts["evaluation_unit"] = "contract_outcome"

    summary = {
        "candidate_binary_resolved_markets": start_markets,
        "outcome_rows_after_market_filter": len(outcomes),
        "eligible_markets": int(contracts["market_id"].nunique()),
        "eligible_contract_rows": len(contracts),
        "dropped_markets": start_markets - int(contracts["market_id"].nunique()),
        "markets_with_exactly_one_winner": int((grouped["winner_rows"] == 1).sum()),
        "markets_with_two_priced_outcomes": int((grouped["priced_rows"] == 2).sum()),
    }
    return contracts, summary


def metric_row(section: str, group: str, metric: str, value: Any, notes: str = "") -> dict[str, Any]:
    """Return one standard reliability summary row."""
    return {"section": section, "group": group, "metric": metric, "value": value, "notes": notes}


def brier_score(frame: pd.DataFrame) -> float:
    """Return mean squared probability error."""
    return round(float(frame["squared_error"].mean()), 6)


def log_loss(frame: pd.DataFrame) -> float:
    """Return binary log loss with clipped probabilities."""
    probabilities = frame["probability"].clip(1e-15, 1 - 1e-15)
    actuals = frame["actual"]
    losses = -(actuals * probabilities.map(math.log) + (1 - actuals) * (1 - probabilities).map(math.log))
    return round(float(losses.mean()), 6)


def add_metric_rows(rows: list[dict[str, Any]], section: str, group: str, frame: pd.DataFrame) -> None:
    """Add reliability metrics for a frame."""
    if frame.empty:
        return
    rows.extend(
        [
            metric_row(section, group, "contract_rows", len(frame)),
            metric_row(section, group, "markets", int(frame["market_id"].nunique())),
            metric_row(section, group, "mean_probability", round(float(frame["probability"].mean()), 6)),
            metric_row(section, group, "actual_win_rate", round(float(frame["actual"].mean()), 6)),
            metric_row(section, group, "brier_score", brier_score(frame), "Lower is better."),
            metric_row(section, group, "log_loss", log_loss(frame), "Lower is better; probabilities are clipped."),
        ]
    )


def build_summary(contracts: pd.DataFrame, coverage: dict[str, Any]) -> list[dict[str, Any]]:
    """Build compact coverage and metric summaries."""
    rows = [
        metric_row("coverage", "all", key, value)
        for key, value in coverage.items()
    ]
    add_metric_rows(rows, "metrics", "all", contracts)

    for category, category_frame in contracts.groupby("clean_category", sort=True):
        add_metric_rows(rows, "category_metrics", str(category), category_frame)

    return rows


def build_cutoff_targets(contracts: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Create cutoff targets for one horizon before market close."""
    columns = [
        "market_id",
        "event_id",
        "outcome_index",
        "outcome_name",
        "actual",
        "clean_category",
        "market_volume",
        "market_liquidity",
        "days_open",
        "price_points",
        "market_closed_time",
    ]
    targets = contracts[contracts["has_price_history"]].loc[:, columns].copy()
    closed = pd.to_datetime(targets["market_closed_time"], errors="coerce", utc=True)
    targets = targets[closed.notna()].copy()
    closed = closed[closed.notna()]
    targets["horizon_days"] = horizon_days
    targets["cutoff_timestamp"] = (closed.astype("int64") // 10**9) - (horizon_days * 86_400)
    targets["cutoff_time"] = pd.to_datetime(targets["cutoff_timestamp"], unit="s", utc=True)
    targets["target_id"] = range(len(targets))
    return targets


def lookup_cutoff_prices(targets: pd.DataFrame) -> pd.DataFrame:
    """Find the latest observed outcome price at or before each cutoff."""
    lookup_columns = ["target_id", "market_id", "outcome_index", "cutoff_timestamp"]
    rows = targets[lookup_columns].itertuples(index=False, name=None)
    query = """
        select
            t.target_id,
            ph.timestamp as forecast_timestamp,
            ph.datetime_utc as forecast_time,
            ph.price as forecast_probability
        from targets t
        left join price_history ph
          on ph.market_id = t.market_id
         and ph.outcome_index = t.outcome_index
         and ph.timestamp = (
            select max(timestamp)
            from price_history p2
            where p2.market_id = t.market_id
              and p2.outcome_index = t.outcome_index
              and p2.timestamp <= t.cutoff_timestamp
         )
        where ph.price is not null
    """
    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "create temp table targets "
            "(target_id integer, market_id text, outcome_index integer, cutoff_timestamp integer)"
        )
        cursor.executemany("insert into targets values (?, ?, ?, ?)", rows)
        cursor.execute("create index idx_temp_targets on targets(market_id, outcome_index)")
        prices = pd.read_sql_query(query, connection)
    return prices


def build_cutoff_contracts(contracts: pd.DataFrame) -> pd.DataFrame:
    """Build historical cutoff probability rows for reliability evaluation."""
    frames: list[pd.DataFrame] = []
    for horizon_days in HORIZON_DAYS:
        targets = build_cutoff_targets(contracts, horizon_days)
        prices = lookup_cutoff_prices(targets)
        merged = targets.merge(prices, on="target_id", how="inner", validate="one_to_one")
        merged["error"] = merged["forecast_probability"] - merged["actual"]
        merged["squared_error"] = merged["error"] ** 2
        merged["probability_source"] = "price_history.last_observed_before_cutoff"
        frames.append(merged.drop(columns=["target_id"]))
        print(
            f"Horizon {horizon_days}d: {len(merged):,} contract rows "
            f"from {merged['market_id'].nunique():,} markets",
            flush=True,
        )
    return pd.concat(frames, ignore_index=True)


def add_cutoff_summary(rows: list[dict[str, Any]], cutoff_contracts: pd.DataFrame) -> None:
    """Add historical cutoff coverage and metrics to the summary."""
    rows.extend(
        [
            metric_row("cutoff_coverage", "all", "contract_rows", len(cutoff_contracts)),
            metric_row("cutoff_coverage", "all", "markets", int(cutoff_contracts["market_id"].nunique())),
            metric_row(
                "cutoff_coverage",
                "all",
                "horizons",
                ",".join(str(day) for day in sorted(cutoff_contracts["horizon_days"].unique())),
            ),
        ]
    )

    metric_frame = cutoff_contracts.rename(columns={"forecast_probability": "probability"})
    for horizon_days, horizon_frame in metric_frame.groupby("horizon_days", sort=True):
        add_metric_rows(rows, "cutoff_metrics", f"{horizon_days}_days_before_close", horizon_frame)


def write_outputs(contracts: pd.DataFrame, cutoff_contracts: pd.DataFrame, rows: list[dict[str, Any]]) -> None:
    """Write the reliability dataset and summary CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    contracts.to_parquet(OUTPUT_PATH, index=False)
    cutoff_contracts.to_parquet(CUTOFF_OUTPUT_PATH, index=False)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build the reliability baseline dataset."""
    require_inputs()
    markets = load_markets()
    outcomes = load_outcomes()
    contracts, coverage = build_contracts(markets, outcomes)
    cutoff_contracts = build_cutoff_contracts(contracts)
    rows = build_summary(contracts, coverage)
    add_cutoff_summary(rows, cutoff_contracts)
    write_outputs(contracts, cutoff_contracts, rows)

    print(f"Wrote reliability contracts to {OUTPUT_PATH}", flush=True)
    print(f"Wrote reliability cutoffs to {CUTOFF_OUTPUT_PATH}", flush=True)
    print(f"Wrote reliability summary to {SUMMARY_PATH}", flush=True)
    print(f"Eligible markets: {contracts['market_id'].nunique():,}", flush=True)
    print(f"Contract rows: {len(contracts):,}", flush=True)
    print(f"Overall Brier score: {brier_score(contracts)}", flush=True)


if __name__ == "__main__":
    main()
