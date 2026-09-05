"""Backtest aggregation methods using explicit as-of cutoffs."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from aggregation_methods import aggregate_forecasts, brier_score, log_loss

DATABASE_PATH = ROOT / "data" / "database" / "markets.db"
ANALYSIS_DIR = ROOT / "data" / "outputs" / "analysis"
FORECAST_OUTPUT = ANALYSIS_DIR / "aggregation_backtest_forecasts.parquet"
EVALUATION_OUTPUT = ANALYSIS_DIR / "aggregation_backtest_evaluation.parquet"
SUMMARY_OUTPUT = ANALYSIS_DIR / "aggregation_backtest_run_summary.csv"

HORIZON_DAYS = [30, 14, 7, 1]
DEFAULT_TARGETS = 80
TOP_CANDIDATES = 60
SELECTED_CONFIG = "support_volume_stronger"


def clean_id(value: Any) -> str:
    """Normalise ids loaded from CSV/Parquet."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def to_timestamp(series: pd.Series) -> pd.Series:
    """Return seconds since epoch for datetime-like values."""
    values = pd.to_datetime(series, errors="coerce", utc=True)
    return values.astype("int64") // 10**9


def load_markets() -> pd.DataFrame:
    """Load market fields needed for historical retrieval and target sampling."""
    columns = [
        "market_id",
        "event_id",
        "event_title",
        "market_question",
        "primary_tag",
        "market_volume",
        "market_liquidity",
        "market_created_at",
        "market_closed_time",
        "outcome_count",
        "outcome_names",
        "winner_outcome_names",
        "has_price_history",
        "price_points",
        "is_resolved",
        "retrieval_text",
    ]
    corpus = pd.read_parquet(ANALYSIS_DIR / "retrieval_corpus.parquet")
    details = pd.read_parquet(ANALYSIS_DIR / "markets_analysis.parquet", columns=[column for column in columns if column != "retrieval_text"])
    details["market_id"] = details["market_id"].map(clean_id)
    corpus["market_id"] = corpus["market_id"].map(clean_id)
    markets = details.merge(
        corpus[["market_id", "clean_category", "retrieval_text"]],
        on="market_id",
        how="left",
        validate="one_to_one",
    )
    markets["clean_category"] = markets["clean_category"].fillna("Other")
    markets["event_id"] = markets["event_id"].map(clean_id)
    markets["created_ts"] = to_timestamp(markets["market_created_at"])
    markets["closed_ts"] = to_timestamp(markets["market_closed_time"])
    markets["market_volume"] = pd.to_numeric(markets["market_volume"], errors="coerce").fillna(0)
    markets["market_liquidity"] = pd.to_numeric(markets["market_liquidity"], errors="coerce").fillna(0)
    return markets


def yes_outcome_index(outcome_names: Any) -> int | None:
    """Return the index of a Yes outcome, if present."""
    if pd.isna(outcome_names):
        return None
    names = [part.strip().lower() for part in str(outcome_names).split(";")]
    return names.index("yes") if "yes" in names else None


def yes_actual(winner_names: Any) -> int | None:
    """Return whether Yes was the resolved winning outcome."""
    if pd.isna(winner_names):
        return None
    return 1 if "yes" in [part.strip().lower() for part in str(winner_names).split(";")] else 0


def parse_event_ids(value: Any) -> list[str]:
    """Parse semicolon-separated relevant event ids."""
    if pd.isna(value):
        return []
    return [clean_id(part) for part in str(value).split(";") if clean_id(part)]


def load_queries_and_retrieval() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load labelled business queries and event-aware retrieval rows."""
    queries = pd.read_csv(ROOT / "config" / "retrieval_eval_queries.csv")
    queries["relevant_event_id_list"] = queries["relevant_event_ids"].map(parse_event_ids)
    retrieval = pd.read_csv(ANALYSIS_DIR / "event_aware_retrieval_results.csv")
    retrieval = retrieval[retrieval["config_name"].eq(SELECTED_CONFIG)].copy()
    retrieval["event_id"] = retrieval["event_id"].map(clean_id)
    return queries, retrieval


def sample_targets(markets: pd.DataFrame, queries: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Select resolved binary Yes/No targets from labelled relevant events."""
    targets = markets[
        markets["is_resolved"]
        & markets["has_price_history"]
        & markets["outcome_count"].eq(2)
        & markets["closed_ts"].notna()
    ].copy()
    targets["yes_index"] = targets["outcome_names"].map(yes_outcome_index)
    targets["actual"] = targets["winner_outcome_names"].map(yes_actual)
    targets = targets[targets["yes_index"].notna() & targets["actual"].notna()].copy()
    query_rows: list[pd.DataFrame] = []
    for query in queries.itertuples(index=False):
        relevant_events = set(query.relevant_event_id_list)
        if not relevant_events:
            continue
        query_targets = targets[targets["event_id"].isin(relevant_events)].copy()
        if query_targets.empty:
            continue
        query_targets = query_targets.sort_values("market_volume", ascending=False).head(2)
        query_targets["query_id"] = query.query_id
        query_targets["query"] = query.query
        query_rows.append(query_targets)
    if not query_rows:
        return targets.head(0).copy()
    sampled = pd.concat(query_rows, ignore_index=True)
    return sampled.sort_values(["clean_category", "market_volume"], ascending=[True, False]).head(limit).copy()


def lookup_prices_as_of(market_indexes: list[tuple[str, int]], as_of: int) -> pd.DataFrame:
    """Return latest observed outcome prices at or before the cutoff."""
    if not market_indexes:
        return pd.DataFrame(columns=["market_id", "outcome_index", "forecast_timestamp", "forecast_time", "probability"])
    rows = [(market_id, int(index), int(as_of)) for market_id, index in market_indexes]
    query = """
        select
            t.market_id,
            t.outcome_index,
            ph.timestamp as forecast_timestamp,
            ph.datetime_utc as forecast_time,
            ph.price as probability
        from targets t
        left join price_history ph
          on ph.market_id = t.market_id
         and ph.outcome_index = t.outcome_index
         and ph.timestamp = (
            select max(timestamp)
            from price_history p2
            where p2.market_id = t.market_id
              and p2.outcome_index = t.outcome_index
              and p2.timestamp <= t.as_of
         )
        where ph.price is not null
    """
    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute("drop table if exists targets")
        cursor.execute("create temp table targets (market_id text, outcome_index integer, as_of integer)")
        cursor.executemany("insert into targets values (?, ?, ?)", rows)
        prices = pd.read_sql_query(query, connection)
    prices["market_id"] = prices["market_id"].map(clean_id)
    return prices


def retrieved_evidence_as_of(target: pd.Series, as_of: int, markets: pd.DataFrame, retrieval: pd.DataFrame) -> pd.DataFrame:
    """Return retrieved markets available by cutoff using precomputed event-aware retrieval rows."""
    query_results = retrieval[retrieval["query_id"].eq(str(target["query_id"]))].sort_values("event_rank").head(10)
    event_scores = {clean_id(row.event_id): float(row.event_score) for row in query_results.itertuples(index=False)}
    candidates = markets[
        markets["event_id"].isin(event_scores)
        & markets["created_ts"].le(as_of)
        & markets["outcome_count"].eq(2)
        & markets["has_price_history"]
        & ~markets["market_id"].eq(clean_id(target["market_id"]))
    ].copy()
    candidates["yes_index"] = candidates["outcome_names"].map(yes_outcome_index)
    candidates = candidates[candidates["yes_index"].notna()].copy()
    candidates["similarity_score"] = candidates["event_id"].map(event_scores)
    candidates["is_direct_target"] = False
    return candidates.sort_values(["similarity_score", "market_volume"], ascending=[False, False]).head(TOP_CANDIDATES)


def build_forecasts_for_target(target: pd.Series, markets: pd.DataFrame, retrieval: pd.DataFrame) -> list[dict[str, Any]]:
    """Build forecast rows for all horizons and methods for one target."""
    rows: list[dict[str, Any]] = []
    target_market_id = clean_id(target["market_id"])
    query_text = str(target.get("query", target["market_question"]))
    target_yes_index = int(target["yes_index"])

    for horizon in HORIZON_DAYS:
        as_of = int(target["closed_ts"]) - horizon * 86_400
        if as_of <= int(target["created_ts"]):
            continue
        retrieved = retrieved_evidence_as_of(target, as_of, markets, retrieval)
        direct = target.to_frame().T.copy()
        direct["yes_index"] = target_yes_index
        direct["similarity_score"] = 1.0
        direct["is_direct_target"] = True
        evidence = pd.concat([direct, retrieved], ignore_index=True, sort=False)
        market_indexes = [(clean_id(row.market_id), int(row.yes_index)) for row in evidence.itertuples(index=False)]
        prices = lookup_prices_as_of(market_indexes, as_of)
        evidence["market_id"] = evidence["market_id"].map(clean_id)
        evidence = evidence.merge(prices, on="market_id", how="inner", validate="one_to_one")
        if evidence.empty:
            continue
        aggregates = aggregate_forecasts(evidence, target_category=str(target["clean_category"]))
        for aggregate in aggregates.itertuples(index=False):
            rows.append(
                {
                    "forecast_mode": "historical",
                    "database_source": "data/database/markets.db",
                    "target_market_id": target_market_id,
                    "target_event_id": clean_id(target["event_id"]),
                    "query_id": str(target.get("query_id", "")),
                    "query": query_text,
                    "target_question": query_text,
                    "target_category": str(target["clean_category"]),
                    "as_of_timestamp": as_of,
                    "as_of_time": pd.to_datetime(as_of, unit="s", utc=True).isoformat(),
                    "horizon_days": horizon,
                    "method": aggregate.method,
                    "forecast_probability": aggregate.forecast_probability,
                    "markets_used": aggregate.markets_used,
                    "events_used": aggregate.events_used,
                    "weight_sum": aggregate.weight_sum,
                    "warnings": aggregate.warnings,
                    "max_forecast_timestamp": int(evidence["forecast_timestamp"].max()),
                    "max_forecast_time": str(evidence.sort_values("forecast_timestamp").iloc[-1]["forecast_time"]),
                    "outcome_joined": False,
                }
            )
    return rows


def evaluate_forecasts(forecasts: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Join eventual outcomes only after forecast rows have been generated."""
    labels = targets[["market_id", "actual", "winner_outcome_names"]].copy()
    labels["market_id"] = labels["market_id"].map(clean_id)
    evaluated = forecasts.merge(labels, left_on="target_market_id", right_on="market_id", how="left", validate="many_to_one")
    evaluated = evaluated.drop(columns=["market_id"])
    evaluated["actual"] = evaluated["actual"].astype("int8")
    evaluated["brier_score"] = evaluated.apply(lambda row: brier_score(row["forecast_probability"], row["actual"]), axis=1)
    evaluated["log_loss"] = evaluated.apply(lambda row: log_loss(row["forecast_probability"], row["actual"]), axis=1)
    evaluated["outcome_joined"] = True
    return evaluated


def run_backtest(target_limit: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the bounded aggregation backtest."""
    markets = load_markets()
    queries, retrieval = load_queries_and_retrieval()
    targets = sample_targets(markets, queries, target_limit)
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(targets.itertuples(index=False), start=1):
        rows.extend(build_forecasts_for_target(pd.Series(target._asdict()), markets, retrieval))
        if index % 10 == 0:
            print(f"Processed {index}/{len(targets)} targets", flush=True)
    forecasts = pd.DataFrame(rows)
    evaluated = evaluate_forecasts(forecasts, targets) if not forecasts.empty else forecasts.copy()
    summary = pd.DataFrame(
        [
            {"metric": "target_limit", "value": target_limit},
            {"metric": "sampled_targets", "value": len(targets)},
            {"metric": "forecast_rows", "value": len(forecasts)},
            {"metric": "evaluated_rows", "value": len(evaluated)},
            {"metric": "horizons", "value": ",".join(str(day) for day in HORIZON_DAYS)},
            {"metric": "methods", "value": ",".join(sorted(forecasts["method"].unique())) if not forecasts.empty else ""},
            {
                "metric": "leakage_check_rows_after_as_of",
                "value": int((forecasts["max_forecast_timestamp"] > forecasts["as_of_timestamp"]).sum()) if not forecasts.empty else 0,
            },
        ]
    )
    return forecasts, evaluated, summary


def main() -> None:
    """Run and persist historical aggregation backtest outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-limit", type=int, default=DEFAULT_TARGETS)
    args = parser.parse_args()

    forecasts, evaluated, summary = run_backtest(args.target_limit)
    forecasts.to_parquet(FORECAST_OUTPUT, index=False)
    evaluated.to_parquet(EVALUATION_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    print(f"Wrote forecast rows to {FORECAST_OUTPUT}", flush=True)
    print(f"Wrote evaluated rows to {EVALUATION_OUTPUT}", flush=True)
    print(f"Wrote run summary to {SUMMARY_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
