"""Build current-mode premium forecast examples from latest retrieved markets."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from aggregation_methods import aggregate_forecasts

ANALYSIS_DIR = ROOT / "data" / "outputs" / "analysis"
OUTPUT_JSON = ANALYSIS_DIR / "current_forecast_examples.json"
OUTPUT_JS = ROOT / "website" / "data" / "current_forecast_data.js"
SUMMARY_CSV = ANALYSIS_DIR / "current_forecast_examples_summary.csv"

SELECTED_QUERY_IDS = ["q01", "q03", "q10", "q20", "q23"]
SELECTED_CONFIG = "support_volume_stronger"
TOP_EVENTS = 10
TOP_MARKETS = 80


def clean_id(value: Any) -> str:
    """Normalise IDs loaded from parquet/CSV."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def yes_probability(outcome_names: Any, current_prices: Any) -> float | None:
    """Return the current Yes probability from semicolon separated market fields."""
    if pd.isna(outcome_names) or pd.isna(current_prices):
        return None
    names = [part.strip().lower() for part in str(outcome_names).split(";")]
    prices = [pd.to_numeric(part.strip(), errors="coerce") for part in str(current_prices).split(";")]
    if "yes" not in names:
        return None
    index = names.index("yes")
    if index >= len(prices) or pd.isna(prices[index]):
        return None
    probability = float(prices[index])
    return probability if 0 <= probability <= 1 else None


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load query labels, retrieval rows, and market details."""
    queries = pd.read_csv(ROOT / "config" / "retrieval_eval_queries.csv")
    queries = queries[queries["query_id"].isin(SELECTED_QUERY_IDS)].copy()
    retrieval = pd.read_csv(ANALYSIS_DIR / "event_aware_retrieval_results.csv")
    retrieval = retrieval[retrieval["config_name"].eq(SELECTED_CONFIG)].copy()
    corpus = pd.read_parquet(
        ANALYSIS_DIR / "retrieval_corpus.parquet",
        columns=[
            "market_id",
            "event_id",
            "event_title",
            "market_question",
            "clean_category",
            "market_volume",
            "has_price_history",
            "price_points",
            "is_resolved",
        ],
    )
    details = pd.read_parquet(
        ANALYSIS_DIR / "markets_analysis.parquet",
        columns=["market_id", "outcome_names", "current_prices", "market_liquidity"],
    )
    corpus["market_id"] = corpus["market_id"].map(clean_id)
    corpus["event_id"] = corpus["event_id"].map(clean_id)
    details["market_id"] = details["market_id"].map(clean_id)
    markets = corpus.merge(details, on="market_id", how="left", validate="one_to_one")
    markets["probability"] = markets.apply(lambda row: yes_probability(row["outcome_names"], row["current_prices"]), axis=1)
    return queries, retrieval, markets


def build_example(query: pd.Series, retrieval: pd.DataFrame, markets: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build one current forecast example and flat summary rows."""
    query_id = str(query.query_id)
    query_results = retrieval[retrieval["query_id"].eq(query_id)].sort_values("event_rank").head(TOP_EVENTS)
    event_scores = {clean_id(row.event_id): float(row.event_score) for row in query_results.itertuples(index=False)}
    event_ranks = {clean_id(row.event_id): int(row.event_rank) for row in query_results.itertuples(index=False)}
    evidence = markets[markets["event_id"].isin(event_scores)].copy()
    evidence["similarity_score"] = evidence["event_id"].map(event_scores)
    evidence["retrieval_rank"] = evidence["event_id"].map(event_ranks)
    evidence["is_direct_target"] = False
    evidence = evidence[evidence["probability"].notna()].sort_values(
        ["retrieval_rank", "market_volume"], ascending=[True, False]
    ).head(TOP_MARKETS)

    aggregates = aggregate_forecasts(evidence)
    selected = aggregates[aggregates["method"].eq("similarity_volume_weighted")]
    if selected.empty and not aggregates.empty:
        selected = aggregates.head(1)

    summary = {
        "queryId": query_id,
        "query": str(query.query),
        "forecastMode": "current",
        "databaseSource": "data/database/markets.db",
        "asOf": None,
        "accuracyEvaluated": False,
        "eventCount": int(evidence["event_id"].nunique()),
        "marketCount": int(len(evidence)),
        "resolvedMarkets": int(evidence["is_resolved"].sum()),
        "priceHistoryMarkets": int(evidence["has_price_history"].sum()),
        "totalVolume": round(float(pd.to_numeric(evidence["market_volume"], errors="coerce").fillna(0).sum()), 2),
        "selectedMethod": "" if selected.empty else str(selected.iloc[0]["method"]),
        "selectedForecastProbability": None if selected.empty else float(selected.iloc[0]["forecast_probability"]),
    }
    payload = {
        "summary": summary,
        "aggregationMethods": aggregates.to_dict(orient="records"),
        "evidenceMarkets": [
            {
                "marketId": clean_id(row.market_id),
                "eventId": clean_id(row.event_id),
                "eventTitle": str(row.event_title),
                "question": str(row.market_question),
                "category": str(row.clean_category),
                "retrievalRank": int(row.retrieval_rank),
                "similarityScore": round(float(row.similarity_score), 6),
                "probability": round(float(row.probability), 6),
                "volume": round(float(pd.to_numeric(row.market_volume, errors="coerce") or 0), 2),
                "hasPriceHistory": bool(row.has_price_history),
                "isResolvedInSnapshot": bool(row.is_resolved),
            }
            for row in evidence.head(30).itertuples(index=False)
        ],
    }
    rows = [{**summary, **row} for row in payload["aggregationMethods"]]
    return payload, rows


def main() -> None:
    """Write current forecast examples for premium dashboard evidence."""
    queries, retrieval, markets = load_inputs()
    examples: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for _, query in queries.iterrows():
        example, rows = build_example(query, retrieval, markets)
        examples.append(example)
        summary_rows.extend(rows)

    payload = {
        "forecastMode": "current",
        "accuracyEvaluated": False,
        "methodNote": "Current forecasts use latest available snapshot data and deliberately omit accuracy metrics.",
        "examples": examples,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUTPUT_JS.write_text(f"window.CURRENT_FORECAST_DATA = {json.dumps(payload, indent=2)};\n", encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote current forecast examples to {OUTPUT_JSON}", flush=True)
    print(f"Wrote dashboard current forecast data to {OUTPUT_JS}", flush=True)
    print(f"Wrote current forecast summary to {SUMMARY_CSV}", flush=True)


if __name__ == "__main__":
    main()
