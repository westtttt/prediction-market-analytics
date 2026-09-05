"""Runtime premium query retrieval and current forecast aggregation."""

from __future__ import annotations

import math
import re
import sqlite3
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL_DIR = ROOT / "scripts" / "stage_05_retrieval"
FORECASTING_DIR = ROOT / "scripts" / "stage_06_forecasting"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(FORECASTING_DIR))

from aggregation_methods import aggregate_forecasts
from run_event_aware_retrieval import (
    CANDIDATE_K,
    TOP_EVENT_MARKETS,
    RerankConfig,
    build_candidates,
    build_event_features,
    fit_tfidf,
)


ANALYSIS_DIR = ROOT / "data" / "outputs" / "analysis"
DATABASE_PATH = ROOT / "data" / "database" / "markets.db"
SELECTED_CONFIG = RerankConfig("support_volume_stronger", 0.80, 0.10, 0.05, 0.05)
TOP_EVENTS = 10
TOP_MARKETS_PER_EVENT = 4
TOP_ANALYSIS_MARKETS = 60
TOP_FORECAST_MARKETS = 80


def clean_id(value: Any) -> str:
    """Normalise IDs loaded from Parquet/CSV for joins and JSON output."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def finite_float(value: Any, default: float = 0.0) -> float:
    """Convert numeric-like values to finite floats for JSON output."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number) or not math.isfinite(float(number)):
        return default
    return float(number)


def percent(part: float, whole: float) -> float:
    """Return a percentage rounded for dashboard display."""
    return 0.0 if whole <= 0 else round((part / whole) * 100, 2)


def expanded_query_text(query_text: str) -> str:
    """Add transparent domain terms for common shorthand forecast questions."""
    text = query_text.lower()
    expansions: list[str] = []
    if re.search(r"\btrump\b", text) and re.search(r"\b(re[- ]?elected|reelection|president|presidential)\b", text):
        expansions.append("2024 US presidential election winner Donald Trump president")
    if re.search(r"\bfed\b|\bfederal reserve\b", text) and re.search(r"\b(rate|rates|cut|cuts|hike|hikes)\b", text):
        expansions.append("Federal Reserve interest rate cuts FOMC")
    if re.search(r"\bbitcoin\b|\bbtc\b", text) and re.search(r"\b(price|prices|usd|dollar|2025)\b", text):
        expansions.append("Bitcoin BTC price USD")
    return " ".join([query_text, *expansions])


def parse_price_pairs(outcomes: Any, prices_value: Any) -> list[tuple[int, str, float]]:
    """Return outcome/probability pairs from semicolon-separated market fields."""
    if pd.isna(prices_value):
        return []
    names = [] if pd.isna(outcomes) else [part.strip() for part in str(outcomes).split(";")]
    prices: list[float] = []
    for part in str(prices_value).split(";"):
        number = pd.to_numeric(part.strip(), errors="coerce")
        if not pd.isna(number):
            prices.append(float(number))
    return [(index, names[index] if index < len(names) else f"Outcome {index + 1}", price) for index, price in enumerate(prices)]


def display_probability(outcomes: Any, prices_value: Any) -> tuple[int, str, float] | None:
    """Return the probability that best represents the proposition."""
    pairs = parse_price_pairs(outcomes, prices_value)
    if not pairs:
        return None
    yes_pair = next((pair for pair in pairs if pair[1].strip().lower() == "yes"), None)
    return yes_pair or max(pairs, key=lambda item: item[2])


def yes_probability(outcomes: Any, prices_value: Any) -> float | None:
    """Return the Yes probability for binary aggregation when available."""
    pairs = parse_price_pairs(outcomes, prices_value)
    yes_pair = next((pair for pair in pairs if pair[1].strip().lower() == "yes"), None)
    if yes_pair is None:
        return None
    probability = float(yes_pair[2])
    return probability if 0 <= probability <= 1 else None


def compact_market(row: pd.Series) -> dict[str, Any]:
    """Build a compact market record for the dashboard."""
    headline = display_probability(row.get("outcome_names"), row.get("current_prices"))
    winner = row.get("winner_outcome_names")
    return {
        "marketId": clean_id(row.get("market_id")),
        "question": str(row.get("market_question", "")),
        "volume": round(finite_float(row.get("market_volume")), 2),
        "hasPriceHistory": bool(row.get("has_price_history")),
        "pricePoints": int(finite_float(row.get("price_points"))),
        "isResolved": bool(row.get("is_resolved")),
        "displayOutcome": "" if headline is None else headline[1],
        "displayProbability": None if headline is None else round(headline[2], 4),
        "winner": "" if pd.isna(winner) else str(winner),
    }


def display_outcome_index(row: pd.Series) -> int:
    """Return the outcome index used for display/time-series analysis."""
    headline = display_probability(row.get("outcome_names"), row.get("current_prices"))
    return 0 if headline is None else int(headline[0])


def downsample_points(frame: pd.DataFrame, max_points: int = 70) -> list[dict[str, Any]]:
    """Downsample one market's price history for browser-friendly charting."""
    frame = frame.sort_values("timestamp")
    if len(frame) > max_points:
        keep = sorted({round(index * (len(frame) - 1) / (max_points - 1)) for index in range(max_points)})
        frame = frame.iloc[keep]
    return [{"time": str(row.datetime_utc), "price": round(finite_float(row.price), 4)} for row in frame.itertuples(index=False)]


def fetch_history(markets: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Fetch display-outcome price-history summaries for selected markets."""
    if markets.empty:
        return {}
    market_ids = [clean_id(value) for value in markets["market_id"]]
    outcome_indexes = {clean_id(row.market_id): display_outcome_index(row) for _, row in markets.iterrows()}
    placeholders = ",".join("?" for _ in market_ids)
    query = f"""
        select market_id, outcome_index, timestamp, datetime_utc, price
        from price_history
        where market_id in ({placeholders})
        order by market_id, outcome_index, timestamp
    """
    with sqlite3.connect(DATABASE_PATH) as connection:
        history = pd.read_sql_query(query, connection, params=market_ids)
    if history.empty:
        return {}
    history["market_id"] = history["market_id"].map(clean_id)
    history = history[history.apply(lambda row: int(row["outcome_index"]) == outcome_indexes.get(row["market_id"], 0), axis=1)]
    summaries: dict[str, dict[str, Any]] = {}
    for market_id, frame in history.groupby("market_id", sort=False):
        frame = frame.sort_values("timestamp")
        first_price = finite_float(frame["price"].iloc[0])
        latest_price = finite_float(frame["price"].iloc[-1])
        summaries[market_id] = {
            "firstProbability": round(first_price, 4),
            "latestHistoryProbability": round(latest_price, 4),
            "change": round(latest_price - first_price, 4),
            "range": round(finite_float(frame["price"].max()) - finite_float(frame["price"].min()), 4),
            "timeSeries": downsample_points(frame),
        }
    return summaries


def warning_messages(summary: dict[str, Any], category_counts: Counter[str]) -> list[str]:
    """Generate simple evidence-quality warnings for a retrieved set."""
    warnings: list[str] = []
    if summary["eventCount"] < 5:
        warnings.append("Thin evidence set: fewer than five unique retrieved events.")
    if summary["priceHistoryCoveragePercent"] < 60:
        warnings.append("Price-history coverage is limited, so movement and cutoff analysis are weaker.")
    if summary["resolvedPercent"] < 50:
        warnings.append("Most retrieved markets are unresolved, limiting outcome-based reliability checks.")
    if summary["medianVolume"] < 1_000:
        warnings.append("Median market volume is low, so individual market signals should be treated cautiously.")
    if summary["topEventVolumeSharePercent"] > 75:
        warnings.append("Evidence is concentrated in one event family rather than spread across many events.")
    if len(category_counts) > 2:
        warnings.append("Retrieved evidence spans several categories, suggesting a broad or ambiguous query.")
    return warnings


def interpretation(summary: dict[str, Any], warnings: list[str]) -> str:
    """Create a short deterministic interpretation for the dashboard."""
    if not warnings and summary["priceHistoryCoveragePercent"] >= 80 and summary["medianVolume"] >= 10_000:
        return "The retrieved set is broad enough for target analysis and has strong market-history coverage."
    if summary["resolvedPercent"] < 50:
        return "The retrieved set is useful for current monitoring, but historical outcome evaluation is limited."
    if summary["topEventVolumeSharePercent"] > 75:
        return "The evidence is highly concentrated in one event family, so event-level grouping is important."
    if summary["priceHistoryCoveragePercent"] < 60:
        return "The retrieved markets are relevant, but sparse price histories weaken probability-movement analysis."
    return "The retrieved set is usable for premium evidence review, with caveats shown in the warning checks."


@lru_cache(maxsize=1)
def runtime_index() -> dict[str, Any]:
    """Load retrieval artifacts and fit the TF-IDF candidate generator once."""
    corpus = pd.read_parquet(
        ANALYSIS_DIR / "retrieval_corpus.parquet",
        columns=[
            "market_id",
            "event_id",
            "event_title",
            "market_question",
            "clean_category",
            "primary_tag",
            "market_volume",
            "has_price_history",
            "price_points",
            "is_resolved",
        ],
    )
    details = pd.read_parquet(
        ANALYSIS_DIR / "markets_analysis.parquet",
        columns=["market_id", "outcome_names", "current_prices", "winner_outcome_names", "market_liquidity"],
    )
    for frame in [corpus, details]:
        frame["market_id"] = frame["market_id"].map(clean_id)
    corpus["event_id"] = corpus["event_id"].map(clean_id)
    corpus["event_id"] = corpus["event_id"].astype("string")
    corpus["market_volume"] = pd.to_numeric(corpus["market_volume"], errors="coerce").fillna(0)
    markets = corpus.merge(details, on="market_id", how="left", validate="one_to_one")
    vectorizer, matrix = fit_tfidf(corpus)
    return {"corpus": corpus, "markets": markets, "vectorizer": vectorizer, "matrix": matrix}


def retrieve_events(query_text: str) -> pd.DataFrame:
    """Run the dissertation event-aware retrieval method for an arbitrary query."""
    index = runtime_index()
    retrieval_query = expanded_query_text(query_text)
    query = pd.Series({"query": retrieval_query})
    candidates = build_candidates(query, index["corpus"], index["vectorizer"], index["matrix"])
    events = build_event_features(candidates)
    return rerank_runtime_events(events, retrieval_query)


def rerank_runtime_events(events: pd.DataFrame, query_text: str) -> pd.DataFrame:
    """Score candidate events with the evaluated reranker plus query year scope."""
    reranked = events.copy()
    reranked["event_score"] = (
        SELECTED_CONFIG.best_weight * reranked["best_tfidf_score"]
        + SELECTED_CONFIG.mean_top_weight * reranked["mean_top_tfidf_score"]
        + SELECTED_CONFIG.support_weight * reranked["support_signal"]
        + SELECTED_CONFIG.volume_weight * reranked["volume_signal"]
    )
    years = set(re.findall(r"\b(20\d{2})\b", query_text))
    if years:
        event_text = (
            reranked["event_title"].fillna("").astype(str)
            + " "
            + reranked["representative_market_question"].fillna("").astype(str)
        ).str.lower()
        reranked["year_scope_signal"] = event_text.map(lambda value: any(year in value for year in years))
        reranked["event_score"] = reranked["event_score"] + reranked["year_scope_signal"].astype(float) * 0.12
    reranked = reranked.sort_values(
        ["event_score", "best_tfidf_score", "first_market_candidate_rank"],
        ascending=[False, False, True],
    ).head(TOP_EVENTS)
    reranked["event_rank"] = range(1, len(reranked) + 1)
    return reranked


def build_query_forecast(query_text: str) -> dict[str, Any]:
    """Build a dashboard-ready target-analysis and current-forecast response."""
    query_text = " ".join(str(query_text).split())
    if len(query_text) < 3:
        raise ValueError("Enter at least three characters for the prediction query.")

    events = retrieve_events(query_text)
    event_ids = [clean_id(value) for value in events["event_id"]]
    markets = runtime_index()["markets"]
    evidence = markets[markets["event_id"].isin(event_ids)].copy()
    event_scores = {clean_id(row.event_id): float(row.event_score) for row in events.itertuples(index=False)}
    event_ranks = {clean_id(row.event_id): int(row.event_rank) for row in events.itertuples(index=False)}
    evidence["retrieval_rank"] = evidence["event_id"].map(event_ranks)
    evidence["retrieval_score"] = evidence["event_id"].map(event_scores)
    evidence["probability"] = evidence.apply(lambda row: yes_probability(row["outcome_names"], row["current_prices"]), axis=1)

    event_count = int(evidence["event_id"].nunique())
    market_count = int(len(evidence))
    resolved_count = int(evidence["is_resolved"].sum())
    history_count = int(evidence["has_price_history"].sum())
    total_volume = finite_float(evidence["market_volume"].sum())
    median_volume = finite_float(evidence["market_volume"].median())
    median_price_points = finite_float(evidence.loc[evidence["has_price_history"], "price_points"].median())
    category_counts = Counter(str(value) for value in evidence["clean_category"].dropna())
    dominant_category = category_counts.most_common(1)[0][0] if category_counts else ""

    history_seed = evidence.sort_values(["retrieval_rank", "market_volume"], ascending=[True, False]).head(TOP_ANALYSIS_MARKETS)
    history_by_market = fetch_history(history_seed)
    event_summaries: list[dict[str, Any]] = []
    for result in events.itertuples(index=False):
        event_id = clean_id(result.event_id)
        frame = evidence[evidence["event_id"].eq(event_id)].sort_values("market_volume", ascending=False)
        event_volume = finite_float(frame["market_volume"].sum())
        top_markets = [compact_market(row) for _, row in frame.head(TOP_MARKETS_PER_EVENT).iterrows()]
        for market in top_markets:
            market["timeSeries"] = history_by_market.get(market["marketId"], {}).get("timeSeries", [])
        event_summaries.append(
            {
                "eventId": event_id,
                "rank": int(result.event_rank),
                "score": round(finite_float(result.event_score), 6),
                "title": str(result.event_title),
                "category": str(result.clean_category),
                "markets": int(len(frame)),
                "resolvedMarkets": int(frame["is_resolved"].sum()),
                "priceHistoryCoveragePercent": percent(frame["has_price_history"].sum(), len(frame)),
                "volume": round(event_volume, 2),
                "volumeSharePercent": percent(event_volume, total_volume),
                "maxPricePoints": int(finite_float(frame["price_points"].max())),
                "topMarkets": top_markets,
            }
        )

    analysis_markets = []
    for _, row in evidence.sort_values("market_volume", ascending=False).head(TOP_ANALYSIS_MARKETS).iterrows():
        market = compact_market(row)
        history = history_by_market.get(clean_id(row["market_id"]), {})
        market.update(
            {
                "eventId": clean_id(row["event_id"]),
                "eventTitle": str(row["event_title"]),
                "category": str(row["clean_category"]),
                "retrievalRank": int(finite_float(row["retrieval_rank"])),
                "firstProbability": history.get("firstProbability"),
                "latestHistoryProbability": history.get("latestHistoryProbability"),
                "change": history.get("change"),
                "range": history.get("range"),
                "timeSeries": history.get("timeSeries", []),
            }
        )
        analysis_markets.append(market)

    top_event_volume = max((row["volume"] for row in event_summaries), default=0.0)
    display_probabilities = [
        None if (headline := display_probability(row.outcome_names, row.current_prices)) is None else headline[2]
        for row in evidence.itertuples(index=False)
    ]
    display_probabilities = [value for value in display_probabilities if value is not None]
    summary = {
        "queryId": f"runtime:{query_text.lower()}",
        "query": query_text,
        "expectedCategory": dominant_category,
        "eventCount": event_count,
        "marketCount": market_count,
        "resolvedMarkets": resolved_count,
        "unresolvedMarkets": market_count - resolved_count,
        "resolvedPercent": percent(resolved_count, market_count),
        "priceHistoryMarkets": history_count,
        "priceHistoryCoveragePercent": percent(history_count, market_count),
        "totalVolume": round(total_volume, 2),
        "medianVolume": round(median_volume, 2),
        "medianPricePoints": round(median_price_points, 2),
        "topEventVolumeSharePercent": percent(top_event_volume, total_volume),
        "dominantCategory": dominant_category,
        "dominantCategorySharePercent": percent(category_counts.get(dominant_category, 0), market_count),
        "probabilityMedian": None if not display_probabilities else round(float(pd.Series(display_probabilities).median()), 4),
        "probabilityIqr": None
        if not display_probabilities
        else round(float(pd.Series(display_probabilities).quantile(0.75) - pd.Series(display_probabilities).quantile(0.25)), 4),
    }
    warnings = warning_messages(summary, category_counts)
    target_payload = {
        "summary": summary,
        "events": event_summaries,
        "categoryBreakdown": [
            {"category": category, "markets": count, "sharePercent": percent(count, market_count)}
            for category, count in category_counts.most_common()
        ],
        "analysisMarkets": analysis_markets,
        "moverMarkets": sorted(
            [
                market
                for market in analysis_markets
                if market.get("change") is not None and market.get("firstProbability") is not None
            ],
            key=lambda market: abs(float(market["change"])),
            reverse=True,
        )[:12],
        "reliabilityContext": [],
        "warnings": warnings,
        "interpretation": interpretation(summary, warnings),
    }

    forecast_evidence = evidence[evidence["probability"].notna()].sort_values(
        ["retrieval_rank", "market_volume"], ascending=[True, False]
    ).head(TOP_FORECAST_MARKETS)
    forecast_evidence = forecast_evidence.assign(
        similarity_score=forecast_evidence["event_id"].map(event_scores),
        is_direct_target=False,
    )
    aggregates = aggregate_forecasts(forecast_evidence, dominant_category)
    selected = aggregates[aggregates["method"].eq("similarity_volume_weighted")]
    if selected.empty and not aggregates.empty:
        selected = aggregates.head(1)
    forecast_summary = {
        "queryId": summary["queryId"],
        "query": query_text,
        "forecastMode": "current snapshot",
        "databaseSource": "data/database/markets.db",
        "asOf": None,
        "accuracyEvaluated": False,
        "eventCount": int(forecast_evidence["event_id"].nunique()),
        "marketCount": int(len(forecast_evidence)),
        "resolvedMarkets": int(forecast_evidence["is_resolved"].sum()),
        "priceHistoryMarkets": int(forecast_evidence["has_price_history"].sum()),
        "totalVolume": round(float(pd.to_numeric(forecast_evidence["market_volume"], errors="coerce").fillna(0).sum()), 2),
        "selectedMethod": "" if selected.empty else str(selected.iloc[0]["method"]),
        "selectedForecastProbability": None if selected.empty else float(selected.iloc[0]["forecast_probability"]),
    }
    forecast_payload = {
        "summary": forecast_summary,
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
            for row in forecast_evidence.head(30).itertuples(index=False)
        ],
    }
    return {
        "retrieval": {
            "method": SELECTED_CONFIG.name,
            "candidateMarkets": CANDIDATE_K,
            "eventTopMarkets": TOP_EVENT_MARKETS,
            "source": "frozen Polymarket snapshot",
        },
        "target": target_payload,
        "forecast": forecast_payload,
    }
