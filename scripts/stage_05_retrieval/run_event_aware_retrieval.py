"""Run event-aware reranking over tuned TF-IDF market candidates."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "data" / "outputs" / "analysis" / "retrieval_corpus.parquet"
QUERY_PATH = ROOT / "config" / "retrieval_eval_queries.csv"
RESULTS_PATH = ROOT / "data" / "outputs" / "analysis" / "event_aware_retrieval_results.csv"
SUMMARY_PATH = ROOT / "data" / "outputs" / "analysis" / "event_aware_retrieval_summary.csv"
TUNING_PATH = ROOT / "data" / "outputs" / "analysis" / "event_aware_retrieval_tuning.csv"
COMPARISON_PATH = ROOT / "data" / "outputs" / "analysis" / "retrieval_comparison_summary.csv"

CANDIDATE_K = 100
TOP_K = 10
TOP_EVENT_MARKETS = 3

CORPUS_COLUMNS = [
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
]


@dataclass(frozen=True)
class RerankConfig:
    """One event-aware reranking configuration."""

    name: str
    best_weight: float
    mean_top_weight: float
    support_weight: float
    volume_weight: float


CONFIGS = [
    RerankConfig("tfidf_unique_event_baseline", 1.00, 0.00, 0.00, 0.00),
    RerankConfig("support_light", 1.00, 0.00, 0.03, 0.00),
    RerankConfig("support_mean", 0.85, 0.15, 0.03, 0.00),
    RerankConfig("support_volume_light", 0.85, 0.10, 0.03, 0.02),
    RerankConfig("support_volume_stronger", 0.80, 0.10, 0.05, 0.05),
]


def require_inputs() -> None:
    """Fail clearly if required inputs are missing."""
    missing = [path for path in [CORPUS_PATH, QUERY_PATH] if not path.exists()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing input file(s): {paths}")


def load_corpus() -> pd.DataFrame:
    """Load retrieval fields and normalise types."""
    corpus = pd.read_parquet(CORPUS_PATH, columns=CORPUS_COLUMNS)
    corpus["event_id"] = corpus["event_id"].astype("string")
    corpus["market_volume"] = pd.to_numeric(corpus["market_volume"], errors="coerce").fillna(0)
    return corpus


def load_queries() -> pd.DataFrame:
    """Load labelled retrieval queries."""
    queries = pd.read_csv(QUERY_PATH)
    queries["relevant_event_id_set"] = queries["relevant_event_ids"].map(parse_event_ids)
    return queries


def build_text(corpus: pd.DataFrame) -> pd.Series:
    """Build the tuned compact event-heavy TF-IDF text."""
    event = corpus["event_title"].fillna("").astype("string")
    question = corpus["market_question"].fillna("").astype("string")
    category = corpus["clean_category"].fillna("").astype("string")
    tag = corpus["primary_tag"].fillna("").astype("string")
    return event + " " + event + " " + question + " " + category + " " + tag


def fit_tfidf(corpus: pd.DataFrame) -> tuple[TfidfVectorizer, Any]:
    """Fit the selected TF-IDF candidate generator."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=80_000,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(build_text(corpus).fillna(""))
    return vectorizer, matrix


def top_candidate_indexes(scores: np.ndarray, candidate_k: int) -> np.ndarray:
    """Return candidate indexes sorted by descending score."""
    limit = min(candidate_k, len(scores))
    indexes = np.argpartition(scores, -limit)[-limit:]
    return indexes[np.argsort(scores[indexes])[::-1]]


def build_candidates(query: pd.Series, corpus: pd.DataFrame, vectorizer: TfidfVectorizer, matrix: Any) -> pd.DataFrame:
    """Return top market candidates for one query."""
    query_vector = vectorizer.transform([str(query["query"])])
    scores = (query_vector @ matrix.T).toarray().ravel()
    indexes = top_candidate_indexes(scores, CANDIDATE_K)
    candidates = corpus.iloc[indexes].copy()
    candidates["tfidf_score"] = scores[indexes]
    candidates["candidate_rank"] = range(1, len(candidates) + 1)
    return candidates


def build_event_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Aggregate market candidates into event-level features."""
    candidate_events = []
    for event_id, frame in candidates.groupby("event_id", sort=False):
        ordered = frame.sort_values("candidate_rank")
        top_scores = ordered["tfidf_score"].head(TOP_EVENT_MARKETS)
        best = ordered.iloc[0]
        candidate_events.append(
            {
                "event_id": event_id,
                "event_title": best["event_title"],
                "representative_market_id": best["market_id"],
                "representative_market_question": best["market_question"],
                "clean_category": best["clean_category"],
                "primary_tag": best["primary_tag"],
                "best_tfidf_score": float(top_scores.iloc[0]),
                "mean_top_tfidf_score": float(top_scores.mean()),
                "candidate_market_count": int(len(frame)),
                "first_market_candidate_rank": int(best["candidate_rank"]),
                "event_market_volume": float(frame["market_volume"].sum()),
                "has_price_history_count": int(frame["has_price_history"].sum()),
                "max_price_points": int(pd.to_numeric(frame["price_points"], errors="coerce").fillna(0).max()),
                "is_resolved_any": bool(frame["is_resolved"].any()),
            }
        )
    events = pd.DataFrame(candidate_events)
    events["support_signal"] = normalise(np.log1p(events["candidate_market_count"]))
    events["volume_signal"] = normalise(np.log1p(events["event_market_volume"]))
    return events


def normalise(series: pd.Series) -> pd.Series:
    """Min-max normalise a numeric series within one query."""
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum == minimum:
        return pd.Series(0.0, index=series.index)
    return (values - minimum) / (maximum - minimum)


def rerank_events(events: pd.DataFrame, config: RerankConfig) -> pd.DataFrame:
    """Score and rank candidate events."""
    reranked = events.copy()
    reranked["event_score"] = (
        config.best_weight * reranked["best_tfidf_score"]
        + config.mean_top_weight * reranked["mean_top_tfidf_score"]
        + config.support_weight * reranked["support_signal"]
        + config.volume_weight * reranked["volume_signal"]
    )
    reranked = reranked.sort_values(
        ["event_score", "best_tfidf_score", "first_market_candidate_rank"],
        ascending=[False, False, True],
    ).head(TOP_K)
    reranked["event_rank"] = range(1, len(reranked) + 1)
    return reranked


def evaluate_configs(corpus: pd.DataFrame, queries: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    """Evaluate each reranking configuration."""
    vectorizer, matrix = fit_tfidf(corpus)
    results_by_config = {config.name: [] for config in CONFIGS}

    for query in queries.itertuples(index=False):
        query_series = pd.Series(query._asdict())
        candidates = build_candidates(query_series, corpus, vectorizer, matrix)
        events = build_event_features(candidates)
        relevant_ids = query_series["relevant_event_id_set"]
        for config in CONFIGS:
            reranked = rerank_events(events, config)
            for item in reranked.itertuples(index=False):
                results_by_config[config.name].append(result_row(query_series, item, config, str(item.event_id) in relevant_ids))

    result_frames = {name: pd.DataFrame(rows) for name, rows in results_by_config.items()}
    tuning_rows = [build_tuning_row(config, result_frames[config.name]) for config in CONFIGS]
    return tuning_rows, result_frames


def result_row(query: pd.Series, item: Any, config: RerankConfig, event_relevant: bool) -> dict[str, Any]:
    """Return one event-level retrieval result row."""
    return {
        "config_name": config.name,
        "query_id": query["query_id"],
        "query": query["query"],
        "query_type": query["query_type"],
        "expected_category": query["expected_category"],
        "relevant_event_ids": query["relevant_event_ids"],
        "event_rank": item.event_rank,
        "event_score": round(float(item.event_score), 6),
        "best_tfidf_score": round(float(item.best_tfidf_score), 6),
        "mean_top_tfidf_score": round(float(item.mean_top_tfidf_score), 6),
        "support_signal": round(float(item.support_signal), 6),
        "volume_signal": round(float(item.volume_signal), 6),
        "candidate_market_count": item.candidate_market_count,
        "first_market_candidate_rank": item.first_market_candidate_rank,
        "event_id": item.event_id,
        "event_title": item.event_title,
        "representative_market_id": item.representative_market_id,
        "representative_market_question": item.representative_market_question,
        "clean_category": item.clean_category,
        "primary_tag": item.primary_tag,
        "event_market_volume": round(float(item.event_market_volume), 4),
        "has_price_history_count": item.has_price_history_count,
        "max_price_points": item.max_price_points,
        "is_resolved_any": item.is_resolved_any,
        "event_relevant": event_relevant,
        "category_match": item.clean_category == query["expected_category"],
    }


def build_tuning_row(config: RerankConfig, results: pd.DataFrame) -> dict[str, Any]:
    """Build one compact evaluation row for a reranker."""
    return {
        "config_name": config.name,
        "best_weight": config.best_weight,
        "mean_top_weight": config.mean_top_weight,
        "support_weight": config.support_weight,
        "volume_weight": config.volume_weight,
        "event_precision_at_1": precision_at_k(results, 1),
        "event_precision_at_5": precision_at_k(results, 5),
        "event_precision_at_10": precision_at_k(results, 10),
        "event_hit_at_1": hit_at_k(results, 1),
        "event_hit_at_5": hit_at_k(results, 5),
        "event_hit_at_10": hit_at_k(results, 10),
        "event_mrr": mean_reciprocal_rank(results),
        "top1_category_match_percent": category_hit_at_k(results, 1),
        "top10_any_category_match_percent": category_hit_at_k(results, 10),
    }


def select_config(tuning_rows: list[dict[str, Any]]) -> str:
    """Select the strongest reranker using a transparent ranking rule."""
    frame = pd.DataFrame(tuning_rows)
    frame = frame.sort_values(
        ["event_mrr", "event_hit_at_1", "event_hit_at_5", "event_hit_at_10"],
        ascending=[False, False, False, False],
    )
    return str(frame.iloc[0]["config_name"])


def build_summary(best_name: str, tuning_rows: list[dict[str, Any]], best_results: pd.DataFrame) -> list[dict[str, Any]]:
    """Build compact summary rows for the selected reranker."""
    best = next(item for item in tuning_rows if item["config_name"] == best_name)
    baseline = next(item for item in tuning_rows if item["config_name"] == "tfidf_unique_event_baseline")
    rows = [
        row("setup", "queries", int(best_results["query_id"].nunique())),
        row("setup", "candidate_k", CANDIDATE_K),
        row("setup", "top_k", TOP_K),
        row("setup", "best_config", best_name),
        row("setup", "tfidf_candidate_generator", "compact_event_heavy_1_2"),
        row("event_relevance", "precision_at_1", best["event_precision_at_1"]),
        row("event_relevance", "precision_at_5", best["event_precision_at_5"]),
        row("event_relevance", "precision_at_10", best["event_precision_at_10"]),
        row("event_relevance", "hit_at_1", best["event_hit_at_1"]),
        row("event_relevance", "hit_at_5", best["event_hit_at_5"]),
        row("event_relevance", "hit_at_10", best["event_hit_at_10"]),
        row("event_relevance", "mrr", best["event_mrr"]),
        row("baseline_comparison", "baseline_hit_at_1", baseline["event_hit_at_1"]),
        row("baseline_comparison", "baseline_hit_at_5", baseline["event_hit_at_5"]),
        row("baseline_comparison", "baseline_hit_at_10", baseline["event_hit_at_10"]),
        row("baseline_comparison", "baseline_mrr", baseline["event_mrr"]),
        row("baseline_comparison", "mrr_change", round(float(best["event_mrr"]) - float(baseline["event_mrr"]), 4)),
        row("baseline_comparison", "hit_at_1_change", round(float(best["event_hit_at_1"]) - float(baseline["event_hit_at_1"]), 2)),
    ]
    for query_type, frame in best_results.groupby("query_type", sort=True):
        rows.append(row("query_type", f"{query_type}_hit_at_10", hit_at_k(frame, 10)))
        rows.append(row("query_type", f"{query_type}_mrr", mean_reciprocal_rank(frame)))
    return rows


def build_comparison_rows(tuning_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a compact baseline/reranker comparison table."""
    rows = []
    for item in tuning_rows:
        rows.append(
            {
                "method": item["config_name"],
                "hit_at_1": item["event_hit_at_1"],
                "hit_at_5": item["event_hit_at_5"],
                "hit_at_10": item["event_hit_at_10"],
                "mrr": item["event_mrr"],
                "precision_at_5": item["event_precision_at_5"],
                "precision_at_10": item["event_precision_at_10"],
            }
        )
    return rows


def parse_event_ids(value: Any) -> set[str]:
    """Parse semicolon-separated event IDs."""
    if pd.isna(value):
        return set()
    return {item.strip() for item in str(value).split(";") if item.strip()}


def precision_at_k(results: pd.DataFrame, k: int) -> float:
    """Return mean event precision at k."""
    scores = []
    for _, frame in results[results["event_rank"].le(k)].groupby("query_id"):
        denominator = min(k, len(frame))
        if denominator:
            scores.append(float(frame["event_relevant"].sum()) / denominator)
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


def hit_at_k(results: pd.DataFrame, k: int) -> float:
    """Return percentage of queries with a relevant event in top k."""
    hits = results[results["event_rank"].le(k)].groupby("query_id")["event_relevant"].any()
    return pct(hits.sum(), len(hits))


def mean_reciprocal_rank(results: pd.DataFrame) -> float:
    """Return mean reciprocal rank for the first relevant event."""
    reciprocal_ranks = []
    for _, frame in results.groupby("query_id"):
        relevant = frame[frame["event_relevant"]].sort_values("event_rank")
        reciprocal_ranks.append(0.0 if relevant.empty else 1.0 / float(relevant.iloc[0]["event_rank"]))
    if not reciprocal_ranks:
        return 0.0
    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)


def category_hit_at_k(results: pd.DataFrame, k: int) -> float:
    """Return percentage of queries with expected category represented in top k."""
    hits = results[results["event_rank"].le(k)].groupby("query_id")["category_match"].any()
    return pct(hits.sum(), len(hits))


def row(section: str, metric: str, value: Any, notes: str = "") -> dict[str, Any]:
    """Return one summary row."""
    return {"section": section, "metric": metric, "value": value, "notes": notes}


def pct(part: Any, whole: Any) -> float:
    """Return a rounded percentage."""
    denominator = float(whole or 0)
    if denominator == 0:
        return 0.0
    return round(float(part or 0) / denominator * 100, 2)


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionary rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run event-aware reranking and save comparison outputs."""
    require_inputs()
    corpus = load_corpus()
    queries = load_queries()
    print(f"Loaded {len(corpus):,} documents and {len(queries):,} queries", flush=True)
    tuning_rows, result_frames = evaluate_configs(corpus, queries)
    best_name = select_config(tuning_rows)
    best_results = result_frames[best_name]

    best_results.to_csv(RESULTS_PATH, index=False)
    write_dict_rows(TUNING_PATH, tuning_rows)
    write_dict_rows(SUMMARY_PATH, build_summary(best_name, tuning_rows, best_results))
    write_dict_rows(COMPARISON_PATH, build_comparison_rows(tuning_rows))

    print(f"Selected reranker: {best_name}", flush=True)
    print(f"Wrote event-aware results to {RESULTS_PATH}", flush=True)
    print(f"Wrote event-aware summary to {SUMMARY_PATH}", flush=True)
    print(f"Wrote reranker tuning to {TUNING_PATH}", flush=True)
    print(f"Wrote retrieval comparison to {COMPARISON_PATH}", flush=True)


if __name__ == "__main__":
    main()
