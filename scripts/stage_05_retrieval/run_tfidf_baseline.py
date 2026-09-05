"""Run and tune a TF-IDF retrieval baseline over the market corpus."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "data" / "outputs" / "analysis" / "retrieval_corpus.parquet"
QUERY_PATH = ROOT / "config" / "retrieval_eval_queries.csv"
RESULTS_PATH = ROOT / "data" / "outputs" / "analysis" / "tfidf_baseline_results.csv"
EVENT_RESULTS_PATH = ROOT / "data" / "outputs" / "analysis" / "tfidf_baseline_event_results.csv"
SUMMARY_PATH = ROOT / "data" / "outputs" / "analysis" / "tfidf_baseline_summary.csv"
TUNING_PATH = ROOT / "data" / "outputs" / "analysis" / "tfidf_baseline_tuning.csv"

TOP_K = 10
CANDIDATE_K = 100


@dataclass(frozen=True)
class TfidfConfig:
    """One TF-IDF configuration to evaluate."""

    name: str
    text_variant: str
    ngram_range: tuple[int, int]
    min_df: int
    max_features: int


CONFIGS = [
    TfidfConfig("compact_question_heavy_1_2", "compact_question_heavy", (1, 2), 2, 80_000),
    TfidfConfig("compact_market_event_1_2", "compact_market_event", (1, 2), 2, 80_000),
    TfidfConfig("compact_event_heavy_1_2", "compact_event_heavy", (1, 2), 2, 80_000),
    TfidfConfig("compact_market_event_1_3", "compact_market_event", (1, 3), 2, 120_000),
]

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
    "retrieval_text",
]


def require_inputs() -> None:
    """Fail clearly if required inputs are missing."""
    missing = [path for path in [CORPUS_PATH, QUERY_PATH] if not path.exists()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing input file(s): {paths}")


def load_corpus() -> pd.DataFrame:
    """Load the retrieval corpus."""
    corpus = pd.read_parquet(CORPUS_PATH, columns=CORPUS_COLUMNS)
    corpus["event_id"] = corpus["event_id"].astype("string")
    corpus["market_volume"] = pd.to_numeric(corpus["market_volume"], errors="coerce").fillna(0)
    return corpus


def load_queries() -> pd.DataFrame:
    """Load predefined retrieval queries."""
    queries = pd.read_csv(QUERY_PATH)
    queries["relevant_event_id_set"] = queries["relevant_event_ids"].map(parse_event_ids)
    return queries


def build_text(corpus: pd.DataFrame, variant: str) -> pd.Series:
    """Build the text representation for one TF-IDF variant."""
    question = corpus["market_question"].fillna("").astype("string")
    event = corpus["event_title"].fillna("").astype("string")
    category = corpus["clean_category"].fillna("").astype("string")
    tag = corpus["primary_tag"].fillna("").astype("string")

    if variant == "current_weighted":
        return corpus["retrieval_text"].fillna("").astype("string")
    if variant == "compact_question_heavy":
        return question + " " + question + " " + question + " " + event + " " + category + " " + tag
    if variant == "compact_market_event":
        return question + " " + question + " " + event + " " + category + " " + tag
    if variant == "compact_event_heavy":
        return event + " " + event + " " + question + " " + category + " " + tag
    raise ValueError(f"Unknown text variant: {variant}")


def fit_vectorizer(text: pd.Series, config: TfidfConfig) -> tuple[TfidfVectorizer, sparse.spmatrix]:
    """Fit one TF-IDF vectorizer."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_features=config.max_features,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(text.fillna(""))
    return vectorizer, matrix


def top_candidate_indexes(scores: np.ndarray, candidate_k: int) -> np.ndarray:
    """Return candidate indexes sorted by descending score."""
    limit = min(candidate_k, len(scores))
    if limit == len(scores):
        indexes = np.argsort(scores)[::-1]
    else:
        indexes = np.argpartition(scores, -limit)[-limit:]
        indexes = indexes[np.argsort(scores[indexes])[::-1]]
    return indexes


def run_query(
    query: pd.Series,
    corpus: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix: sparse.spmatrix,
    config: TfidfConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one query and return market-row and event-level results."""
    query_vector = vectorizer.transform([str(query["query"])])
    scores = (query_vector @ matrix.T).toarray().ravel()
    candidate_indexes = top_candidate_indexes(scores, CANDIDATE_K)
    relevant_event_ids = query["relevant_event_id_set"]

    market_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    seen_events: set[str] = set()

    for candidate_rank, index in enumerate(candidate_indexes, start=1):
        item = corpus.iloc[int(index)]
        event_id = str(item["event_id"])
        result = result_row(
            query=query,
            item=item,
            score=float(scores[index]),
            config=config,
            rank=candidate_rank,
            event_rank=None,
            event_relevant=event_id in relevant_event_ids,
        )
        if len(market_rows) < TOP_K:
            market_rows.append({key: value for key, value in result.items() if key != "event_rank"})
        if event_id not in seen_events:
            seen_events.add(event_id)
            event_result = result.copy()
            event_result["event_rank"] = len(event_rows) + 1
            event_result["first_market_candidate_rank"] = candidate_rank
            event_rows.append(event_result)
        if len(event_rows) >= TOP_K and len(market_rows) >= TOP_K:
            break

    return market_rows, event_rows[:TOP_K]


def result_row(
    query: pd.Series,
    item: pd.Series,
    score: float,
    config: TfidfConfig,
    rank: int,
    event_rank: int | None,
    event_relevant: bool,
) -> dict[str, Any]:
    """Return one detailed retrieval result row."""
    return {
        "config_name": config.name,
        "query_id": query["query_id"],
        "query": query["query"],
        "query_type": query.get("query_type", ""),
        "expected_category": query["expected_category"],
        "relevant_event_ids": query.get("relevant_event_ids", ""),
        "rank": rank,
        "event_rank": event_rank,
        "score": round(score, 6),
        "market_id": item["market_id"],
        "event_id": item["event_id"],
        "event_title": item["event_title"],
        "market_question": item["market_question"],
        "clean_category": item["clean_category"],
        "market_volume": item["market_volume"],
        "has_price_history": item["has_price_history"],
        "price_points": item["price_points"],
        "is_resolved": item["is_resolved"],
        "category_match": item["clean_category"] == query["expected_category"],
        "event_relevant": event_relevant,
    }


def evaluate_config(corpus: pd.DataFrame, queries: pd.DataFrame, config: TfidfConfig) -> dict[str, Any]:
    """Fit and evaluate one TF-IDF configuration."""
    text = build_text(corpus, config.text_variant)
    vectorizer, matrix = fit_vectorizer(text, config)
    market_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for query in queries.itertuples(index=False):
        query_market_rows, query_event_rows = run_query(pd.Series(query._asdict()), corpus, vectorizer, matrix, config)
        market_rows.extend(query_market_rows)
        event_rows.extend(query_event_rows)

    market_results = pd.DataFrame(market_rows)
    event_results = pd.DataFrame(event_rows)
    summary = {
        "config_name": config.name,
        "text_variant": config.text_variant,
        "ngram_range": f"{config.ngram_range[0]}-{config.ngram_range[1]}",
        "min_df": config.min_df,
        "max_features": config.max_features,
        "matrix_shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
        "market_hit_at_1": hit_at_k(market_results, "rank", 1),
        "market_hit_at_5": hit_at_k(market_results, "rank", 5),
        "market_hit_at_10": hit_at_k(market_results, "rank", 10),
        "market_mrr": mean_reciprocal_rank(market_results, "rank"),
        "event_precision_at_1": precision_at_k(event_results, "event_rank", 1),
        "event_precision_at_5": precision_at_k(event_results, "event_rank", 5),
        "event_precision_at_10": precision_at_k(event_results, "event_rank", 10),
        "event_hit_at_1": hit_at_k(event_results, "event_rank", 1),
        "event_hit_at_5": hit_at_k(event_results, "event_rank", 5),
        "event_hit_at_10": hit_at_k(event_results, "event_rank", 10),
        "event_mrr": mean_reciprocal_rank(event_results, "event_rank"),
        "top1_category_match_percent": category_hit_at_k(market_results, 1),
        "top10_any_category_match_percent": category_hit_at_k(market_results, 10),
        "market_results": market_results,
        "event_results": event_results,
    }
    return summary


def parse_event_ids(value: Any) -> set[str]:
    """Parse semicolon-separated event IDs."""
    if pd.isna(value):
        return set()
    return {item.strip() for item in str(value).split(";") if item.strip()}


def precision_at_k(results: pd.DataFrame, rank_column: str, k: int) -> float:
    """Return mean precision at k."""
    scores = []
    for _, frame in results[results[rank_column].le(k)].groupby("query_id"):
        denominator = min(k, len(frame))
        if denominator:
            scores.append(float(frame["event_relevant"].sum()) / denominator)
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


def hit_at_k(results: pd.DataFrame, rank_column: str, k: int) -> float:
    """Return percentage of queries with at least one relevant event in top k."""
    hits = results[results[rank_column].le(k)].groupby("query_id")["event_relevant"].any()
    return pct(hits.sum(), len(hits))


def mean_reciprocal_rank(results: pd.DataFrame, rank_column: str) -> float:
    """Return mean reciprocal rank for the first relevant result."""
    reciprocal_ranks = []
    for _, frame in results.groupby("query_id"):
        relevant = frame[frame["event_relevant"]].sort_values(rank_column)
        reciprocal_ranks.append(0.0 if relevant.empty else 1.0 / float(relevant.iloc[0][rank_column]))
    if not reciprocal_ranks:
        return 0.0
    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)


def category_hit_at_k(results: pd.DataFrame, k: int) -> float:
    """Return percentage of queries with expected category represented in top k."""
    hits = results[results["rank"].le(k)].groupby("query_id")["category_match"].any()
    return pct(hits.sum(), len(hits))


def best_config(tuning_rows: list[dict[str, Any]]) -> str:
    """Choose the strongest config with a simple, transparent ranking rule."""
    frame = pd.DataFrame(tuning_rows)
    frame = frame.sort_values(
        ["event_mrr", "event_hit_at_1", "event_hit_at_5", "event_hit_at_10"],
        ascending=[False, False, False, False],
    )
    return str(frame.iloc[0]["config_name"])


def build_summary(best: dict[str, Any], corpus: pd.DataFrame, queries: pd.DataFrame) -> list[dict[str, Any]]:
    """Build compact summary rows for the selected TF-IDF configuration."""
    rows = [
        row("setup", "documents", len(corpus)),
        row("setup", "queries", len(queries)),
        row("setup", "top_k", TOP_K),
        row("setup", "candidate_k", CANDIDATE_K),
        row("setup", "best_config", best["config_name"]),
        row("setup", "text_variant", best["text_variant"]),
        row("setup", "ngram_range", best["ngram_range"]),
        row("setup", "min_df", best["min_df"]),
        row("setup", "max_features", best["max_features"]),
        row("setup", "matrix_shape", best["matrix_shape"]),
        row("event_relevance", "precision_at_1", best["event_precision_at_1"]),
        row("event_relevance", "precision_at_5", best["event_precision_at_5"]),
        row("event_relevance", "precision_at_10", best["event_precision_at_10"]),
        row("event_relevance", "hit_at_1", best["event_hit_at_1"]),
        row("event_relevance", "hit_at_5", best["event_hit_at_5"]),
        row("event_relevance", "hit_at_10", best["event_hit_at_10"]),
        row("event_relevance", "mrr", best["event_mrr"]),
        row("market_row_diagnostic", "hit_at_1", best["market_hit_at_1"]),
        row("market_row_diagnostic", "hit_at_5", best["market_hit_at_5"]),
        row("market_row_diagnostic", "hit_at_10", best["market_hit_at_10"]),
        row("market_row_diagnostic", "mrr", best["market_mrr"]),
        row("category_check", "top1_category_match_percent", best["top1_category_match_percent"]),
        row("category_check", "top10_any_category_match_percent", best["top10_any_category_match_percent"]),
    ]

    event_results = best["event_results"]
    for query_type, frame in event_results.groupby("query_type", sort=True):
        rows.append(row("query_type", f"{query_type}_hit_at_10", hit_at_k(frame, "event_rank", 10)))
        rows.append(row("query_type", f"{query_type}_mrr", mean_reciprocal_rank(frame, "event_rank")))
    return rows


def row(section: str, metric: str, value: Any, notes: str = "") -> dict[str, Any]:
    """Return one summary row."""
    return {"section": section, "metric": metric, "value": value, "notes": notes}


def pct(part: Any, whole: Any) -> float:
    """Return a rounded percentage."""
    denominator = float(whole or 0)
    if denominator == 0:
        return 0.0
    return round(float(part or 0) / denominator * 100, 2)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tuning(rows: list[dict[str, Any]]) -> None:
    """Write compact tuning results without embedded result dataframes."""
    fields = [
        "config_name",
        "text_variant",
        "ngram_range",
        "min_df",
        "max_features",
        "matrix_shape",
        "event_precision_at_1",
        "event_precision_at_5",
        "event_precision_at_10",
        "event_hit_at_1",
        "event_hit_at_5",
        "event_hit_at_10",
        "event_mrr",
        "market_hit_at_1",
        "market_hit_at_5",
        "market_hit_at_10",
        "market_mrr",
        "top1_category_match_percent",
        "top10_any_category_match_percent",
    ]
    TUNING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TUNING_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow({field: item[field] for field in fields})


def main() -> None:
    """Run TF-IDF tuning and save the selected baseline outputs."""
    require_inputs()
    corpus = load_corpus()
    queries = load_queries()
    print(f"Loaded {len(corpus):,} documents and {len(queries):,} queries", flush=True)

    evaluated: list[dict[str, Any]] = []
    for config in CONFIGS:
        print(f"Evaluating {config.name}", flush=True)
        evaluated.append(evaluate_config(corpus, queries, config))

    selected_name = best_config(evaluated)
    selected = next(item for item in evaluated if item["config_name"] == selected_name)
    selected["market_results"].to_csv(RESULTS_PATH, index=False)
    selected["event_results"].to_csv(EVENT_RESULTS_PATH, index=False)
    write_tuning(evaluated)
    write_rows(SUMMARY_PATH, build_summary(selected, corpus, queries))
    print(f"Selected TF-IDF config: {selected_name}", flush=True)
    print(f"Wrote market results to {RESULTS_PATH}", flush=True)
    print(f"Wrote event results to {EVENT_RESULTS_PATH}", flush=True)
    print(f"Wrote tuning summary to {TUNING_PATH}", flush=True)
    print(f"Wrote baseline summary to {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
