"""Analyse whether event-aware or contrastive retrieval is justified."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "data" / "outputs" / "analysis" / "retrieval_corpus.parquet"
QUERY_PATH = ROOT / "config" / "retrieval_eval_queries.csv"
SUMMARY_PATH = ROOT / "data" / "outputs" / "analysis" / "retrieval_readiness_summary.csv"
DIAGNOSTICS_PATH = ROOT / "data" / "outputs" / "analysis" / "retrieval_candidate_diagnostics.csv"

CANDIDATE_K = 100
TOP_K = 10

CORPUS_COLUMNS = [
    "market_id",
    "event_id",
    "event_title",
    "market_question",
    "clean_category",
    "primary_tag",
    "market_volume",
]


def require_inputs() -> None:
    """Fail clearly if required files are missing."""
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
    """Build the selected compact event-heavy TF-IDF text."""
    event = corpus["event_title"].fillna("").astype("string")
    question = corpus["market_question"].fillna("").astype("string")
    category = corpus["clean_category"].fillna("").astype("string")
    tag = corpus["primary_tag"].fillna("").astype("string")
    return event + " " + event + " " + question + " " + category + " " + tag


def parse_event_ids(value: Any) -> set[str]:
    """Parse semicolon-separated event IDs."""
    if pd.isna(value):
        return set()
    return {item.strip() for item in str(value).split(";") if item.strip()}


def top_candidate_indexes(scores: np.ndarray, candidate_k: int) -> np.ndarray:
    """Return candidate indexes sorted by descending score."""
    limit = min(candidate_k, len(scores))
    indexes = np.argpartition(scores, -limit)[-limit:]
    return indexes[np.argsort(scores[indexes])[::-1]]


def event_size_summary(corpus: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarise event sizes and possible weak-supervision pairs."""
    sizes = corpus.groupby("event_id", sort=False).size()
    positive_pairs = int(((sizes * (sizes - 1)) // 2).sum())
    rows = [
        row("event_structure", "events", int(len(sizes))),
        row("event_structure", "markets", int(len(corpus))),
        row("event_structure", "median_markets_per_event", float(sizes.median())),
        row("event_structure", "mean_markets_per_event", round(float(sizes.mean()), 4)),
        row("event_structure", "p90_markets_per_event", float(sizes.quantile(0.90))),
        row("event_structure", "p95_markets_per_event", float(sizes.quantile(0.95))),
        row("event_structure", "p99_markets_per_event", float(sizes.quantile(0.99))),
        row("event_structure", "max_markets_per_event", int(sizes.max())),
        row("event_structure", "single_market_events_percent", pct((sizes == 1).sum(), len(sizes))),
        row("event_structure", "multi_market_events_percent", pct((sizes >= 2).sum(), len(sizes))),
        row("event_structure", "events_with_5_plus_markets_percent", pct((sizes >= 5).sum(), len(sizes))),
        row("event_structure", "events_with_10_plus_markets_percent", pct((sizes >= 10).sum(), len(sizes))),
        row("contrastive_pairs", "positive_market_pairs_same_event", positive_pairs),
        row("contrastive_pairs", "events_with_positive_pairs", int((sizes >= 2).sum())),
    ]
    return rows


def labelled_event_summary(corpus: pd.DataFrame, queries: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarise labelled relevant events in the corpus."""
    event_sizes = corpus.groupby("event_id", sort=False).size()
    labelled_ids = sorted({event_id for ids in queries["relevant_event_id_set"] for event_id in ids})
    labelled_sizes = event_sizes.reindex(labelled_ids).dropna()
    return [
        row("labelled_events", "labelled_event_ids", len(labelled_ids)),
        row("labelled_events", "labelled_events_found_in_corpus", int(labelled_sizes.size)),
        row("labelled_events", "median_markets_per_labelled_event", float(labelled_sizes.median())),
        row("labelled_events", "mean_markets_per_labelled_event", round(float(labelled_sizes.mean()), 4)),
        row("labelled_events", "labelled_events_with_2_plus_markets_percent", pct((labelled_sizes >= 2).sum(), len(labelled_sizes))),
        row("labelled_events", "labelled_events_with_5_plus_markets_percent", pct((labelled_sizes >= 5).sum(), len(labelled_sizes))),
    ]


def candidate_diagnostics(corpus: pd.DataFrame, queries: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Check whether TF-IDF candidates contain labelled events for each query."""
    text = build_text(corpus)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=80_000,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(text.fillna(""))

    rows: list[dict[str, Any]] = []
    for query in queries.itertuples(index=False):
        query_series = pd.Series(query._asdict())
        relevant_ids = query_series["relevant_event_id_set"]
        query_vector = vectorizer.transform([str(query_series["query"])])
        scores = (query_vector @ matrix.T).toarray().ravel()
        indexes = top_candidate_indexes(scores, CANDIDATE_K)
        candidates = corpus.iloc[indexes].copy()
        candidates["score"] = scores[indexes]
        candidates["candidate_rank"] = range(1, len(candidates) + 1)

        first_event_rows = candidates.drop_duplicates("event_id", keep="first").reset_index(drop=True)
        first_event_rows["event_rank"] = first_event_rows.index + 1
        relevant_candidates = candidates[candidates["event_id"].isin(relevant_ids)]
        relevant_events = first_event_rows[first_event_rows["event_id"].isin(relevant_ids)]
        top_market_rows = candidates.head(TOP_K)
        top_event_rows = first_event_rows.head(TOP_K)

        top_event = first_event_rows.iloc[0]
        rows.append(
            {
                "query_id": query_series["query_id"],
                "query": query_series["query"],
                "query_type": query_series["query_type"],
                "relevant_event_ids": query_series["relevant_event_ids"],
                "relevant_in_top_100_market_candidates": not relevant_candidates.empty,
                "relevant_in_top_10_unique_events": not relevant_events[relevant_events["event_rank"] <= TOP_K].empty,
                "first_relevant_market_rank": blank_if_missing(relevant_candidates["candidate_rank"].min()),
                "first_relevant_event_rank": blank_if_missing(relevant_events["event_rank"].min()),
                "top_10_market_unique_events": int(top_market_rows["event_id"].nunique()),
                "top_10_event_repeated_market_rows": int(TOP_K - top_market_rows["event_id"].nunique()),
                "top_event_id": top_event["event_id"],
                "top_event_title": top_event["event_title"],
                "top_event_market_count_in_candidates": int((candidates["event_id"] == top_event["event_id"]).sum()),
                "top_event_score": round(float(top_event["score"]), 6),
                "diagnosis": diagnose(relevant_candidates, relevant_events),
            }
        )

    diagnostics = pd.DataFrame(rows)
    summary = [
        row("candidate_coverage", "queries", len(diagnostics)),
        row(
            "candidate_coverage",
            "relevant_in_top_100_market_candidates_percent",
            pct(diagnostics["relevant_in_top_100_market_candidates"].sum(), len(diagnostics)),
        ),
        row(
            "candidate_coverage",
            "relevant_in_top_10_unique_events_percent",
            pct(diagnostics["relevant_in_top_10_unique_events"].sum(), len(diagnostics)),
        ),
        row(
            "candidate_coverage",
            "median_first_relevant_event_rank",
            float(pd.to_numeric(diagnostics["first_relevant_event_rank"], errors="coerce").median()),
        ),
        row(
            "candidate_coverage",
            "mean_top_10_market_unique_events",
            round(float(diagnostics["top_10_market_unique_events"].mean()), 4),
        ),
        row(
            "candidate_coverage",
            "queries_with_repeated_top_10_market_events",
            int((diagnostics["top_10_event_repeated_market_rows"] > 0).sum()),
        ),
    ]
    for diagnosis, count in diagnostics["diagnosis"].value_counts().sort_index().items():
        summary.append(row("diagnosis", str(diagnosis), int(count)))
    return diagnostics, summary


def diagnose(relevant_candidates: pd.DataFrame, relevant_events: pd.DataFrame) -> str:
    """Classify whether a query looks rerankable or representation-limited."""
    if relevant_events.empty:
        return "representation_or_label_scope_issue"
    first_event_rank = int(relevant_events["event_rank"].min())
    if not relevant_candidates.empty:
        if first_event_rank == 1:
            return "already_top_1"
        if first_event_rank <= TOP_K:
            return "top_1_reranking_candidate"
        return "top_10_reranking_candidate"
    return "representation_or_label_scope_issue"


def blank_if_missing(value: Any) -> Any:
    """Return a blank cell for missing numeric diagnostics."""
    if pd.isna(value):
        return ""
    return int(value)


def row(section: str, metric: str, value: Any, notes: str = "") -> dict[str, Any]:
    """Return one summary row."""
    return {"section": section, "metric": metric, "value": value, "notes": notes}


def pct(part: Any, whole: Any) -> float:
    """Return a rounded percentage."""
    denominator = float(whole or 0)
    if denominator == 0:
        return 0.0
    return round(float(part or 0) / denominator * 100, 2)


def write_outputs(summary: list[dict[str, Any]], diagnostics: pd.DataFrame) -> None:
    """Write compact readiness outputs."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False)


def main() -> None:
    """Run retrieval readiness diagnostics."""
    require_inputs()
    corpus = load_corpus()
    queries = load_queries()
    summary = event_size_summary(corpus)
    summary.extend(labelled_event_summary(corpus, queries))
    diagnostics, candidate_summary = candidate_diagnostics(corpus, queries)
    summary.extend(candidate_summary)
    write_outputs(summary, diagnostics)
    print(f"Wrote readiness summary to {SUMMARY_PATH}")
    print(f"Wrote candidate diagnostics to {DIAGNOSTICS_PATH}")


if __name__ == "__main__":
    main()
