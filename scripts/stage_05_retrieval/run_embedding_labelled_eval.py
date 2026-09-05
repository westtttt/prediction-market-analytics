"""Evaluate embedding retrievers on the labelled business-query set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "data" / "outputs" / "analysis" / "retrieval_corpus.parquet"
QUERY_PATH = ROOT / "config" / "retrieval_eval_queries.csv"
MODEL_DIR = ROOT / "data" / "outputs" / "models" / "contrastive_retriever" / "finetuned_small_eval"
OUTPUT_DIR = ROOT / "data" / "outputs" / "analysis"

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42
MAX_CANDIDATES = 100_000
ENCODE_BATCH_SIZE = 128
TOP_KS = [1, 5, 10, 20, 50]
RESULT_TOP_K = max(TOP_KS)

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


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(MODEL_DIR))
    parser.add_argument("--base-model-name", default=BASE_MODEL_NAME)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--encode-batch-size", type=int, default=ENCODE_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def require_inputs(args: argparse.Namespace) -> None:
    """Fail clearly if required inputs are missing."""
    missing = [path for path in [CORPUS_PATH, QUERY_PATH] if not path.exists()]
    if not Path(args.model_path).exists():
        missing.append(Path(args.model_path))
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing input file(s): {paths}")


def load_queries() -> pd.DataFrame:
    """Load labelled retrieval queries."""
    queries = pd.read_csv(QUERY_PATH)
    queries["relevant_event_id_set"] = queries["relevant_event_ids"].map(parse_event_ids)
    return queries


def load_candidates(max_candidates: int, seed: int) -> pd.DataFrame:
    """Load a candidate market pool that includes every labelled relevant event."""
    corpus = pd.read_parquet(CORPUS_PATH, columns=CORPUS_COLUMNS)
    corpus["event_id"] = corpus["event_id"].astype("string")
    corpus["market_id"] = corpus["market_id"].astype("string")
    corpus["market_volume"] = pd.to_numeric(corpus["market_volume"], errors="coerce").fillna(0)
    corpus["candidate_text"] = build_candidate_text(corpus)

    relevant_ids = {event_id for ids in load_queries()["relevant_event_id_set"] for event_id in ids}
    required = corpus[corpus["event_id"].astype(str).isin(relevant_ids)].copy()
    optional = corpus[~corpus["event_id"].astype(str).isin(relevant_ids)].copy()
    optional = optional.sort_values("market_volume", ascending=False)
    remaining = max(max_candidates - len(required), 0)
    if len(optional) > remaining:
        high_volume = optional.head(max(remaining // 2, 0))
        rest_needed = remaining - len(high_volume)
        remainder = optional.iloc[len(high_volume) :]
        if rest_needed > 0 and len(remainder) > rest_needed:
            remainder = remainder.sample(n=rest_needed, random_state=seed)
        optional = pd.concat([high_volume, remainder], ignore_index=True)
    candidates = pd.concat([required, optional], ignore_index=True)
    return candidates.drop_duplicates("market_id").reset_index(drop=True)


def build_candidate_text(corpus: pd.DataFrame) -> pd.Series:
    """Build candidate-facing text for embedding retrieval."""
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


def load_models(args: argparse.Namespace) -> dict[str, SentenceTransformer]:
    """Load pretrained and fine-tuned embedding models."""
    return {
        "embedding_pretrained": SentenceTransformer(args.base_model_name, local_files_only=True),
        "embedding_contrastive_finetuned": SentenceTransformer(args.model_path, local_files_only=True),
    }


def encode_texts(model: SentenceTransformer, texts: pd.Series, batch_size: int) -> np.ndarray:
    """Encode and normalise text embeddings."""
    embeddings = model.encode(
        texts.fillna("").astype(str).tolist(),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.astype(np.float32)


def top_indexes(scores: np.ndarray, k: int) -> np.ndarray:
    """Return top-k indexes sorted by descending score."""
    limit = min(k, len(scores))
    indexes = np.argpartition(scores, -limit)[-limit:]
    return indexes[np.argsort(scores[indexes])[::-1]]


def evaluate_model(
    method: str,
    model: SentenceTransformer,
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    """Evaluate one embedding model on the labelled query set."""
    query_embeddings = encode_texts(model, queries["query"], batch_size)
    candidate_embeddings = encode_texts(model, candidates["candidate_text"], batch_size)
    rows = []

    for query_index, query in enumerate(queries.itertuples(index=False)):
        query_series = pd.Series(query._asdict())
        relevant_ids = query_series["relevant_event_id_set"]
        scores = candidate_embeddings @ query_embeddings[query_index]
        indexes = top_indexes(scores, min(500, len(scores)))
        ranked = candidates.iloc[indexes].copy()
        ranked["score"] = scores[indexes]
        event_ranked = ranked.drop_duplicates("event_id", keep="first").head(RESULT_TOP_K).reset_index(drop=True)

        for event_rank, item in enumerate(event_ranked.itertuples(index=False), start=1):
            rows.append(
                {
                    "method": method,
                    "query_id": query_series["query_id"],
                    "query": query_series["query"],
                    "query_type": query_series["query_type"],
                    "expected_category": query_series["expected_category"],
                    "relevant_event_ids": query_series["relevant_event_ids"],
                    "event_rank": event_rank,
                    "score": round(float(item.score), 6),
                    "event_id": item.event_id,
                    "event_title": item.event_title,
                    "representative_market_id": item.market_id,
                    "representative_market_question": item.market_question,
                    "clean_category": item.clean_category,
                    "primary_tag": item.primary_tag,
                    "market_volume": item.market_volume,
                    "has_price_history": item.has_price_history,
                    "price_points": item.price_points,
                    "is_resolved": item.is_resolved,
                    "event_relevant": str(item.event_id) in relevant_ids,
                    "category_match": item.clean_category == query_series["expected_category"],
                }
            )

    return pd.DataFrame(rows)


def precision_at_k(results: pd.DataFrame, k: int) -> float:
    """Return mean event precision at k."""
    scores = []
    for _, frame in results[results["event_rank"].le(k)].groupby("query_id"):
        denominator = min(k, len(frame))
        if denominator:
            scores.append(float(frame["event_relevant"].sum()) / denominator)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def hit_at_k(results: pd.DataFrame, queries: pd.DataFrame, k: int) -> float:
    """Return percentage of labelled queries with a relevant event in top k."""
    hits = results[results["event_rank"].le(k)].groupby("query_id")["event_relevant"].any()
    return pct(hits.sum(), len(queries))


def mean_reciprocal_rank(results: pd.DataFrame, queries: pd.DataFrame) -> float:
    """Return mean reciprocal rank for the first relevant event."""
    reciprocal_ranks = []
    for query_id in queries["query_id"]:
        frame = results[results["query_id"].eq(query_id)]
        relevant = frame[frame["event_relevant"]].sort_values("event_rank")
        reciprocal_ranks.append(0.0 if relevant.empty else 1.0 / float(relevant.iloc[0]["event_rank"]))
    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else 0.0


def category_hit_at_k(results: pd.DataFrame, k: int) -> float:
    """Return percentage of queries with expected category represented in top k."""
    hits = results[results["event_rank"].le(k)].groupby("query_id")["category_match"].any()
    return pct(hits.sum(), len(hits))


def pct(part: Any, whole: Any) -> float:
    """Return a rounded percentage."""
    denominator = float(whole or 0)
    if denominator == 0:
        return 0.0
    return round(float(part or 0) / denominator * 100, 2)


def build_summary(method: str, results: pd.DataFrame, queries: pd.DataFrame, candidates: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    """Build one compact method summary row."""
    return {
        "method": method,
        "queries": int(len(queries)),
        "candidate_markets": int(len(candidates)),
        "candidate_events": int(candidates["event_id"].nunique()),
        "candidate_note": "labelled relevant events plus high-volume/sample corpus subset",
        "hit_at_1": hit_at_k(results, queries, 1),
        "hit_at_5": hit_at_k(results, queries, 5),
        "hit_at_10": hit_at_k(results, queries, 10),
        "hit_at_20": hit_at_k(results, queries, 20),
        "hit_at_50": hit_at_k(results, queries, 50),
        "mrr": mean_reciprocal_rank(results, queries),
        "precision_at_5": precision_at_k(results, 5),
        "precision_at_10": precision_at_k(results, 10),
        "top1_category_match_percent": category_hit_at_k(results, 1),
        "top10_any_category_match_percent": category_hit_at_k(results, 10),
        "model_path": args.model_path if method == "embedding_contrastive_finetuned" else args.base_model_name,
        "max_candidates": args.max_candidates,
    }


def write_outputs(results: pd.DataFrame, summary_rows: list[dict[str, Any]]) -> None:
    """Write detailed and summary outputs."""
    results_path = OUTPUT_DIR / "embedding_labelled_eval_results.csv"
    summary_path = OUTPUT_DIR / "embedding_labelled_eval_summary.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    """Run labelled-query evaluation for embedding retrievers."""
    args = parse_args()
    require_inputs(args)
    queries = load_queries()
    candidates = load_candidates(args.max_candidates, args.seed)
    print(f"Loaded {len(queries):,} queries and {len(candidates):,} candidate markets", flush=True)

    result_frames = []
    summary_rows = []
    for method, model in load_models(args).items():
        print(f"Evaluating {method}", flush=True)
        results = evaluate_model(method, model, queries, candidates, args.encode_batch_size)
        result_frames.append(results)
        summary_rows.append(build_summary(method, results, queries, candidates, args))

    all_results = pd.concat(result_frames, ignore_index=True)
    write_outputs(all_results, summary_rows)
    print(f"Wrote embedding labelled evaluation to {OUTPUT_DIR / 'embedding_labelled_eval_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
