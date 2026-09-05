"""Build a compact market-level retrieval corpus."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MARKETS_PATH = ROOT / "data" / "outputs" / "analysis" / "markets_analysis.parquet"
CATEGORIES_PATH = ROOT / "data" / "outputs" / "analysis" / "market_categories.parquet"
OUTPUT_PATH = ROOT / "data" / "outputs" / "analysis" / "retrieval_corpus.parquet"
SUMMARY_PATH = ROOT / "data" / "outputs" / "analysis" / "retrieval_corpus_summary.csv"
SAMPLE_PATH = ROOT / "data" / "outputs" / "analysis" / "retrieval_corpus_sample.csv"

TEXT_COLUMNS = [
    "event_title",
    "market_question",
    "event_description",
    "market_description",
    "tag_labels",
    "primary_tag",
]

MARKET_COLUMNS = [
    "market_id",
    "event_id",
    "event_title",
    "market_question",
    "event_description",
    "market_description",
    "tag_labels",
    "primary_tag",
    "market_volume",
    "has_price_history",
    "price_points",
    "is_resolved",
    "outcome_count",
]

DESCRIPTION_LIMIT = 500
SAMPLE_ROWS = 1_000


def require_inputs() -> None:
    """Fail clearly if required generated datasets are missing."""
    missing = [path for path in [MARKETS_PATH, CATEGORIES_PATH] if not path.exists()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing input dataset(s): {paths}")


def clean_text(value: Any, limit: int | None = None) -> str:
    """Normalize whitespace and optionally truncate long text."""
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if limit is not None and len(text) > limit:
        return text[:limit].rsplit(" ", 1)[0]
    return text


def build_text(row: pd.Series) -> str:
    """Build the weighted retrieval text for one market."""
    parts = [
        clean_text(row["market_question"]),
        clean_text(row["market_question"]),
        clean_text(row["event_title"]),
        clean_text(row["clean_category"]),
        clean_text(row["primary_tag"]),
        clean_text(row["tag_labels"]),
        clean_text(row["market_description"], DESCRIPTION_LIMIT),
        clean_text(row["event_description"], DESCRIPTION_LIMIT),
    ]
    return " ".join(part for part in parts if part)


def metric_row(section: str, metric: str, value: Any, notes: str = "") -> dict[str, Any]:
    """Return one standard summary row."""
    return {"section": section, "metric": metric, "value": value, "notes": notes}


def load_data() -> pd.DataFrame:
    """Load market text, metadata, and cleaned categories."""
    require_inputs()
    markets = pd.read_parquet(MARKETS_PATH, columns=MARKET_COLUMNS)
    categories = pd.read_parquet(CATEGORIES_PATH, columns=["market_id", "clean_category"])
    frame = markets.merge(categories, on="market_id", how="left", validate="one_to_one")
    frame["clean_category"] = frame["clean_category"].fillna("Other")
    return frame


def build_corpus(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the retrieval corpus dataframe."""
    corpus = frame.copy()
    corpus["retrieval_text"] = corpus.apply(build_text, axis=1)
    corpus["text_length"] = corpus["retrieval_text"].str.len()
    corpus["has_tags"] = corpus["tag_labels"].notna() & corpus["tag_labels"].astype("string").str.strip().ne("")
    corpus["has_description"] = (
        corpus["market_description"].notna() & corpus["market_description"].astype("string").str.strip().ne("")
    )
    keep_columns = [
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
        "outcome_count",
        "has_tags",
        "has_description",
        "text_length",
        "retrieval_text",
    ]
    return corpus.loc[:, keep_columns]


def build_summary(corpus: pd.DataFrame) -> list[dict[str, Any]]:
    """Build compact corpus summary rows."""
    return [
        metric_row("coverage", "documents", len(corpus)),
        metric_row("coverage", "events", int(corpus["event_id"].nunique())),
        metric_row("coverage", "categories", int(corpus["clean_category"].nunique())),
        metric_row("coverage", "has_tags_percent", pct(corpus["has_tags"].sum(), len(corpus))),
        metric_row("coverage", "has_description_percent", pct(corpus["has_description"].sum(), len(corpus))),
        metric_row("coverage", "has_price_history_percent", pct(corpus["has_price_history"].sum(), len(corpus))),
        metric_row("text", "median_text_length", round(float(corpus["text_length"].median()), 2)),
        metric_row("text", "p95_text_length", round(float(corpus["text_length"].quantile(0.95)), 2)),
        metric_row("text", "max_text_length", int(corpus["text_length"].max())),
    ]


def pct(part: Any, whole: Any) -> float:
    """Return a rounded percentage."""
    denominator = float(whole or 0)
    if denominator == 0:
        return 0.0
    return round(float(part or 0) / denominator * 100, 2)


def write_outputs(corpus: pd.DataFrame, rows: list[dict[str, Any]]) -> None:
    """Write corpus, sample, and summary outputs."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(OUTPUT_PATH, index=False)
    corpus.head(SAMPLE_ROWS).to_csv(SAMPLE_PATH, index=False)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build the retrieval corpus."""
    frame = load_data()
    corpus = build_corpus(frame)
    rows = build_summary(corpus)
    write_outputs(corpus, rows)
    print(f"Wrote retrieval corpus to {OUTPUT_PATH}", flush=True)
    print(f"Wrote retrieval summary to {SUMMARY_PATH}", flush=True)
    print(f"Documents: {len(corpus):,}", flush=True)
    print(f"Events: {corpus['event_id'].nunique():,}", flush=True)


if __name__ == "__main__":
    main()
