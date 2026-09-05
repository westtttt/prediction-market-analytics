"""Build a small, reproducible example table for historical backtesting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "data" / "outputs" / "analysis"
EVALUATION_PATH = ANALYSIS_DIR / "aggregation_backtest_evaluation.parquet"
MARKETS_PATH = ANALYSIS_DIR / "markets_analysis.parquet"
CSV_OUTPUT = ANALYSIS_DIR / "aggregation_backtest_examples.csv"
MD_OUTPUT = ANALYSIS_DIR / "aggregation_backtest_examples.md"

EXAMPLE_TARGETS = [
    "253591",  # 2024 US presidential election winner example.
    "516861",  # Bitcoin price in 2025 example.
    "516729",  # Fed rate cuts in 2025 example.
]

EXAMPLE_METHODS = [
    "direct_market",
    "top_similarity",
    "same_category_top5_similarity_weighted",
    "similarity_volume_weighted",
    "unweighted_mean",
]

EXAMPLE_HORIZONS = [7, 1]


def clean_id(value: object) -> str:
    """Return a stable string id."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def load_target_details(target_ids: list[str]) -> pd.DataFrame:
    """Load readable event and market labels for selected target markets."""
    columns = [
        "market_id",
        "event_id",
        "event_title",
        "market_question",
        "market_volume",
        "outcome_names",
        "winner_outcome_names",
    ]
    markets = pd.read_parquet(MARKETS_PATH, columns=columns)
    markets["market_id"] = markets["market_id"].map(clean_id)
    markets["event_id"] = markets["event_id"].map(clean_id)
    return markets[markets["market_id"].isin(target_ids)].copy()


def build_examples() -> pd.DataFrame:
    """Return reproducible example rows from the saved backtest output."""
    evaluation = pd.read_parquet(EVALUATION_PATH)
    evaluation["target_market_id"] = evaluation["target_market_id"].map(clean_id)
    evaluation["target_event_id"] = evaluation["target_event_id"].map(clean_id)
    examples = evaluation[
        evaluation["target_market_id"].isin(EXAMPLE_TARGETS)
        & evaluation["method"].isin(EXAMPLE_METHODS)
        & evaluation["horizon_days"].isin(EXAMPLE_HORIZONS)
    ].copy()
    details = load_target_details(EXAMPLE_TARGETS)
    examples = examples.merge(
        details,
        left_on="target_market_id",
        right_on="market_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_target"),
    )
    examples["result"] = examples["actual"].map({1: "Yes won", 0: "Yes did not win"})
    examples["forecast_probability"] = examples["forecast_probability"].round(6)
    examples["brier_score"] = examples["brier_score"].round(6)
    examples["log_loss"] = examples["log_loss"].round(6)
    examples["market_volume"] = pd.to_numeric(examples["market_volume"], errors="coerce").round(2)
    output_columns = [
        "query_id",
        "query",
        "event_title",
        "market_question",
        "target_category",
        "target_market_id",
        "target_event_id",
        "horizon_days",
        "as_of_time",
        "method",
        "forecast_probability",
        "result",
        "brier_score",
        "log_loss",
        "markets_used",
        "events_used",
    ]
    return examples[output_columns].sort_values(["query_id", "target_market_id", "horizon_days", "method"]).copy()


def method_definitions() -> pd.DataFrame:
    """Return dissertation-ready definitions for aggregation methods."""
    rows = [
        {
            "method": "direct_market",
            "definition": "Uses the target market's own historical Yes price at the cutoff.",
            "role": "Primary baseline where a direct target market exists.",
        },
        {
            "method": "top_similarity",
            "definition": "Uses the Yes price from the single highest-ranked retrieved evidence market.",
            "role": "Retrieval-based fallback baseline.",
        },
        {
            "method": "unweighted_mean",
            "definition": "Averages all usable retrieved evidence-market probabilities equally.",
            "role": "Naive broad aggregation baseline.",
        },
        {
            "method": "median",
            "definition": "Uses the median probability across all usable retrieved evidence markets.",
            "role": "Robust broad aggregation baseline.",
        },
        {
            "method": "similarity_weighted_mean",
            "definition": "Averages evidence probabilities using retrieval similarity as the weight.",
            "role": "Tests whether stronger retrieval matches should contribute more.",
        },
        {
            "method": "liquidity_weighted_mean",
            "definition": "Averages evidence probabilities using liquidity where available, otherwise volume.",
            "role": "Tests whether activity and market depth improve aggregation.",
        },
        {
            "method": "similarity_volume_weighted",
            "definition": "Weights evidence by retrieval similarity multiplied by a log-scaled activity measure.",
            "role": "Current premium selected method and broad weighted aggregation test.",
        },
        {
            "method": "top5_similarity_weighted_mean",
            "definition": "Applies similarity weighting only to the five highest-ranked evidence markets.",
            "role": "Tests whether restricting to the strongest evidence reduces dilution.",
        },
        {
            "method": "min_volume_similarity_volume_weighted",
            "definition": "Applies similarity-volume weighting after removing evidence markets below a minimum volume threshold.",
            "role": "Tests whether low-volume evidence weakens aggregation.",
        },
        {
            "method": "same_category_similarity_weighted",
            "definition": "Similarity-weights only evidence markets in the same cleaned category as the target.",
            "role": "Tests category filtering as a relevance constraint.",
        },
        {
            "method": "same_category_top5_similarity_weighted",
            "definition": "Similarity-weights the top five same-category evidence markets.",
            "role": "Best-performing filtered aggregation variant in the bounded run.",
        },
        {
            "method": "event_grouped_similarity_mean",
            "definition": "First summarises probabilities within each event, then combines event-level probabilities by similarity.",
            "role": "Tests event-level aggregation rather than market-row aggregation.",
        },
    ]
    return pd.DataFrame(rows)


def write_markdown(examples: pd.DataFrame, definitions: pd.DataFrame) -> None:
    """Write a compact Markdown evidence table for dissertation drafting."""
    def markdown_table(frame: pd.DataFrame) -> str:
        headers = list(frame.columns)
        rows = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in frame.itertuples(index=False):
            values = [str(value).replace("|", "\\|") for value in row]
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join(rows)

    lines = [
        "# Historical Backtest Examples",
        "",
        "These examples are generated from the saved historical aggregation backtest output. They are intended for dissertation drafting and reproducibility checks.",
        "",
        "## Aggregation Method Definitions",
        "",
        markdown_table(definitions),
        "",
        "## Selected Backtest Examples",
        "",
        markdown_table(examples),
        "",
    ]
    MD_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Build example CSV and Markdown outputs."""
    examples = build_examples()
    definitions = method_definitions()
    examples.to_csv(CSV_OUTPUT, index=False)
    write_markdown(examples, definitions)
    print(f"Wrote {len(examples)} example rows to {CSV_OUTPUT}", flush=True)
    print(f"Wrote example markdown to {MD_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
