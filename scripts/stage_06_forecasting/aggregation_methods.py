"""Transparent aggregation methods for retrieved prediction-market evidence."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


EPSILON = 1e-15


def clip_probability(value: float) -> float:
    """Return a finite probability in the closed interval [0, 1]."""
    if not math.isfinite(float(value)):
        return math.nan
    return min(1.0, max(0.0, float(value)))


def brier_score(probability: float, actual: int) -> float:
    """Return squared probability error for one binary forecast."""
    p = clip_probability(probability)
    return (p - int(actual)) ** 2


def log_loss(probability: float, actual: int) -> float:
    """Return clipped binary log loss for one forecast."""
    p = min(1.0 - EPSILON, max(EPSILON, clip_probability(probability)))
    y = int(actual)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def evidence_warnings(frame: pd.DataFrame) -> list[str]:
    """Return simple warning labels for an evidence set."""
    warnings: list[str] = []
    if frame.empty:
        return ["no_usable_evidence"]
    if len(frame) < 5:
        warnings.append("thin_evidence_set")
    if frame["event_id"].nunique() < 3:
        warnings.append("low_event_diversity")
    if pd.to_numeric(frame.get("market_volume", pd.Series(dtype=float)), errors="coerce").fillna(0).median() < 1_000:
        warnings.append("low_median_volume")
    if pd.to_numeric(frame.get("probability", pd.Series(dtype=float)), errors="coerce").notna().sum() < len(frame):
        warnings.append("missing_probabilities")
    return warnings


def _weighted_mean(frame: pd.DataFrame, weight_column: str) -> tuple[float, float]:
    probabilities = pd.to_numeric(frame["probability"], errors="coerce")
    weights = pd.to_numeric(frame[weight_column], errors="coerce").fillna(0).clip(lower=0)
    usable = probabilities.notna() & weights.gt(0)
    if not usable.any():
        return float(probabilities.dropna().mean()), 0.0
    return float((probabilities[usable] * weights[usable]).sum() / weights[usable].sum()), float(weights[usable].sum())


def aggregate_forecasts(evidence: pd.DataFrame, target_category: str | None = None) -> pd.DataFrame:
    """Run all aggregation methods over an evidence set.

    Expected columns are: market_id, event_id, probability, similarity_score,
    market_volume, market_liquidity, and is_direct_target.
    """
    if evidence.empty:
        return pd.DataFrame(columns=["method", "forecast_probability", "markets_used", "events_used", "weight_sum", "warnings"])

    frame = evidence.copy()
    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce").map(clip_probability)
    frame["similarity_score"] = pd.to_numeric(frame.get("similarity_score", 0), errors="coerce").fillna(0).clip(lower=0)
    frame["market_volume"] = pd.to_numeric(frame.get("market_volume", 0), errors="coerce").fillna(0).clip(lower=0)
    frame["market_liquidity"] = pd.to_numeric(frame.get("market_liquidity", 0), errors="coerce").fillna(0).clip(lower=0)
    frame["liquidity_or_volume"] = frame["market_liquidity"].where(frame["market_liquidity"].gt(0), frame["market_volume"])
    frame["similarity_volume_weight"] = frame["similarity_score"] * (1 + frame["liquidity_or_volume"].map(math.log1p))
    frame["clean_category"] = frame.get("clean_category", "")
    frame = frame[frame["probability"].notna()].copy()

    warnings = "|".join(evidence_warnings(frame))
    rows: list[dict[str, Any]] = []

    def add(method: str, probability: float, markets: pd.DataFrame, weight_sum: float = 0.0) -> None:
        if math.isfinite(float(probability)):
            rows.append(
                {
                    "method": method,
                    "forecast_probability": round(clip_probability(probability), 6),
                    "markets_used": int(len(markets)),
                    "events_used": int(markets["event_id"].nunique()) if "event_id" in markets else 0,
                    "weight_sum": round(float(weight_sum), 6),
                    "warnings": warnings,
                }
            )

    direct = frame[frame.get("is_direct_target", False).astype(bool)]
    if not direct.empty:
        add("direct_market", float(direct.iloc[0]["probability"]), direct.head(1), 1.0)

    ranked = frame.sort_values(["similarity_score", "market_volume"], ascending=[False, False])
    add("top_similarity", float(ranked.iloc[0]["probability"]), ranked.head(1), float(ranked.iloc[0]["similarity_score"]))
    add("unweighted_mean", float(frame["probability"].mean()), frame, float(len(frame)))
    add("median", float(frame["probability"].median()), frame, float(len(frame)))

    probability, weight_sum = _weighted_mean(frame, "similarity_score")
    add("similarity_weighted_mean", probability, frame, weight_sum)

    probability, weight_sum = _weighted_mean(frame, "liquidity_or_volume")
    add("liquidity_weighted_mean", probability, frame, weight_sum)

    probability, weight_sum = _weighted_mean(frame, "similarity_volume_weight")
    add("similarity_volume_weighted", probability, frame, weight_sum)

    top_five = ranked.head(5)
    probability, weight_sum = _weighted_mean(top_five, "similarity_score")
    add("top5_similarity_weighted_mean", probability, top_five, weight_sum)

    liquid = frame[frame["market_volume"].ge(1_000)].copy()
    if not liquid.empty:
        probability, weight_sum = _weighted_mean(liquid, "similarity_volume_weight")
        add("min_volume_similarity_volume_weighted", probability, liquid, weight_sum)

    if target_category:
        same_category = frame[frame["clean_category"].astype(str).eq(str(target_category))].copy()
        if not same_category.empty:
            probability, weight_sum = _weighted_mean(same_category, "similarity_score")
            add("same_category_similarity_weighted", probability, same_category, weight_sum)
            same_category_top_five = same_category.sort_values(
                ["similarity_score", "market_volume"], ascending=[False, False]
            ).head(5)
            probability, weight_sum = _weighted_mean(same_category_top_five, "similarity_score")
            add("same_category_top5_similarity_weighted", probability, same_category_top_five, weight_sum)

    event_rows = []
    for event_id, event_frame in frame.groupby("event_id", sort=False):
        event_probability, event_weight = _weighted_mean(event_frame, "similarity_score")
        if math.isfinite(float(event_probability)):
            event_rows.append(
                {
                    "event_id": event_id,
                    "probability": event_probability,
                    "similarity_score": max(event_weight, float(event_frame["similarity_score"].max())),
                }
            )
    if event_rows:
        event_frame = pd.DataFrame(event_rows)
        probability, weight_sum = _weighted_mean(event_frame, "similarity_score")
        add("event_grouped_similarity_mean", probability, event_frame, weight_sum)

    return pd.DataFrame(rows)
