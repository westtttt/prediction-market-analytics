"""Command-line bridge for the final website forecast API."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from query_forecast_service import build_query_forecast


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the final website response focused on the first forecast page."""
    target = payload.get("target", {})
    forecast = payload.get("forecast", {})
    return {
        "retrieval": payload.get("retrieval", {}),
        "target": {
            "summary": target.get("summary", {}),
            "events": [
                {
                    "eventId": event.get("eventId"),
                    "rank": event.get("rank"),
                    "score": event.get("score"),
                    "title": event.get("title"),
                    "category": event.get("category"),
                    "markets": event.get("markets"),
                    "volume": event.get("volume"),
                    "priceHistoryCoveragePercent": event.get("priceHistoryCoveragePercent"),
                }
                for event in target.get("events", [])[:5]
            ],
            "warnings": target.get("warnings", [])[:4],
            "interpretation": target.get("interpretation", ""),
        },
        "forecast": {
            "summary": forecast.get("summary", {}),
            "aggregationMethods": forecast.get("aggregationMethods", []),
            "evidenceMarkets": forecast.get("evidenceMarkets", [])[:12],
        },
    }


def main() -> int:
    """Run the tested query forecast service and print JSON for Node."""
    try:
        payload: dict[str, Any] = json.load(sys.stdin)
        query = str(payload.get("query", "")).strip()
        print(json.dumps(compact_payload(build_query_forecast(query))), flush=True)
        return 0
    except ValueError as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr, flush=True)
        return 2
    except Exception as error:  # pragma: no cover - defensive bridge path
        print(json.dumps({"error": f"Forecast service failed: {error}"}), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
