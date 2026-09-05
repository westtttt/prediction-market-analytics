

"""Functions for interacting with the Polymarket CLOB API.

The CLOB (Central Limit Order Book) API exposes historical trading data for
individual market outcomes. This module provides endpoint-specific functions
for retrieving historical price trajectories. Shared HTTP session, retry,
timeout, and JSON handling are delegated to http_session.py.
"""

from __future__ import annotations

from typing import Any

import requests

from .http_session import PolymarketAPIError, get_json

CLOB_BASE_URL = "https://clob.polymarket.com"
PRICE_HISTORY_ENDPOINT = "/prices-history"


def fetch_clob_price_history(
    *,
    clob_token_id: str,
    interval: str = "max",
    fidelity: int = 1440,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch historical prices for a single CLOB outcome token."""

    response = get_json(
        CLOB_BASE_URL,
        PRICE_HISTORY_ENDPOINT,
        params={
            "market": clob_token_id,
            "interval": interval,
            "fidelity": fidelity,
        },
        session=session,
    )

    if not isinstance(response, dict):
        raise PolymarketAPIError(
            "Expected CLOB price-history response to be a dictionary."
        )

    if "history" not in response:
        raise PolymarketAPIError(
            "CLOB price-history response did not contain a 'history' field."
        )

    history = response["history"]
    if not isinstance(history, list):
        raise PolymarketAPIError(
            "CLOB price-history 'history' field was not a list."
        )

    return response