"""Shared HTTP helpers for Polymarket API access.

This module keeps low-level request behaviour in one place so that Gamma and
CLOB API modules can focus on endpoint-specific logic. It provides a reusable
requests session with retry behaviour and a small JSON helper for GET requests.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_BACKOFF_FACTOR = 0.5
DEFAULT_TOTAL_RETRIES = 3
DEFAULT_STATUS_FORCE_LIST = (429, 500, 502, 503, 504)


class PolymarketAPIError(RuntimeError):
    """Raised when a Polymarket API response cannot be retrieved or parsed."""


def create_session(
    *,
    total_retries: int = DEFAULT_TOTAL_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
) -> requests.Session:
    """Create a requests session with retry behaviour for transient failures."""

    retry_strategy = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=DEFAULT_STATUS_FORCE_LIST,
        allowed_methods={"GET"},
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "prediction-market-analytics-dashboard/1.0",
        }
    )

    return session


def build_url(base_url: str, endpoint: str) -> str:
    """Join an API base URL and endpoint safely."""

    return urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/"))


def get_json(
    base_url: str,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    session: requests.Session | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    """Perform a GET request and return the decoded JSON payload."""

    active_session = session or create_session()
    url = build_url(base_url, endpoint)

    try:
        response = active_session.get(
            url,
            params=params,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise PolymarketAPIError(f"GET request failed for {url}: {exc}") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise PolymarketAPIError(f"Invalid JSON response from {url}: {exc}") from exc
