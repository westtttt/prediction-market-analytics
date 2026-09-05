"""
Reusable Polymarket API helper functions.

This module centralises API access so that collection scripts do not each
implement their own request, retry, timeout, and pagination logic.

Current focus:
- Polymarket Gamma API event collection
- Offset and keyset event pagination
- Raw JSON collection before database loading

Example usage:

    from api.polymarket_gamma_api import fetch_gamma_events_keyset_page

    events, next_cursor, raw_response = fetch_gamma_events_keyset_page(limit=100)
"""

from __future__ import annotations

import time
from typing import Any

import requests


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 2


class PolymarketAPIError(RuntimeError):
    """
    Raised when a Polymarket API request repeatedly fails.
    """


SessionParams = dict[str, str | int | float | bool | None]


def create_session() -> requests.Session:
    """
    Create a requests session for Polymarket API calls.

    A session is used within each script run to reuse HTTP connection handling
    without requiring any persistent service or long-running connection.
    """

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "prediction-market-analytics-dashboard/1.0",
        }
    )
    return session


def clean_params(params: SessionParams | None) -> dict[str, str | int | float | bool]:
    """
    Remove None-valued parameters before sending a request.
    """

    if params is None:
        return {}

    return {key: value for key, value in params.items() if value is not None}


def get_json(
    endpoint: str,
    params: SessionParams | None = None,
    *,
    base_url: str = GAMMA_BASE_URL,
    session: requests.Session | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: int = DEFAULT_BACKOFF_SECONDS,
) -> Any:
    """
    Request JSON from a Polymarket API endpoint.

    Parameters
    ----------
    endpoint:
        API endpoint path, for example ``/events``.
    params:
        Optional query parameters.
    base_url:
        Base API URL. Defaults to the Gamma API.
    session:
        Optional existing requests session. If omitted, a temporary session is
        created for the request.
    timeout_seconds:
        Request timeout.
    max_retries:
        Number of request attempts before failing.
    backoff_seconds:
        Base sleep duration between failed attempts. The wait increases with
        each attempt.

    Returns
    -------
    Any
        Parsed JSON response.

    Raises
    ------
    PolymarketAPIError
        If all retry attempts fail.
    """

    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"

    url = f"{base_url}{endpoint}"
    request_params = clean_params(params)
    active_session = session or create_session()
    close_session = session is None

    last_error: Exception | None = None

    try:
        for attempt in range(1, max_retries + 1):
            try:
                response = active_session.get(
                    url,
                    params=request_params,
                    timeout=timeout_seconds,
                )
                response.raise_for_status()
                return response.json()

            except (requests.RequestException, ValueError) as error:
                last_error = error

                if attempt == max_retries:
                    break

                sleep_time = backoff_seconds * attempt
                time.sleep(sleep_time)

        raise PolymarketAPIError(
            "Polymarket API request failed after "
            f"{max_retries} attempts: {url} params={request_params}"
        ) from last_error

    finally:
        if close_session:
            active_session.close()


def fetch_gamma_events_page(
    *,
    limit: int = 100,
    offset: int = 0,
    active: bool | None = None,
    closed: bool | None = None,
    order: str = "volume",
    ascending: bool = False,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch a single page of events from the Gamma API.

    Parameters mirror the event endpoint options currently useful for the
    dissertation pipeline. Additional filters can be added here if needed.
    """

    params: SessionParams = {
        "limit": limit,
        "offset": offset,
        "active": str(active).lower() if active is not None else None,
        "closed": str(closed).lower() if closed is not None else None,
        "order": order,
        "ascending": str(ascending).lower(),
    }

    data = get_json("/events", params=params, session=session)

    if not isinstance(data, list):
        raise PolymarketAPIError(
            f"Expected /events response to be a list, received {type(data)}"
        )

    return data


def extract_keyset_events(data: Any) -> list[dict[str, Any]]:
    """
    Extract event records from a keyset response.

    The Gamma keyset endpoint may return either a bare list or a dictionary
    containing a list under a key such as ``data`` or ``events``. This helper
    keeps the collector tolerant of small response-shape differences while the
    endpoint is being investigated.
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["data", "events", "results"]:
            value = data.get(key)

            if isinstance(value, list):
                return value

    raise PolymarketAPIError(
        "Expected keyset response to contain a list of events, "
        f"received {type(data)}"
    )


def extract_next_cursor(data: Any) -> str | None:
    """
    Extract the next cursor from a keyset response.
    """

    if not isinstance(data, dict):
        return None

    for key in ["next_cursor", "nextCursor", "next", "cursor"]:
        value = data.get(key)

        if value is not None:
            return str(value)

    return None


def fetch_gamma_events_keyset_page(
    *,
    limit: int = 100,
    after_cursor: str | None = None,
    active: bool | None = None,
    closed: bool | None = None,
    order: str = "volume",
    ascending: bool = False,
    session: requests.Session | None = None,
) -> tuple[list[dict[str, Any]], str | None, Any]:
    """
    Fetch a single page of events from the Gamma API keyset endpoint.

    Keyset pagination should be preferred over offset pagination for large
    event collections because it avoids high-offset API limits.

    Returns
    -------
    tuple[list[dict[str, Any]], str | None, Any]
        Event records, next cursor, and the raw parsed response.
    """

    params: SessionParams = {
        "limit": limit,
        "after_cursor": after_cursor,
        "active": str(active).lower() if active is not None else None,
        "closed": str(closed).lower() if closed is not None else None,
        "order": order,
        "ascending": str(ascending).lower(),
    }

    data = get_json("/events/keyset", params=params, session=session)
    events = extract_keyset_events(data)
    next_cursor = extract_next_cursor(data)

    return events, next_cursor, data


def iter_gamma_events_keyset(
    *,
    limit: int = 100,
    max_pages: int | None = None,
    active: bool | None = None,
    closed: bool | None = None,
    order: str = "volume",
    ascending: bool = False,
    sleep_seconds: float = 0.25,
) -> list[dict[str, Any]]:
    """
    Fetch Gamma API events using keyset pagination.

    The function follows ``next_cursor`` values until the endpoint returns no
    further cursor, an empty page, or ``max_pages`` is reached.
    """

    all_events: list[dict[str, Any]] = []
    after_cursor: str | None = None
    pages_fetched = 0
    seen_cursors: set[str] = set()

    with create_session() as session:
        while True:
            if max_pages is not None and pages_fetched >= max_pages:
                break

            page, next_cursor, _raw_response = fetch_gamma_events_keyset_page(
                limit=limit,
                after_cursor=after_cursor,
                active=active,
                closed=closed,
                order=order,
                ascending=ascending,
                session=session,
            )

            if not page:
                break

            all_events.extend(page)
            pages_fetched += 1

            if next_cursor is None:
                break

            if next_cursor in seen_cursors:
                raise PolymarketAPIError(
                    f"Repeated keyset cursor encountered: {next_cursor}"
                )

            seen_cursors.add(next_cursor)
            after_cursor = next_cursor
            time.sleep(sleep_seconds)

    return all_events


def iter_gamma_events(
    *,
    limit: int = 100,
    start_offset: int = 0,
    max_pages: int | None = None,
    active: bool | None = None,
    closed: bool | None = None,
    order: str = "volume",
    ascending: bool = False,
    sleep_seconds: float = 0.25,
) -> list[dict[str, Any]]:
    """
    Fetch paginated Gamma API events.

    The function stops when the endpoint returns an empty page, a short page,
    or when ``max_pages`` has been reached.
    """

    all_events: list[dict[str, Any]] = []
    offset = start_offset
    pages_fetched = 0

    with create_session() as session:
        while True:
            if max_pages is not None and pages_fetched >= max_pages:
                break

            page = fetch_gamma_events_page(
                limit=limit,
                offset=offset,
                active=active,
                closed=closed,
                order=order,
                ascending=ascending,
                session=session,
            )

            if not page:
                break

            all_events.extend(page)
            pages_fetched += 1

            if len(page) < limit:
                break

            offset += limit
            time.sleep(sleep_seconds)

    return all_events


def count_nested_markets(events: list[dict[str, Any]]) -> int:
    """
    Count nested market objects contained in a list of event records.
    """

    count = 0

    for event in events:
        markets = event.get("markets")

        if isinstance(markets, list):
            count += sum(1 for market in markets if isinstance(market, dict))

    return count


if __name__ == "__main__":
    sample_events, next_cursor, _raw_response = fetch_gamma_events_keyset_page(limit=5)

    print(f"Fetched events: {len(sample_events)}")
    print(f"Nested markets: {count_nested_markets(sample_events)}")
    print(f"Next cursor: {next_cursor}")

    for event in sample_events:
        print(f"{event.get('id')} | {event.get('title')}")
