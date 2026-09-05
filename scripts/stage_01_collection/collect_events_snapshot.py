"""
Collect raw Polymarket event data using Gamma API keyset pagination.

Purpose
-------
This script is the production-style event collector for the event-first data
architecture. It fetches Polymarket events through the Gamma API keyset endpoint
and saves the raw API response data before any database loading or
normalisation.

Raw files are saved to:

    data/raw/polymarket/events/

The loader will later read these raw JSON files and populate the event-first
SQLite database.

Run from the project root:

    python3 scripts/stage_01_collection/collect_events_snapshot.py

For a small test run, set MAX_PAGES below to a small number, for example 5.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from api.polymarket_gamma_api import (  # noqa: E402
    PolymarketAPIError,
    count_nested_markets,
    create_session,
    fetch_gamma_events_keyset_page,
)


RAW_EVENTS_DIR = Path("data/raw/polymarket/events")
RAW_EVENTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LIMIT = 100
DEFAULT_SLEEP_SECONDS = 0.05
DEFAULT_MAX_PAGES: int | None = 2
PROGRESS_EVERY_N_PAGES = 10


def utc_now_iso() -> str:
    """
    Return the current UTC time as an ISO-formatted string.
    """

    return datetime.now(timezone.utc).isoformat()


def timestamp_for_filename() -> str:
    """
    Return a UTC timestamp suitable for filenames.
    """

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def get_event_id(event: dict[str, Any]) -> str | None:
    """
    Extract an event ID as a string.
    """

    event_id = event.get("id")

    if event_id is None:
        return None

    return str(event_id)


def count_event_tags(events: list[dict[str, Any]]) -> int:
    """
    Count event-level tag objects across a list of events.
    """

    total_tags = 0

    for event in events:
        tags = event.get("tags")

        if isinstance(tags, list):
            total_tags += sum(1 for tag in tags if isinstance(tag, dict))

    return total_tags


def collect_events_keyset(
    *,
    limit: int = DEFAULT_LIMIT,
    max_pages: int | None = DEFAULT_MAX_PAGES,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    active: bool | None = None,
    closed: bool | None = None,
    order: str = "volume",
    ascending: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Collect events using keyset pagination and return raw records plus metadata.

    The collection follows next-cursor values until the endpoint returns no next
    cursor, an empty page, a repeated cursor, an API error, or MAX_PAGES is
    reached.
    """

    events_by_id: dict[str, dict[str, Any]] = {}
    duplicate_event_ids: Counter[str] = Counter()
    page_summaries: list[dict[str, Any]] = []
    seen_cursors: set[str] = set()

    after_cursor: str | None = None
    page_number = 1
    stopped_reason = "unknown"
    total_records_returned = 0

    collection_started_at_utc = utc_now_iso()
    collection_started_timestamp = time.perf_counter()
    progress_checkpoint_timestamp = None

    with create_session() as session:
        while True:
            if max_pages is not None and page_number > max_pages:
                stopped_reason = "max_pages_reached"
                break

            try:
                page_events, next_cursor, _raw_response = fetch_gamma_events_keyset_page(
                    limit=limit,
                    after_cursor=after_cursor,
                    active=active,
                    closed=closed,
                    order=order,
                    ascending=ascending,
                    session=session,
                )
            except PolymarketAPIError as error:
                stopped_reason = "api_error"
                page_summaries.append(
                    {
                        "page_number": page_number,
                        "after_cursor": after_cursor,
                        "events_returned": 0,
                        "nested_markets_returned": 0,
                        "event_tags_returned": 0,
                        "next_cursor_present": False,
                        "error_message": str(error),
                    }
                )
                print(f"API error on page {page_number}: {error}")
                break

            events_returned = len(page_events)
            nested_markets_returned = count_nested_markets(page_events)
            event_tags_returned = count_event_tags(page_events)
            next_cursor_present = next_cursor is not None
            total_records_returned += events_returned

            duplicate_count_on_page = 0

            for event in page_events:
                event_id = get_event_id(event)

                if event_id is None:
                    continue

                if event_id in events_by_id:
                    duplicate_event_ids[event_id] += 1
                    duplicate_count_on_page += 1

                events_by_id[event_id] = event

            page_summaries.append(
                {
                    "page_number": page_number,
                    "after_cursor": after_cursor,
                    "events_returned": events_returned,
                    "nested_markets_returned": nested_markets_returned,
                    "event_tags_returned": event_tags_returned,
                    "duplicate_events_on_page": duplicate_count_on_page,
                    "unique_events_so_far": len(events_by_id),
                    "next_cursor_present": next_cursor_present,
                    "next_cursor": next_cursor,
                }
            )

            if page_number == 1:
                elapsed_since_start = (
                    time.perf_counter() - collection_started_timestamp
                )

                print(
                    f"Page {page_number}: "
                    f"events={events_returned}, "
                    f"unique_events={len(events_by_id)}, "
                    f"nested_markets={nested_markets_returned}, "
                    f"event_tags={event_tags_returned}, "
                    f"next_cursor={next_cursor_present}, "
                    f"elapsed_since_start={elapsed_since_start:.2f}s"
                )

                progress_checkpoint_timestamp = time.perf_counter()

            elif page_number % PROGRESS_EVERY_N_PAGES == 0:
                current_timestamp = time.perf_counter()
                interval_seconds = current_timestamp - progress_checkpoint_timestamp
                progress_checkpoint_timestamp = current_timestamp

                print(
                    f"Page {page_number}: "
                    f"events={events_returned}, "
                    f"unique_events={len(events_by_id)}, "
                    f"nested_markets={nested_markets_returned}, "
                    f"event_tags={event_tags_returned}, "
                    f"next_cursor={next_cursor_present}, "
                    f"time_for_last_{PROGRESS_EVERY_N_PAGES}_pages={interval_seconds:.2f}s"
                )

            if events_returned == 0:
                stopped_reason = "empty_page"
                break

            if next_cursor is None:
                stopped_reason = "no_next_cursor"
                break

            if next_cursor in seen_cursors:
                stopped_reason = "repeated_cursor"
                break

            seen_cursors.add(next_cursor)
            after_cursor = next_cursor
            page_number += 1
            time.sleep(sleep_seconds)

    collection_finished_at_utc = utc_now_iso()
    unique_events = list(events_by_id.values())

    metadata = {
        "collection_started_at_utc": collection_started_at_utc,
        "collection_finished_at_utc": collection_finished_at_utc,
        "method": "gamma_events_keyset",
        "limit": limit,
        "max_pages": max_pages,
        "sleep_seconds": sleep_seconds,
        "active": active,
        "closed": closed,
        "order": order,
        "ascending": ascending,
        "stopped_reason": stopped_reason,
        "pages_requested": len(page_summaries),
        "total_records_returned": total_records_returned,
        "total_unique_events": len(unique_events),
        "duplicate_event_count": total_records_returned - len(unique_events),
        "duplicate_event_ids": dict(duplicate_event_ids),
        "total_nested_markets": count_nested_markets(unique_events),
        "total_event_tags": count_event_tags(unique_events),
        "page_summaries": page_summaries,
    }

    return unique_events, metadata


def parse_optional_bool(value: str) -> bool:
    """Parse common command-line boolean strings."""

    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Use true or false.")


def save_raw_events(events: list[dict[str, Any]], metadata: dict[str, Any]) -> Path:
    """
    Save collected events and collection metadata to a raw JSON snapshot.
    """

    timestamp = timestamp_for_filename()
    output_path = RAW_EVENTS_DIR / f"polymarket_events_keyset_{timestamp}.json"

    payload = {
        "metadata": metadata,
        "events": events,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return output_path


def print_collection_summary(metadata: dict[str, Any], output_path: Path) -> None:
    """
    Print a concise collection summary.
    """

    print()
    print("=" * 80)
    print("Polymarket Event Collection Summary")
    print("=" * 80)
    print(f"Stopped reason: {metadata['stopped_reason']}")
    print(f"Pages requested: {metadata['pages_requested']}")
    print(f"Total records returned: {metadata['total_records_returned']}")
    print(f"Total unique events: {metadata['total_unique_events']}")
    print(f"Duplicate event count: {metadata['duplicate_event_count']}")
    print(f"Total nested markets: {metadata['total_nested_markets']}")
    print(f"Total event-level tags: {metadata['total_event_tags']}")
    print()
    print("Saved raw event snapshot")
    print("-" * 80)
    print(output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Collect raw Polymarket event snapshots from the Gamma API."
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="Maximum number of pages to collect. Use 0 for no cap.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Delay between API pages.",
    )
    parser.add_argument(
        "--active",
        type=parse_optional_bool,
        default=None,
        help="Optionally restrict events by active=true/false.",
    )
    parser.add_argument(
        "--closed",
        type=parse_optional_bool,
        default=None,
        help="Optionally restrict events by closed=true/false.",
    )
    parser.add_argument("--order", type=str, default="volume")
    parser.add_argument("--ascending", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_pages = None if args.max_pages == 0 else args.max_pages
    events, metadata = collect_events_keyset(
        limit=args.limit,
        max_pages=max_pages,
        sleep_seconds=args.sleep_seconds,
        active=args.active,
        closed=args.closed,
        order=args.order,
        ascending=args.ascending,
    )
    output_path = save_raw_events(events, metadata)
    print_collection_summary(metadata, output_path)


if __name__ == "__main__":
    main()
