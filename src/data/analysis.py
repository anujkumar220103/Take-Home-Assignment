"""Dataset profiling and analysis functions for Step 1."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .validation import read_rows

TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n"}
ID_PATTERN = re.compile(r"[^,;|]+")


def parse_id_list(value: str) -> list[str]:
    """Parse TWCS response-id fields, which may contain several comma-separated IDs."""
    if not value.strip():
        return []
    return [item.strip() for item in ID_PATTERN.split(value) if item.strip()]


def parse_datetime(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        return None


def profile_dataset(input_path: Path) -> dict:
    row_count = 0
    unique_tweet_ids: set[str] = set()
    unique_conversation_ids: set[str] = set()
    inbound_counts = Counter()
    text_lengths: list[int] = []
    created_dates: list[datetime] = []
    column_value_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in read_rows(input_path):
        row_count += 1
        unique_tweet_ids.add(row.get("tweet_id", ""))
        inbound_value = row.get("inbound", "").strip().lower()
        inbound_counts[inbound_value] += 1
        text_lengths.append(len(row.get("text", "")))
        created_at = parse_datetime(row.get("created_at", ""))
        if created_at:
            created_dates.append(created_at)
        for response_id in parse_id_list(row.get("response_tweet_id", "")):
            unique_conversation_ids.add(response_id)
        for column in ("author_id", "in_response_to_tweet_id"):
            value = row.get(column, "").strip()
            if value:
                column_value_counts[column][value] += 1

    return {
        "row_count": row_count,
        "unique_tweet_count": len(unique_tweet_ids - {""}),
        "inbound_counts": dict(inbound_counts),
        "text_length": _numeric_summary(text_lengths),
        "date_range": _date_range(created_dates),
        "unique_author_count": len(column_value_counts["author_id"]),
        "rows_with_parent_tweet": sum(
            count for value, count in column_value_counts["in_response_to_tweet_id"].items()
            if value
        ),
    }


def analyze_data_quality(input_path: Path) -> dict:
    row_count = 0
    blank_counts = Counter()
    duplicate_tweet_ids = Counter()
    invalid_inbound_values = Counter()
    invalid_dates = 0
    empty_text_rows = 0

    for row in read_rows(input_path):
        row_count += 1
        for column, value in row.items():
            if not value.strip():
                blank_counts[column] += 1
        tweet_id = row.get("tweet_id", "").strip()
        if tweet_id:
            duplicate_tweet_ids[tweet_id] += 1
        inbound = row.get("inbound", "").strip().lower()
        if inbound not in TRUE_VALUES and inbound not in {"0", "false", "f", "no", "n"}:
            invalid_inbound_values[inbound] += 1
        if parse_datetime(row.get("created_at", "")) is None:
            invalid_dates += 1
        if not row.get("text", "").strip():
            empty_text_rows += 1

    duplicate_rows = sum(count - 1 for count in duplicate_tweet_ids.values() if count > 1)
    return {
        "row_count": row_count,
        "blank_value_counts": dict(blank_counts),
        "duplicate_tweet_id_rows": duplicate_rows,
        "invalid_inbound_values": dict(invalid_inbound_values),
        "invalid_created_at_rows": invalid_dates,
        "empty_text_rows": empty_text_rows,
    }


def analyze_brands(input_path: Path) -> dict:
    """Rank likely support accounts; TWCS does not contain human-readable brand names."""
    author_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for row in read_rows(input_path):
        author_id = row.get("author_id", "").strip()
        if not author_id:
            continue
        author_stats[author_id]["rows"] += 1
        inbound_value = row.get("inbound", "").strip().lower()
        if inbound_value in TRUE_VALUES:
            author_stats[author_id]["inbound_rows"] += 1
        elif inbound_value in FALSE_VALUES:
            author_stats[author_id]["outbound_rows"] += 1
        if row.get("response_tweet_id", "").strip():
            author_stats[author_id]["response_link_rows"] += 1

    ranked = []
    for author_id, stats in author_stats.items():
        rows = stats["rows"]
        outbound_rows = stats["outbound_rows"]
        ranked.append(
            {
                "author_id": author_id,
                "rows": rows,
                "inbound_rows": stats["inbound_rows"],
                "outbound_rows": outbound_rows,
                "response_link_rows": stats["response_link_rows"],
                "outbound_share": round(outbound_rows / rows, 6) if rows else 0.0,
            }
        )
    ranked.sort(key=lambda item: (item["outbound_rows"], item["rows"]), reverse=True)
    return {
        "identification_note": (
            "The standard TWCS file has author IDs, not brand-name labels. "
            "Candidates are ranked by outbound activity and row volume; IDs must be mapped externally."
        ),
        "candidate_count": len(ranked),
        "top_candidates": ranked[:20],
    }


def analyze_conversations(input_path: Path) -> dict:
    rows_by_tweet_id: dict[str, dict[str, str]] = {}
    response_edges: list[tuple[str, str]] = []
    root_count = 0
    for row in read_rows(input_path):
        tweet_id = row.get("tweet_id", "").strip()
        if tweet_id:
            rows_by_tweet_id[tweet_id] = row
        parent_id = row.get("in_response_to_tweet_id", "").strip()
        if parent_id and tweet_id:
            response_edges.append((parent_id, tweet_id))
        if not parent_id:
            root_count += 1

    known_parent_edges = sum(parent in rows_by_tweet_id for parent, _ in response_edges)
    child_counts = Counter(parent for parent, _ in response_edges)
    return {
        "row_count": len(rows_by_tweet_id),
        "root_tweet_count": root_count,
        "reply_edge_count": len(response_edges),
        "edges_with_known_parent": known_parent_edges,
        "edges_with_missing_parent": len(response_edges) - known_parent_edges,
        "tweets_with_multiple_replies": sum(count > 1 for count in child_counts.values()),
        "structure": (
            "Rows represent tweets. in_response_to_tweet_id links a reply to one parent; "
            "response_tweet_id may list one or more child IDs. Missing parent rows are retained."
        ),
    }


def choose_brand(brand_analysis: dict) -> dict:
    candidates = brand_analysis["top_candidates"]
    if not candidates:
        raise ValueError("No author IDs were available for brand selection.")
    selected = candidates[0]
    return {
        "recommended_author_id": selected["author_id"],
        "reason": (
            "Selected the candidate with the highest count of outbound rows (inbound=false), "
            "using total row count as the tie-breaker. This is a proxy for support activity, "
            "not a verified brand identity."
        ),
        "statistics": selected,
    }


def analyze_brand_suitability(input_path: Path, brand_id: str) -> dict:
    """Measure whether a brand has enough linked support history for later steps."""
    rows_by_tweet_id: dict[str, dict[str, str]] = {}
    brand_rows: list[dict[str, str]] = []
    for row in read_rows(input_path):
        tweet_id = row.get("tweet_id", "").strip()
        if tweet_id:
            rows_by_tweet_id[tweet_id] = row
        if row.get("author_id", "").strip() == brand_id:
            brand_rows.append(row)

    brand_rows_with_parent = 0
    brand_replies_to_customers = 0
    known_parent_rows = 0
    usable_conversations = 0
    multi_turn_conversations = 0
    customer_authors: set[str] = set()
    for row in brand_rows:
        parent_id = row.get("in_response_to_tweet_id", "").strip()
        if not parent_id:
            continue
        brand_rows_with_parent += 1
        parent_row = rows_by_tweet_id.get(parent_id)
        if parent_row is None:
            continue
        known_parent_rows += 1
        if parent_row.get("inbound", "").strip().lower() in TRUE_VALUES:
            brand_replies_to_customers += 1
            usable_conversations += 1
            if (
                parent_row.get("in_response_to_tweet_id", "").strip()
                or row.get("response_tweet_id", "").strip()
            ):
                multi_turn_conversations += 1
            customer_id = parent_row.get("author_id", "").strip()
            if customer_id:
                customer_authors.add(customer_id)

    return {
        "brand_id": brand_id,
        "brand_rows": len(brand_rows),
        "brand_rows_with_parent": brand_rows_with_parent,
        "brand_rows_with_known_parent": known_parent_rows,
        "brand_replies_to_customer_rows": brand_replies_to_customers,
        "usable_conversations": usable_conversations,
        "multi_turn_conversations": multi_turn_conversations,
        "unique_customers_with_direct_brand_reply": len(customer_authors),
        "response_link_rows": sum(
            bool(row.get("response_tweet_id", "").strip()) for row in brand_rows
        ),
        "sufficient_for_150_to_250_example_set": brand_replies_to_customers >= 250,
        "suitability_reason": (
            "The selected brand has enough direct replies to inbound customer rows to support "
            "intent discovery, response retrieval, and a 150-250 example review set. "
            "This is a volume check; semantic diversity and response quality require later analysis."
        ),
    }


def create_processed_data(input_path: Path, output_path: Path, selected_author_id: str) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_rows = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("The input CSV has no header row.")
        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row.get("author_id", "").strip() == selected_author_id:
                    writer.writerow(row)
                    written_rows += 1
    return written_rows


def create_sample(
    input_path: Path,
    output_path: Path,
    sample_size: int,
    seed: int,
    selected_brand_id: str,
) -> dict:
    """Sample real brand-centered conversation units for development work.

    A unit contains a selected brand reply, its known parent, and known direct
    children. Rows are never synthesized. Units are selected from message-count
    and text-length strata so small and longer interactions are represented.
    """
    import random

    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")

    rows_by_id: dict[str, dict[str, str]] = {}
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    brand_reply_ids: list[str] = []
    fieldnames: list[str] | None = None
    for row in read_rows(input_path):
        if fieldnames is None:
            fieldnames = list(row.keys())
        tweet_id = row.get("tweet_id", "").strip()
        if not tweet_id:
            continue
        rows_by_id[tweet_id] = row
        parent_id = row.get("in_response_to_tweet_id", "").strip()
        if parent_id:
            children_by_parent[parent_id].append(tweet_id)
        if row.get("author_id", "").strip() == selected_brand_id:
            if row.get("inbound", "").strip().lower() in FALSE_VALUES:
                brand_reply_ids.append(tweet_id)

    if not fieldnames:
        raise ValueError("The input CSV has no data rows.")
    units = []
    for brand_reply_id in brand_reply_ids:
        unit_ids = {brand_reply_id}
        brand_reply = rows_by_id[brand_reply_id]
        parent_id = brand_reply.get("in_response_to_tweet_id", "").strip()
        if parent_id in rows_by_id:
            unit_ids.add(parent_id)
            grandparent_id = rows_by_id[parent_id].get(
                "in_response_to_tweet_id", ""
            ).strip()
            if grandparent_id in rows_by_id:
                unit_ids.add(grandparent_id)
        unit_ids.update(
            child_id
            for child_id in children_by_parent.get(brand_reply_id, [])
            if child_id in rows_by_id
        )
        rows = [rows_by_id[tweet_id] for tweet_id in unit_ids]
        units.append(
            {
                "tweet_ids": unit_ids,
                "rows": rows,
                "message_count": len(rows),
                "text_length": sum(len(row.get("text", "")) for row in rows),
                "has_customer_parent": bool(
                    parent_id
                    and parent_id in rows_by_id
                    and rows_by_id[parent_id].get("inbound", "").strip().lower()
                    in TRUE_VALUES
                ),
            }
        )

    if not units:
        raise ValueError(f"No outbound rows found for selected brand: {selected_brand_id}")

    random_generator = random.Random(seed)
    for unit in units:
        message_bucket = "short" if unit["message_count"] <= 2 else "multi_turn"
        length_bucket = "short_text" if unit["text_length"] <= 200 else "long_text"
        unit["stratum"] = f"{message_bucket}_{length_bucket}"
    units_by_stratum: dict[str, list[dict]] = defaultdict(list)
    for unit in units:
        units_by_stratum[unit["stratum"]].append(unit)
    for stratum_units in units_by_stratum.values():
        random_generator.shuffle(stratum_units)

    selected_units: list[dict] = []
    strata = list(units_by_stratum.values())
    while len(selected_units) < sample_size and any(strata):
        for stratum_units in strata:
            if stratum_units and len(selected_units) < sample_size:
                selected_units.append(stratum_units.pop())

    selected_ids = set().union(*(unit["tweet_ids"] for unit in selected_units))
    selected_rows = [rows_by_id[tweet_id] for tweet_id in rows_by_id if tweet_id in selected_ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)
    return {
        "sample_rows": len(selected_rows),
        "sample_units": len(selected_units),
        "brand_reply_rows_available": len(brand_reply_ids),
        "customer_parent_rows_in_sample": sum(
            unit["has_customer_parent"] for unit in selected_units
        ),
        "multi_turn_units_in_sample": sum(
            unit["message_count"] > 2 for unit in selected_units
        ),
        "strata_in_sample": dict(Counter(unit["stratum"] for unit in selected_units)),
        "strategy": (
            "Deterministically shuffled real conversation units centered on selected-brand "
            "outbound replies. Each unit includes the known parent, one known grandparent, "
            "and known direct children; units are round-robin sampled across message-count "
            "and aggregate text-length strata."
        ),
    }


def _numeric_summary(values: Iterable[int]) -> dict[str, float | int]:
    values = list(values)
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    ordered_values = sorted(values)
    midpoint = len(ordered_values) // 2
    median = ordered_values[midpoint]
    if len(ordered_values) % 2 == 0:
        median = (ordered_values[midpoint - 1] + ordered_values[midpoint]) / 2
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
        "median": median,
    }


def _date_range(values: list[datetime]) -> dict[str, str | None]:
    if not values:
        return {"earliest": None, "latest": None}
    return {"earliest": min(values).isoformat(), "latest": max(values).isoformat()}
