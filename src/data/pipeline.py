"""Orchestration for the complete Step 1 dataset pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from .analysis import (
    analyze_brands,
    analyze_brand_suitability,
    analyze_conversations,
    analyze_data_quality,
    choose_brand,
    create_processed_data,
    create_sample,
    profile_dataset,
)
from .validation import validate_input


def run_step1(
    input_path: Path,
    reports_dir: Path,
    processed_path: Path,
    sample_path: Path,
    sample_size: int,
    seed: int,
) -> dict:
    validation = validate_input(input_path)
    if not validation.is_valid:
        missing = ", ".join(validation.missing_required_columns)
        raise ValueError(f"Missing required columns: {missing}")

    profiling = profile_dataset(input_path)
    data_quality = analyze_data_quality(input_path)
    brands = analyze_brands(input_path)
    conversations = analyze_conversations(input_path)
    recommendation = choose_brand(brands)
    suitability = analyze_brand_suitability(
        input_path, recommendation["recommended_author_id"]
    )
    processed_rows = create_processed_data(
        input_path, processed_path, recommendation["recommended_author_id"]
    )
    sample = create_sample(
        input_path,
        sample_path,
        sample_size,
        seed,
        recommendation["recommended_author_id"],
    )

    results = {
        "input": str(input_path),
        "validation": {
            "row_count": validation.row_count,
            "columns": validation.columns,
            "blank_value_counts": validation.blank_value_counts,
        },
        "profiling": profiling,
        "data_quality": data_quality,
        "brands": brands,
        "conversations": conversations,
        "recommendation": recommendation,
        "suitability": suitability,
        "outputs": {
            "processed_path": str(processed_path),
            "processed_rows": processed_rows,
            "sample_path": str(sample_path),
            **sample,
            "sample_seed": seed,
        },
    }
    _write_reports(results, reports_dir)
    return results


def _write_reports(results: dict, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    for report_name, report in results.items():
        if report_name in {"input", "outputs"}:
            continue
        report_path = reports_dir / f"{report_name}.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    (reports_dir / "step1_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "step1_summary.md").write_text(
        _build_summary(results), encoding="utf-8"
    )


def _build_summary(results: dict) -> str:
    validation = results["validation"]
    profiling = results["profiling"]
    quality = results["data_quality"]
    conversations = results["conversations"]
    brands = results["brands"]
    recommendation = results["recommendation"]
    suitability = results["suitability"]
    outputs = results["outputs"]
    columns = {
        "tweet_id": "tweet identifier",
        "author_id": "tweet author; brand accounts appear as named support handles",
        "inbound": "whether the tweet is inbound to the support account",
        "created_at": "tweet creation timestamp",
        "text": "tweet message text",
        "response_tweet_id": "child response tweet ID or IDs",
        "in_response_to_tweet_id": "parent tweet ID for a reply",
    }
    column_lines = "\n".join(
        f"- `{column}`: {columns.get(column, 'additional dataset column')}"
        for column in validation["columns"]
    )
    candidate_lines = "\n".join(
        f"- `{candidate['author_id']}`: {candidate['outbound_rows']:,} outbound rows, "
        f"{candidate['response_link_rows']:,} response links"
        for candidate in brands["top_candidates"][:10]
    )
    return f"""# Step 1 Dataset Summary

## Dataset

- Rows: **{profiling['row_count']:,}**
- Unique tweets: **{profiling['unique_tweet_count']:,}**
- Unique authors: **{profiling['unique_author_count']:,}**
- Date range: **{profiling['date_range']['earliest']}** to **{profiling['date_range']['latest']}**
- Inbound rows: **{profiling['inbound_counts'].get('true', 0):,}**
- Outbound rows: **{profiling['inbound_counts'].get('false', 0):,}**

## Columns

{column_lines}

## Data Quality

- Blank values: `response_tweet_id` {validation['blank_value_counts'].get('response_tweet_id', 0):,}; `in_response_to_tweet_id` {validation['blank_value_counts'].get('in_response_to_tweet_id', 0):,}; no blanks in other observed columns.
- Duplicate tweet ID rows: **{quality['duplicate_tweet_id_rows']:,}**
- Invalid inbound values: **{sum(quality['invalid_inbound_values'].values()):,}**
- Invalid timestamps: **{quality['invalid_created_at_rows']:,}**
- Empty text rows: **{quality['empty_text_rows']:,}**
- Message length in characters: min **{profiling['text_length']['min']}**, median **{profiling['text_length']['median']}**, mean **{profiling['text_length']['mean']}**, max **{profiling['text_length']['max']}**.

## Conversation Structure

{conversations['structure']}

- Root tweets: **{conversations['root_tweet_count']:,}**
- Reply edges: **{conversations['reply_edge_count']:,}**
- Edges with known parents: **{conversations['edges_with_known_parent']:,}**
- Edges with missing parents: **{conversations['edges_with_missing_parent']:,}**
- Tweets with multiple replies: **{conversations['tweets_with_multiple_replies']:,}**

Important corner cases are missing parent rows, blank response-link fields on root or terminal tweets, multiple replies from one parent, and the fact that the file contains author IDs rather than an explicit brand-name column.

## Brand Analysis

Top candidates by outbound support rows:

{candidate_lines}

### Recommendation

Recommended brand handle: **`{recommendation['recommended_author_id']}`**.

Evidence: it has **{recommendation['statistics']['outbound_rows']:,}** outbound rows, **{recommendation['statistics']['response_link_rows']:,}** response links, and ranks first among **{brands['candidate_count']:,}** author IDs by outbound support activity. The selection is a dataset handle recommendation, not an externally verified legal brand identity.

Suitability for later work:

- Brand rows: **{suitability['brand_rows']:,}**
- Brand replies with known parents: **{suitability['brand_rows_with_known_parent']:,}**
- Direct replies to inbound customer rows: **{suitability['brand_replies_to_customer_rows']:,}**
- Usable brand-centered conversations: **{suitability['usable_conversations']:,}**
- Unique customers with a direct brand reply: **{suitability['unique_customers_with_direct_brand_reply']:,}**
- Multi-turn conversations available: **{suitability['multi_turn_conversations']:,}**
- 150-250 example-set volume check: **{'passes' if suitability['sufficient_for_150_to_250_example_set'] else 'does not pass'}**

This volume is sufficient for the requested later-stage minimums, but it does not establish intent diversity, retrieval quality, or golden-set quality. Those must be assessed in later steps.

## Sampling and Preprocessing

- Development sample: **{outputs['sample_rows']:,}** rows across **{outputs['sample_units']:,}** real conversation units, centered on `{recommendation['recommended_author_id']}` outbound replies.
- Sampling strategy: {outputs['strategy']}
- Sample composition: **{outputs['customer_parent_rows_in_sample']:,}** units have a customer parent and **{outputs['multi_turn_units_in_sample']:,}** are multi-turn units; strata: `{outputs['strata_in_sample']}`.
- Random seed: **{outputs['sample_seed']}**.
- Processed output: rows authored by the recommended handle, preserving the original CSV columns and values. The development sample preserves original rows, IDs, text, and available parent/child links.
- No text normalization, deduplication, imputation, or semantic filtering was applied; raw text is preserved.

## Files

- `reports/step1_results.json` and individual JSON reports
- `reports/step1_summary.md`
- `data/processed/recommended_brand.csv` ({outputs['processed_rows']:,} rows)
- `data/samples/development_sample.csv` ({outputs['sample_rows']:,} rows)

## Limitations

- The dataset has no explicit brand-name field, so handle names are used as brand proxies.
- The sample is reproducible and conversation-centered, but stratification is based on structural and text-length proxies rather than semantic intent labels.
- Parent-child linkage is incomplete for **{conversations['edges_with_missing_parent']:,}** edges; missing source tweets were retained rather than fabricated.
- No Step 2 or later AI functionality is included.
"""
