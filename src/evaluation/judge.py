"""Parse precomputed judge responses without making live API calls."""

from __future__ import annotations

import json


def parse_judge_json(raw_text: str) -> dict:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Judge output is not valid JSON: {error}") from error
    required = {"helpfulness", "groundedness", "safety", "rationale"}
    missing = required - parsed.keys()
    if missing:
        raise ValueError(f"Judge output is missing fields: {sorted(missing)}")
    for field in ("helpfulness", "groundedness", "safety"):
        if not isinstance(parsed[field], (int, float)) or not 0 <= parsed[field] <= 1:
            raise ValueError(f"Judge score {field} must be between 0 and 1.")
    if not isinstance(parsed["rationale"], str) or not parsed["rationale"].strip():
        raise ValueError("Judge rationale must be non-empty text.")
    return parsed