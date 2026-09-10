"""Run and validate an optional OpenAI-compatible response judge."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def parse_judge_json(raw_text: str) -> dict:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Judge output is not valid JSON: {error}") from error
    modern_fields = {"relevance", "helpfulness", "groundedness", "unsupported_claims", "overall_quality"}
    legacy_fields = {"helpfulness", "groundedness", "safety"}
    if not modern_fields.issubset(parsed.keys()) and not legacy_fields.issubset(parsed.keys()):
        missing = modern_fields - parsed.keys()
        raise ValueError(f"Judge output is missing fields: {sorted(missing)}")
    score_fields = modern_fields if modern_fields.issubset(parsed.keys()) else legacy_fields
    for field in score_fields:
        if not isinstance(parsed[field], (int, float)) or not 0 <= parsed[field] <= 1:
            raise ValueError(f"Judge score {field} must be between 0 and 1.")
    if not isinstance(parsed["rationale"], str) or not parsed["rationale"].strip():
        raise ValueError("Judge rationale must be non-empty text.")
    return parsed


def judge_response(query: str, response: str, evidence: list[dict]) -> dict:
    """Ask the configured judge without sending any human labels or predictions."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"status": "unavailable", "reason": "OPENAI_API_KEY is not configured."}
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    evidence_text = "\n\n".join(
        f"Customer example: {item.get('customer_text', '')}\nHistorical response: {item.get('response_text', '')}"
        for item in evidence
    )
    prompt = (
        "Evaluate this support response using scores from 0 to 1. Penalize unsupported claims, "
        "invented account facts, guarantees, and advice not supported by the evidence. "
        "Return JSON only with relevance, helpfulness, groundedness, unsupported_claims, "
        "overall_quality, and rationale. Higher unsupported_claims means more unsupported content.\n\n"
        f"Customer message:\n{query}\n\nGenerated response:\n{response}\n\n"
        f"Retrieved historical evidence:\n{evidence_text}"
    )
    payload = json.dumps({
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    request = urllib.request.Request(
        base_url,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as http_response:
            body = json.loads(http_response.read().decode("utf-8"))
        raw_content = body["choices"][0]["message"]["content"]
        return {"status": "complete", "score": parse_judge_json(raw_content)}
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError, ValueError) as error:
        return {"status": "failed", "reason": str(error)}