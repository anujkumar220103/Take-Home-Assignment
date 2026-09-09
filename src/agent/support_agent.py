"""Retrieval-grounded AmazonHelp support agent for Step 3."""

from __future__ import annotations

import csv
import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import Pipeline

from src.data.step2_intents import (
    INTENT_DESCRIPTIONS,
    discover_intents,
    train_classifier,
)

DEFAULT_TOP_K = 3
MIN_RETRIEVAL_SCORE = 0.08
MIN_CLASSIFIER_CONFIDENCE = 0.45
RANKING_WEIGHTS = {
    "text_similarity": 0.60,
    "intent_compatibility": 0.20,
    "response_availability": 0.10,
    "conversation_quality": 0.10,
}
ACCOUNT_SPECIFIC_PATTERNS = (
    r"\bwhere exactly is my (?:order|package)\b",
    r"\b(?:check|track) my (?:order|refund|package)\b",
    r"\bwhy was i charged\b",
    r"\bchange my (?:delivery )?address\b",
    r"\b(cancel|update|change) my order\b",
)


def load_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return list(csv.DictReader(input_file))


def build_cases(input_path: Path, selected_brand_id: str = "AmazonHelp") -> list[dict[str, str]]:
    """Build cases only from observed customer parent and brand reply rows."""
    rows = load_rows(input_path)
    rows_by_id = {row.get("tweet_id", "").strip(): row for row in rows}
    cases = []
    for response in rows:
        if response.get("author_id", "").strip() != selected_brand_id:
            continue
        parent_id = response.get("in_response_to_tweet_id", "").strip()
        parent = rows_by_id.get(parent_id)
        if not parent or parent.get("inbound", "").strip().lower() != "true":
            continue
        cases.append(
            {
                "customer_tweet_id": parent_id,
                "response_tweet_id": response.get("tweet_id", "").strip(),
                "customer_text": parent.get("text", ""),
                "response_text": response.get("text", ""),
                "created_at": response.get("created_at", ""),
                "conversation_group": parent_id,
                "intent": "",
                "response_available": bool(response.get("text", "").strip()),
                "conversation_quality": _conversation_quality(parent, response, rows_by_id),
            }
        )
    return cases


def build_retrieval_index(
    cases: list[dict[str, str]], labeled_customer_rows: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    if not cases:
        raise ValueError("No linked customer-to-brand cases were found.")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
    )
    matrix = vectorizer.fit_transform(case["customer_text"] for case in cases)
    if labeled_customer_rows:
        labeled_customer_rows = discover_intents(labeled_customer_rows)
        labels_by_text = {
            row.get("text", ""): row.get("intent", "")
            for row in labeled_customer_rows
        }
        for case in cases:
            case["intent"] = labels_by_text.get(case["customer_text"], "")
    discovered_cases = discover_intents(
        [{"text": case["customer_text"]} for case in cases]
    )
    for case, discovered_case in zip(cases, discovered_cases):
        if not case["intent"]:
            case["intent"] = discovered_case["intent"]
    return {"cases": cases, "vectorizer": vectorizer, "matrix": matrix}


def classify_query(query: str, labeled_customer_rows: list[dict[str, str]]) -> dict[str, Any]:
    labeled_rows = discover_intents(labeled_customer_rows)
    classifier: Pipeline = train_classifier(labeled_rows)
    probabilities = classifier.predict_proba([query])[0]
    predicted_index = int(probabilities.argmax())
    predicted_intent = classifier.classes_[predicted_index]
    return {
        "intent": predicted_intent,
        "confidence": round(float(probabilities[predicted_index]), 4),
        "classifier": classifier,
    }


def retrieve_cases(
    query: str,
    index: dict[str, Any],
    top_k: int = DEFAULT_TOP_K,
    predicted_intent: str = "",
    before_created_at: str | None = None,
    excluded_conversation_group: str | None = None,
) -> list[dict[str, Any]]:
    query_vector = index["vectorizer"].transform([query])
    scores = linear_kernel(query_vector, index["matrix"]).ravel()
    ranked_cases = []
    for index_value in scores.argsort()[::-1]:
        case = index["cases"][int(index_value)]
        if excluded_conversation_group and case.get("conversation_group") == excluded_conversation_group:
            continue
        if before_created_at and not _is_before(case.get("created_at", ""), before_created_at):
            continue
        text_similarity = float(scores[index_value])
        intent_compatibility = 1.0 if predicted_intent and case.get("intent") == predicted_intent else 0.0
        response_availability = 1.0 if case.get("response_available") else 0.0
        conversation_quality = float(case.get("conversation_quality", 0.0))
        ranking_score = (
            RANKING_WEIGHTS["text_similarity"] * text_similarity
            + RANKING_WEIGHTS["intent_compatibility"] * intent_compatibility
            + RANKING_WEIGHTS["response_availability"] * response_availability
            + RANKING_WEIGHTS["conversation_quality"] * conversation_quality
        )
        ranked_cases.append(
            {
                **case,
                "text_similarity": round(text_similarity, 4),
                "intent_compatibility": intent_compatibility,
                "response_availability": response_availability,
                "conversation_quality": round(conversation_quality, 4),
                "retrieval_score": round(ranking_score, 4),
            }
        )
        if len(ranked_cases) >= top_k:
            break
    return ranked_cases


def choose_action(
    query: str, classification: dict[str, Any], retrieved_cases: list[dict[str, Any]]
) -> dict[str, Any]:
    top_score = retrieved_cases[0]["retrieval_score"] if retrieved_cases else 0.0
    text_similarity = retrieved_cases[0].get("text_similarity", 0.0) if retrieved_cases else 0.0
    confidence = classification["confidence"]
    reasons = []
    if is_account_specific_request(query):
        reasons.append("the request needs private account or order access that this agent does not have")
    if not retrieved_cases:
        reasons.append("no linked historical case was retrieved")
    if top_score < MIN_RETRIEVAL_SCORE or text_similarity < MIN_RETRIEVAL_SCORE:
        reasons.append(
            f"retrieval evidence is below {MIN_RETRIEVAL_SCORE:.2f} "
            f"(ranked={top_score:.4f}, text={text_similarity:.4f})"
        )
    if confidence < MIN_CLASSIFIER_CONFIDENCE:
        reasons.append(
            f"intent confidence {confidence:.4f} is below {MIN_CLASSIFIER_CONFIDENCE:.2f}"
        )
    if reasons:
        return {
            "action": "escalate",
            "reason": "; ".join(reasons),
            "top_retrieval_score": top_score,
            "top_text_similarity": text_similarity,
            "evidence_strength": "weak" if top_score < MIN_RETRIEVAL_SCORE else "mixed",
        }
    return {
        "action": "handle",
        "reason": "intent confidence and historical retrieval score passed the configured thresholds",
        "top_retrieval_score": top_score,
        "top_text_similarity": text_similarity,
        "evidence_strength": "strong",
    }


def grounded_fallback_reply(query: str, classification: dict[str, Any], retrieved_cases: list[dict[str, Any]]) -> str:
    if is_account_specific_request(query):
        return (
            "I cannot access your private order, payment, refund, or account details. "
            "I found historical support guidance, but a support specialist must check or change the account."
        )
    if not retrieved_cases:
        return "I could not find a sufficiently similar historical case, so a support specialist should review this request."
    topic_by_intent = {
        "delivery_tracking": "the order's available tracking and delivery information",
        "delivery_delay": "the order's promised delivery date and tracking information",
        "returns_refunds": "the order and return or refund details",
        "order_changes_cancellation": "the order details and the requested change",
        "missing_wrong_damaged_item": "the item and order details",
        "account_login": "the account access or password-reset details",
        "payment_billing": "the charge or payment details",
        "prime_subscription": "the Prime membership or subscription details",
    }
    topic = topic_by_intent.get(classification["intent"], "the issue and order details")
    return (
        f"I cannot access your account directly. Based on a similar historical "
        f"{classification['intent'].replace('_', ' ')} case, please review {topic} "
        "and contact an Amazon support specialist if you need account-specific help."
    )


def generate_reply(
    query: str,
    classification: dict[str, Any],
    retrieved_cases: list[dict[str, Any]],
    use_llm: bool = True,
) -> dict[str, Any]:
    """Use an optional OpenAI-compatible endpoint, otherwise use grounded fallback."""
    if not use_llm or not os.getenv("OPENAI_API_KEY"):
        return {"mode": "mock", "text": grounded_fallback_reply(query, classification, retrieved_cases)}
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    evidence = "\n\n".join(
        f"Customer: {case['customer_text']}\nHistorical response: {case['response_text']}"
        for case in retrieved_cases
    )
    prompt = (
        "You are an AmazonHelp support assistant. You cannot access private accounts or perform actions. "
        "Answer only using the historical evidence. "
        "Do not invent policies, order details, refunds, or guarantees. If evidence is insufficient, "
        "say that a specialist should review the case.\n\n"
        f"Customer message: {query}\nIntent: {classification['intent']}\nEvidence:\n{evidence}"
    )
    payload = json.dumps(
        {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"].strip()
        return {"mode": "llm", "text": text}
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as error:
        return {
            "mode": "mock_after_llm_error",
            "text": grounded_fallback_reply(query, classification, retrieved_cases),
            "error": str(error),
        }


def answer_query(
    query: str,
    index: dict[str, Any],
    labeled_customer_rows: list[dict[str, str]],
    use_llm: bool = False,
    before_created_at: str | None = None,
    excluded_conversation_group: str | None = None,
) -> dict[str, Any]:
    classification = classify_query(query, labeled_customer_rows)
    retrieved_cases = retrieve_cases(
        query,
        index,
        predicted_intent=classification["intent"],
        before_created_at=before_created_at,
        excluded_conversation_group=excluded_conversation_group,
    )
    decision = choose_action(query, classification, retrieved_cases)
    reply = generate_reply(query, classification, retrieved_cases, use_llm and decision["action"] == "handle")
    return {
        "query": query,
        "intent": classification["intent"],
        "intent_description": INTENT_DESCRIPTIONS.get(classification["intent"], ""),
        "intent_confidence": classification["confidence"],
        "input_message": query,
        "retrieved_cases": retrieved_cases,
        "decision": decision,
        "evidence_strength": decision["evidence_strength"],
        "reply": reply,
    }


def write_index_summary(output_path: Path, cases: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "indexed_cases": len(cases),
                "fields": ["customer_tweet_id", "response_tweet_id", "customer_text", "response_text"],
                "note": "Only observed customer-parent and AmazonHelp-response pairs are indexed.",
                "ranking_weights": RANKING_WEIGHTS,
                "thresholds": {
                    "retrieval_score": MIN_RETRIEVAL_SCORE,
                    "intent_confidence": MIN_CLASSIFIER_CONFIDENCE,
                    "status": "provisional; Step 4 must tune and evaluate them",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def is_account_specific_request(query: str) -> bool:
    normalized_query = query.lower()
    return any(re.search(pattern, normalized_query) for pattern in ACCOUNT_SPECIFIC_PATTERNS)


def _conversation_quality(
    parent: dict[str, str], response: dict[str, str], rows_by_id: dict[str, dict[str, str]]
) -> float:
    score = 0.5
    if parent.get("in_response_to_tweet_id", "").strip() in rows_by_id:
        score += 0.25
    if response.get("response_tweet_id", "").strip():
        score += 0.25
    return min(score, 1.0)


def _is_before(case_timestamp: str, query_timestamp: str) -> bool:
    try:
        return datetime.strptime(case_timestamp, "%a %b %d %H:%M:%S %z %Y") < datetime.strptime(
            query_timestamp, "%a %b %d %H:%M:%S %z %Y"
        )
    except ValueError:
        return False