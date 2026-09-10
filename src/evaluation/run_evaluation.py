"""Run the complete offline-first Step 4 evaluation on the frozen golden set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.agent.support_agent import answer_query, build_cases, build_retrieval_index, load_rows
from src.data.step2_intents import INTENT_DESCRIPTIONS, discover_intents, train_classifier
from .agreement import rating_agreement
from .baselines import majority_baseline, simple_tfidf_baseline
from .judge import judge_response
from .metrics import classification_metrics, escalation_metrics, quality_summary, validate_human_labels

DEFAULT_GOLDEN = Path("data/golden/golden_set.csv")
DEFAULT_RAW = Path("data/raw/twcs.csv")
DEFAULT_CLASSIFIER = Path("data/samples/development_sample.csv")
DEFAULT_RESULTS = Path("reports/evaluation_results.json")
DEFAULT_REPORT = Path("reports/step4_evaluation_report.md")
DEFAULT_FAILURES = Path("reports/step4_failure_analysis.md")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return list(csv.DictReader(input_file))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round(value: float) -> float:
    return round(float(value), 6)


def _main_predictions(golden_rows: list[dict[str, str]], raw_path: Path, classifier_path: Path) -> list[dict]:
    classifier_rows = [row for row in load_rows(classifier_path) if row.get("inbound", "").lower() == "true"]
    cases = build_cases(raw_path)
    index = build_retrieval_index(cases, classifier_rows)
    frozen_classifier = train_classifier(discover_intents(classifier_rows))
    predictions = []
    for row in golden_rows:
        result = answer_query(
            row["input_message"], index, classifier_rows, use_llm=False, classifier=frozen_classifier
        )
        decision = result["decision"]
        predictions.append(
            {
                "example_id": row["example_id"],
                "input_message": row["input_message"],
                "model_predicted_intent": result["intent"],
                "intent_confidence": result["intent_confidence"],
                "model_action": "AUTO_HANDLE" if decision["action"] == "handle" else "ESCALATE_TO_HUMAN",
                "decision_reason": decision["reason"],
                "top_retrieval_score": decision["top_retrieval_score"],
                "top_text_similarity": decision["top_text_similarity"],
                "evidence_strength": result["evidence_strength"],
                "generated_response": result["reply"]["text"],
                "retrieved_evidence": [
                    {
                        "customer_text": case["customer_text"],
                        "response_text": case["response_text"],
                        "retrieval_score": case["retrieval_score"],
                        "text_similarity": case["text_similarity"],
                    }
                    for case in result["retrieved_cases"]
                ],
            }
        )
    return predictions


def _baseline_metrics(name: str, actual: list[str], predicted: list[str]) -> dict:
    return {"system": name, **classification_metrics(actual, predicted, list(INTENT_DESCRIPTIONS))}


def _judge_results(predictions: list[dict]) -> tuple[list[dict], dict]:
    results = []
    for prediction in predictions:
        judged = judge_response(
            prediction["input_message"], prediction["generated_response"], prediction["retrieved_evidence"]
        )
        results.append({"example_id": prediction["example_id"], **judged})
    completed = [item for item in results if item["status"] == "complete"]
    summary = {"requested": len(results), "completed": len(completed), "coverage": len(completed) / len(results) if results else 0.0}
    if not completed:
        summary["status"] = "unavailable" if results and results[0]["status"] == "unavailable" else "no successful judgments"
        summary["reason"] = results[0].get("reason", "No judge results available.") if results else "No rows available."
        return results, summary
    scores = [item["score"] for item in completed]
    fields = ("relevance", "helpfulness", "groundedness", "unsupported_claims", "overall_quality")
    summary.update({field: _round(sum(score[field] for score in scores) / len(scores)) for field in fields})
    return results, summary


def _agreement(golden_rows: list[dict[str, str]], judge_results: list[dict]) -> dict:
    by_id = {item["example_id"]: item for item in judge_results if item.get("status") == "complete"}
    human = []
    judge = []
    for row in golden_rows:
        item = by_id.get(row["example_id"])
        if item:
            human.append(float(row["human_reply_quality"]))
            judge.append(float(item["score"]["overall_quality"]))
    if not human:
        return {"joint_count": 0, "status": "unavailable"}
    return {"status": "available", **rating_agreement(human, judge)}


def _failure_rows(golden_rows: list[dict[str, str]], predictions: list[dict]) -> list[dict]:
    golden_by_id = {row["example_id"]: row for row in golden_rows}
    candidates = []
    for prediction in predictions:
        row = golden_by_id[prediction["example_id"]]
        intent_error = prediction["model_predicted_intent"] != row["human_intent"]
        false_auto = row["human_escalation"] == "ESCALATE_TO_HUMAN" and prediction["model_action"] == "AUTO_HANDLE"
        false_escalation = row["human_escalation"] == "AUTO_HANDLE" and prediction["model_action"] == "ESCALATE_TO_HUMAN"
        poor_quality = float(row["human_reply_quality"]) <= 0.5
        poor_grounding = float(row["human_grounding_quality"]) <= 0.5
        weak_evidence = prediction["top_text_similarity"] < 0.08
        priority = (false_auto, intent_error, false_escalation, poor_grounding, poor_quality, weak_evidence)
        if any(priority):
            candidates.append(
                {
                    "example_id": row["example_id"],
                    "customer_message": row["input_message"],
                    "human_intent": row["human_intent"],
                    "predicted_intent": prediction["model_predicted_intent"],
                    "human_escalation": row["human_escalation"],
                    "predicted_action": prediction["model_action"],
                    "generated_response": prediction["generated_response"],
                    "relevant_evidence": prediction["retrieved_evidence"][:2],
                    "human_reply_quality": row["human_reply_quality"],
                    "human_grounding_quality": row["human_grounding_quality"],
                    "failure_reasons": {
                        "incorrect_intent": intent_error,
                        "false_auto_handle": false_auto,
                        "false_escalation": false_escalation,
                        "poor_reply_quality": poor_quality,
                        "poor_grounding": poor_grounding,
                        "weak_retrieval": weak_evidence,
                    },
                }
            )
    candidates.sort(key=lambda item: tuple(not value for value in item["failure_reasons"].values()))
    return candidates[:5]


def _write_failure_report(failures: list[dict], path: Path) -> None:
    lines = ["# Step 4 Failure Analysis", "", "These examples are selected only from the frozen, human-labelled golden set and actual main-agent predictions.", ""]
    if not failures:
        lines.append("No prioritized failure candidates were observed.")
    for index, failure in enumerate(failures, start=1):
        lines.extend([
            f"## Failure {index}: {failure['example_id']}",
            f"- Customer message: {failure['customer_message']}",
            f"- Human intent / predicted intent: `{failure['human_intent']}` / `{failure['predicted_intent']}`",
            f"- Human escalation / predicted action: `{failure['human_escalation']}` / `{failure['predicted_action']}`",
            f"- Reply quality / grounding quality: `{failure['human_reply_quality']}` / `{failure['human_grounding_quality']}`",
            f"- Generated response: {failure['generated_response']}",
            f"- Failure reasons: `{json.dumps(failure['failure_reasons'], sort_keys=True)}`",
            f"- Relevant evidence: `{json.dumps(failure['relevant_evidence'], ensure_ascii=True)}`",
            "- Explanation: The listed mismatch or quality score is the observed evidence; no additional cause is inferred.",
            "- Improvement hypothesis: Review this category in a future development split before changing frozen evaluation settings.",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(results: dict, path: Path) -> None:
    main = results["main_agent"]["intent"]
    baselines = results["baselines"]
    lines = [
        "# Step 4 Evaluation Report", "", "## Dataset", "",
        f"- Golden-set size: **{results['dataset']['size']}**",
        f"- Annotation completeness: **{results['dataset']['complete_rows']} / {results['dataset']['size']}**",
        f"- Frozen golden SHA-256 before/after: `{results['golden_integrity']['before']}` / `{results['golden_integrity']['after']}`",
        "", "## Intent Results", "", f"- Main agent: `{json.dumps(main, indent=2)}`",
        "", "## Escalation Results", "", f"- `{json.dumps(results['main_agent']['escalation'], indent=2)}`",
        "", "## Reply Quality", "", f"- Reply: `{json.dumps(results['quality']['reply'], indent=2)}`",
        f"- Grounding: `{json.dumps(results['quality']['grounding'], indent=2)}`",
        "", "## Baselines", "", "| System | Accuracy | Macro Precision | Macro Recall | Macro F1 |", "|---|---:|---:|---:|---:|",
    ]
    for item in [baselines["trivial"], baselines["simple"], {"system": "Main agent", **main}]:
        lines.append(f"| {item['system']} | {item['accuracy']:.4f} | {item['macro_precision']:.4f} | {item['macro_recall']:.4f} | {item['macro_f1']:.4f} |")
    lines.extend([
        "", "## LLM Judge", "", f"- `{json.dumps(results['judge'], indent=2)}`",
        "", "## Human/Judge Agreement", "", f"- `{json.dumps(results['agreement'], indent=2)}`",
        "", "## Failure Analysis", "", "See `reports/step4_failure_analysis.md` for five prioritized real examples when candidates exist.",
        "", "## Limitations", "", "The golden set is a finite, coverage-oriented sample from one support handle. Human ratings are not model-generated scores. LLM judge results are unavailable unless OPENAI_API_KEY is configured, and any API failures reduce judge coverage.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a manually labelled Step 4 golden set.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--classifier-input", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    arguments = parser.parse_args()
    try:
        rows = load_csv(arguments.golden)
        validate_human_labels(rows, set(INTENT_DESCRIPTIONS))
        golden_before = _file_hash(arguments.golden)
        predictions = _main_predictions(rows, arguments.raw, arguments.classifier_input)
        actual_intents = [row["human_intent"] for row in rows]
        main_intents = [item["model_predicted_intent"] for item in predictions]
        main_actions = [item["model_action"] for item in predictions]
        human_actions = [row["human_escalation"] for row in rows]
        development_rows = load_rows(arguments.classifier_input)
        silver_training_rows = [row for row in development_rows if row.get("inbound", "").lower() == "true"]
        labeled_development_rows = discover_intents(silver_training_rows)
        trivial_predictions = majority_baseline(
            [{"human_intent": row["intent"]} for row in labeled_development_rows],
            rows,
        )
        simple_predictions = simple_tfidf_baseline(silver_training_rows, rows)
        judge_results, judge_summary = _judge_results(predictions)
        evaluation = {
            "dataset": {"path": str(arguments.golden), "size": len(rows), "complete_rows": sum(row.get("annotation_status") == "complete" for row in rows)},
            "seed": 20260910,
            "main_agent": {
                "intent": classification_metrics(actual_intents, main_intents, list(INTENT_DESCRIPTIONS)),
                "escalation": escalation_metrics(human_actions, main_actions),
                "predictions": predictions,
            },
            "baselines": {
                "trivial": _baseline_metrics("Trivial baseline", actual_intents, trivial_predictions),
                "simple": _baseline_metrics("Simple TF-IDF baseline", actual_intents, simple_predictions),
            },
            "quality": {
                "reply": quality_summary([dict(row, model_action=action) for row, action in zip(rows, main_actions)], "human_reply_quality", "model_action"),
                "grounding": quality_summary([dict(row, model_action=action) for row, action in zip(rows, main_actions)], "human_grounding_quality", "model_action"),
            },
            "judge": judge_summary,
            "judge_results": judge_results,
        }
        evaluation["agreement"] = _agreement(rows, judge_results)
        failures = _failure_rows(rows, predictions)
        evaluation["failure_candidates"] = failures
        golden_after = _file_hash(arguments.golden)
        evaluation["golden_integrity"] = {"before": golden_before, "after": golden_after, "unchanged": golden_before == golden_after}
        if golden_before != golden_after:
            raise ValueError("Golden set changed during evaluation; refusing to report results.")
        arguments.results.parent.mkdir(parents=True, exist_ok=True)
        arguments.results.write_text(json.dumps(evaluation, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        _write_report(evaluation, arguments.report)
        _write_failure_report(failures, arguments.failures)
    except (FileNotFoundError, ValueError) as error:
        print(f"Evaluation is not ready: {error}")
        return 1
    print(f"Evaluated {len(rows)} frozen golden rows.")
    print(f"Main intent accuracy: {evaluation['main_agent']['intent']['accuracy']:.4f}")
    print(f"Main intent macro F1: {evaluation['main_agent']['intent']['macro_f1']:.4f}")
    print(f"False auto-handle rate: {evaluation['main_agent']['escalation']['false_auto_handle_rate']:.4f}")
    print(f"LLM judge coverage: {evaluation['judge']['completed']} / {evaluation['judge']['requested']}")
    print(f"Wrote {arguments.results}, {arguments.report}, and {arguments.failures}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())