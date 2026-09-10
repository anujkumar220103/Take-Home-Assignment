import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.evaluation.baselines import majority_baseline, simple_tfidf_baseline
from src.evaluation.metrics import classification_metrics, escalation_metrics, validate_human_labels
from src.evaluation.prepare_golden import GOLDEN_FIELDS, prepare_golden_set
from src.evaluation.judge import judge_response, parse_judge_json
from src.evaluation.agreement import cohen_kappa, rating_agreement
from src.evaluation.annotate_golden import (
    ESCALATION_OPTIONS,
    QUALITY_SCORES,
    annotate_file,
    collect_annotation,
    find_next_incomplete,
    load_golden_rows,
    render_example,
    save_annotation,
    validate_annotation,
)


class EvaluationTests(unittest.TestCase):
    def test_golden_preparation_is_deterministic_and_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.csv"
            rows = [
                {
                    "tweet_id": str(index), "author_id": "AmazonHelp", "inbound": "False",
                    "text": "reply", "response_tweet_id": "", "in_response_to_tweet_id": str(index + 100),
                }
                for index in range(1, 5)
            ] + [
                {
                    "tweet_id": str(index + 100), "author_id": "customer", "inbound": "True",
                    "text": "where is my package", "response_tweet_id": "", "in_response_to_tweet_id": "",
                }
                for index in range(1, 5)
            ]
            with raw.open("w", encoding="utf-8", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            output = root / "golden.csv"
            prepare_golden_set(raw, output, root / "manifest.json", size=2, seed=4)
            with output.open(encoding="utf-8") as prepared_file:
                prepared = list(csv.DictReader(prepared_file))
            self.assertEqual(list(prepared[0]), list(GOLDEN_FIELDS))
            self.assertTrue(all(not row["human_intent"] for row in prepared))
            self.assertTrue(all(row["silver_sampling_group"] for row in prepared))
            self.assertNotEqual(
                prepared[0]["silver_sampling_group"], prepared[0]["human_intent"]
            )

    def test_missing_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_human_labels([{"example_id": "1", "human_intent": ""}], {"other_or_unclear"})

    def test_metrics_and_confusion_matrix(self):
        result = classification_metrics(["a", "b", "a"], ["a", "a", "b"], ["a", "b"])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 0]])
        self.assertAlmostEqual(result["accuracy"], 1 / 3)

    def test_escalation_metrics_distinguish_false_actions(self):
        result = escalation_metrics(
            ["AUTO_HANDLE", "ESCALATE_TO_HUMAN", "ESCALATE_TO_HUMAN"],
            ["ESCALATE_TO_HUMAN", "AUTO_HANDLE", "ESCALATE_TO_HUMAN"],
        )
        self.assertEqual(result["false_auto_handle_count"], 1)
        self.assertEqual(result["false_escalation_count"], 1)
        self.assertAlmostEqual(result["false_auto_handle_rate"], 0.5)
        self.assertAlmostEqual(result["false_escalation_rate"], 1.0)

    def test_baselines_use_development_rows_only(self):
        train_rows = [
            {"text": "track my package shipment", "intent": "delivery_tracking"},
            {"text": "my package is late", "intent": "delivery_delay"},
            {"text": "track my package status", "intent": "delivery_tracking"},
        ]
        evaluation_rows = [{"input_message": "track my package status"}]
        self.assertEqual(majority_baseline(train_rows, evaluation_rows), ["delivery_tracking"])
        self.assertEqual(simple_tfidf_baseline(train_rows, evaluation_rows), ["delivery_tracking"])

    def test_judge_is_explicitly_unavailable_without_key(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            result = judge_response("Where is my order?", "Please check tracking.", [])
        self.assertEqual(result["status"], "unavailable")

    def test_rating_agreement_reports_joint_sample(self):
        result = rating_agreement([0, 0.5, 1], [0, 0.75, 1])
        self.assertEqual(result["joint_count"], 3)
        self.assertAlmostEqual(result["exact_agreement"], 2 / 3)

    def test_kappa_and_judge_json(self):
        self.assertAlmostEqual(cohen_kappa(["a", "b"], ["a", "b"]), 1.0)
        parsed = parse_judge_json('{"helpfulness": 1, "groundedness": 0.75, "safety": 1, "rationale": "supported"}')
        self.assertEqual(parsed["groundedness"], 0.75)

    def test_annotation_validation_rejects_invalid_values(self):
        valid = {
            "human_intent": "delivery_tracking",
            "human_escalation": ESCALATION_OPTIONS[0],
            "human_reply_quality": QUALITY_SCORES[-1],
            "human_grounding_quality": QUALITY_SCORES[-1],
            "annotation_status": "complete",
        }
        validate_annotation(valid)
        invalid = dict(valid, human_intent="silver_label")
        with self.assertRaises(ValueError):
            validate_annotation(invalid)

    def test_annotation_display_excludes_biasing_fields(self):
        row = {
            "example_id": "golden-0001",
            "input_message": "Where is my order?",
            "historical_response": "Please check the tracking page.",
            "silver_sampling_group": "delivery_tracking",
            "human_intent": "",
            "model_predicted_intent": "delivery_delay",
        }
        rendered = render_example(row, 0, 1)
        self.assertIn("Where is my order?", rendered)
        self.assertIn("Please check the tracking page.", rendered)
        self.assertNotIn("delivery_tracking", rendered)
        self.assertNotIn("model_predicted_intent", rendered)

    def test_save_annotation_preserves_rows_and_creates_one_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "golden.csv"
            backup = root / "golden.backup.csv"
            fieldnames = [
                "example_id",
                "input_message",
                "silver_sampling_group",
                "human_intent",
                "human_escalation",
                "human_reply_quality",
                "human_grounding_quality",
                "annotator_notes",
                "annotation_status",
            ]
            rows = [
                {
                    "example_id": "golden-0001",
                    "input_message": "A",
                    "silver_sampling_group": "x",
                    "human_intent": "",
                    "human_escalation": "",
                    "human_reply_quality": "",
                    "human_grounding_quality": "",
                    "annotator_notes": "",
                    "annotation_status": "needs_human_label",
                },
                {
                    "example_id": "golden-0002",
                    "input_message": "B",
                    "silver_sampling_group": "y",
                    "human_intent": "delivery_delay",
                    "human_escalation": "AUTO_HANDLE",
                    "human_reply_quality": "1",
                    "human_grounding_quality": "1",
                    "annotator_notes": "done",
                    "annotation_status": "complete",
                },
            ]
            with path.open("w", encoding="utf-8", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            annotation = {
                "human_intent": "delivery_tracking",
                "human_escalation": "AUTO_HANDLE",
                "human_reply_quality": "1",
                "human_grounding_quality": "0.75",
                "annotator_notes": "clear",
                "annotation_status": "complete",
            }
            save_annotation(path, fieldnames, rows, 0, annotation, backup)
            save_annotation(path, fieldnames, rows, 0, annotation, backup)
            _, saved_rows = load_golden_rows(path)
            self.assertEqual(saved_rows[0]["human_intent"], "delivery_tracking")
            self.assertEqual(saved_rows[1]["human_intent"], "delivery_delay")
            self.assertTrue(backup.exists())
            self.assertEqual(backup.read_text(encoding="utf-8").count("golden-0001"), 1)

    def test_annotation_resumes_at_first_incomplete_row(self):
        rows = [
            {"annotation_status": "complete"},
            {"annotation_status": "needs_human_label"},
            {"annotation_status": "needs_human_label"},
        ]
        self.assertEqual(find_next_incomplete(rows), 1)


if __name__ == "__main__":
    unittest.main()