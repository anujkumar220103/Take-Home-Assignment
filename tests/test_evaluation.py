import csv
import tempfile
import unittest
from pathlib import Path

from src.evaluation.metrics import classification_metrics, validate_human_labels
from src.evaluation.prepare_golden import GOLDEN_FIELDS, prepare_golden_set
from src.evaluation.judge import parse_judge_json
from src.evaluation.agreement import cohen_kappa


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

    def test_kappa_and_judge_json(self):
        self.assertAlmostEqual(cohen_kappa(["a", "b"], ["a", "b"]), 1.0)
        parsed = parse_judge_json('{"helpfulness": 1, "groundedness": 0.75, "safety": 1, "rationale": "supported"}')
        self.assertEqual(parsed["groundedness"], 0.75)


if __name__ == "__main__":
    unittest.main()