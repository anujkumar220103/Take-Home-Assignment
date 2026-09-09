import csv
import tempfile
import unittest
from pathlib import Path

from src.data.analysis import analyze_conversations, create_sample
from src.data.pipeline import run_step1
from src.data.validation import validate_input


CSV_CONTENT = """tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id
1,brand-a,false,Mon Jan 01 00:00:00 +0000 2024,Hello,2,
2,customer-1,true,Mon Jan 01 00:01:00 +0000 2024,Need help,,1
3,brand-a,false,Mon Jan 01 00:02:00 +0000 2024,Reply,4,2
4,customer-1,true,Mon Jan 01 00:03:00 +0000 2024,Thanks,,3
"""


class Step1Tests(unittest.TestCase):
    def test_validation_and_conversation_links(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "twcs.csv"
            input_path.write_text(CSV_CONTENT, encoding="utf-8")
            validation = validate_input(input_path)
            conversation = analyze_conversations(input_path)

            self.assertTrue(validation.is_valid)
            self.assertEqual(validation.row_count, 4)
            self.assertEqual(conversation["reply_edge_count"], 3)
            self.assertEqual(conversation["edges_with_missing_parent"], 0)

    def test_sampling_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "twcs.csv"
            first_sample = Path(directory) / "first.csv"
            second_sample = Path(directory) / "second.csv"
            input_path.write_text(CSV_CONTENT, encoding="utf-8")
            first_result = create_sample(
                input_path, first_sample, sample_size=2, seed=7, selected_brand_id="brand-a"
            )
            second_result = create_sample(
                input_path, second_sample, sample_size=2, seed=7, selected_brand_id="brand-a"
            )
            self.assertEqual(first_sample.read_text(), second_sample.read_text())
            self.assertEqual(first_result, second_result)
            self.assertEqual(first_result["customer_parent_rows_in_sample"], 1)

    def test_pipeline_creates_step1_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "twcs.csv"
            input_path.write_text(CSV_CONTENT, encoding="utf-8")

            results = run_step1(
                input_path=input_path,
                reports_dir=root / "reports",
                processed_path=root / "processed.csv",
                sample_path=root / "sample.csv",
                sample_size=2,
                seed=7,
            )

            self.assertEqual(results["profiling"]["row_count"], 4)
            self.assertEqual(results["recommendation"]["recommended_author_id"], "brand-a")
            self.assertFalse(results["suitability"]["sufficient_for_150_to_250_example_set"])
            self.assertTrue((root / "reports" / "step1_results.json").exists())
            self.assertTrue((root / "reports" / "step1_summary.md").exists())
            self.assertTrue((root / "processed.csv").exists())
            self.assertTrue((root / "sample.csv").exists())
            with (root / "sample.csv").open(encoding="utf-8") as sample_file:
                sample_rows = list(csv.DictReader(sample_file))
            self.assertEqual(
                {row["author_id"] for row in sample_rows}, {"brand-a", "customer-1"}
            )


if __name__ == "__main__":
    unittest.main()
