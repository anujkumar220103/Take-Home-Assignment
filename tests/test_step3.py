import tempfile
import unittest
from pathlib import Path

from src.agent.support_agent import (
    answer_query,
    build_cases,
    build_retrieval_index,
    choose_action,
    is_account_specific_request,
    load_rows,
    retrieve_cases,
)


CSV_CONTENT = """tweet_id,author_id,inbound,text,response_tweet_id,in_response_to_tweet_id
1,customer-1,True,Where is my package?,,2
2,AmazonHelp,False,Please check the tracking page for the delivery status.,3,1
3,AmazonHelp,False,Let us know if the package does not arrive.,,1
4,customer-2,True,I forgot my password and cannot login.,,5
5,AmazonHelp,False,Please use the password reset flow for account access.,,4
"""


class Step3Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.input_path = Path(self.directory.name) / "sample.csv"
        self.input_path.write_text(CSV_CONTENT, encoding="utf-8")
        rows = load_rows(self.input_path)
        self.customer_rows = [row for row in rows if row["inbound"] == "True"]
        self.cases = build_cases(self.input_path)
        self.index = build_retrieval_index(self.cases)

    def tearDown(self):
        self.directory.cleanup()

    def test_index_contains_only_linked_brand_cases(self):
        self.assertEqual(len(self.cases), 3)
        self.assertEqual(self.cases[0]["customer_tweet_id"], "1")
        self.assertEqual(
            {case["response_tweet_id"] for case in self.cases}, {"2", "3", "5"}
        )

    def test_answer_retrieves_grounded_case_and_explains_decision(self):
        result = answer_query("Where is my package?", self.index, self.customer_rows)
        self.assertEqual(result["decision"]["action"], "handle")
        self.assertTrue(result["decision"]["reason"])
        self.assertEqual(result["reply"]["mode"], "mock")
        self.assertIn("cannot access", result["reply"]["text"])
        self.assertNotEqual(
            result["reply"]["text"], result["retrieved_cases"][0]["response_text"]
        )
        self.assertGreater(result["retrieved_cases"][0]["retrieval_score"], 0)
        self.assertIn("input_message", result)
        self.assertIn("evidence_strength", result)
        self.assertIn("top_text_similarity", result["decision"])

    def test_weak_retrieval_escalates_with_reason(self):
        decision = choose_action(
            "Tell me something unrelated",
            {"confidence": 0.9},
            [{"retrieval_score": 0.1, "text_similarity": 0.0}],
        )
        self.assertEqual(decision["action"], "escalate")
        self.assertIn("below", decision["reason"])

    def test_low_intent_confidence_escalates(self):
        decision = choose_action(
            "I need help",
            {"confidence": 0.2},
            [{"retrieval_score": 0.8, "text_similarity": 0.8}],
        )
        self.assertEqual(decision["action"], "escalate")
        self.assertIn("confidence", decision["reason"])

    def test_account_specific_request_does_not_claim_access(self):
        self.assertTrue(is_account_specific_request("Where exactly is my order?"))
        result = answer_query(
            "Where exactly is my order?", self.index, self.customer_rows
        )
        self.assertEqual(result["decision"]["action"], "escalate")
        self.assertIn("cannot access", result["reply"]["text"])

    def test_temporal_and_conversation_filters_exclude_cases(self):
        filtered_cases = retrieve_cases(
            "Where is my package?",
            self.index,
            predicted_intent="delivery_tracking",
            before_created_at="Wed Jan 03 00:00:00 +0000 2024",
            excluded_conversation_group="1",
        )
        self.assertEqual(filtered_cases, [])


if __name__ == "__main__":
    unittest.main()