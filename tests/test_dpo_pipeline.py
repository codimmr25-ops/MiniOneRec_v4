import json
import os
import sys
import tempfile
import unittest

import pandas as pd

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data import DPODataset
from build_dpo_pairs import build_dpo_records
from build_preference_data import build_preference_records


class TestPreferenceBuilder(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            [
                {
                    "user_id": "A1",
                    "history_item_id": "[1, 2]",
                    "history_item_title": "['Alpha Drill', 'Beta Clamp']",
                    "history_item_sid": "['<a_1><b_1><c_1>', '<a_2><b_2><c_2>']",
                    "item_id": 3,
                    "item_title": "Gamma Press",
                    "item_sid": "<a_3><b_3><c_3>",
                },
                {
                    "user_id": "A2",
                    "history_item_id": "[]",
                    "history_item_title": "[]",
                    "history_item_sid": "[]",
                    "item_id": 4,
                    "item_title": "Delta Gauge",
                    "item_sid": "<a_4><b_4><c_4>",
                },
            ]
        )
        self.item_meta = {
            "3": {"title": "Gamma Press", "description": "['High precision press for lab work.']"},
            "4": {"title": "Delta Gauge", "description": ""},
        }
        self.indices = {
            "1": ["<a_1>", "<b_1>", "<c_1>"],
            "2": ["<a_2>", "<b_2>", "<c_2>"],
            "3": ["<a_3>", "<b_3>", "<c_3>"],
            "4": ["<a_4>", "<b_4>", "<c_4>"],
        }

    def test_build_preference_records_retries_and_skips_short_history(self):
        state = {"calls": 0}

        def flaky_llm(model_name, prompt_list, max_tokens, api_info):
            state["calls"] += 1
            if state["calls"] == 1:
                raise TimeoutError("temporary timeout")
            return ["The user prefers precision industrial tools."]

        records, failures = build_preference_records(
            dataframe=self.df,
            item_meta=self.item_meta,
            sid_index=self.indices,
            llm_fn=flaky_llm,
            provider="minimax",
            llm_model="MiniMax-M2.5",
            dataset_label="industrial and scientific items",
            batch_size=1,
            max_attempts=2,
            preference_mode="target_conditioned_debug",
        )

        self.assertEqual(state["calls"], 2)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["user"], "A1")
        self.assertEqual(records[0]["target_item_sid"], "<a_3><b_3><c_3>")
        self.assertEqual(records[0]["preference_source"], "minimax:MiniMax-M2.5:target_conditioned_debug")
        self.assertEqual(records[0]["user_preference"], "The user prefers precision industrial tools.")
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["user"], "A2")
        self.assertEqual(failures[0]["reason"], "insufficient_history")


class TestDpoPairBuilder(unittest.TestCase):
    def test_build_dpo_records_prefers_model_hard_negative(self):
        preference_records = [
            {
                "user": "A1",
                "split": "train",
                "history_item_sid": ["<a_1><b_1><c_1>", "<a_2><b_2><c_2>"],
                "target_item_sid": "<a_3><b_3><c_3>",
                "user_preference": "Prefers precise tools.",
            }
        ]

        def candidate_fn(prompts):
            self.assertEqual(len(prompts), 1)
            return [["<a_3><b_3><c_3>", "<a_9><b_9><c_9>"]]

        records, failures = build_dpo_records(
            preference_records=preference_records,
            candidate_generator=candidate_fn,
            fallback_sids=["<a_7><b_7><c_7>", "<a_9><b_9><c_9>"],
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["chosen"], "<a_3><b_3><c_3>\n")
        self.assertEqual(records[0]["rejected"], "<a_9><b_9><c_9>\n")
        self.assertEqual(records[0]["negative_source"], "model_hard_negative")
        self.assertEqual(records[0]["rejected_rank"], 2)
        self.assertIn("The user has interacted with items", records[0]["prompt"])
        self.assertEqual(failures, [])

    def test_build_dpo_records_falls_back_when_model_has_no_wrong_candidate(self):
        preference_records = [
            {
                "user": "A2",
                "split": "valid",
                "history_item_sid": ["<a_1><b_1><c_1>"],
                "target_item_sid": "<a_3><b_3><c_3>",
                "user_preference": "Prefers measurement tools.",
            }
        ]

        def candidate_fn(prompts):
            return [["<a_3><b_3><c_3>", "<a_3><b_3><c_3>"]]

        records, failures = build_dpo_records(
            preference_records=preference_records,
            candidate_generator=candidate_fn,
            fallback_sids=["<a_3><b_3><c_3>", "<a_8><b_8><c_8>"],
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["rejected"], "<a_8><b_8><c_8>\n")
        self.assertEqual(records[0]["negative_source"], "random_fallback")
        self.assertEqual(failures, [])


class TestDpoDataset(unittest.TestCase):
    def test_dpo_dataset_normalizes_completions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "pairs.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "prompt": "### User Input:\nPredict.\n\n### Response:\n",
                            "chosen": " <a_3><b_3><c_3> ",
                            "rejected": "<a_8><b_8><c_8>\n",
                        }
                    )
                    + "\n"
                )

            dataset = DPODataset(jsonl_path)
            sample = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(sample["prompt"], "### User Input:\nPredict.\n\n### Response:\n")
            self.assertEqual(sample["chosen"], "<a_3><b_3><c_3>\n")
            self.assertEqual(sample["rejected"], "<a_8><b_8><c_8>\n")


if __name__ == "__main__":
    unittest.main()
