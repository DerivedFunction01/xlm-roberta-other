from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from datasets import DatasetDict

from shared.building import rows_to_dataset_dict, split_indices
from shared.cache import load_dataset_cache, save_dataset_cache
from safety.data import REDACTED_TOKEN, build_flat_examples, build_label_vocabulary, row_to_examples


class BuildingAndSafetyDataTests(unittest.TestCase):
    def test_split_indices_respects_sizes(self) -> None:
        split_map = split_indices(20, val_size=0.25, test_size=0.25, seed=7)
        self.assertEqual(len(split_map["train"]), 10)
        self.assertEqual(len(split_map["val"]), 5)
        self.assertEqual(len(split_map["test"]), 5)
        self.assertEqual(len(set(split_map["train"] + split_map["val"] + split_map["test"])), 20)

    def test_cache_round_trip(self) -> None:
        rows = [
            {"text": "a", "labels": [1.0, 0.0], "binary_label": 0},
            {"text": "b", "labels": [0.0, 1.0], "binary_label": 1},
        ]
        dataset = rows_to_dataset_dict(rows, val_size=0.5, test_size=0.0, seed=13)
        self.assertIsInstance(dataset, DatasetDict)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            meta_path = Path(tmpdir) / "cache.meta.json"
            meta = {"cache_version": 1, "name": "unit-test"}
            save_dataset_cache(dataset, cache_dir, meta_path=meta_path, meta=meta)

            loaded = load_dataset_cache(cache_dir, meta_path=meta_path, expected_meta=meta)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded["train"]), len(dataset["train"]))
            self.assertEqual(len(loaded["val"]), len(dataset["val"]))
            self.assertEqual(len(loaded["test"]), len(dataset["test"]))

    def test_row_to_examples_and_flat_examples(self) -> None:
        raw_rows = [
            {
                "id": "abc",
                "prompt": REDACTED_TOKEN,
                "response": "Stay safe.",
                "prompt_label": "unsafe",
                "response_label": "safe",
                "violated_categories": "self-harm, harassment",
                "prompt_label_source": "human",
                "response_label_source": "llm_jury",
                "tag": "generic",
                "language": "en",
            }
        ]
        examples = row_to_examples(raw_rows[0], drop_redacted=True, augment=False)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["role"], "response")
        self.assertEqual(examples[0]["binary_label"], 0)

        flat = build_flat_examples(raw_rows, drop_redacted=True, augment=False)
        self.assertEqual(len(flat), 1)
        self.assertEqual(flat[0]["text"], "Stay safe.")

        vocab = build_label_vocabulary(
            [
                {"categories": ["self-harm", "harassment"]},
                {"categories": ["harassment"]},
                {"categories": ["harassment"]},
            ],
            min_label_count=2,
        )
        self.assertEqual(vocab, ["harassment"])


if __name__ == "__main__":
    unittest.main()
