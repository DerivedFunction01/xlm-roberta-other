from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any

import numpy as np
from datasets import DatasetDict
from sklearn.preprocessing import MultiLabelBinarizer

from building import rows_to_dataset_dict
from cache import load_dataset_cache, save_dataset_cache
from fetch import load_safety_guard_dataset
from paths import PATHS
from text_utils.mutations import MutationConfig, TextMutator

REDACTED_TOKEN = "REDACTED"
SAFETY_CACHE_VERSION = 1


def parse_categories(raw_str: str | None) -> list[str]:
    """Split comma-separated category strings into a cleaned list."""
    if not raw_str or not isinstance(raw_str, str):
        return []
    return [category.strip() for category in raw_str.split(",") if category.strip()]


def _example_from_text(
    *,
    text: str,
    binary_label: int,
    categories: list[str],
    role: str,
    language: str,
    prompt_label_source: str | None,
    response_label_source: str | None,
    tag: str,
    source_id: str,
) -> dict[str, Any]:
    return {
        "text": text,
        "binary_label": binary_label,
        "categories": categories,
        "role": role,
        "language": language,
        "prompt_label_source": prompt_label_source,
        "response_label_source": response_label_source,
        "tag": tag,
        "source_id": source_id,
    }


def row_to_examples(
    row: dict[str, Any],
    *,
    drop_redacted: bool = True,
    include_response: bool = True,
    augment: bool = True,
    mutator: TextMutator | None = None,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    rng = rng or random.Random()
    mutator = mutator or TextMutator(MutationConfig(keep_original=True))

    categories = parse_categories(row.get("violated_categories"))
    language = str(row.get("language", "en") or "en")
    source_id = str(row.get("id", ""))
    tag = str(row.get("tag", "generic") or "generic")
    prompt_label_source = str(row.get("prompt_label_source", "") or "")
    response_label_source = row.get("response_label_source")
    response_label_source_str = str(response_label_source) if response_label_source not in (None, "") else None

    prompt_text = row.get("prompt", "")
    if not (drop_redacted and prompt_text == REDACTED_TOKEN):
        prompt_text = str(prompt_text or "").strip()
        if prompt_text:
            base = _example_from_text(
                text=prompt_text,
                binary_label=1 if row.get("prompt_label") == "unsafe" else 0,
                categories=categories,
                role="prompt",
                language=language,
                prompt_label_source=prompt_label_source or None,
                response_label_source=response_label_source_str,
                tag=tag,
                source_id=source_id,
            )
            examples.append(base)
            if augment:
                for variant in mutator.augment(prompt_text, rng=rng, lang=language):
                    if variant != prompt_text:
                        examples.append({**base, "text": variant})

    if include_response:
        response_text = row.get("response")
        response_label = row.get("response_label")
        if (
            response_text is not None
            and isinstance(response_text, str)
            and response_text.strip()
            and response_label not in (None, "", "null")
        ):
            response_text = response_text.strip()
            base = _example_from_text(
                text=response_text,
                binary_label=1 if response_label == "unsafe" else 0,
                categories=categories,
                role="response",
                language=language,
                prompt_label_source=prompt_label_source or None,
                response_label_source=response_label_source_str,
                tag=tag,
                source_id=source_id,
            )
            examples.append(base)
            if augment:
                for variant in mutator.augment(response_text, rng=rng, lang=language):
                    if variant != response_text:
                        examples.append({**base, "text": variant})

    return examples


def build_flat_examples(
    raw_rows: list[dict[str, Any]],
    *,
    drop_redacted: bool = True,
    augment: bool = True,
    mutator: TextMutator | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    flat_examples: list[dict[str, Any]] = []
    for row in raw_rows:
        flat_examples.extend(
            row_to_examples(
                row,
                drop_redacted=drop_redacted,
                augment=augment,
                mutator=mutator,
                rng=rng,
            )
        )
    return flat_examples


def build_label_vocabulary(examples: list[dict[str, Any]], *, min_label_count: int = 10) -> list[str]:
    category_counter: Counter[str] = Counter()
    for example in examples:
        category_counter.update(example.get("categories", []))
    known_categories = sorted(category for category, count in category_counter.items() if count >= min_label_count)
    if not known_categories:
        raise ValueError("No labels met min_label_count; reduce the threshold or check the source dataset")
    return known_categories


def binarize_examples(examples: list[dict[str, Any]], known_categories: list[str]) -> list[dict[str, Any]]:
    label_set = set(known_categories)
    mlb = MultiLabelBinarizer(classes=known_categories)
    mlb.fit([known_categories])

    for example in examples:
        filtered = [category for category in example.get("categories", []) if category in label_set]
        example["labels"] = mlb.transform([filtered])[0].astype(np.float32).tolist()
    return examples


def build_safety_classifier_dataset(
    *,
    dataset_split: str = "train",
    drop_redacted: bool = True,
    augment: bool = True,
    min_label_count: int = 10,
    val_size: float = 0.05,
    test_size: float = 0.05,
    seed: int = 42,
    force_rebuild: bool = False,
    cache_dir: str | None = None,
    cache_meta_path: str | None = None,
) -> tuple[DatasetDict, list[str], dict[str, int], dict[int, str], dict[str, Any]]:
    cache_dir = cache_dir or PATHS["safety_classifier"]["cache_dir"]
    cache_meta_path = cache_meta_path or PATHS["safety_classifier"]["cache_meta"]
    expected_meta = {
        "cache_version": SAFETY_CACHE_VERSION,
        "dataset_name": "nvidia/Nemotron-Safety-Guard-Dataset-v3",
        "dataset_split": dataset_split,
        "drop_redacted": drop_redacted,
        "augment": augment,
        "min_label_count": min_label_count,
        "val_size": val_size,
        "test_size": test_size,
        "seed": seed,
    }

    if not force_rebuild:
        cached = load_dataset_cache(cache_dir, meta_path=cache_meta_path, expected_meta=expected_meta)
        if cached is not None:
            with open(cache_meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            known_categories = list(meta.get("known_categories", []))
            label2id = dict(meta.get("label2id", {}))
            id2label = {int(k): v for k, v in meta.get("id2label", {}).items()} if isinstance(meta.get("id2label"), dict) else {}
            return cached, known_categories, label2id, id2label, meta

    raw = load_safety_guard_dataset(split=dataset_split)
    raw_rows = [dict(row) for row in raw]
    flat_examples = build_flat_examples(
        raw_rows,
        drop_redacted=drop_redacted,
        augment=augment,
        seed=seed,
    )

    known_categories = build_label_vocabulary(flat_examples, min_label_count=min_label_count)
    flat_examples = binarize_examples(flat_examples, known_categories)
    label2id = {category: idx for idx, category in enumerate(known_categories)}
    id2label = {idx: category for category, idx in label2id.items()}

    columns = list(flat_examples[0].keys())
    dataset = rows_to_dataset_dict(
        flat_examples,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        columns=columns,
    )

    meta = {
        **expected_meta,
        "known_categories": known_categories,
        "label2id": label2id,
        "id2label": id2label,
        "num_examples": len(flat_examples),
        "split_sizes": {split_name: len(split) for split_name, split in dataset.items()},
    }
    save_dataset_cache(dataset, cache_dir, meta_path=cache_meta_path, meta=meta)
    return dataset, known_categories, label2id, id2label, meta
