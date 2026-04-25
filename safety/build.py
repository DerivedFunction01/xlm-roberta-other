from __future__ import annotations

import json
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any

import numpy as np
from datasets import DatasetDict
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from shared.building import rows_to_dataset_dict
from shared.cache import ensure_cache_meta, load_dataset_cache, save_dataset_cache
from shared.fetch import load_safety_guard_dataset
from shared.paths import PATHS
from shared.tokenization import tokenize_dataset_dict
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


def _chunked(seq: list[tuple[int, dict[str, Any]]], chunk_size: int) -> list[list[tuple[int, dict[str, Any]]]]:
    return [seq[i : i + chunk_size] for i in range(0, len(seq), chunk_size)]


def _build_flat_examples_chunk(
    chunk: list[tuple[int, dict[str, Any]]],
    *,
    drop_redacted: bool,
    augment: bool,
    mutator: TextMutator | None,
    seed: int,
) -> list[dict[str, Any]]:
    chunk_examples: list[dict[str, Any]] = []
    for row_index, row in chunk:
        rng = random.Random(seed + row_index)
        chunk_examples.extend(
            row_to_examples(
                row,
                drop_redacted=drop_redacted,
                augment=augment,
                mutator=mutator,
                rng=rng,
            )
        )
    return chunk_examples


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
    indexed_rows = list(enumerate(raw_rows))
    if len(indexed_rows) < 1000 or (os.cpu_count() or 1) <= 1:
        rng = random.Random(seed)
        flat_examples: list[dict[str, Any]] = []
        for _, row in tqdm(indexed_rows, desc="Building safety examples", unit="row"):
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

    workers = max(1, (os.cpu_count() or 2) - 1)
    chunk_size = max(1, len(indexed_rows) // (workers * 4))
    chunks = _chunked(indexed_rows, chunk_size)
    flat_examples = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        chunk_worker = partial(
            _build_flat_examples_chunk,
            drop_redacted=drop_redacted,
            augment=augment,
            mutator=mutator,
            seed=seed,
        )
        for chunk_examples in tqdm(
            executor.map(chunk_worker, chunks, chunksize=1),
            total=len(chunks),
            desc="Building safety examples",
            unit="chunk",
        ):
            flat_examples.extend(chunk_examples)
    return flat_examples


def build_label_vocabulary(examples: list[dict[str, Any]], *, min_label_count: int = 10) -> list[str]:
    category_counter: Counter[str] = Counter()
    for example in tqdm(examples, desc="Counting safety labels", unit="example"):
        category_counter.update(example.get("categories", []))
    known_categories = sorted(category for category, count in category_counter.items() if count >= min_label_count)
    if not known_categories:
        raise ValueError("No labels met min_label_count; reduce the threshold or check the source dataset")
    return known_categories


def binarize_examples(examples: list[dict[str, Any]], known_categories: list[str]) -> list[dict[str, Any]]:
    label_set = set(known_categories)
    mlb = MultiLabelBinarizer(classes=known_categories)
    mlb.fit([known_categories])

    for example in tqdm(examples, desc="Binarizing safety labels", unit="example"):
        filtered = [category for category in example.get("categories", []) if category in label_set]
        example["labels"] = mlb.transform([filtered])[0].astype(np.float32).tolist()
    return examples


def summarize_label_distribution(
    examples: list[dict[str, Any]],
    known_categories: list[str],
) -> dict[str, Any]:
    binary_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()

    for example in tqdm(examples, desc="Summarizing safety labels", unit="example"):
        binary_label = "unsafe" if int(example.get("binary_label", 0)) == 1 else "safe"
        binary_counter[binary_label] += 1
        category_counter.update(category for category in example.get("categories", []) if category in known_categories)

    total_examples = len(examples)
    return {
        "num_examples": total_examples,
        "binary_label_counts": dict(binary_counter),
        "binary_label_rates": {
            label: (count / total_examples) if total_examples else 0.0
            for label, count in binary_counter.items()
        },
        "category_positive_counts": dict(category_counter),
        "category_positive_rates": {
            category: (category_counter.get(category, 0) / total_examples) if total_examples else 0.0
            for category in known_categories
        },
    }


def summarize_dataset_label_distribution(
    dataset: DatasetDict,
    known_categories: list[str],
) -> dict[str, Any]:
    binary_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    total_examples = 0

    for split_name, split in tqdm(dataset.items(), desc="Summarizing safety label distribution", unit="split"):
        for example in tqdm(split, desc=f"Counting {split_name}", unit="example", leave=False):
            total_examples += 1
            binary_label = "unsafe" if int(example.get("binary_label", 0)) == 1 else "safe"
            binary_counter[binary_label] += 1
            category_values = example.get("categories", [])
            if isinstance(category_values, list):
                category_counter.update(category for category in category_values if category in known_categories)

    return {
        "num_examples": total_examples,
        "binary_label_counts": dict(binary_counter),
        "binary_label_rates": {
            label: (count / total_examples) if total_examples else 0.0
            for label, count in binary_counter.items()
        },
        "category_positive_counts": dict(category_counter),
        "category_positive_rates": {
            category: (category_counter.get(category, 0) / total_examples) if total_examples else 0.0
            for category in known_categories
        },
    }


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
    cache_dir: str = PATHS["safety"]["raw_cache_dir"],
    cache_meta_path: str = PATHS["safety"]["raw_cache_meta"],
) -> tuple[DatasetDict, list[str], dict[str, int], dict[int, str], dict[str, Any]]:
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
            if "binary_label_counts" not in meta or "category_positive_counts" not in meta:
                label_distribution = summarize_dataset_label_distribution(cached, known_categories)
                meta = {**meta, **label_distribution}
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
    label_distribution = summarize_label_distribution(flat_examples, known_categories)
    label2id = {category: idx for idx, category in enumerate(known_categories)}
    id2label = {idx: category for category, idx in label2id.items()}
    binary_label2id = {"safe": 0, "unsafe": 1}
    binary_id2label = {idx: label for label, idx in binary_label2id.items()}

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
        "binary_label2id": binary_label2id,
        "binary_id2label": binary_id2label,
        **label_distribution,
        "split_sizes": {split_name: len(split) for split_name, split in dataset.items()},
    }
    save_dataset_cache(dataset, cache_dir, meta_path=cache_meta_path, meta=meta)
    return dataset, known_categories, label2id, id2label, meta


def build_and_cache_safety_dataset(
    *,
    model_name: str,
    max_length: int,
    dataset_split: str = "train",
    drop_redacted: bool = True,
    augment: bool = True,
    min_label_count: int = 10,
    val_size: float = 0.05,
    test_size: float = 0.05,
    seed: int = 42,
    force_rebuild: bool = False,
) -> tuple[DatasetDict, list[str], dict[str, int], dict[int, str], dict[str, Any]]:
    raw_cache_dir = PATHS["safety"]["raw_cache_dir"]
    raw_cache_meta = PATHS["safety"]["raw_cache_meta"]
    tokenized_cache_dir = PATHS["safety"]["tokenized_cache_dir"]
    tokenized_cache_meta = PATHS["safety"]["tokenized_cache_meta"]
    binary_label2id = {"safe": 0, "unsafe": 1}
    binary_id2label = {0: "safe", 1: "unsafe"}

    raw_dataset, known_categories, label2id, id2label, raw_meta_state = build_safety_classifier_dataset(
        dataset_split=dataset_split,
        drop_redacted=drop_redacted,
        augment=augment,
        min_label_count=min_label_count,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        force_rebuild=force_rebuild,
        cache_dir=raw_cache_dir,
        cache_meta_path=raw_cache_meta,
    )

    raw_meta = {
        "cache_version": SAFETY_CACHE_VERSION,
        "dataset_kind": "safety_raw",
        "dataset_split": dataset_split,
        "drop_redacted": drop_redacted,
        "augment": augment,
        "min_label_count": min_label_count,
        "val_size": val_size,
        "test_size": test_size,
        "seed": seed,
        "binary_label2id": binary_label2id,
        "binary_id2label": binary_id2label,
    }

    tokenized_meta = {
        "cache_version": SAFETY_CACHE_VERSION,
        "dataset_kind": "safety_tokenized",
        "model_name": model_name,
        "max_length": max_length,
        "dataset_split": dataset_split,
        "drop_redacted": drop_redacted,
        "augment": augment,
        "min_label_count": min_label_count,
        "val_size": val_size,
        "test_size": test_size,
        "seed": seed,
        "label2id": label2id,
        "id2label": id2label,
        "binary_label2id": binary_label2id,
        "binary_id2label": binary_id2label,
    }

    if not force_rebuild:
        ensure_cache_meta(tokenized_cache_dir, meta_path=tokenized_cache_meta, meta=tokenized_meta)
        loaded = load_dataset_cache(tokenized_cache_dir, meta_path=tokenized_cache_meta, expected_meta=tokenized_meta)
        if loaded is not None:
            with open(tokenized_cache_meta, encoding="utf-8") as f:
                meta = json.load(f)
            known_categories = list(meta.get("known_categories", []))
            label2id = dict(meta.get("label2id", {}))
            id2label = {int(k): v for k, v in meta.get("id2label", {}).items()} if isinstance(meta.get("id2label"), dict) else {}
            if "binary_label_counts" not in meta or "category_positive_counts" not in meta:
                label_distribution = summarize_dataset_label_distribution(raw_dataset, known_categories)
                meta = {**meta, **label_distribution}
            return loaded, known_categories, label2id, id2label, meta

    tokenized = tokenize_dataset_dict(
        raw_dataset,
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        kind="text",
        max_length=max_length,
        text_columns=("text",),
    )
    tokenized = DatasetDict(
        {
            split_name: split.remove_columns(
                [col for col in split.column_names if col not in {"input_ids", "attention_mask", "labels", "binary_label"}]
            )
            for split_name, split in tokenized.items()
        }
    )
    label_distribution = summarize_dataset_label_distribution(raw_dataset, known_categories)
    tokenized_meta = {
        **tokenized_meta,
        "known_categories": known_categories,
        "label2id": label2id,
        "id2label": id2label,
        "binary_label2id": binary_label2id,
        "binary_id2label": binary_id2label,
        **label_distribution,
        "split_sizes": {split_name: len(split) for split_name, split in tokenized.items()},
    }
    save_dataset_cache(tokenized, tokenized_cache_dir, meta_path=tokenized_cache_meta, meta=tokenized_meta)
    return tokenized, known_categories, label2id, id2label, tokenized_meta
