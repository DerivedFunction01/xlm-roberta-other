from __future__ import annotations

import random
from typing import Any

from datasets import Dataset, DatasetDict, Value, concatenate_datasets
from transformers import AutoTokenizer

from shared.cache import save_dataset_cache, load_dataset_cache
from shared.paths import PATHS
from shared.tokenization import tokenize_dataset_dict
from shared.fetch import load_mnli_dataset, load_xnli_dataset

LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def _select(ds, n, *, seed: int):
    """Shuffle + select n rows; pass None to keep all."""
    return ds.shuffle(seed=seed).select(range(n)) if n else ds.shuffle(seed=seed)


def _mnli_schema(batch):
    return {"premise": batch["premise"], "hypothesis": batch["hypothesis"], "label": batch["label"]}


def flatten_xnli(batch, indices, langs: list[str]):
    """Expand a batch of multi-lang rows into one row per language."""
    premises = []
    hypotheses = []
    labels = []
    langs_out = []
    source_ids = []

    for premise_row, hypothesis_row, label, source_id in zip(
        batch["premise"],
        batch["hypothesis"],
        batch["label"],
        indices,
    ):
        hyp_translations = dict(zip(hypothesis_row["language"], hypothesis_row["translation"]))
        for lang in langs:
            premises.append(premise_row[lang])
            hypotheses.append(hyp_translations[lang])
            labels.append(label)
            langs_out.append(lang)
            source_ids.append(source_id)

    return {
        "premise": premises,
        "hypothesis": hypotheses,
        "label": labels,
        "lang": langs_out,
        "source_id": source_ids,
    }


def build_xnli_datasets(
    flat_ds,
    pool_size,
    val_size,
    same_pct,
    cross_pct,
    langs,
    *,
    seed: int = 42,
):
    """Returns (train_dataset, val_dataset) as HF Datasets."""
    rng = random.Random(seed)

    if pool_size is None:
        pool_size = len(flat_ds)
    if val_size is None:
        raise ValueError("xnli_val_size must be set to an integer")

    n_same = int(pool_size * same_pct / 100)
    n_cross = int(pool_size * cross_pct / 100)

    by_source: dict[int, list[dict[str, Any]]] = {}
    for row in flat_ds:
        by_source.setdefault(int(row["source_id"]), []).append(row)

    by_example = list(by_source.values())
    if any(len(variants) != len(langs) for variants in by_example):
        raise ValueError("Unexpected flattened XNLI group size; check flatten_xnli()")
    rng.shuffle(by_example)

    same_rows = []
    for variants in by_example:
        if len(same_rows) >= n_same:
            break
        v = rng.choice(variants)
        same_rows.append({"premise": v["premise"], "hypothesis": v["hypothesis"], "label": v["label"]})

    cross_rows = []
    for variants in by_example:
        if len(cross_rows) >= n_cross:
            break
        if len(variants) < 2:
            continue
        src = rng.choice(variants)
        tgt = rng.choice([v for v in variants if v["lang"] != src["lang"]])
        cross_rows.append({"premise": src["premise"], "hypothesis": tgt["hypothesis"], "label": src["label"]})

    val_cut = val_size // 2
    val_cut = min(val_cut, len(same_rows), len(cross_rows))

    val_rows = same_rows[:val_cut] + cross_rows[:val_cut]
    train_rows = same_rows[val_cut:] + cross_rows[val_cut:]
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    def _to_hf(rows):
        return Dataset.from_dict(
            {
                "premise": [r["premise"] for r in rows],
                "hypothesis": [r["hypothesis"] for r in rows],
                "label": [r["label"] for r in rows],
            }
        )

    return _to_hf(train_rows), _to_hf(val_rows)


def build_nli_datasets(
    *,
    mnli_train_size=None,
    mnli_val_size=None,
    xnli_pool_size=None,
    xnli_val_size=400,
    same_lang_pct=50,
    cross_lang_pct=50,
    xnli_languages=None,
    seed: int = 42,
):
    if xnli_languages is None:
        xnli_languages = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]

    mnli_raw = load_mnli_dataset()
    mnli_train = _select(mnli_raw["train"], mnli_train_size, seed=seed)
    mnli_val = _select(mnli_raw["validation_matched"], mnli_val_size, seed=seed)
    _keep = ["premise", "hypothesis", "label"]
    mnli_train = mnli_train.map(_mnli_schema, batched=True, remove_columns=[c for c in mnli_train.column_names if c not in _keep])
    mnli_val = mnli_val.map(_mnli_schema, batched=True, remove_columns=[c for c in mnli_val.column_names if c not in _keep])
    mnli_train = mnli_train.cast_column("label", Value("int64"))
    mnli_val = mnli_val.cast_column("label", Value("int64"))

    xnli_raw = load_xnli_dataset()
    xnli_dev = xnli_raw["validation"]
    xnli_flat = xnli_dev.map(
        lambda batch, indices: flatten_xnli(batch, indices, xnli_languages),
        batched=True,
        with_indices=True,
        remove_columns=xnli_dev.column_names,
        desc="Flattening XNLI",
    )

    xnli_train, xnli_val = build_xnli_datasets(
        xnli_flat,
        pool_size=xnli_pool_size,
        val_size=xnli_val_size,
        same_pct=same_lang_pct,
        cross_pct=cross_lang_pct,
        langs=xnli_languages,
        seed=seed,
    )
    xnli_train = xnli_train.cast_column("label", Value("int64"))
    xnli_val = xnli_val.cast_column("label", Value("int64"))

    train_dataset = concatenate_datasets([mnli_train, xnli_train]).shuffle(seed=seed)
    eval_dataset = concatenate_datasets([mnli_val, xnli_val]).shuffle(seed=seed)
    return train_dataset, eval_dataset


def build_and_cache_nli_dataset(
    *,
    model_name: str,
    max_length: int,
    mnli_train_size=None,
    mnli_val_size=None,
    xnli_pool_size=None,
    xnli_val_size=400,
    same_lang_pct=50,
    cross_lang_pct=50,
    xnli_languages=None,
    seed: int = 42,
    force_rebuild: bool = False,
) -> tuple[Dataset, Dataset, dict[str, Any]]:
    raw_meta = {
        "cache_version": 1,
        "dataset_kind": "nli_raw",
        "mnli_train_size": mnli_train_size,
        "mnli_val_size": mnli_val_size,
        "xnli_pool_size": xnli_pool_size,
        "xnli_val_size": xnli_val_size,
        "same_lang_pct": same_lang_pct,
        "cross_lang_pct": cross_lang_pct,
        "xnli_languages": xnli_languages,
        "seed": seed,
    }
    tokenized_meta = {
        "cache_version": 1,
        "dataset_kind": "nli_tokenized",
        "model_name": model_name,
        "max_length": max_length,
        "mnli_train_size": mnli_train_size,
        "mnli_val_size": mnli_val_size,
        "xnli_pool_size": xnli_pool_size,
        "xnli_val_size": xnli_val_size,
        "same_lang_pct": same_lang_pct,
        "cross_lang_pct": cross_lang_pct,
        "xnli_languages": xnli_languages,
        "seed": seed,
    }
    raw_cache_dir = PATHS["nli"]["raw_cache_dir"]
    raw_cache_meta = PATHS["nli"]["raw_cache_meta"]
    tokenized_cache_dir = PATHS["nli"]["tokenized_cache_dir"]
    tokenized_cache_meta = PATHS["nli"]["tokenized_cache_meta"]

    if not force_rebuild:
        loaded = load_dataset_cache(tokenized_cache_dir, meta_path=tokenized_cache_meta, expected_meta=tokenized_meta)
        if loaded is not None:
            return loaded["train"], loaded["val"], tokenized_meta

    train_dataset, eval_dataset = build_nli_datasets(
        mnli_train_size=mnli_train_size,
        mnli_val_size=mnli_val_size,
        xnli_pool_size=xnli_pool_size,
        xnli_val_size=xnli_val_size,
        same_lang_pct=same_lang_pct,
        cross_lang_pct=cross_lang_pct,
        xnli_languages=xnli_languages,
        seed=seed,
    )
    raw_dataset = DatasetDict({"train": train_dataset, "val": eval_dataset})
    save_dataset_cache(raw_dataset, raw_cache_dir, meta_path=raw_cache_meta, meta=raw_meta)

    tokenized = tokenize_dataset_dict(
        raw_dataset,
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        kind="pair",
        max_length=max_length,
        text_columns=("premise", "hypothesis"),
    )
    tokenized = DatasetDict(
        {
            split_name: split.remove_columns(
                [col for col in split.column_names if col not in {"input_ids", "attention_mask", "label"}]
            )
            for split_name, split in tokenized.items()
        }
    )
    save_dataset_cache(tokenized, tokenized_cache_dir, meta_path=tokenized_cache_meta, meta=tokenized_meta)
    return tokenized["train"], tokenized["val"], tokenized_meta
