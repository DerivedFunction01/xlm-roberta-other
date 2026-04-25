from __future__ import annotations

# %%
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

# %%
CONFIG = {
    "model_name": "xlm-roberta-base",
    "output_dir": "./xlmr-safety-guard",
    "max_length": 512,
    "num_train_epochs": 2,
    "steps": 1000,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": True,
    "dataloader_num_workers": 4,
    "seed": 42,
    "threshold": 0.5,
}

BASE_DIR = Path(".")
TOKENIZED_CACHE_DIR = BASE_DIR / ".cache" / "xlm_roberta_other" / "safety" / "tokenized"
TOKENIZED_CACHE_META = TOKENIZED_CACHE_DIR / "dataset.meta.json"

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
warnings.filterwarnings("ignore", category=UserWarning)


# %%
def load_cached_dataset(cache_dir: Path) -> DatasetDict:
    split_files = {path.stem: str(path) for path in sorted(cache_dir.glob("*.parquet"))}
    if not split_files:
        raise FileNotFoundError(f"No parquet splits found in {cache_dir}")
    return load_dataset("parquet", data_files=split_files)


def make_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_train_epochs"],
        learning_rate=CONFIG["learning_rate"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
        warmup_ratio=CONFIG["warmup_ratio"],
        weight_decay=CONFIG["weight_decay"],
        fp16=CONFIG["fp16"],
        dataloader_num_workers=CONFIG["dataloader_num_workers"],
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=CONFIG["steps"],
        save_steps=CONFIG["steps"],
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        seed=CONFIG["seed"],
        report_to="tensorboard",
        label_names=["labels"],
        push_to_hub=True
    )


# %%
if not TOKENIZED_CACHE_META.exists():
    raise RuntimeError("Tokenized safety cache not found. Run the build script first.")

with TOKENIZED_CACHE_META.open(encoding="utf-8") as f:
    meta = json.load(f)

if meta.get("model_name") != CONFIG["model_name"] or meta.get("max_length") != CONFIG["max_length"]:
    raise RuntimeError("Safety cache metadata does not match the current config.")

ds = load_cached_dataset(TOKENIZED_CACHE_DIR)
known_categories = list(meta.get("known_categories", []))
label2id = dict(meta.get("label2id", {}))
id2label = {int(k): v for k, v in meta.get("id2label", {}).items()}

print(f"  Known labels: {len(known_categories)}")
print(f"  Train: {len(ds['train']):,}")
print(f"  Val:   {len(ds['val']):,}")
print(f"  Test:  {len(ds['test']):,}")

for split_name in ("train", "val", "test"):
    ds[split_name].set_format("torch", columns=["input_ids", "attention_mask", "labels"])


# %%
print(f"Loading model: {CONFIG['model_name']}")
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=len(known_categories),
    id2label=id2label,
    label2id=label2id,
    problem_type="multi_label_classification",
)

threshold = CONFIG["threshold"]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)

    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="micro", zero_division=0)
    recall = recall_score(labels, preds, average="micro", zero_division=0)
    per_label_f1 = f1_score(labels, preds, average=None, zero_division=0)
    per_label_report = {f"f1_{known_categories[i]}": float(per_label_f1[i]) for i in range(len(known_categories))}

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        **per_label_report,
    }


trainer = Trainer(
    model=model,
    args=make_training_args(),
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    compute_metrics=compute_metrics,
)


# %%
trainer.train()
print("\nEvaluating on test split ...")
print(trainer.evaluate(ds["test"]))

trainer.save_model(CONFIG["output_dir"])
print(f"\nModel saved to: {CONFIG['output_dir']}")
