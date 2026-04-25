from __future__ import annotations

# %%
import json
import random
from pathlib import Path

import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

# %%
CONFIG = {
    "model_name": "xlm-roberta-base",
    "output_dir": "./xlmr-xnli-mnli",
    "max_length": 256,
    "num_train_epochs": 2,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "warmup_ratio": 0.1,
    "weight_decay": 0.06,
    "fp16": True,
    "seed": 42,
}

BASE_DIR = Path(".")
HF_TOKEN_PATH = BASE_DIR / "hf_token"
TOKENIZED_CACHE_DIR = BASE_DIR / ".cache" / "xlm_roberta_other" / "nli" / "tokenized"
TOKENIZED_CACHE_META = TOKENIZED_CACHE_DIR / "dataset.meta.json"
LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

if not HF_TOKEN_PATH.exists():
    raise FileNotFoundError(f"Missing Hugging Face token file: {HF_TOKEN_PATH}")

HF_TOKEN = HF_TOKEN_PATH.read_text(encoding="utf-8").strip()
if not HF_TOKEN:
    raise ValueError(f"Hugging Face token file is empty: {HF_TOKEN_PATH}")

login(token=HF_TOKEN, add_to_git_credential=False)


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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=CONFIG["seed"],
        report_to="none",
    )


# %%
if not TOKENIZED_CACHE_META.exists():
    raise RuntimeError("Tokenized NLI cache not found. Run the build script first.")

with TOKENIZED_CACHE_META.open(encoding="utf-8") as f:
    meta = json.load(f)

if meta.get("model_name") != CONFIG["model_name"] or meta.get("max_length") != CONFIG["max_length"]:
    raise RuntimeError("NLI cache metadata does not match the current config.")

cached = load_cached_dataset(TOKENIZED_CACHE_DIR)
train_dataset = cached["train"]
eval_dataset = cached["val"]
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

print(f"Final train: {len(train_dataset):,} rows")
print(f"Final eval:  {len(eval_dataset):,} rows")


# %%
print(f"Loading model: {CONFIG['model_name']}")
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    token=HF_TOKEN,
)

accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)


trainer = Trainer(
    model=model,
    args=make_training_args(),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)


# %%
trainer.train()
trainer.save_model(CONFIG["output_dir"])
print(f"\nModel saved to: {CONFIG['output_dir']}")
