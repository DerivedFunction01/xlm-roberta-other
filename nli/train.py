from __future__ import annotations

# %%
import math
import json
import random
from pathlib import Path

import evaluate
import numpy as np
import torch
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
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "evals_per_epoch": 2,
    "saves_per_epoch": 2,
    "logging_steps": 100,
    "weight_decay": 0.06,
    "fp16": True,
    "seed": 42,
}

WARMUP_RATIO = 0.1

BASE_DIR = Path(".")
HF_TOKEN_PATH = BASE_DIR / "hf_token"
TOKENIZED_CACHE_DIR = BASE_DIR / ".cache" / "xlm_roberta_other" / "nli" / "tokenized"
TOKENIZED_CACHE_META = TOKENIZED_CACHE_DIR / "dataset.meta.json"
LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

if not HF_TOKEN_PATH.exists():
    HF_TOKEN = None
else:
    HF_TOKEN = HF_TOKEN_PATH.read_text(encoding="utf-8").strip() or None
    if HF_TOKEN is not None:
        login(token=HF_TOKEN, add_to_git_credential=False)


# %%
def load_cached_dataset(cache_dir: Path) -> DatasetDict:
    split_files = {path.stem: str(path) for path in sorted(cache_dir.glob("*.parquet"))}
    if not split_files:
        raise FileNotFoundError(f"No parquet splits found in {cache_dir}")
    return load_dataset("parquet", data_files=split_files)


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return max(1, torch.cuda.device_count())


def get_effective_batch_size() -> int:
    return (
        CONFIG["per_device_train_batch_size"]
        * CONFIG["gradient_accumulation_steps"]
        * get_world_size()
    )


def compute_warmup_steps(train_size: int) -> int:
    steps_per_epoch = math.ceil(train_size / get_effective_batch_size())
    total_steps = steps_per_epoch * CONFIG["num_train_epochs"]
    return max(1, int(total_steps * WARMUP_RATIO))


def compute_step_interval(train_size: int, *, checkpoints_per_epoch: int) -> int:
    steps_per_epoch = math.ceil(train_size / get_effective_batch_size())
    return max(1, math.ceil(steps_per_epoch / checkpoints_per_epoch))


def make_training_args(*, warmup_steps: int, eval_steps: int, save_steps: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_train_epochs"],
        learning_rate=CONFIG["learning_rate"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        logging_strategy="steps",
        logging_steps=CONFIG["logging_steps"],
        warmup_steps=warmup_steps,
        weight_decay=CONFIG["weight_decay"],
        fp16=CONFIG["fp16"],
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=CONFIG["seed"],
        report_to="tensorboard",
        push_to_hub=True,
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
print(f"GPU count:   {get_world_size()}")
warmup_steps = compute_warmup_steps(len(train_dataset))
eval_interval = compute_step_interval(len(train_dataset), checkpoints_per_epoch=CONFIG["evals_per_epoch"])
save_interval = compute_step_interval(len(train_dataset), checkpoints_per_epoch=CONFIG["saves_per_epoch"])
print(f"Warmup steps: {warmup_steps}")
print(f"Eval interval: {eval_interval}")
print(f"Save interval: {save_interval}")


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
    args=make_training_args(
        warmup_steps=warmup_steps,
        eval_steps=eval_interval,
        save_steps=save_interval,
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)


# %%
trainer.train()
trainer.save_model(CONFIG["output_dir"])
print(f"\nModel saved to: {CONFIG['output_dir']}")
trainer.push_to_hub()
