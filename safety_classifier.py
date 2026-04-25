from __future__ import annotations

# %% [markdown]
# # XLM-RoBERTa Safety Guard Fine-tuning
# Fine-tunes xlm-roberta-base on nvidia/Nemotron-Safety-Guard-Dataset-v3.

# %% [markdown]
# ## Config

# %%
CONFIG = {
    "model_name": "xlm-roberta-base",
    "output_dir": "./xlmr-safety-guard",
    "dataset_split": "train",
    "val_size": 0.05,
    "test_size": 0.05,
    "drop_redacted": True,
    "augment": True,
    "min_label_count": 10,
    "max_length": 256,
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": True,
    "dataloader_num_workers": 4,
    "seed": 42,
    "bce_pos_weight": 10.0,
    "threshold": 0.5,
    "force_rebuild_cache": False,
}

# %%
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from paths import PATHS
from safety_data import build_safety_classifier_dataset

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
warnings.filterwarnings("ignore", category=UserWarning)

# %%
print("Building / loading safety dataset cache ...")
ds, known_categories, label2id, id2label, meta = build_safety_classifier_dataset(
    dataset_split=CONFIG["dataset_split"],
    drop_redacted=CONFIG["drop_redacted"],
    augment=CONFIG["augment"],
    min_label_count=CONFIG["min_label_count"],
    val_size=CONFIG["val_size"],
    test_size=CONFIG["test_size"],
    seed=CONFIG["seed"],
    force_rebuild=CONFIG["force_rebuild_cache"],
    cache_dir=PATHS["safety_classifier"]["cache_dir"],
    cache_meta_path=PATHS["safety_classifier"]["cache_meta"],
)

print(f"  Known labels: {len(known_categories)}")
print(f"  Train: {len(ds['train']):,}")
print(f"  Val:   {len(ds['val']):,}")
print(f"  Test:  {len(ds['test']):,}")

# %%
print(f"\nLoading tokenizer: {CONFIG['model_name']}")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])


def tokenize(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        max_length=CONFIG["max_length"],
        padding="max_length",
    )
    enc["labels"] = batch["labels"]
    return enc


keep_columns = ["text", "binary_label", "labels", "role", "language", "prompt_label_source", "response_label_source", "tag", "source_id"]
for split_name in ("train", "val", "test"):
    ds[split_name] = ds[split_name].map(tokenize, batched=True, remove_columns=[c for c in ds[split_name].column_names if c in keep_columns])
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


class WeightedBCETrainer(Trainer):
    """Trainer that uses BCEWithLogitsLoss with optional pos_weight."""

    def __init__(self, *args, pos_weight: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if pos_weight is not None:
            pw = torch.full((len(known_categories),), pos_weight)
            self._loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self._loss_fn = nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        loss = self._loss_fn(outputs.logits.to(labels.device), labels)
        return (loss, outputs) if return_outputs else loss


THRESHOLD = CONFIG["threshold"]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= THRESHOLD).astype(int)
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


training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_train_epochs"],
    learning_rate=CONFIG["learning_rate"],
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
    warmup_ratio=CONFIG["warmup_ratio"],
    weight_decay=CONFIG["weight_decay"],
    fp16=CONFIG["fp16"],
    dataloader_num_workers=CONFIG["dataloader_num_workers"],
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    seed=CONFIG["seed"],
    report_to="none",
    label_names=["labels"],
)

trainer = WeightedBCETrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    pos_weight=CONFIG["bce_pos_weight"],
)

trainer.train()
print("\nEvaluating on test split ...")
print(trainer.evaluate(ds["test"]))

trainer.save_model(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])
print(f"\nModel saved to: {CONFIG['output_dir']}")
