from __future__ import annotations

# %%
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import DatasetDict, load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    Trainer,
    TrainingArguments,
    XLMRobertaConfig,
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

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


class XLMRobertaTwoHeadForSafety(XLMRobertaPreTrainedModel):
    def __init__(self, config: XLMRobertaConfig):
        super().__init__(config)
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.binary_classifier = nn.Linear(config.hidden_size, 1)
        self.category_classifier = nn.Linear(config.hidden_size, config.num_category_labels)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        binary_label=None,
        **kwargs,
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)

        binary_logits = self.binary_classifier(pooled_output).squeeze(-1)
        category_logits = self.category_classifier(pooled_output)
        logits = torch.cat([binary_logits.unsqueeze(-1), category_logits], dim=-1)

        loss = None
        if labels is not None and binary_label is not None:
            binary_loss = nn.functional.binary_cross_entropy_with_logits(binary_logits, binary_label.float())
            category_loss = nn.functional.binary_cross_entropy_with_logits(category_logits, labels.float())
            loss = binary_loss + category_loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def make_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_train_epochs"],
        learning_rate=CONFIG["learning_rate"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        warmup_ratio=CONFIG["warmup_ratio"],
        weight_decay=CONFIG["weight_decay"],
        fp16=CONFIG["fp16"],
        dataloader_num_workers=CONFIG["dataloader_num_workers"],
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=CONFIG["steps"],
        save_steps=CONFIG["steps"],
        load_best_model_at_end=True,
        metric_for_best_model="binary_f1",
        greater_is_better=True,
        seed=CONFIG["seed"],
        report_to="none",
        label_names=["labels", "binary_label"],
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
if not known_categories:
    raise RuntimeError("Safety cache metadata did not include any category labels.")

label2id = {category: idx for idx, category in enumerate(known_categories)}
id2label = {idx: category for category, idx in label2id.items()}
binary_label2id = {"safe": 0, "unsafe": 1}
binary_id2label = {0: "safe", 1: "unsafe"}

print(f"  Known categories: {len(known_categories)}")
print(f"  Train: {len(ds['train']):,}")
print(f"  Val:   {len(ds['val']):,}")
print(f"  Test:  {len(ds['test']):,}")

for split_name in ("train", "val", "test"):
    ds[split_name].set_format("torch", columns=["input_ids", "attention_mask", "labels", "binary_label"])


# %%
print(f"Loading model: {CONFIG['model_name']}")
model_config = XLMRobertaConfig.from_pretrained(CONFIG["model_name"])
model_config.num_category_labels = len(known_categories)
model_config.label2id = label2id
model_config.id2label = id2label
model_config.binary_label2id = binary_label2id
model_config.binary_id2label = binary_id2label
model = XLMRobertaTwoHeadForSafety.from_pretrained(
    CONFIG["model_name"],
    config=model_config,
    ignore_mismatched_sizes=True,
)

threshold = CONFIG["threshold"]


def compute_metrics(eval_pred):
    predictions, label_ids = eval_pred
    if not isinstance(label_ids, tuple) or len(label_ids) != 2:
        raise RuntimeError("Expected binary and category labels from Trainer")

    category_labels, binary_labels = label_ids
    predictions = np.asarray(predictions)
    binary_logits = predictions[:, 0]
    category_logits = predictions[:, 1:]

    binary_probs = 1 / (1 + np.exp(-binary_logits))
    category_probs = 1 / (1 + np.exp(-category_logits))

    binary_preds = (binary_probs >= threshold).astype(int)
    category_preds = (category_probs >= threshold).astype(int)

    binary_labels = binary_labels.astype(int)
    category_labels = category_labels.astype(int)

    binary_f1 = f1_score(binary_labels, binary_preds, zero_division=0)
    binary_precision = precision_score(binary_labels, binary_preds, zero_division=0)
    binary_recall = recall_score(binary_labels, binary_preds, zero_division=0)
    category_micro_f1 = f1_score(category_labels, category_preds, average="micro", zero_division=0)
    category_macro_f1 = f1_score(category_labels, category_preds, average="macro", zero_division=0)
    category_precision = precision_score(category_labels, category_preds, average="micro", zero_division=0)
    category_recall = recall_score(category_labels, category_preds, average="micro", zero_division=0)
    per_category_f1 = f1_score(category_labels, category_preds, average=None, zero_division=0)

    per_category_report = {
        f"f1_{known_categories[i]}": float(per_category_f1[i])
        for i in range(len(known_categories))
    }

    return {
        "binary_f1": float(binary_f1),
        "binary_precision": float(binary_precision),
        "binary_recall": float(binary_recall),
        "category_micro_f1": float(category_micro_f1),
        "category_macro_f1": float(category_macro_f1),
        "category_precision": float(category_precision),
        "category_recall": float(category_recall),
        **per_category_report,
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
