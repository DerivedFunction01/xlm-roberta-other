# %% [markdown]
# # XLM-RoBERTa Safety Guard Fine-tuning
# Fine-tunes xlm-roberta-base on the nvidia/Nemotron-Safety-Guard-Dataset-v3 dataset.
# Treats prompt and response as separate training examples with independent labels.
# Uses multi-label classification over the violated_categories taxonomy.

# %% [markdown]
# ## Config

# %%
CONFIG = {
    # ── Model ───────────────────────────────────────────────────────────────
    "model_name": "xlm-roberta-base",
    "output_dir": "./xlmr-safety-guard",
    # ── Dataset ─────────────────────────────────────────────────────────────
    "dataset_name": "nvidia/Nemotron-Safety-Guard-Dataset-v3",
    "dataset_split": "train",  # adjust if the dataset has explicit splits
    "val_size": 0.05,  # fraction of data held out for validation
    "test_size": 0.05,  # fraction of data held out for final test
    # ── Filtering ───────────────────────────────────────────────────────────
    # Drop redacted prompts that have not been reconstructed
    "drop_redacted": True,
    # Minimum number of training examples per label (rarer labels are kept as-is)
    "min_label_count": 10,
    # ── Tokenisation ────────────────────────────────────────────────────────
    "max_length": 256,
    # ── Training ────────────────────────────────────────────────────────────
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": True,
    "dataloader_num_workers": 4,
    "seed": 42,
    # ── Loss ────────────────────────────────────────────────────────────────
    # pos_weight scalar applied uniformly across all labels to counter class imbalance.
    # Set to None to disable; tune upward (e.g. 5–20) if recall on unsafe is too low.
    "bce_pos_weight": 10.0,
    # ── Classification threshold ────────────────────────────────────────────
    # Applied to sigmoid outputs at inference / eval time.
    "threshold": 0.5,
}

# Special token used for redacted prompts
REDACTED_TOKEN = "REDACTED"

# %% [markdown]
# ## Imports

# %%
import random
import warnings
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

warnings.filterwarnings("ignore", category=UserWarning)

# %% [markdown]
# ## Load Dataset

# %%
print(f"Loading {CONFIG['dataset_name']} ...")
raw = load_dataset(CONFIG["dataset_name"], split=CONFIG["dataset_split"])
print(f"  Raw rows: {len(raw):,}")
print(f"  Columns:  {raw.column_names}")

# %% [markdown]
# ## Build Flat Examples
#
# Each original row may yield up to two training examples:
#   1. The **prompt** example  — always present (unless redacted/dropped)
#   2. The **response** example — only when `response` is non-null
#
# Each example gets:
#   - `text`             : the string to classify
#   - `binary_label`     : 0 (safe) or 1 (unsafe) — useful for auxiliary metrics
#   - `categories`       : list of violated category strings (empty list if safe)
#   - `role`             : "prompt" | "response"
#   - `language`         : ISO 639-1 code


# %%
def parse_categories(raw_str: str) -> list[str]:
    """Split comma-separated category string into a cleaned list."""
    if not raw_str or not isinstance(raw_str, str):
        return []
    return [c.strip() for c in raw_str.split(",") if c.strip()]


def row_to_examples(row: dict) -> list[dict]:
    examples = []

    # ── Prompt example ───────────────────────────────────────────────────
    prompt_text = row.get("prompt", "")
    if CONFIG["drop_redacted"] and prompt_text == REDACTED_TOKEN:
        pass  # skip; no external dataset reconstruction implemented here
    elif prompt_text and isinstance(prompt_text, str):
        examples.append(
            {
                "text": prompt_text,
                "binary_label": 1 if row["prompt_label"] == "unsafe" else 0,
                "categories": parse_categories(row.get("violated_categories", "")),
                "role": "prompt",
                "language": row.get("language", "en"),
            }
        )

    # ── Response example ─────────────────────────────────────────────────
    response_text = row.get("response")
    response_label = row.get("response_label")
    if (
        response_text is not None
        and isinstance(response_text, str)
        and response_text.strip()
        and response_label not in (None, "", "null")
    ):
        examples.append(
            {
                "text": response_text,
                "binary_label": 1 if response_label == "unsafe" else 0,
                "categories": parse_categories(row.get("violated_categories", "")),
                "role": "response",
                "language": row.get("language", "en"),
            }
        )

    return examples


print("Building flat examples ...")
all_examples: list[dict] = []
for row in raw:
    all_examples.extend(row_to_examples(row))

print(f"  Total flat examples: {len(all_examples):,}")

role_counts = Counter(e["role"] for e in all_examples)
print(f"  Prompt examples:     {role_counts['prompt']:,}")
print(f"  Response examples:   {role_counts['response']:,}")

safe_count = sum(1 for e in all_examples if e["binary_label"] == 0)
unsafe_count = sum(1 for e in all_examples if e["binary_label"] == 1)
print(f"  Safe:   {safe_count:,}  ({100 * safe_count / len(all_examples):.1f}%)")
print(f"  Unsafe: {unsafe_count:,}  ({100 * unsafe_count / len(all_examples):.1f}%)")

# %% [markdown]
# ## Build Label Vocabulary
#
# We collect every unique violated category that appears at least
# `min_label_count` times and fit a MultiLabelBinarizer over them.
# Rare categories are lumped under an "other" catch-all.

# %%
category_counter: Counter = Counter()
for e in all_examples:
    category_counter.update(e["categories"])

print(f"\nAll violated categories ({len(category_counter)} unique):")
for cat, cnt in category_counter.most_common():
    print(f"  {cnt:>6,}  {cat}")

# Filter rare categories
min_count = CONFIG["min_label_count"]
known_categories = sorted(
    cat for cat, cnt in category_counter.items() if cnt >= min_count
)

# Include a generic "unsafe" label so that safe examples have an all-zero
# target vector and unsafe examples without a known category get a signal.
# We do NOT add it explicitly; safe → all zeros is the correct multi-hot.
print(f"\nKept categories (≥{min_count} occurrences): {len(known_categories)}")
for c in known_categories:
    print(f"  {c}")
    
mlb = MultiLabelBinarizer(classes=known_categories)
mlb.fit([known_categories])  # fit on full vocabulary to fix column order

NUM_LABELS = len(known_categories)
print(f"\nnum_labels = {NUM_LABELS}")

# %% [markdown]
# ## Binarize Labels


# %%
def binarize(example: dict) -> dict:
    """Convert category list to multi-hot numpy array."""
    # Keep only known categories; rare ones are silently dropped
    filtered = [c for c in example["categories"] if c in set(known_categories)]
    multi_hot = mlb.transform([filtered])[0].astype(np.float32)
    example["labels"] = multi_hot.tolist()
    return example


all_examples = [binarize(e) for e in all_examples]

# %% [markdown]
# ## Train / Val / Test Split
#
# Stratified split is non-trivial for multi-label data; we use a simple
# random split and rely on dataset size for approximate balance.

# %%
rng = np.random.default_rng(CONFIG["seed"])
indices = np.arange(len(all_examples))
rng.shuffle(indices)

n_test = int(len(indices) * CONFIG["test_size"])
n_val = int(len(indices) * CONFIG["val_size"])

test_idx = indices[:n_test]
val_idx = indices[n_test : n_test + n_val]
train_idx = indices[n_test + n_val :]


def make_hf_dataset(idxs: np.ndarray) -> Dataset:
    rows = [all_examples[i] for i in idxs]
    return Dataset.from_dict(
        {
            "text": [r["text"] for r in rows],
            "binary_label": [r["binary_label"] for r in rows],
            "labels": [r["labels"] for r in rows],
            "role": [r["role"] for r in rows],
            "language": [r["language"] for r in rows],
        }
    )


ds = DatasetDict(
    {
        "train": make_hf_dataset(train_idx),
        "val": make_hf_dataset(val_idx),
        "test": make_hf_dataset(test_idx),
    }
)

print(f"\nSplit sizes:")
print(f"  Train: {len(ds['train']):,}")
print(f"  Val:   {len(ds['val']):,}")
print(f"  Test:  {len(ds['test']):,}")

# %% [markdown]
# ## Tokenise

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
    # HuggingFace Trainer expects `labels` as a list/tensor — keep as-is
    enc["labels"] = batch["labels"]
    return enc


col_remove = ["text", "role", "language", "binary_label"]

ds["train"] = ds["train"].map(tokenize, batched=True, remove_columns=col_remove)
ds["val"] = ds["val"].map(tokenize, batched=True, remove_columns=col_remove)
ds["test"] = ds["test"].map(tokenize, batched=True, remove_columns=col_remove)

ds["train"].set_format("torch", columns=["input_ids", "attention_mask", "labels"])
ds["val"].set_format("torch", columns=["input_ids", "attention_mask", "labels"])
ds["test"].set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# %% [markdown]
# ## Model

# %%
id2label = {i: cat for i, cat in enumerate(known_categories)}
label2id = {cat: i for i, cat in id2label.items()}

print(f"Loading model: {CONFIG['model_name']}")
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    problem_type="multi_label_classification",
)

# %% [markdown]
# ## Custom Trainer with Weighted BCE Loss
#
# Safety data is heavily skewed toward safe examples.
# `pos_weight` upweights the positive (violated) class per label.


# %%
class WeightedBCETrainer(Trainer):
    """Trainer that uses BCEWithLogitsLoss with optional pos_weight."""

    def __init__(self, *args, pos_weight: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if pos_weight is not None:
            pw = torch.full((NUM_LABELS,), pos_weight)
            self._loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self._loss_fn = nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self._loss_fn(logits.to(labels.device), labels)
        return (loss, outputs) if return_outputs else loss


# %% [markdown]
# ## Metrics
#
# For multi-label classification we report:
#   - Micro-F1  (aggregates TP/FP/FN across all labels)
#   - Macro-F1  (unweighted mean per-label F1)
#   - Micro-Precision / Micro-Recall

# %%
THRESHOLD = CONFIG["threshold"]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= THRESHOLD).astype(int)
    labels = labels.astype(int)

    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="micro", zero_division=0)
    recall = recall_score(labels, preds, average="micro", zero_division=0)

    # Per-label F1 for inspection
    per_label_f1 = f1_score(labels, preds, average=None, zero_division=0)
    per_label_report = {
        f"f1_{known_categories[i]}": float(per_label_f1[i])
        for i in range(len(known_categories))
    }

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        **per_label_report,
    }


# %% [markdown]
# ## Train

# %%
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
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    seed=CONFIG["seed"],
    report_to="none",
    label_names=["labels"],  # needed for multi-label
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

# %% [markdown]
# ## Evaluate on Test Set

# %%
print("\nRunning final evaluation on test set ...")
test_results = trainer.evaluate(ds["test"])

print("\nTest results:")
for k, v in sorted(test_results.items()):
    print(f"  {k:<40} {v:.4f}")

# %% [markdown]
# ## Inference Helper
#
# Demonstrates how to run the saved model on new text at inference time.


# %%
def predict(
    texts: list[str],
    model_path: str = CONFIG["output_dir"],
    threshold: float = THRESHOLD,
) -> list[dict]:
    """
    Run safety classification on a list of strings.

    Returns a list of dicts with:
      - "text"             : the input string
      - "is_unsafe"        : bool
      - "violated_categories" : list of violated category names
      - "scores"           : dict {category: sigmoid_score}
    """
    from transformers import pipeline

    clf = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        top_k=None,  # return scores for all labels
        device=0 if torch.cuda.is_available() else -1,
    )

    results = []
    for text, output in zip(
        texts, clf(texts, truncation=True, max_length=CONFIG["max_length"])
    ):
        scores = {item["label"]: item["score"] for item in output}
        violated = [cat for cat, score in scores.items() if score >= threshold]
        results.append(
            {
                "text": text,
                "is_unsafe": len(violated) > 0,
                "violated_categories": violated,
                "scores": scores,
            }
        )
    return results


# %% [markdown]
# ## Save

# %%
trainer.save_model(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])

# Also save the label vocabulary so it can be reconstructed offline
import json, os

vocab_path = os.path.join(CONFIG["output_dir"], "category_vocab.json")
with open(vocab_path, "w") as f:
    json.dump({"categories": known_categories}, f, indent=2)

print(f"\nModel saved to:        {CONFIG['output_dir']}")
print(f"Category vocab saved to: {vocab_path}")
