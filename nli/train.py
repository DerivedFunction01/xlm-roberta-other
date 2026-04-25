from __future__ import annotations

CONFIG = {
    "model_name": "xlm-roberta-base",
    "output_dir": "./xlmr-xnli-mnli",
    "mnli_train_size": None,
    "mnli_val_size": None,
    "xnli_pool_size": None,
    "xnli_val_size": 400,
    "same_lang_pct": 50,
    "cross_lang_pct": 50,
    "xnli_languages": ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"],
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

import random

import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from .data import ID2LABEL, LABEL2ID, build_nli_datasets

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


def main() -> None:
    print("Building NLI datasets...")
    train_dataset, eval_dataset = build_nli_datasets(
        mnli_train_size=CONFIG["mnli_train_size"],
        mnli_val_size=CONFIG["mnli_val_size"],
        xnli_pool_size=CONFIG["xnli_pool_size"],
        xnli_val_size=CONFIG["xnli_val_size"],
        same_lang_pct=CONFIG["same_lang_pct"],
        cross_lang_pct=CONFIG["cross_lang_pct"],
        xnli_languages=CONFIG["xnli_languages"],
        seed=CONFIG["seed"],
    )

    print(f"Final train: {len(train_dataset):,} rows")
    print(f"Final eval:  {len(eval_dataset):,} rows")

    print(f"\nLoading tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    def tokenize(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=CONFIG["max_length"],
            padding="max_length",
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    print(f"Loading model: {CONFIG['model_name']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=preds, references=labels)

    training_args = TrainingArguments(
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    print(f"\nModel saved to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()

