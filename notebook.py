# %% [markdown]
# # XLM-RoBERTa NLI Fine-tuning
# Replicates the MoritzLaurer mDeBERTa-v3-base-mnli-xnli approach using xlm-roberta-base.
# XNLI data is split into same-language pairs and cross-language pairs (mixed premise/hypothesis langs).

# %% [markdown]
# ## Config

# %%
CONFIG = {
    # ── Model ──────────────────────────────────────────────────────────────
    "model_name": "xlm-roberta-base",
    "output_dir": "./xlmr-nli-out",

    # ── Data sizes ─────────────────────────────────────────────────────────
    # Set to None to use the full split
    "mnli_train_size":      2000,   # rows sampled from MNLI train
    "mnli_val_size":        500,    # rows sampled from MNLI validation_matched
    "xnli_pool_size":       2400,   # total XNLI rows to draw from before train/val split
    "xnli_val_size":        400,    # how many of those go to validation

    # ── XNLI language mix ──────────────────────────────────────────────────
    # Percentages must sum to 100.
    # same_lang_pct:  premise and hypothesis are in the same language
    # cross_lang_pct: premise and hypothesis are in different languages
    "same_lang_pct":  50,
    "cross_lang_pct": 50,

    # Languages available in XNLI
    "xnli_languages": [
        "ar", "bg", "de", "el", "en", "es",
        "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh",
    ],

    # ── Tokenisation ───────────────────────────────────────────────────────
    "max_length": 256,

    # ── Training (mirrors Laurer's hyperparams) ────────────────────────────
    "num_train_epochs":            2,
    "learning_rate":               2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size":  16,
    "warmup_ratio":                0.1,
    "weight_decay":                0.06,
    "fp16":                        True,   # XLM-R supports fp16 (mDeBERTa does not)
    "seed":                        42,
}

# Label mapping — MNLI and XNLI both use 0/1/2 in this order
LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

assert CONFIG["same_lang_pct"] + CONFIG["cross_lang_pct"] == 100, \
    "same_lang_pct + cross_lang_pct must equal 100"

# %% [markdown]
# ## Imports

# %%
import random
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

# %% [markdown]
# ## Load MultiNLI

# %%
print("Loading MultiNLI...")
mnli_raw = load_dataset("nyu-mll/multi_nli")

def _select(ds, n):
    """Shuffle + select n rows; pass None to keep all."""
    return ds.shuffle(seed=CONFIG["seed"]).select(range(n)) if n else ds.shuffle(seed=CONFIG["seed"])

mnli_train = _select(mnli_raw["train"],                CONFIG["mnli_train_size"])
mnli_val   = _select(mnli_raw["validation_matched"],   CONFIG["mnli_val_size"])

# Normalise to flat schema: premise, hypothesis, label (int)
_keep = ["premise", "hypothesis", "label"]

def _mnli_schema(batch):
    return {"premise": batch["premise"], "hypothesis": batch["hypothesis"], "label": batch["label"]}

mnli_train = mnli_train.map(_mnli_schema, batched=True,
                            remove_columns=[c for c in mnli_train.column_names if c not in _keep])
mnli_val   = mnli_val.map(_mnli_schema,   batched=True,
                          remove_columns=[c for c in mnli_val.column_names   if c not in _keep])

print(f"  MNLI train: {len(mnli_train):,} rows")
print(f"  MNLI val:   {len(mnli_val):,} rows")

# %% [markdown]
# ## Load & flatten XNLI
#
# `facebook/xnli` with `"all_languages"` stores each row as:
#   - `premise`:    dict  {lang_code: text, ...}
#   - `hypothesis`: dict  {"language": [...], "translation": [...]}
#   - `label`:      int   (0=entailment, 1=neutral, 2=contradiction)
#
# We flatten so every row is a single (premise, hypothesis, label, lang) tuple.
# `lang` tracks the premise language, which lets us build cross-lang pairs later.

# %%
print("Loading XNLI (all languages)...")
xnli_raw = load_dataset("facebook/xnli", "all_languages")

# Use the dev split — professionally translated, not machine-translated
# (Laurer explicitly avoids the machine-translated XNLI train split)
xnli_dev = xnli_raw["validation"]

LANGS = CONFIG["xnli_languages"]
# {
#     "premise": [
#         "وقال، ماما، لقد عدت للمنزل.",
#         "И той каза: Мамо, у дома съм.",
#         "und er hat gesagt, Mama ich bin daheim.",
#         "Και είπε, Μαμά, έφτασα στο σπίτι.",
#         "And he said, Mama, I'm home.",
#         "Y él dijo: Mamá, estoy en casa.",
#         "Et il a dit, maman, je suis à la maison.",
#         "और उसने कहा, माँ, मैं घर आया हूं।",
#         "И он сказал: Мама, я дома.",
#         "Naye akasema, Mama, niko nyumbani.",
#         "และเขาพูดว่า, ม่าม๊า ผมอยู่บ้าน",
#         "Ve Anne, evdeyim dedi.",
#         "اور اس نے کہا امّی، میں گھر آگیا ہوں۔",
#         "Và anh ấy nói, Mẹ, con đã về nhà.",
#         "他说，妈妈，我回来了。",
#     ],
#     "hypothesis": [
#         "اتصل بأمه حالما أوصلته حافلة المدرسية.",
#         "Той се обади на майка си веднага щом училищният автобус го е оставил.",
#         "Er rief seine Mutter an, sobald er aus dem Schulbus stieg.",
#         "Τηλεφώνησε στη μαμά του μόλις το σχολικό λεωφορείο τον άφησε.",
#         "He called his mom as soon as the school bus dropped him off.",
#         "Llamó a su madre tan pronto como el autobús escolar lo dejó.",
#         "Il a appelé sa mère dès que le bus scolaire l'a déposé.",
#         "जब ही उसकी स्कूल बस ने उसे उतरा उसने अपनी माँ को बुलाया",
#         "Он позвал маму, как только вышел из школьного автобуса.",
#         "Alimwita mama yake mara tu basi ya shule ilipomshukisha.",
#         "เขาโทรหาเเม่ของเขาอย่างรวดเร็วหลังจากที่รถโรงเรียนส่งเขาเเล้ว",
#         "Okul servisi onu bırakır bırakmaz annesini aradı.",
#         "اسنے اپنی امی کو فوران فون کیا جیسے ھی سکوول بس نے اسے اتارا",
#         "Ngay khi xuống xe buýt của trường, anh ấy gọi cho mẹ.",
#         "校车把他放下后，他立即给他妈妈打了电话。",
#     ],
#     "label": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     "lang": [
#         "ar",
#         "bg",
#         "de",
#         "el",
#         "en",
#         "es",
#         "fr",
#         "hi",
#         "ru",
#         "sw",
#         "th",
#         "tr",
#         "ur",
#         "vi",
#         "zh",
#     ],
# }
def flatten_xnli(example):
    """Expand one multi-lang row into one row per language."""
    hyp_translations = example["hypothesis"]["translation"]   # ordered by LANGS
    return {
        "premise":    [example["premise"][lang]    for lang in LANGS],
        "hypothesis": [hyp_translations[i]         for i, _ in enumerate(LANGS)],
        "label":      [example["label"]]            * len(LANGS),
        "lang":       list(LANGS),
    }

xnli_flat = xnli_dev.map(
    flatten_xnli,
    batched=False,
    remove_columns=xnli_dev.column_names,
)

print(f"  XNLI flattened: {len(xnli_flat):,} rows  ({len(xnli_dev):,} originals × {len(LANGS)} langs)")

# %% [markdown]
# ## Build same-language and cross-language XNLI pairs
#
# **Same-language**: premise[lang] paired with hypothesis[lang] — native to the dataset.
#
# **Cross-language**: premise[lang_A] paired with hypothesis[lang_B], A ≠ B.
# Because all hypotheses are translations of the same underlying sentence, the label is preserved.
# We pair rows from the *same original example* but different language variants.

# %%
def build_xnli_datasets(flat_ds, pool_size, val_size, same_pct, cross_pct, langs, seed=42):
    """
    Returns (train_dataset, val_dataset) as HF Datasets.

    pool_size  : total rows to sample from flat_ds before train/val split
    val_size   : how many rows go to validation (drawn equally from same & cross)
    same_pct   : % of pool that should be same-language pairs
    cross_pct  : % of pool that should be cross-language pairs
    """
    rng = random.Random(seed)

    n_same  = int(pool_size * same_pct  / 100)
    n_cross = int(pool_size * cross_pct / 100)
    n_lang  = len(langs)

    # ── Group flat rows by original XNLI example ───────────────────────────
    # The flat dataset has n_lang consecutive rows per original example.
    n_orig = len(flat_ds) // n_lang
    by_example = [
        [flat_ds[i * n_lang + j] for j in range(n_lang)]
        for i in range(n_orig)
    ]
    rng.shuffle(by_example)

    # ── Same-language rows ─────────────────────────────────────────────────
    same_rows = []
    for variants in by_example:
        if len(same_rows) >= n_same:
            break
        v = rng.choice(variants)
        same_rows.append({"premise": v["premise"], "hypothesis": v["hypothesis"], "label": v["label"]})

    # ── Cross-language rows ────────────────────────────────────────────────
    cross_rows = []
    for variants in by_example:
        if len(cross_rows) >= n_cross:
            break
        if len(variants) < 2:
            continue
        src = rng.choice(variants)
        tgt = rng.choice([v for v in variants if v["lang"] != src["lang"]])
        cross_rows.append({"premise": src["premise"], "hypothesis": tgt["hypothesis"], "label": src["label"]})

    print(f"  XNLI same-lang rows (pool):  {len(same_rows):,}")
    print(f"  XNLI cross-lang rows (pool): {len(cross_rows):,}")

    # ── Train / val split ─────────────────────────────────────────────────
    val_cut = val_size // 2   # half from same, half from cross
    val_cut = min(val_cut, len(same_rows), len(cross_rows))

    val_rows   = same_rows[:val_cut]  + cross_rows[:val_cut]
    train_rows = same_rows[val_cut:]  + cross_rows[val_cut:]
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    def _to_hf(rows):
        return Dataset.from_dict({
            "premise":    [r["premise"]    for r in rows],
            "hypothesis": [r["hypothesis"] for r in rows],
            "label":      [r["label"]      for r in rows],
        })

    return _to_hf(train_rows), _to_hf(val_rows)


xnli_train, xnli_val = build_xnli_datasets(
    xnli_flat,
    pool_size  = CONFIG["xnli_pool_size"],
    val_size   = CONFIG["xnli_val_size"],
    same_pct   = CONFIG["same_lang_pct"],
    cross_pct  = CONFIG["cross_lang_pct"],
    langs      = CONFIG["xnli_languages"],
    seed       = CONFIG["seed"],
)

print(f"  XNLI train: {len(xnli_train):,}")
print(f"  XNLI val:   {len(xnli_val):,}")

# %% [markdown]
# ## Concatenate all training & validation data

# %%
train_dataset = concatenate_datasets([mnli_train, xnli_train]).shuffle(seed=CONFIG["seed"])
eval_dataset  = concatenate_datasets([mnli_val,   xnli_val  ]).shuffle(seed=CONFIG["seed"])

print(f"\nFinal train: {len(train_dataset):,} rows")
print(f"Final eval:  {len(eval_dataset):,}  rows")

# %% [markdown]
# ## Tokenise

# %%
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
eval_dataset  = eval_dataset.map( tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format( "torch", columns=["input_ids", "attention_mask", "label"])

# %% [markdown]
# ## Model

# %%
print(f"Loading model: {CONFIG['model_name']}")
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# %% [markdown]
# ## Metrics

# %%
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

# %% [markdown]
# ## Train

# %%
training_args = TrainingArguments(
    output_dir                  = CONFIG["output_dir"],
    num_train_epochs            = CONFIG["num_train_epochs"],
    learning_rate               = CONFIG["learning_rate"],
    per_device_train_batch_size = CONFIG["per_device_train_batch_size"],
    per_device_eval_batch_size  = CONFIG["per_device_eval_batch_size"],
    warmup_ratio                = CONFIG["warmup_ratio"],
    weight_decay                = CONFIG["weight_decay"],
    fp16                        = CONFIG["fp16"],
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "accuracy",
    seed                        = CONFIG["seed"],
    report_to                   = "none",
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = eval_dataset,
    tokenizer       = tokenizer,
    compute_metrics = compute_metrics,
)

trainer.train()

# %% [markdown]
# ## Save

# %%
trainer.save_model(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])
print(f"\nModel saved to: {CONFIG['output_dir']}")
