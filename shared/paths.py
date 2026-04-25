from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_ROOT = ROOT_DIR / ".cache" / "xlm_roberta_other"

PATHS: dict[str, Any] = {
    "root_dir": str(ROOT_DIR),
    "cache_root": str(CACHE_ROOT),
    "nli": {
        "raw_cache_dir": str(CACHE_ROOT / "nli" / "raw"),
        "raw_cache_meta": str(CACHE_ROOT / "nli" / "raw" / "dataset.meta.json"),
        "tokenized_cache_dir": str(CACHE_ROOT / "nli" / "tokenized"),
        "tokenized_cache_meta": str(CACHE_ROOT / "nli" / "tokenized" / "dataset.meta.json"),
    },
    "safety": {
        "raw_cache_dir": str(CACHE_ROOT / "safety" / "raw"),
        "raw_cache_meta": str(CACHE_ROOT / "safety" / "raw" / "dataset.meta.json"),
        "tokenized_cache_dir": str(CACHE_ROOT / "safety" / "tokenized"),
        "tokenized_cache_meta": str(CACHE_ROOT / "safety" / "tokenized" / "dataset.meta.json"),
    },
}

for path in [
    CACHE_ROOT,
    Path(PATHS["nli"]["raw_cache_dir"]),
    Path(PATHS["nli"]["tokenized_cache_dir"]),
    Path(PATHS["safety"]["raw_cache_dir"]),
    Path(PATHS["safety"]["tokenized_cache_dir"]),
]:
    path.mkdir(parents=True, exist_ok=True)
