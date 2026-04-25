from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_ROOT = ROOT_DIR / ".cache" / "xlm_roberta_other"

PATHS: dict[str, Any] = {
    "root_dir": str(ROOT_DIR),
    "cache_root": str(CACHE_ROOT),
    "nli": {
        "cache_dir": str(CACHE_ROOT / "nli"),
        "cache_meta": str(CACHE_ROOT / "nli" / "dataset.meta.json"),
    },
    "safety": {
        "cache_dir": str(CACHE_ROOT / "safety"),
        "cache_meta": str(CACHE_ROOT / "safety" / "dataset.meta.json"),
    },
}

for path in [
    CACHE_ROOT,
    Path(PATHS["nli"]["cache_dir"]),
    Path(PATHS["safety"]["cache_dir"]),
]:
    path.mkdir(parents=True, exist_ok=True)

