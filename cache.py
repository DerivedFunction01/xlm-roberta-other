from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk

from io_utils import write_json_atomic

DEFAULT_CACHE_META_NAME = "dataset.meta.json"


def save_dataset_cache(
    dataset: DatasetDict,
    cache_dir: str | Path,
    *,
    meta_path: str | Path | None = None,
    meta: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> None:
    """Persist a DatasetDict to disk with an optional manifest."""
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing cache dir: {cache_dir}")
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_dir))

    if meta is not None:
        resolved_meta_path = Path(meta_path) if meta_path is not None else cache_dir / DEFAULT_CACHE_META_NAME
        write_json_atomic(str(resolved_meta_path), meta)


def load_dataset_cache(
    cache_dir: str | Path,
    *,
    meta_path: str | Path | None = None,
    expected_meta: dict[str, Any] | None = None,
) -> DatasetDict | None:
    """Load a cached DatasetDict if the manifest matches the expected metadata."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return None

    resolved_meta_path = Path(meta_path) if meta_path is not None else cache_dir / DEFAULT_CACHE_META_NAME
    if expected_meta is not None:
        if not resolved_meta_path.exists():
            return None
        try:
            with resolved_meta_path.open(encoding="utf-8") as f:
                cached_meta = json.load(f)
        except Exception:
            return None
        for key, value in expected_meta.items():
            if cached_meta.get(key) != value:
                return None

    try:
        loaded = load_from_disk(str(cache_dir))
    except Exception:
        return None
    if not isinstance(loaded, DatasetDict):
        return None
    return loaded

