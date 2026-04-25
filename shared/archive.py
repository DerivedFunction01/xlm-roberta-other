from __future__ import annotations

import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from shared.cache import load_dataset_cache
from shared.paths import CACHE_ROOT

ARTIFACT_ROOT = Path("artifacts")
ARCHIVE_NAME_TEMPLATE = "xlm_roberta_other_{subdir_name}_cache.zip"


def archive_path_for(subdir_name: str, artifact_root: Path = ARTIFACT_ROOT) -> Path:
    return artifact_root / ARCHIVE_NAME_TEMPLATE.format(subdir_name=subdir_name)


def ensure_cache_archive_extracted(
    subdir_name: str,
    *,
    artifact_root: Path = ARTIFACT_ROOT,
    cache_root: Path = CACHE_ROOT,
) -> Path:
    target_dir = cache_root / subdir_name
    tokenized_cache_dir = target_dir / "tokenized"
    if load_dataset_cache(tokenized_cache_dir) is not None:
        return target_dir

    archive_path = archive_path_for(subdir_name, artifact_root=artifact_root)
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing cache archive: {archive_path}")

    cache_root = cache_root.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            extracted_path = (cache_root / info.filename).resolve()
            if extracted_path != cache_root and cache_root not in extracted_path.parents:
                raise ValueError(f"Unsafe archive member path: {info.filename}")
        archive.extractall(cache_root)

    if load_dataset_cache(tokenized_cache_dir) is None:
        raise RuntimeError(f"Archive extraction did not restore a usable cache: {tokenized_cache_dir}")
    return target_dir


def ensure_cache_archives_extracted(
    subdir_names: Iterable[str],
    *,
    artifact_root: Path = ARTIFACT_ROOT,
    cache_root: Path = CACHE_ROOT,
) -> list[Path]:
    subdir_names = list(subdir_names)
    if not subdir_names:
        return []

    max_workers = min(len(subdir_names), os.cpu_count() or 1)
    results: list[Path] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                ensure_cache_archive_extracted,
                subdir_name,
                artifact_root=artifact_root,
                cache_root=cache_root,
            ): subdir_name
            for subdir_name in subdir_names
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting caches", unit="cache"):
            results.append(future.result())
    return results
