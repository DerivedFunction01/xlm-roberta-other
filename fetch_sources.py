from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from nli.build import build_and_cache_nli_dataset
from safety.build import build_and_cache_safety_dataset
from shared.archive import cache_has_archive, cache_has_tokenized_cache, ensure_cache_archive_extracted
from shared.paths import CACHE_ROOT
from tqdm.auto import tqdm

ARTIFACT_ROOT = Path("artifacts")
TOKENIZED_SUBDIRS = ["nli", "safety"]


def build_all_sources() -> None:
    steps = [
        ("Building NLI caches", lambda: build_and_cache_nli_dataset(model_name="xlm-roberta-base", max_length=256)),
        ("Building safety caches", lambda: build_and_cache_safety_dataset(model_name="xlm-roberta-base", max_length=512)),
    ]
    for label, action in tqdm(steps, desc="Fetch/build stages", unit="stage"):
        print(label + " ...")
        action()


def validate_tokenized_cache(subdir_name: str) -> Path:
    tokenized_dir = CACHE_ROOT / subdir_name / "tokenized"
    if not tokenized_dir.exists():
        raise FileNotFoundError(f"Tokenized cache directory does not exist: {tokenized_dir}")

    meta_path = tokenized_dir / "dataset.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Tokenized cache metadata not found: {meta_path}")

    with meta_path.open(encoding="utf-8") as f:
        json.load(f)

    return tokenized_dir


def zip_cache_subdir(subdir_name: str) -> Path:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    tokenized_dir = validate_tokenized_cache(subdir_name)

    archive_base = ARTIFACT_ROOT / f"xlm_roberta_other_{subdir_name}_cache"
    archive_path = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=CACHE_ROOT,
            base_dir=f"{subdir_name}/tokenized",
        )
    )
    print(f"Created cache archive: {archive_path}")
    return archive_path


def reconcile_cache_subdir(subdir_name: str) -> Path:
    has_cache = cache_has_tokenized_cache(subdir_name)
    has_archive = cache_has_archive(subdir_name)

    if has_cache and has_archive:
        print(f"{subdir_name}: cache and archive already exist, skipping")
        return Path("artifacts") / f"xlm_roberta_other_{subdir_name}_cache.zip"

    if has_cache and not has_archive:
        return zip_cache_subdir(subdir_name)

    if has_archive and not has_cache:
        print(f"{subdir_name}: tokenized cache missing, restoring from archive")
        ensure_cache_archive_extracted(subdir_name)
        return Path("artifacts") / f"xlm_roberta_other_{subdir_name}_cache.zip"

    raise FileNotFoundError(
        f"{subdir_name}: neither tokenized cache nor archive exists. Build the cache first or restore it from artifacts."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package cached tokenized datasets into zip archives.")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build datasets before packaging them. Omit this to only zip existing tokenized caches.",
    )
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=TOKENIZED_SUBDIRS,
        help="Cache subdirectories to package (default: nli safety).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build:
        build_all_sources()
    for subdir_name in tqdm(args.subdirs, desc="Archiving caches", unit="cache"):
        reconcile_cache_subdir(subdir_name)


if __name__ == "__main__":
    main()
