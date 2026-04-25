from __future__ import annotations

import shutil
from pathlib import Path

from nli.build import build_and_cache_nli_dataset
from safety.build import build_and_cache_safety_dataset
from shared.paths import CACHE_ROOT

ARTIFACT_ROOT = Path("artifacts")


def build_all_sources() -> None:
    print("Building NLI caches ...")
    build_and_cache_nli_dataset(
        model_name="xlm-roberta-base",
        max_length=256,
    )

    print("Building safety caches ...")
    build_and_cache_safety_dataset(
        model_name="xlm-roberta-base",
        max_length=512,
    )


def zip_cache_subdir(subdir_name: str) -> Path:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    source_dir = CACHE_ROOT / subdir_name
    if not source_dir.exists():
        raise FileNotFoundError(f"Cache directory does not exist: {source_dir}")

    archive_base = ARTIFACT_ROOT / f"xlm_roberta_other_{subdir_name}_cache"
    archive_path = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=CACHE_ROOT,
            base_dir=subdir_name,
        )
    )
    print(f"Created cache archive: {archive_path}")
    return archive_path


def main() -> None:
    build_all_sources()
    zip_cache_subdir("nli")
    zip_cache_subdir("safety")


if __name__ == "__main__":
    main()
