from __future__ import annotations

import shutil
from pathlib import Path

from nli.build import build_and_cache_nli_dataset
from safety.build import build_and_cache_safety_dataset
from shared.paths import CACHE_ROOT

ARTIFACT_ROOT = Path("artifacts")
ARCHIVE_NAME = "xlm_roberta_other_cache"


def build_all_sources() -> None:
    print("Building NLI caches ...")
    build_and_cache_nli_dataset(
        model_name="xlm-roberta-base",
        max_length=256,
    )

    print("Building safety caches ...")
    build_and_cache_safety_dataset(
        model_name="xlm-roberta-base",
        max_length=256,
    )


def zip_cache_root() -> Path:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    archive_base = ARTIFACT_ROOT / ARCHIVE_NAME
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=CACHE_ROOT))
    print(f"Created cache archive: {archive_path}")
    return archive_path


def main() -> None:
    build_all_sources()
    zip_cache_root()


if __name__ == "__main__":
    main()
