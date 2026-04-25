from __future__ import annotations

import shutil
from pathlib import Path

from nli.build import build_and_cache_nli_dataset
from safety.build import build_and_cache_safety_dataset
from shared.paths import CACHE_ROOT
from tqdm.auto import tqdm

ARTIFACT_ROOT = Path("artifacts")


def build_all_sources() -> None:
    steps = [
        ("Building NLI caches", lambda: build_and_cache_nli_dataset(model_name="xlm-roberta-base", max_length=256)),
        ("Building safety caches", lambda: build_and_cache_safety_dataset(model_name="xlm-roberta-base", max_length=512)),
    ]
    for label, action in tqdm(steps, desc="Fetch/build stages", unit="stage"):
        print(label + " ...")
        action()


def zip_cache_subdir(subdir_name: str) -> Path:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    tokenized_dir = CACHE_ROOT / subdir_name / "tokenized"
    if not tokenized_dir.exists():
        raise FileNotFoundError(f"Tokenized cache directory does not exist: {tokenized_dir}")

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


def main() -> None:
    build_all_sources()
    for subdir_name in tqdm(["nli", "safety"], desc="Archiving caches", unit="cache"):
        zip_cache_subdir(subdir_name)


if __name__ == "__main__":
    main()
