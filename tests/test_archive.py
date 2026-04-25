from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from shared.archive import ensure_cache_archive_extracted


class ArchiveTests(unittest.TestCase):
    def test_extracts_archive_into_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            artifact_root = tmp_path / "artifacts"
            cache_root = tmp_path / ".cache" / "xlm_roberta_other"
            archive_dir = artifact_root / "xlm_roberta_other_nli_cache.zip"
            artifact_root.mkdir(parents=True, exist_ok=True)
            cache_root.mkdir(parents=True, exist_ok=True)
            nli_dir = cache_root / "nli"
            tokenized_dir = nli_dir / "tokenized"
            tokenized_dir.mkdir(parents=True, exist_ok=True)

            source_dir = tmp_path / "source"
            (source_dir / "nli" / "tokenized").mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"text": ["hello"], "label": [1]}).to_parquet(
                source_dir / "nli" / "tokenized" / "train.parquet",
                index=False,
            )

            with zipfile.ZipFile(archive_dir, "w") as archive:
                archive.write(
                    source_dir / "nli" / "tokenized" / "train.parquet",
                    arcname="nli/tokenized/train.parquet",
                )

            restored = ensure_cache_archive_extracted(
                "nli",
                artifact_root=artifact_root,
                cache_root=cache_root,
            )

            self.assertEqual(restored, nli_dir)
            self.assertTrue((tokenized_dir / "train.parquet").exists())


if __name__ == "__main__":
    unittest.main()
