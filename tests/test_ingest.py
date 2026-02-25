from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from data.ingest import ingest_sources


class IngestTests(unittest.TestCase):
    def test_ingest_with_license_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir(parents=True)
            (root / "LICENSE").write_text("MIT License", encoding="utf-8")
            (root / "src").mkdir()
            (root / "src" / "Hero.tsx").write_text(
                "export default function Hero(){return <button className='bg-blue-500'>Hi</button>}",
                encoding="utf-8",
            )

            out = Path(tmp) / "out"
            summary = ingest_sources([root], out)

            self.assertEqual(summary.accepted_files, 1)
            manifest_lines = (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(manifest_lines), 1)
            payload = json.loads(manifest_lines[0])
            self.assertEqual(payload["spdx"], "MIT")
            self.assertTrue(payload["react_like"])


if __name__ == "__main__":
    unittest.main()
