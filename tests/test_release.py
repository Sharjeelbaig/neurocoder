from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from release.hf_package import build_hf_package
from release.ollama_package import build_ollama_package
from release.quantize import write_dummy_gguf


class ReleaseTests(unittest.TestCase):
    def test_package_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text('{"vocab": {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}}', encoding="utf-8")

            hf = build_hf_package(
                output_dir=root / "hf",
                tokenizer_json=tokenizer,
                model_config={"context_length": 4096},
                model_weights=None,
                license_text="MIT",
            )
            self.assertIn("config.json", hf.files_written)
            self.assertTrue((root / "hf" / "model.safetensors").exists())

            gguf = root / "model.gguf"
            write_dummy_gguf(gguf)
            ollama = build_ollama_package(root / "ollama", gguf)
            self.assertIn("Modelfile", ollama.files_written)
            self.assertTrue((root / "ollama" / "Modelfile").exists())


if __name__ == "__main__":
    unittest.main()
