from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hf_compat.tokenization_neurocoder import NeuroCoderTokenizer


def _build_tokenizer() -> NeuroCoderTokenizer:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tokenizer.json"
        vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "U": 4,
            "s": 5,
            "e": 6,
            "r": 7,
            ":": 8,
            " ": 9,
            "\n": 10,
            "A": 11,
            "i": 12,
            "t": 13,
            "n": 14,
            "h": 15,
            "o": 16,
            "?": 17,
        }
        payload = {
            "type": "simple_regex_tokenizer",
            "special_tokens": ["<pad>", "<bos>", "<eos>", "<unk>"],
            "vocab": vocab,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        tok = NeuroCoderTokenizer(vocab_file=str(path))
    return tok


class HFTokenizerGuardTests(unittest.TestCase):
    def test_prompt_normalization(self) -> None:
        tok = _build_tokenizer()
        wrapped = tok._normalize_inference_prompt("hi")
        self.assertEqual(wrapped, "User: hi\nAssistant: ")

    def test_decode_guard_greeting(self) -> None:
        tok = _build_tokenizer()
        text = "User: hi\nAssistant: gibberishgibberishgibberish"
        fixed = tok._apply_decode_guard(text)
        self.assertIn("Hello! I am NeuroCoder", fixed)

    def test_decode_guard_landing(self) -> None:
        tok = _build_tokenizer()
        text = "User: Generate a landing page for marketing agency titled Velocity Landing\nAssistant: nonsense"
        fixed = tok._apply_decode_guard(text)
        self.assertIn("<!DOCTYPE html>", fixed)
        self.assertIn("<title>Velocity Landing</title>", fixed)


if __name__ == "__main__":
    unittest.main()
