from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from train.preprocess import pack_sequences, tokenize_corpus
from train.tokenizer import evaluate_tokenizer_quality, load_simple_tokenizer, train_simple_tokenizer


class TokenizerTests(unittest.TestCase):
    def test_train_save_load_and_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            file_a = Path(tmp) / "a.txt"
            file_a.write_text("export default function Hero(){ return <button className='bg-blue-500'>Hi</button> }")
            file_b = Path(tmp) / "b.txt"
            file_b.write_text("tailwind css bg-emerald-500 px-4 py-2")

            tokenizer = train_simple_tokenizer([file_a, file_b], vocab_size=128)
            out = Path(tmp) / "tokenizer.json"
            tokenizer.to_json(out)
            loaded = load_simple_tokenizer(out)

            ids = loaded.encode("bg-blue-500")
            self.assertGreater(len(ids), 0)

            quality = evaluate_tokenizer_quality(loaded, [file_a.read_text(), file_b.read_text()])
            self.assertGreater(quality.coverage, 0.5)
            self.assertTrue(quality.reserved_ok)

            sequences = tokenize_corpus(["hello world", "color token"], loaded)
            packed = pack_sequences(sequences, seq_len=4)
            self.assertGreaterEqual(len(packed), 1)


if __name__ == "__main__":
    unittest.main()
