"""Transformers tokenizer for NeuroCoder remote-code loading."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from transformers import PreTrainedTokenizer

TOKEN_PATTERN = re.compile(r"\s+|[A-Za-z_][A-Za-z0-9_]*|\d+|\S")
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class NeuroCoderTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file: str | None = None, **kwargs: Any) -> None:
        self.vocab: dict[str, int] = {}
        self.id_to_token: list[str] = []

        if vocab_file is not None:
            payload = json.loads(Path(vocab_file).read_text(encoding="utf-8"))
            self.vocab = {str(k): int(v) for k, v in payload.get("vocab", {}).items()}
            max_id = max(self.vocab.values()) if self.vocab else -1
            self.id_to_token = ["<unk>"] * (max_id + 1)
            for token, idx in self.vocab.items():
                self.id_to_token[idx] = token

        if not self.vocab:
            self.vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
            self.id_to_token = SPECIAL_TOKENS[:]

        kwargs.setdefault("bos_token", "<bos>")
        kwargs.setdefault("eos_token", "<eos>")
        kwargs.setdefault("unk_token", "<unk>")
        kwargs.setdefault("pad_token", "<pad>")
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def _tokenize(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token, 0))

    def _convert_id_to_token(self, index: int) -> str:
        if 0 <= index < len(self.id_to_token):
            return self.id_to_token[index]
        return self.unk_token

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0: list[int], token_ids_1: list[int] | None = None) -> list[int]:
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        file_name = "tokenizer.json" if filename_prefix is None else f"{filename_prefix}-tokenizer.json"
        out_path = out_dir / file_name
        payload = {
            "type": "simple_regex_tokenizer",
            "special_tokens": SPECIAL_TOKENS,
            "vocab": self.vocab,
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return (str(out_path),)
