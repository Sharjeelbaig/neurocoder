"""Tokenizer training utilities with no-pretrained fallback implementation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S")

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


@dataclass(slots=True)
class SimpleTokenizer:
    vocab: dict[str, int]
    id_to_token: list[str]

    @property
    def unk_id(self) -> int:
        return self.vocab["<unk>"]

    @property
    def eos_id(self) -> int:
        return self.vocab["<eos>"]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for token in TOKEN_PATTERN.findall(text):
            if token in self.vocab:
                ids.append(self.vocab[token])
                continue
            # character fallback keeps coverage deterministic.
            for char in token:
                ids.append(self.vocab.get(char, self.unk_id))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        tokens = [self.id_to_token[idx] if 0 <= idx < len(self.id_to_token) else "<unk>" for idx in ids]
        return " ".join(tokens)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "simple_regex_tokenizer",
            "special_tokens": SPECIAL_TOKENS,
            "vocab": self.vocab,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@dataclass(slots=True)
class TokenizerQuality:
    coverage: float
    fragmentation: float
    reserved_ok: bool



def train_simple_tokenizer(
    corpus_files: Iterable[Path],
    vocab_size: int = 32_000,
) -> SimpleTokenizer:
    counts: dict[str, int] = {}

    for path in corpus_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in TOKEN_PATTERN.findall(text):
            counts[token] = counts.get(token, 0) + 1

    max_main_vocab = max(vocab_size - len(SPECIAL_TOKENS), 0)
    most_common = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_main_vocab]

    id_to_token = SPECIAL_TOKENS + [token for token, _ in most_common]
    vocab = {token: idx for idx, token in enumerate(id_to_token)}
    return SimpleTokenizer(vocab=vocab, id_to_token=id_to_token)


def load_simple_tokenizer(path: Path) -> SimpleTokenizer:
    payload = json.loads(path.read_text(encoding="utf-8"))
    vocab = payload["vocab"]
    max_id = max(vocab.values()) if vocab else -1
    id_to_token = ["<unk>"] * (max_id + 1)
    for token, idx in vocab.items():
        id_to_token[idx] = token
    return SimpleTokenizer(vocab=vocab, id_to_token=id_to_token)


def evaluate_tokenizer_quality(tokenizer: SimpleTokenizer, samples: Iterable[str]) -> TokenizerQuality:
    total_tokens = 0
    known_tokens = 0
    produced_ids = 0
    rough_words = 0

    for sample in samples:
        tokens = TOKEN_PATTERN.findall(sample)
        total_tokens += len(tokens)
        rough_words += max(len(sample.split()), 1)
        ids = tokenizer.encode(sample)
        produced_ids += len(ids)
        known_tokens += sum(1 for token in tokens if token in tokenizer.vocab)

    coverage = (known_tokens / total_tokens) if total_tokens else 1.0
    fragmentation = (produced_ids / rough_words) if rough_words else 0.0
    reserved_ok = all(token in tokenizer.vocab for token in SPECIAL_TOKENS)
    return TokenizerQuality(coverage=coverage, fragmentation=fragmentation, reserved_ok=reserved_ok)
