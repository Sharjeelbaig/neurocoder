"""Sequence packing and masking utilities for causal language modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from train.tokenizer import SimpleTokenizer


@dataclass(slots=True)
class PackedSequence:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]



def tokenize_corpus(texts: Iterable[str], tokenizer: SimpleTokenizer, append_eos: bool = True) -> list[list[int]]:
    sequences: list[list[int]] = []
    for text in texts:
        ids = tokenizer.encode(text)
        if append_eos:
            ids.append(tokenizer.eos_id)
        sequences.append(ids)
    return sequences


def pack_sequences(token_sequences: Iterable[list[int]], seq_len: int = 4096) -> list[PackedSequence]:
    packed: list[PackedSequence] = []
    buffer: list[int] = []

    def flush(chunk: list[int]) -> None:
        labels = chunk[1:] + [-100]
        packed.append(
            PackedSequence(
                input_ids=chunk,
                labels=labels,
                attention_mask=[1] * len(chunk),
            )
        )

    for sequence in token_sequences:
        buffer.extend(sequence)
        while len(buffer) >= seq_len:
            chunk = buffer[:seq_len]
            flush(chunk)
            buffer = buffer[seq_len:]

    if buffer:
        flush(buffer)

    return packed


def pad_batch(batch: list[PackedSequence], pad_id: int) -> PackedSequence:
    max_len = max(len(item.input_ids) for item in batch)
    input_ids: list[int] = []
    labels: list[int] = []
    attention_mask: list[int] = []

    for item in batch:
        pad = max_len - len(item.input_ids)
        input_ids.extend(item.input_ids + [pad_id] * pad)
        labels.extend(item.labels + [-100] * pad)
        attention_mask.extend(item.attention_mask + [0] * pad)

    return PackedSequence(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
