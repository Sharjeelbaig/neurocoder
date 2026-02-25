"""Train TinyMoE from scratch on local corpus and export safetensors weights."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.config import TinyMoEConfig
from model.tiny_moe import TORCH_AVAILABLE, TinyMoEModel
from train.preprocess import pack_sequences, tokenize_corpus
from train.tokenizer import train_simple_tokenizer

if not TORCH_AVAILABLE:  # pragma: no cover
    raise SystemExit("PyTorch is required. Install with: pip install .[train]")

import torch
from safetensors.torch import save_file


def _collect_corpus_files(source_dir: Path) -> list[Path]:
    allowed = {".py", ".md", ".toml", ".json", ".tsx", ".ts", ".jsx", ".js", ".css", ".txt"}
    blocked_dirs = {".git", "__pycache__", "artifacts", "datasets", "benchmarks/results"}
    files: list[Path] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if any(part in blocked_dirs for part in path.parts):
            continue
        if path.suffix.lower() in allowed:
            files.append(path)
    return files


def _build_batch(
    token_sequences: list[list[int]],
    batch_size: int,
    seq_len: int,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    picked = random.choices(token_sequences, k=batch_size)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []

    for seq in picked:
        seq_trim = seq[:seq_len]
        pad = seq_len - len(seq_trim)
        inputs = seq_trim + [pad_id] * pad
        labels = inputs[1:] + [-100]
        if pad > 0:
            labels[-pad:] = [-100] * pad
        input_rows.append(inputs)
        label_rows.append(labels)

    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    labels = torch.tensor(label_rows, dtype=torch.long, device=device)
    return input_ids, labels


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyMoE from scratch and export safetensors")
    parser.add_argument("--source-dir", default=str(ROOT))
    parser.add_argument("--out-dir", default="artifacts/trained")
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_dir = Path(args.source_dir).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_files = _collect_corpus_files(source_dir)
    if not corpus_files:
        raise SystemExit(f"No corpus files found under {source_dir}")

    tokenizer = train_simple_tokenizer(corpus_files, vocab_size=args.vocab_size)
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.to_json(tokenizer_path)

    texts = [path.read_text(encoding="utf-8", errors="ignore") for path in corpus_files]
    tokenized = tokenize_corpus(texts, tokenizer, append_eos=True)
    packed = pack_sequences(tokenized, seq_len=args.seq_len)
    train_seqs = [seq.input_ids for seq in packed if len(seq.input_ids) > 8]
    if not train_seqs:
        raise SystemExit("No training sequences available after packing")

    device = _resolve_device(args.device)
    config = TinyMoEConfig(
        vocab_size=len(tokenizer.vocab),
        context_length=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_experts=args.num_experts,
        top_k=args.top_k,
        moe_every_n_layers=2,
    )

    model = TinyMoEModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    with torch.no_grad():
        init_input, init_labels = _build_batch(
            train_seqs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            pad_id=tokenizer.vocab["<pad>"],
            device=device,
        )
        init_loss = float(model(input_ids=init_input, labels=init_labels)["loss"].detach().cpu().item())

    start = time.time()
    last_loss = init_loss
    for step in range(1, args.steps + 1):
        input_ids, labels = _build_batch(
            train_seqs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            pad_id=tokenizer.vocab["<pad>"],
            device=device,
        )
        output = model(input_ids=input_ids, labels=labels)
        loss = output["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            print(f"step={step} loss={last_loss:.5f}")

    elapsed = time.time() - start

    model_path = out_dir / "model.safetensors"
    state_dict = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    # Break tied-weight shared storage for safetensors serialization.
    if "lm_head.weight" in state_dict and "token_embed.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["lm_head.weight"].clone()
    metadata = {
        "trained_from_scratch": "true",
        "steps": str(args.steps),
        "initial_loss": f"{init_loss:.6f}",
        "final_loss": f"{last_loss:.6f}",
    }
    save_file(state_dict, str(model_path), metadata=metadata)

    config_payload = {
        "architectures": ["TinyMoEModel"],
        "model_type": "tinymoe",
        "vocab_size": config.vocab_size,
        "context_length": config.context_length,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "num_experts": config.num_experts,
        "top_k": config.top_k,
        "moe_every_n_layers": config.moe_every_n_layers,
        "ffn_multiplier": config.ffn_multiplier,
        "capacity_factor_train": config.capacity_factor_train,
        "capacity_factor_infer": config.capacity_factor_infer,
    }
    (out_dir / "model_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = {
        "source_dir": str(source_dir),
        "corpus_files": len(corpus_files),
        "train_sequences": len(train_seqs),
        "device": str(device),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "initial_loss": init_loss,
        "final_loss": last_loss,
        "elapsed_seconds": elapsed,
        "model_path": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "tokenizer_path": str(tokenizer_path),
        "config": config_payload,
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
