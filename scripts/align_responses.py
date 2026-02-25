"""Targeted full-model alignment on critical prompt-response samples."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import load_file, save_file

from model.config import TinyMoEConfig
from model.tiny_moe import TinyMoEModel
from train.tokenizer import load_simple_tokenizer



def load_model(model_dir: Path) -> tuple[TinyMoEModel, TinyMoEConfig, object]:
    cfg = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
    config = TinyMoEConfig(
        vocab_size=int(cfg["vocab_size"]),
        context_length=int(cfg["context_length"]),
        hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        num_experts=int(cfg["num_experts"]),
        top_k=int(cfg.get("top_k", 2)),
        moe_every_n_layers=int(cfg.get("moe_every_n_layers", 2)),
        ffn_multiplier=int(cfg.get("ffn_multiplier", 4)),
        capacity_factor_train=float(cfg.get("capacity_factor_train", 1.25)),
        capacity_factor_infer=float(cfg.get("capacity_factor_infer", 1.0)),
    )
    tokenizer = load_simple_tokenizer(model_dir / "tokenizer.json")
    model = TinyMoEModel(config)
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=False)
    return model, config, tokenizer



def build_sequences(tokenizer, dataset_path: Path, seq_len: int) -> list[list[int]]:
    raw = dataset_path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in raw.split("\n\n") if block.strip()]
    seqs: list[list[int]] = []

    for block in blocks:
        ids = tokenizer.encode(block)
        if not ids:
            continue
        ids.append(tokenizer.eos_id)
        if len(ids) <= seq_len:
            seqs.append(ids)
            continue

        for start in range(0, len(ids), seq_len - 32):
            chunk = ids[start : start + seq_len]
            if len(chunk) >= 16:
                seqs.append(chunk)
            if start + seq_len >= len(ids):
                break

    return seqs



def make_batch(seqs: list[list[int]], batch_size: int, seq_len: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    sampled = random.choices(seqs, k=batch_size)
    x_rows: list[list[int]] = []
    y_rows: list[list[int]] = []
    for seq in sampled:
        seq = seq[:seq_len]
        pad = seq_len - len(seq)
        x = seq + [pad_id] * pad
        y = x[1:] + [-100]
        if pad > 0:
            y[-pad:] = [-100] * pad
        x_rows.append(x)
        y_rows.append(y)
    return torch.tensor(x_rows, dtype=torch.long), torch.tensor(y_rows, dtype=torch.long)



def main() -> None:
    parser = argparse.ArgumentParser(description="Align NeuroCoder on critical prompts")
    parser.add_argument("--model-dir", default="artifacts/trained_v5")
    parser.add_argument("--dataset", default="datasets/curriculum/critical_alignment.txt")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = Path(args.model_dir).resolve()
    dataset_path = Path(args.dataset).resolve()

    model, _, tokenizer = load_model(model_dir)
    device = torch.device(args.device)
    model.to(device)
    model.train()

    seqs = build_sequences(tokenizer, dataset_path, seq_len=args.seq_len)
    if not seqs:
        raise SystemExit("No alignment sequences built")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        x, y = make_batch(seqs, args.batch_size, args.seq_len, tokenizer.vocab["<pad>"])
        x = x.to(device)
        y = y.to(device)
        out = model(input_ids=x, labels=y)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            print(f"align_step={step} loss={float(loss.detach().cpu().item()):.6f}")

    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if "lm_head.weight" in state and "token_embed.weight" in state:
        state["lm_head.weight"] = state["lm_head.weight"].clone()
    save_file(state, str(model_dir / "model.safetensors"), metadata={"aligned": "true"})

    summary = {
        "model_dir": str(model_dir),
        "dataset": str(dataset_path),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "num_sequences": len(seqs),
    }
    (model_dir / "alignment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
