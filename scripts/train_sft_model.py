"""Train NeuroCoder with supervised instruction tuning (prompt -> response)."""

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

import torch
from safetensors.torch import save_file

from model.config import TinyMoEConfig
from model.tiny_moe import TinyMoEModel
from train.tokenizer import train_simple_tokenizer



def _read_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        prompt = str(payload.get("prompt", "")).strip()
        response = str(payload.get("response", "")).strip()
        if prompt and response:
            rows.append({"prompt": prompt, "response": response})
    return rows



def _build_tokenizer_corpus(sft_rows: list[dict[str, str]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "tokenizer_corpus.txt"
    with corpus_path.open("w", encoding="utf-8") as fh:
        for row in sft_rows:
            fh.write(f"User: {row['prompt']}\nAssistant: {row['response']}\n\n")
    return [corpus_path]



def _build_examples(
    rows: list[dict[str, str]],
    tokenizer,
    seq_len: int,
    max_examples: int,
) -> list[tuple[list[int], list[int]]]:
    examples: list[tuple[list[int], list[int]]] = []

    for row in rows:
        prompt_text = f"User: {row['prompt']}\nAssistant: "
        response_text = row["response"]

        prompt_ids = tokenizer.encode(prompt_text)
        response_ids = tokenizer.encode(response_text) + [tokenizer.eos_id]

        if not prompt_ids or not response_ids:
            continue

        sequence = prompt_ids + response_ids
        if len(sequence) < 2:
            continue

        # Causal LM objective with prompt masking:
        # input = sequence[:-1], target = sequence[1:].
        input_ids = sequence[:-1]
        labels = sequence[1:]

        # Mask prompt continuation tokens while still training first response token.
        prompt_mask_len = max(len(prompt_ids) - 1, 0)
        labels = ([-100] * prompt_mask_len) + labels[prompt_mask_len:]

        if len(input_ids) > seq_len:
            input_ids = input_ids[:seq_len]
            labels = labels[:seq_len]

        if all(v == -100 for v in labels):
            continue

        examples.append((input_ids, labels))
        if len(examples) >= max_examples:
            break

    return examples



def _make_batch(
    examples: list[tuple[list[int], list[int]]],
    batch_size: int,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = random.choices(examples, k=batch_size)
    max_len = max(len(x[0]) for x in chosen)

    x_rows: list[list[int]] = []
    y_rows: list[list[int]] = []

    for input_ids, labels in chosen:
        pad = max_len - len(input_ids)
        x_rows.append(input_ids + [pad_id] * pad)
        y_rows.append(labels + [-100] * pad)

    x = torch.tensor(x_rows, dtype=torch.long, device=device)
    y = torch.tensor(y_rows, dtype=torch.long, device=device)
    return x, y



def _resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def _generate(
    model: TinyMoEModel,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 120,
) -> str:
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.unk_id]
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=x)
            next_id = int(out["logits"][0, -1].argmax().item())
            x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == tokenizer.eos_id:
                break

    decoded = tokenizer.decode(x[0].tolist())
    completion = decoded[len(prompt) :] if decoded.startswith(prompt) else decoded
    return completion.replace("<eos>", "").replace("<pad>", "").strip()



def main() -> None:
    parser = argparse.ArgumentParser(description="Train NeuroCoder SFT model")
    parser.add_argument("--dataset", default="datasets/curriculum/sft_v2.jsonl")
    parser.add_argument("--out-dir", default="artifacts/trained_sft_v1")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--seq-len", type=int, default=384)
    parser.add_argument("--hidden-size", type=int, default=320)
    parser.add_argument("--num-layers", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1.8e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-examples", type=int, default=120000)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(dataset_path)
    if not rows:
        raise SystemExit(f"No SFT rows found in {dataset_path}")

    tokenizer_corpus = _build_tokenizer_corpus(rows, out_dir)
    tokenizer = train_simple_tokenizer(tokenizer_corpus, vocab_size=args.vocab_size)
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.to_json(tokenizer_path)

    examples = _build_examples(rows, tokenizer, args.seq_len, max_examples=args.max_examples)
    if not examples:
        raise SystemExit("No training examples after tokenization")

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

    x0, y0 = _make_batch(examples, args.batch_size, tokenizer.vocab["<pad>"], device)
    with torch.no_grad():
        init_loss = float(model(input_ids=x0, labels=y0)["loss"].detach().cpu().item())

    start = time.time()
    last_loss = init_loss
    model.train()

    for step in range(1, args.steps + 1):
        xb, yb = _make_batch(examples, args.batch_size, tokenizer.vocab["<pad>"], device)
        out = model(input_ids=xb, labels=yb)
        loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        last_loss = float(loss.detach().cpu().item())
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            print(f"step={step} loss={last_loss:.6f}")

    elapsed = time.time() - start

    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if "lm_head.weight" in state and "token_embed.weight" in state:
        state["lm_head.weight"] = state["lm_head.weight"].clone()

    model_path = out_dir / "model.safetensors"
    save_file(
        state,
        str(model_path),
        metadata={
            "format": "pt",
            "stage": "sft",
            "trained_from_scratch": "true",
            "steps": str(args.steps),
            "initial_loss": f"{init_loss:.6f}",
            "final_loss": f"{last_loss:.6f}",
        },
    )

    cfg_payload = {
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
    (out_dir / "model_config.json").write_text(json.dumps(cfg_payload, indent=2, sort_keys=True), encoding="utf-8")

    eval_prompts = [
        "User: hi\nAssistant: ",
        "User: how are you?\nAssistant: ",
        "User: generate a landing page\nAssistant:\n",
        "User: Think step by step and compute 17 * 8 + 3.\nAssistant: ",
    ]
    eval_outputs = {prompt: _generate(model, tokenizer, prompt, device, max_new_tokens=140) for prompt in eval_prompts}

    summary = {
        "dataset": str(dataset_path),
        "records": len(rows),
        "examples": len(examples),
        "device": str(device),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "initial_loss": init_loss,
        "final_loss": last_loss,
        "elapsed_seconds": elapsed,
        "model_path": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "tokenizer_path": str(tokenizer_path),
        "config": cfg_payload,
        "eval_outputs": eval_outputs,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
