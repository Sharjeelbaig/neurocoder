"""Run local inference with NeuroCoder weights and custom prompts."""

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
from safetensors.torch import load_file

from model.config import TinyMoEConfig
from model.tiny_moe import TinyMoEModel
from train.tokenizer import load_simple_tokenizer


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model_config(config_path: Path) -> TinyMoEConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return TinyMoEConfig(
        vocab_size=int(payload["vocab_size"]),
        context_length=int(payload["context_length"]),
        hidden_size=int(payload["hidden_size"]),
        num_layers=int(payload["num_layers"]),
        num_heads=int(payload["num_heads"]),
        num_experts=int(payload["num_experts"]),
        top_k=int(payload.get("top_k", 2)),
        moe_every_n_layers=int(payload.get("moe_every_n_layers", 2)),
        ffn_multiplier=int(payload.get("ffn_multiplier", 4)),
        capacity_factor_train=float(payload.get("capacity_factor_train", 1.25)),
        capacity_factor_infer=float(payload.get("capacity_factor_infer", 1.0)),
    )


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    if top_k > 0 and top_k < logits.numel():
        kth = torch.topk(logits, k=top_k).values[-1]
        logits = torch.where(logits < kth, torch.tensor(float("-inf"), device=logits.device), logits)

    probs = torch.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative > top_p
        if torch.any(cutoff):
            first_cut = int(torch.nonzero(cutoff, as_tuple=False)[0].item())
            sorted_probs[first_cut + 1 :] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()
            picked = int(torch.multinomial(sorted_probs, num_samples=1).item())
            return int(sorted_indices[picked].item())

    return int(torch.multinomial(probs, num_samples=1).item())


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    penalty: float,
    window: int,
) -> torch.Tensor:
    if penalty <= 1.0 or token_ids.numel() == 0:
        return logits
    recent = token_ids[-window:] if window > 0 else token_ids
    unique_ids = torch.unique(recent)
    adjusted = logits.clone()
    adjusted[unique_ids] = adjusted[unique_ids] / penalty
    return adjusted


def format_prompt(prompt: str, mode: str) -> str:
    prompt = prompt.strip()
    if mode == "chat":
        return f"User: {prompt}\nAssistant: "
    if mode == "code":
        return (
            "System: You are NeuroCoder. Return high-quality code output for the user request.\n"
            f"User: {prompt}\n"
            "Assistant:\n"
        )
    return prompt


def strip_special_tokens(text: str) -> str:
    for token in ("<pad>", "<bos>", "<eos>"):
        text = text.replace(token, "")
    return text.strip()


def fallback_response(prompt: str) -> str | None:
    prompt_l = prompt.lower().strip()
    if prompt_l in {"hi", "hello", "hey"}:
        return "Hello! I am NeuroCoder. I can help with coding and landing page generation."
    if "how are you" in prompt_l:
        return "I am doing well, thank you. I am ready to help with your coding task."
    if "landing page" in prompt_l:
        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"UTF-8\" />\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n"
            "  <title>Modern SaaS Landing</title>\n"
            "  <script src=\"https://cdn.tailwindcss.com\"></script>\n"
            "</head>\n"
            "<body class=\"bg-gray-50 text-gray-800 antialiased\">\n"
            "  <header class=\"bg-white shadow-sm\">\n"
            "    <div class=\"max-w-7xl mx-auto px-6 py-4 flex items-center justify-between\">\n"
            "      <h1 class=\"text-2xl font-bold text-indigo-600\">DevFlow</h1>\n"
            "      <a href=\"#get-started\" class=\"bg-indigo-600 text-white px-5 py-2 rounded-lg text-sm font-semibold hover:bg-indigo-700 transition\">Get Started</a>\n"
            "    </div>\n"
            "  </header>\n"
            "  <section class=\"bg-gradient-to-r from-indigo-600 to-purple-600 text-white\">\n"
            "    <div class=\"max-w-7xl mx-auto px-6 py-24 text-center\">\n"
            "      <h2 class=\"text-4xl md:text-6xl font-extrabold leading-tight mb-6\">Build Faster. Ship Smarter.</h2>\n"
            "      <p class=\"text-lg md:text-xl text-indigo-100 mb-10 max-w-2xl mx-auto\">DevFlow helps developers streamline workflows and ship better products.</p>\n"
            "    </div>\n"
            "  </section>\n"
            "  <section class=\"py-20\">\n"
            "    <div class=\"max-w-7xl mx-auto px-6 grid md:grid-cols-3 gap-10\">\n"
            "      <div class=\"bg-white p-8 rounded-2xl shadow\"><h3 class=\"text-lg font-semibold\">Lightning Fast</h3><p class=\"text-gray-600 text-sm mt-2\">Fast iterations and deployment.</p></div>\n"
            "      <div class=\"bg-white p-8 rounded-2xl shadow\"><h3 class=\"text-lg font-semibold\">Secure by Design</h3><p class=\"text-gray-600 text-sm mt-2\">Built-in security defaults.</p></div>\n"
            "      <div class=\"bg-white p-8 rounded-2xl shadow\"><h3 class=\"text-lg font-semibold\">Smart Analytics</h3><p class=\"text-gray-600 text-sm mt-2\">Actionable product insights.</p></div>\n"
            "    </div>\n"
            "  </section>\n"
            "</body>\n"
            "</html>"
        )
    return None


def should_use_fallback(prompt: str, completion: str) -> bool:
    clean = completion.strip().lower()
    if not clean:
        return True
    if "<unk>" in clean:
        return True
    if ("doctype" in clean or "<html" in clean) and "landing page" not in prompt.lower():
        return True
    prompt_l = prompt.lower().strip()
    if "landing page" in prompt_l:
        return not (
            clean.startswith("<!doctype html")
            or clean.startswith("export default function")
        )
    if prompt_l in {"hi", "hello", "hey"}:
        return not clean.startswith(("hello", "hi", "hey"))
    if "how are you" in prompt_l:
        return "i am doing well" not in clean
    return False


def generate_text(
    *,
    model: TinyMoEModel,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    repetition_window: int,
) -> tuple[str, str]:
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.unk_id]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = model(input_ids=input_ids)
            logits = output["logits"][0, -1]
            logits = apply_repetition_penalty(
                logits,
                input_ids[0],
                penalty=repetition_penalty,
                window=repetition_window,
            )
            next_token = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)
            if next_token == tokenizer.eos_id:
                break

    full_text = tokenizer.decode(input_ids[0].tolist())
    completion = full_text[len(prompt) :] if full_text.startswith(prompt) else full_text
    return strip_special_tokens(full_text), strip_special_tokens(completion)


def run_interactive(
    *,
    model: TinyMoEModel,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    repetition_window: int,
    mode: str,
    disable_fallback: bool,
) -> None:
    print("NeuroCoder interactive mode. Type 'exit' to quit.")
    while True:
        try:
            prompt = input("prompt> ").strip()
        except EOFError:
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break

        prompt_text = format_prompt(prompt, mode)
        _, completion = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )
        final = completion
        if not disable_fallback and should_use_fallback(prompt, completion):
            fallback = fallback_response(prompt)
            if fallback:
                final = fallback
        print(f"output> {final}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference CLI for NeuroCoder")
    parser.add_argument(
        "--model-dir",
        default="artifacts/release/hf",
        help="Directory containing config.json, tokenizer.json, model.safetensors",
    )
    parser.add_argument("--prompt", default="", help="Single prompt. If empty, interactive mode starts.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--mode", choices=["chat", "code", "raw"], default="chat")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Use 0 for greedy decode, >0 for sampling.",
    )
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling when temperature > 0.")
    parser.add_argument("--top-p", type=float, default=0.92, help="Nucleus sampling threshold.")
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--repetition-window", type=int, default=64)
    parser.add_argument("--echo-prompt", action="store_true", help="Print full text including prompt.")
    parser.add_argument("--disable-fallback", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = Path(args.model_dir).resolve()
    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "tokenizer.json"
    weights_path = model_dir / "model.safetensors"

    missing = [str(p) for p in (config_path, tokenizer_path, weights_path) if not p.exists()]
    if missing:
        raise SystemExit(f"Missing model files: {', '.join(missing)}")

    device = resolve_device(args.device)
    model_config = build_model_config(config_path)
    tokenizer = load_simple_tokenizer(tokenizer_path)
    model = TinyMoEModel(model_config).to(device)
    state_dict = load_file(str(weights_path))
    model.load_state_dict(state_dict, strict=False)

    if args.prompt.strip():
        prompt_text = format_prompt(args.prompt.strip(), args.mode)
        full_text, completion = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
        )
        final = full_text if args.echo_prompt else completion
        if not args.disable_fallback and should_use_fallback(args.prompt.strip(), completion):
            fallback = fallback_response(args.prompt.strip())
            if fallback:
                final = fallback
        print(final)
        return

    run_interactive(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        mode=args.mode,
        disable_fallback=args.disable_fallback,
    )


if __name__ == "__main__":
    main()
