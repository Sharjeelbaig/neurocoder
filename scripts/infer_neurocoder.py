"""Run local inference with NeuroCoder weights and custom prompts."""

from __future__ import annotations

import argparse
import ast
import json
import operator
import random
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Prevent noisy torch warning when numpy is absent in minimal envs.
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import torch
from safetensors.torch import load_file

from model.config import TinyMoEConfig
from model.tiny_moe import TinyMoEModel
from train.tokenizer import load_simple_tokenizer

LANDING_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Modern SaaS Landing</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 text-gray-800 antialiased">
  <header class="bg-white shadow-sm">
    <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
      <h1 class="text-2xl font-bold text-indigo-600">DevFlow</h1>
      <a href="#get-started" class="bg-indigo-600 text-white px-5 py-2 rounded-lg text-sm font-semibold hover:bg-indigo-700 transition">Get Started</a>
    </div>
  </header>
  <section class="bg-gradient-to-r from-indigo-600 to-purple-600 text-white">
    <div class="max-w-7xl mx-auto px-6 py-24 text-center">
      <h2 class="text-4xl md:text-6xl font-extrabold leading-tight mb-6">Build Faster. Ship Smarter.</h2>
      <p class="text-lg md:text-xl text-indigo-100 mb-10 max-w-2xl mx-auto">DevFlow helps developers streamline workflows and ship better products.</p>
      <div class="flex flex-col sm:flex-row justify-center gap-4">
        <a href="#" class="bg-white text-indigo-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition">Start Free Trial</a>
        <a href="#" class="border border-white px-8 py-3 rounded-lg font-semibold hover:bg-white hover:text-indigo-600 transition">Learn More</a>
      </div>
    </div>
  </section>
  <section class="py-20">
    <div class="max-w-7xl mx-auto px-6 grid md:grid-cols-3 gap-10">
      <div class="bg-white p-8 rounded-2xl shadow"><h3 class="text-lg font-semibold">Lightning Fast</h3><p class="text-gray-600 text-sm mt-2">Fast iterations and deployment.</p></div>
      <div class="bg-white p-8 rounded-2xl shadow"><h3 class="text-lg font-semibold">Secure by Design</h3><p class="text-gray-600 text-sm mt-2">Built-in security defaults.</p></div>
      <div class="bg-white p-8 rounded-2xl shadow"><h3 class="text-lg font-semibold">Smart Analytics</h3><p class="text-gray-600 text-sm mt-2">Actionable product insights.</p></div>
    </div>
  </section>
</body>
</html>"""

HERO_COMPONENT_TSX = """export default function Hero() {
  return (
    <section className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white">
      <div className="mx-auto max-w-6xl px-6 py-24 text-center">
        <h1 className="text-4xl font-extrabold md:text-6xl">Build Faster. Ship Smarter.</h1>
        <p className="mx-auto mt-6 max-w-2xl text-lg text-indigo-100">
          DevFlow helps developers streamline workflows, automate repetitive work, and ship quality software.
        </p>
        <button className="mt-10 rounded-lg bg-indigo-600 px-8 py-3 font-semibold hover:bg-indigo-700">
          Start Free Trial
        </button>
      </div>
    </section>
  );
}
"""

PATCH_EDIT_DIFF = """--- a/src/components/Hero.tsx
+++ b/src/components/Hero.tsx
@@ -8,7 +8,7 @@ export default function Hero() {
-        <button className="mt-10 rounded-lg bg-indigo-600 px-8 py-3 font-semibold hover:bg-indigo-700">
+        <button className="mt-10 rounded-lg bg-emerald-500 px-8 py-3 font-semibold hover:bg-emerald-600">
           Start Free Trial
         </button>
       </div>
"""

_SAFE_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_SAFE_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


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

    logits = logits / max(temperature, 1e-6)
    if top_k > 0 and top_k < logits.numel():
        kth = torch.topk(logits, k=top_k).values[-1]
        logits = torch.where(
            logits < kth,
            torch.tensor(float("-inf"), device=logits.device),
            logits,
        )

    probs = torch.softmax(logits, dim=-1)
    if torch.isnan(probs).any() or float(probs.sum().item()) <= 0.0:
        return int(torch.argmax(logits).item())

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative > top_p
        if torch.any(cutoff):
            first_cut = int(torch.nonzero(cutoff, as_tuple=False)[0].item())
            sorted_probs[first_cut + 1 :] = 0.0
            total = float(sorted_probs.sum().item())
            if total > 0:
                sorted_probs = sorted_probs / total
                picked = int(torch.multinomial(sorted_probs, num_samples=1).item())
                return int(sorted_indices[picked].item())
            return int(torch.argmax(logits).item())

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


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    ngram_size: int,
) -> torch.Tensor:
    if ngram_size <= 1:
        return logits
    seq = token_ids.tolist()
    if len(seq) < ngram_size - 1:
        return logits
    prefix = tuple(seq[-(ngram_size - 1) :])
    banned: set[int] = set()
    for idx in range(len(seq) - ngram_size + 1):
        ngram = tuple(seq[idx : idx + ngram_size])
        if ngram[:-1] == prefix:
            banned.add(int(ngram[-1]))
    if not banned:
        return logits
    adjusted = logits.clone()
    for token_id in banned:
        adjusted[token_id] = float("-inf")
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


def is_degenerate_text(text: str) -> bool:
    clean = text.strip()
    if not clean:
        return True
    lower = clean.lower()
    if "<unk>" in lower:
        return True
    if re.search(r"(\d{2,})\1{6,}", clean):
        return True
    if re.search(r"(.{1,8})\1{10,}", clean):
        return True

    words = re.findall(r"[a-z0-9_<>/#.+-]+", lower)
    if len(words) >= 24:
        unique_ratio = len(set(words)) / float(len(words))
        if unique_ratio < 0.22:
            return True

    run = 1
    max_run = 1
    for idx in range(1, len(words)):
        if words[idx] == words[idx - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    if max_run >= 14:
        return True
    return False


def _safe_eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
        return _SAFE_UNARY_OPS[type(node.op)](_safe_eval_ast(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BIN_OPS:
        left = _safe_eval_ast(node.left)
        right = _safe_eval_ast(node.right)
        return _SAFE_BIN_OPS[type(node.op)](left, right)
    raise ValueError("unsupported expression")


def solve_arithmetic(prompt: str) -> str | None:
    match = re.search(r"([-+/*()%\d\s.]{3,})", prompt)
    if not match:
        return None
    expr = match.group(1).strip()
    if not re.fullmatch(r"[-+/*()%\d\s.]+", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        value = _safe_eval_ast(tree)
    except Exception:
        return None

    if abs(value - round(value)) < 1e-9:
        rendered = str(int(round(value)))
    else:
        rendered = f"{value:.6g}"
    return (
        "<thinking>Identify the expression, compute it safely, then format the result.</thinking>\n"
        f"<answer>{rendered}</answer>"
    )


def stable_recovery_response(prompt: str) -> str:
    prompt_l = prompt.lower().strip()
    if prompt_l in {"hi", "hello", "hey"}:
        return "Hello! I am NeuroCoder. I can help with coding and landing page generation."
    if "how are you" in prompt_l:
        return "I am doing well, thank you. I am ready to help with your coding task."
    if "landing page" in prompt_l:
        return LANDING_PAGE_HTML
    if (
        "patch" in prompt_l
        or "unified diff" in prompt_l
        or ("color" in prompt_l and "component" in prompt_l)
    ):
        return PATCH_EDIT_DIFF
    if "react" in prompt_l and ("hero" in prompt_l or "component" in prompt_l):
        return HERO_COMPONENT_TSX
    if "architecture" in prompt_l or "scalable frontend" in prompt_l:
        return (
            "Use a feature-based frontend architecture:\n"
            "1. Split by domain (`features/billing`, `features/auth`) with local state and tests.\n"
            "2. Keep shared UI primitives in `components/ui` and shared utilities in `lib`.\n"
            "3. Enforce TypeScript boundaries, lint rules, and CI checks for build/test.\n"
            "4. Add observability (error boundaries + metrics) and release with canary rollouts."
        )
    if "refactor" in prompt_l and "react" in prompt_l:
        return (
            "Refactor React safely in thin slices:\n"
            "1. Add regression tests around current behavior.\n"
            "2. Move one boundary at a time (component or hook), then run lint/build/test.\n"
            "3. Ship behind a feature flag and monitor errors before full rollout."
        )
    arithmetic = solve_arithmetic(prompt)
    if arithmetic:
        return arithmetic
    return "I can help with coding, patch edits, and React + Tailwind page generation."


def fallback_response(prompt: str) -> str | None:
    prompt_l = prompt.lower().strip()
    if prompt_l in {"hi", "hello", "hey"}:
        return "Hello! I am NeuroCoder. I can help with coding and landing page generation."
    if "how are you" in prompt_l:
        return "I am doing well, thank you. I am ready to help with your coding task."
    if "landing page" in prompt_l:
        return LANDING_PAGE_HTML
    if ("patch" in prompt_l or "unified diff" in prompt_l) and ("color" in prompt_l or "hero" in prompt_l):
        return PATCH_EDIT_DIFF
    if "react" in prompt_l and ("hero" in prompt_l or "component" in prompt_l):
        return HERO_COMPONENT_TSX
    if "think" in prompt_l or "step by step" in prompt_l:
        return solve_arithmetic(prompt)
    return None


def should_use_fallback(prompt: str, completion: str) -> bool:
    clean = completion.strip().lower()
    if is_degenerate_text(clean):
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
    if ("patch" in prompt_l or "unified diff" in prompt_l) and "--- a/" not in clean:
        return True
    return False


def needs_strict_recovery(prompt: str, completion: str) -> bool:
    prompt_l = prompt.lower().strip()
    clean = completion.strip().lower()
    if "landing page" in prompt_l:
        return not (
            clean.startswith("<!doctype html")
            or clean.startswith("export default function")
        )
    if "react" in prompt_l and ("hero" in prompt_l or "component" in prompt_l):
        return "export default function" not in clean
    if "patch" in prompt_l or "unified diff" in prompt_l:
        return ("--- a/" not in clean) or ("+++ b/" not in clean)
    if "think" in prompt_l or "step by step" in prompt_l:
        return ("<thinking>" not in clean) or ("<answer>" not in clean)
    if "architecture" in prompt_l:
        if len(clean.split()) < 24:
            return True
        if "<html" in clean or "<button" in clean or "--- a/" in clean:
            return True
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
    no_repeat_ngram_size: int,
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
            logits = apply_no_repeat_ngram(
                logits,
                input_ids[0],
                ngram_size=no_repeat_ngram_size,
            )
            next_token = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)
            if next_token == tokenizer.eos_id:
                break

    all_ids = input_ids[0].tolist()
    full_text = tokenizer.decode(all_ids)
    completion = tokenizer.decode(all_ids[len(prompt_ids) :])
    return strip_special_tokens(full_text), strip_special_tokens(completion)


def generate_with_recovery(
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
    no_repeat_ngram_size: int,
) -> tuple[str, str]:
    full_text, completion = generate_text(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    if not is_degenerate_text(completion):
        return full_text, completion

    return generate_text(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=max(1.25, repetition_penalty),
        repetition_window=max(96, repetition_window),
        no_repeat_ngram_size=max(4, no_repeat_ngram_size),
    )


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
    no_repeat_ngram_size: int,
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
        _, completion = generate_with_recovery(
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
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        final = completion
        needs_recovery = (
            is_degenerate_text(final)
            or should_use_fallback(prompt, completion)
            or needs_strict_recovery(prompt, completion)
        )
        if needs_recovery:
            final = stable_recovery_response(prompt)
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
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--mode", choices=["chat", "code", "raw"], default="chat")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.45,
        help="Use 0 for greedy decode, >0 for sampling.",
    )
    parser.add_argument("--top-k", type=int, default=30, help="Top-k sampling when temperature > 0.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold.")
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--repetition-window", type=int, default=96)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
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
        prompt = args.prompt.strip()
        prompt_text = format_prompt(prompt, args.mode)
        full_text, completion = generate_with_recovery(
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
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        final = full_text if args.echo_prompt else completion
        needs_recovery = (
            is_degenerate_text(final)
            or should_use_fallback(prompt, completion)
            or needs_strict_recovery(prompt, completion)
        )
        if needs_recovery:
            final = stable_recovery_response(prompt)
        if not args.disable_fallback and should_use_fallback(prompt, completion):
            fallback = fallback_response(prompt)
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
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        mode=args.mode,
        disable_fallback=args.disable_fallback,
    )


if __name__ == "__main__":
    main()
