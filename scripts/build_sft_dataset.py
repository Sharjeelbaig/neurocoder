"""Build a large synthetic SFT dataset for NeuroCoder prompt-response training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

GREETINGS = ["hi", "hello", "hey", "good morning", "good evening", "yo"]
HOW_ARE_YOU = ["how are you?", "how are you doing?", "you okay?", "how is it going?"]
BRANDS = ["DevFlow", "NeuroStack", "CodePulse", "LaunchGrid", "ShipCraft", "ByteForge", "StackPilot", "FlowBeam"]
COLORS = ["indigo", "emerald", "sky", "violet", "rose", "teal", "amber", "blue"]
SECTIONS = ["hero", "pricing", "features", "testimonials", "faq", "cta", "footer"]
TECH_STACKS = [
    "React + Tailwind",
    "HTML + Tailwind",
    "React + TypeScript + Tailwind",
    "Vite + React + Tailwind",
]

TOOL_HINTS = [
    "Return unified diff only.",
    "Provide tool-compatible JSON output.",
    "Keep changes minimal and focused.",
    "Preserve formatting and avoid unrelated edits.",
]


LONG_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>{brand} Landing</title>
  <script src=\"https://cdn.tailwindcss.com\"></script>
</head>
<body class=\"bg-gray-50 text-gray-800 antialiased\">
  <header class=\"bg-white shadow-sm\">
    <div class=\"max-w-7xl mx-auto px-6 py-4 flex items-center justify-between\">
      <h1 class=\"text-2xl font-bold text-{color}-600\">{brand}</h1>
      <a href=\"#get-started\" class=\"bg-{color}-600 text-white px-5 py-2 rounded-lg text-sm font-semibold hover:bg-{color}-700 transition\">Get Started</a>
    </div>
  </header>
  <section class=\"bg-gradient-to-r from-{color}-600 to-{color}-800 text-white\">
    <div class=\"max-w-7xl mx-auto px-6 py-24 text-center\">
      <h2 class=\"text-4xl md:text-6xl font-extrabold leading-tight mb-6\">Build Faster. Ship Smarter.</h2>
      <p class=\"text-lg md:text-xl text-{color}-100 mb-10 max-w-2xl mx-auto\">{brand} helps developers streamline workflows and ship better products.</p>
      <div class=\"flex justify-center gap-4\">
        <a href=\"#\" class=\"bg-white text-{color}-600 px-8 py-3 rounded-lg font-semibold\">Start Free Trial</a>
        <a href=\"#\" class=\"border border-white px-8 py-3 rounded-lg font-semibold\">Learn More</a>
      </div>
    </div>
  </section>
  <section class=\"py-20\">
    <div class=\"max-w-7xl mx-auto px-6 grid md:grid-cols-3 gap-10\">
      <div class=\"bg-white p-8 rounded-2xl shadow\"><h3 class=\"text-lg font-semibold\">Lightning Fast</h3><p class=\"text-gray-600 text-sm mt-2\">Fast iterations and deployment.</p></div>
      <div class=\"bg-white p-8 rounded-2xl shadow\"><h3 class=\"text-lg font-semibold\">Secure by Design</h3><p class=\"text-gray-600 text-sm mt-2\">Built-in security defaults.</p></div>
      <div class=\"bg-white p-8 rounded-2xl shadow\"><h3 class=\"text-lg font-semibold\">Smart Analytics</h3><p class=\"text-gray-600 text-sm mt-2\">Actionable product insights.</p></div>
    </div>
  </section>
  <section id=\"pricing\" class=\"bg-gray-100 py-20\">
    <div class=\"max-w-7xl mx-auto px-6 text-center\">
      <h3 class=\"text-3xl font-bold mb-4\">Simple Pricing</h3>
      <div class=\"grid md:grid-cols-3 gap-8 mt-10\">
        <div class=\"bg-white p-8 rounded-2xl shadow\"><h4 class=\"text-lg font-semibold\">Starter</h4><p class=\"text-4xl font-bold\">$0</p></div>
        <div class=\"bg-{color}-600 text-white p-8 rounded-2xl shadow-lg\"><h4 class=\"text-lg font-semibold\">Pro</h4><p class=\"text-4xl font-bold\">$29</p></div>
        <div class=\"bg-white p-8 rounded-2xl shadow\"><h4 class=\"text-lg font-semibold\">Enterprise</h4><p class=\"text-4xl font-bold\">$99</p></div>
      </div>
    </div>
  </section>
  <footer class=\"bg-white border-t py-8\"><div class=\"max-w-7xl mx-auto px-6 text-center text-sm text-gray-500\">© 2026 {brand}. All rights reserved.</div></footer>
</body>
</html>"""


def build_diff(color_from: str, color_to: str, section: str) -> str:
    return (
        f"--- a/src/components/{section.capitalize()}.tsx\n"
        f"+++ b/src/components/{section.capitalize()}.tsx\n"
        "@@ -3,7 +3,7 @@ export default function Component() {\n"
        f"-  <button className=\"rounded-lg bg-{color_from}-600 px-6 py-3 text-white\">Start</button>\n"
        f"+  <button className=\"rounded-lg bg-{color_to}-600 px-6 py-3 text-white\">Start</button>\n"
        " }\n"
    )


def add_chat_examples(records: list[dict[str, str]], n: int) -> None:
    for _ in range(n):
        records.append(
            {
                "prompt": random.choice(GREETINGS),
                "response": random.choice(
                    [
                        "Hello! I am NeuroCoder. I can help with coding and landing page generation.",
                        "Hi! Ready to help with your coding request.",
                        "Hey! Ask me for code generation or patch edits.",
                    ]
                ),
            }
        )
        records.append(
            {
                "prompt": random.choice(HOW_ARE_YOU),
                "response": random.choice(
                    [
                        "I am doing well, thank you. I am ready to help with your coding task.",
                        "I am good and ready to help. Tell me what to build.",
                        "Doing well. Share your task and I will provide code-focused help.",
                    ]
                ),
            }
        )
        stack = random.choice(TECH_STACKS)
        records.append(
            {
                "prompt": "what can you do?",
                "response": (
                    f"I specialize in {stack}, landing page generation, code refactoring, and unified diff patch edits."
                ),
            }
        )


def add_reasoning_examples(records: list[dict[str, str]], n: int) -> None:
    for _ in range(n):
        a = random.randint(2, 40)
        b = random.randint(2, 40)
        c = random.randint(2, 40)
        records.append(
            {
                "prompt": f"Think step by step and compute {a} * {b} + {c}.",
                "response": (
                    f"<thinking>First multiply {a} by {b}. Then add {c} to the product.</thinking>\n"
                    f"<answer>{a * b + c}</answer>"
                ),
            }
        )

        color = random.choice(COLORS)
        brand = random.choice(BRANDS)
        records.append(
            {
                "prompt": "reason about building a landing page and then provide the final code",
                "response": (
                    "<thinking>Need hero, features, pricing, and a clear CTA. Keep semantic structure and Tailwind utility classes.</thinking>\n"
                    f"<answer>{LONG_HTML_TEMPLATE.format(brand=brand, color=color)}</answer>"
                ),
            }
        )


def add_code_examples(records: list[dict[str, str]], n: int) -> None:
    for _ in range(n):
        color = random.choice(COLORS)
        brand = random.choice(BRANDS)
        section = random.choice(SECTIONS)

        records.append(
            {
                "prompt": "generate a landing page",
                "response": LONG_HTML_TEMPLATE.format(brand=brand, color=color),
            }
        )

        records.append(
            {
                "prompt": f"build a {random.choice(TECH_STACKS)} landing page for {brand}",
                "response": (
                    "export default function LandingPage() {\n"
                    "  return (\n"
                    f"    <main className=\"min-h-screen bg-gray-50 text-gray-900\">\n"
                    f"      <header className=\"border-b bg-white\"><div className=\"mx-auto max-w-6xl px-6 py-4\"><h1 className=\"text-2xl font-bold text-{color}-600\">{brand}</h1></div></header>\n"
                    "      <section className=\"mx-auto max-w-6xl px-6 py-20\">\n"
                    "        <h2 className=\"text-5xl font-extrabold\">Build Faster. Ship Smarter.</h2>\n"
                    "      </section>\n"
                    "    </main>\n"
                    "  );\n"
                    "}\n"
                ),
            }
        )

        c1, c2 = random.sample(COLORS, 2)
        records.append(
            {
                "prompt": f"change the button color from {c1}-600 to {c2}-600 in {section} component. {random.choice(TOOL_HINTS)}",
                "response": build_diff(c1, c2, section),
            }
        )
        records.append(
            {
                "prompt": "return tool-compatible action for patch edit",
                "response": (
                    '{"tool":"patch_edit","format":"unified_diff","path":"src/components/Hero.tsx","safety":"validate_apply_then_build"}'
                ),
            }
        )


def add_complex_qa(records: list[dict[str, str]], n: int) -> None:
    questions = [
        "How should I structure a frontend codebase for maintainability?",
        "What is a safe rollout strategy for UI refactors?",
        "How do I avoid regressions when changing Tailwind classes?",
        "How can I make code generation outputs deterministic?",
    ]
    answers = [
        "Use feature-based folders, shared UI primitives, strict linting, and snapshot tests for key components.",
        "Ship in stages: baseline tests, canary release, metrics check, then full rollout with rollback guardrails.",
        "Use focused diffs, run visual checks, and gate changes through lint/build/test before merge.",
        "Constrain output schemas, set fixed decoding parameters, and validate outputs with deterministic checks.",
    ]

    for _ in range(n):
        idx = random.randrange(len(questions))
        records.append({"prompt": questions[idx], "response": answers[idx]})


def write_outputs(records: list[dict[str, str]], out_jsonl: Path, out_txt: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    blocks = [f"User: {row['prompt']}\nAssistant: {row['response']}" for row in records]
    out_txt.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NeuroCoder SFT dataset")
    parser.add_argument("--out-jsonl", default="datasets/curriculum/sft_v2.jsonl")
    parser.add_argument("--out-txt", default="datasets/curriculum/sft_v2.txt")
    parser.add_argument("--chat", type=int, default=10000)
    parser.add_argument("--code", type=int, default=14000)
    parser.add_argument("--reasoning", type=int, default=4000)
    parser.add_argument("--complex", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    records: list[dict[str, str]] = []

    add_chat_examples(records, args.chat)
    add_code_examples(records, args.code)
    add_reasoning_examples(records, args.reasoning)
    add_complex_qa(records, args.complex)

    random.shuffle(records)

    out_jsonl = Path(args.out_jsonl)
    out_txt = Path(args.out_txt)
    write_outputs(records, out_jsonl, out_txt)

    print(f"wrote records={len(records)} jsonl={out_jsonl} txt={out_txt}")


if __name__ == "__main__":
    main()
