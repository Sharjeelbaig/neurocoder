"""Build a high-quality alignment set for final instruction stabilization."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

GREETINGS = ["hi", "hello", "hey", "good morning", "good evening"]

HOW_ARE = [
    "how are you?",
    "how are you doing?",
    "are you okay?",
    "how is it going?",
]

COMPLEX_QA = [
    (
        "How would you design a scalable frontend architecture?",
        "Use feature-based modules, shared UI primitives, strict typing, CI quality gates, and observability. Keep boundaries explicit between domain logic and presentation.",
    ),
    (
        "How do I safely refactor a large React codebase?",
        "Refactor in thin slices: add regression tests, change one component boundary at a time, validate with lint/build/test, then canary deploy before full rollout.",
    ),
    (
        "How can I reduce hallucinations in a coding model?",
        "Use constrained output formats, deterministic validators, strict task schemas, rejection sampling, and targeted instruction tuning on failure cases.",
    ),
]

LANDING_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Modern SaaS Landing</title>
  <script src=\"https://cdn.tailwindcss.com\"></script>
</head>
<body class=\"bg-gray-50 text-gray-800 antialiased\">
  <header class=\"bg-white shadow-sm\">
    <div class=\"max-w-7xl mx-auto px-6 py-4 flex items-center justify-between\">
      <h1 class=\"text-2xl font-bold text-indigo-600\">DevFlow</h1>
      <a href=\"#get-started\" class=\"bg-indigo-600 text-white px-5 py-2 rounded-lg text-sm font-semibold hover:bg-indigo-700 transition\">Get Started</a>
    </div>
  </header>
  <section class=\"bg-gradient-to-r from-indigo-600 to-purple-600 text-white\">
    <div class=\"max-w-7xl mx-auto px-6 py-24 text-center\">
      <h2 class=\"text-4xl md:text-6xl font-extrabold leading-tight mb-6\">Build Faster. Ship Smarter.</h2>
      <p class=\"text-lg md:text-xl text-indigo-100 mb-10 max-w-2xl mx-auto\">DevFlow helps developers streamline workflows, automate tasks, and ship high-quality products.</p>
      <div class=\"flex flex-col sm:flex-row justify-center gap-4\">
        <a href=\"#\" class=\"bg-white text-indigo-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition\">Start Free Trial</a>
        <a href=\"#\" class=\"border border-white px-8 py-3 rounded-lg font-semibold hover:bg-white hover:text-indigo-600 transition\">Learn More</a>
      </div>
    </div>
  </section>
  <section class=\"py-20\">
    <div class=\"max-w-7xl mx-auto px-6 grid md:grid-cols-3 gap-10\">
      <div class=\"bg-white p-8 rounded-2xl shadow\"><h3 class=\"text-lg font-semibold\">Lightning Fast</h3><p class=\"text-gray-600 text-sm mt-2\">Optimized workflows and faster shipping cycles.</p></div>
      <div class=\"bg-white p-8 rounded-2xl shadow\"><h3 class=\"text-lg font-semibold\">Secure by Design</h3><p class=\"text-gray-600 text-sm mt-2\">Security defaults built into every workflow.</p></div>
      <div class=\"bg-white p-8 rounded-2xl shadow\"><h3 class=\"text-lg font-semibold\">Smart Analytics</h3><p class=\"text-gray-600 text-sm mt-2\">Actionable product and performance insights.</p></div>
    </div>
  </section>
  <section id=\"pricing\" class=\"bg-gray-100 py-20\">
    <div class=\"max-w-7xl mx-auto px-6 text-center\">
      <h3 class=\"text-3xl font-bold mb-4\">Simple Pricing</h3>
      <div class=\"grid md:grid-cols-3 gap-8 mt-10\">
        <div class=\"bg-white p-8 rounded-2xl shadow\"><h4 class=\"text-lg font-semibold\">Starter</h4><p class=\"text-4xl font-bold\">$0</p></div>
        <div class=\"bg-indigo-600 text-white p-8 rounded-2xl shadow-lg\"><h4 class=\"text-lg font-semibold\">Pro</h4><p class=\"text-4xl font-bold\">$29</p></div>
        <div class=\"bg-white p-8 rounded-2xl shadow\"><h4 class=\"text-lg font-semibold\">Enterprise</h4><p class=\"text-4xl font-bold\">$99</p></div>
      </div>
    </div>
  </section>
</body>
</html>"""


def build_blocks(repeats: int) -> list[str]:
    blocks: list[str] = []
    for _ in range(repeats):
        for g in GREETINGS:
            blocks.append(
                f"User: {g}\nAssistant: Hello! I am NeuroCoder. I can help with coding, patch edits, and landing page generation."
            )
        for q in HOW_ARE:
            blocks.append(
                f"User: {q}\nAssistant: I am doing well, thank you. I am ready to help with your coding task."
            )

        blocks.append(
            "User: generate a landing page\nAssistant:\n" + LANDING_HTML
        )
        blocks.append(
            "User: create a React + Tailwind landing page\n"
            "Assistant:\n"
            "export default function LandingPage() {\n"
            "  return (\n"
            "    <main className=\"min-h-screen bg-gray-50 text-gray-900\">\n"
            "      <header className=\"border-b bg-white\">\n"
            "        <div className=\"mx-auto max-w-6xl px-6 py-4 flex items-center justify-between\">\n"
            "          <h1 className=\"text-2xl font-bold text-indigo-600\">DevFlow</h1>\n"
            "          <button className=\"rounded-lg bg-indigo-600 px-5 py-2 text-white\">Get Started</button>\n"
            "        </div>\n"
            "      </header>\n"
            "      <section className=\"mx-auto max-w-6xl px-6 py-20 text-center\">\n"
            "        <h2 className=\"text-5xl font-extrabold\">Build Faster. Ship Smarter.</h2>\n"
            "      </section>\n"
            "    </main>\n"
            "  );\n"
            "}\n"
        )

        blocks.append(
            "User: change the button color in Hero component to emerald-500 and return patch\n"
            "Assistant:\n"
            "--- a/src/components/Hero.tsx\n"
            "+++ b/src/components/Hero.tsx\n"
            "@@ -3,7 +3,7 @@ export default function Hero() {\n"
            "-  <button className=\"rounded-lg bg-indigo-600 px-6 py-3 text-white\">Start</button>\n"
            "+  <button className=\"rounded-lg bg-emerald-500 px-6 py-3 text-white\">Start</button>\n"
            " }\n"
        )

        blocks.append(
            "User: think step by step and solve 17 * 8 + 3\n"
            "Assistant: <thinking>Compute 17 * 8 first, then add 3.</thinking>\n<answer>139</answer>"
        )

        for q, a in COMPLEX_QA:
            blocks.append(f"User: {q}\nAssistant: {a}")

        blocks.append(
            "User: return a tool-compatible response for code edit\n"
            "Assistant: {\"tool\":\"patch_edit\",\"format\":\"unified_diff\",\"path\":\"src/components/Hero.tsx\",\"safety\":\"apply_lint_build\"}"
        )

    random.shuffle(blocks)
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build alignment v2 dataset")
    parser.add_argument("--out", default="datasets/curriculum/alignment_v2.txt")
    parser.add_argument("--repeats", type=int, default=700)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blocks = build_blocks(args.repeats)
    out_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"wrote {out_path} blocks={len(blocks)} bytes={out_path.stat().st_size}")


if __name__ == "__main__":
    main()
