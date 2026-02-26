"""Build high-priority task-fix alignment data for NeuroCoder v4."""

from __future__ import annotations

import argparse
from pathlib import Path
import random


GREETING_BLOCKS = [
    "User: hi\nAssistant: Hello! I am NeuroCoder. I can help with coding, patch edits, and landing page generation.",
    "User: hello\nAssistant: Hello! I am NeuroCoder. I can help with coding, patch edits, and landing page generation.",
    "User: hey\nAssistant: Hello! I am NeuroCoder. I can help with coding, patch edits, and landing page generation.",
    "User: how are you?\nAssistant: I am doing well, thank you. I am ready to help with your coding task.",
]


PY_FUNC = """def reverse_string(value: str) -> str:
    \"\"\"Return the reversed version of the input string.\"\"\"
    return value[::-1]
"""


PY_BLOCKS = [
    f"User: Write a python function to reverse a string\nAssistant:\n{PY_FUNC}",
    f"User: give python code to reverse a string\nAssistant:\n{PY_FUNC}",
]


LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Velocity Landing</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 text-gray-800 antialiased">
  <header class="bg-white shadow-sm">
    <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
      <h1 class="text-2xl font-bold text-indigo-600">Velocity</h1>
      <a href="#get-started" class="bg-indigo-600 text-white px-5 py-2 rounded-lg text-sm font-semibold hover:bg-indigo-700 transition">Get Started</a>
    </div>
  </header>
  <section class="bg-gradient-to-r from-indigo-600 to-purple-600 text-white">
    <div class="max-w-7xl mx-auto px-6 py-24 text-center">
      <h2 class="text-4xl md:text-6xl font-extrabold leading-tight mb-6">Build Faster. Ship Smarter.</h2>
      <p class="text-lg md:text-xl text-indigo-100 mb-10 max-w-2xl mx-auto">Velocity helps teams streamline workflows and ship better products.</p>
      <div class="flex flex-col sm:flex-row justify-center gap-4">
        <a href="#" class="bg-white text-indigo-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition">Start Free Trial</a>
        <a href="#" class="border border-white px-8 py-3 rounded-lg font-semibold hover:bg-white hover:text-indigo-600 transition">Learn More</a>
      </div>
    </div>
  </section>
</body>
</html>"""


LANDING_BLOCKS = [
    f"User: Generate a landing page for marketing agency titled Velocity Landing\nAssistant:\n{LANDING_HTML}",
    f"User: Generate a landing page for marketing agency, but title should be \"Velocity Landing\"\nAssistant:\n{LANDING_HTML}",
    f"User: generate a landing page for marketing agency\nAssistant:\n{LANDING_HTML}",
]


PATCH_DIFF = """--- a/src/components/Hero.tsx
+++ b/src/components/Hero.tsx
@@ -8,7 +8,7 @@ export default function Hero() {
-        <button className="mt-10 rounded-lg bg-indigo-600 px-8 py-3 font-semibold hover:bg-indigo-700">
+        <button className="mt-10 rounded-lg bg-blue-500 px-8 py-3 font-semibold hover:bg-blue-600">
           Start Free Trial
         </button>
       </div>"""


PATCH_BLOCKS = [
    f"User: Provide a unified diff to change hero button color to blue-500\nAssistant:\n{PATCH_DIFF}",
    f"User: change hero button color to blue-500 and return unified diff\nAssistant:\n{PATCH_DIFF}",
]


REASON_BLOCKS = [
    "User: Think step by step and solve 17 * 8 + 3\nAssistant: <Think>Compute 17 * 8 first, then add 3.</Think>\n<Answer>139</Answer>",
]


def build_blocks(repeats: int) -> list[str]:
    chunks = GREETING_BLOCKS + PY_BLOCKS + LANDING_BLOCKS + PATCH_BLOCKS + REASON_BLOCKS
    rows: list[str] = []
    for _ in range(repeats):
        rows.extend(chunks)
    random.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build task-fix alignment set v4")
    parser.add_argument("--out", default="datasets/curriculum/alignment_taskfix_v4.txt")
    parser.add_argument("--repeats", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_blocks(args.repeats)
    out.write_text("\n\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {out} blocks={len(rows)} bytes={out.stat().st_size}")


if __name__ == "__main__":
    main()

