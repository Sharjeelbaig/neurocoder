"""Build a synthetic curriculum corpus for coding-chat alignment."""

from __future__ import annotations

import argparse
from pathlib import Path
import random


GREETINGS = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "good evening",
    "hiya",
]

GREETING_RESPONSES = [
    "Hello! I am NeuroCoder. I can help with coding tasks.",
    "Hi! Ready to help with React, Tailwind, and patch edits.",
    "Hey! Tell me what code you want to generate or modify.",
]

HOW_ARE_YOU_RESPONSES = [
    "I am doing well. I am ready to help with your code request.",
    "Doing great. Share your coding task and I will help.",
    "I am good, thanks. What should we build today?",
]

BRANDS = [
    "DevFlow",
    "NeuroStack",
    "CodePulse",
    "LaunchGrid",
    "ShipCraft",
    "ByteForge",
]

PRIMARY_COLORS = [
    "indigo",
    "emerald",
    "sky",
    "violet",
    "rose",
    "teal",
]


def build_html_landing(brand: str, color: str) -> str:
    return f"""<!DOCTYPE html>
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
      <p class=\"text-lg md:text-xl text-{color}-100 mb-10 max-w-2xl mx-auto\">{brand} helps developers automate workflows and ship reliable products.</p>
      <a href=\"#pricing\" class=\"bg-white text-{color}-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition\">Start Free Trial</a>
    </div>
  </section>
  <section id=\"features\" class=\"py-20\">
    <div class=\"max-w-7xl mx-auto px-6 grid md:grid-cols-3 gap-10\">
      <article class=\"bg-white p-8 rounded-2xl shadow\"><h3 class=\"text-lg font-semibold\">Fast Builds</h3><p class=\"text-gray-600 text-sm mt-2\">Faster build cycles with smart automation.</p></article>
      <article class=\"bg-white p-8 rounded-2xl shadow\"><h3 class=\"text-lg font-semibold\">Secure by Design</h3><p class=\"text-gray-600 text-sm mt-2\">Security-first developer workflows.</p></article>
      <article class=\"bg-white p-8 rounded-2xl shadow\"><h3 class=\"text-lg font-semibold\">Live Analytics</h3><p class=\"text-gray-600 text-sm mt-2\">Real-time operational visibility.</p></article>
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


def build_react_landing(brand: str, color: str) -> str:
    return f"""export default function LandingPage() {{
  return (
    <main className=\"min-h-screen bg-gray-50 text-gray-900\">
      <header className=\"border-b bg-white\">
        <div className=\"mx-auto max-w-6xl px-6 py-4 flex items-center justify-between\">
          <h1 className=\"text-2xl font-bold text-{color}-600\">{brand}</h1>
          <button className=\"rounded-lg bg-{color}-600 px-5 py-2 text-white\">Get Started</button>
        </div>
      </header>
      <section className=\"mx-auto max-w-6xl px-6 py-20 text-center\">
        <h2 className=\"text-5xl font-extrabold\">Build Faster. Ship Smarter.</h2>
        <p className=\"mt-4 text-lg text-gray-600\">{brand} helps teams ship reliable software.</p>
      </section>
    </main>
  );
}}"""


def build_patch_example(color_from: str, color_to: str) -> str:
    return f"""--- a/src/components/Hero.tsx
+++ b/src/components/Hero.tsx
@@ -3,7 +3,7 @@ export default function Hero() {{
-      <button className=\"rounded-lg bg-{color_from}-600 px-6 py-3 text-white\">Start</button>
+      <button className=\"rounded-lg bg-{color_to}-600 px-6 py-3 text-white\">Start</button>
 }}"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NeuroCoder curriculum corpus")
    parser.add_argument("--out", default="datasets/curriculum/coding_chat_v1.txt")
    parser.add_argument("--chat-samples", type=int, default=6000)
    parser.add_argument("--landing-samples", type=int, default=2000)
    parser.add_argument("--patch-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blocks: list[str] = []

    for _ in range(args.chat_samples):
        greeting = random.choice(GREETINGS)
        response = random.choice(GREETING_RESPONSES)
        blocks.append(f"User: {greeting}\nAssistant: {response}")

        blocks.append(
            "User: how are you?\n"
            f"Assistant: {random.choice(HOW_ARE_YOU_RESPONSES)}"
        )

        blocks.append(
            "User: can you use tools?\n"
            "Assistant: Yes. I can return tool-compatible outputs like unified diffs and structured JSON."
        )

        blocks.append(
            "User: what is your speciality?\n"
            "Assistant: I specialize in coding tasks, especially React + Tailwind landing pages and code patch edits."
        )

    for _ in range(args.landing_samples):
        brand = random.choice(BRANDS)
        color = random.choice(PRIMARY_COLORS)
        html = build_html_landing(brand, color)
        react = build_react_landing(brand, color)

        blocks.append(
            "User: generate a landing page\n"
            "Assistant:\n" + html
        )

        blocks.append(
            "User: create a React + Tailwind landing page\n"
            "Assistant:\n" + react
        )

        blocks.append(
            "User: generate a landing page with reasoning\n"
            "Assistant: <thinking>Need a modern SaaS layout with hero, features, pricing, CTA, footer.</thinking>\n"
            "<answer>" + html + "</answer>"
        )

    for _ in range(args.patch_samples):
        c1, c2 = random.sample(PRIMARY_COLORS, 2)
        patch = build_patch_example(c1, c2)
        blocks.append(
            f"User: change button color from {c1}-600 to {c2}-600 in Hero component\n"
            "Assistant:\n" + patch
        )
        blocks.append(
            "User: provide tool-compatible output for this edit\n"
            "Assistant: {\"tool\":\"patch_edit\",\"path\":\"src/components/Hero.tsx\",\"format\":\"unified_diff\"}"
        )

    text = "\n\n".join(blocks) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path} bytes={out_path.stat().st_size} blocks={len(blocks)}")


if __name__ == "__main__":
    main()
