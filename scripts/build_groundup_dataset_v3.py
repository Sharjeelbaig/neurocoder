"""Build a ground-up NeuroCoder dataset with strict style/privacy constraints.

This generator intentionally avoids copying external site content. It creates:
1) landing-page generation examples with html/css/js multi-file outputs,
2) patch-edit unified diff examples,
3) compact reasoning-style examples with <Think>/<Answer> blocks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re

AGENCY_NAMES = [
    "Auraloop",
    "CodeHarbor",
    "NovaForge",
    "Shipline",
    "VectorHive",
    "LumenBuild",
    "OrbitCraft",
    "PulseBridge",
]

VERTICALS = [
    "software agency",
    "ai agency",
    "developer tooling startup",
    "product engineering studio",
    "automation consultancy",
]

ACCENTS = ["cyan", "emerald", "sky", "teal", "indigo", "rose"]
HEADLINES = [
    "We build software that ships.",
    "Engineering teams that move fast.",
    "AI products built for production.",
    "From idea to launch in weeks.",
]

THINK_PLANS = [
    "Need a clear hero, trust signals, services, process, faq, and cta. Keep semantic html and accessible structure.",
    "User asked for a modern agency landing page. Provide three files: index.html, styles.css, and script.js with practical interactions.",
    "Use a clean visual hierarchy and responsive layout. Keep css maintainable and js behavior-focused.",
]

PATCH_TARGETS = [
    ("src/components/Hero.tsx", "bg-indigo-600", "bg-emerald-500"),
    ("src/components/Hero.tsx", "hover:bg-indigo-700", "hover:bg-emerald-600"),
    ("src/components/Navbar.tsx", "text-indigo-600", "text-sky-600"),
    ("src/components/Pricing.tsx", "bg-rose-600", "bg-cyan-600"),
]

PROMPT_VARIANTS = [
    "Generate a landing page for {vertical} {agency}",
    "Create a modern landing page for {agency}, a {vertical}",
    "Build a complete landing page for {agency} ({vertical})",
]

THINK_VARIANTS = [
    "Think step by step and then provide code.",
    "Use <Think> and then final implementation.",
    "Return your plan briefly before files.",
]

REASONING_PROMPTS = [
    ("Think step by step and solve 19 * 7 + 4.", "<Answer>137</Answer>"),
    ("Think step by step and solve 1148583*a = 1148360*a - 5352.", "<Answer>-24</Answer>"),
    (
        "If Thorn thought the same thing as Mance, did Thorn agree with Mance?",
        "<Answer>yes</Answer>",
    ),
]


def _sanitize(text: str, banned_terms: list[str]) -> str:
    out = text
    for term in banned_terms:
        if not term:
            continue
        out = re.sub(re.escape(term), "Agency", out, flags=re.IGNORECASE)
    return out


def _build_index_html(agency: str, accent: str, headline: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{agency} - Software Agency</title>
  <meta name="description" content="{agency} builds high-quality digital products for modern teams." />
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <header class="site-header">
    <div class="container row">
      <a class="brand" href="#">{agency}</a>
      <nav class="nav">
        <a href="#services">Services</a>
        <a href="#process">Process</a>
        <a href="#faq">FAQ</a>
        <a href="#contact" class="btn btn-sm">Get Started</a>
      </nav>
    </div>
  </header>

  <main>
    <section class="hero">
      <div class="container">
        <p class="eyebrow">Production-focused agency</p>
        <h1>{headline}</h1>
        <p class="subtitle">We design and ship web products, automation systems, and developer tools with measurable impact.</p>
        <div class="hero-actions">
          <a class="btn" href="#contact">Talk to us</a>
          <a class="btn btn-ghost" href="#services">See services</a>
        </div>
      </div>
      <div class="hero-glow accent-{accent}" aria-hidden="true"></div>
    </section>

    <section id="services" class="section">
      <div class="container grid-3">
        <article class="card"><h3>Web Platforms</h3><p>End-to-end product engineering with maintainable architecture and fast delivery.</p></article>
        <article class="card"><h3>AI Systems</h3><p>Task-specific assistants, retrieval pipelines, and workflow automation.</p></article>
        <article class="card"><h3>Growth Loops</h3><p>Instrumentation, analytics, and iterative experimentation for product teams.</p></article>
      </div>
    </section>

    <section id="process" class="section section-soft">
      <div class="container timeline">
        <div><strong>01 Discovery</strong><p>Clarify goals, constraints, and success metrics.</p></div>
        <div><strong>02 Build</strong><p>Implement quickly with predictable milestones.</p></div>
        <div><strong>03 Launch</strong><p>Deploy, monitor, and iterate from real usage.</p></div>
      </div>
    </section>

    <section id="faq" class="section">
      <div class="container faq">
        <details><summary>How long does a project take?</summary><p>Most engagements ship an initial version in 4-8 weeks.</p></details>
        <details><summary>Can you work with our stack?</summary><p>Yes. We integrate with existing React, Node, and Python systems.</p></details>
      </div>
    </section>
  </main>

  <footer id="contact" class="site-footer">
    <div class="container row">
      <p>Ready to build?</p>
      <a class="btn btn-sm" href="mailto:hello@example.com">Contact us</a>
    </div>
  </footer>

  <script src="script.js"></script>
</body>
</html>"""


def _build_css(accent: str) -> str:
    accent_color = {
        "cyan": "#06b6d4",
        "emerald": "#10b981",
        "sky": "#0ea5e9",
        "teal": "#14b8a6",
        "indigo": "#6366f1",
        "rose": "#f43f5e",
    }[accent]
    return f"""/* NeuroCoder dataset sample stylesheet */
:root {{
  --bg: #0f1220;
  --bg-soft: #171b2c;
  --text: #f3f4f6;
  --muted: #b6bdd3;
  --accent: {accent_color};
  --line: rgba(255, 255, 255, 0.1);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
}}
.container {{ width: min(1100px, 92%); margin: 0 auto; }}
.row {{ display: flex; align-items: center; justify-content: space-between; gap: 1rem; }}
.site-header {{ position: sticky; top: 0; backdrop-filter: blur(8px); background: rgba(15, 18, 32, 0.75); border-bottom: 1px solid var(--line); }}
.site-header .container {{ padding: 1rem 0; }}
.brand {{ color: var(--text); font-weight: 700; text-decoration: none; }}
.nav a {{ color: var(--muted); text-decoration: none; margin-left: 1rem; }}
.btn {{ background: var(--accent); color: white; border-radius: 999px; padding: 0.75rem 1.1rem; text-decoration: none; border: 0; display: inline-block; }}
.btn-sm {{ padding: 0.5rem 0.9rem; }}
.btn-ghost {{ background: transparent; border: 1px solid var(--line); color: var(--text); }}
.hero {{ position: relative; padding: 6rem 0 4rem; overflow: hidden; }}
.hero h1 {{ font-size: clamp(2rem, 4vw, 3.8rem); margin: 0 0 0.8rem; letter-spacing: -0.03em; }}
.eyebrow {{ color: var(--accent); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; }}
.subtitle {{ color: var(--muted); max-width: 60ch; }}
.hero-actions {{ display: flex; gap: 0.8rem; flex-wrap: wrap; margin-top: 1.5rem; }}
.hero-glow {{ position: absolute; right: -8rem; top: -8rem; width: 22rem; height: 22rem; border-radius: 50%; filter: blur(50px); opacity: 0.25; background: var(--accent); }}
.section {{ padding: 4rem 0; }}
.section-soft {{ background: var(--bg-soft); border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }}
.grid-3 {{ display: grid; gap: 1rem; grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.card {{ border: 1px solid var(--line); border-radius: 14px; padding: 1.1rem; background: rgba(255, 255, 255, 0.02); }}
.timeline {{ display: grid; gap: 1rem; grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.faq details {{ border-bottom: 1px solid var(--line); padding: 0.9rem 0; }}
.site-footer {{ border-top: 1px solid var(--line); padding: 2rem 0; }}
@media (max-width: 860px) {{
  .nav a {{ display: none; }}
  .grid-3, .timeline {{ grid-template-columns: 1fr; }}
}}
"""


def _build_js() -> str:
    return """/* NeuroCoder dataset sample script */
(function () {
  var links = document.querySelectorAll('a[href^="#"]');
  links.forEach(function (link) {
    link.addEventListener('click', function (e) {
      var id = link.getAttribute('href');
      if (!id || id.length < 2) return;
      var target = document.querySelector(id);
      if (!target) return;
      e.preventDefault();
      window.scrollTo({ top: target.offsetTop - 72, behavior: 'smooth' });
    });
  });

  var header = document.querySelector('.site-header');
  if (header) {
    window.addEventListener('scroll', function () {
      header.style.borderBottomColor = window.scrollY > 18 ? 'rgba(255,255,255,0.16)' : 'rgba(255,255,255,0.10)';
    }, { passive: true });
  }
})();"""


def _build_landing_response(agency: str, accent: str, headline: str) -> str:
    index_html = _build_index_html(agency=agency, accent=accent, headline=headline)
    styles_css = _build_css(accent=accent)
    script_js = _build_js()
    think = random.choice(THINK_PLANS)
    return (
        f"<Think>{think}</Think>\n"
        "Create a file `index.html` and paste this code:\n"
        "```html\n"
        f"{index_html}\n"
        "```\n\n"
        "Create a file `styles.css` and paste this code:\n"
        "```css\n"
        f"{styles_css}\n"
        "```\n\n"
        "Create a file `script.js` and paste this code:\n"
        "```js\n"
        f"{script_js}\n"
        "```"
    )


def _build_patch_response(path: str, old: str, new: str) -> str:
    patch = (
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -7,7 +7,7 @@ export default function Component() {\n"
        f"-  <button className=\"rounded-lg {old} px-6 py-3 text-white\">Start</button>\n"
        f"+  <button className=\"rounded-lg {new} px-6 py-3 text-white\">Start</button>\n"
        " }\n"
    )
    return (
        "<Think>Need a minimal and safe edit. Only change the requested class token and preserve structure.</Think>\n"
        f"{patch}"
    )


def _build_records(
    *,
    landing_count: int,
    patch_count: int,
    reasoning_count: int,
    banned_terms: list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for _ in range(landing_count):
        agency = random.choice(AGENCY_NAMES)
        vertical = random.choice(VERTICALS)
        accent = random.choice(ACCENTS)
        headline = random.choice(HEADLINES)
        prompt = random.choice(PROMPT_VARIANTS).format(vertical=vertical, agency=agency)
        prompt = f"{prompt}. {random.choice(THINK_VARIANTS)}"
        response = _build_landing_response(agency=agency, accent=accent, headline=headline)
        rows.append(
            {
                "prompt": _sanitize(prompt, banned_terms),
                "response": _sanitize(response, banned_terms),
                "task_type": "page_generate",
            }
        )

    for _ in range(patch_count):
        path, old, new = random.choice(PATCH_TARGETS)
        prompt = f"Change `{old}` to `{new}` in `{path}` and return unified diff only."
        response = _build_patch_response(path=path, old=old, new=new)
        rows.append(
            {
                "prompt": _sanitize(prompt, banned_terms),
                "response": _sanitize(response, banned_terms),
                "task_type": "patch_edit",
            }
        )

    for idx in range(reasoning_count):
        prompt, answer = REASONING_PROMPTS[idx % len(REASONING_PROMPTS)]
        response = (
            "<Think>Identify the objective, compute or infer the result, then return a concise final answer.</Think>\n"
            f"{answer}"
        )
        rows.append(
            {
                "prompt": _sanitize(prompt, banned_terms),
                "response": _sanitize(response, banned_terms),
                "task_type": "reasoning",
            }
        )

    random.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NeuroCoder ground-up dataset v3")
    parser.add_argument("--out-jsonl", default="datasets/groundup/neurocoder_v3.jsonl")
    parser.add_argument("--out-txt", default="datasets/groundup/neurocoder_v3.txt")
    parser.add_argument("--manifest", default="datasets/groundup/neurocoder_v3_manifest.json")
    parser.add_argument("--landing-count", type=int, default=8000)
    parser.add_argument("--patch-count", type=int, default=6000)
    parser.add_argument("--reasoning-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ban-term",
        action="append",
        default=["inferencia"],
        help="Case-insensitive term to scrub from prompts/responses.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    banned_terms = [term.strip() for term in args.ban_term if term.strip()]

    rows = _build_records(
        landing_count=args.landing_count,
        patch_count=args.patch_count,
        reasoning_count=args.reasoning_count,
        banned_terms=banned_terms,
    )

    out_jsonl = Path(args.out_jsonl).resolve()
    out_txt = Path(args.out_txt).resolve()
    manifest_path = Path(args.manifest).resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    blocks = [f"User: {row['prompt']}\nAssistant: {row['response']}" for row in rows]
    out_txt.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

    counts: dict[str, int] = {"page_generate": 0, "patch_edit": 0, "reasoning": 0}
    for row in rows:
        counts[row["task_type"]] = counts.get(row["task_type"], 0) + 1

    manifest = {
        "records": len(rows),
        "counts": counts,
        "out_jsonl": str(out_jsonl),
        "out_txt": str(out_txt),
        "bytes_jsonl": out_jsonl.stat().st_size,
        "bytes_txt": out_txt.stat().st_size,
        "banned_terms": banned_terms,
        "seed": args.seed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
