"""Task runtime for page generation and patch-style code edits."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from infer.diff_utils import generate_unified_diff, validate_unified_diff
from infer.schemas import OutputFile, TaskRequest, TaskResponse, ValidationResult
from infer.validators import apply_and_validate

_COLOR_TOKEN_RE = re.compile(
    r"\b(bg|text|border)-(slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-\d{2,3}\b"
)


class ModelAdapter(Protocol):
    """Adapter contract for model-generated outputs."""

    def generate_patch(self, request: TaskRequest, prompt: str) -> str:
        ...

    def generate_page(self, request: TaskRequest, prompt: str) -> list[OutputFile]:
        ...


@dataclass(slots=True)
class HeuristicModelAdapter:
    """Deterministic fallback adapter used before trained checkpoints are available."""

    default_new_color: str = "emerald-500"

    def generate_patch(self, request: TaskRequest, prompt: str) -> str:
        path, before = _select_target_file(request)
        if not path:
            return ""
        after = _apply_color_edit(before, request.instruction, self.default_new_color)
        if after == before:
            return ""
        return generate_unified_diff(path, before, after)

    def generate_page(self, request: TaskRequest, prompt: str) -> list[OutputFile]:
        hero_title = _extract_title(request.instruction)
        app_file = OutputFile(
            path="src/App.tsx",
            content=(
                "import Hero from './components/Hero';\n"
                "import Pricing from './components/Pricing';\n"
                "\n"
                "export default function App(){\n"
                "  return (\n"
                "    <main className=\"min-h-screen bg-slate-950 text-white\">\n"
                "      <Hero />\n"
                "      <Pricing />\n"
                "    </main>\n"
                "  );\n"
                "}\n"
            ),
        )
        hero_file = OutputFile(
            path="src/components/Hero.tsx",
            content=(
                "export default function Hero(){\n"
                "  return (\n"
                "    <section className=\"mx-auto max-w-5xl px-6 py-20\">\n"
                f"      <h1 className=\"text-5xl font-bold\">{hero_title}</h1>\n"
                "      <p className=\"mt-4 text-slate-300\">Built with TinyMoE Coder.</p>\n"
                "      <button className=\"mt-8 rounded-lg bg-emerald-500 px-5 py-3 font-semibold text-white\">\n"
                "        Get Started\n"
                "      </button>\n"
                "    </section>\n"
                "  );\n"
                "}\n"
            ),
        )
        pricing_file = OutputFile(
            path="src/components/Pricing.tsx",
            content=(
                "const tiers = [\n"
                "  { name: 'Starter', price: '$9' },\n"
                "  { name: 'Growth', price: '$29' },\n"
                "  { name: 'Scale', price: '$99' },\n"
                "];\n"
                "\n"
                "export default function Pricing(){\n"
                "  return (\n"
                "    <section className=\"mx-auto grid max-w-5xl gap-6 px-6 pb-24 md:grid-cols-3\">\n"
                "      {tiers.map((tier) => (\n"
                "        <article key={tier.name} className=\"rounded-2xl border border-slate-800 bg-slate-900 p-6\">\n"
                "          <h2 className=\"text-xl font-semibold\">{tier.name}</h2>\n"
                "          <p className=\"mt-2 text-4xl font-bold\">{tier.price}</p>\n"
                "        </article>\n"
                "      ))}\n"
                "    </section>\n"
                "  );\n"
                "}\n"
            ),
        )
        return [app_file, hero_file, pricing_file]


@dataclass(slots=True)
class TaskService:
    adapter: ModelAdapter
    max_repair_attempts: int = 1

    def handle(self, request: TaskRequest) -> TaskResponse:
        prompt = self._compile_prompt(request)

        if request.task_type == "page_generate":
            generated_files = self.adapter.generate_page(request, prompt)
            file_map = {file.path: file.content for file in generated_files}
            _, lint_ok, build_ok, notes = apply_and_validate(file_map, _noop_patch())
            return TaskResponse.create(
                status="ok",
                files=generated_files,
                validation=ValidationResult(
                    apply_ok=True,
                    lint_ok=lint_ok,
                    build_ok=build_ok,
                    notes=notes,
                ),
            )

        patch = self._constrained_patch(self.adapter.generate_patch(request, prompt))
        files_map = {item.path: item.content for item in request.files}
        apply_result, lint_ok, build_ok, notes = apply_and_validate(files_map, patch)

        if apply_result.ok and lint_ok and build_ok:
            return TaskResponse.create(
                status="ok",
                patch=patch,
                validation=ValidationResult(True, lint_ok, build_ok, notes),
            )

        repaired_patch = patch
        repaired_notes = list(notes)
        for _ in range(self.max_repair_attempts):
            repaired_patch = self._repair_patch(request, repaired_patch, repaired_notes)
            apply_result, lint_ok, build_ok, repair_notes = apply_and_validate(files_map, repaired_patch)
            repaired_notes.extend(repair_notes)
            if apply_result.ok and lint_ok and build_ok:
                return TaskResponse.create(
                    status="ok",
                    patch=repaired_patch,
                    validation=ValidationResult(True, lint_ok, build_ok, repaired_notes),
                )

        status = "needs_retry" if apply_result.ok else "failed"
        return TaskResponse.create(
            status=status,
            patch=repaired_patch,
            validation=ValidationResult(apply_result.ok, lint_ok, build_ok, repaired_notes),
        )

    def _compile_prompt(self, request: TaskRequest) -> str:
        file_names = ", ".join(item.path for item in request.files) if request.files else "(no files)"
        return (
            "You are TinyMoE Coder for React+Tailwind.\n"
            f"Task type: {request.task_type}\n"
            f"Instruction: {request.instruction}\n"
            f"Files: {file_names}\n"
            "Output constraints: unified diff for patch edits."
        )

    def _constrained_patch(self, raw_output: str) -> str:
        if not raw_output.strip():
            return ""
        lines = raw_output.splitlines()
        start_idx = None
        for idx, line in enumerate(lines):
            if line.startswith("--- ") or line.startswith("diff --git "):
                start_idx = idx
                break
        if start_idx is None:
            return raw_output.strip() + "\n"
        candidate = "\n".join(lines[start_idx:]).strip() + "\n"
        return candidate

    def _repair_patch(self, request: TaskRequest, patch: str, notes: list[str]) -> str:
        valid, _ = validate_unified_diff(patch)
        if valid:
            return patch

        regenerated = self.adapter.generate_patch(request, self._compile_prompt(request) + "\nRepair invalid patch.")
        if regenerated.strip():
            return self._constrained_patch(regenerated)
        return patch


def _noop_patch() -> str:
    return "--- a/__noop__\n+++ b/__noop__\n@@ -1 +1 @@\n-placeholder\n+placeholder\n"


def _select_target_file(request: TaskRequest) -> tuple[str, str]:
    preferred = [
        item for item in request.files if "hero" in item.path.lower() and item.path.endswith((".tsx", ".jsx"))
    ]
    if preferred:
        return preferred[0].path, preferred[0].content
    if request.files:
        return request.files[0].path, request.files[0].content
    return "", ""


def _extract_title(instruction: str) -> str:
    cleaned = instruction.strip().capitalize()
    if not cleaned:
        return "Launch faster with a focused coding SLM"
    if len(cleaned) > 80:
        cleaned = cleaned[:77] + "..."
    return cleaned


def _apply_color_edit(content: str, instruction: str, fallback_color: str) -> str:
    target_color_match = re.search(
        r"(?:to|into|as)\s+(?:bg-|text-|border-)?([a-z]+-\d{2,3})",
        instruction.lower(),
    )
    new_color = target_color_match.group(1) if target_color_match else fallback_color
    new_token = None

    if "text" in instruction.lower():
        new_token = f"text-{new_color}" if not new_color.startswith("text-") else new_color
    elif "border" in instruction.lower():
        new_token = f"border-{new_color}" if not new_color.startswith("border-") else new_color
    else:
        new_token = f"bg-{new_color}" if not new_color.startswith("bg-") else new_color

    match = _COLOR_TOKEN_RE.search(content)
    if not match:
        return content
    return content.replace(match.group(0), new_token, 1)
