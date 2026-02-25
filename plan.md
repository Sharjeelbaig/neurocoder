NeuroCoder (Small Coder SLM) - 12-Week From-Scratch Launch Plan
Summary
Build a from-scratch, CPU-first small language model specialized for:

Landing page generation in React + Tailwind.
Code-edit requests returned as unified diff patches (example: “change primary color in Hero component”).
The system will use a sparse MoE decoder model plus deterministic helpers to hit quality targets with minimal active parameters and local latency under 3 seconds.

Scope and Success Criteria
In scope: Frontend-only generation/editing for React+Tailwind landing pages.
Out of scope: General backend coding, multi-framework support, multilingual instructions, broad coding parity.
v1 success gates:
Quantized runtime fits <=4 GB RAM on CPU-first deployment.
Median edit request latency <3s for typical patch tasks.
Narrow benchmark score reaches >=95% of Qwen 7B baseline on this project’s custom task suite.
Patch apply rate >=92%.
Build/lint pass after patch >=85%.
Model Architecture (Decision-Complete)
Model family: Decoder-only causal Transformer with sparse MoE FFNs.
Tokenizer: Train from scratch, 32k vocab, BPE/Unigram hybrid, code-aware normalization.
Context length: 4096 tokens.
Core config:
20 transformer blocks.
Hidden size 1024, 16 heads, RoPE positional encoding.
MoE placement: every other block (10 MoE blocks + 10 dense blocks).
Per MoE block: 8 experts, top-2 routing.
Router: load-balancing auxiliary loss + z-loss stabilization.
Capacity factors: train 1.25, inference 1.0.
Estimated params:
Total params: ~1.0B-1.2B.
Active params/token: ~170M-220M.
Inference format:
4-bit weight quantization for deployment.
KV cache quantized to minimize RAM pressure.
Target deployed artifact: <=2 GB weights, total process memory <=4 GB.
Data Strategy (No External LLM Use)
License policy: permissive only (MIT/BSD/Apache/CC), SPDX-filtered at ingest.
Data domains:
React+Tailwind landing page repos/templates/components.
UI style systems and design token examples under permissive licenses.
Programmatic instruction/patch pair generation (non-LLM synthetic only).
Dataset splits:
Pretraining corpus: ~15B-25B tokens, domain-focused code/text mix.
Instruction corpus: 3M-6M examples, format-constrained.
Validation/test frozen early and never reused for training.
Synthetic generators (deterministic):
Color/theme edits.
Component-level style/layout adjustments.
Section insert/remove/reorder.
Accessibility and responsiveness patch tasks.
Training Pipeline
Framework: PyTorch + custom MoE modules (open-source tooling allowed, no pretrained artifacts).
Infrastructure: hybrid.
On-prem: fast iteration, unit-scale ablations.
Cloud GPU rentals: main pretraining and final tuning runs.
Stages:
Stage A - Data ingest + legal filter + dedup + tokenizer training.
Stage B - Base pretraining from random init on domain corpus.
Stage C - Supervised instruction tuning (full-model tuning, no PEFT).
Stage D - Preference optimization using rule-based scoring pairs (no external reward model).
Stage E - Quantization-aware validation and CPU latency tuning.
Training controls:
Deterministic seeds and run manifests.
Checkpoint every fixed token interval.
Expert load monitoring with hard alarms for collapse/skew.
Inference System Design
Generation style: constrained decoding to unified diff schema for edit mode.
Deterministic helper stack (required):
Prompt compiler enforcing task schema and file constraints.
Patch parser/validator (git apply --check equivalent logic + syntax checks).
React+Tailwind lint/build validator.
Auto-repair pass with the same model if first patch fails validation.
Modes:
page_generate: new landing page scaffold/files.
patch_edit: minimal diff over provided files.
patch_validate: static validation only.
Public APIs / Interfaces / Types
Input schema (TaskRequest):
task_type: page_generate | patch_edit
instruction: string
files: array of { path: string, content: string }
constraints: { framework: "react-tailwind", output: "unified_diff" }
Output schema (TaskResponse):
status: ok | needs_retry | failed
patch: unified diff string (for patch_edit)
files: array of full files (for page_generate)
validation: { apply_ok: bool, build_ok: bool, lint_ok: bool, notes: string[] }
Training sample schema (TrainExample):
id, source_license, task_type, instruction, context_files, target_patch, metadata.
Baseline comparator interface:
Freeze benchmark runner that executes both models on identical prompts and checks exact metrics.
Evaluation and Benchmark Suite
Suite composition (NarrowCoderBench-v1):
400 landing-page generation prompts.
500 patch-edit tasks on realistic repo snapshots.
100 adversarial/edge tasks (ambiguous instructions, conflicting style constraints).
Metrics:
Patch apply success.
Lint/build/test pass rates.
Structural correctness for unified diff format.
Visual/layout checks for generated pages (deterministic screenshot diff thresholds).
Latency and memory on target CPU hardware.
Parity rule:
Compare against Qwen 7B baseline only for evaluation.
Pass if aggregate weighted score >=95% of baseline on this narrow suite.
12-Week Execution Plan
Weeks 1-2: bootstrap repo, data contracts, license filtering, tokenizer training, benchmark harness.
Weeks 3-5: initial MoE pretraining runs + stability debugging (router balance, expert utilization).
Weeks 6-7: scale pretraining to target tokens, freeze candidate checkpoints.
Weeks 8-9: instruction tuning + rule-based preference optimization; strict diff-format specialization.
Week 10: quantization + CPU runtime optimization + helper stack integration.
Week 11: benchmark against baseline, regression triage, failure taxonomy fixes.
Week 12: hardening, reproducibility pass, deployment package, go/no-go report.
Test Cases and Scenarios
Model correctness:
“Change button color in Hero component” produces minimal unified diff touching only relevant files.
“Create SaaS landing page with pricing/testimonials/CTA” generates valid React+Tailwind files.
Format safety:
Invalid diff lines are rejected by validator and repaired automatically.
Ambiguous instruction returns needs_retry with actionable notes.
Build integrity:
Generated/edited code passes parser + lint + build checks.
Performance:
Median and p95 latency under target on CPU hardware.
Memory stays below 4 GB during realistic workloads.
MoE reliability:
Expert load distribution remains within configured skew bounds across evaluation set.
Assumptions and Defaults
“From scratch” means no pretrained weights/tokenizers or external LLM-generated training data.
Open-source training/inference libraries are allowed.
Hybrid infra is available with enough cloud GPU hours within $20k-$50k.
On-prem resources are sufficient for smaller iteration jobs.
English-only instructions in v1.
React+Tailwind only in v1.
Unified diff is the single edit output contract.
Qwen baseline is used only as an evaluation reference, never as training signal.