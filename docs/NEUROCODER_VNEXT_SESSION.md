# NeuroCoder vNext Full Coding Session

This runbook is the long-form, end-to-end execution flow to move NeuroCoder from unstable demo behavior to a production narrow-scope coding model.

## Objectives

1. Keep scope narrow: landing pages + patch edits + concise coding assistant behavior.
2. Make HF usage standard (`AutoTokenizer` + `AutoModelForCausalLM`) and fast enough for local use.
3. Reduce looped/hallucinated outputs with better data, training stages, and decoding defaults.
4. Keep privacy guardrails: do not include proprietary site/source content in dataset artifacts.

## Hard Constraints

1. No private/proprietary content in datasets.
2. Redact banned terms (default includes `inferencia`) in generated data.
3. Maintain from-scratch training path (no PEFT/LoRA dependency).
4. Publish only reproducible artifacts (`config.json`, `model.safetensors`, tokenizer files, remote-code files).

## Dataset Strategy

### Ground-Up Dataset (Primary)

Use `scripts/build_groundup_dataset_v3.py`:

1. Generates multi-file landing-page responses (`index.html`, `styles.css`, `script.js`).
2. Generates unified-diff patch edits for React/Tailwind components.
3. Generates compact reasoning examples with `<Think>` and `<Answer>`.
4. Applies banned-term scrubbing to avoid leaking sensitive branding/context.

Output:

1. `datasets/groundup/neurocoder_v3.jsonl`
2. `datasets/groundup/neurocoder_v3.txt`
3. `datasets/groundup/neurocoder_v3_manifest.json`

### Optional External Blend (Secondary)

Use as additional supervision only after license checks and format cleanup:

1. [smirki/UIGEN-T1.1-TAILWIND](https://huggingface.co/datasets/smirki/UIGEN-T1.1-TAILWIND)
2. [crownelius/GLM-5.0-25000x](https://huggingface.co/datasets/crownelius/GLM-5.0-25000x)
3. [crownelius/Opus-4.6-Reasoning-3300x](https://huggingface.co/datasets/crownelius/Opus-4.6-Reasoning-3300x)

Builder:

1. `scripts/build_hf_mix_dataset.py`

## Model/Runtime Improvements in vNext

### Inference Speed

1. Added KV cache support to HF remote-code model (`past_key_values` path).
2. Enabled `use_cache` in exported config/generation defaults.

### HF Compatibility

1. Safetensors packaging now ensures metadata includes `format=pt`.
2. Added `generation_config.json` in HF package for sane default decoding.
3. Kept `AutoConfig`/`AutoModelForCausalLM`/`AutoTokenizer` `auto_map` compatible.

### Output Stability

1. Default generation config now includes repetition + n-gram anti-loop controls.
2. Regression suite checks both fallback-disabled and fallback-enabled behavior.

## Execution (Single Command Session)

Dry-run (prints commands):

```bash
python3 scripts/run_vnext_session.py --profile fast
```

Execute full session:

```bash
python3 scripts/run_vnext_session.py --execute --profile full
```

This performs:

1. Ground-up dataset generation.
2. Optional HF mix dataset generation.
3. From-scratch SFT training on merged corpus.
4. Alignment pass on ground-up text corpus.
5. HF/Ollama packaging.
6. Inference regression suite.

## Acceptance Gates

1. Inference suite pass-rate: 100% for critical task set.
2. Standard HF load path works in external env:
   1. `AutoTokenizer.from_pretrained("Sharjeelbaig/neurocoder", trust_remote_code=True)`
   2. `AutoModelForCausalLM.from_pretrained("Sharjeelbaig/neurocoder", trust_remote_code=True)`
3. No `safetensors format` loading error in modern `transformers`.
4. Model card examples produce coherent output under recommended generation config.

## Publish Checklist

1. Package:

```bash
python3 scripts/package_release.py \
  --tokenizer artifacts/trained_vnext_full/tokenizer.json \
  --weights artifacts/trained_vnext_full/model.safetensors \
  --model-config artifacts/trained_vnext_full/model_config.json \
  --out artifacts/release_vnext_full \
  --model-name neurocoder
```

2. Upload:

```bash
hf upload Sharjeelbaig/neurocoder artifacts/release_vnext_full/hf --repo-type model
```

3. Verify:

```bash
hf download Sharjeelbaig/neurocoder --repo-type model --local-dir /tmp/neurocoder_verify
python3 scripts/test_inference_suite.py --model-dir /tmp/neurocoder_verify
```

## Notes on "Comparable to Large Models"

Narrow-task parity is realistic; broad parity with frontier general models is not realistic at this size. The practical path is:

1. Very strong task-specific data quality.
2. Strict output contracts and validators.
3. Fast decode path with caching.
4. Tool-compatible response patterns.

