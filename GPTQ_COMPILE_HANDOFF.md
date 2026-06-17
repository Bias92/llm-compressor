# GPTQ torch.compile — handoff / status

Self-contained summary so any agent (Codex, a fresh session, a human) can pick this up cold.
Scratch doc — not part of the eventual PR. Claims are scoped to what was actually observed; see the
"NOT yet verified" section before treating anything as a result.

## Goal
Speed up GPTQ weight quantization with `torch.compile` — llm-compressor issue
https://github.com/vllm-project/llm-compressor/issues/1496 . Applies the torch.compile approach from
PR https://github.com/vllm-project/llm-compressor/pull/2384 (MSE-observer compile — **currently OPEN /
unmerged**) to GPTQ. "Pattern reference", not a merged dependency.

## Branch / commits
- Branch `claude/gptq-compile-modern`, based on `torch-compile-observers` (the dev line where gptq lives
  at `src/llmcompressor/modifiers/gptq/` and MSE compile lives in `src/llmcompressor/observers/mse_quant.py`).
  On fork `Bias92/llm-compressor`.
- **Pushed (remote + pod clone head = `90396f02`):**
  - `ee1d283d` perf: make GPTQ compatible with torch.compile  ← the actual change
  - `90396f02` chore(scratch): GPTQ compile benchmark (`bench_gptq_compile.py`, NOT for PR)
- **LOCAL ONLY, NOT pushed:** `47b60853` fix(scratch): recipe YAML indentation. The pod clone is at
  `90396f02` + a manual heredoc fix applied in-place for the same YAML issue.
- Not opened as a PR.

## What was implemented (only `modifiers/gptq/gptq_quantize.py`, ~3 edits)
The GPTQ algorithm is untouched. The per-column fake-quant is extracted into a module-level function
with a compiled variant, gated by a flag. The eager outer loop (Cholesky + sequential error-feedback
recurrence) is unchanged.

1. Module-level: `_enable_gptq_compile` + `set_gptq_compile`/`get_gptq_compile`;
   `torch._dynamo.config.capture_scalar_outputs = True`;
   `_quantize_column(...)` that just calls `fake_quantize(...)`; `_quantize_column_compiled =
   torch.compile(_quantize_column, dynamic=True)` created once at import.
2. In `quantize_weight` before the loop: `quantize_column = _quantize_column_compiled if
   get_gptq_compile() else _quantize_column`; pre-build the CHANNEL-strategy `column_quant_args` once
   (was `copy(quant_args)` per column).
3. In the per-column loop: each strategy (TENSOR/CHANNEL/GROUP/TENSOR_GROUP/BLOCK) selects the scale/zp
   slice eagerly and calls `quantize_column(...)`; error-propagation matmuls stay eager.

**Gating caveat (reviewer will ask):** the global `set_gptq_compile` matches the `set_torch_compile`
global that exists in THIS branch's `observers/mse_quant.py`. BUT #2384's gating API evolved during
review and the live PR may use a different mechanism (e.g. session state / `set_observer_compile`).
Verify against current #2384 and align the gating before opening a PR. Flag is set directly by the
bench; not wired into oneshot/dataset_args yet.

## Environment that runs (RunPod A100 SXM 80GB, "RunPod PyTorch 2.4.0" template)
```bash
cd /workspace
pip install -U compressed-tensors transformers datasets accelerate   # also upgrades torch -> 2.12.0+cu130
pip uninstall -y torchvision torchaudio                               # REQUIRED: ABI-broken vs torch 2.12
git clone https://github.com/Bias92/llm-compressor.git llmc-modern
cd llmc-modern && git checkout claude/gptq-compile-modern
pip install -e . --no-deps
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.12.0+cu130 True
python -c "import llmcompressor; from llmcompressor.modifiers.gptq.gptq_quantize import get_gptq_compile; print('OK', get_gptq_compile())"
```
torch note: `torch==2.10` is NOT on the cu124 wheel index (`download.pytorch.org/whl/cu124` maxes at
2.6.0; cu121 maxes 2.5.1) — don't pin it there. Default PyPI ships the 2.1x line as CUDA-13 builds;
`pip install -U` lands `torch 2.12.0+cu130`, which runs on this pod's A100 driver. The Feb-based branch
`claude/xenodochial-...` is UNRUNNABLE against any real compressed-tensors (mixed-era CT API).

## Benchmark observations (Qwen3-8B, W4A16 group, A100 80GB) — one-process bench (see caveat)
| config | speedup (e2e) | eager peak | compiled peak | notes |
|---|---|---|---|---|
| weight-only, 64 smp / 4096 | **~1.10x** (986→893s) | 35 GB | 50–66 GB | --runs 3 medians |
| +activation, 64 / 4096 | — | 66 GB | **OOM** | compiled exceeded 80 GB in this process |
| +activation, 32 / 2048 (lighter) | **1.14x** (854→746s) | 35 GB | 50–66 GB | --runs 1; OOM avoided by lighter cfg + `expandable_segments:True`. cold 50GB / warm 66GB (warm>cold = one-process accumulation) |

## Conclusion (calibrated)
In the tested configs, **GPTQ per-column compile is ~1.1x (weight-only 1.10x, +activation 1.14x),
with a ~1.5x memory overhead and an OOM on the full +activation config (64/4096 on 80GB).** Both are
below the 1.3x "worth it" bar. Likely reason (Amdahl): the compiled `fake_quantize` is a small
slice of GPTQ's time; the hot path (Hessian Cholesky inverse + per-column error-propagation matmuls +
the sequential recurrence) stays eager. +activation did NOT produce a bigger win (unlike MSE #2384,
where the compiled thing IS the observer and +act calls it heavily) — for GPTQ, +act adds work to the
eager forward/observer path, not to the compiled per-column kernel.

## NOT yet verified (do these before any PR claim)
1. **Correctness** — calling `fake_quantize` makes it LIKELY identical to eager, but UNPROVEN. Add a
   unit test asserting compiled == eager across all 5 strategies (TENSOR/CHANNEL/GROUP/TENSOR_GROUP/BLOCK)
   and with `global_scale` (fp4 path). (A test for this exists on the old Feb branch; port + adapt it.)
2. **Clean memory** — the bench runs warmup+eager+cold+warm in ONE process, so peak memory accumulates
   (warm peak > cold peak) and that contributed to the +act OOM. The "compile adds ~1.5x peak" figure is
   NOT confirmed; re-measure one-config-per-FRESH-process recording both `max_memory_allocated()` and
   `max_memory_reserved()` + the OOM site.
3. **Recompiles** — observed 0 in the W4A16-group config tested; other shapes/strategies/`global_scale`
   presence may produce additional graphs. State as "0 recompiles in tested config", not a blanket claim.
4. **+activation e2e number** — only per-module times observed so far; the e2e table for the lighter
   +act run was still pending when this was written.

## Must-fix before any PR (from code review)
- **Gating is wrong for a PR.** The process-global `_enable_gptq_compile` / `set_gptq_compile` has no
  session isolation and isn't serializable via recipe/oneshot — if the bench sets it True and doesn't
  reset, every later GPTQ run takes the compiled path. Replace with a `GPTQModifier.enable_torch_compile`
  attribute (or thread `quantize_weight(..., enable_torch_compile=...)` explicitly), aligned with #2384's
  final API. The bench currently relies on the global; that's fine for benching, not for the PR.
- **`torch._dynamo.config.capture_scalar_outputs = True` is probably unnecessary here and too broad.**
  It's a global Dynamo config flip at import, and its comment cites `calculate_qparams` — but this
  helper only calls `fake_quantize` (scale/zp are precomputed eagerly and passed in; `calculate_qparams`
  is NOT on this path). Test compiling `_quantize_column` WITHOUT it; if it works, remove it (or move it
  into a scoped compile-config). The comment is also inaccurate (`calculate_qparams` vs the `calculate_range`
  that fake_quantize actually hits).
- (Already in "NOT yet verified": correctness unit test across 5 strategies + global_scale; recompile
  claim scoped to tested config; fresh-process memory re-measure.)

## Next options
1. **Block kernel** — compile more of the loop (vectorize per-column scale lookup into scale_map/zero_map;
   the `8fb026ee` / aladerran direction from PR #2320). Bigger compilable fraction → maybe bigger speedup,
   but harder (sequential recurrence) and likely worse memory.
2. **Accept ~1.1x** weight-only + memory cost = weak PR; reviewer may pass.
3. Re-measure cleanly (fresh-process + correctness test), take the honest numbers to the maintainer, decide.

Decision bar: e2e **≥1.3x → worth pursuing**; ~1.1x → weak; persistent OOM → shelve; fast-but-memory-heavy
→ default-off + docs + memory mitigation.
