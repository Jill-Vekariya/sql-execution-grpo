# Reproduce Phases 0–3

Run under Linux with Python 3.12 and uv 0.13.7. WSL2 works for CPU verifier tests.
The core/test/data extras do not install the large training/generation stack.

```bash
uv sync --frozen --extra test --extra data
uv run --no-sync pytest --junitxml=results/tests-reproduction.xml
uv run --no-sync python scripts/fetch_train.py --dataset spider --database-budget-mb 512
uv run --no-sync python scripts/fetch_train.py --dataset bird --database-budget-mb 12000
uv run --no-sync python -m sqlrl.data.prepare
uv run --no-sync python -m sqlrl.data.audit_gold --output results/gold-audit-reproduction
```

The BIRD archive is several GB; downloading all its TRAIN databases is deliberate.
Source reports list omitted databases when a lower download cap is chosen.
The fetcher never opens an official dev/test member. Train annotation members
and database hashes are saved; compare checksums against the committed manifest
before reproducing a research result. Upstream datasets can change.

On a CUDA machine, use the exact pinned training environment:

```bash
uv sync --frozen --extra train --extra data
uv run --no-sync python scripts/freeze_training_corpus.py
uv run --no-sync python -m sqlrl.eval.generate --config configs/base_zero.json
uv run --no-sync python -m sqlrl.eval.run --predictions results/runs/base_zero_seed42/predictions.jsonl --output results/evaluations/base_zero_seed42
uv run --no-sync python -m sqlrl.eval.generate --config configs/base_few.json
uv run --no-sync python -m sqlrl.train_sft.run --config configs/sft.json
uv run --no-sync python -m sqlrl.eval.generate --config configs/sft_eval.json
uv run --no-sync python -m sqlrl.eval.run --predictions results/runs/sft_eval_seed42/predictions.jsonl --output results/evaluations/sft_seed42
```

Before research reproduction, use the model revision already committed rather
than rerunning pin_model.py against a moving main branch. The pinning command
is for setting up a newly declared experiment. Adapt run output directories to
new unique names; existing evidence directories are never overwritten.

For frontier inference, set the chosen model, endpoint, verified per-token prices
and a positive budget in configs/frontier.json and export SQLRL_API_KEY locally.
Never commit a key. The same prompt is used. The generic HTTP backend uses the
chat-completions wire format; a selected provider/model may need a verified
adapter for its token-limit parameter or reasoning controls. It has not been
validated against a paid frontier model. No API request is made with the default
unset model and zero budget.

vLLM 0.32.0 is locked as an optional generation extra. A separately hosted vLLM
endpoint can be used with the HTTP generator after explicitly configuring it.
Colocated vLLM/GRPO GPU operation is deferred to Phase 5 and is not claimed tested.

The independent CPU API integration uses a random tiny Qwen3 fixture, two LoRA
optimizer steps and adapter saving. It produces no benchmark accuracy:

```bash
uv venv /tmp/sqlrl-cpu-smoke
uv pip sync --python /tmp/sqlrl-cpu-smoke/bin/python requirements-cpu-smoke.lock --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
/tmp/sqlrl-cpu-smoke/bin/python scripts/smoke_sft_cpu.py
```

The CPU-specific Torch build has its own hash-pinned lockfile; it does not replace
the GPU experiment environment. Existing smoke-result directories are preserved.

Gold-audit errors are not silently corrected. Preserve original labels and
record proposed corrections separately. No final-set command exists in this
phase's code. Phase 9 requires a frozen checkpoint, immutable manifest, tag and
a dedicated single-use evaluation implementation.
