# Execute the remaining Phase 2–3 work

The code checkpoint is published. Research completion still requires actual
base, SFT and frontier runs; no synthetic fixture can substitute for them.

Use an already connected Linux CUDA machine with native BF16 support. Exact
VRAM requirements have not been measured on a GPU. Install the committed
Python 3.12/uv environment with `uv sync --frozen --extra train --extra data`.
Use the TRAIN-only download/preparation commands in [REPRODUCE.md](REPRODUCE.md).
Then restore the published eligible file using the committed IDs and hashes:

```bash
uv run --no-sync python -m sqlrl.data.restore_eligible
```

For this existing experiment, do not rerun `freeze_training_corpus.py`: a new
gold audit on different hardware can produce different timeout exclusions.
The restore command verifies the complete prepared training hash, manifest hash
and resulting eligible-file hash; it preserves an existing file if it differs.
The runner verifies the committed eligible training pool and validation IDs,
original annotation hashes/labels, and database bytes. It never opens official
dev/final data or chooses a new training subset.

Store provider settings in a private local JSON file outside this repository:

```json
{
  "model": "REPLACE_WITH_VERSIONED_PROVIDER_MODEL",
  "endpoint": "https://REPLACE_WITH_PROVIDER/v1/chat/completions",
  "api_key_env": "SQLRL_API_KEY",
  "budget_usd": 0,
  "input_usd_per_million": 0,
  "output_usd_per_million": 0
}
```

Replace the model, endpoint, prices and spending cap with verified values. Set
the credential in the execution environment; never put it into a JSON file or
chat. The provider must support the existing chat-completions adapter, including
its `max_tokens` argument, and return usage. Provider compatibility must be
checked before a research run; a network failure can still incur charges, so
the generator never automatically retries an API request.

Inspect the complete plan without starting any model or API work:

```bash
uv run --no-sync python -m sqlrl.complete_phase03 \
  --output results/my_new_readiness_check
```

On the configured GPU, supply its actual hourly rate and a new output path:

```bash
uv run --no-sync python -m sqlrl.complete_phase03 \
  --output results/phase03_measured_attempt01 \
  --gpu-usd-per-hour ACTUAL_RATE \
  --frontier-config /private/path/frontier.json \
  --execute
```

Use a rate of zero only for compute known to be free. The script does not rent
or shut down hardware, authorize a subscription, or enforce a cloud billing
limit. An allocated rented machine continues to incur provider charges outside
the script; manage that allocation separately. Existing output directories are
rejected. A failed attempt is preserved and a new attempt starts from scratch.

The command runs zero-shot and few-shot generation/evaluation, LoRA SFT, SFT
generation/evaluation, the frontier baseline, paired database-cluster bootstrap
comparisons, and a measured cost report. It writes `comparison.json` and
`comparison.md` only after all stages succeed. GPU costs measure process wall
time, including model loading, at the supplied rate; downloads/setup and idle
time outside the measured stages are excluded. Provider API usage supplies
frontier costs. The SFT cost-per-accuracy-point value is undefined for a zero
or negative gain. No RL training or final-test evaluation is included.

Free Colab GPU availability and GPU type are not guaranteed. Check the actual
assigned hardware rather than changing this frozen BF16 configuration silently.
Sources: [Colab FAQ](https://research.google.com/colaboratory/faq.html),
[PyTorch BF16 check](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html).
