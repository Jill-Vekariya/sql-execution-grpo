# Run Phases 0–3 with a ₹0 cash budget

This is the explicitly amended `student_free_nf4_fp16_v1` experiment. The same
pinned Qwen3-4B model, 12,908 training IDs, 2,304 validation IDs, prompts and SQL
verifier are retained. Both local base systems and SFT use NF4 weights with FP16
compute. SFT is therefore QLoRA. This is not the original, unrun BF16 reference.
Gemini 3.8 Flash is the named free hosted baseline; it is not claimed to be the
latest unrestricted flagship model. No final-test data or RL training is used.

## Account setup

Use a phone-verified Kaggle account with available free GPU quota. Keep the
notebook **private**, enable Internet, and choose free T4 GPU access. No paid
cloud resources are provisioned. Colab is a possible alternative when its
assigned GPU and session limits support the same locked environment.

For Gemini, create an API key in a Google AI Studio project displaying **Free
tier**, with billing disabled. Add it to Kaggle Secrets as `GEMINI_API_KEY` and
enable notebook access. The code cannot infer billing status from a key. Only
after checking that project yourself, set `BILLING_DISABLED_CONFIRMED=True` in
the API cell. Never put a key in a notebook cell, output, dataset or chat.
Student verification/phone verification and new credentials require the account
owner's participation. Nothing in this repository claims benefits have already
been activated.

## Notebook and data

Upload `notebooks/Free_Phase03_Kaggle.ipynb` to your private Kaggle Code area.
It embeds the necessary project source, so no private GitHub token is needed.
The locked CUDA 12.6 environment is separate from the original CUDA 13 lock.
The actual GPU must still pass the notebook's CUDA and architecture checks.
GPU integration and memory use remain unverified until run on assigned hardware.

The TRAIN-only sources occupy approximately 31 GB of databases plus 18 GB of
archives. They go under temporary scratch space outside `/kaggle/working`;
check free disk before downloading. The persistent output contains a source snapshot,
processed records, predictions, adapter and trainer checkpoints. Final/test
annotations and databases are never fetched. Original source, DB and manifest
hashes are checked before the frozen eligible file is restored.

Run the local stages first. Each request/finished stage is saved. SFT checkpoints
are saved every 10 optimizer steps in the free profile (100 remains the original
reference default). On interruption, save the notebook output
privately and attach it as input next session; set `RESUME_FROM` to that prior
`sqlrl_free_run` directory before unpacking. Use the same embedded source and run root.
The runner refuses changed source/config/data. An interrupted SFT run without
a complete trainer checkpoint must be preserved and restarted under a new root.
The session timer stops child processes before a typical notebook time limit;
it does not keep a notebook alive or bypass provider quotas.

## API quotas and results

Run the Gemini cell separately, initially with `MAX_REQUESTS=20`. Consult your
project's actual quota; no fixed daily allowance is promised. HTTP 429 or a
transport failure pauses the run without dropping completed questions. Resume
later on the **same project/account**, respecting its quota. A changed provider
modelVersion pauses the run so versions are not mixed. No paid fallback exists.

LOW thinking and the 512-total-output-token cap are logged. Reasoning can consume
that cap, so truncated/empty responses count as failures; this limitation must be
disclosed in the report. All 2,304 predictions are required before evaluation.
Partial API output is progress, not a final accuracy result.

After every system finishes, the report cell produces `comparison.json` and
`comparison.md`. Cash cost is zero only for the confirmed free allocation/project;
GPU time and provider token use are still recorded. GPU time lost to a hard
termination is not invented. A zero-cash experiment can still require several
sessions or weeks. Full completion within a particular quota is not guaranteed.

Primary references checked for this profile:

- https://www.kaggle.com/docs/notebooks
- https://www.kaggle.com/docs/efficient-gpu-usage
- https://ai.google.dev/gemini-api/docs/pricing
- https://ai.google.dev/gemini-api/docs/generate-content/text-generation
- https://ai.google.dev/gemini-api/docs/latest-model
- https://huggingface.co/docs/peft/developer_guides/quantization
- https://pytorch.org/get-started/previous-versions/
