# Recorded batch-eight free profile

The optional student_free_nf4_fp16_batch8_v2 run changes inference batching only.
It preserves the pinned Qwen revision, NF4/FP16 weights and compute, all 12,908
eligible training IDs and 2,304 validation IDs, seed, prompts, three-shot
selection, context/output limits, SQL verifier and training hyperparameters.
Official final-test data remains sealed. This is a separately recorded
operational amendment to the serial student_free_nf4_fp16_v1 profile.

## Measured diagnostic

Eight fixed first validation questions were tested for token identity and exact
answer equality against saved serial outputs; gold answers were not used to
select a batch size. Batch four took 10.025393 seconds; batch eight took
6.903841 seconds, compared with 28.520275 seconds for the same serial questions.
All eight answers matched in both tests. Batch eight used a peak 3,384,800,768
bytes of allocated GPU memory in its own process. The earlier serial worker
remained allocated but was paused during measurement, then resumed in finally.
The source and receipt identify this as a diagnostic, not an accuracy result.
A first tokenization preflight failed before generation and its log was kept.

The observed batch-eight speedup was 4.131 times on these eight questions.
This is not a guarantee for the entire corpus or longer few-shot prompts.

## Evidence and resumption

Use a new root results/free_runs_batch8_v2; do not combine serial and batched
predictions. Preserve the original serial run, source and configuration.
The source under src is unchanged. Batched generation additionally records
the SHA256 of scripts/run_batched_generation.py in each immutable run config,
so changing that script prevents resuming a different implementation.
Predictions are written atomically after each completed batch and must match
the exact frozen validation prefix on resume. A CUDA error stops the batch
and preserves that prefix. It does not fill the rest with manufactured errors.

Each prediction's latency_s is its shared batch wall time. The separately
named amortized_batch_seconds field measures throughput. Do not compare these
latencies as if the batched run served isolated serial requests. Input-token
counts exclude padding; output-token counts include the first EOS and exclude
subsequent padding. The original 4,096 input and 512 output limits are retained.

The training command remains sqlrl.train_sft.run with microbatch one,
accumulation 16, one epoch, learning rate 0.0002, maximum length 4,608 and
the same LoRA settings. Ten-step checkpoints remain enabled. A fresh SFT stage
is deferred if fewer than 30 minutes remain in the bounded session, preserving
completed baselines and avoiding an attempt with no complete checkpoint.

Run scripts/run_free_batched.py --stage local with a session bound within the
provider's actual free allowance. Reuse the same source, config, root and full
trainer checkpoint when continuing. The hosted Gemini baseline still uses the
documented REST profile, free project and owner-managed credential; no paid
fallback or undocumented credential route is introduced.
