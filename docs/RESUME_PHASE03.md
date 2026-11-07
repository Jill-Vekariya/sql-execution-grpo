# Phases 0–3 resume checkpoint — 7 November 2026

Scope: Phases 0–3 only, using the amended student_free_nf4_fp16_v1 experiment and a zero cash budget. No RL or final-test work is authorized yet. The user paused the work, requested all state saved, then authorized continuing.

## Durable project and notebooks

- Private repository: https://github.com/Jill-Vekariya/sql-execution-grpo
- Saved Colab notebook: https://colab.research.google.com/drive/1cTMxhhGlbZT_1pF3T7n2YD1APCK5maNZ
- Private Kaggle notebook: https://www.kaggle.com/code/jillvekariya/free-phase03-kaggle
- Colab embedded source snapshot SHA256: bf30e56913aabd272ee8c8d759c038e62246e6a66105602e98645a73b099b48e
- Full original TRAIN data, audit evidence and frozen cohorts are retained on the user's computer. The portable notebook downloads TRAIN only and verifies the original hashes.

## Verified state

- 91 CPU/CI tests passed before the pause. Latest prior successful CI: commit 7f68f493f23a6c3d710eb9a62eeaea02ec404afe, run 42290225313.
- Colab free tier connected to a Tesla T4. Its locked environment and CUDA kernel probe succeeded. See results/colab_gpu_probe.json.
- Initial Colab setup completed, including the notebook's pytest invocation. Subprocess stdout was not captured in the output cell; the separate GPU probe printed the exact versions.
- Colab resource panel showed no subscription and zero compute units; a free GPU runtime was nevertheless allocated. The displayed session estimate was up to 4 hours 50 minutes, not a guaranteed allowance.
- Colab public ai.list_models() returned google/gemini-3.5-flash-lite and google/gemini-3.8-flash.
- The public ai.generate_text signature exposes prompt, model_name and stream only. This is NOT yet a validated replacement for the REST baseline's output cap, token receipts and modelVersion guard. No hosted model prediction was generated.
- Kaggle GPU and internet were blocked pending account-owner phone verification. Do not request that handoff repeatedly; Colab is the working free alternative.

## Pending experiments (do not mark Phase 3 complete)

- No Qwen zero-shot/few-shot predictions, QLoRA adapter, SFT validation predictions or Gemini benchmark predictions have been produced.
- No model accuracy or final comparison report exists. Gold SQL audit success is not model accuracy.
- Colab TRAIN download/preparation cell and local experiment cell had not been run before the pause.
- API key/billing confirmation has not been supplied. Do not invent credentials, activate billing, or silently replace the named model.

## Resume safely

1. Open the saved Colab notebook and restore the same Google account through secure browser authentication if needed. Browser sign-in may expire between turns.
2. Check current free GPU availability and runtime contents. Runtime files and installed packages are temporary; re-run the source/setup cells if needed. The source and displayed GPU receipt remain saved in the Drive notebook.
3. Capture setup subprocess output in a log/print it when diagnosing errors. Hide the giant source code cell to avoid slow DOM inspection.
4. Download/prepare the exact TRAIN-only cohort, then run the local stages. Preserve complete checkpoints and predictions before the session expires. Use a session bound shorter than the account's current displayed runtime allowance.
5. Check checkpoint frequency against measured T4 throughput before long SFT. The current implementation saves every 100 optimizer steps; more frequent operational checkpoints may be needed for short free sessions. Do not change corpus IDs, training hyperparameters, model revision or evaluation cohort to make the run cheaper.
6. Investigate whether a documented free Colab hosted API can preserve the baseline's explicit output budget and response metadata. Otherwise the owner must supply the confirmed-free Gemini API credential securely. Do not use undocumented credential or browser internals.
7. Evaluate only complete 2,304-question prediction sets, then generate comparison.md/comparison.json. Keep final-test data sealed.

## Browser export caveat

A checkpoint zip was created in the first temporary Colab runtime, but files.download did not yield a confirmed browser download before the pause. Do not claim that zip was durably exported. No trained weights or model predictions were in it. Preserve the saved Drive notebook and local source instead, and use a verified export path on the next runtime.
