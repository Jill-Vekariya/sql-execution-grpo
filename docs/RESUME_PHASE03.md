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
- Before the pause, neither Colab TRAIN preparation nor local experiments had run. After the user continued, a fresh free T4 runtime was connected, the 10-step checkpoint policy was applied, and all 91 tests passed in 3.31 seconds. TRAIN-only preparation was launched in a background process with prepare-data.log; its successful completion and all model experiments still require verification.
- API key/billing confirmation has not been supplied. Do not invent credentials, activate billing, or silently replace the named model.

## Resume safely

1. Open the saved Colab notebook and restore the same Google account through secure browser authentication if needed. Browser sign-in may expire between turns.
2. Check current free GPU availability and runtime contents. Runtime files and installed packages are temporary; re-run the source/setup cells if needed. The source and displayed GPU receipt remain saved in the Drive notebook.
3. Capture setup subprocess output in a log/print it when diagnosing errors. Hide the giant source code cell to avoid slow DOM inspection.
4. Download/prepare the exact TRAIN-only cohort, then run the local stages. Preserve complete checkpoints and predictions before the session expires. Use a session bound shorter than the account's current displayed runtime allowance.
5. The free SFT config now saves every 10 optimizer steps; the original reference default remains 100. Check this against measured T4 throughput before long SFT. The current saved Drive notebook has an explicit resume-policy cell that applies this exact source/config amendment before model runs. Newly generated notebooks embed it directly. Colab sessions are bounded to 10,800 seconds of local stages, with earlier setup time also counted against the provider's runtime lifetime. Do not change corpus IDs, training hyperparameters, model revision or evaluation cohort to make the run cheaper.
6. Investigate whether a documented free Colab hosted API can preserve the baseline's explicit output budget and response metadata. Otherwise the owner must supply the confirmed-free Gemini API credential securely. Do not use undocumented credential or browser internals.
7. Evaluate only complete 2,304-question prediction sets, then generate comparison.md/comparison.json. Keep final-test data sealed.

## Browser export caveat

A checkpoint zip was created in the first temporary Colab runtime, but files.download did not yield a confirmed browser download before the pause. Do not claim that zip was durably exported. No trained weights or model predictions were in it. Preserve the saved Drive notebook and local source instead, and use a verified export path on the next runtime.


## Owner approval recorded on 7 November 2026

The user explicitly approved the standard Colab Google Drive connection after being told that notebook code can access Drive files, and instructed completion of all Phase 0–3 work. Preserve that approval when resuming; do not ask for the same consent again. The zero cash budget remains binding. Drive OAuth completion and actual checkpoint persistence still need visible verification. Account authentication, creation of a new API credential, or a provider Terms acceptance may still require the owner under the relevant browser controls. This record is authorization history, not proof that the experiments are complete.


## Active Colab continuation checkpoint — 7 November 2026

- Latest source commit is 0f50ae73147db01ed452f920342121a717ac6509. GitHub CI 42356512415 passed. The earlier d5e7097 checkpoint also passed CI.
- In the fresh afternoon Colab session, the pinned GPU environment was restored. The source-policy cell again passed all 91 tests, in 2.99 seconds, and started TRAIN preparation (last observed PID 4538). No new model accuracy is claimed.
- A daemon model_controller was started in the notebook. Its verified initial controller_status.json says waiting_for_verified_train_data with cash budget zero. It waits for successful preparation, then launches scripts/run_free.py --stage local with at most 10,800 seconds and an additional four-hour bound measured from the session source-snapshot creation. It records status, logs and stage outcomes; it never marks Phase 3 complete while the hosted baseline is pending.
- The current saved Drive notebook contains the additional controller and storage cells. At this checkpoint, cells are: 1 original source; 2 failed Drive FUSE mount/helper; 3 pinned setup; 4 public Drive API authentication; 5 source policy/tests/TRAIN subprocess; 6 bounded model controller; 7 preparation status; 8 earlier unused gated mount; 9 original preparation; 10 original local-stage runner; 11 original Gemini; 12 report; 13 unconfirmed browser export; 14 GPU probe. Do not run all these cells blindly: they include retained diagnostic attempts and duplicate launch cells.
- The Drive FUSE route failed with credential propagation was unsuccessful after granting only the previously approved Drive file scope. The OAuth screen also offered unrelated Photos/contacts/message scopes, which were left unselected. No successful Drive mount or durable model backup has been verified.
- The supported public google.colab.auth.authenticate_user() route requests access to Google Drive AND Google Cloud data. Its first prompt was declined by the agent to keep preparation separate from this additional authorization. The owner must review the actual Google authorization prompt for that broader scope. The agent is preparing a manual handoff; existing approval for Drive files remains valid.
- After successful owner authorization, cell 4 obtains standard Application Default Credentials through public APIs (never print or copy credential values), builds Drive v3, and checks storageQuota. Then implement/verify private project-folder checkpoint upload using the supported Drive API before claiming persistence. Drive API uploads should stay within current free thresholds, respect storage quota, and never enable billing.
- Browser downloads through both the Python files.download route and the notebook File menu have not yielded a confirmed local file. Do not claim they succeeded. chrome://downloads was explicitly blocked by browser URL policy; do not attempt alternate surfaces or internal browser state to bypass that block.
- Google sign-in initially returned to a signed-out modal; one fresh target-tab check and secure sign-in restored a working session. Current active notebook tab is 3 in browser cdp/1. Runtime variables and tabs can expire between turns; inspect current state before restarting anything. Saved output cells with empty execution counts are historical output, not proof a new session ran them.
- Within the current kernel, do not re-run the original source unpacking after the SFT source amendment: its immutable snapshot guard will correctly reject the changed source. Reuse the live variables/controller. On a genuinely new runtime, restore the same source, then apply the identical tested amendment before resuming saved runs.


## Authorization and backup follow-up

- Re-running the public Drive API auth cell succeeded without another visible permission action. It returned a valid Drive storage quota through public Google APIs. No credential values were printed or copied. This supersedes the earlier pending-auth blocker; do not request that same Drive authorization again.
- A private_checkpoint_backup daemon is running in saved notebook cell 7. It waits for valid Drive API credentials, then archives PERSIST only (processed TRAIN cohorts, results, logs and source snapshot; excludes raw databases, model cache and credentials) every 300 seconds. It checks archive integrity, a 1 GiB transfer cap, existing free storage, and remote MD5/size. It retains only the newest complete SFT checkpoint in each backup. Verify backup_status.json before claiming a durable backup.
- Cells after 6 shifted by one: 7 backup worker; 8 concise live status; 9 old unused gated mount; 10 old prepare; 11 old local runner; 12 hosted optional key; 13 report; 14 unconfirmed export; 15 GPU probe. Keep the original launch cells untouched during the active run.
- AI Studio API Keys opened successfully under the same Google account and showed no imported API keys. An attempt to open Create API key was rejected by automatic approval review because it initiates sensitive credential creation without a specifically authorized project/scope. Do not bypass this rejection. The owner must create/review the credential and free project, and save it securely in Colab rather than chat. No API key was created and billing remains untouched.

- The first real Drive backup succeeded: 668,896 bytes, remote MD5 3712b6e79717301176d7cbf26b7d26b5 matched local, SHA256 8c8f3f6d6fe0ddb76c5cb162e4744bf56ca0a352a929cfa23559ed86d2d4a836. Exact observed receipt is saved in results/colab_drive_backup_probe.json. It contains no trained checkpoint yet. The worker updates the same private file; read its newest receipt before restoring.

- The AI Studio tab subsequently redirected to /app/onboarding. It requires the owner to acknowledge the displayed age/professional-use eligibility and accept Gemini API Additional Terms and Starter Tier Additional Terms via Continue. The agent has not checked or accepted these. Hand off this actual onboarding page for owner review, then owner creates the key in a Free tier project with billing disabled and saves GEMINI_API_KEY in Colab Secrets. No new credential or paid subscription is authorized by the agent.
- Commit 34376a137d79869b6f699e9fac30fdb6c3628c2c passed GitHub CI 42359274747. The following docs-only continuation does not alter running model source. The last observed extraction count was 204/209, with prep still running and both controller/backup threads alive; no model predictions had started at that observation.
