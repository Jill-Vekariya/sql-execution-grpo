# Decisions

## D001 — Scope and authority
The user authorized uninterrupted work through Phases 0, 1, 2 and 3 and a GitHub
push. Phases 4–10 remain future work. Use a new repository so existing projects
are unaffected. Record blocked work rather than asserting runs happened.

## D002 — Model and compute
Choose Qwen/Qwen3-4B-Instruct-2507, with its Hub commit pinned in configs.
Qwen3.5-4B is the newer alternative considered. This is a compatibility and
experimental-simplicity choice, not measured evidence of superior SQL accuracy.
The detected laptop has Intel integrated graphics and 16 GB RAM, no CUDA GPU.
Use WSL Linux for enforced process limits. No paid cloud GPU or API allowance
was supplied; paid jobs are not provisioned. The frontier model remains unset.

## D003 — Data version and splits
Use Spider **1.0**, not Spider 2.0, and original BIRD-SQL 2023 TRAIN annotations.
Do not silently replace BIRD training with its later filtered 6,601-example
subset. Select 15% of each training corpus's database IDs for project validation
with seed 42, then remove normalized duplicate questions, preserving validation.
The normalizer uses Unicode NFKC, casefold and collapsed whitespace. It does
not claim to identify paraphrases. Train/val database identifiers are disjoint;
available file hashes also must be disjoint across those splits.

## D004 — Final-set access and the audit conflict
Reserve both official dev sets for the single Phase 9 final evaluation. Do not
open their questions, gold SQL or database contents in Phases 0–3. The ZIP central
directory may be read to locate TRAIN members, but no dev/test member is opened.
Spider's shared tables.json contains schema metadata; only TRAIN database
metadata is retained. Final split checksums and cross-final question deduplication
cannot honestly be completed without accessing the sealed data. Their manifest
fields therefore remain explicitly pending. This resolves the conflicting
requests in favor of keeping the final evaluation unseen. The Phase 9 protocol
must disclose any discovered overlap without retuning the frozen model.

## D005 — Download budget versus research corpus
Training metadata is fully audited. Initially cap selective database downloads
for practical local verification; sources logs identify every omitted DB.
Missing DBs do **not** silently become a smaller research corpus: benchmark
generation and SFT fail preflight until the declared corpus is available.
Gold-audit results report both the available-data and entire-corpus denominator.

## D006 — SQLite protection
Each query runs in a fresh Python interpreter with a read-only immutable URI,
SQLite authorizer, disabled extensions, query-only mode, address-space/CPU/file
limits, progress interrupt, parent wall timeout and bounded rows/result bytes.
No partial result is scored as correct. WAL snapshots are rejected. SQL parsing
is also done inside the child, so parser CPU/memory abuse cannot stall the
training process before SQLite runs. Linux/WSL is required; unsupported platforms
fail closed. These protections target SQL on
trusted benchmark DBs, not OS-level isolation against malicious database files
or SQLite engine exploits. Hash static database files before use.

A measured audit with SQLGlot imported separately in every child increased
startup overhead and timeout counts. The final executor therefore uses a small
SELECT/WITH prefix check and SQLite's own parser inside the bounded child;
SQLite's authorizer and read-only URI remain the enforcement boundaries. Full
SQLGlot AST parsing is used for result-order/reward analysis, not for starting
each executor process. Both earlier audit versions are retained.

## D007 — Comparator contract
Columns retain their positional meaning; aliases are irrelevant. Column count
is checked even for empty results. Preserve duplicate rows (bag semantics).
Enforce row order only for the outer gold ORDER BY. Nested ORDER BY and window
ORDER BY do not define outer result order. Gold ordering ties remain a limitation:
this contract can reject a different valid ordering within tied keys. NULL equals
NULL, not zero/empty string. SQLite numeric values use rtol=1e-6, atol=1e-8;
text is not coerced to a number. Integer/integer and mixed integer/float answers
must be exactly equal; only float/float comparisons receive tolerance, so a
nearby large integer cannot exploit relative tolerance. Nonfinite floats are
rejected. Unordered
approximate matching uses one-to-one bipartite matching, capped at 500 rows when
exact bags differ; exceeding that cap is a recorded comparison failure.
This stricter metric differs from official benchmark evaluators and is named
explicitly; no leaderboard-comparable claim is made.

## D008 — Prompts and baselines
Use schema DDL, question, BIRD evidence when supplied, and exactly one SQL fence.
Do not include sample values: this reduces retrieval variability and avoids
query-answer leakage through sampling. BIRD may suffer without value grounding;
the same policy applies to every system. Few-shot uses three deterministic
examples from distinct TRAIN databases; their IDs are saved. The same validation
questions, schema and evidence are used across systems. Greedy local generation,
512 completion tokens and a 4096-token prompt limit are fixed in configs;
overlength prompts are failures, not silently truncated or omitted.

## D009 — Uncertainty and cost
Use 2,000 percentile bootstrap replicates resampling databases, seed 42.
Report query-weighted accuracy; resampling databases respects cross-question
dependence. One/small numbers of databases yield weak intervals and must be
disclosed. Paired differences use the same resampled database groups.
Unknown token counts/costs are null, never zero. Local inference cost requires
hardware rate and time; the script does not invent a dollar rate.

## D010 — SFT
One CUDA device, one preregistered epoch, LoRA rank 16/alpha 32/dropout 0.05 on Qwen attention and
MLP projections; completion-only loss, no packing, BF16, batch 1, accumulation 16,
learning rate 2e-4. All values are in configs. Overlength training examples are
excluded with IDs/counts saved rather than truncating the gold answer. The SFT
reference uses the final epoch; intermediate validation loss is diagnostic only.
Execution evaluation follows using the common generator/verifier. Model
training, successful adapter saving and validation EX are distinct milestones.

## D011 — Provisional reward and cheating
Reward code exists and is tested before any training. Phase 4 has not finalized
the RL reward. Execution is 1; valid bonus 0.02; format +0.01/-0.01; timeout/huge
output -0.05. Robust mode withholds execution credit for unverified empty results
and obvious table-free/literal-only answer shortcuts. Raw single-snapshot EX is
logged separately. These heuristics can reject legitimate queries and do not
prove semantic equivalence or eliminate hardcoding. Counterfactual/test-suite
databases are the planned stronger check. No RL reward-improvement claim exists.

## D012 — Dependencies
Check official model cards, TRL SFT/GRPO documentation, PEFT LoRA documentation,
SQLGlot docs and SQLite authorizer docs before implementing. Pin exact direct
versions and a transitive uv.lock. Newest independent package versions are not
assumed compatible: TRL 1.17.1's vLLM extra requires vLLM<0.33; vLLM 0.32 requires
Transformers<5.20 and Torch 2.13.0. Use the resolved matrix and save resolver logs.
GPU execution remains unverified until a real GPU smoke test succeeds.

## D013 — Shared training eligibility
Keep raw train/val manifests intact. Before benchmark training, freeze an
additional train_eligible manifest using only training gold execution status
and the pinned tokenizer's declared SFT context budget. The same eligible IDs
are the SFT, future RL and few-shot training pool. Exclusions are saved with
their reasons and checksums; original labels are never rewritten. This favors
examples supported by the executor/resource budget and can change the query-type
distribution. Report that limitation. Validation is not filtered: nonexecuting
gold remains visible in the evaluation denominator and gold-error count.

## D014 - Resume verification, 18 November 2026

Recheck the existing checkout rather than recreate the project. The fresh audit
verifies saved train/val/eligible manifest hashes, database/question disjointness,
and 111 passing CPU checks without touching held-out content. Existing locks,
model revision, prompts, training parameters, and partial run artifacts remain
unchanged. See results/resume_audit_20261118/summary.json.

The latest saved GPU progress is historical. Verify the newest private backup
and a current free runtime before resuming a prefix or claiming current progress.
A passing verifier suite is not completed Phase 2 inference or Phase 3 SFT.

The current request explicitly requires a model choice before GPU work.
Recommend retaining Qwen3-4B-Instruct-2507 for continuity; offer Qwen3.5-4B
as a newer, separately configured experiment. No SQL best-model claim is made.
The existing zero cash budget remains the ceiling pending any user amendment.

Spider 1.0's official test is public according to the maintainer page.
Keep official dev as the existing final cohort; correct the earlier hidden-test
rationale rather than change the experiment after partial model generation.
Final checksum/dedup work remains pending under D004.

## D015 - Delegated model choice, 18 November 2026

The user delegated the choice ("best for me i trust you"). Retain the pinned
Qwen3-4B-Instruct-2507 and student_free_nf4_fp16_batch8_v2 configuration,
seed 42, unchanged cohorts, and zero cash budget. This chooses continuity
and verified integration, not a claim of best measured SQL capability.

The saved Colab notebook has later historical progress than the repository
handoff: 1,448/2,304 zero-shot predictions at 2026-11-07T10:46:40 UTC.
The newest actual backup remains unverified. A fresh free T4 connected and
ran the original source cell with the expected source hash; no new benchmark
generation or SFT restarted. Details: results/continuation_choice_20261118.json.

Automatic approval review rejected the clean-notebook upload to Google Colab
and the notebook's Allow prompt for Google Drive AND Google Cloud data.
These actions remain blocked; do not route them through another mechanism.
Explicit owner approval for that payload/destination and exact access scope
is needed before resuming backup restoration. Earlier approval records are
retained as history, not treated as a way to bypass these current rejections.

## D016 - Imported baseline evidence and resumed work, 18 November 2026

The user's subsequent "done" resolved the precise clean-notebook upload and
Drive/Google Cloud authorization handoffs. The clean notebook was uploaded and
the existing standard auth completed. The archive was restored and verified
before restarting the private backup worker. No new access scope or paid
resource was added beyond the disclosed approval.

The restored full zero-shot run is reportable: all 2,304 predictions, config,
environment, execution scores and summary were exported and hashed. A separate
CPU replay reproduces the entire summary, including the seeded database-cluster
bootstrap interval. Imported files remain immutable; later live progress is
recorded separately. No incomplete-prefix accuracy is reported.

The laptop's stale BIRD metadata fallback schemas were replaced only after the
saved prompt schemas reproduced the full validation file SHA256 exactly.
Question/evidence/gold/ID fields were unchanged. The stale local file is kept,
and results/validation_schema_investigation_20261118.json documents the proof.
Do not waive data pins just because a file has the right IDs or row count.

Keep the same frozen model, quantization, source, prompts, seed and full cohorts.
TRAIN reconstruction passed, and the bounded controller resumed few-shot
generation rather than regenerating the completed zero-shot run. No checkpoint
selection, training hyperparameter or reward was changed after seeing VAL.
Costs remain null where unmeasured. A free-GPU price setting alone does not
justify inventing measured inference cost. Shared batch latency must be labeled.

## D017 — Resume evidence and gold failure accounting, 19 November 2026

Keep the delegated model/revision, NF4/FP16 profile, seed, cohorts and zero cash
budget fixed. Saved outputs are replayed separately from new benchmark execution.
The fresh baseline replay also checks the processed VAL bytes; it neither
executes benchmark SQL nor calls a model. Final-test content stays sealed.

Classify every saved gold failure exactly once before discussing the missed
near-100-percent execution target. The new diagnostic report preserves every
case ID and source hash, separates resource limits from SQLite errors, and
does not infer annotation noise from an error alone. Regression checks reject
dropped or relabelled failures. No cohort or metric changes follow from it.

The current notebook restore reached the same Drive/Google Cloud prompt,
but automatic approval review rejected Allow for missing action-time confirmation.
Preserve that blocker and the exact scope; do not bypass it through shell,
Drive API credential extraction or a different upload mechanism. Neither
historical notebook output nor the user-reported 872-answer count establishes
the newest backup's current progress. See results/resume_20261119.json.
