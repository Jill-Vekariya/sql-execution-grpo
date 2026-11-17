# Failures and unresolved work

Append concrete observations; do not close an item without saved evidence.

| ID | Observation | Evidence / disposition |
|---|---|---|
| F001 | No CUDA GPU on the connected laptop. | results/environment.json; full Qwen inference/SFT blocked. |
| F002 | Docker CLI installed, Linux engine not running. | WSL used; no Docker execution claimed. |
| F003 | No frontier API model, positive budget, or SQLRL_API_KEY configured. | configs/frontier.json and run status; frontier benchmark blocked. |
| F004 | First environment bootstrap attempted logging before the empty results directory had been transferred. | Created the directory and reran bootstrap; this did not involve data/training. |
| F005 | Final checksums and cross-final duplicate audit pending. | Final data remains sealed; see D004. |
| F006 | Early local metadata requests returned buffered output late. | They eventually succeeded; this was not evidence of network denial. |
| F007 | First executor suite: 56 passed, one failed. A too-low RLIMIT_AS did not invalidate already-mapped interpreter memory. | results/tests-initial.xml; fixed by checking current virtual address space before applying the cap. Rerun evidence is retained separately. |
| F008 | BIRD's official training databases are inside a compressed nested ZIP of about 9 GB; selective outer-ZIP reads cannot extract individual DBs. | fetcher explicitly records omitted DBs at the initial 512 MiB download cap. A full-download path is implemented for a larger explicit cap. |
| F009 | An initial PowerShell-to-WSL shell loop expanded a config variable incorrectly, producing a missing configs/.json CLI error before generation. | Replaced the shell loop with scripts/run_preflights.py using argument arrays. No API or model call occurred. |
| F010 | Initial CPU smoke dependency sync omitted the CPU wheel index used during resolution. | Initial install stderr preserved; rerun uses the PyTorch CPU index and the same hashed lockfile. |
| F011 | Desktop Commander stopped responding during the full BIRD download; a write returned "Timeout - no response from device" and a terminal call timed out. | Work continued in the local Linux workspace. The device reconnected after the user requested continuation; the tested source and reports were then synchronized for publishing. |
| F012 | A local audit was started while BIRD extraction was still incomplete. | That report is retained separately under gold_audit_partial_extraction; the primary audit was rerun only after all 209 database snapshots were present. |
| F013 | The tokenizer client could not initialize under the workspace's SOCKS proxy because its optional socksio dependency was absent. | The initial traceback is preserved. socksio is explicitly pinned for training and the CPU environment. |
| F014 | The initial context audit incorrectly counted the two fields of the current tokenizer's BatchEncoding instead of input_ids. | Detected from an impossible 2-token distribution before benchmark training. The incorrect eligibility report is retained; the corrected count has a regression test and eligibility is regenerated. |
| F015 | The first GitHub CI run passed 76 tests and failed a test that assumed 4,000 redundant parentheses must overflow SQLite's parser. The runner instead returned the correct value promptly. | CI run 42278188866 preserves the failure. The test now verifies bounded execution and checks the exact value if the SQLite build accepts the query; executor protections are unchanged. Follow-up CI evidence is recorded in publication_status.json. |

Gold query errors, missing databases and resource-bound failures are recorded
individually in results/gold_audit/queries.jsonl. Their existence is not evidence
that a label is incorrect; inspect the error and original annotation first.

## Resume observations, 18 November 2026

- F016: The saved Spider literature audit incorrectly said official test labels
  were hidden. The maintainer page states the test was released in 2024.
  Corrected the rationale with prior text retained in
  results/literature_audit.json. Existing official-dev final cohort unchanged;
  no held-out content opened.
- F017: Saved Colab progress and old GPU receipts do not confirm a current
  runtime or the newest private backup. Current restoration/GPU status is
  unverified, not a demonstrated runtime failure. See
  results/resume_audit_20261118/summary.json. No inference/SFT completion claimed.
- F018: Current web access could not open the vLLM GRPO documentation URL.
  Recorded in results/resume_audit_20261118/upstream_review.json; no training
  API implementation or dependency change was made on that basis.

- F019 (18 November 2026): Automatic approval review rejected uploading the
  clean private-project resume notebook to Google Colab because the specific
  payload/destination lacked explicit end-user authorization. No upload occurred.
- F020 (18 November 2026): Refreshing the existing notebook's Drive connection
  reached an Allow dialog for Google Drive and Google Cloud data. Automatic
  approval review rejected clicking Allow without explicit authorization for
  that exact scope. Restoration and GPU work remain paused. Both rejections
  and the actual saved progress are in results/continuation_choice_20261118.json.


## Resolved handoffs and evidence import, 18 November 2026

- F019/F020 follow-up: the user explicitly completed/approved the disclosed
  notebook upload and Drive/Google Cloud access. Clean upload, standard auth
  and archive restoration then succeeded. The earlier rejections remain history;
  they are no longer the current restoration blocker. Evidence:
  results/resume_verified_20261118.json.
- F021: both Colab files.download and a File-browser download failed to produce
  an observed native download. A timed-out wait also reset the automation
  runtime. No successful download was claimed. Instead, a read-only visible
  notebook artifact transferred the results-only ZIP in chunks; byte length,
  ZIP integrity, archive SHA256 and every member hash were verified before
  import. Evidence: results/colab_import_20261118/import_manifest.json.
- F022: the first import failed its processed-VAL checksum assertion. All 1,276
  BIRD schemas on the laptop were stale metadata fallbacks; the BIRD database
  files were absent there, so an attempted local actual-DDL read also failed.
  Recovering only schema text from the saved prompts reproduced the entire
  Colab VAL file byte for byte. The old local file was preserved. No question,
  gold SQL or cohort changed and no benchmark was rerun to resolve this.
  Evidence: results/validation_schema_investigation_20261118.json.
- F023: the restored full baseline has 26 gold-error cases and unavailable
  difficulty labels; these are visible evaluation limitations. Keep them in
  the full denominator and report difficulty as unknown. Do not silently
  filter them or relabel them as model failures or confirmed label noise.
  Evidence: results/free_runs_batch8_v2/base_zero_evaluation.
- F003 follow-up: Colab Secrets showed no saved secret. The zero cash budget
  and owner-managed Free tier/billing-disabled credential gate remain; no
  hosted API request, key creation or provider-terms acceptance occurred.

- F024: a nested Windows-to-WSL quoted commit command truncated the unpublished
  title and did not push. A WSL read-only remote lookup then stalled; a headless
  retry returned 128 and gh was absent in WSL. The title was corrected before
  publication, and already-authenticated native Windows Git/gh pushed the
  verified commit successfully. The stalled read-only request was stopped.
  Training/data/results were unaffected. Evidence:
  results/publication_checkpoint_20261118.json.
