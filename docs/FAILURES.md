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
