# Dataset audit

The maintained machine-readable record is `results/literature_audit.json` for
upstream claims and `results/data_audit.json` for locally measured train/val
counts and checksums. Published totals include held-out splits, unlike this
project's training corpus. Spider 1.0's official test was publicly released;
this project retains official dev as its preregistered final cohort. BIRD's
official test labels remain hidden. No held-out content was opened to check
these public dataset-status statements.

| Corpus | Upstream license | Published total | Official source |
|---|---|---|---|
| Spider 1.0 | CC BY-SA 4.0 | 10,181 questions; 200 DBs | https://yale-lily.github.io/spider |
| BIRD-SQL 2023 | CC BY-SA 4.0, announced April 2024 | 12,751 pairs; 95 DBs; 33.4 GB | https://bird-bench.github.io/ |

These are the maintainers' corpus totals. Actual official TRAIN metadata counts,
available local databases, duplicate removals and validation memberships come
from the saved audit, not the table above. Use the original BIRD train corpus;
the later filtered subset is a different training distribution.

Spider's authors document past annotation/label revisions. BIRD's authors
document dev cleaning, a filtered training release and more recent Mini-Dev
corrections. Successful execution alone does not establish that SQL expresses
the question correctly. Gold queries that fail execution are logged as unresolved
cases, with their original source IDs preserved; no automatic relabeling occurs.

Official dev serves as the final test for both corpora. It is not the validation
set. Project validation is made solely from database-disjoint official TRAIN
data. Reading held-out questions to remove cross-final duplicates would conflict
with the explicit no-access-before-Phase-9 rule; that check is pending and never
reported as completed. The final descriptor checksum is likewise null until a
separate Phase 9 protocol seals an actual final manifest.

The selective ZIP downloader opens TRAIN annotation/database members only.
Spider tables.json is shared schema metadata and is filtered to TRAIN IDs.
BIRD's DBs are compressed inside a second archive. At a small initial download
cap only metadata is available; the source report explicitly lists omissions.
The full TRAIN archive can be downloaded with a larger download-byte cap, using
the same pinned annotations and split assignments. Missing data blocks research
generation/training rather than silently changing its evaluation denominator.

Raw datasets/databases are not checked into Git. Attribution and ShareAlike
obligations apply to dataset-derived material. Review original database
provenance before redistributing database contents; the code's MIT license does
not override any dataset license.
