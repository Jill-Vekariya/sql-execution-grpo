# Read and rewrite the three core components

## 1. Executor

Start with `src/sqlrl/executor/runner.py`. `Limits` holds the declared resource
budget; `QueryResult` makes successful execution and failures explicit.
`execute()` starts one interpreter per query and kills it at the wall timeout.
It never scores truncated output as a complete result.

Then read `worker.py`. It uses a small SELECT/WITH prefix allow-list, SQLite's
own parser, a read-only immutable connection and an authorizer that rejects
write operations. The prefix check is not the security boundary: a WITH statement
can contain a write, so SQLite authorization must remain enabled. The child has
memory, CPU, file and output bounds. This protects SQL execution on trusted
benchmark snapshots; it is not an OS security boundary against malicious DB
binaries or SQLite vulnerabilities.

Read `compare.py` last. `matches()` checks the number of columns, preserves
duplicate rows and uses one-to-one matching when row order is irrelevant. Only
an outer gold ORDER BY makes row order matter. Integers and mixed int/float
answers must be exactly equal. Float/float values use the documented tolerance.
Large approximate bag matches fail explicitly rather than run an unbounded
quadratic comparison. Changing any of these rules can change reported accuracy.

The executor tests cover writes, WITH writes, multiple statements, extension/file
access, timeouts, row/result/memory bounds, special URI characters and pathological
parser input. Comparison tests cover duplicates, empty results, NULLs, column
order, numeric types and large integer near misses.

## 2. Reward

Read `src/sqlrl/reward/__init__.py`. `score()` returns individual reward terms;
`Reward.total` adds them. Execution contributes 1, valid execution contributes
0.02, and format contributes +0.01 or -0.01. Excessive output and timeout penalties
are separate. This is a provisional pre-training reward, not a validated GRPO
configuration.

`raw_execution` records single-snapshot agreement. `execution` can withhold that
credit for obvious literal/table-free shortcuts or unverified empty results.
Counterfactual databases can demonstrate when a coincidentally correct query
fails under changed contents. The heuristics can reject legitimate solutions;
they are not a semantic-equivalence proof. Phase 4 must freeze and ablate this
choice before any RL result is claimed.

Inspect `results/synthetic_checks/attack_rewards.jsonl` after running the fixture.
These are real component outputs from hand-written attacks, not samples learned
by a trained model. The cheating analysis document keeps that distinction clear.

## 3. Evaluation

Read `src/sqlrl/eval/run.py`. `evaluate()` requires exactly one prediction per
question ID, including failed generations. It executes gold and predicted SQL,
extracts the answer, calls the shared comparator and saves each outcome. Gold
errors, invalid SQL and timeout failures remain visible in the denominator.
Changing the policy for nonexecuting gold labels requires a new documented
eligibility decision, shared by SFT and RL.

`prompts.py` holds the fixed prompt and answer extractor. The schema, question
and BIRD evidence are shared across systems. Few-shot examples come from train
databases only; they are saved by ID. No sample-value retrieval is added silently.

`statistics.py` computes database-cluster bootstrap intervals and paired
differences. Questions from one database are not treated as independent bootstrap
units. Difficulty labels absent from the upstream training annotations remain
`unknown`; the code does not invent official benchmark difficulty labels.

`generate.py` handles model/API generation separately from verification. It saves
raw answers, token counts, errors and any measured cost. No positive API budget,
model or credentials are supplied by default. Unknown costs are null. The HTTP
provider adapter must be verified against the eventual frontier model before a
paid comparison is run.

## What the CPU SFT smoke proves

`scripts/smoke_sft_cpu.py` performs two optimizer steps on a randomly initialized
tiny Qwen3 model. It checks the pinned TRL/PEFT interfaces, finite loss, changed
LoRA weights and adapter saving. It does not test a pretrained 4B model, SQL
generalization, a useful adapter, or any claimed accuracy gain.

The full SFT implementation is `src/sqlrl/train_sft/run.py`. Its actual benchmark
run is blocked until CUDA compute and a complete, frozen corpus are available.
The final test remains outside these Phase 0–3 entry points.
