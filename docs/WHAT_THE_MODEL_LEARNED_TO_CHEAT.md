# What the model learned to cheat

**No GRPO model has been trained. No learned cheating has been measured.**

The Phase 0–3 verifier tests anticipated exploit classes: empty-answer agreement,
literal answer hardcoding, table-free unions, LIMIT truncation, duplicate removal,
column permutation, loose numeric/type comparison, multi-statement writes,
extension/file access, recursive hangs and oversized outputs. The saved synthetic
attack report shows the actual component rewards for these examples. Synthetic
attacks are not observations about a trained model.

Single-snapshot execution correctness cannot prove semantic correctness. An
incorrect query can coincidentally produce the right empty/constant result.
The provisional robust reward flags obvious shortcuts and withholds empty-result
credit without a nonempty counterfactual witness. This can also suppress genuine
solutions. Phase 4 must test and ablate that tradeoff before GRPO is run.

Phase 7 will add actual question, gold SQL, generated SQL, reward components and
execution evidence from saved runs, covering the largest validation wins/losses
and highest-reward training generations. Until then, no error taxonomy counts
or improvement claims are populated.
