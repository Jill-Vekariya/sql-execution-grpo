# Data licenses and attribution

The MIT license covers project-authored software and synthetic fixtures.
Dataset-derived split manifests and benchmark query excerpts in result reports
remain under **Creative Commons Attribution-ShareAlike 4.0**:
https://creativecommons.org/licenses/by-sa/4.0/ . No raw databases are committed.

- **Spider**: Tao Yu et al., *Spider: A Large-Scale Human-Labeled Dataset for
  Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*, EMNLP 2018.
  Source and license declaration: https://yale-lily.github.io/spider .
- **BIRD**: Jinyang Li et al., *Can LLM Already Serve as A Database Interface?
  A BIg Bench for Large-Scale Database Grounded Text-to-SQLs*, NeurIPS 2023.
  Source and April 2024 license update: https://bird-bench.github.io/ .

Changes made here: deterministic database-held-out train/validation partitioning,
normalized-question duplicate removal and conversion into project manifests.
Original gold SQL is not corrected in place. Upstream corpus claims and links
are preserved in results/literature_audit.json. Individual underlying database
sources can have additional terms/provenance; consult them before redistribution.
