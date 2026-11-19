# Saved gold-query failure analysis

All 720 failures from the saved 16,371-query TRAIN/VAL audit are accounted for exactly once.
This report classifies saved evidence. It runs no queries and opens no final-test content.

| Observed mechanism | Queries |
|---|---:|
| compound select ordering | 2 |
| missing column | 208 |
| missing table | 142 |
| resource memory limit | 28 |
| resource result limit | 5 |
| resource row limit | 138 |
| resource timeout | 197 |

## Interpretation

- Missing table/column errors establish incompatibility between saved SQL and the frozen SQLite snapshot; they do not identify which source is at fault.
- Resource-limit failures do not establish annotation errors.
- No label, database, training eligibility, comparator or resource limit was changed.
- The near-100-percent gold-execution target remains unmet.
- Gold-error VAL questions remain in the evaluation denominator.

## Most frequent SQL errors

| Saved SQLite error | Count |
|---|---:|
| no such table: Order Details | 72 |
| no such column: T2.CompanyName | 49 |
| no such table: EmployeeTerritories | 35 |
| no such table: Territories | 19 |
| no such column: CompanyName | 15 |
| no such column: T1.CompanyName | 12 |
| no such column: Salary | 11 |
| no such table: PersonPhone | 10 |
| no such column: Title | 9 |
| no such column: T1.UnitPrice | 6 |

Full case IDs, split/database breakdowns, prior relaxed probes and source hashes: [gold_failure_analysis_20261119.json](../results/gold_failure_analysis_20261119.json).
