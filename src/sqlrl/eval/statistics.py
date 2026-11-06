"""Database-cluster bootstrap respects the database-held-out design."""
import random
from collections import defaultdict


def _percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    lo, hi = int(index), min(int(index) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def bootstrap_ci(values, *, groups=None, seed=42, samples=2000):
    if not values:
        return [None, None]
    groups = groups if groups is not None else list(range(len(values)))
    if len(values) != len(groups):
        raise ValueError("Values and group IDs must align")
    by_group = defaultdict(list)
    for group, value in zip(groups, values):
        by_group[group].append(value)
    clusters = [(sum(v), len(v)) for _, v in sorted(by_group.items())]
    rng, estimates = random.Random(seed), []
    for _ in range(samples):
        draw = rng.choices(clusters, k=len(clusters))
        estimates.append(sum(x[0] for x in draw) / sum(x[1] for x in draw))
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def paired_comparison(a, b, groups, *, seed=42, samples=2000):
    if len(a) != len(b) or len(a) != len(groups):
        raise ValueError("Paired systems must contain the same aligned questions")
    delta = [y - x for x, y in zip(a, b)]
    ci = bootstrap_ci(delta, groups=groups, seed=seed, samples=samples)
    return {"wins": sum(y > x for x, y in zip(a, b)), "losses": sum(y < x for x, y in zip(a, b)),
            "ties": sum(y == x for x, y in zip(a, b)), "delta": sum(delta)/len(delta) if delta else None,
            "delta_ci95": ci, "distinguishable_at_95pct": bool(ci[0] is not None and (ci[0] > 0 or ci[1] < 0))}


def summarize(rows, *, seed=42, samples=2000):
    if not rows:
        raise ValueError("Cannot evaluate an empty split")
    n = len(rows)
    groups = [r.get("dataset", "") + ":" + r["db_id"] for r in rows]
    tokens = [r["output_tokens"] for r in rows if r.get("output_tokens") is not None]
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    generation_latencies = [r["generation_latency_s"] for r in rows if r.get("generation_latency_s") is not None]
    difficulties = sorted({r.get("difficulty", "unknown") for r in rows})
    return {
        "n": n, "databases": len(set(groups)), "execution_accuracy": sum(r["correct"] for r in rows)/n,
        "execution_accuracy_ci95": bootstrap_ci([int(r["correct"]) for r in rows], groups=groups, seed=seed, samples=samples),
        "ci_method": "database_cluster_percentile_bootstrap", "bootstrap_seed": seed, "bootstrap_samples": samples,
        "valid_sql_rate": sum(r["valid"] for r in rows)/n,
        "timeout_rate": sum(r["status"] == "timeout" for r in rows)/n,
        "gold_errors": sum(r["status"] == "gold_error" for r in rows),
        "average_output_tokens": sum(tokens)/len(tokens) if tokens else None,
        "token_counts_complete": len(tokens) == n,
        "total_cost_usd": sum(costs) if len(costs) == n else None,
        "average_latency_s": sum(r["latency_s"] for r in rows)/n,
        "average_generation_latency_s": sum(generation_latencies)/len(generation_latencies) if len(generation_latencies) == n else None,
        "by_difficulty": {d: {"n": sum(r.get("difficulty", "unknown") == d for r in rows),
            "accuracy": sum(r["correct"] for r in rows if r.get("difficulty", "unknown") == d)/sum(r.get("difficulty", "unknown") == d for r in rows),
            "ci95":bootstrap_ci([int(r["correct"]) for r in rows if r.get("difficulty", "unknown") == d],
                groups=[g for r,g in zip(rows,groups) if r.get("difficulty", "unknown") == d], seed=seed, samples=samples)} for d in difficulties},
    }
