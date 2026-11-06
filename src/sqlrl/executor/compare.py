"""Bag semantics, strict column order, NULL equality, numeric tolerance."""
import json
import math
from collections import Counter, deque


def _cell(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        # Integer/count answers must not gain a relative tolerance at large
        # magnitudes. Mixed int/float values must also be exactly equal.
        if isinstance(a,int) or isinstance(b,int):
            return a == b
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8)
    return type(a) is type(b) and a == b


def _row(a, b):
    return len(a) == len(b) and all(_cell(x, y) for x, y in zip(a, b))


def matches(gold, pred, *, ordered=False, gold_columns=None, pred_columns=None):
    if gold_columns is not None and pred_columns is not None and gold_columns != pred_columns:
        return False
    if len(gold) != len(pred):
        return False
    if ordered:
        return all(_row(a, b) for a, b in zip(gold, pred))
    # Fast path preserves multiplicity. Reject nonfinite numbers even if equal.
    try:
        canonical = lambda rows: Counter(json.dumps(r, sort_keys=True, allow_nan=False) for r in rows)
        if canonical(gold) == canonical(pred):
            return True
    except ValueError:
        return False
    # Approximate equality is not transitive. Use a one-to-one bipartite match,
    # not sorting/rounding/set conversion. Bound the quadratic fallback.
    if len(gold) > 500:
        raise ValueError("comparison_limit: over 500 rows require exact bag equality")
    edges = [[j for j, b in enumerate(pred) if _row(a, b)] for a in gold]
    assigned = {}  # predicted index -> gold index
    for root in range(len(gold)):
        queue, parent, seen = deque([root]), {}, {root}
        endpoint = None
        while queue and endpoint is None:
            current = queue.popleft()
            for j in edges[current]:
                if j in parent:
                    continue
                parent[j] = current
                if j not in assigned:
                    endpoint = j
                    break
                other = assigned[j]
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        if endpoint is None:
            return False
        while endpoint is not None:
            i = parent[endpoint]
            previous = next((j for j, owner in assigned.items() if owner == i), None)
            assigned[endpoint] = i
            endpoint = previous
    return True
