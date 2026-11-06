"""Provisional, readable reward. Phase 4 must freeze and ablate its design."""
from dataclasses import dataclass, field

from sqlglot import exp

from sqlrl.eval.prompts import extract_sql
from sqlrl.executor import Limits, execute, matches, ordered_gold, parse_query


@dataclass
class Reward:
    execution: float = 0.0
    raw_execution: float = 0.0
    valid: float = 0.0
    format: float = 0.0
    penalty: float = 0.0
    flags: list[str] = field(default_factory=list)

    @property
    def total(self):
        return self.execution + self.valid + self.format + self.penalty


def score(db_path, gold_sql, answer, *, robust=True, counterfactual_databases=(), limits=Limits()):
    reward = Reward()
    if len(answer) > limits.max_sql_chars:
        reward.penalty = -0.05
        reward.flags.append("output_length_limit")
        return reward
    sql, formatted = extract_sql(answer)
    reward.format = 0.01 if formatted else -0.01
    if sql is None:
        return reward
    gold = execute(db_path, gold_sql, limits)
    if gold.status != "ok":
        raise ValueError("Gold query failed: " + gold.status)
    pred = execute(db_path, sql, limits)
    if pred.status != "ok":
        if pred.status == "timeout":
            reward.penalty = -0.05
        reward.flags.append(pred.status)
        return reward
    reward.valid = 0.02
    try:
        reward.raw_execution = float(matches(gold.rows, pred.rows, ordered=ordered_gold(gold_sql), gold_columns=gold.columns, pred_columns=pred.columns))
    except ValueError:
        reward.flags.append("comparison_limit")
    reward.execution = reward.raw_execution
    if robust and reward.execution:
        gold_tree, pred_tree = parse_query(gold_sql), parse_query(sql)
        gold_tables = list(gold_tree.find_all(exp.Table))
        pred_tables = list(pred_tree.find_all(exp.Table))
        if gold_tables and not pred_tables:
            reward.flags.append("table_free_answer")
        # This is a deliberately narrow heuristic, not a proof against hardcoding.
        projections = getattr(pred_tree, "expressions", [])
        if gold_tables and projections and all(isinstance(x.unalias(), exp.Literal) for x in projections):
            reward.flags.append("literal_only_projection")
        witnessed_nonempty = bool(gold.rows)
        for alternate in counterfactual_databases:
            other_gold, other_pred = execute(alternate, gold_sql, limits), execute(alternate, sql, limits)
            if other_gold.status != "ok" or other_pred.status != "ok":
                reward.flags.append("counterfactual_execution_error")
                continue
            witnessed_nonempty |= bool(other_gold.rows)
            if not matches(other_gold.rows, other_pred.rows, ordered=ordered_gold(gold_sql), gold_columns=other_gold.columns, pred_columns=other_pred.columns):
                reward.flags.append("counterfactual_mismatch")
        if not witnessed_nonempty:
            reward.flags.append("empty_result_unverified")
        if reward.flags:
            reward.execution = 0.0
    # No arbitrary runtime shaping yet: machine load would add confounding noise.
    return reward
