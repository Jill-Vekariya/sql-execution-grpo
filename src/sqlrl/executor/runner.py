"""Parent-side API. Each SQLite execution gets a new isolated interpreter."""
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import sqlglot
from sqlglot import exp


@dataclass(frozen=True)
class Limits:
    timeout_s: float = 3.0
    max_rows: int = 10000
    memory_mb: int = 256
    max_sql_chars: int = 20000
    max_result_bytes: int = 2000000


@dataclass
class QueryResult:
    status: str
    rows: list = field(default_factory=list)
    columns: int = 0
    elapsed_s: float = 0.0
    error: str | None = None


def parse_query(sql):
    statements = sqlglot.parse(sql, read="sqlite")
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise ValueError("Exactly one SELECT/WITH read query is allowed")
    tree = statements[0]
    for node in tree.walk():
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Command)):
            raise ValueError("Mutating statements are forbidden, including inside WITH")
    return tree


def ordered_gold(sql):
    """Only the outer query's ORDER BY defines the output ordering contract."""
    return parse_query(sql).args.get("order") is not None


def execute(db_path, sql, limits=Limits()):
    start = time.monotonic()
    if sys.platform != "linux":
        return QueryResult("unsupported_platform", error="Use Linux/WSL for enforced resource limits")
    if len(sql) > limits.max_sql_chars:
        return QueryResult("rejected", error="SQL length limit")
    path = Path(db_path).resolve()
    if not path.is_file():
        return QueryResult("missing_database", error="Database file is missing")
    # Parsing also happens in the bounded child: pathological SQL must not
    # freeze the training process before SQLite is even reached.
    # Immutable reads ignore WAL files. Require a checkpointed, static snapshot.
    if Path(str(path) + "-wal").exists():
        return QueryResult("rejected", error="WAL snapshot must be checkpointed first")
    worker = Path(__file__).with_name("worker.py")
    process = subprocess.Popen(
        [sys.executable, "-I", str(worker), str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
    )
    try:
        output, _ = process.communicate(json.dumps({"sql": sql, "limits": asdict(limits)}), timeout=limits.timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return QueryResult("timeout", elapsed_s=time.monotonic() - start)
    try:
        result = QueryResult(**json.loads(output))
    except (ValueError, TypeError):
        result = QueryResult("worker_error", error=f"Worker exited {process.returncode}; resource cap may have fired")
    result.elapsed_s = time.monotonic() - start
    return result
