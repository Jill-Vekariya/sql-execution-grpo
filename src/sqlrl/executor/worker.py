"""Small isolated child. Never import an ML library or the training model.

This bounds accidental/adversarial SQL on trusted benchmark DB files. It is not
an OS sandbox for malicious SQLite binaries or SQLite engine vulnerabilities.
"""
import json
import math
import os
import re
import resource
import sqlite3
import sys
import time
from pathlib import Path


def require_read_prefix(sql):
    """Allow SELECT/WITH after whitespace/comments; SQLite parses the body.

    This prefix check is not the security boundary. The read-only connection
    and SQLite authorizer below deny writes, including writes inside WITH.
    sqlite3.execute also rejects multiple statements.
    """
    remaining = sql.lstrip()
    while remaining.startswith(("--", "/*")):
        if remaining.startswith("--"):
            end = remaining.find("\n")
            if end < 0:
                raise ValueError("No SELECT/WITH statement")
            remaining = remaining[end+1:].lstrip()
        else:
            end = remaining.find("*/", 2)
            if end < 0:
                raise ValueError("Unclosed SQL comment")
            remaining = remaining[end+2:].lstrip()
    token = re.match(r"[A-Za-z_]+", remaining)
    if token is None or token.group().upper() not in {"SELECT", "WITH"}:
        raise ValueError("Only SELECT/WITH statements are allowed")


def run(path, sql, limits):
    cap = limits["memory_mb"] * 1024 * 1024
    # RLIMIT_AS blocks future allocation but does not evict existing mappings.
    # A cap below the interpreter's baseline must therefore fail explicitly.
    current_vms = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    if current_vms > cap:
        return {"status": "memory_limit", "error": "Cap is below interpreter address space"}
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    cpu = max(1, math.ceil(limits["timeout_s"]) + 1)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    try:
        require_read_prefix(sql)
    except ValueError as error:
        return {"status": "rejected", "error": str(error)[:300]}
    con = sqlite3.connect(Path(path).as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        con.enable_load_extension(False)
        con.execute("PRAGMA query_only=ON")
        con.execute("PRAGMA trusted_schema=OFF")
        con.execute("PRAGMA temp_store=MEMORY")
        con.execute("PRAGMA cache_size=-2048")
        con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1000000)
        con.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, limits["max_sql_chars"])
        con.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 200)
        con.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100)
        con.setlimit(sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 50)

        def authorize(action, first, second, database, source):
            if action == sqlite3.SQLITE_FUNCTION:
                name = (second or first or "").lower()
                return sqlite3.SQLITE_DENY if name in {"load_extension", "readfile", "writefile", "eval", "fts3_tokenizer"} else sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ:
                if database not in {None, "main"} or (first or "").lower().startswith(("pragma_", "sqlite_")):
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_OK if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE} else sqlite3.SQLITE_DENY

        con.set_authorizer(authorize)
        deadline = time.monotonic() + limits["timeout_s"]
        con.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        cursor = con.execute(sql)
        columns = len(cursor.description or [])
        rows, byte_count = [], 0
        for row in cursor:
            if len(rows) >= limits["max_rows"]:
                return {"status": "row_limit"}
            values = [{"blob_hex": x.hex()} if isinstance(x, bytes) else x for x in row]
            encoded = json.dumps(values, allow_nan=False)
            byte_count += len(encoded.encode("utf-8"))
            if byte_count > limits["max_result_bytes"]:
                return {"status": "result_limit"}
            rows.append(values)
        return {"status": "ok", "rows": rows, "columns": columns}
    finally:
        con.close()


if __name__ == "__main__":
    try:
        request = json.load(sys.stdin)
        result = run(sys.argv[1], request["sql"], request["limits"])
    except MemoryError:
        result = {"status": "memory_limit"}
    except (sqlite3.Error, ValueError, OSError) as error:
        status = "timeout" if "interrupted" in str(error) else "sql_error"
        result = {"status": status, "error": str(error)[:300]}
    print(json.dumps(result, allow_nan=False))
