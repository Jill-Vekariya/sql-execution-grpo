import argparse
import json
from pathlib import Path

from sqlrl.common import sha256, write_json
from sqlrl.eval import paired_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    a, b = [[json.loads(line) for line in Path(path).read_text().splitlines()] for path in [args.a,args.b]]
    by_a, by_b = {r["id"]:r for r in a}, {r["id"]:r for r in b}
    if len(by_a) != len(a) or len(by_b) != len(b) or set(by_a) != set(by_b):
        raise ValueError("Paired comparisons require the same unique question IDs")
    keys = sorted(by_a)
    result = paired_comparison([int(by_a[k]["correct"]) for k in keys], [int(by_b[k]["correct"]) for k in keys], [by_a[k].get("dataset", "") + ":" + by_a[k]["db_id"] for k in keys])
    write_json(args.output, dict(result, inputs={args.a:sha256(args.a), args.b:sha256(args.b)}))


if __name__ == "__main__":
    main()
