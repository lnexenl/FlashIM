"""Merge JSONL launch logs into one deterministic entry per distinct compiler input."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def merge(paths):
    specializations = defaultdict(lambda: defaultdict(set))
    for path in paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            route = tuple(record.pop("route"))
            arches = record.pop("architectures")
            key = json.dumps(record, sort_keys=True, separators=(",", ":"))
            specializations[key][route].update(arches)
    if not specializations:
        raise ValueError("empty launch log")
    entries = []
    for key, routes in specializations.items():
        record = json.loads(key)
        entry = {"kernel": record.pop("kernel"), **record}
        entry["routes"] = [[list(route), sorted(arches)] for route, arches in sorted(routes.items())]
        entries.append(json.dumps(entry, separators=(",", ":")))
    return "\n".join(sorted(entries)) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("triton.jsonl"))
    args = parser.parse_args()
    args.output.write_text(merge(args.logs))
