import json
import math
import os
from statistics import mean, pstdev

from absl import app, flags


FLAGS = flags.FLAGS
flags.DEFINE_multi_string(
    "input",
    [],
    "Run directories or compare_summary.json files to aggregate. Repeat the flag for multiple runs.",
)


def _flatten(prefix, value, output):
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            _flatten(next_prefix, item, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            output[prefix] = float(value)


def _load_summary(path):
    summary_path = path
    if os.path.isdir(path):
        summary_path = os.path.join(path, "compare_summary.json")
    with open(summary_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main(_):
    if not FLAGS.input:
        raise ValueError("At least one --input must be provided.")

    rows = [_load_summary(path) for path in FLAGS.input]
    flattened = []
    for row in rows:
        current = {}
        _flatten("", row, current)
        flattened.append(current)

    all_keys = sorted({key for row in flattened for key in row.keys()})
    aggregate = {
        "num_runs": len(rows),
        "inputs": list(FLAGS.input),
        "metrics": {},
    }

    for key in all_keys:
        values = [row[key] for row in flattened if key in row]
        if not values:
            continue
        aggregate["metrics"][key] = {
            "count": len(values),
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    app.run(main)
