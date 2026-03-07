import glob
import json
import os
from statistics import mean

from absl import app, flags


FLAGS = flags.FLAGS
flags.DEFINE_string("input", None, "Path to a merged results.jsonl file or a rerank run directory.")


def _load_rows(input_path: str):
    if os.path.isdir(input_path):
        merged_path = os.path.join(input_path, "results.jsonl")
        jsonl_candidates = [merged_path] if os.path.exists(merged_path) else sorted(
            glob.glob(os.path.join(input_path, "results_rank*.jsonl"))
        )
    else:
        jsonl_candidates = [input_path]

    rows = []
    for jsonl_path in jsonl_candidates:
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _mean_selected_metric(rows, method_name, metric_name):
    values = []
    for row in rows:
        method_metrics = row.get("selected_metrics", {}).get(method_name, {})
        if metric_name in method_metrics:
            values.append(float(method_metrics[metric_name]))
    return mean(values) if values else None


def _mean_score_margin(rows, score_name):
    values = []
    for row in rows:
        score_values = row.get("scores", {}).get(score_name)
        if score_values:
            values.extend(float(value) for value in score_values)
    return mean(values) if values else None


def _counterfactual_accuracy(rows):
    values = []
    for row in rows:
        cope_scores = row.get("scores", {}).get("cope")
        if not cope_scores:
            continue
        selected_index = row.get("selected_index", {}).get("cope", 0)
        values.append(1.0 if float(cope_scores[selected_index]) > 0 else 0.0)
    return mean(values) if values else None


def main(_):
    if not FLAGS.input:
        raise ValueError("--input is required.")

    rows = _load_rows(FLAGS.input)
    if not rows:
        raise ValueError("No result rows found.")

    summary = {
        "num_prompts": len(rows),
        "counterfactual_discrimination_accuracy": _counterfactual_accuracy(rows),
        "mean_raw_score": _mean_score_margin(rows, "raw"),
        "mean_pmi_score": _mean_score_margin(rows, "pmi"),
        "mean_cope_score": _mean_score_margin(rows, "cope"),
        "mean_cope_lse_score": _mean_score_margin(rows, "cope_lse"),
    }

    metric_names = set()
    for row in rows:
        metric_names.update(row.get("metrics", {}).keys())

    for metric_name in sorted(metric_names):
        for method_name in ["single", "raw", "pmi", "cope", "cope_lse", "primary"]:
            metric_value = _mean_selected_metric(rows, method_name, metric_name)
            if metric_value is not None:
                summary[f"{method_name}_{metric_name}"] = metric_value

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app.run(main)
