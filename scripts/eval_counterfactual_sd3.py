import glob
import json
import os
from collections import Counter
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


def _selection_disagreement(rows, method_a, method_b):
    return sum(
        row.get("selected_index", {}).get(method_a) != row.get("selected_index", {}).get(method_b)
        for row in rows
    )


def _cross_eval_mean(rows, method_name, score_name):
    values = []
    for row in rows:
        idx = row.get("selected_index", {}).get(method_name)
        if idx is None:
            continue
        score_values = row.get("scores", {}).get(score_name)
        if score_values is None:
            continue
        values.append(float(score_values[idx]))
    return mean(values) if values else None


def _positive_cope_margin(rows, method_name):
    values = []
    for row in rows:
        idx = row.get("selected_index", {}).get(method_name)
        if idx is None:
            continue
        cope_scores = row.get("scores", {}).get("cope")
        if cope_scores is None:
            continue
        values.append(float(cope_scores[idx]) > 0)
    if not values:
        return None
    return {
        "count": int(sum(values)),
        "total": len(values),
        "rate": float(sum(values) / len(values)),
    }


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

    methods = ["single", "raw", "pmi", "cope", "cope_lse"]
    score_names = ["raw", "pmi", "cope", "cope_lse"]

    summary["negative_mode_counts"] = dict(
        Counter(row.get("negative_mode", "unknown") for row in rows)
    )
    summary["selection_disagreement"] = {}
    for index, method_a in enumerate(methods):
        for method_b in methods[index + 1:]:
            summary["selection_disagreement"][f"{method_a}_vs_{method_b}"] = _selection_disagreement(
                rows,
                method_a,
                method_b,
            )

    summary["cross_eval_mean"] = {}
    for score_name in score_names:
        summary["cross_eval_mean"][score_name] = {}
        for method_name in methods:
            value = _cross_eval_mean(rows, method_name, score_name)
            if value is not None:
                summary["cross_eval_mean"][score_name][method_name] = value

    summary["positive_cope_margin"] = {}
    for method_name in methods:
        result = _positive_cope_margin(rows, method_name)
        if result is not None:
            summary["positive_cope_margin"][method_name] = result

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
