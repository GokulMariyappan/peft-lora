"""Collect completed/failed runs into the Module-2 comparison table."""
import glob
import json
import os

from common import RUNS, RESULTS, write_csv


def read_json(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not read {path}: {exc}")
        return None


rows = []
base_eval = None
for run_dir in sorted(glob.glob(os.path.join(RUNS, "*"))):
    if not os.path.isdir(run_dir):
        continue
    name = os.path.basename(run_dir)
    if name.startswith("base-"):
        base_eval = read_json(os.path.join(run_dir, "eval.json"))
        continue

    metrics = read_json(os.path.join(run_dir, "metrics.json"))
    evaluation = read_json(os.path.join(run_dir, "eval.json"))
    manifest = read_json(os.path.join(run_dir, "manifest.json"))
    failure_path = os.path.join(run_dir, "FAILED.txt")
    merge_flag_path = os.path.join(run_dir, "MERGE_FLAGGED.txt")
    merged_ok = os.path.isfile(os.path.join(run_dir, "merged", "merge_meta.json"))

    if metrics is None:
        rows.append({
            "run": name,
            "method": "FAILED" if os.path.isfile(failure_path) else "INCOMPLETE",
            "failure": os.path.isfile(failure_path),
            "merge_status": "flagged" if os.path.isfile(merge_flag_path) else "not_run",
        })
        continue

    rows.append({
        "run": name,
        "method": metrics.get("method"),
        "gsm8k_acc": evaluation.get("gsm8k_acc") if evaluation else None,
        "n_gsm": evaluation.get("n_gsm") if evaluation else None,
        "val_ppl": evaluation.get("val_ppl") if evaluation else None,
        "lr": manifest.get("lr") if manifest else None,
        "peak_mem_gib": metrics.get("peak_mem_gib"),
        "sec_per_step": metrics.get("sec_per_step_median"),
        "delta_vs_lora": None,
        "seeds": 1,
        "trainable": metrics.get("trainable_params"),
        "pct": metrics.get("trainable_pct"),
        "steps": metrics.get("steps"),
        "base_precision": metrics.get("base_precision"),
        "loss_last10": round(metrics["loss_last10_mean"], 3) if metrics.get("loss_last10_mean") is not None else None,
        "merge_status": "passed" if merged_ok else ("flagged" if os.path.isfile(merge_flag_path) else "not_run"),
        "failure": False,
    })

lora = next((row for row in rows if row.get("method") == "lora" and row.get("gsm8k_acc") is not None), None)
if lora:
    for row in rows:
        if row.get("gsm8k_acc") is not None:
            row["delta_vs_lora"] = round(row["gsm8k_acc"] - lora["gsm8k_acc"], 2)

columns = [
    "method", "gsm8k_acc", "val_ppl", "lr", "peak_mem_gib", "sec_per_step",
    "delta_vs_lora", "seeds", "trainable", "pct", "steps", "base_precision", "merge_status",
]
lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
if base_eval:
    lines.append(
        "| base (no FT) | "
        + " | ".join(str(value) for value in [base_eval.get("gsm8k_acc"), base_eval.get("val_ppl"), "n/a", "", "", "", "", "", "", "", "4bit", "n/a"])
        + " |"
    )
for row in rows:
    lines.append("| " + " | ".join(str(row.get(column, "")) if row.get(column) is not None else "" for column in columns) + " |")

os.makedirs(RESULTS, exist_ok=True)
with open(os.path.join(RESULTS, "week5_results.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
if rows:
    csv_columns = columns + ["run", "n_gsm", "loss_last10", "failure"]
    write_csv(os.path.join(RESULTS, "week5_results.csv"), [
        {column: row.get(column) for column in csv_columns} for row in rows
    ])
print("\n".join(lines))
