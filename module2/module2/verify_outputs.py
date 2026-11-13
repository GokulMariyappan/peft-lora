"""Audit expected Module-2 files without loading model weights or requiring a GPU."""
import argparse
import csv
import hashlib
import json
import os

METHODS = ("lora", "dora", "rslora", "pissa")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(path, errors, label=None):
    if not os.path.isfile(path):
        errors.append(f"Missing {label or path}")
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", default="2e-4")
    args = parser.parse_args()

    root = "."
    errors, warnings = [], []
    data_dir = os.path.join(root, "data", args.model)
    check_file(os.path.join(data_dir, "train.jsonl"), errors, "training JSONL")
    check_file(os.path.join(data_dir, "val.jsonl"), errors, "validation JSONL")

    counts_path = os.path.join(root, "results", f"param_counts_{args.model}.json")
    if check_file(counts_path, errors, "parameter-count report"):
        counts = load_json(counts_path)
        variants = counts if isinstance(counts, list) else counts.get("variants", [])
        seen = {row.get("method") for row in variants}
        for method in METHODS:
            if method not in seen:
                errors.append(f"Parameter report is missing method: {method}")

    for method in METHODS:
        run_id = f"{args.model}-{method}-fixed-lr{args.lr}-s{args.seed}"
        run_dir = os.path.join(root, "runs", run_id)
        if not os.path.isdir(run_dir):
            errors.append(f"Missing run directory: {run_id}")
            continue
        if os.path.isfile(os.path.join(run_dir, "FAILED.txt")):
            errors.append(f"Training failed: {run_id} (see FAILED.txt)")
            continue

        required = ["run_config.yaml", "metrics.json", "manifest.json", "eval.json", "samples.csv"]
        for filename in required:
            check_file(os.path.join(run_dir, filename), errors, f"{run_id}/{filename}")

        adapter_file = None
        for filename in ("adapter_model.safetensors", "adapter_model.bin"):
            candidate = os.path.join(run_dir, "adapter", filename)
            if os.path.isfile(candidate):
                adapter_file = candidate
                break
        if adapter_file is None:
            errors.append(f"Missing adapter weights: {run_id}/adapter/")
        else:
            manifest_path = os.path.join(run_dir, "manifest.json")
            if os.path.isfile(manifest_path):
                manifest = load_json(manifest_path)
                actual_hash = sha256(adapter_file)
                if manifest.get("adapter_sha256") != actual_hash:
                    errors.append(f"Adapter SHA256 mismatch: {run_id}")

        if method == "pissa":
            check_file(os.path.join(run_dir, "adapter_init", "adapter_model.safetensors"), errors,
                       f"{run_id}/adapter_init/adapter_model.safetensors")

        merge_meta = os.path.join(run_dir, "merged", "merge_meta.json")
        merge_flag = os.path.join(run_dir, "MERGE_FLAGGED.txt")
        if os.path.isfile(merge_meta):
            metadata = load_json(merge_meta)
            if not metadata.get("parity", {}).get("passed"):
                errors.append(f"Merged export does not record passed parity: {run_id}")
        elif os.path.isfile(merge_flag):
            warnings.append(f"Merge parity flagged for {run_id}; adapter retained, no merged export")
        else:
            warnings.append(f"Merge has not been run for {run_id}")

        samples_path = os.path.join(run_dir, "samples.csv")
        if os.path.isfile(samples_path):
            with open(samples_path, newline="", encoding="utf-8") as f:
                sample_count = sum(1 for _ in csv.DictReader(f))
            if sample_count != 20:
                warnings.append(f"{run_id}: expected 20 saved samples, found {sample_count}")

    base_eval = os.path.join(root, "runs", f"base-{args.model}", "eval.json")
    check_file(base_eval, errors, "base-model evaluation")
    for filename in ("week5_results.md", "week5_results.csv"):
        check_file(os.path.join(root, "results", filename), errors, f"results/{filename}")

    print("MODULE 2 ARTIFACT AUDIT")
    print(f"Errors: {len(errors)} | Warnings: {len(warnings)}")
    for item in errors:
        print("ERROR:", item)
    for item in warnings:
        print("WARNING:", item)
    if not errors:
        print("Required artifacts are present. Review warnings and inspect metrics before reporting conclusions.")
    raise SystemExit(1 if errors else 0)
