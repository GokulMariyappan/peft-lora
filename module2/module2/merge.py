"""Merge an adapter into an FP16 base, verify numerical/output parity, then export."""
import argparse
import json
import os
import shutil
import subprocess
import sys

import torch
import yaml
import transformers
import peft
from peft import PeftModel

from common import (
    load_tokenizer, load_base, load_gsm8k, make_prompt, generate,
    sha256_file, cleanup,
)


@torch.no_grad()
def probe(model, tokenizer, prompts, max_new=96):
    logits, texts = [], []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model(**encoded).logits[0, -1].float().cpu()
        logits.append(output)
        texts.append(generate(model, tokenizer, [prompt], max_new, 1)[0])
    return logits, texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--max_new", type=int, default=96)
    parser.add_argument("--logit_atol", type=float, default=0.05)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found. Merging requires a CUDA runtime in this implementation.")

    config_path = os.path.join(args.run_dir, "run_config.yaml")
    manifest_path = os.path.join(args.run_dir, "manifest.json")
    if not os.path.isfile(config_path) or not os.path.isfile(manifest_path):
        raise FileNotFoundError("Run is incomplete: run_config.yaml and manifest.json are required")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    method = config["method"].lower()
    if method == "pissa" and not manifest.get("pissa_converted"):
        raise SystemExit("REFUSED: raw PiSSA adapter is not converted to a standard LoRA delta")

    model_id = config["model"]["id"]
    revision = config["model"].get("revision")
    adapter_dir = os.path.join(args.run_dir, "adapter")
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Missing adapter directory: {adapter_dir}")

    tokenizer = load_tokenizer(model_id, revision)
    base = load_base(model_id, revision, "fp16")  # Merge validation is deliberately FP16.
    adapted = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).eval()

    gsm = load_gsm8k(3)
    prompts = [make_prompt(question) for question, _ in gsm]
    if len(prompts) != 3:
        raise RuntimeError(f"Expected 3 parity prompts, received {len(prompts)}")

    unmerged_logits, unmerged_text = probe(adapted, tokenizer, prompts, args.max_new)
    merged_model = adapted.merge_and_unload().eval()
    merged_logits, merged_text = probe(merged_model, tokenizer, prompts, args.max_new)

    max_diffs = [float((before - after).abs().max().item()) for before, after in zip(unmerged_logits, merged_logits)]
    allclose = [
        bool(torch.allclose(before, after, atol=args.logit_atol, rtol=5e-3))
        for before, after in zip(unmerged_logits, merged_logits)
    ]
    same_text = [before == after for before, after in zip(unmerged_text, merged_text)]
    same_argmax = [int(before.argmax()) == int(after.argmax()) for before, after in zip(unmerged_logits, merged_logits)]
    passed = all(allclose) and all(same_text) and all(same_argmax)
    parity = {
        "passed": passed,
        "texts_identical": same_text,
        "argmax_identical": same_argmax,
        "logits_within_tolerance": allclose,
        "max_abs_logit_diff": max_diffs,
        "logit_atol": args.logit_atol,
        "logit_rtol": 5e-3,
        "n_prompts": len(prompts),
        "comparison_precision": "fp16",
    }
    print(json.dumps(parity, indent=2))

    output_dir = os.path.join(args.run_dir, "merged")
    flag_path = os.path.join(args.run_dir, "MERGE_FLAGGED.txt")
    if not passed:
        # Never leave an old successful export beside a newly failed parity result.
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        with open(flag_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(parity, indent=2))
        raise SystemExit("Parity FAILED: adapter retained, merged export withheld")

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.isfile(weights_path):
        weights_path = os.path.join(adapter_dir, "adapter_model.bin")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError("Could not locate adapter_model.safetensors or adapter_model.bin")
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_commit = None

    metadata = {
        "base_id": model_id,
        "base_revision": revision,
        "adapter_path": adapter_dir,
        "adapter_sha256": sha256_file(weights_path),
        "method": method,
        "parity": parity,
        "git_commit": git_commit,
        "versions": {
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "torch": torch.__version__,
        },
    }
    with open(os.path.join(output_dir, "merge_meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    if os.path.exists(flag_path):
        os.remove(flag_path)
    print(f"Saved verified merged model -> {output_dir}")
    del adapted, merged_model
    cleanup()
