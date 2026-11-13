"""Evaluate base or adapter: answer-token perplexity, GSM8K accuracy and samples."""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from peft import PeftModel

from common import (
    MODELS, RUNS, short_name, get_revision, load_tokenizer, load_base,
    get_splits, encode_rows, collate, make_prompt, load_gsm8k,
    generate, extract_answer, write_csv, cleanup,
)


@torch.no_grad()
def perplexity(model, rows, pad_id, batch_size=4):
    if not rows:
        raise ValueError("No validation rows available for perplexity evaluation")
    total_nll, total_tokens = 0.0, 0
    loader = DataLoader(rows, batch_size=batch_size, collate_fn=lambda batch: collate(batch, pad_id))
    for batch in loader:
        batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits
        shifted_logits = logits[:, :-1, :].float()
        shifted_labels = batch["labels"][:, 1:]
        total_nll += F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ).item()
        total_tokens += int((shifted_labels != -100).sum().item())
    if total_tokens == 0:
        raise ValueError("Validation set contains no answer tokens")
    return math.exp(total_nll / total_tokens)


def bootstrap_ci(correct, seed=0, samples=2000):
    values = np.asarray(correct, dtype=np.float32)
    rng = np.random.default_rng(seed)
    boot = [100.0 * rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)]
    return [round(float(np.percentile(boot, 2.5)), 1), round(float(np.percentile(boot, 97.5)), 1)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir")
    parser.add_argument("--base_only", action="store_true")
    parser.add_argument("--model", default="qwen3-0.6b", choices=list(MODELS))
    parser.add_argument("--n_gsm", type=int, default=100)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--max_new", type=int, default=256)
    parser.add_argument("--bs_gen", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found. In Colab select Runtime > Change runtime type > T4 GPU.")
    if args.base_only == bool(args.run_dir):
        raise SystemExit("Choose exactly one: --base_only or --run_dir PATH")
    if min(args.n_gsm, args.n_val, args.max_new, args.bs_gen) < 1:
        raise SystemExit("Evaluation counts and generation settings must be positive")

    if args.base_only:
        model_id = MODELS[args.model]
        revision = get_revision(model_id)
        precision = "4bit"
        run_dir = os.path.join(RUNS, f"base-{args.model}")
        adapter_dir = None
    else:
        run_dir = args.run_dir
        config_path = os.path.join(run_dir, "run_config.yaml")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Missing run configuration: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        model_id = config["model"]["id"]
        revision = config["model"].get("revision")
        precision = config.get("base_precision", "4bit")
        adapter_dir = os.path.join(run_dir, "adapter")
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"Missing trained adapter: {adapter_dir}")

    os.makedirs(run_dir, exist_ok=True)
    short = short_name(model_id)
    tokenizer = load_tokenizer(model_id, revision)
    model = load_base(model_id, revision, precision)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    splits = get_splits(tokenizer, short)
    val_rows = encode_rows(tokenizer, splits["val"][:args.n_val])
    ppl = perplexity(model, val_rows, tokenizer.pad_token_id)
    print(f"Validation perplexity ({len(val_rows)} examples): {ppl:.3f}", flush=True)

    gsm = load_gsm8k(args.n_gsm)
    prompts = [make_prompt(question) for question, _ in gsm]
    generations = generate(model, tokenizer, prompts, args.max_new, args.bs_gen)
    rows, correct = [], []
    for index, ((question, gold), generation) in enumerate(zip(gsm, generations)):
        prediction = extract_answer(generation)
        is_correct = prediction is not None and gold is not None and abs(prediction - gold) < 1e-6
        correct.append(is_correct)
        rows.append({
            "id": index,
            "question": question,
            "gold": gold,
            "pred": prediction,
            "correct": is_correct,
            "generation": generation,
        })
    if not correct:
        raise RuntimeError("GSM8K evaluation returned no examples")
    accuracy = 100.0 * float(np.mean(correct))
    confidence_interval = bootstrap_ci(correct)
    print(
        f"GSM8K accuracy ({len(gsm)} problems): {accuracy:.1f}% "
        f"95% bootstrap CI {confidence_interval}",
        flush=True,
    )

    write_csv(os.path.join(run_dir, "samples.csv"), rows[:20])
    evaluation = {
        "val_ppl": round(ppl, 4),
        "gsm8k_acc": round(accuracy, 2),
        "gsm8k_ci95": confidence_interval,
        "n_gsm": len(gsm),
        "n_val": len(val_rows),
        "max_new_tokens": args.max_new,
        "base_precision": precision,
        "base_only": args.base_only,
        "base_model_id": model_id,
        "base_model_revision": revision,
    }
    with open(os.path.join(run_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)

    del model
    cleanup()
