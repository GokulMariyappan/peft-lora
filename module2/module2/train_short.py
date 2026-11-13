"""Run short, controlled adapter smoke tests and save reproducible run artifacts."""
import argparse
import json
import os
import statistics
import time
import traceback

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from common import (
    MODELS, TARGETS, RUNS, MAX_LEN, seed_everything, fmt_lr, cleanup,
    sha256_file, get_revision, load_tokenizer, get_splits, encode_rows, collate,
)
from registry import build_peft_model, count_params, needs_weight_conversion


def _adapter_weights_path(adapter_dir):
    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        path = os.path.join(adapter_dir, filename)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def run_one(args, method):
    method = method.lower()
    seed_everything(args.seed)
    model_id = MODELS[args.model]
    revision = get_revision(model_id)
    run_id = f"{args.model}-{method}-fixed-lr{fmt_lr(args.lr)}-s{args.seed}"
    run_dir = os.path.join(RUNS, run_id)
    os.makedirs(run_dir, exist_ok=True)

    tokenizer = load_tokenizer(model_id, revision)
    splits = get_splits(tokenizer, args.model)
    train_rows = encode_rows(tokenizer, splits["train"])
    if len(train_rows) != len(splits["train"]):
        raise RuntimeError(
            f"Tokenization retained {len(train_rows)}/{len(splits['train'])} training rows. "
            "The cached dataset and tokenizer may not match."
        )
    required_examples = args.steps * args.bs * args.accum
    if required_examples > len(train_rows):
        raise ValueError(
            f"Requested {required_examples} examples ({args.steps} steps x "
            f"{args.bs} batch x {args.accum} accumulation), but only "
            f"{len(train_rows)} training examples are available."
        )

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_rows,
        batch_size=args.bs,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
        drop_last=False,
    )

    cleanup()
    torch.cuda.reset_peak_memory_stats()
    model, precision, note = build_peft_model(model_id, revision, method)
    trainable, total, pct = count_params(model)

    init_dir = os.path.join(run_dir, "adapter_init")
    if needs_weight_conversion(method):
        model.save_pretrained(init_dir, safe_serialization=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.03 * args.steps)),
        num_training_steps=args.steps,
    )
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    iterator = iter(loader)
    step_times, step_losses = [], []
    optimizer.zero_grad(set_to_none=True)

    for step in range(args.steps):
        torch.cuda.synchronize()
        start = time.time()
        accumulated_loss = 0.0
        for _ in range(args.accum):
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError("Training DataLoader exhausted before requested steps completed") from exc
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / args.accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at optimizer step {step + 1}")
            scaler.scale(loss).backward()
            accumulated_loss += float(loss.detach().item())

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at optimizer step {step + 1}")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        step_times.append(time.time() - start)
        step_losses.append(accumulated_loss)
        if step % 10 == 0 or step == args.steps - 1:
            print(
                f"[{run_id}] step {step + 1}/{args.steps} "
                f"loss={accumulated_loss:.4f} time={step_times[-1]:.2f}s",
                flush=True,
            )

    peak_memory = torch.cuda.max_memory_allocated() / (2 ** 30)
    warmup_count = min(5, max(0, len(step_times) - 1))
    measured_times = step_times[warmup_count:] or step_times
    metrics = {
        "method": method,
        "steps": args.steps,
        "trainable_params": trainable,
        "trainable_pct": round(pct, 3),
        "loss_first10_mean": sum(step_losses[:min(10, len(step_losses))]) / min(10, len(step_losses)),
        "loss_last10_mean": sum(step_losses[-min(10, len(step_losses)):]) / min(10, len(step_losses)),
        "any_nan": not all(torch.isfinite(torch.tensor(value)).item() for value in step_losses),
        "peak_mem_gib": round(peak_memory, 3),
        "sec_per_step_median": round(statistics.median(measured_times), 3),
        "effective_batch": args.bs * args.accum,
        "examples_seen": required_examples,
        "base_precision": precision,
        "note": note,
    }

    adapter_dir = os.path.join(run_dir, "adapter")
    if needs_weight_conversion(method):
        model.save_pretrained(
            adapter_dir,
            safe_serialization=True,
            path_initial_model_for_weight_conversion=init_dir,
        )
        converted = True
    else:
        model.save_pretrained(adapter_dir, safe_serialization=True)
        converted = False

    config = {
        "run_id": run_id,
        "model": {"id": model_id, "revision": revision},
        "method": method,
        "strategy": "fixed",
        "lr": args.lr,
        "seed": args.seed,
        "data": {
            "dataset": "meta-math/MetaMathQA",
            "train_n": len(splits["train"]),
            "val_n": len(splits["val"]),
            "max_len": MAX_LEN,
        },
        "quant": {"type": "nf4", "double": True, "compute_dtype": "float16"},
        "lora": {"r": 16, "alpha": 16, "dropout": 0.05, "targets": TARGETS},
        "train": {
            "mode": "short_smoke_test",
            "per_device_bs": args.bs,
            "grad_accum": args.accum,
            "effective_batch": args.bs * args.accum,
            "steps": args.steps,
            "examples_seen": required_examples,
            "fraction_of_train_set_seen": round(required_examples / len(train_rows), 4),
        },
        "tracker": {"backend": "none", "project": "module2-smoke"},
        "base_precision": precision,
    }
    with open(os.path.join(run_dir, "run_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    weights_path = _adapter_weights_path(adapter_dir)
    manifest = {
        "adapter_sha256": sha256_file(weights_path),
        "adapter_weights_file": os.path.basename(weights_path),
        "base_model_revision": revision,
        "lr": args.lr,
        "seed": args.seed,
        "strategy": "fixed",
        "runs_used": 1,
        "method": method,
        "pissa_converted": converted,
        "base_precision": precision,
        "torch_version": torch.__version__,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(metrics, indent=2))
    del model, optimizer, scheduler, scaler
    cleanup()
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-0.6b", choices=list(MODELS))
    parser.add_argument("--methods", nargs="+", default=["lora", "dora", "rslora", "pissa"], choices=list(["lora", "dora", "rslora", "pissa"]))
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found. In Colab select Runtime > Change runtime type > T4 GPU.")
    if args.steps < 1 or args.bs < 1 or args.accum < 1 or args.lr <= 0:
        raise SystemExit("steps, bs, accum and lr must be positive")

    failures = []
    for method in args.methods:
        try:
            run_one(args, method)
        except Exception:
            run_id = f"{args.model}-{method}-fixed-lr{fmt_lr(args.lr)}-s{args.seed}"
            run_dir = os.path.join(RUNS, run_id)
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "FAILED.txt"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            failures.append(method)
            print(f"!! {method} FAILED; see {run_dir}/FAILED.txt", flush=True)
            cleanup()

    if failures:
        raise SystemExit(f"Training failed for: {', '.join(failures)}")
