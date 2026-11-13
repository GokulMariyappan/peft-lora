"""Variant registry and trainable-parameter audit for Module 2."""
import argparse
import json
import os

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from common import MODELS, TARGETS, RESULTS, load_base, get_revision, cleanup


class UnknownMethodError(ValueError):
    pass


# Interface Contract v1.0: the variant flag is the only method-level change.
BASE_KW = dict(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=TARGETS,
    bias="none",
    task_type="CAUSAL_LM",
)
VARIANTS = {
    "lora": {},
    "dora": {"use_dora": True},
    "rslora": {"use_rslora": True},
    "pissa": {"init_lora_weights": "pissa_niter_4"},
}


def get_config(name, **overrides):
    key = name.lower()
    if key not in VARIANTS:
        raise UnknownMethodError(f"'{name}' not in registry {list(VARIANTS)}")
    return LoraConfig(**{**BASE_KW, **VARIANTS[key], **overrides})


def needs_weight_conversion(name):
    return name.lower() == "pissa"


def count_params(model):
    """Return trainable count, approximate total count, and trainable percentage."""
    trainable = total = 0
    for param in model.parameters():
        # bitsandbytes stores 4-bit values packed; x2 is the project's stated estimate.
        multiplier = 2 if param.__class__.__name__ == "Params4bit" else 1
        total += param.numel() * multiplier
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total, (100.0 * trainable / total if total else 0.0)


def _prep(base, precision):
    if precision == "4bit":
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    else:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    if hasattr(base, "config"):
        base.config.use_cache = False
    return base


def _is_quantized_pissa_compatibility_error(exc):
    """Do not hide OOM or unrelated programming errors behind the FP16 fallback."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return False
    message = str(exc).lower()
    if "out of memory" in message or "cuda error: out of memory" in message:
        return False
    markers = (
        "4bit", "4-bit", "quantiz", "params4bit", "quantized", "not supported",
        "unsupported", "cannot access data pointer", "data pointer",
    )
    return any(marker in message for marker in markers)


def build_peft_model(model_id, revision, method, precision="4bit"):
    """Load base + adapter. PiSSA uses FP16 fallback only for quantization incompatibility."""
    method = method.lower()
    if method not in VARIANTS:
        raise UnknownMethodError(f"'{method}' not in registry {list(VARIANTS)}")

    note = ""
    try:
        base = load_base(model_id, revision, precision)
        model = get_peft_model(_prep(base, precision), get_config(method))
    except Exception as exc:
        if not (precision == "4bit" and method == "pissa" and _is_quantized_pissa_compatibility_error(exc)):
            raise
        cleanup()
        precision = "fp16"
        note = (
            f"PiSSA 4-bit initialization was incompatible ({type(exc).__name__}: "
            f"{str(exc)[:180]}); retried with an FP16 base. Memory is not comparable "
            "with 4-bit runs."
        )
        base = load_base(model_id, revision, precision)
        model = get_peft_model(_prep(base, precision), get_config(method))

    # Keep trainable adapter weights in FP32 while the frozen base uses FP16 compute.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return model, precision, note


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-0.6b", choices=list(MODELS))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found. In Colab select Runtime > Change runtime type > T4 GPU.")

    model_id = MODELS[args.model]
    revision = get_revision(model_id)
    rows = []
    for method in VARIANTS:
        model, precision, note = build_peft_model(model_id, revision, method)
        trainable, total, pct = count_params(model)
        row = {
            "method": method,
            "trainable": trainable,
            "total": total,
            "pct": round(pct, 3),
            "base_precision": precision,
            "note": note,
        }
        rows.append(row)
        print(
            f"{method:7s} trainable={trainable:>12,} total={total:>14,} "
            f"({pct:.3f}%) base={precision} {note}"
        )
        del model
        cleanup()

    os.makedirs(RESULTS, exist_ok=True)
    with open(f"{RESULTS}/param_counts_{args.model}.json", "w", encoding="utf-8") as f:
        json.dump({"model_id": model_id, "revision": revision, "variants": rows}, f, indent=2)
