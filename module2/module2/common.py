"""Shared helpers for Module-2 (follows Interface Contract v1.0)."""
import csv, gc, hashlib, json, os, random, re
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELS = {"qwen3-0.6b": "Qwen/Qwen3-0.6B-Base", "gemma3-1b": "google/gemma-3-1b-pt"}
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_LEN = 512
PROMPT = ("Below is a math problem. Solve it step by step and end with "
          "'The answer is: <number>'.\n\n### Question:\n{instruction}\n\n### Answer:\n")
RUNS, DATA, RESULTS = "runs", "data", "results"


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def fmt_lr(lr):  # 2e-4 -> "2e-4" (contract run_id format)
    m, e = f"{lr:.0e}".split("e")
    return f"{m}e{int(e)}"


def short_name(model_id):
    return next(k for k, v in MODELS.items() if v == model_id)


def cleanup():
    gc.collect(); torch.cuda.empty_cache()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_revision(model_id):
    """Resolve and return the current Hub commit hash; warn if offline/unavailable."""
    try:
        from huggingface_hub import model_info
        revision = model_info(model_id).sha
        if not revision:
            raise RuntimeError("Hugging Face did not return a commit hash")
        return revision
    except Exception as exc:
        print(f"WARNING: could not resolve an immutable revision for {model_id}: {exc}")
        print("WARNING: this run will use the model repository's current default revision.")
        return None


def make_prompt(q):
    return PROMPT.format(instruction=q)


def load_tokenizer(model_id, revision=None):
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # only affects tok(...) calls used for generation
    return tok


def load_base(model_id, revision=None, precision="4bit"):
    kw = dict(revision=revision, device_map={"": 0})
    if precision == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    else:  # "fp16" (used for PiSSA fallback and for merging)
        kw["torch_dtype"] = torch.float16
    return AutoModelForCausalLM.from_pretrained(model_id, **kw)


# ---------------------------------------------------------------- data
def encode(tok, q, a, max_len=MAX_LEN):
    """Returns (input_ids, labels) with prompt labels = -100, or None if too long."""
    p = tok(make_prompt(q))["input_ids"]  # adds BOS for Gemma, nothing for Qwen
    o = tok(a + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(p) + len(o) > max_len:
        return None
    return p + o, [-100] * len(p) + o


def get_splits(tok, short, train_n=5000, val_n=500, seed=42, pool=20000):
    """Create a deterministic, length-filtered MetaMathQA split in contract JSONL format."""
    d = os.path.join(DATA, short)
    tp, vp = os.path.join(d, "train.jsonl"), os.path.join(d, "val.jsonl")

    def read(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    expected = train_n + val_n
    cached_ok = False
    if os.path.exists(tp) and os.path.exists(vp):
        try:
            cached_ok = len(read(tp)) == train_n and len(read(vp)) == val_n
        except (OSError, json.JSONDecodeError):
            cached_ok = False

    if not cached_ok:
        os.makedirs(d, exist_ok=True)
        source = load_dataset("meta-math/MetaMathQA", split="train")
        source = source.shuffle(seed=seed)
        source = source.select(range(min(pool, len(source))))
        seen, rows = set(), []
        for i, r in enumerate(source):
            q, a = str(r.get("query", "")).strip(), str(r.get("response", "")).strip()
            if not q or not a or q in seen:
                continue
            seen.add(q)
            if encode(tok, q, a) is None:
                continue
            rows.append({"id": f"mm{i}", "instruction": q, "input": "", "output": a})
            if len(rows) == expected:
                break
        if len(rows) != expected:
            raise RuntimeError(
                f"Only collected {len(rows)} usable examples from the first {min(pool, len(source))} "
                f"MetaMathQA rows; need {expected}. Increase pool in get_splits()."
            )
        for path, part in ((tp, rows[:train_n]), (vp, rows[train_n:])):
            with open(path, "w", encoding="utf-8") as f:
                for row in part:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_rows, val_rows = read(tp), read(vp)
    if len(train_rows) != train_n or len(val_rows) != val_n:
        raise RuntimeError(f"Invalid cached split sizes: train={len(train_rows)}, val={len(val_rows)}")
    return {"train": train_rows, "val": val_rows}


def encode_rows(tok, rows):
    out = []
    for r in rows:
        e = encode(tok, r["instruction"], r["output"])
        if e:
            out.append({"input_ids": e[0], "labels": e[1]})
    return out


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id); lab = torch.full((len(batch), L), -100)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"]); lab[i, :n] = torch.tensor(b["labels"]); att[i, :n] = 1
    return {"input_ids": ids, "attention_mask": att, "labels": lab}


# ---------------------------------------------------------------- GSM8K
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _to_float(s):
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def extract_answer(text):
    """Text after 'The answer is:' (first number), else last number in output."""
    text = text.split("###")[0]  # cut off hallucinated next question
    if "The answer is:" in text:
        nums = _NUM.findall(text.split("The answer is:")[-1])
        return _to_float(nums[0]) if nums else None
    nums = _NUM.findall(text)
    return _to_float(nums[-1]) if nums else None


def load_gsm8k(n):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(n, len(ds))))
    return [(r["question"], _to_float(r["answer"].split("####")[-1].strip())) for r in ds]


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=256, bs=16):
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
