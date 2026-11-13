# Module 2 v1 — Group 2: registry, short runs, evaluation and merge

This project implements the Group 2 workflow for LoRA, DoRA, rsLoRA and PiSSA on a free Google Colab T4 GPU. Start with `qwen3-0.6b` (`Qwen/Qwen3-0.6B-Base`). The 60-step configuration is a **smoke test**, not a full fine-tuning run and not enough evidence to rank methods.

## 1. Colab setup

Select **Runtime → Change runtime type → T4 GPU**. Then run:

```python
!nvidia-smi
!pip -q install "transformers>=4.51" "peft>=0.14" "bitsandbytes>=0.45" datasets accelerate pyyaml
```

Upload the `module2/` folder (or clone the repository), then:

```python
%cd /content/module2
!python -m py_compile common.py registry.py train_short.py evaluate.py merge.py make_table.py verify_outputs.py
```

After the first successful run, record the actual environment so the experiment can be reproduced:

```python
!pip freeze > requirements-lock.txt
```

## 2. Run the pipeline

```python
# A. Attach each adapter configuration and report trainable parameters.
!python registry.py --model qwen3-0.6b

# B. Train all four methods: 60 optimizer steps, effective batch 16, seed 42.
!python train_short.py --model qwen3-0.6b --methods lora dora rslora pissa --steps 60 --lr 2e-4 --seed 42

# C. Evaluate the untouched base model.
!python evaluate.py --base_only --model qwen3-0.6b

# D. Evaluate each completed adapter run. Failed/incomplete folders are skipped.
!for d in runs/qwen3-0.6b-*; do if [ -f "$d/run_config.yaml" ] && [ -f "$d/adapter/adapter_config.json" ]; then python evaluate.py --run_dir "$d"; fi; done

# E. Merge each adapter into an FP16 base and verify parity.
!for d in runs/qwen3-0.6b-*; do if [ -f "$d/run_config.yaml" ] && [ -f "$d/adapter/adapter_config.json" ]; then python merge.py --run_dir "$d" || echo "Merge flagged/failed for $d"; fi; done

# F. Build the comparison table and audit the artifacts.
!python make_table.py
!python verify_outputs.py --model qwen3-0.6b --seed 42 --lr 2e-4
```

If training runs out of GPU memory, restart the runtime and use a smaller micro-batch while preserving effective batch 16:

```python
!python train_short.py --model qwen3-0.6b --methods lora dora rslora pissa --steps 60 --lr 2e-4 --seed 42 --bs 2 --accum 8
```

Do not rerun training over an existing completed experiment unless you intend to replace it. Copy or download the existing `runs/`, `data/` and `results/` folders first.

## 3. Interface defaults

- Base model: `Qwen/Qwen3-0.6B-Base`.
- LoRA rank `r=16`, alpha `16`, dropout `0.05`.
- Targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- Quantization: NF4, double quantization, FP16 compute.
- Dataset: `meta-math/MetaMathQA`, 5,000 train / 500 validation rows, max sequence length 512, deterministic seed 42.
- Smoke test: 60 optimizer steps, per-device batch 4, accumulation 4 (effective batch 16), learning rate `2e-4`.
- Evaluation: 200 validation examples for answer-token perplexity and first 100 GSM8K test questions for greedy accuracy (configurable in `evaluate.py`).

The data split builder filters over-length rows and checks that the requested 5,000/500 examples were actually collected. It does not silently write undersized splits.

## 4. Expected artifacts

```text
module2/
├── common.py
├── registry.py
├── train_short.py
├── evaluate.py
├── merge.py
├── make_table.py
├── verify_outputs.py
├── data/qwen3-0.6b/{train,val}.jsonl
├── runs/
│   ├── base-qwen3-0.6b/{eval.json,samples.csv}
│   └── qwen3-0.6b-{lora,dora,rslora,pissa}-fixed-lr2e-4-s42/
│       ├── adapter/
│       ├── adapter_init/              # PiSSA only
│       ├── run_config.yaml
│       ├── metrics.json
│       ├── manifest.json
│       ├── eval.json
│       ├── samples.csv
│       └── merged/                    # only when parity passes
│           ├── model.safetensors      # actual sharded/unsharded filename may vary
│           ├── tokenizer files
│           └── merge_meta.json
└── results/
    ├── param_counts_qwen3-0.6b.json
    ├── week5_results.md
    └── week5_results.csv
```

The registry JSON stores the resolved model revision and a `variants` array with trainable/total parameter counts. The expected architecture counts in the design document (LoRA/rsLoRA/PiSSA 10,092,544; DoRA 10,436,608) are reference values to verify, not hard-coded outputs. A mismatch must be investigated against the loaded model architecture and PEFT version.

## 5. Important interpretation and limitations

- **Smoke test only:** 60 optimizer steps at effective batch 16 consume 960 examples per method (19.2% of 5,000). Do not describe this as one epoch or as a final performance ranking.
- **Seed:** the default is 42. All methods reset the same seed and use the same shuffled data order. This is one seed; proper comparisons need multiple seeds and longer runs.
- **rsLoRA:** with alpha 16 and rank 16, its scale is `alpha/sqrt(r)=4`, compared with standard LoRA's `alpha/r=1`. The same learning rate is therefore not necessarily an equally calibrated comparison.
- **PiSSA fallback:** only quantization-compatibility failures trigger the FP16 fallback. Out-of-memory and unrelated errors are not hidden. An FP16 PiSSA run has different memory use and must not be compared directly with 4-bit runs.
- **Evaluation base revision:** the base-only evaluation resolves a Hub revision; adapter runs record and reuse their resolved revision. If revision resolution fails, the scripts warn that the mutable default revision is being used.
- **Merge parity:** merging is checked against the unmerged adapter on an FP16 base, not against the 4-bit training model. It checks logits within tolerance, next-token argmax and generated text on three GSM8K prompts. A failed check writes `MERGE_FLAGGED.txt` and withholds the merged export.
- **Configuration contract:** `train.mode`, `train.steps`, `train.examples_seen`, `train.fraction_of_train_set_seen` and `base_precision` are implementation metadata beyond a minimal contract. Group 1 should approve these additions before claiming strict Interface Contract v1.0 compliance.
- **No fake outputs:** evaluation and merge artifacts are written only after those stages execute. The results table marks incomplete or failed runs rather than inventing metrics.

## 6. Files at a glance

- `common.py`: model loading, prompt/tokenization, deterministic dataset creation, batching, GSM8K and file helpers.
- `registry.py`: shared LoRA configuration, method flags, PiSSA fallback and parameter counts.
- `train_short.py`: isolated short training runs, metrics, adapter serialization and manifests.
- `evaluate.py`: validation perplexity, GSM8K accuracy/CI and 20 saved generations.
- `merge.py`: FP16 merge, numerical/output parity check and export gating.
- `make_table.py`: Markdown/CSV results aggregation.
- `verify_outputs.py`: offline artifact audit and adapter SHA256 verification.
