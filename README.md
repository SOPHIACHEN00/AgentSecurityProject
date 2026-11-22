# AgentSecurityProject

This repository trains Qwen to decide whether a tool should have default access to a data attribute for a given task. Inputs follow the order `query`, `tool`, `data`. The target is exactly `Yes` or `No`. A `Yes` label represents task-level permission. Individual consent is a separate decision.

## Dataset

`data/processed/wu_ci_work_sft_reviewed.jsonl` contains 1,444 reviewed decisions from 169 queries, with 871 Yes and 573 No labels. Each line contains a chat prompt, an assistant completion and a metadata field called `query_id`. The model receives the prompt. The split algorithm uses the ID to keep all decisions for a query together.

The reviewed data adapts the tool-specific permission lists in [Wu et al.'s dataset](https://github.com/llm-platform-security/ai-agent-permissions/tree/ec777f9d81d662c0df8b75c1aba9e381680f1e77) and the task and data attributes in [Microsoft CI-Work](https://github.com/microsoft/CI-Work/tree/50e6b18a606467e4ec3cac4a57e0c6cca8eb2752). The release converts these sources to reviewed binary permission decisions and formats them for chat SFT. The respective upstream notices are in [licenses/Wu.txt](licenses/Wu.txt) and [licenses/CI-Work.txt](licenses/CI-Work.txt).

## Setup

Use Linux with a BF16-capable CUDA GPU. The pinned environment targets a single RTX 5090 with CUDA 12.8 PyTorch wheels and Python 3.13. Run commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The first model load downloads a pinned revision from Hugging Face. For offline use, set `SFT_MODELS_DIR` to a folder containing the model snapshots, named `Qwen3-4B-Instruct-2507` and `Qwen3-8B`. Each snapshot needs its model and tokenizer files and a text file named `REVISION` with the commit pinned in `src/sft/common.py`.

## Train and evaluate one fold

The commands use seed 42 and fold 1 by default. Four folds supply training data and the fifth supplies held-out evaluation data. Output directories must be new. To run Qwen3-8B, use `--model 8b` throughout and choose new output directories.

```bash
python src/sft/evaluate.py --model 4b --output-dir runs/4b-base
python src/sft/train.py --model 4b --output-dir runs/4b-train
python src/sft/evaluate.py --model 4b --adapter runs/4b-train/adapter --output-dir runs/4b-sft
```

Training uses two epochs, BF16, an effective batch size of eight, and LoRA on the attention projections. Qwen3-4B-Instruct-2507 uses four examples per device step with two accumulation steps. Qwen3-8B uses two examples with four accumulation steps. The learning rate peaks at 0.00005 after a 5% warmup and decays linearly. Only the answer and end-of-turn token contribute to the loss. Training logs loss at each optimizer update and saves the final adapter. Evaluation uses greedy decoding, with thinking disabled for Qwen3-8B.

## Repeated five-fold cross-validation

```bash
# Inspect fold assignments without loading weights.
python src/sft/cross_validate.py --prepare-only --output-dir runs/fold-plan

# Smoke test both sizes on the first fold of seed 42, five updates each.
python src/sft/cross_validate.py --smoke --output-dir runs/smoke

# Run all 30 fits and paired base/adapter evaluations.
python src/sft/cross_validate.py --output-dir runs/cv
```

Each of the seeds 42, 43 and 44 defines five folds grouped by query ID and balanced by data source and group size. Both model sizes use the same folds. Each run starts from pinned base weights. The held-out fold is used after the fixed training schedule. For each repeat, every example contributes one held-out prediction per model and condition.

Each evaluation writes `predictions.jsonl` and `metrics.json`. The cross-validation runner pools predictions within each repeat and writes the mean and sample standard deviation across repeats to `summary.json`. Reported metrics include accuracy, Macro-F1, FP (gold No, predicted Yes), FN (gold Yes, predicted No), FPR and FNR. Other outputs count as incorrect and are also reported separately. A smoke summary covers only its single held-out fold.

### Qwen3.5-0.8B and Qwen3.5-2B

These checkpoints use a newer Transformers runtime. The RTX 5090 experiment used PyTorch 2.11.0+cu128, Transformers 5.4.0, PEFT 0.19.1, Accelerate 1.12.0, huggingface-hub 1.33.0, flash-linear-attention and fla-core 0.4.2. Download the revisions listed in `src/sft/qwen35_cv.py` to `models/Qwen3.5-0.8B` and `models/Qwen3.5-2B`, and write the matching commit hash in each snapshot's `REVISION` file.

```bash
python src/sft/qwen35_cv.py --run runs/qwen35-3x5
python src/sft/report_qwen35_cv.py --run runs/qwen35-3x5
```

The runner uses the same query partitions and evaluates each original checkpoint once on all rows. Each fold starts from the original weights, updates the language model's full-attention Q/K/V/O adapters for two epochs with batches of eight, and decodes held-out decisions greedily with thinking disabled. It saves each update's loss, learning rate and gradient norm in `steps.jsonl`. Reporting uses NumPy and Matplotlib, checks the predictions against the original examples, and writes per-repeat scores, a summary, and English training/validation figures.
