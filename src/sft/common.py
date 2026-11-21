"""Shared model configuration and deterministic query-grouped splits."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/wu_ci_work_sft_reviewed.jsonl"
# Repository ID, pinned revision, per-device batch size and accumulation steps.
MODELS = {
    "4b": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554", 4, 2),
    "8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", 2, 4),
}


def experiment_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def dataset():
    raw = DATA.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines()]
    return rows, hashlib.sha256(raw).hexdigest()


def group_folds(rows, seed):
    groups = defaultdict(list)
    tasks, decisions = {}, set()
    for row in rows:
        qid = row["query_id"]
        text = row["prompt"][-1]["content"]
        task = " ".join(text.rsplit("\ntool: ", 1)[0].split()).casefold()
        if tasks.setdefault(qid, task) != task or text in decisions:
            raise ValueError("Repeated decision or inconsistent query ID")
        if row["completion"][0]["content"] not in ("Yes", "No"):
            raise ValueError("Invalid permission label")
        decisions.add(text)
        groups[qid].append(row)
    if len(tasks) != len(set(tasks.values())):
        raise ValueError("Duplicate task text across query IDs")
    rng = random.Random(seed)
    folds = [set() for _ in range(5)]
    # Wu identifiers are numeric. CI-Work uses UUIDs.
    for wu in (True, False):
        ids = sorted(qid for qid in groups if qid.isdigit() == wu)
        rng.shuffle(ids)
        ids.sort(key=lambda qid: len(groups[qid]))
        for start in range(0, len(ids), 5):
            order = list(range(5))
            rng.shuffle(order)
            for qid, index in zip(ids[start:start + 5], order):
                folds[index].add(qid)
    if not all(folds) or set.union(*folds) != set(groups) or sum(map(len, folds)) != len(groups):
        raise ValueError("Invalid grouped five-fold partition")
    return folds


def partition(args):
    rows, digest = dataset()
    held = group_folds(rows, args.seed)[args.fold - 1]
    train = [row for row in rows if row["query_id"] not in held]
    test = [row for row in rows if row["query_id"] in held]
    model_id, revision, _, _ = MODELS[args.model]
    info = dict(model_id=model_id, model_revision=revision, seed=args.seed, fold=args.fold,
                dataset_sha256=digest, test_query_ids=sorted(held))
    return train, test, info


def load_model(size):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Use a BF16-capable CUDA GPU")
    model_id, revision, _, _ = MODELS[size]
    location, options = model_id, {"revision": revision}
    if "SFT_MODELS_DIR" in os.environ:
        location = ROOT / os.environ["SFT_MODELS_DIR"] / model_id.split("/")[1]
        if (location / "REVISION").read_text().strip() != revision:
            raise ValueError(f"Unexpected model revision for {model_id}")
        options = {"local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(location, **options)
    model = AutoModelForCausalLM.from_pretrained(location, **options, dtype=torch.bfloat16,
                                                attn_implementation="sdpa", device_map={"": 0})
    return model, tokenizer


def permission_prompt(tokenizer, messages, **options):
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         enable_thinking=False, **options)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def new_run_dir(relative):
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("runs",):
        raise ValueError("Use a new project-relative output path under runs/")
    out = ROOT / relative
    out.mkdir(parents=True, exist_ok=False)
    return out
