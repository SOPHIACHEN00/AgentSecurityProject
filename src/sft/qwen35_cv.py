"""Repeated query-grouped CV for the text-only permission task with Qwen3.5."""
import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForImageTextToText, AutoTokenizer,
                          get_linear_schedule_with_warmup, set_seed)

from common import ROOT, dataset, group_folds, permission_prompt, write_json
from evaluate import permission_metrics, predict

REVISIONS = {"0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
             "2B": "15852e8c16360a2fea060d615a32b45270f8a8fc"}
DATA_SHA = "f28d5f72d47e8ab27fc5780b0be95b08e2f870cb384b7ed6f2d7175795dc1a94"
BATCH = 8


def load(size):
    folder = ROOT / "models" / f"Qwen3.5-{size}"
    if (folder / "REVISION").read_text().strip() != REVISIONS[size]:
        raise ValueError("Unexpected model revision")
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model, info = AutoModelForImageTextToText.from_pretrained(
        folder, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"]:
        raise ValueError(f"Incomplete checkpoint: {info}")
    return model, tokenizer


def encode(tokenizer, rows):
    examples = []
    for row in rows:
        prefix = permission_prompt(tokenizer, row["prompt"], return_dict=False)
        answer = row["completion"][0]["content"]
        token_ids = tokenizer.encode(answer, add_special_tokens=False)
        if len(token_ids) != 1 or tokenizer.decode(token_ids) != answer:
            raise ValueError("Expected a single Yes/No token")
        sequence = prefix + token_ids + [tokenizer.eos_token_id]
        if len(sequence) > 512:
            raise ValueError("Example exceeds the original context limit")
        examples.append(torch.tensor(sequence, dtype=torch.long))
    return examples


def collate(examples, pad_id):
    ids = torch.nn.utils.rnn.pad_sequence(examples, batch_first=True,
                                          padding_value=pad_id, padding_side="left").cuda()
    lengths = torch.tensor([len(x) for x in examples], device=ids.device)
    mask = torch.arange(ids.shape[1], device=ids.device) >= ids.shape[1] - lengths[:, None]
    return ids, mask.long()


def answer_loss(model, batch):
    ids, mask = batch
    # Left padding aligns the two supervised positions. Earlier logits never enter the loss.
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=3).logits
    return torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                             ids[:, -2:].reshape(-1))


@torch.inference_mode()
def split_loss(model, batches):
    total = sum(answer_loss(model, b).item() * len(b[0]) for b in batches)
    return total / sum(len(b[0]) for b in batches)


def fit(model, tok, train, held, seed, size, out):
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                        target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)")
    model = get_peft_model(model, config)
    model.peft_config["default"].base_model_name_or_path = f"Qwen/Qwen3.5-{size}"
    model.peft_config["default"].revision = REVISIONS[size]
    examples = encode(tok, train)
    validation = encode(tok, held)
    def eval_batches(rows):
        ordered = sorted(rows, key=len)
        return [collate(ordered[i:i+16], tok.pad_token_id) for i in range(0, len(ordered), 16)]
    train_batches, val_batches = eval_batches(examples), eval_batches(validation)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=0.0)
    steps = math.ceil(len(train) / BATCH) * 2
    sched = get_linear_schedule_with_warmup(opt, math.ceil(0.05 * steps), steps)
    curves = []
    def measure(step):
        model.eval()
        curves.append(dict(step=step, training=split_loss(model, train_batches),
                           validation=split_loss(model, val_batches)))
        write_json(out / "loss.json", curves)
        print(step, curves[-1], flush=True)
        model.train()
    measure(0)
    step = 0
    with (out / "steps.jsonl").open("w") as log:
        for epoch in range(2):
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
            for i in range(0, len(order), BATCH):
                batch = collate([examples[j] for j in order[i:i+BATCH]], tok.pad_token_id)
                loss = answer_loss(model, batch)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                lr = opt.param_groups[0]["lr"]
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                print(json.dumps(dict(step=step, loss=loss.item(), grad_norm=norm.item(),
                                      learning_rate=lr), allow_nan=False), file=log, flush=True)
                if step % 32 == 0 or step == steps:
                    measure(step)
    if step != steps or not any(p.detach().count_nonzero().item() for n,p in model.named_parameters()
                                if "lora_B." in n):
        raise RuntimeError("Incomplete fit")
    model.save_pretrained(out / "adapter", save_embedding_layers=False)
    return model


def job(size, rows, held_ids, seed, run, destination):
    if (destination / "metrics.json").exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print("START", destination.relative_to(run), flush=True)
    set_seed(seed)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        out = Path(tmp) / "result"
        out.mkdir()
        model, tokenizer = load(size)
        held = [r for r in rows if r["query_id"] in held_ids] if held_ids else rows
        if held_ids:
            train = [r for r in rows if r["query_id"] not in held_ids]
            model = fit(model, tokenizer, train, held, seed, size, out)
        model.eval()
        results = predict(model, tokenizer, held)
        with (out / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in results:
                print(json.dumps(row, ensure_ascii=False), file=stream)
        metrics = permission_metrics(results)
        write_json(out / "metrics.json", dict(**metrics, revision=REVISIONS[size], seed=seed))
        out.rename(destination)
        print("DONE", destination.relative_to(run), metrics, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference-plan", type=Path)
    parser.add_argument("--models", choices=REVISIONS, nargs="+", default=list(REVISIONS))
    args = parser.parse_args()
    if args.run.is_absolute() or args.run.parts[:1] != ("runs",) or ".." in args.run.parts:
        parser.error("Use a project-relative run directory under runs/")
    rows, digest = dataset()
    if digest != DATA_SHA or len(rows) != 1444:
        raise ValueError("Dataset differs from the prior CV")
    plan = {s: [sorted(ids) for ids in group_folds(rows, s)] for s in (42, 43, 44)}
    if args.reference_plan:
        original = json.loads((ROOT / args.reference_plan).read_bytes())
        if digest != original["source_sha256"] or len(original["folds"]) != 15 or any(
                plan[f["seed"]][f["fold"]-1] != f["test_query_ids"] for f in original["folds"]):
            raise ValueError("Held-out assignments differ from prior models")
    run = ROOT / args.run
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "folds.json", plan)
    for size in args.models:
        job(size, rows, None, 42, run, run / size / "base")
    for seed, folds in plan.items():
        for fold, ids in enumerate(folds, 1):
            for size in args.models:
                job(size, rows, set(ids), seed, run, run / size / f"seed-{seed}" / f"fold-{fold}")


if __name__ == "__main__":
    main()
