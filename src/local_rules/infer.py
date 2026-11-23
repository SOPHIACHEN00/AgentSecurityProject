"""Run frozen BF16 Qwen weights on the shared, ranked 32-demonstration prompts."""
import argparse
from collections import defaultdict
import inspect
import json
from pathlib import Path
import time

import torch
import transformers
from transformers import AutoTokenizer, GenerationConfig
from transformers.generation import GenerationMixin

from prepare import MODELS, ROOT, read_jsonl, run_directory, source, verified, write_json


def load(name):
    model_dir = ROOT / "models" / name
    if (model_dir / "REVISION").read_text().strip() != MODELS[name]:
        raise ValueError(f"Unexpected model revision for {name}")
    if "5090" not in torch.cuda.get_device_name(0) or not torch.cuda.is_bf16_supported():
        raise ValueError("Expected the BF16-capable RTX 5090")
    if name.startswith("Qwen3.5-"):
        from transformers import AutoModelForImageTextToText as loader
    else:
        from transformers import AutoModelForCausalLM as loader
    model, info = loader.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16,
                                          device_map={"": 0}, attn_implementation="sdpa", output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"]:
        raise ValueError(f"Incomplete weights for {name}")
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="left")
    return model.eval(), tok


def messages_for(record, rows):
    target = rows[record["row"]-1]
    messages = [target["prompt"][0]]
    for case in record["cases"]:
        example = rows[case-1]
        if example["query_id"] == target["query_id"]:
            raise ValueError("Test query entered its own context")
        messages.extend((example["prompt"][1], example["completion"][0]))
    messages.append(target["prompt"][1])
    if len(messages) != 66 or len(set(record["cases"])) != 32:
        raise ValueError("Expected 32 complete user/assistant pairs and the target")
    return messages


def tokenize(tok, rows, records, limit):
    options = dict(add_generation_prompt=True, enable_thinking=False)
    strings = []
    for record in records:
        messages = messages_for(record, rows)
        rendered = tok.apply_chat_template(messages, tokenize=False, **options)
        if rendered.count("<|im_start|>user\n") != 33 or rendered.count("<|im_start|>assistant\n") != 33:
            raise ValueError("Chat template omitted a turn")
        strings.append(rendered)
    sequences = tok(strings, add_special_tokens=False, padding=False, truncation=False,
                    return_attention_mask=False)["input_ids"]
    longest = max(range(len(sequences)), key=lambda i: len(sequences[i]))
    ref = tok.apply_chat_template(messages_for(records[longest], rows), tokenize=True,
                                  return_dict=False, **options)
    if sequences[longest] != ref or len(ref) + 8 > limit:
        raise ValueError("Tokenized context differs from native template or exceeds the model limit")
    return sequences


@torch.inference_mode()
def decode(model, tok, ids, generation, options):
    try:
        batch = tok.pad({"input_ids": ids}, padding=True, return_attention_mask=True, return_tensors="pt").to(model.device)
        output = model.generate(**batch, generation_config=generation, do_sample=False, **options)
        texts = tok.batch_decode(output[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        return [(raw, len(ids)) for raw in texts]
    except torch.OutOfMemoryError:
        if len(ids) == 1:
            raise
    torch.cuda.empty_cache()
    mid = len(ids) // 2
    return decode(model, tok, ids[:mid], generation, options) + decode(model, tok, ids[mid:], generation, options)


def infer(run, name, batch_size):
    rows, _ = source()
    manifest = verified(run)
    raw = read_jsonl(run / "retrieval.jsonl")
    groups = defaultdict(list)
    for r in raw:
        groups[r["seed"], r["fold"], r["percent"]].append(r)
    folder = run / "predictions" / name
    folder.mkdir(parents=True, exist_ok=True)
    model, tok = load(name)
    text_config = getattr(model.config, "text_config", model.config)
    ctx = min(tok.model_max_length, text_config.max_position_embeddings)
    generation = GenerationConfig(do_sample=False, max_new_tokens=8, use_cache=True,
                                  eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    options = {"use_model_defaults": False} if "use_model_defaults" in inspect.signature(
        GenerationMixin.generate).parameters else {}
    config = dict(model=name, revision=MODELS[name], gpu=torch.cuda.get_device_name(0),
                  torch=torch.__version__, transformers=transformers.__version__, dtype=str(model.dtype),
                  context_limit=ctx, max_new_tokens=8, decoding="greedy", enable_thinking=False,
                  retrieval_sha256=manifest["retrieval_sha256"], truncation=False, k=32,
                  batch_size=batch_size, generation_switches=options)
    settings = folder / "settings.json"
    if settings.exists() and json.loads(settings.read_bytes()) != config:
        raise ValueError("Inference settings differ from the existing run")
    write_json(settings, config)
    print("MODEL", config, flush=True)
    for (seed, fold, percent), records in sorted(groups.items()):
        file = folder / f"seed-{seed}-fold-{fold}-p{percent}.jsonl"
        if file.exists():
            previous = read_jsonl(file)
            if [r["row"] for r in previous] != [r["row"] for r in records]:
                raise ValueError(f"Incomplete previous shard: {file.name}")
            continue
        start = time.monotonic()
        sequences = tokenize(tok, rows, records, ctx)
        order = sorted(range(len(records)), key=lambda j: len(sequences[j]))
        results = [None] * len(records)
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset+batch_size]
            answers = decode(model, tok, [sequences[i] for i in indices], generation, options)
            for j, (raw_answer, used_batch) in zip(indices, answers, strict=True):
                key = records[j]["row"]
                source_row = rows[key-1]
                pred = raw_answer.strip()
                results[j] = dict(seed=seed, fold=fold, percent=percent, model=name, row=key,
                                  query_id=source_row["query_id"], input=source_row["prompt"][-1]["content"],
                                  gold=source_row["completion"][0]["content"], raw=raw_answer,
                                  prediction=pred if pred in ("Yes", "No") else "Invalid",
                                  input_tokens=len(sequences[j]), inference_batch=used_batch)
            if offset == 0 or (offset + batch_size) % 64 == 0:
                print("BATCH", name, seed, fold, percent, min(offset+batch_size, len(records)),
                      len(records), round(time.monotonic()-start, 1), flush=True)
        temp = file.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8", newline="\n") as out:
            for result in results:
                print(json.dumps(result, ensure_ascii=False), file=out)
        temp.replace(file)
        print("DONE", name, seed, fold, percent, len(results), "sec", round(time.monotonic()-start, 2),
              "max_tokens", max(map(len, sequences)), "peak_GiB",
              round(torch.cuda.max_memory_allocated()/2**30, 2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    infer(run_directory(args.run), args.model, args.batch_size)
