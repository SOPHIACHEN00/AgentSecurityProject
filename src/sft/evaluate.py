"""Evaluate the base model or a saved adapter on held-out permission decisions."""
from collections import Counter
import json
from pathlib import Path

from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
import torch
from transformers import GenerationConfig

from common import ROOT, experiment_parser, load_model, new_run_dir, partition, permission_prompt, write_json


@torch.inference_mode()
def predict(model, tokenizer, rows):
    """Decode with the same greedy settings during validation and standalone evaluation."""
    generation = GenerationConfig(do_sample=False, max_new_tokens=8, use_cache=True,
                                  eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    results = []
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(rows), 8):
            batch = rows[start:start+8]
            inputs = permission_prompt(tokenizer, [row["prompt"] for row in batch],
                                       padding=True, return_tensors="pt", return_dict=True).to(model.device)
            output = model.generate(**inputs, generation_config=generation,
                                    use_model_defaults=False, do_sample=False)
            texts = tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            for row, raw in zip(batch, texts, strict=True):
                answer = raw.strip()
                results.append({"query_id": row["query_id"], "input": row["prompt"][-1]["content"],
                                "gold": row["completion"][0]["content"], "raw": raw,
                                "prediction": answer if answer in ("Yes", "No") else "Invalid"})
    finally:
        tokenizer.padding_side = previous_padding
    return results


def permission_metrics(results):
    gold = [r["gold"] for r in results]
    pred = [r["prediction"] for r in results]
    confusion = Counter(zip(gold, pred))
    totals = Counter(gold)
    fp, fn = confusion["No", "Yes"], confusion["Yes", "No"]
    return {
        "n": len(results), "queries": len({r["query_id"] for r in results}),
        "positive_label": "Yes", "gold_counts": dict(totals), "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=["Yes", "No"], average="macro", zero_division=0),
        "invalid_outputs": pred.count("Invalid"),
        "TP": confusion["Yes", "Yes"], "TN": confusion["No", "No"], "FP": fp, "FN": fn,
        "FPR": fp / totals["No"] if totals["No"] else None,
        "FNR": fn / totals["Yes"] if totals["Yes"] else None,
    }



@torch.inference_mode()
def main():
    parser = experiment_parser(__doc__)
    parser.add_argument("--adapter", type=Path, help="Path to a training run's adapter")
    args = parser.parse_args()
    _, rows, info = partition(args)
    model, tokenizer = load_model(args.model)
    if args.adapter:
        folder = ROOT / args.adapter
        training = json.loads((folder.parent / "training.json").read_text(encoding="utf-8"))
        if any(training[key] != value for key, value in info.items()):
            raise ValueError("Adapter and evaluation must use the same model, dataset, seed and fold")
        model = PeftModel.from_pretrained(model, folder, is_trainable=False)
    model.eval()
    out = new_run_dir(args.output_dir)
    results = predict(model, tokenizer, rows)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = permission_metrics(results)
    write_json(out / "metrics.json", {**info, "adapter": str(args.adapter) if args.adapter else None, **metrics})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
