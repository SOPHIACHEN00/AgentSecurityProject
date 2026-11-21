"""Run paired five-fold cross-validation and summarize out-of-fold predictions."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys

from common import MODELS, ROOT, dataset, group_folds, new_run_dir, write_json

HERE = Path(__file__).resolve().parent
METRICS = ("accuracy", "macro_f1", "FP", "FN", "FPR", "FNR", "invalid_outputs")


def invoke(script, shared, output, extra=()):
    subprocess.run([sys.executable, str(HERE / script), *shared, "--output-dir", str(output), *extra],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true", help="Write the fold assignments and exit")
    parser.add_argument("--smoke", action="store_true", help="First seed and fold only, five optimizer updates")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models):
        parser.error("Provide distinct seeds and model sizes")
    out = new_run_dir(args.output_dir)
    rows, digest = dataset()
    seeds = args.seeds[:1] if args.smoke else args.seeds
    plan = {seed: [sorted(fold) for fold in group_folds(rows, seed)] for seed in seeds}
    write_json(out / "plan.json", {"dataset_sha256": digest, "folds": plan, "models": args.models,
                                   "smoke": args.smoke, "n": len(rows)})
    if args.prepare_only:
        return
    summaries = []
    for seed in seeds:
        for size in args.models:
            predictions = {"base": [], "sft": []}
            for index, ids in enumerate(plan[seed], 1):
                if args.smoke and index > 1:
                    break
                folder = args.output_dir / f"seed-{seed}" / f"fold-{index}" / size
                shared = ["--model", size, "--seed", str(seed), "--fold", str(index)]
                invoke("evaluate.py", shared, folder / "base")
                invoke("train.py", shared, folder / "train", ["--max-steps", "5"] if args.smoke else [])
                invoke("evaluate.py", shared, folder / "sft", ["--adapter", str(folder / "train/adapter")])
                expected = {(r["query_id"], r["prompt"][-1]["content"]) for r in rows if r["query_id"] in ids}
                for kind, combined in predictions.items():
                    file = ROOT / folder / kind / "predictions.jsonl"
                    batch = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
                    keys = {(r["query_id"], r["input"]) for r in batch}
                    if keys != expected or len(batch) != len(expected):
                        raise ValueError("Evaluation did not cover the held-out fold exactly once")
                    combined.extend(batch)
            from evaluate import permission_metrics
            for kind, combined in predictions.items():
                if not args.smoke and len(combined) != len(rows):
                    raise ValueError("Incomplete out-of-fold coverage")
                summaries.append({"seed": seed, "model": size, "kind": kind, **permission_metrics(combined)})
    summary = {"scope": "smoke: first fold" if args.smoke else "pooled out-of-fold per repeat",
               "per_repeat": summaries, "across_repeats": []}
    for size in args.models:
        for kind in ("base", "sft"):
            repeats = [r for r in summaries if r["model"] == size and r["kind"] == kind]
            stats = {}
            for name in METRICS:
                values = [r[name] for r in repeats]
                stats[name] = {"mean": mean(values), "sd": stdev(values) if len(values) > 1 else None}
            summary["across_repeats"].append({"model": size, "kind": kind, "repeats": len(repeats), **stats})
    write_json(out / "summary.json", summary)


if __name__ == "__main__":
    main()
