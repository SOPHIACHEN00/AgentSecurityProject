"""Validate all OOF decisions, pool by repeat, and plot paired same-weight losses."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, dataset, group_folds

METRICS = ("accuracy", "macro_f1", "FP", "FN", "FPR", "FNR", "invalid")
STEPS = np.arange(0, 289, 32)


def decisions(folder, expected):
    rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_bytes().splitlines()]
    actual = {(r["query_id"], r["input"]): r["gold"] for r in rows}
    if len(actual) != len(rows) or actual != expected:
        raise ValueError(f"Decision or gold-label mismatch in {folder}")
    return rows


def score(rows):
    counts = Counter((r["gold"], r["prediction"]) for r in rows)
    tp, tn, fp, fn = [counts[k] for k in (("Yes", "Yes"), ("No", "No"), ("No", "Yes"), ("Yes", "No"))]
    iy, ino = counts["Yes", "Invalid"], counts["No", "Invalid"]
    if tp + tn + fp + fn + iy + ino != len(rows):
        raise ValueError("Unexpected label in predictions")
    return dict(accuracy=(tp + tn) / len(rows),
                macro_f1=(tp / (2*tp + fp + fn + iy) + tn / (2*tn + fn + fp + ino)),
                FP=fp, FN=fn, FPR=fp / (tn + fp + ino), FNR=fn / (tp + fn + iy), invalid=iy + ino)


def save_csv(file, rows):
    with file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(series, dest):
    top = math.ceil(max(np.percentile(v, 75, axis=0).max() for v in series.values()) * 1.08 * 10) / 10
    for size, values in series.items():
        fig, ax = plt.subplots(figsize=(8, 4.65), layout="constrained")
        for i, (label, color, style, marker) in enumerate([
                ("Training", "#126d83", "-", "o"), ("Validation", "#ad4927", "--", "s")]):
            traces = values[:, i, :]
            low, high = np.percentile(traces, [25, 75], axis=0)
            ax.fill_between(STEPS, low, high, color=color, alpha=.13, linewidth=0)
            ax.plot(STEPS, traces.mean(axis=0), style, color=color, marker=marker,
                    lw=2, ms=4.6, label=label)
        ax.set_title(f"Qwen3.5-{size} training and validation loss", loc="left", fontsize=15, pad=13)
        ax.set(xlim=(-4, 292), ylim=(0, top), xticks=STEPS, xlabel="Optimizer step",
               ylabel="Mean loss per supervised token")
        ax.grid(axis="y", alpha=.2)
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(frameon=False)
        fig.savefig(dest / f"cv-qwen35-{size.lower()}-loss.png", dpi=205)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    run, figures = ROOT / args.run, ROOT / args.figure_dir
    figures.mkdir(parents=True, exist_ok=True)
    rows, digest = dataset()
    if digest != "f28d5f72d47e8ab27fc5780b0be95b08e2f870cb384b7ed6f2d7175795dc1a94":
        raise ValueError("Unexpected source data")
    gold = {(r["query_id"], r["prompt"][-1]["content"]): r["completion"][0]["content"] for r in rows}
    plan = json.loads((run / "folds.json").read_bytes())
    summaries, repeats, series = [], [], {}
    for size in ("0.8B", "2B"):
        base = score(decisions(run / size / "base", gold))
        summaries.append(dict(model=size, weights="Raw", **{k: {"mean": v, "sd": None} for k,v in base.items()}))
        curves = []
        for seed in (42, 43, 44):
            combined = []
            for i, ids in enumerate(group_folds(rows, seed), 1):
                if sorted(ids) != plan[str(seed)][i-1]:
                    raise ValueError("Changed fold membership")
                held = {k:v for k,v in gold.items() if k[0] in ids}
                directory = run / size / f"seed-{seed}" / f"fold-{i}"
                predictions = decisions(directory, held)
                combined.extend(predictions)
                measured = score(predictions)
                original = json.loads((directory / "metrics.json").read_bytes())
                for key, value in measured.items():
                    saved = original["invalid_outputs" if key == "invalid" else key]
                    if not math.isclose(value, saved, abs_tol=1e-12):
                        raise ValueError(f"Metric mismatch in {directory}: {key}")
                last = 2 * math.ceil((len(gold) - len(held)) / 8)
                logs = [json.loads(line) for line in (directory / "steps.jsonl").read_bytes().splitlines()]
                if [entry["step"] for entry in logs] != list(range(1, last + 1)) or any(
                        not math.isfinite(entry[k]) for entry in logs for k in ("loss", "grad_norm", "learning_rate")):
                    raise ValueError(f"Incomplete optimizer history in {directory}")
                points = json.loads((directory / "loss.json").read_bytes())
                if [p["step"] for p in points] != sorted(set(STEPS.tolist()) | {last}):
                    raise ValueError(f"Unexpected loss measurement steps in {directory}")
                traces = [[p[key] for p in points if p["step"] in STEPS] for key in ("training", "validation")]
                if not np.isfinite(traces).all():
                    raise ValueError("Nonfinite loss")
                curves.append(traces)
            if len(combined) != len(gold):
                raise ValueError("Incomplete OOF coverage")
            repeats.append(dict(model=size, seed=seed, **score(combined)))
        stats = {}
        for k in METRICS:
            vals = [r[k] for r in repeats if r["model"] == size]
            stats[k] = {"mean": mean(vals), "sd": stdev(vals)}
        summaries.append(dict(model=size, weights="SFT", **stats))
        series[size] = np.asarray(curves)
    save_csv(run / "per-repeat.csv", repeats)
    averaged = []
    for size, values in series.items():
        for step, (training, validation) in zip(STEPS, values.mean(axis=0).T, strict=True):
            averaged.append(dict(model=size, step=int(step), training=training, validation=validation))
    save_csv(run / "loss-by-step.csv", averaged)
    (run / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    render(series, figures)
    for r in summaries:
        print(r)
    for size, traces in series.items():
        print(size, "Mean loss at steps 0 and 288 (train, validation)", traces.mean(axis=0)[:, [0, -1]])


if __name__ == "__main__":
    main()
