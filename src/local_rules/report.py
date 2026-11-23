"""Audit the shared libraries, retrievals and OOF decisions before reporting."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from prepare import (MODELS, PERCENT, ROOT, SEEDS, libraries, read_jsonl,
                     run_directory, source, verified, write_json)

METRICS = ("accuracy", "macro_f1", "FP", "FN", "FPR", "FNR", "invalid_outputs")
METHODS = ("Raw", "Local Rules 25%", "Local Rules 50%", "Local Rules 100%", "SFT")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def audit_inputs(run, rows, plan, manifest):
    folds = {(d["seed"], d["fold"]): d for d in manifest["folds"]}
    allowed, expected, targets = {}, set(), {}
    all_ids = {r["query_id"] for r in rows}
    for f in plan["folds"]:
        seed, fold = f["seed"], f["fold"]
        spec = folds[seed, fold]
        held = set(f["test_query_ids"])
        test = {i for i,r in enumerate(rows, 1) if r["query_id"] in held}
        targets[seed, fold] = test
        if spec["test_query_ids"] != f["test_query_ids"] or sorted(test) != spec["test_rows"]:
            raise ValueError("Held-out group mismatch")
        for p, ids in libraries(all_ids - held, seed, fold):
            lib = spec["libraries"][str(p)]
            indices = {i for i,r in enumerate(rows, 1) if r["query_id"] in ids}
            labels = Counter(rows[i-1]["completion"][0]["content"] for i in indices)
            if (sorted(ids), len(ids), len(indices), labels) != (
                    lib["query_ids"], lib["n_queries"], lib["n_cases"], lib["labels"]):
                raise ValueError("Library membership or label counts differ")
            allowed[seed, fold, p] = indices
            expected.update((seed, fold, p, i) for i in test)
    seen = set()
    for r in read_jsonl(run / "retrieval.jsonl"):
        group = (r["seed"], r["fold"], r["percent"])
        key = (*group, r["row"])
        ids, scores = r["cases"], r["scores"]
        if key in seen or key not in expected or len(set(ids)) != 32 or not set(ids) <= allowed[group]:
            raise ValueError("Duplicate or out-of-library case or target")
        if len(ids) != 32 or len(scores) != 32 or not all(math.isfinite(x) and 0 <= x <= 1+1e-10 for x in scores):
            raise ValueError("Invalid ranked scores")
        order = list(zip(scores, ids))
        if order != sorted(order, key=lambda x: (-x[0], x[1])):
            raise ValueError("Retrieval ordering or tie-breaking mismatch")
        seen.add(key)
    if seen != expected or len(seen) != 12996:
        raise ValueError("Incomplete retrieval keys")
    for seed in SEEDS:
        counts = Counter(i for fold in range(1, 6) for i in targets[seed, fold])
        if len(counts) != len(rows) or set(counts.values()) != {1}:
            raise ValueError("Each decision must be held out once per repeat")
    return targets


def scored(model, method, seed, values, rows):
    if set(values) != set(range(1, len(rows)+1)):
        raise ValueError(f"Incomplete decisions: {model}, {method}, {seed}")
    c = Counter((rows[i-1]["completion"][0]["content"], pred) for i,pred in values.items())
    tp, tn, fp, fn = (c[k] for k in (("Yes","Yes"), ("No","No"), ("No","Yes"), ("Yes","No")))
    iy, ino = c["Yes","Invalid"], c["No","Invalid"]
    if tp+tn+fp+fn+iy+ino != len(rows):
        raise ValueError("Unexpected prediction")
    f1 = tp/(2*tp + fp + fn + iy) + tn/(2*tn + fp + fn + ino)
    return dict(model=model, method=method, seed=seed, n=len(rows), TP=tp, TN=tn,
                accuracy=(tp+tn)/len(rows), macro_f1=f1, FP=fp, FN=fn,
                FPR=fp/(tn+fp+ino), FNR=fn/(tp+fn+iy), invalid_outputs=iy+ino)


def validate_prediction(r, row):
    if (r["query_id"],r["input"],r["gold"]) != (row["query_id"],row["prompt"][1]["content"],row["completion"][0]["content"]):
        raise ValueError("Prediction does not match its source row")
    stripped = r["raw"].strip()
    if r["prediction"] != (stripped if stripped in ("Yes", "No") else "Invalid"):
        raise ValueError("Prediction parser mismatch")


def new_results(run, rows, targets, manifest):
    scores, audit = [], {}
    for model, revision in MODELS.items():
        folder = run / "predictions" / model
        cfg = json.loads((folder / "settings.json").read_bytes())
        if (cfg["revision"], cfg["retrieval_sha256"], cfg["max_new_tokens"], cfg["enable_thinking"],
            cfg["truncation"], cfg["decoding"], cfg["k"]) != (revision, manifest["retrieval_sha256"], 8, False,
                                                               False, "greedy", 32):
            raise ValueError("Inference configuration mismatch")
        lengths = []
        for seed in SEEDS:
            for p in PERCENT:
                values = {}
                for fold in range(1,6):
                    records = read_jsonl(folder / f"seed-{seed}-fold-{fold}-p{p}.jsonl")
                    if {r["row"] for r in records} != targets[seed,fold] or len(records) != len(targets[seed,fold]):
                        raise ValueError("Foreign or duplicated test row")
                    for r in records:
                        if (r["model"], r["seed"], r["fold"], r["percent"]) != (model, seed, fold, p):
                            raise ValueError("Misidentified prediction")
                        validate_prediction(r, rows[r["row"]-1])
                        if not 0 < r["input_tokens"] <= cfg["context_limit"]-8:
                            raise ValueError("Context overflow")
                        lengths.append(r["input_tokens"])
                        values[r["row"]] = r["prediction"]
                scores.append(scored(model, f"Local Rules {p}%", seed, values, rows))
        if len(lengths) != 12996 or len(list(folder.glob("seed-*.jsonl"))) != 45:
            raise ValueError("Extra or missing predictions")
        audit[model] = dict(n=len(lengths), min_input_tokens=min(lengths), max_input_tokens=max(lengths),
                            context_limit=cfg["context_limit"], revision=revision)
    return scores, audit


def verify_old_metrics(actual, ref, invalid_key="invalid_outputs"):
    for name in METRICS:
        if not math.isclose(actual[name], float(ref[invalid_key if name == "invalid_outputs" else name]), abs_tol=1e-12):
            raise ValueError(f"Historical result mismatch: {actual['model']} {actual['method']} {name}")


def published_history():
    """Read the previously audited Raw and SFT repeats for the comparison."""
    records = read_csv(ROOT / "results/cross-validation-reference.csv")
    expected = {(model, "Raw", None) for model in MODELS}
    expected.update((model, "SFT", seed) for model in MODELS for seed in SEEDS)
    for row in records:
        row["seed"] = int(row["seed"]) if row["seed"] else None
        for field in ("n", "TP", "TN", "FP", "FN", "invalid_outputs"):
            row[field] = int(row[field])
        for field in ("accuracy", "macro_f1", "FPR", "FNR"):
            row[field] = float(row[field])
        key = (row["model"], row["method"], row["seed"])
        if key not in expected or row["n"] != 1444:
            raise ValueError(f"Unexpected reference score: {key}")
        expected.remove(key)
    if expected:
        raise ValueError(f"Missing reference scores: {expected}")
    return records


def history(rows, targets, original, old35):
    id_for = {(r["query_id"],r["prompt"][1]["content"],r["completion"][0]["content"]):i for i,r in enumerate(rows,1)}
    old_rows = read_csv(original / "oof-predictions.csv")
    old_means = read_csv(original / "cv-summary.csv")
    old_seeds = read_csv(original / "per-repeat.csv")
    results = []
    for model, size in (("Qwen3-4B-Instruct-2507","4b"), ("Qwen3-8B","8b")):
        base, tuned = {}, defaultdict(dict)
        for r in old_rows:
            i = id_for[r["query_id"],r["input"],r["gold"]]
            seed, fold = int(r["seed"]), int(r["fold"])
            if i not in targets[seed,fold] or i in tuned[seed]:
                raise ValueError("Mismatched old Qwen3 fold")
            tuned[seed][i] = r[f"{size}_sft"]
            if base.setdefault(i,r[f"{size}_base"]) != r[f"{size}_base"]:
                raise ValueError("Inconsistent raw prediction")
        item = scored(model, "Raw", None, base, rows)
        verify_old_metrics(item, next(v for v in old_means if v["model"]==size and v["stage"]=="baseline"))
        results.append(item)
        for seed in SEEDS:
            item = scored(model, "SFT", seed, tuned[seed], rows)
            verify_old_metrics(item, next(v for v in old_seeds if v["model"]==size and int(v["seed"])==seed))
            results.append(item)
    old_seeds = read_csv(old35 / "per-repeat.csv")
    for size in ("2B","0.8B"):
        model = "Qwen3.5-" + size
        for seed in (None, *SEEDS):
            values = {}
            for fold in (range(1,6) if seed is not None else [None]):
                path = old35 / size / (f"seed-{seed}/fold-{fold}" if fold is not None else "base")
                m = json.loads((path / "metrics.json").read_bytes())
                if m["revision"] != MODELS[model]:
                    raise ValueError("Historical model revision differs")
                for r in read_jsonl(path / "predictions.jsonl"):
                    i = id_for[r["query_id"],r["input"],r["gold"]]
                    if i in values or (fold is not None and i not in targets[seed,fold]):
                        raise ValueError("Mismatched old Qwen3.5 decision")
                    validate_prediction(r, rows[i-1])
                    values[i] = r["prediction"]
            item = scored(model, "Raw" if seed is None else "SFT", seed, values, rows)
            if seed is None:
                verify_old_metrics(item, m)
            else:
                verify_old_metrics(item, next(v for v in old_seeds if v["model"]==size and int(v["seed"])==seed), "invalid")
            results.append(item)
    return results


def display(stat, name):
    percent = name in ("accuracy", "FPR", "FNR")
    factor = 100 if percent else 1
    places = 4 if name == "macro_f1" else 2
    if stat["repeats"] == 1 and name in ("FP", "FN", "invalid_outputs"):
        return str(int(stat[name]))
    val = f"{stat[name]*factor:.{places}f}"
    if stat["repeats"] > 1:
        val += f" ± {stat[name+'_sd']*factor:.{places}f}"
    return val


def summarize(run, report, qwen3_history=None, qwen35_history=None):
    rows, plan = source()
    manifest = verified(run)
    targets = audit_inputs(run, rows, plan, manifest)
    results, audit = new_results(run, rows, targets, manifest)
    if (qwen3_history is None) != (qwen35_history is None):
        raise ValueError("Provide both historical run directories for a prediction-level audit")
    if qwen3_history is None:
        results.extend(published_history())
    else:
        results.extend(history(rows, targets, qwen3_history, qwen35_history))
    write_csv(run / "per-seed.csv", results)
    summary, differences = [], []
    lines = ["# 本地案例库大小对权限判断的影响", "",
             "四个模型回答相同的 1,444 道题，共 169 个 query，871 个 Yes、573 个 No。沿用种子 42、43、44 的五折划分。"
             "25%、50%、100% 表示训练侧 query_id 的比例，每题由 TF-IDF 检索 32 条已标注案例。",
             "", "Raw 是原始模型一次全量评估的实测值。其余行先合并每个种子的五折预测，再报告三个种子的均值 ± 样本标准差。"
             "Yes 表示任务层面的默认许可。FP 为错误放行，FN 为错误拒绝。FPR 的分母为 573 条应拒绝的访问，FNR 的分母为 871 条应放行的访问。无效输出计错并单独列出。",
             "", "四个模型共用 12,996 组检索结果，合计 51,984 条新增预测。每折案例库只含训练侧 query，各条件都评估同一批留出题。"
             "以下 FP 和 FN 的差值均为每 1,444 条判断的平均条数。", ""]
    for model in MODELS:
        lines.extend([f"## {model}", "", "| 方法 | Accuracy (%) | Macro-F1 | FP | FN | FPR (%) | FNR (%) | 无效输出数 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        by_method = {}
        for method in METHODS:
            group = [r for r in results if (r["model"], r["method"]) == (model,method)]
            if {r["seed"] for r in group} != ({None} if method=="Raw" else set(SEEDS)):
                raise ValueError("Incorrect repeat coverage")
            stat = dict(model=model, method=method, repeats=len(group))
            for name in METRICS:
                vals = [r[name] for r in group]
                stat[name] = mean(vals)
                stat[name+"_sd"] = stdev(vals) if len(group)>1 else None
            summary.append(stat)
            by_method[method] = stat
            label = "Raw + " + method + " (32 cases)" if method.startswith("Local") else method
            lines.append("| " + " | ".join([label]+[display(stat,k) for k in METRICS]) + " |")
        lines.append("")
        for left,right in (("Local Rules 50%","Local Rules 25%"),("Local Rules 100%","Local Rules 50%"),
                           ("Local Rules 100%","Local Rules 25%"),("Local Rules 100%","Raw"),("Local Rules 100%","SFT")):
            x,y = by_method[left],by_method[right]
            differences.append(dict(model=model,left=left,right=right,accuracy_pp=100*(x["accuracy"]-y["accuracy"]),
                                    macro_f1=x["macro_f1"]-y["macro_f1"], FP=x["FP"]-y["FP"], FN=x["FN"]-y["FN"]))
        small, full = by_method["Local Rules 25%"], by_method["Local Rules 100%"]
        lines.append(f"案例库从 25% 扩到 100% 后，平均准确率变化 {100*(full['accuracy']-small['accuracy']):+.2f} 个百分点，"
                     f"FP 变化 {full['FP']-small['FP']:+.2f} 条，FN 变化 {full['FN']-small['FN']:+.2f} 条。")
        for ref in ("Raw", "SFT"):
            full, other = by_method["Local Rules 100%"], by_method[ref]
            lines.append(f"100% 案例库相对 {ref} 的准确率差为 {100*(full['accuracy']-other['accuracy']):+.2f} 个百分点，"
                         f"FP 差为 {full['FP']-other['FP']:+.2f} 条，FN 差为 {full['FN']-other['FN']:+.2f} 条。")
        lines.append("")
    write_csv(run / "summary.csv", summary)
    write_csv(run / "differences.csv", differences)
    write_json(run / "audit.json", dict(retrievals=12996, new_predictions=sum(m["n"] for m in audit.values()),
                                       models=audit, source_sha256=manifest["dataset_sha256"],
                                       plan_sha256=manifest["plan_sha256"], retrieval_sha256=manifest["retrieval_sha256"]))
    report.parent.mkdir(parents=True, exist_ok=True)
    temp = report.with_suffix(".md.tmp")
    temp.write_text("\n".join(lines), encoding="utf-8")
    temp.replace(report)
    print("Verified 51,984 local-rules predictions; included the audited Raw and SFT scores.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--qwen3-history", type=Path, help="Optional original Qwen3 run for rescoring Raw/SFT")
    parser.add_argument("--qwen35-history", type=Path, help="Companion Qwen3.5 run for rescoring Raw/SFT")
    args = parser.parse_args()
    summarize(run_directory(args.run), ROOT / "results/local-rules-comparison.md",
              ROOT / args.qwen3_history if args.qwen3_history else None,
              ROOT / args.qwen35_history if args.qwen35_history else None)
