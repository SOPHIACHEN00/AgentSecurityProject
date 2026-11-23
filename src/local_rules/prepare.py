"""Prepare one frozen collection of folds, nested libraries and shared TF-IDF hits."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("data/processed/wu_ci_work_sft_reviewed.jsonl")
PLAN = Path("data/splits/permission-cv-plan.json")
DATA_HASH = "f28d5f72d47e8ab27fc5780b0be95b08e2f870cb384b7ed6f2d7175795dc1a94"
PLAN_HASH = "a03b5b269a4d653a920f3cc0c37a074628eac3777b2bcad1a9937b5506ef789e"
SEEDS = (42, 43, 44)
PERCENT = (25, 50, 100)
MODELS = {
    "Qwen3-4B-Instruct-2507": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
}


def run_directory(name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("runs",) or len(relative.parts) < 2:
        raise ValueError("Expected a project-relative experiment directory under runs/")
    return ROOT / relative


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, content):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream]


def source():
    if sha(ROOT / DATA) != DATA_HASH or sha(ROOT / PLAN) != PLAN_HASH:
        raise ValueError("Source dataset or historical fold manifest changed")
    rows = read_jsonl(ROOT / DATA)
    qids = {r["query_id"] for r in rows}
    counts = Counter(r["completion"][0]["content"] for r in rows)
    if len(rows) != 1444 or len(qids) != 169 or counts != {"Yes": 871, "No": 573}:
        raise ValueError("Unexpected source counts")
    if len({r["prompt"][-1]["content"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source input")
    if len({r["prompt"][0]["content"] for r in rows}) != 1 or any(
        [m["role"] for m in r["prompt"]] != ["system", "user"] or
        [m["role"] for m in r["completion"]] != ["assistant"] for r in rows
    ):
        raise ValueError("Unexpected message schema")
    plan = json.loads((ROOT / PLAN).read_bytes())
    if plan["source_sha256"] != DATA_HASH or plan["seeds"] != list(SEEDS) or len(plan["folds"]) != 15:
        raise ValueError("Historical plan does not match the source")
    return rows, plan


def libraries(train_ids, seed, fold):
    wu = sorted(q for q in train_ids if q.isdigit())
    ci = sorted(train_ids - set(wu))
    rng = random.Random(100 * seed + fold)
    rng.shuffle(wu)
    rng.shuffle(ci)
    previous = set()
    for percent in PERCENT:
        m = math.ceil(percent * len(train_ids) / 100)
        n_wu = math.floor(m * len(wu) / len(train_ids) + 0.5)
        ids = set(wu[:n_wu] + ci[:m-n_wu])
        if len(ids) != m or not previous <= ids <= train_ids:
            raise ValueError("Invalid nested library")
        yield percent, ids
        previous = ids


def prepare(run):
    import numpy as np
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    rows, plan = source()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "manifest.json").exists():
        verified(run)
        return
    qids = {r["query_id"] for r in rows}
    texts = [r["prompt"][-1]["content"] for r in rows]
    manifest = {"dataset": DATA.as_posix(), "dataset_sha256": DATA_HASH, "plan": PLAN.as_posix(),
                "plan_sha256": PLAN_HASH, "models": MODELS, "seeds": SEEDS, "source_rows": 1444,
                "line_numbering": "one-based JSONL lines", "sklearn": sklearn.__version__,
                "tfidf": {"ngram_range": [1, 2], "norm": "l2"}, "k": 32, "folds": []}
    all_keys = set()
    partial = run / "retrieval.jsonl.tmp"
    with partial.open("w", encoding="utf-8", newline="\n") as out:
        for f in plan["folds"]:
            seed, fold = f["seed"], f["fold"]
            held = set(f["test_query_ids"])
            test = [i for i,r in enumerate(rows) if r["query_id"] in held]
            if len(test) != f["test_n"] or len(rows) - len(test) != f["train_n"]:
                raise ValueError("Historical fold count mismatch")
            parts = {}
            for percent, ids in libraries(qids - held, seed, fold):
                indices = np.array([i for i,r in enumerate(rows) if r["query_id"] in ids])
                counts = Counter(rows[i]["completion"][0]["content"] for i in indices)
                parts[str(percent)] = {"query_ids": sorted(ids), "n_queries": len(ids),
                                       "n_cases": len(indices), "labels": counts}
                vec = TfidfVectorizer(ngram_range=(1, 2), norm="l2")
                index = vec.fit_transform([texts[i] for i in indices])
                scores = cosine_similarity(vec.transform([texts[i] for i in test]), index)
                for i, similarities in zip(test, scores, strict=True):
                    best = np.lexsort((indices, -similarities))[:32]
                    key = (seed, fold, percent, i+1)
                    if key in all_keys or len(best) != 32 or not np.isfinite(similarities[best]).all():
                        raise ValueError("Invalid retrieval")
                    all_keys.add(key)
                    record = dict(seed=seed, fold=fold, percent=percent, row=i+1,
                                  cases=(indices[best]+1).tolist(), scores=similarities[best].tolist())
                    out.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            if (seed, fold) == (42, 1):
                check = [(v["n_queries"], v["n_cases"]) for v in parts.values()]
                if (len(held), len(test), check) != (34, 288, [(34, 286), (68, 557), (135, 1156)]):
                    raise ValueError(f"Reference sampling check failed: {check}")
            manifest["folds"].append(dict(seed=seed, fold=fold, test_query_ids=sorted(held),
                                          test_rows=[i+1 for i in test], libraries=parts))
    if len(all_keys) != 12996:
        raise ValueError("Incomplete retrieval coverage")
    partial.replace(run / "retrieval.jsonl")
    manifest["retrieval_sha256"] = sha(run / "retrieval.jsonl")
    manifest["retrievals"] = len(all_keys)
    write_json(run / "manifest.json", manifest)
    print("Shared retrievals:", len(all_keys), manifest["retrieval_sha256"], flush=True)


def verified(run):
    manifest = json.loads((run / "manifest.json").read_bytes())
    if (manifest["dataset_sha256"] != DATA_HASH or manifest["plan_sha256"] != PLAN_HASH
            or sha(run / "retrieval.jsonl") != manifest["retrieval_sha256"] or manifest["models"] != MODELS):
        raise ValueError("Frozen experiment inputs changed")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    prepare(run_directory(parser.parse_args().run))
