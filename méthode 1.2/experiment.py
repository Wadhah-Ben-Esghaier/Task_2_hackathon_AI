"""Sélection, calibration et évaluation locale sur les neuf références."""
import argparse
import hashlib
import json
import os
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from design import FLIPS, METHOD1, PROMPTS, REFS, ROOT, read_records, sample
from cluster import Config, cluster, model_distances, read_truth, score, write_csv
from data import build_answers
from signals import flips_distance, margins, panels


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(f.name, path)


def fingerprint():
    return hashlib.sha256(json.dumps(PROMPTS, sort_keys=True).encode()).hexdigest()


def parts(records, endpoints, arm, starts, cache):
    ids, repeats = arm["ids"], arm["repeats"]
    key = (tuple(ids), repeats)
    if key not in cache:
        rows = sample(records, endpoints, ids, starts[0], repeats)
        matrix = model_distances(build_answers(rows, endpoints=endpoints), Config())[-1]
        if not np.isfinite(matrix).all():
            raise ValueError("Distance textuelle non finie. Vérifier le plan équilibré.")
        cache[key] = matrix
    if not arm["flips"]:
        return cache[key], None, None
    binary = sample(records, endpoints, FLIPS, starts[1], 5)
    matrix, coverage = flips_distance(binary, endpoints)
    return cache[key], matrix, coverage


def robust_scale(matrix, keep):
    values = matrix[keep]
    if not values.size:
        return None
    q25, q75 = np.percentile(values, [25, 75])
    if q75 - q25 <= 1e-12:
        return None
    return [float(np.median(values)), float(q75 - q25)]


def combine(text, flips, coverage, scales):
    if scales is None:
        return text
    # Les échelles sont apprises lors de la calibration puis gelées.
    (mt, st), (mf, sf) = scales
    z = (text - mt) / st
    fused = np.where(coverage >= 2, 0.5 * (z + (flips - mf) / sf), z)
    np.fill_diagonal(fused, 0)
    return fused


def as_rows(endpoints, labels, family="P"):
    return [(ep, family, f"{family}_{int(label)}") for ep, label in zip(endpoints, labels)]


def threshold(matrix, endpoints, truth):
    values = squareform(matrix, checks=True)
    shift = min(0., values.min())
    heights = np.unique(linkage(values - shift, method="average")[:, 2] + shift)
    cuts = [float(heights[0] - max(1., abs(heights[0])) * 1e-8),
            *((heights[:-1] + heights[1:]) / 2), float(heights[-1] + max(1., abs(heights[-1])) * 1e-8)]
    scores = [score(truth, as_rows(endpoints, cluster(matrix, t)))["model"] for t in cuts]
    winners = [float(t) for t, s in zip(cuts, scores) if s == max(scores)]
    return winners[(len(winners) - 1) // 2]


def fit(records, endpoints, truth, seed):
    margin, names = margins(records, endpoints, truth)
    plan = {"version": 1, "prompts_sha256": fingerprint(), "seed": seed,
            "contrasts": names, "margins": margin, "arms": {}}
    cache = {}
    for name, (ids, repeats, binary) in panels(margin, seed).items():
        arm = dict(ids=ids, repeats=repeats, flips=binary, scales=None,
                   budget=len(ids) * repeats + 20 * int(binary))
        text, flips, coverage = parts(records, endpoints, arm, (4, 0), cache)
        if binary:
            upper = np.triu(np.ones(text.shape, dtype=bool), 1)
            scales = [robust_scale(text, upper), robust_scale(flips, upper & (coverage >= 2))]
            if all(scales):
                arm["scales"] = scales
        matrix = combine(text, flips, coverage, arm["scales"])
        arm["threshold"] = threshold(matrix, endpoints, truth)
        plan["arms"][name] = arm
    return plan


def read_plan(path):
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan["version"] != 1 or plan["prompts_sha256"] != fingerprint():
        raise ValueError("Plan incompatible avec la version ou le texte des prompts.")
    return plan


def export_prompts(plan, arm_name, directory):
    """Manifeste utilisable hors ligne, une liste JSONL par nombre de répétitions."""
    arm = plan["arms"][arm_name]
    directory.mkdir(parents=True, exist_ok=True)
    groups = [(arm["ids"], arm["repeats"])]
    if arm["flips"]:
        groups.append((list(FLIPS), 5))
    for ids, repeats in groups:
        path = directory / f"{arm_name}_repeat{repeats}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for pid in ids:
                f.write(json.dumps({"id": pid, "prompt": PROMPTS[pid]}, ensure_ascii=False) + "\n")
        print(f"{path}: {len(ids)} prompts x {repeats}")


def evaluate(records, plan, endpoints, truth, out=None, held=None):
    cache, result = {}, {}
    for name, arm in plan["arms"].items():
        text, flips, coverage = parts(records, endpoints, arm, (8, 5), cache)
        matrix = combine(text, flips, coverage, arm["scales"])
        rows = as_rows(endpoints, cluster(matrix, arm["threshold"]))
        if out is not None:
            write_csv(rows, out / f"{name}.csv")
        pairs = []
        for i, j in combinations(range(len(endpoints)), 2):
            a, b = endpoints[i], endpoints[j]
            if held is not None and held not in (a, b):
                continue
            pairs.append(dict(a=a, b=b, same=truth[a][:2] == truth[b][:2],
                              predicted=rows[i][1:] == rows[j][1:], distance=float(matrix[i,j]),
                              alphabets=int(coverage[i,j]) if coverage is not None else None))
        result[name] = dict(budget=arm["budget"], threshold=arm["threshold"],
                            hybrid_active=arm["scales"] is not None, pairs=pairs,
                            **metrics(pairs))
    return result


def metrics(pairs):
    from cluster import mcc
    tp = sum(p["same"] and p["predicted"] for p in pairs)
    tn = sum(not p["same"] and not p["predicted"] for p in pairs)
    fp = sum(not p["same"] and p["predicted"] for p in pairs)
    fn = len(pairs) - tp - tn - fp
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, mcc=mcc(tp, tn, fp, fn))


def print_report(result):
    for name, row in result.items():
        print(f"{name:19} MCC={row['mcc']:.3f}  FP={row['fp']:2}  FN={row['fn']:2}")


def leave_one_out(records, truth, seed):
    pooled = {}
    folds = {}
    for held in REFS:
        train = [e for e in REFS if e != held]
        plan = fit(records, train, truth, seed)
        rows = evaluate(records, plan, REFS, truth, held=held)
        folds[held] = rows
        for name, row in rows.items():
            pooled.setdefault(name, []).extend(row["pairs"])
    return dict(folds=folds, pooled={name: metrics(pairs) for name, pairs in pooled.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fit", "test", "loo", "export"])
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--plan", type=Path, default=ROOT / "runs/plan.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arm", default="margin_flips_100")
    args = parser.parse_args()
    if args.action == "export":
        export_prompts(read_plan(args.plan), args.arm, args.out)
        return
    if args.answers is None:
        parser.error("--answers est nécessaire pour cette action.")
    truth = read_truth(METHOD1 / "practice_answers.csv")
    records = read_records(args.answers, REFS)
    if args.action == "fit":
        save_json(args.plan, fit(records, REFS, truth, args.seed))
        print(args.plan)
    elif args.action == "test":
        args.out.mkdir(parents=True, exist_ok=True)
        result = evaluate(records, read_plan(args.plan), REFS, truth, args.out)
        save_json(args.out / "test.json", result)
        print_report(result)
    else:
        result = leave_one_out(records, truth, args.seed)
        save_json(args.out / "loo.json", result)
        print_report(result["pooled"])


if __name__ == "__main__":
    main()
