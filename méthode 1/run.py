"""Point d'entrée en ligne de commande.

Sur les endpoints d'entraînement (réponses connues) : régler le seuil, voir les erreurs.
    python run.py ref --answers reference_answers.jsonl --truth practice_answers.csv

Sur les endpoints notés : produire submission.csv avec le seuil trouvé.
    python run.py ep --answers answers.jsonl --model-threshold 3.2 --out submission.csv

Options utiles : --tokenizer char|word|hf:gpt2   --kernel hamming|ngram|onehot
                 --summary mean|max|median       --method average|complete
                 --n-families K                  --family-threshold T

Pour l'appeler depuis solve() de task_template.py :
    from run import solve_from_answers
    def solve(endpoints):
        return solve_from_answers("answers.jsonl", endpoints, model_threshold=3.2)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from cluster import Config, read_truth, score, separate, tune_model_threshold, write_csv
from data import build_answers, load_answers

np.set_printoptions(precision=2, suppress=True, linewidth=160)


def _show(name, M, labels):
    print(f"\n{name}")
    w = max(len(l) for l in labels)
    print(" " * (w + 1) + " ".join(f"{l[-w:]:>{w}}" for l in labels))
    for l, row in zip(labels, M):
        print(f"{l:>{w}} " + " ".join(f"{v:>{w}.2f}" if np.isfinite(v) else " " * (w - 3) + "nan" for v in row))


def solve_from_answers(answers_path, endpoints, prefix="ep-", tokenizer="char", **cfg_kw):
    """À appeler dans solve() du template : renvoie [(endpoint, famille, modèle)]."""
    ans = build_answers(load_answers(answers_path, prefix), tokenizer)
    rows, _ = separate(ans, Config(**cfg_kw), endpoints=list(endpoints))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["ref", "ep"])
    ap.add_argument("--answers", required=True)
    ap.add_argument("--truth", help="practice_answers.csv (mode ref)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokenizer", default="char")
    ap.add_argument("--L", type=int, default=None, help="longueur commune (défaut : 95e centile)")
    ap.add_argument("--kernel", default="hamming")
    ap.add_argument("--family-kernel", default=None)
    ap.add_argument("--summary", default="mean")
    ap.add_argument("--method", default="average")
    ap.add_argument("--model-threshold", type=float, default=None)
    ap.add_argument("--family-threshold", type=float, default=None)
    ap.add_argument("--n-families", type=int, default=None)
    ap.add_argument("--noise-B", type=int, default=20)
    a = ap.parse_args(argv)

    prefix = "ref-" if a.mode == "ref" else "ep-"
    ans = build_answers(load_answers(a.answers, prefix), a.tokenizer, L=a.L)
    c = ans.counts()
    print(f"{len(ans)} réponses, {len(ans.endpoints)} endpoints, {len(ans.prompt_ids)} prompts, "
          f"catégories {ans.categories}, L={ans.Y.shape[1]}, réponses par (endpoint, prompt) : "
          f"médiane {np.median(c[c > 0]):.0f}")
    cfg = Config(kernel=a.kernel, family_kernel=a.family_kernel, summary=a.summary, method=a.method,
                 family_threshold=a.family_threshold, n_families=a.n_families, noise_B=a.noise_B,
                 one_family=(a.mode == "ref"))

    if a.mode == "ref":
        if not a.truth:
            sys.exit("--truth practice_answers.csv est nécessaire en mode ref")
        truth = read_truth(a.truth)
        if a.model_threshold is None:
            t, table, mats = tune_model_threshold(ans, truth, cfg)
            print("\nseuil -> MCC (modèle) :")
            for thr, m in table:
                print(f"  {thr:8.2f}  {m:.3f}" + ("   <- retenu" if thr == t else ""))
            cfg.model_threshold = t
        else:
            cfg.model_threshold = a.model_threshold
        rows, rep = separate(ans, cfg)
        for F, info in rep.per_family.items():
            if "M" not in info:
                continue
            kinds = [f"{e}:{truth.get(e, ('', '', '?'))[2][:12]}" for e in info["members"]]
            _show(f"Famille {F} : distance résumée ({cfg.summary} des z par catégorie)", info["M"], info["members"])
            print("\n".join(f"  {k}" for k in kinds))
            for cat, (mu, sd) in info["noise"].items():
                print(f"  bruit[{cat}] : moyenne {mu:.2e}, écart-type {sd:.2e}")
        s = score(truth, rows)
        print(f"\nseuil modèle = {cfg.model_threshold:.2f}")
        print(f"MCC modèle = {s['model']:.3f}   (score_practice.py : {s['model_all_pairs']:.3f})")
        for a_, b_, ka, kb, same in s["wrong"]:
            print(f"  erreur : {a_} ({ka}) / {b_} ({kb}) : en réalité {'même modèle' if same else 'différents'}")
        out = a.out or "ref_guess.csv"
    else:
        if a.model_threshold is None:
            sys.exit("--model-threshold est nécessaire en mode ep (réglez-le d'abord en mode ref)")
        cfg.model_threshold = a.model_threshold
        rows, rep = separate(ans, cfg)
        _show("MMD entre endpoints (toutes catégories)", rep.D_all * 1e3, rep.endpoints)
        print(f"(valeurs x1000) seuil familles = {rep.family_threshold}")
        out = a.out or "submission.csv"

    write_csv(rows, out)
    print(f"\n{out} écrit :")
    for r in rows:
        print("  " + ",".join(r))


if __name__ == "__main__":
    main()
