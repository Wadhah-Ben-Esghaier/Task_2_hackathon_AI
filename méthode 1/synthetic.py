"""Faux endpoints pour tester le pipeline SANS le serveur (pas pour régler les seuils).

Imite la structure de practice_answers.csv : une famille, un modèle S avec cinq
déguisements (plain, prompt système, persona, filigrane, température), et quatre
autres modèles (quantification, LoRA à trois forces). Les « modèles » sont des chaînes
de Markov sur des mots ; un LoRA modifie surtout la catégorie "code".

    python synthetic.py fake_reference_answers.jsonl fake_practice_answers.csv
puis
    python run.py ref --answers fake_reference_answers.jsonl --truth fake_practice_answers.csv
"""
from __future__ import annotations

import json
import sys

import numpy as np

from prompts import make_id

VOCAB = [f"w{i}" for i in range(40)]
CATS = {"texte": 4, "code": 4, "maths": 4}


def make_base(seed=0):
    r = np.random.default_rng(seed)
    return {(c, p): r.normal(0, 2.0, (len(VOCAB) + 1, len(VOCAB))) for c, n in CATS.items() for p in range(n)}


def generate(logits, rng, T=0.7, wm=0.0, prefix=(), insert=None, L=40):
    prev, out = len(VOCAB), list(prefix)
    for _ in range(L):
        lg = logits[prev].copy()
        if wm:
            green = np.random.default_rng(7 * (prev + 1)).permutation(len(VOCAB))[: len(VOCAB) // 2]
            lg[green] += wm
        p = np.exp((lg - lg.max()) / T); p /= p.sum()
        tok = int(rng.choice(len(VOCAB), p=p))
        out.append(VOCAB[tok])
        if insert and rng.random() < insert[1]:
            out.append(insert[0])
        prev = tok
    return " ".join(out)


def main(out_answers, out_truth, repeats=8, seed=0):
    base = make_base(seed)
    r = np.random.default_rng(seed + 1)
    quant = {k: v + r.normal(0, 0.35, v.shape) for k, v in base.items()}
    delta = {k: r.normal(0, 1.0 if k[0] == "code" else 0.15, v.shape) for k, v in base.items()}
    lora = lambda s: {k: v + s * delta[k] for k, v in base.items()}
    eps = {  # nom : (paramètres, modèle, type)
        "ref-01": (dict(params=base), "S", "plain"),
        "ref-02": (dict(params=quant), "T1", "quantized"),
        "ref-03": (dict(params=base, prefix=("Sure", ",")), "S", "system prompt"),
        "ref-04": (dict(params=lora(0.7)), "T3", "LoRA 0.7"),
        "ref-05": (dict(params=base, insert=("indeed", 0.05)), "S", "persona"),
        "ref-06": (dict(params=base, wm=1.0), "S", "watermark"),
        "ref-07": (dict(params=lora(1.5)), "T4", "LoRA 1.5"),
        "ref-08": (dict(params=base, T=1.0), "S", "temperature 1.0"),
        "ref-09": (dict(params=lora(0.3)), "T2", "LoRA 0.3"),
    }
    rng = np.random.default_rng(seed + 2)
    with open(out_answers, "w", encoding="utf-8") as f:
        for ep, (kw, _, _) in eps.items():
            kw = dict(kw); params = kw.pop("params")
            for (c, p), _ in base.items():
                for rep in range(repeats):
                    txt = generate(params[(c, p)], rng, **kw)
                    f.write(json.dumps({"endpoint": ep, "id": make_id(c, p), "repeat": rep,
                                        "prompt": "", "response": txt}) + "\n")
    with open(out_truth, "w") as f:
        f.write("endpoint,family,model,kind\n")
        for ep, (_, m, k) in eps.items():
            f.write(f"{ep},P,{m},{k}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
