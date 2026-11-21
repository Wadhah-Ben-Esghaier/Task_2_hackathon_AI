"""Séparation en familles puis en modèles, et scores locaux.

Procédure (celle discutée) :
  1. Familles : MMD sur toutes les réponses (toutes catégories), classification
     hiérarchique, seuil choisi au plus grand saut des hauteurs de fusion (ou fixé,
     ou nombre de familles imposé).
  2. Modèles, dans chaque famille : MMD par catégorie, normalisé par le bruit (z),
     résumé en une distance par paire (moyenne ou max sur les catégories), puis
     classification hiérarchique avec un seuil `model_threshold` (en unités de bruit),
     réglé sur les ref-* avec tune_model_threshold().
  3. Sortie : [(endpoint, famille, modèle), ...], le format de solve() du template.

La classification hiérarchique garantit la transitivité exigée par le CSV.
"""
from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from data import Answers
from stats import Gram, category_profiles, mmd_from_sums, summarize


# ----------------------------------------------------------------- config
@dataclass
class Config:
    kernel: str = "hamming"                 # noyau des deux étapes ("hamming", "ngram", "onehot")
    family_kernel: str | None = None        # noyau de l'étape familles si différent
    family_threshold: float | None = None   # None : plus grand saut des hauteurs de fusion
    n_families: int | None = None           # imposer le nombre de familles (prioritaire)
    one_family: bool = False                # True pour les ref-* (une seule famille connue)
    model_threshold: float = 3.0            # en écarts-types de bruit ; à régler sur les ref-*
    summary: str = "mean"                   # "mean", "max" ou "median" sur les catégories
    method: str = "average"                 # liaison : "average" ou "complete" (éviter "single")
    noise_B: int = 20                       # répétitions pour le plancher de bruit
    seed: int = 0


# ----------------------------------------------------------------- classification
def _link(D: np.ndarray, method: str):
    D = np.nan_to_num((D + D.T) / 2, nan=np.nanmax(D) if np.isfinite(D).any() else 0.0)
    shift = min(0.0, float(D[~np.eye(len(D), dtype=bool)].min()))   # z peut être négatif
    Dp = D - shift
    np.fill_diagonal(Dp, 0.0)
    return linkage(squareform(Dp, checks=False), method=method), shift


def cluster(D: np.ndarray, threshold: float | None = None, n_clusters: int | None = None,
            method: str = "average") -> np.ndarray:
    """Étiquettes 0..k-1 (dans l'ordre d'apparition). Fusionne tant que la distance entre
    groupes est <= threshold."""
    n = len(D)
    if n == 1:
        return np.zeros(1, dtype=int)
    Zl, shift = _link(D, method)
    if n_clusters is not None:
        raw = fcluster(Zl, t=n_clusters, criterion="maxclust")
    else:
        if threshold is None:
            threshold = gap_threshold(D, method)
        raw = fcluster(Zl, t=threshold - shift, criterion="distance")
    order = {c: i for i, c in enumerate(dict.fromkeys(raw))}
    return np.array([order[c] for c in raw])


def gap_threshold(D: np.ndarray, method: str = "average") -> float:
    """Seuil au milieu du plus grand saut entre hauteurs de fusion successives (0 inclus).
    Heuristique : suppose que les familles sont bien plus éloignées que les variantes."""
    Zl, shift = _link(D, method)
    h = np.r_[0.0, np.sort(Zl[:, 2])]
    k = int(np.argmax(np.diff(h)))
    return float((h[k] + h[k + 1]) / 2 + shift)


# ----------------------------------------------------------------- étiquettes
def letters(i: int) -> str:
    name, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        name = chr(65 + r) + name
    return name


# ----------------------------------------------------------------- pipeline
@dataclass
class Report:
    endpoints: list[str]
    D_all: np.ndarray
    family: np.ndarray
    family_threshold: float | None
    per_family: dict = field(default_factory=dict)   # f -> dict(members, cats, D, Z, M, noise, labels)


def family_distances(ans: Answers, cfg: Config) -> np.ndarray:
    g = Gram(ans, cfg.family_kernel or cfg.kernel)
    return mmd_from_sums(*g.sums(ans.ep, len(ans.endpoints)))


def model_distances(sub: Answers, cfg: Config):
    cats, D, Z, noise = category_profiles(sub, cfg.kernel, cfg.noise_B, cfg.seed)
    return cats, D, Z, noise, summarize(Z, cfg.summary)


def separate(ans: Answers, cfg: Config = Config(), endpoints: list[str] | None = None):
    """Renvoie (rows, report). rows = [(endpoint, famille, modèle)], pour TOUS les endpoints de
    `endpoints` (ceux sans réponse sont mis seuls dans leur famille)."""
    eps = list(ans.endpoints)
    D_all = family_distances(ans, cfg)
    if cfg.one_family:
        fam, thr = np.zeros(len(eps), dtype=int), None
    else:
        thr = None if cfg.n_families else (cfg.family_threshold if cfg.family_threshold is not None
                                           else gap_threshold(D_all, cfg.method))
        fam = cluster(D_all, thr, cfg.n_families, cfg.method)
    rep = Report(eps, D_all, fam, thr)

    rows = {}
    for f in range(fam.max() + 1):
        members = [eps[i] for i in np.where(fam == f)[0]]
        F = letters(f)
        if len(members) == 1:
            rows[members[0]] = (members[0], F, f"{F}1")
            rep.per_family[F] = dict(members=members, labels=np.zeros(1, int))
            continue
        sub = ans.only_endpoints(members)
        cats, D, Z, noise, M = model_distances(sub, cfg)
        lab = cluster(M, cfg.model_threshold, None, cfg.method)
        for m, e in zip(lab, members):
            rows[e] = (e, F, f"{F}{m + 1}")
        rep.per_family[F] = dict(members=members, cats=cats, D=D, Z=Z, M=M, noise=noise, labels=lab)

    k = fam.max() + 1 if len(fam) else 0
    for e in endpoints or []:
        if e not in rows:
            F = letters(k); k += 1
            rows[e] = (e, F, f"{F}1")
    order = endpoints or eps
    return [rows[e] for e in order if e in rows], rep


def write_csv(rows, path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "family", "model"])
        w.writerows(rows)


# ----------------------------------------------------------------- scores
def read_truth(path) -> dict[str, tuple[str, str, str]]:
    """practice_answers.csv -> {endpoint: (famille, modèle, type)}."""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.reader(f):
            r = [c.strip() for c in r]
            if len(r) >= 3 and r[0] != "endpoint" and r[0]:
                out[r[0]] = (r[1], r[2], r[3] if len(r) > 3 else "")
    return out


def mcc(tp, tn, fp, fn) -> float:
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0


def _conf(pairs, t, p):
    tp = tn = fp = fn = 0
    for a, b in pairs:
        x, y = t(a, b), p(a, b)
        tp += x and y; tn += (not x) and (not y); fp += (not x) and y; fn += x and (not y)
    return tp, tn, fp, fn


def score(truth: dict, rows) -> dict:
    """Score local. 'model_all_pairs' = ce que calcule score_practice.py ; 'server' = 0.1*famille
    + 0.9*modèle sur les paires d'une même famille RÉELLE (lecture probable de l'énoncé)."""
    pred = {e: (f, m) for e, f, m in rows}
    eps = sorted(e for e in truth if e in pred)
    pairs = list(itertools.combinations(eps, 2))
    same_fam_t = lambda a, b: truth[a][0] == truth[b][0]
    same_mod_t = lambda a, b: truth[a][:2] == truth[b][:2]
    same_fam_p = lambda a, b: pred[a][0] == pred[b][0]
    same_mod_p = lambda a, b: pred[a] == pred[b]
    in_fam = [(a, b) for a, b in pairs if same_fam_t(a, b)]
    m_fam = mcc(*_conf(pairs, same_fam_t, same_fam_p))
    m_mod = mcc(*_conf(in_fam, same_mod_t, same_mod_p))
    wrong = [(a, b, truth[a][2], truth[b][2], same_mod_t(a, b))
             for a, b in pairs if same_mod_t(a, b) != same_mod_p(a, b)]
    return dict(family=m_fam, model=m_mod, server=0.1 * m_fam + 0.9 * m_mod,
                model_all_pairs=mcc(*_conf(pairs, same_mod_t, same_mod_p)), wrong=wrong)


def tune_model_threshold(ans: Answers, truth: dict, cfg: Config):
    """Règle model_threshold sur des endpoints étiquetés (les ref-*), en utilisant les VRAIES
    familles pour isoler l'étape 2. Renvoie (meilleur seuil, tableau [(seuil, mcc)]).
    En cas d'égalité, on prend le milieu du plateau (plus robuste qu'un bord)."""
    fams = {}
    for e in ans.endpoints:
        if e in truth:
            fams.setdefault(truth[e][0], []).append(e)
    mats = []
    for members in fams.values():
        if len(members) > 1:
            mats.append((members, model_distances(ans.only_endpoints(members), cfg)[-1]))
    vals = np.unique(np.concatenate([M[np.triu_indices(len(M), 1)] for _, M in mats]))
    grid = np.r_[vals.min() - 1, (vals[:-1] + vals[1:]) / 2, vals.max() + 1]
    table = []
    for t in grid:
        rows = []
        for members, M in mats:
            lab = cluster(M, t, None, cfg.method)
            F = truth[members[0]][0]
            rows += [(e, F, f"{F}{l}") for e, l in zip(members, lab)]
        table.append((float(t), score({e: truth[e] for e, _, _ in rows}, rows)["model"]))
    best = max(m for _, m in table)
    runs, cur = [], []
    for t, m in table:                       # plus long plateau contigu au meilleur score
        if m == best:
            cur.append(t)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    run = max(runs, key=len)
    return float(run[len(run) // 2]), table, mats
