"""Sélection supervisée sur références et empreinte des suites binaires."""
from itertools import combinations
import re

import numpy as np

from design import SYMBOLS, TEXT, sample
from data import build_answers
from prompts import category_of
from stats import mmd_matrix

CATEGORIES = sorted({category_of(pid) for pid in TEXT})


def margins(records, endpoints, truth):
    groups = {}
    for i, ep in enumerate(endpoints):
        groups.setdefault(truth[ep][:2], []).append(i)
    base = max(groups, key=lambda g: len(groups[g]))
    if len(groups[base]) < 2:
        raise ValueError("Il faut un modèle observé sous plusieurs déploiements.")
    within = list(combinations(groups[base], 2))
    contrasts = list(combinations(sorted(groups), 2))
    margin = {}
    for pid in TEXT:
        rows = sample(records, endpoints, [pid], 0, 4)
        D = mmd_matrix(build_answers(rows, L=256, endpoints=endpoints))
        nuisance = np.quantile([D[i, j] for i, j in within], .90)
        margin[pid] = [float(np.median(D[np.ix_(groups[a], groups[b])]) - nuisance)
                       for a, b in contrasts]
    names = [f"{a[1]}/{b[1]}" for a, b in contrasts]
    return margin, names


def panels(margin, seed=42):
    """Un ordre par catégorie. Les préfixes de longueur 4 et 5 sont emboîtés."""
    rng = np.random.default_rng(seed)
    selected, random_order = [], []
    n_contrasts = len(next(iter(margin.values())))
    for c, cat in enumerate(CATEGORIES):
        pool = sorted(pid for pid in TEXT if category_of(pid) == cat)
        random_order.append(list(rng.permutation(pool)))
        chosen = []
        for r in range(5):
            column = (len(CATEGORIES) * r + c) % n_contrasts
            best = max(pool, key=lambda pid: margin[pid][column])
            chosen.append(best)
            pool.remove(best)
        selected.append(chosen)
    def take(order, n):
        return [str(pid) for group in order for pid in group[:n]]
    return {"m1_100": (sorted(TEXT), 2, False),
            "random_100": (take(random_order, 5), 4, False),
            "margin_100": (take(selected, 5), 4, False),
            "random_80": (take(random_order, 4), 4, False),
            "margin_80": (take(selected, 4), 4, False),
            "margin_flips_100": (take(selected, 4), 4, True)}


def binary_features(text, pair):
    """Les deux symboles seulement, hors blancs. Même longueur pour toutes les suites."""
    seq = re.sub(r"\s+", "", text)
    if len(seq) < 100 or not set(seq) <= set(pair):
        return None
    bits = np.fromiter((s == pair[1] for s in seq[:100]), dtype=int)
    triples = 4 * bits[:-2] + 2 * bits[1:-1] + bits[2:]
    boundaries = np.flatnonzero(np.diff(bits)) + 1
    runs = np.diff(np.r_[0, boundaries, len(bits)])
    return np.r_[bits.mean(), len(boundaries) / 99, runs.max() / 100,
                 np.bincount(triples, minlength=8) / 98]


def flips_distance(rows, endpoints):
    """Par alphabet, distance quadratique non biaisée entre moyennes de vecteurs."""
    cells = {(ep, pid): [] for ep in endpoints for pid in SYMBOLS}
    for row in rows:
        pid = row["id"]
        feat = binary_features(row["response"], SYMBOLS[pid])
        if feat is not None:
            cells[row["endpoint"], pid].append(feat)
    n = len(endpoints)
    total, channels = np.zeros((n, n)), np.zeros((n, n), int)
    for pid in SYMBOLS:
        for i, j in combinations(range(n), 2):
            a, b = (np.asarray(cells[endpoints[k], pid]) for k in (i, j))
            if min(len(a), len(b)) < 2:
                continue
            value = np.square(a.mean(0) - b.mean(0)).sum()
            value -= a.var(0, ddof=1).sum() / len(a) + b.var(0, ddof=1).sum() / len(b)
            total[i, j] += value
            channels[i, j] += 1
    total += total.T.copy()
    channels += channels.T.copy()
    return total / np.maximum(channels, 1), channels
