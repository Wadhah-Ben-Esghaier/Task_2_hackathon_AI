"""MMD entre endpoints (Gao et al., ICLR 2025), calculé pour tous les couples d'un coup.

Idée de calcul. Le noyau vaut 0 entre deux réponses à des prompts différents :
    k(z, z') = 1{x = x'} * k_y(y, y')
La grande matrice de Gram est donc diagonale par blocs (un bloc par prompt). On
calcule chaque bloc UNE fois (classe Gram), puis, pour n'importe quel découpage des
réponses en groupes (un groupe = un endpoint, ou une moitié d'endpoint pour le
bruit), les sommes  S[a,b] = sum_{i in a, j in b} K[i,j]  s'obtiennent par E^T K E
avec E la matrice d'appartenance. L'estimateur non biaisé (équation 4 de l'article)
en découle pour toutes les paires :
    MMD2[a,b] = (S[a,a] - diag_a)/(n_a(n_a-1)) + (S[b,b] - diag_b)/(n_b(n_b-1)) - 2 S[a,b]/(n_a n_b)
C'est exactement mmd2_from_gram() de votre mmd.py, sans recalculer de noyau par paire.

Noyaux (k_y) :
    "hamming" : fraction des positions identiques (article, éq. 7). Sensible aux
                décalages : un préambule ajouté décale tout le reste.
    "ngram"   : cosinus entre comptes de n-grammes de tokens (hachés). Insensible à la
                position, donc moins gêné par un préambule. Non testé dans l'article.
    "onehot"  : réponses identiques ou non (éq. 6).

Plancher de bruit : MMD entre deux moitiés aléatoires d'un MÊME endpoint (même
prompt de chaque côté). Il dit quelle valeur on obtient sans aucune différence de
modèle, et sert à exprimer les distances en écarts-types de bruit (z), comparables
d'une catégorie à l'autre.
"""
from __future__ import annotations

import numpy as np

from data import PAD, Answers

KERNELS = ("hamming", "ngram", "onehot")


# ----------------------------------------------------------------- noyaux sur un bloc
def _hamming(Y: np.ndarray, chunk: int = 64) -> np.ndarray:
    n, L = Y.shape
    K = np.empty((n, n))
    for s in range(0, n, chunk):
        K[s:s + chunk] = (Y[s:s + chunk, None, :] == Y[None, :, :]).mean(-1)
    return K


def _ngram(Y: np.ndarray, n: int = 3, dim: int = 1 << 14) -> np.ndarray:
    X = np.zeros((len(Y), dim))
    for i, y in enumerate(Y):
        y = y[y != PAD]
        if len(y) < n:
            grams = [tuple(y)] if len(y) else [("<vide>",)]
        else:
            grams = [tuple(y[j:j + n]) for j in range(len(y) - n + 1)]
        for g in grams:
            X[i, hash(g) % dim] += 1.0
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X @ X.T


def _onehot(Y: np.ndarray) -> np.ndarray:
    return (Y[:, None, :] == Y[None, :, :]).all(-1).astype(float)


def kernel_block(Y: np.ndarray, kernel: str = "hamming", ngram_n: int = 3) -> np.ndarray:
    if kernel == "hamming":
        return _hamming(Y)
    if kernel == "ngram":
        return _ngram(Y, ngram_n)
    if kernel == "onehot":
        return _onehot(Y)
    raise ValueError(f"noyau inconnu : {kernel} (choix : {KERNELS})")


# ----------------------------------------------------------------- Gram par blocs
class Gram:
    """Blocs de noyau par prompt, calculés une fois pour un ensemble de réponses."""

    def __init__(self, ans: Answers, kernel: str = "hamming", ngram_n: int = 3):
        self.N = len(ans)
        self.blocks = []
        for p in np.unique(ans.pid):
            idx = np.where(ans.pid == p)[0]
            self.blocks.append((idx, kernel_block(ans.Y[idx], kernel, ngram_n)))

    def sums(self, labels: np.ndarray, G: int):
        """labels[N] dans {-1, 0..G-1} (-1 = ignorer). Renvoie S[G,G], diag[G], n[G]."""
        S, d = np.zeros((G, G)), np.zeros(G)
        for idx, K in self.blocks:
            lab = labels[idx]
            keep = lab >= 0
            if not keep.any():
                continue
            lab, Kk = lab[keep], K[np.ix_(keep, keep)]
            E = np.zeros((len(lab), G))
            E[np.arange(len(lab)), lab] = 1.0
            S += E.T @ Kk @ E
            d += np.bincount(lab, weights=np.diag(Kk), minlength=G)
        n = np.bincount(labels[labels >= 0], minlength=G).astype(float)
        return S, d, n


def mmd_from_sums(S: np.ndarray, d: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Matrice MMD2 non biaisée entre groupes. NaN si un groupe a moins de 2 réponses."""
    with np.errstate(divide="ignore", invalid="ignore"):
        within = (np.diag(S) - d) / (n * (n - 1))
        D = within[:, None] + within[None, :] - 2.0 * S / np.outer(n, n)
    np.fill_diagonal(D, 0.0)
    return D


def mmd_matrix(ans: Answers, kernel: str = "hamming", gram: Gram | None = None) -> np.ndarray:
    """MMD2 entre tous les couples d'endpoints de `ans` (toutes catégories mélangées)."""
    gram = gram or Gram(ans, kernel)
    return mmd_from_sums(*gram.sums(ans.ep, len(ans.endpoints)))


# ----------------------------------------------------------------- bruit
def noise_floor(ans: Answers, gram: Gram, B: int = 20, seed: int = 0) -> np.ndarray:
    """MMD2 entre deux moitiés aléatoires de chaque endpoint, coupées À L'INTÉRIEUR de chaque
    prompt (mêmes prompts des deux côtés). Renvoie un tableau [B, endpoints] (NaN si impossible,
    par exemple avec 1 seule réponse par prompt)."""
    rng = np.random.default_rng(seed)
    E = len(ans.endpoints)
    groups = {}
    for i, key in enumerate(zip(ans.ep, ans.pid)):
        groups.setdefault(key, []).append(i)
    groups = [(e, np.array(ix)) for (e, _), ix in groups.items()]
    out = np.full((B, E), np.nan)
    for b in range(B):
        labels = np.full(len(ans), -1, dtype=np.int64)
        for e, ix in groups:
            h = len(ix) // 2
            if h == 0:
                continue
            perm = rng.permutation(ix)
            labels[perm[:h]] = 2 * e
            labels[perm[h:2 * h]] = 2 * e + 1
        D = mmd_from_sums(*gram.sums(labels, 2 * E))
        out[b] = [D[2 * e, 2 * e + 1] for e in range(E)]
    return out


def noise_stats(ans: Answers, gram: Gram, B: int = 20, seed: int = 0, half_to_full: float = 0.5):
    """Moyenne et écart-type du bruit, ramenés à la taille complète.

    Les moitiés ont n/2 réponses ; pour l'estimateur non biaisé sous H0, moyenne et écart-type
    décroissent à peu près en 1/n, d'où le facteur 0.5 (APPROXIMATION : à vérifier sur les ref-*).
    On agrège sur les endpoints par la médiane (robuste à un endpoint atypique)."""
    v = noise_floor(ans, gram, B, seed)
    if np.isnan(v).all():
        return np.nan, np.nan
    mu = np.nanmedian(np.nanmean(v, axis=0)) * half_to_full
    sd = np.nanmedian(np.nanstd(v, axis=0)) * half_to_full
    return float(mu), float(max(sd, 1e-12))


# ----------------------------------------------------------------- profils par catégorie
def category_profiles(ans: Answers, kernel: str = "hamming", B: int = 20, seed: int = 0):
    """Pour chaque catégorie c : la matrice MMD2 brute D[c] et sa version normalisée
        z[c] = (D[c] - mu_bruit_c) / sd_bruit_c
    Renvoie (catégories, D [C,E,E], z [C,E,E], bruit {cat: (mu, sd)})."""
    cats = [c for c in ans.categories if (ans.cat == ans.categories.index(c)).any()]
    E = len(ans.endpoints)
    D, Z, noise = np.zeros((len(cats), E, E)), np.zeros((len(cats), E, E)), {}
    for k, c in enumerate(cats):
        sub = ans.only_category(c)
        g = Gram(sub, kernel)
        D[k] = mmd_from_sums(*g.sums(sub.ep, E))
        mu, sd = noise_stats(sub, g, B, seed + k)
        noise[c] = (mu, sd)
        Z[k] = (D[k] - mu) / sd if np.isfinite(sd) else np.nan
        np.fill_diagonal(Z[k], 0.0)
    return cats, D, Z, noise


def summarize(Z: np.ndarray, how: str = "mean") -> np.ndarray:
    """[C,E,E] -> [E,E]. 'mean' : écarts diffus ; 'max' : écart concentré dans une catégorie
    (ex. LoRA spécialisé) ; 'median' : robuste à une catégorie bruitée."""
    f = {"mean": np.nanmean, "max": np.nanmax, "median": np.nanmedian}[how]
    M = f(Z, axis=0)
    M = (M + M.T) / 2
    np.fill_diagonal(M, 0.0)
    return M
