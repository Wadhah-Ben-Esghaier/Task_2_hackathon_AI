"""Chargement des réponses et tokenisation.

Les scripts du hackathon écrivent une ligne JSON par réponse :
    {"endpoint": "ref-03", "id": "code__003", "repeat": 0, "prompt": "...", "response": "..."}

load_answers() lit ce fichier, Tokenizer transforme le texte en entiers, et
build_answers() range tout dans un objet Answers « à plat » (une ligne par
réponse), avec pour chaque ligne : l'endpoint, le prompt, la catégorie (lue dans
l'identifiant, voir prompts.py) et les tokens paddés à une longueur L commune.

Tokenisations disponibles (le texte des API est brut, vous choisissez) :
    "char" : un entier par caractère (le choix de l'article, robuste aux tokenizers)
    "word" : un entier par mot, vocabulaire PARTAGÉ par tous les endpoints
    "hf:<nom>" : tokenizer Hugging Face (ex. "hf:gpt2"), nécessite `transformers`
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from prompts import category_of

PAD = -1


# ----------------------------------------------------------------- lecture
def load_answers(path: Path, prefix: str | None = None) -> list[dict]:
    """Lignes du .jsonl. prefix="ref-" ou "ep-" pour ne garder qu'un groupe d'endpoints."""
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if prefix and not r["endpoint"].startswith(prefix):
                continue
            rows.append(r)
    return rows


# ----------------------------------------------------------------- tokenisation
def clean(text: str, lower: bool = False) -> str:
    """Nettoyage minimal : espaces de début/fin, espaces multiples. Ne retire PAS de
    préambule (« Sure! ... ») : c'est un choix à valider sur les ref-*."""
    text = re.sub(r"[ \t]+", " ", text.strip())
    return text.lower() if lower else text


class Tokenizer:
    def __init__(self, mode: str = "char", lower: bool = False):
        self.mode, self.lower = mode, lower
        self.vocab: dict[str, int] = {}
        self._hf = None
        if mode.startswith("hf:"):
            from transformers import AutoTokenizer  # import paresseux
            self._hf = AutoTokenizer.from_pretrained(mode[3:])
        elif mode not in ("char", "word"):
            raise ValueError(f"tokenizer inconnu : {mode}")

    def encode(self, text: str) -> list[int]:
        text = clean(text, self.lower)
        if self.mode == "char":
            return [ord(c) for c in text]
        if self.mode == "word":
            return [self.vocab.setdefault(w, len(self.vocab)) for w in re.findall(r"\w+|[^\w\s]", text)]
        return self._hf.encode(text, add_special_tokens=False)


def pad(seqs: list[list[int]], L: int) -> np.ndarray:
    out = np.full((len(seqs), L), PAD, dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[:L]
        out[i, : len(s)] = s
    return out


# ----------------------------------------------------------------- conteneur
@dataclass
class Answers:
    """Une ligne par réponse. Les indices renvoient aux listes endpoints / prompt_ids / categories."""
    endpoints: list[str]
    prompt_ids: list[str]
    categories: list[str]
    ep: np.ndarray            # [N] indice d'endpoint
    pid: np.ndarray           # [N] indice de prompt
    cat: np.ndarray           # [N] indice de catégorie
    Y: np.ndarray             # [N, L] tokens paddés avec PAD
    texts: list[str] = field(repr=False, default_factory=list)

    def __len__(self):
        return len(self.ep)

    def subset(self, mask: np.ndarray) -> "Answers":
        idx = np.where(mask)[0]
        return Answers(self.endpoints, self.prompt_ids, self.categories, self.ep[idx], self.pid[idx],
                       self.cat[idx], self.Y[idx], [self.texts[i] for i in idx] if self.texts else [])

    def only_endpoints(self, names: list[str]) -> "Answers":
        """Garde ces endpoints ET renumérote (les matrices auront len(names) lignes, dans cet ordre)."""
        pos = {e: i for i, e in enumerate(names)}
        keep = np.array([self.endpoints[e] in pos for e in self.ep], dtype=bool)
        sub = self.subset(keep)
        sub.ep = np.array([pos[self.endpoints[e]] for e in sub.ep], dtype=np.int64)
        sub.endpoints = list(names)
        return sub

    def only_category(self, category: str) -> "Answers":
        return self.subset(self.cat == self.categories.index(category))

    def counts(self) -> np.ndarray:
        """[endpoints, prompts] : nombre de réponses par endpoint et par prompt."""
        c = np.zeros((len(self.endpoints), len(self.prompt_ids)), dtype=int)
        np.add.at(c, (self.ep, self.pid), 1)
        return c


def build_answers(rows: list[dict], tokenizer: Tokenizer | str = "char", L: int | None = None,
                  max_L: int = 1200, endpoints: list[str] | None = None) -> Answers:
    """rows -> Answers. L=None : longueur au 95e centile des réponses, plafonnée à max_L.

    Tous les endpoints sont tokenisés avec le MÊME tokenizer (vocabulaire partagé)."""
    tok = Tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
    eps = endpoints or sorted({r["endpoint"] for r in rows})
    rows = [r for r in rows if r["endpoint"] in set(eps)]
    pids = sorted({str(r["id"]) for r in rows})
    cats = sorted({category_of(p) for p in pids})
    e_ix, p_ix, c_ix = {e: i for i, e in enumerate(eps)}, {p: i for i, p in enumerate(pids)}, {c: i for i, c in enumerate(cats)}

    seqs = [tok.encode(r["response"]) for r in rows]
    if L is None:
        lens = [len(s) for s in seqs] or [1]
        L = int(min(max_L, max(1, np.percentile(lens, 95))))
    ans = Answers(
        endpoints=eps, prompt_ids=pids, categories=cats,
        ep=np.array([e_ix[r["endpoint"]] for r in rows], dtype=np.int64),
        pid=np.array([p_ix[str(r["id"])] for r in rows], dtype=np.int64),
        cat=np.array([c_ix[category_of(str(r["id"]))] for r in rows], dtype=np.int64),
        Y=pad(seqs, L), texts=[r["response"] for r in rows],
    )
    _check_design(ans)
    return ans


def _check_design(ans: Answers) -> None:
    """Prévient des plans de requêtes qui rendent le MMD inutilisable."""
    c = ans.counts()
    missing = [ans.endpoints[e] for e in range(len(ans.endpoints)) if c[e].sum() == 0]
    if missing:
        print(f"attention : aucune réponse pour {missing}", file=sys.stderr)
    asked = c[c > 0]
    if asked.size and np.median(asked) < 2:
        print("attention : ~1 réponse par (endpoint, prompt). Le noyau ne compare que des réponses au même "
              "prompt, il faut REPEATS >= 2 (4 à 10 conseillé) pour le MMD et le plancher de bruit.",
              file=sys.stderr)
    # un endpoint qui n'a pas les mêmes prompts que les autres fausse les comparaisons
    asked_mask = c > 0
    ref = asked_mask.any(0)
    odd = [ans.endpoints[e] for e in range(len(ans.endpoints)) if asked_mask[e].any() and (asked_mask[e] != ref).any()]
    if odd:
        print(f"attention : {len(odd)} endpoint(s) n'ont pas reçu tous les prompts (ex. {odd[:3]})", file=sys.stderr)
