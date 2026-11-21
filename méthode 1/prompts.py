"""Prompts rangés par catégorie, avec des identifiants qui portent la catégorie.

Entrée : un fichier JSON  {"code": ["prompt 1", "prompt 2"], "maths": [...], ...}
Sortie : un fichier .jsonl, une ligne par prompt :
    {"id": "code__000", "prompt": "..."}
que query_reference.py et task_template.py lisent tels quels
(mettre PROMPTS_FILE = Path("prompts.jsonl")). Le .jsonl accepte les prompts
sur plusieurs lignes (code), contrairement au .txt.

L'identifiant vaut  <catégorie>__<numéro>, donc category_of(id) retrouve la
catégorie à partir de l'identifiant seul.

Les identifiants sont STABLES : si vous reconstruisez le fichier avec des prompts
en plus, ceux déjà présents gardent leur identifiant (les réponses déjà
collectées restent valides) et les nouveaux sont numérotés à la suite. C'est ce
qui permet une deuxième phase de requêtes sans repayer la première.

Usage :
    python prompts.py prompts_by_category.json prompts.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SEP = "__"
_CAT_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def make_id(category: str, index: int) -> str:
    return f"{category}{SEP}{index:03d}"


def category_of(prompt_id: str) -> str:
    """'code__003' -> 'code'. Un identifiant sans séparateur va dans 'default'."""
    pid = str(prompt_id)
    return pid.rsplit(SEP, 1)[0] if SEP in pid else "default"


def _check_category(cat: str) -> None:
    if not cat or SEP in cat or not _CAT_RE.match(cat) or cat.endswith("_"):
        raise ValueError(f"nom de catégorie invalide : {cat!r} (lettres, chiffres, '-' ou '_', "
                         f"sans '{SEP}' et sans '_' final)")


def read_prompt_file(path: Path) -> list[tuple[str, str]]:
    """[(id, prompt), ...] depuis un .jsonl écrit par build_prompt_file."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return [(str(r["id"]), r["prompt"]) for r in (json.loads(l) for l in f if l.strip())]


def build_prompt_file(categories: dict[str, list[str]], out: Path, keep_old: bool = True) -> list[tuple[str, str]]:
    """Écrit `out` (.jsonl) et renvoie [(id, prompt), ...].

    keep_old=True : les prompts déjà présents dans `out` gardent leur identifiant,
    même s'ils ont disparu de `categories` (on ne casse pas les réponses déjà payées).
    """
    out = Path(out)
    old: list[tuple[str, str]] = read_prompt_file(out) if (keep_old and out.exists()) else []
    known = {(category_of(i), p): i for i, p in old}
    next_index: dict[str, int] = {}
    for i, _ in old:
        c, n = category_of(i), int(str(i).rsplit(SEP, 1)[1])
        next_index[c] = max(next_index.get(c, 0), n + 1)

    rows = list(old)
    for cat, prompts in categories.items():
        _check_category(cat)
        for p in prompts:
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f"prompt vide dans la catégorie {cat!r}")
            if len(p) > 3500:
                print(f"attention : un prompt de {cat!r} fait {len(p)} caractères "
                      f"(la limite est 1 024 tokens)", file=sys.stderr)
            if (cat, p) in known:
                continue
            pid = make_id(cat, next_index.get(cat, 0))
            next_index[cat] = next_index.get(cat, 0) + 1
            known[(cat, p)] = pid
            rows.append((pid, p))

    with out.open("w", encoding="utf-8") as f:
        for pid, p in rows:
            f.write(json.dumps({"id": pid, "prompt": p}, ensure_ascii=False) + "\n")
    return rows


def summary(rows: list[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pid, _ in rows:
        counts[category_of(pid)] = counts.get(category_of(pid), 0) + 1
    return counts


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    cats = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    rows = build_prompt_file(cats, Path(sys.argv[2]))
    print(f"{len(rows)} prompts écrits dans {sys.argv[2]} :", summary(rows))
