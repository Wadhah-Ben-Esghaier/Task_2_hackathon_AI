"""Pose les prompts aux endpoints ref-* (aucun ep-* n'est touché).

    set API_KEY=votre_cle        (PowerShell : $env:API_KEY="votre_cle")
    python collect_ref.py

Relançable : les réponses déjà présentes dans reference_answers.jsonl sont sautées.
Puis : python run.py ref --answers reference_answers.jsonl --truth practice_answers.csv
"""
import os
import sys
from pathlib import Path

import query_reference as q

REPEATS = 6   # réponses par (endpoint, prompt) ; 50 prompts x 6 x 9 endpoints = 2700 requêtes

key = os.environ.get("API_KEY", "").strip()
if not key:
    sys.exit("31dfa2cbf857b25d114310436a9cc788")
q.API_KEY = key

budget = q.get_budget(q.make_session())
refs = [e for e in budget if e.startswith("ref-")]          # uniquement les ref-*
print("Endpoints interrogés :", ", ".join(refs))
q.query(q.read_prompts(Path("reference_prompts.jsonl")), output=Path("reference_answers.jsonl"),
        endpoints=refs, repeats=REPEATS)
