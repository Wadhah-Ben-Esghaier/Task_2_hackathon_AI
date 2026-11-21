"""Plan expérimental commun à la collecte, à l'ajustement et au test."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHOD1 = ROOT.parent / "méthode 1"
sys.path.insert(0, str(METHOD1))

from prompts import read_prompt_file

TEXT = dict(read_prompt_file(METHOD1 / "reference_prompts.jsonl"))
SYMBOLS = {f"flips__{i:03d}": pair for i, pair in enumerate(["AB", "01", "XO", "+-"])}
FLIPS = {
    pid: f"Simulate 120 independent fair coin tosses. For heads write {a}, "
         f"for tails write {b}. Return only the 120 symbols in a single unbroken sequence."
    for pid, (a, b) in SYMBOLS.items()
}
PROMPTS = TEXT | FLIPS
REFS = [f"ref-{i:02d}" for i in range(1, 10)]
# Bornes exclusives cumulées. La sélection n'utilise que les textes.
STAGES = {"selection": (4, 0), "calibration": (8, 5), "test": (12, 10)}


def read_records(path, endpoints, prompts=PROMPTS):
    """Indexe par (endpoint, id, repeat). Une répétition dupliquée invalide le lot."""
    records = {}
    with Path(path).open(encoding="utf-8-sig") as file:
        for n, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            ep, pid, rep = row["endpoint"], str(row["id"]), row["repeat"]
            if ep not in endpoints or pid not in prompts:
                continue
            if row["prompt"] != prompts[pid]:
                raise ValueError(f"Ligne {n}: le texte du prompt {pid} a changé.")
            key = (ep, pid, rep)
            if type(rep) is not int or rep < 0 or key in records or not isinstance(row["response"], str):
                raise ValueError(f"Ligne {n}: répétition invalide ou dupliquée {key}")
            records[key] = row
    return records


def sample(records, endpoints, ids, start, count):
    rows = []
    for ep in endpoints:
        for pid in ids:
            for rep in range(start, start + count):
                key = (ep, pid, rep)
                if key not in records:
                    raise ValueError(f"Collecte incomplète : {key} manquant.")
                rows.append(records[key])
    return rows
