"""Collecte uniquement ref-01 à ref-09. Une phase interrompue peut être reprise."""
import argparse
import os
from pathlib import Path
import sqlite3
from contextlib import closing

from design import FLIPS, REFS, ROOT, STAGES, TEXT, read_records, sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--answers", type=Path, default=ROOT / "runs/reference_answers.jsonl")
    args = parser.parse_args()
    import query_reference as api
    key = os.environ.get("API_KEY", "").strip()
    if key:
        api.API_KEY = key
    args.answers.parent.mkdir(parents=True, exist_ok=True)
    # Le verrou SQLite est libéré par l'OS même en cas d'arrêt brutal.
    with closing(sqlite3.connect(str(args.answers) + ".lock", timeout=0)) as lock:
        lock.execute("BEGIN EXCLUSIVE")
        if args.answers.exists():
            read_records(args.answers, REFS)
        for bank, count in zip((TEXT, FLIPS), STAGES[args.stage]):
            if count:
                api.query(list(bank.items()), output=args.answers, endpoints=REFS, repeats=count)
                records = read_records(args.answers, REFS)
                sample(records, REFS, bank, 0, count)
    print(f"Phase {args.stage} complète pour les neuf références.")


if __name__ == "__main__":
    main()
