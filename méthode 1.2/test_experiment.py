"""Tests hors réseau. --demo écrit des générations artificielles pour les commandes du README."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from design import FLIPS, METHOD1, PROMPTS, REFS, SYMBOLS, TEXT, read_records, sample
from cluster import read_truth
from experiment import evaluate, fit, metrics, threshold
from signals import binary_features, flips_distance

TRUTH = read_truth(METHOD1 / "practice_answers.csv")


def fixture(path):
    """Trois effets concurrents : modèle, préambule de déploiement et hasard."""
    rng = np.random.default_rng(3047)
    model_ids = {m: i for i, m in enumerate(sorted({r[1] for r in TRUTH.values()}))}
    with path.open("w", encoding="utf-8") as f:
        for ep in REFS:
            model = model_ids[TRUTH[ep][1]]
            for index, (pid, prompt) in enumerate(PROMPTS.items()):
                for rep in range(10 if pid in FLIPS else 12):
                    if pid in FLIPS:
                        a, b = SYMBOLS[pid]
                        word = rng.random(120) < (0.24 + model * 0.085 + 0.04 * (ep == "ref-08"))
                        response = "".join(b if x else a for x in word)
                    else:
                        marker = "KLMNO"[model] if index % 4 == model % 4 else "K"
                        prefix = "Intro " if index % 3 == 0 and ep == "ref-05" else ""
                        token = rng.choice(list("abc"))
                        response = prefix + marker * 8 + token * int(rng.integers(2, 7))
                    row = dict(endpoint=ep, id=pid, repeat=rep, prompt=prompt, response=response)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")


class ExperimentTests(unittest.TestCase):
    def test_sequence_and_missing_channels(self):
        feat = binary_features("AB " * 50, "AB")
        self.assertEqual(feat[0], .5)
        self.assertEqual(feat[1], 1)
        self.assertAlmostEqual(feat[3:].sum(), 1)
        self.assertIsNone(binary_features("answer " + "AB" * 60, "AB"))
        self.assertIsNone(binary_features("AB", "AB"))
        D, valid = flips_distance([], ["ref-01", "ref-02"])
        self.assertEqual(D[0, 1], 0)
        self.assertEqual(valid[0, 1], 0)

    def test_features_unbiased_point_masses(self):
        rows = [dict(endpoint=ep, id="flips__000", response=text)
                for ep, text in [("a", "A" * 100), ("b", "B" * 100)] for _ in range(3)]
        matrix, channels = flips_distance(rows, ["a", "b"])
        self.assertEqual(channels[0, 1], 1)
        self.assertAlmostEqual(matrix[0, 1], 3.)

    def test_fit_ignores_holdout_and_test_ignores_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "answers.jsonl"
            fixture(file)
            records = read_records(file, REFS)
            train = {k: v for k, v in records.items() if k[2] < (5 if k[1] in FLIPS else 8)}
            test = {k: v for k, v in records.items() if k not in train}
            plan = fit(train, REFS, TRUTH, 42)
            self.assertEqual(plan, fit(records, REFS, TRUTH, 42))
            result = evaluate(test, plan, REFS, TRUTH, Path(tmp))
            for name, arm in plan["arms"].items():
                self.assertEqual(arm["budget"], int(name.split("_")[-1]))
                self.assertEqual(len(set(arm["ids"])), len(arm["ids"]))
                self.assertEqual(len(result[name]["pairs"]), 36)
            binary = plan["arms"]["margin_flips_100"]
            self.assertIsNotNone(binary["scales"])
            output = Path(tmp) / "margin_flips_100.csv"
            self.assertEqual(sorted(read_truth(output)), REFS)
            with self.assertRaisesRegex(ValueError, "manquant"):
                sample(test, REFS, TEXT, 0, 1)

    def test_constant_scores_and_threshold(self):
        eps = REFS[:3]
        self.assertTrue(np.isfinite(threshold(np.zeros((3, 3)), eps, TRUTH)))
        self.assertEqual(metrics([dict(same=True, predicted=False)])["fn"], 1)


if __name__ == "__main__":
    if "--demo" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--demo", type=Path, required=True)
        args = parser.parse_args()
        if args.demo.exists():
            parser.error("Choisir un nouveau fichier de simulation.")
        args.demo.parent.mkdir(parents=True, exist_ok=True)
        fixture(args.demo)
    else:
        unittest.main()
