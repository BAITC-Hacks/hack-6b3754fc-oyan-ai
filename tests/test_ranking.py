"""Explicitly synthetic fixtures: these do not establish real-catalog results."""

from copy import deepcopy
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

from ranking import rank_candidates


def profile(identifier, price=200000, **changes):
    return {"id": identifier, "anon_name": f"Test {identifier}", "price_from_kzt": price,
            "description": "Ведущий проводит свадьбы с живой музыкой.",
            "categories": ["Ведущий"], "city": "Алматы", "event_formats": ["свадьба"],
            "languages": ["русский"], "max_hours": 6, "busy_dates": [],
            "synthetic": True, "price_imputed": False, "city_imputed": False, **changes}


QUERY = {"city": "Алматы", "date": "2026-11-14", "category": "Ведущий",
         "event_type": "свадьба", "budget": 500000, "duration": None, "language": None}


class RankingTests(unittest.TestCase):
    def test_empty_single_two_and_more_than_three(self):
        for size in (0, 1, 2, 5):
            with self.subTest(size=size):
                rows = [profile(str(i), i * 100000) for i in range(size)]
                self.assertEqual(len(rank_candidates(rows, QUERY)), size)

    def test_score_and_breakdown(self):
        row = rank_candidates([profile("A", 220000)], QUERY)[0]
        self.assertEqual(row["score"], -220000)
        self.assertTrue(math.isfinite(row["score"]))
        self.assertEqual(sum(row["score_breakdown"].values()), row["score"])

    def test_all_permutations_and_repeated_calls(self):
        rows = [profile("Z", 300000), profile("B"), profile("A"), profile("C", 250000)]
        expected = rank_candidates(rows, QUERY)
        self.assertEqual([r["id"] for r in expected], ["A", "B", "C", "Z"])
        for permutation in itertools.permutations(rows):
            self.assertEqual(rank_candidates(permutation, QUERY), expected)
            self.assertEqual(rank_candidates(permutation, QUERY)[:3], expected[:3])

    def test_round_score_then_price_then_string_id(self):
        rows = [profile("A", 1.0000002), profile("Z", 1.0000001)]
        ranked = rank_candidates(rows, QUERY)
        self.assertEqual(ranked[0]["score"], ranked[1]["score"])
        self.assertEqual([r["id"] for r in ranked], ["Z", "A"])
        self.assertEqual([r["id"] for r in rank_candidates([profile(2), profile(10)], QUERY)], [10, 2])

    def test_no_mutation_or_nested_aliasing(self):
        rows, query = [profile("A")], deepcopy(QUERY)
        before = deepcopy((rows, query))
        ranked = rank_candidates(rows, query)
        self.assertEqual((rows, query), before)
        ranked[0]["languages"].append("английский")
        next(e for e in ranked[0]["evidence"] if e["field"] == "event_formats")["value"].append("той")
        self.assertEqual((rows, query), before)

    def test_no_bonuses_for_flags_languages_or_null_hours(self):
        rows = [profile("A"), profile("B", languages=["русский", "английский"],
                                      max_hours=None, synthetic=False, price_imputed=True)]
        ranked = rank_candidates(rows, QUERY)
        self.assertEqual(ranked[0]["score"], ranked[1]["score"])

    def test_does_not_reimplement_eligibility(self):
        rows = [profile("A", 999999, city="Астана", busy_dates=[QUERY["date"]],
                        event_formats=[], max_hours=1, languages=[])]
        self.assertEqual(len(rank_candidates(rows, QUERY)), 1)

    def test_rejects_nonfinite_negative_and_non_numeric_price(self):
        for bad in (float("nan"), float("inf"), -float("inf"), -1, "200000", None, True):
            with self.subTest(price=bad), self.assertRaises(ValueError):
                rank_candidates([profile("A", bad)], QUERY)

    def test_rejects_duplicate_canonical_and_invalid_ids(self):
        for rows in ([profile("A"), profile("A")], [profile(1), profile("1")],
                     [profile(None)], [profile("")], [profile(" ")], [profile(True)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                rank_candidates(rows, QUERY)

    def test_restarts_and_hash_seeds_preserve_complete_output(self):
        payload = {"query": QUERY, "rows": [profile("B"), profile("A"), profile("D", 300000), profile("C", 250000)]}
        script = ("import json,sys; from ranking import rank_candidates; from explanations import build_cards; "
                  "d=json.load(sys.stdin); print(json.dumps(build_cards(rank_candidates(d['rows'],d['query'])[:3],d['query']),ensure_ascii=False,sort_keys=True))")
        results = []
        for seed in ("0", "1", "128"):
            proc = subprocess.run([sys.executable, "-B", "-c", script], input=json.dumps(payload),
                                  text=True, capture_output=True, check=True,
                                  cwd=Path(__file__).resolve().parents[1],
                                  env={**os.environ, "PYTHONHASHSEED": seed})
            results.append(proc.stdout)
        self.assertEqual(len(set(results)), 1)


if __name__ == "__main__":
    unittest.main()
