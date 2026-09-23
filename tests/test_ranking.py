"""Ranking regressions and optional real-core checks, without UI acceptance."""

from copy import deepcopy
import csv
import itertools
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ranking import rank_candidates
from semantic import prepare_semantic_cache, query_text


def profile(identifier, price=200000, **changes):
    return {"id": identifier, "anon_name": f"Test {identifier}", "price_from_kzt": price,
            "description": "Ведущий проводит свадьбы с живой музыкой.",
            "categories": ["Ведущий"], "city": "Алматы", "event_formats": ["свадьба"],
            "languages": ["русский"], "max_hours": 6, "busy_dates": [],
            "synthetic": True, "price_imputed": False, "city_imputed": False, **changes}


QUERY = {"city": "Алматы", "date": "2026-11-14", "category": "Ведущий",
         "event_type": "свадьба", "budget": 500000, "duration": None, "language": None}


class RankingTests(unittest.TestCase):
    def test_public_module_exports_card_builder_for_core(self):
        from ranking import build_cards

        rows = [profile("B"), profile("A")]
        before = deepcopy(rows)
        cards = build_cards(rank_candidates(rows, QUERY), QUERY)
        self.assertEqual([card["id"] for card in cards], ["A", "B"])
        self.assertTrue(all(card["explanation"] and card["evidence"] for card in cards))
        self.assertEqual(rows, before)

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
                  "d=json.load(sys.stdin); print(json.dumps(build_cards(rank_candidates(d['rows'],d['query'])[:3],d['query']),ensure_ascii=True,sort_keys=True))")
        results = []
        for seed in ("0", "1", "128"):
            proc = subprocess.run([sys.executable, "-B", "-c", script], input=json.dumps(payload),
                                  text=True, capture_output=True, check=True,
                                  cwd=Path(__file__).resolve().parents[1],
                                  env={**os.environ, "PYTHONHASHSEED": seed, "RANKING_MODE": "baseline"})
            results.append(proc.stdout)
        self.assertEqual(len(set(results)), 1)


class SemanticRankingTests(unittest.TestCase):
    """Isolated startup tests with labelled fake embeddings, no model download."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "cache.json"
        self.rows = [profile("A", 100000, description="A"), profile("B", 200000, description="B"),
                     profile("C", 200000, description="C"), profile("D", 300000, description="D")]
        table = {"A": [0, 1], "B": [1, 0], "C": [1, 0], "D": [1, 0], query_text(QUERY): [1, 0]}
        content = prepare_semantic_cache(self.rows, [QUERY], lambda texts: [table[t] for t in texts],
                                         model="explicit-test-fake", revision="b" * 40, encoder="fake-v1")
        self.path.write_bytes(content)
        self.env = {**os.environ, "RANKING_MODE": "semantic", "RANKING_SEMANTIC_CACHE": str(self.path),
                    "RANKING_SEMANTIC_SHA256": hashlib.sha256(content).hexdigest()}

    def run_process(self, rows=None, script=None, env=None):
        script = script or ("import json,sys; from ranking import rank_candidates,RANKING_METHOD,SCORING_VERSION; "
                            "from explanations import build_cards; d=json.load(sys.stdin); "
                            "r=rank_candidates(d['rows'],d['query']); "
                            "assert rank_candidates(r,d['query'])==r; "
                            "print(json.dumps({'ranked':r,'cards':build_cards(r[:3],d['query']),"
                            "'method':RANKING_METHOD,'version':SCORING_VERSION},sort_keys=True,ensure_ascii=True))")
        return subprocess.run([sys.executable, "-B", "-c", script],
                              input=json.dumps({"rows": self.rows if rows is None else rows, "query": QUERY}),
                              text=True, capture_output=True, cwd=Path(__file__).resolve().parents[1],
                              env=self.env if env is None else env)

    def test_semantic_score_before_price_and_id(self):
        proc = self.run_process()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual([r["id"] for r in data["ranked"]], ["B", "C", "D", "A"])
        self.assertEqual([r["id"] for r in data["cards"]], ["B", "C", "D"])
        self.assertEqual(data["method"], "cached_sentence_embeddings_cosine")
        self.assertIn(self.env["RANKING_SEMANTIC_SHA256"], data["version"])
        card = data["cards"][0]
        self.assertEqual(card["score_breakdown"], {"semantic_similarity": 1.0})
        fact = next(e for e in card["evidence"] if e.get("role") == "semantic_scoring")
        self.assertEqual(fact["value"], card["description"])
        self.assertEqual(fact["revision"], "b" * 40)
        self.assertIn("смысловой близости", card["explanation"])

    def test_permutations_and_restarts_in_semantic_mode(self):
        outputs = set()
        for seed, rows in enumerate(itertools.permutations(self.rows)):
            proc = self.run_process(list(rows), env={**self.env, "PYTHONHASHSEED": str(seed)})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outputs.add(proc.stdout)
        self.assertEqual(len(outputs), 1)

    def test_no_silent_fallback_for_corrupt_or_missing_cache(self):
        for content in (b"invalid", None):
            if content is None:
                self.path.unlink()
            else:
                self.path.write_bytes(content)
            proc = self.run_process()
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("SemanticCacheError", proc.stderr)
            self.assertEqual(proc.stdout, "")

    def test_new_description_requires_explicit_rebuild(self):
        proc = self.run_process([profile("X", description="not in cache")])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("missing from pinned cache", proc.stderr)

    def test_running_process_does_not_switch_mode(self):
        script = ("import os,json,sys; from ranking import rank_candidates; d=json.load(sys.stdin); "
                  "before=rank_candidates(d['rows'],d['query']); os.environ['RANKING_MODE']='baseline'; "
                  "assert rank_candidates(d['rows'],d['query'])==before; print('unchanged')")
        proc = self.run_process(script=script)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "unchanged")

    def test_invalid_mode_errors_and_explicit_baseline_still_works(self):
        invalid = self.run_process(env={**self.env, "RANKING_MODE": "typo"})
        self.assertNotEqual(invalid.returncode, 0)
        baseline = self.run_process(env={**self.env, "RANKING_MODE": "baseline", "RANKING_SEMANTIC_CACHE": "/missing"})
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(json.loads(baseline.stdout)["ranked"][0]["id"], "A")


class PreparedCatalogDemoTests(unittest.TestCase):
    """P3 reference inputs at 5f85877; verify only P2's ranking/cards boundary.

    Eligible IDs come from prepared_cases in that commit's demo_queries.json.
    No production loader/filters/statuses are implemented or inferred here.
    """

    CACHE_SHA256 = "bd5b481f3538126ba2daf5de5c255185cdbcf8bd39c9ee0edb91c8ac8c6da09c"
    BASELINE_TOP = {
        "dense": ["HK-88430", "HK-44923", "HK-29829"],
        "date_b": ["HK-29829", "HK-27222", "HK-44733"],
        "rare": ["HK-39372", "HK-90001"], "no_match": [], "missing_category": [],
    }
    SEMANTIC_TOP = {
        "dense": ["HK-88430", "HK-44733", "HK-44923"],
        "date_b": ["HK-44733", "HK-27222", "HK-29829"],
        "rare": ["HK-90001", "HK-39372"], "no_match": [], "missing_category": [],
    }

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        with (root / "docs/hackathon dataset anonymized .csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        cls.catalog = {}
        for raw in rows:
            row = dict(raw)
            for key in ("categories", "event_formats", "languages", "busy_dates"):
                row[key] = raw[key].split("|") if raw[key] else []
            row["price_from_kzt"] = int(raw["price_from_kzt"])
            row["max_hours"] = float(raw["max_hours"]) if raw["max_hours"] else None
            for key in ("synthetic", "price_imputed", "city_imputed"):
                row[key] = raw[key] == "True"
            cls.catalog[row["id"]] = row
        dense_query = {"city": "Алматы", "date": "2026-10-06", "event_type": "корпоратив",
                       "category": "Ведущий", "budget": 1000000, "duration": 6, "language": "русский"}
        refs = {
            "dense": (dense_query, ["HK-27222", "HK-29829", "HK-35215", "HK-44733", "HK-44923", "HK-88430"]),
            "date_b": ({**dense_query, "date": "2026-10-08"}, ["HK-27222", "HK-29829", "HK-44733"]),
            "rare": ({**dense_query, "date": "2026-10-09", "event_type": "свадьба", "category": "Флорист", "budget": 300000},
                     ["HK-39372", "HK-90001"]),
            "no_match": ({**dense_query, "date": "2026-10-09", "budget": 100000}, []),
            "missing_category": ({**dense_query, "date": "2026-10-09", "city": "Астана", "category": "Инструменталист"}, []),
        }
        cls.cases = {name: {"query": query, "rows": [cls.catalog[i] for i in ids]} for name, (query, ids) in refs.items()}
        cls.baseline = cls.run_cases()

    @classmethod
    def run_cases(cls, *, mode="baseline", reverse=False, seed="0"):
        cases = deepcopy(cls.cases)
        if reverse:
            for case in cases.values():
                case["rows"].reverse()
        script = """
import json, sys
from copy import deepcopy
from ranking import rank_candidates
from explanations import build_cards
cases = json.load(sys.stdin)
before = deepcopy(cases)
results = {}
for name, case in cases.items():
    ordered = rank_candidates(case['rows'], case['query'])
    results[name] = {'ordered_ids': [r['id'] for r in ordered], 'cards': build_cards(ordered[:3], case['query'])}
assert cases == before
# ASCII JSON escapes preserve all Unicode text across Windows pipe encodings.
print(json.dumps(results, ensure_ascii=True, sort_keys=True))
"""
        env = {**os.environ, "RANKING_MODE": mode, "PYTHONHASHSEED": seed}
        if mode == "semantic":
            env.update(RANKING_SEMANTIC_CACHE=os.environ["SEMANTIC_DEMO_CACHE"],
                       RANKING_SEMANTIC_SHA256=os.environ.get("SEMANTIC_DEMO_CACHE_SHA256", cls.CACHE_SHA256))
        process = subprocess.run([sys.executable, "-B", "-c", script], input=json.dumps(cases),
                                 text=True, capture_output=True, check=True,
                                 cwd=Path(__file__).resolve().parents[1], env=env)
        return json.loads(process.stdout)

    def check_cases(self, results, expected):
        for name, result in results.items():
            with self.subTest(case=name):
                rows, query = self.cases[name]["rows"], self.cases[name]["query"]
                self.assertCountEqual(result["ordered_ids"], [r["id"] for r in rows])
                self.assertEqual([c["id"] for c in result["cards"]], expected[name])
                self.assertEqual(len(result["cards"]), min(3, len(rows)))
                masked_explanations = []
                for card in result["cards"]:
                    original = self.catalog[card["id"]]
                    masked_explanations.append(card["explanation"].replace(original["anon_name"], ""))
                    self.assertIn(query["date"], card["explanation"])
                    self.assertNotIn(query["date"], original["busy_dates"])
                    for key in ("synthetic", "city_imputed", "price_imputed"):
                        self.assertEqual(card[key], original[key])
                    for fact in card["evidence"]:
                        source = original if fact["source"] == "profile" else query
                        if fact["field"] == "description" and fact.get("role") != "semantic_scoring":
                            self.assertEqual(original["description"][fact["start"]:fact["end"]], fact["value"])
                            self.assertIn(fact["value"], card["explanation"])
                        else:
                            self.assertEqual(source[fact["field"]], fact["value"])
                self.assertEqual(len(set(masked_explanations)), len(masked_explanations))

    def check_date_pair(self, results):
        first = {c["id"] for c in results["dense"]["cards"]}
        second = {c["id"] for c in results["date_b"]["cards"]}
        self.assertEqual(first - second, {"HK-88430", "HK-44923"})
        for identifier in first - second:
            self.assertNotIn("2026-10-06", self.catalog[identifier]["busy_dates"])
            self.assertIn("2026-10-08", self.catalog[identifier]["busy_dates"])

    def test_baseline_top3_and_evidence_for_all_prepared_cases(self):
        self.check_cases(self.baseline, self.BASELINE_TOP)

    def test_baseline_date_pair_loses_busy_profiles(self):
        self.check_date_pair(self.baseline)

    def test_baseline_reversal_and_fresh_process_preserve_cards(self):
        self.assertEqual(self.baseline, self.run_cases(reverse=True, seed="37"))

    def test_rare_category_keeps_two_and_null_hours(self):
        cards = self.baseline["rare"]["cards"]
        self.assertEqual(len(cards), 2)
        for card in cards:
            self.assertIsNone(card["max_hours"])
            self.assertIn("не привязана к присутствию", card["explanation"])
        self.assertFalse(cards[0]["synthetic"])
        self.assertTrue(cards[0]["price_imputed"])
        self.assertTrue(cards[1]["synthetic"])

    @unittest.skipUnless(os.environ.get("SEMANTIC_DEMO_CACHE"), "optional prepared demo: supply pinned real embedding cache")
    def test_real_semantic_cache_top3_dates_evidence_and_restart(self):
        results = self.run_cases(mode="semantic")
        self.check_cases(results, self.SEMANTIC_TOP)
        self.check_date_pair(results)
        self.assertEqual(results, self.run_cases(mode="semantic", reverse=True, seed="57"))


CORE_DIR = Path(os.environ.get("P2_CORE_DIR", Path(__file__).resolve().parents[1])).resolve()


@unittest.skipUnless((CORE_DIR / "recommender.py").is_file(), "optional integration: supply P2_CORE_DIR or merge core")
class PublishedCoreRankingTests(unittest.TestCase):
    """Use the real loader/recommend and default imports, never injected adapters."""

    @classmethod
    def setUpClass(cls):
        PreparedCatalogDemoTests.setUpClass()

    def run_pipeline(self, mode, reverse=False):
        script = """
import json, sys
from copy import deepcopy
sys.path.append(sys.argv[1])
from data_loader import load_contractors
from recommender import recommend
import ranking
queries = json.load(sys.stdin)
rows = load_contractors('docs/hackathon dataset anonymized .csv')
if sys.argv[2] == 'reverse':
    rows.reverse()
before = deepcopy((queries, rows))
responses = {name: recommend(query, rows) for name, query in queries.items()}
assert responses == {name: recommend(query, rows) for name, query in queries.items()}
assert (queries, rows) == before
print(json.dumps({'responses': responses, 'method': ranking.RANKING_METHOD,
                  'version': ranking.SCORING_VERSION}, ensure_ascii=True, sort_keys=True))
"""
        env = {**os.environ, "RANKING_MODE": mode, "PYTHONHASHSEED": "37" if reverse else "0"}
        if mode == "semantic":
            env.update(RANKING_SEMANTIC_CACHE=os.environ["SEMANTIC_DEMO_CACHE"],
                       RANKING_SEMANTIC_SHA256=os.environ.get("SEMANTIC_DEMO_CACHE_SHA256", PreparedCatalogDemoTests.CACHE_SHA256))
        queries = {name: case["query"] for name, case in PreparedCatalogDemoTests.cases.items()}
        process = subprocess.run([sys.executable, "-B", "-c", script, str(CORE_DIR), "reverse" if reverse else "original"],
                                 input=json.dumps(queries), text=True, capture_output=True,
                                 cwd=Path(__file__).resolve().parents[1], env=env)
        self.assertEqual(process.returncode, 0, process.stderr)
        return json.loads(process.stdout)

    def check_pipeline(self, mode):
        first, second = self.run_pipeline(mode), self.run_pipeline(mode, reverse=True)
        reference = PreparedCatalogDemoTests.run_cases(mode=mode)
        expected_stats = {"dense": (10, 6, 3), "date_b": (10, 3, 3), "rare": (2, 2, 2),
                          "no_match": (10, 0, 0), "missing_category": (0, 0, 0)}
        for name, response in first["responses"].items():
            with self.subTest(mode=mode, case=name):
                self.assertEqual(response["results"], reference[name]["cards"])
                other = second["responses"][name]
                # Rejections retain catalog order in P1's core; ranking must not depend on it.
                for key in ("status", "query", "results", "stats", "meta", "message"):
                    self.assertEqual(response[key], other[key])
                self.assertEqual(sorted(response["rejections"], key=lambda r: r["id"]),
                                 sorted(other["rejections"], key=lambda r: r["id"]))
                stats = response["stats"]
                self.assertEqual(tuple(stats[k] for k in ("city_category_total", "eligible_count", "returned_count")), expected_stats[name])
                self.assertEqual(stats["city_category_total"], stats["eligible_count"] + sum(stats["rejected_first_reason"].values()))
                status = {"no_match": "no_matching_candidates", "missing_category": "category_not_found"}.get(name, "success")
                self.assertEqual(response["status"], status)
                self.assertEqual(response["meta"]["ranking_method"], first["method"] if status == "success" else "not_run")
                self.assertEqual(response["meta"]["scoring_version"], first["version"] if status == "success" else "not_run")
        rejected = {row["id"]: row for row in first["responses"]["date_b"]["rejections"]}
        for identifier in ("HK-88430", "HK-44923"):
            self.assertEqual(rejected[identifier]["primary_reason"], "busy")
            self.assertIn("busy", rejected[identifier]["reasons"])

    def test_baseline_real_core_default_imports_and_reproducibility(self):
        self.check_pipeline("baseline")

    @unittest.skipUnless(os.environ.get("SEMANTIC_DEMO_CACHE"), "optional integration: supply pinned real embedding cache")
    def test_semantic_real_core_default_imports_and_reproducibility(self):
        self.check_pipeline("semantic")


if __name__ == "__main__":
    unittest.main()
