from copy import deepcopy
import hashlib
import unittest

from explanations import build_cards
from ranking import rank_candidates
from semantic import SemanticCache, prepare_semantic_cache


QUERY = {"city": "Алматы", "date": "2026-11-14", "category": "Ведущий",
         "event_type": "свадьба", "budget": 300000, "duration": 6, "language": None}


def profile(identifier="A", **changes):
    return {"id": identifier, "anon_name": "Синтетический пример " + identifier,
            "price_from_kzt": 220000, "description": "Ведущий с живой музыкой.",
            "event_formats": ["свадьба"], "city": "Алматы", "categories": ["Ведущий"],
            "languages": ["русский"], "max_hours": None, "busy_dates": [],
            "synthetic": True, "price_imputed": False, "city_imputed": False, **changes}


class ExplanationTests(unittest.TestCase):
    def test_exact_amounts_and_two_sentences(self):
        text = build_cards([profile()], QUERY)[0]["explanation"]
        self.assertIn("от 220 000 ₸, на 80 000 ₸ ниже лимита 300 000 ₸", text)
        self.assertIn("формат «свадьба»", text)
        self.assertIn("В описании указано: «Ведущий с живой музыкой»", text)
        self.assertEqual(text.count("."), 2)

    def test_exact_budget_and_fractional_money(self):
        self.assertIn("равна лимиту 300 000", build_cards([profile(price_from_kzt=300000)], QUERY)[0]["explanation"])
        text = build_cards([profile(price_from_kzt=0.1)], {**QUERY, "budget": 0.3})[0]["explanation"]
        self.assertIn("на 0.2 ₸ ниже", text)

    def test_different_factual_quotes_not_names(self):
        rows = [profile("A", description="Ведущий с гитарой."),
                profile("B", description="Ведущий с фортепиано."),
                profile("C", description="Ведущий с викторинами.")]
        cards = build_cards(rank_candidates(rows, QUERY)[:3], QUERY)
        self.assertEqual(len({c["explanation"] for c in cards}), 3)
        for card, row in zip(cards, rows):
            self.assertNotIn(row["anon_name"], card["explanation"])
            self.assertIn(row["description"].rstrip("."), card["explanation"])

    def test_evidence_matches_profile_and_query(self):
        row = profile()
        card = build_cards(rank_candidates([row], QUERY), QUERY)[0]
        for fact in card["evidence"]:
            original = row if fact["source"] == "profile" else QUERY
            if fact["field"] == "description":
                self.assertEqual(original["description"][fact["start"]:fact["end"]], fact["value"])
                self.assertIn(fact["value"], card["explanation"])
            else:
                self.assertEqual(original[fact["field"]], fact["value"])

    def test_flags_null_and_absence_of_promises(self):
        row = profile(price_imputed=True, city_imputed=True)
        card = build_cards([row], QUERY)[0]
        text = card["explanation"]
        self.assertIn("цена проставлена при подготовке данных", text)
        self.assertIn("город «Алматы» проставлен при подготовке данных", text)
        self.assertTrue(card["synthetic"])
        self.assertIsNone(card["max_hours"])
        for prohibited in ("безлимит", "лучший", "идеальн", "гарантирован", "забронирован", "экономия", "единственный"):
            self.assertNotIn(prohibited, text.lower())

    def test_preserves_order_all_fields_no_top3_slice_or_mutation(self):
        rows = rank_candidates([profile(str(i), price_from_kzt=i * 10000) for i in range(5)], QUERY)[::-1]
        original, query = deepcopy(rows), deepcopy(QUERY)
        cards = build_cards(rows, query)
        self.assertEqual([c["id"] for c in cards], [r["id"] for r in rows])
        for card, row in zip(cards, rows):
            for key, value in row.items():
                self.assertEqual(card[key], value)
            self.assertEqual(card["category"], QUERY["category"])
        cards[0]["languages"].append("английский")
        self.assertEqual(rows, original)
        self.assertEqual(query, QUERY)

    def test_empty_and_missing_description(self):
        self.assertEqual(build_cards([], QUERY), [])
        row = profile(description="")
        self.assertIn("Краткая цитата из описания не приведена", build_cards([row], QUERY)[0]["explanation"])

    def test_equal_facts_are_not_fabricated_as_differences(self):
        cards = build_cards([profile("A"), profile("B")], QUERY)
        self.assertEqual(cards[0]["explanation"], cards[1]["explanation"])
        self.assertIn("факты совпадают", cards[0]["explanation"])
        self.assertEqual(cards[0]["explanation"].count("."), 2)

    def test_long_sentence_is_not_misrepresented(self):
        card = build_cards([profile(description="Ведущий " + "музыка " * 70)], QUERY)[0]
        self.assertIn("Краткая цитата из описания не приведена", card["explanation"])
        self.assertFalse(any(e["field"] == "description" for e in card["evidence"]))

    def semantic_row(self):
        row = profile()
        content = prepare_semantic_cache([row], [QUERY], lambda texts: [[1, 0] for t in texts],
                                         model="explicit-test-fake", revision="a" * 40, encoder="fake-v1")
        cache = SemanticCache.from_bytes(content, hashlib.sha256(content).hexdigest())
        row["score"] = 1.0
        row["score_breakdown"] = {"semantic_similarity": 1.0}
        row["evidence"] = [cache.evidence(row["description"], QUERY, 1.0)]
        row["evidence"].append(cache.excerpt(row["description"], QUERY))
        return row

    def test_semantic_evidence_is_preserved_without_aliasing(self):
        row = self.semantic_row()
        before = deepcopy(row)
        card = build_cards([row], QUERY)[0]
        fact = next(e for e in card["evidence"] if e.get("role") == "semantic_scoring")
        self.assertEqual(fact, row["evidence"][0])
        fact["value"] = "changed"
        self.assertEqual(row, before)

    def test_changed_semantic_query_profile_and_score_raise(self):
        for key, value in (("description", "Changed"), ("score", 0.5), ("evidence", [])):
            row = self.semantic_row()
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                build_cards([row], QUERY)
        with self.assertRaises(ValueError):
            build_cards([self.semantic_row()], {**QUERY, "event_type": "той"})

    def test_semantic_excerpt_must_be_literal(self):
        row = self.semantic_row()
        row["evidence"][1]["value"] = "Выдуманное преимущество"
        with self.assertRaises(ValueError):
            build_cards([row], QUERY)


if __name__ == "__main__":
    unittest.main()
