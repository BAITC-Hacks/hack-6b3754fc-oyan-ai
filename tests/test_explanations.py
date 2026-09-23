from copy import deepcopy
import csv
import hashlib
from pathlib import Path
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

    def test_dates_and_requested_language_hours_are_grounded(self):
        row = profile(languages=["русский", "казахский", "английский"], max_hours=10)
        query = {**QUERY, "language": "казахский"}
        card = build_cards([row], query)[0]
        self.assertIn("по календарю нет отметки о занятости на 2026-11-14", card["explanation"])
        self.assertIn("запрошенный язык «казахский» указан в профиле", card["explanation"])
        self.assertIn("заявленные 10 ч покрывают запрошенные 6 ч", card["explanation"])
        self.assertNotIn("английский", card["explanation"])
        self.assertEqual(card["explanation"].count("."), 2)
        for source, field in (("profile", "busy_dates"), ("profile", "max_hours"), ("profile", "languages"),
                              ("query", "date"), ("query", "duration"), ("query", "language")):
            expected = row[field] if source == "profile" else query[field]
            fact = next(e for e in card["evidence"] if e["source"] == source and e["field"] == field)
            self.assertEqual(fact["value"], expected)

    def test_same_quote_and_price_different_hours_have_distinct_facts(self):
        rows = [profile("A", max_hours=8), profile("B", max_hours=10)]
        ranked = rank_candidates(rows, QUERY)
        cards = build_cards(ranked, QUERY)
        self.assertEqual(cards[0]["score"], cards[1]["score"])
        self.assertNotEqual(cards[0]["explanation"], cards[1]["explanation"])
        self.assertIn("заявленные 8 ч", cards[0]["explanation"])
        self.assertIn("заявленные 10 ч", cards[1]["explanation"])
        self.assertNotIn("факты совпадают", cards[0]["explanation"])

    def test_optional_conditions_are_omitted_when_not_requested(self):
        row = profile(max_hours=10, languages=["русский", "английский"])
        for query in ({**QUERY, "duration": None, "language": None},
                      {k: v for k, v in QUERY.items() if k not in ("duration", "language")}):
            with self.subTest(query=query):
                text = build_cards([row], query)[0]["explanation"]
                self.assertNotIn("запрошенн", text)
                self.assertNotIn("английский", text)

    def test_duration_boundary_fractional_and_null_are_honest(self):
        for maximum, duration, expected in ((6, 6, "заявленные 6 ч покрывают запрошенные 6 ч"),
                                             (6.5, 6.25, "заявленные 6.5 ч покрывают запрошенные 6.25 ч"),
                                             (None, 12, "услуга не привязана к присутствию на площадке")):
            with self.subTest(maximum=maximum):
                text = build_cards([profile(max_hours=maximum)], {**QUERY, "duration": duration})[0]["explanation"]
                self.assertIn(expected, text)
                if maximum is None:
                    self.assertNotIn("покрывают", text)
                    self.assertNotIn("безлимит", text)

    def test_missing_fields_do_not_become_assurances(self):
        row = profile()
        for key in ("busy_dates", "max_hours", "languages"):
            del row[key]
        text = build_cards([row], {**QUERY, "language": "русский"})[0]["explanation"]
        for phrase in ("по календарю", "запрошенный язык", "привязана", "покрывают"):
            self.assertNotIn(phrase, text)
        row["busy_dates"] = None
        self.assertNotIn("по календарю", build_cards([row], QUERY)[0]["explanation"])

    def test_contradictory_core_input_is_not_mislabeled_or_filtered(self):
        row = profile(busy_dates=[QUERY["date"]], max_hours=4, languages=["русский"])
        cards = build_cards([row], {**QUERY, "language": "казахский"})
        self.assertEqual([c["id"] for c in cards], [row["id"]])
        text = cards[0]["explanation"]
        self.assertIn("есть отметка о занятости", text)
        self.assertIn("«казахский» не указан", text)
        self.assertIn("заявлено 4 ч, меньше запрошенных 6 ч", text)
        self.assertNotIn("нет отметки", text)
        self.assertNotIn("покрывают", text)

    def test_calendar_evidence_is_a_copy(self):
        row = profile(busy_dates=["2026-12-01"])
        card = build_cards([row], QUERY)[0]
        next(e for e in card["evidence"] if e["field"] == "busy_dates")["value"].append(QUERY["date"])
        self.assertEqual(row["busy_dates"], ["2026-12-01"])
        self.assertEqual(card["busy_dates"], ["2026-12-01"])

    def test_semantic_mode_keeps_score_but_reflects_new_request_conditions(self):
        row = self.semantic_row()
        row["max_hours"] = 10
        row["languages"] = ["русский", "казахский"]
        changed = {**QUERY, "date": "2026-11-15", "language": "казахский", "duration": 10}
        first, second = build_cards([row], QUERY)[0], build_cards([row], changed)[0]
        self.assertNotEqual(first["explanation"], second["explanation"])
        self.assertEqual(first["score_breakdown"], second["score_breakdown"])
        self.assertIn("2026-11-15", second["explanation"])
        self.assertIn("запрошенные 10 ч", second["explanation"])
        self.assertIn("смысловой близости", second["explanation"])

    def test_supplied_csv_profile_reflects_actual_date_and_hours(self):
        """Reference regression on supplied CSV; not a JSONL/core acceptance test."""
        path = Path(__file__).resolve().parents[1] / "docs/hackathon dataset anonymized .csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            raw = next(row for row in csv.DictReader(stream) if row["id"] == "HK-42352")
        row = dict(raw)
        for key in ("categories", "event_formats", "languages", "busy_dates"):
            row[key] = raw[key].split("|")
        row["price_from_kzt"] = int(raw["price_from_kzt"])
        row["max_hours"] = float(raw["max_hours"])
        for key in ("synthetic", "price_imputed", "city_imputed"):
            row[key] = raw[key] == "True"
        first_query = {**QUERY, "budget": 1000000, "date": "2026-09-30", "language": "казахский"}
        second_query = {**first_query, "date": "2026-10-07", "duration": 10}
        self.assertEqual(row["max_hours"], 10)
        self.assertIn("казахский", row["languages"])
        for query in (first_query, second_query):
            self.assertNotIn(query["date"], row["busy_dates"])
        first = build_cards([row], first_query)[0]["explanation"]
        second = build_cards([row], second_query)[0]["explanation"]
        self.assertNotEqual(first, second)
        self.assertIn("запрошенные 6 ч", first)
        self.assertIn("запрошенные 10 ч", second)
        self.assertIn("2026-09-30", first)
        self.assertIn("2026-10-07", second)


if __name__ == "__main__":
    unittest.main()
