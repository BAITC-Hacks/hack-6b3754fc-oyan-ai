import unittest

from semantic import select_description_excerpt


class ExcerptTests(unittest.TestCase):
    def test_exact_source_offsets_and_relevant_clause(self):
        text = "  Работаю давно. Ведущий: живая музыка и викторины! Другой факт."
        result = select_description_excerpt(text, {"category": "Ведущий", "event_type": "свадьба"})
        self.assertEqual(result["value"], "Ведущий: живая музыка и викторины")
        self.assertEqual(text[result["start"]:result["end"]], result["value"])
        self.assertEqual(result["matched_terms"], ["ведущий"])

    def test_no_overlap_is_not_semantic_match(self):
        result = select_description_excerpt("Свадебные церемонии у озера.", {"event_type": "свадьба"})
        self.assertEqual(result["matched_terms"], [])
        self.assertEqual(result["value"], "Свадебные церемонии у озера")

    def test_case_and_yo_normalization(self):
        result = select_description_excerpt("ТЁПЛЫЕ встречи", {"category": "теплые"})
        self.assertEqual(result["matched_terms"], ["теплые"])

    def test_empty_description_and_punctuation(self):
        for text in ("", "   ", "...!?;"):
            self.assertIsNone(select_description_excerpt(text, {}))

    def test_ties_use_first_clause(self):
        result = select_description_excerpt("Ведущий с гитарой. Ведущий с фортепиано.", {"category": "Ведущий"})
        self.assertEqual(result["value"], "Ведущий с гитарой")

    def test_long_sentence_is_not_clipped_before_late_negation(self):
        text = "Ведущий " + "свадьба " * 60 + "не входит в мои услуги."
        self.assertIsNone(select_description_excerpt(text, {"category": "Ведущий"}))

    def test_semicolon_keeps_negative_context(self):
        text = "Не работаю на следующих мероприятиях: свадьба; корпоратив."
        result = select_description_excerpt(text, {"event_type": "корпоратив"})
        self.assertEqual(result["value"], text.rstrip("."))

    def test_abbreviation_and_initials_keep_context(self):
        text = "В г. Алматы свадьбы ведёт А. Иванов."
        self.assertEqual(select_description_excerpt(text, {})["value"], text.rstrip("."))

    def test_negation_is_preserved(self):
        result = select_description_excerpt("Я не ведущий и не провожу свадьбы.", {"category": "Ведущий"})
        self.assertEqual(result["value"], "Я не ведущий и не провожу свадьбы")

    def test_decimal_number_is_not_cut(self):
        result = select_description_excerpt("Выступление 1.5 часа. Работаю один.", {})
        self.assertEqual(result["value"], "Выступление 1.5 часа")

    def test_does_not_infer_style_from_language(self):
        text = "Современный ведущий. Казахский традиционный той."
        self.assertEqual(select_description_excerpt(text, {"language": "казахский"})["value"], "Современный ведущий")


if __name__ == "__main__":
    unittest.main()
