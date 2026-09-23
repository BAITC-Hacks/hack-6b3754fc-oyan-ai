import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from semantic import SemanticCache, SemanticCacheError, _text_chunks, encode_with_local_model, prepare_semantic_cache, query_text, select_description_excerpt


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


class CacheTests(unittest.TestCase):
    """Fake vectors verify mechanics, not model quality or neural inference."""

    def setUp(self):
        self.query = {"event_type": "свадьба", "category": "Ведущий"}
        self.rows = [{"description": "A"}, {"description": "B"}]
        self.table = {"A": [1.0, 0.0], "B": [0.0, 1.0], query_text(self.query): [1.0, 0.0]}
        self.content = self.prepare()

    def prepare(self, rows=None, queries=None, encode=None):
        return prepare_semantic_cache(self.rows if rows is None else rows,
                                      [self.query] if queries is None else queries,
                                      encode or (lambda texts: [self.table[t] for t in texts]),
                                      model="explicit-test-fake", revision="a" * 40, encoder="fake-v1")

    def load(self, content=None):
        content = self.content if content is None else content
        return SemanticCache.from_bytes(content, hashlib.sha256(content).hexdigest())

    def test_cosine_and_finite_range(self):
        cache = self.load()
        self.assertEqual(cache.similarity("A", self.query), 1.0)
        self.assertEqual(cache.similarity("B", self.query), 0.0)

    def test_encoding_is_deduplicated_and_sorted(self):
        calls = []
        def encode(texts):
            calls.append(texts)
            return [self.table[t] for t in texts]
        self.assertEqual(self.prepare(rows=self.rows[::-1] + self.rows, queries=[self.query] * 2, encode=encode), self.content)
        self.assertEqual(calls, [sorted(self.table)])

    def test_unknown_or_changed_text_does_not_fallback(self):
        cache = self.load()
        for description, query in (("A changed", self.query), ("A", {**self.query, "event_type": "той"})):
            with self.assertRaisesRegex(SemanticCacheError, "missing from pinned cache"):
                cache.similarity(description, query)

    def test_query_uses_only_explicit_event_category(self):
        changed = {**self.query, "budget": 999, "city": "Астана", "language": "казахский", "date": "2026-12-31", "duration": 12}
        self.assertEqual(query_text(changed), query_text(self.query))
        self.assertEqual(self.load().similarity("A", changed), 1)

    def test_query_case_and_edge_spaces_share_preparation_and_vectors(self):
        variant = {"event_type": "  СВАДЬБА\t", "category": " ведущий "}
        before = dict(variant)
        self.assertEqual(query_text(variant), query_text(self.query))
        self.assertEqual(self.prepare(queries=[self.query, variant]), self.content)
        cache = self.load()
        self.assertEqual(cache.similarity("A", variant), cache.similarity("A", self.query))
        self.assertEqual(cache.evidence("A", variant, 1), cache.evidence("A", self.query, 1))
        self.assertEqual(variant, before)

    def test_query_normalization_does_not_change_internal_spaces_or_description(self):
        cache = self.load()
        for description, query in (("a", self.query), (" A ", self.query),
                                   ("A", {**self.query, "category": "Ве дущий"})):
            with self.subTest(description=description, query=query), self.assertRaises(SemanticCacheError):
                cache.similarity(description, query)

    def test_old_text_preparation_requires_explicit_rebuild(self):
        data = json.loads(self.content)
        data["text_version"] = "event-category-ru-excerpts320-v1"
        with self.assertRaisesRegex(SemanticCacheError, "text preparation"):
            self.load(json.dumps(data).encode())

    def test_snapshot_is_immutable_even_if_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            path.write_bytes(self.content)
            cache = SemanticCache.from_path(path, hashlib.sha256(self.content).hexdigest())
            path.write_text("broken")
            self.assertEqual(cache.similarity("A", self.query), 1)
            with self.assertRaises(TypeError):
                cache.vectors["changed"] = (0, 1)
            with self.assertRaises(SemanticCacheError):
                SemanticCache.from_path(path, hashlib.sha256(self.content).hexdigest())

    def test_missing_file_hash_and_invalid_json(self):
        with self.assertRaises(SemanticCacheError):
            SemanticCache.from_path("/no-such-semantic-cache.json", "a" * 64)
        with self.assertRaises(SemanticCacheError):
            SemanticCache.from_bytes(self.content, "0" * 64)
        with self.assertRaises(SemanticCacheError):
            self.load(b"not JSON")
        with self.assertRaises(SemanticCacheError):
            self.load(b'{"schema":1,"schema":1}')

    def test_invalid_vectors_fail_before_cache_publication(self):
        for vector in ([0.0, 0.0], [float("nan"), 0], [float("inf"), 0], [True, 0], [], ["1", 0]):
            with self.subTest(vector=vector), self.assertRaises(SemanticCacheError):
                self.prepare(encode=lambda texts: [vector] * len(texts))
        with self.assertRaises(SemanticCacheError):
            self.prepare(encode=lambda texts: [[1, 0]])
        with self.assertRaises(SemanticCacheError):
            self.prepare(encode=lambda texts: [[1, 0], [1], [1, 0]])

    def test_rejects_unpinned_revision_changed_preparation_and_text_hash(self):
        for field, value in (("revision", "main"), ("text_version", "unknown"), ("dimension", 0), ("encoder", "")):
            data = json.loads(self.content)
            data[field] = value
            with self.subTest(field=field), self.assertRaises(SemanticCacheError):
                self.load(json.dumps(data).encode())
        data = json.loads(self.content)
        next(iter(data["entries"].values()))["text"] = "tampered"
        with self.assertRaises(SemanticCacheError):
            self.load(json.dumps(data).encode())

    def test_stable_normalization_for_large_and_tiny_vectors(self):
        for magnitude in (1e300, 1e-300):
            cache = self.load(self.prepare(encode=lambda texts: [[magnitude, magnitude] for t in texts]))
            self.assertEqual(cache.similarity("A", self.query), 1.0)

    def test_semantic_excerpt_prefers_meaning_over_first_sentence(self):
        description = "Первая общая фраза. Церемонии бракосочетания."
        table = {description: [1, 1], "Первая общая фраза": [0, 1],
                 "Церемонии бракосочетания": [1, 0], query_text(self.query): [1, 0]}
        cache = self.load(self.prepare(rows=[{"description": description}], encode=lambda texts: [table[t] for t in texts]))
        excerpt = cache.excerpt(description, self.query)
        self.assertEqual(excerpt["value"], "Церемонии бракосочетания")
        self.assertEqual(description[excerpt["start"]:excerpt["end"]], excerpt["value"])
        self.assertEqual(excerpt["matched_terms"], [])

    def test_semantic_excerpt_tie_uses_source_position(self):
        description = "Первая фраза. Вторая фраза."
        cache = self.load(self.prepare(rows=[{"description": description}], encode=lambda texts: [[1, 0] for t in texts]))
        self.assertEqual(cache.excerpt(description, self.query)["value"], "Первая фраза")


class ChunkTests(unittest.TestCase):
    def test_chunks_keep_all_text_including_last_qualification(self):
        text = "свадьба " * 100 + "но только камерная"
        chunks = _text_chunks(text, lambda part: len(part) + 2, 128)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(chunk) + 2 <= 128 for chunk in chunks))
        self.assertTrue(chunks[-1].endswith("но только камерная"))

    def test_empty_and_unbroken_unicode_text(self):
        self.assertEqual(_text_chunks("", lambda part: 2, 128), [""])
        text = "Ю" * 300
        self.assertEqual("".join(_text_chunks(text, lambda part: len(part) + 2, 128)), text)

    def test_tail_embedding_contributes_to_pooled_result(self):
        class FakeModel:
            max_seq_length = 6
            def tokenizer(self, text, **kwargs):
                return {"input_ids": [0] * (len(text) + 2)}
            def encode(self, chunks, **kwargs):
                self.seen = chunks
                return [[1, 0] if "B" not in text else [0, 1] for text in chunks]
        model = FakeModel()
        vector = encode_with_local_model(model, ["AAAABBBB"])[0]
        self.assertEqual("".join(model.seen), "AAAABBBB")
        self.assertGreater(vector[0], 0)
        self.assertGreater(vector[1], 0)


@unittest.skipUnless(os.environ.get("SEMANTIC_TEST_MODEL_DIR"), "optional local model smoke: set SEMANTIC_TEST_MODEL_DIR")
class LocalModelSmokeTest(unittest.TestCase):
    def test_real_pinned_model_recognizes_paraphrase_and_repeats(self):
        """Opt-in only: no installation, network access or account required here."""
        import importlib.metadata
        import torch
        from sentence_transformers import SentenceTransformer
        for package, expected in (("sentence-transformers", "6.1.0"), ("transformers", "5.17.0"), ("torch", "2.14.0")):
            self.assertEqual(importlib.metadata.version(package), expected)
        path = Path(os.environ["SEMANTIC_TEST_MODEL_DIR"])
        with (path / "model.safetensors").open("rb") as weights:
            self.assertEqual(hashlib.file_digest(weights, "sha256").hexdigest(),
                             "eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        model = SentenceTransformer(str(path), device="cpu", local_files_only=True,
                                    trust_remote_code=False, model_kwargs={"use_safetensors": True})
        model.eval()
        query = {"category": "Фотограф", "event_type": "свадьба"}
        rows = [{"description": "Запечатлеваю церемонии бракосочетания и счастливые моменты молодожёнов."},
                {"description": "Устанавливаю трубы и ремонтирую водопровод."}]
        def prepare():
            return prepare_semantic_cache(rows, [query], lambda texts: encode_with_local_model(model, texts),
                                          model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                          revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
                                          encoder="ST6.1.0-T5.17.0-torch2.14.0-cpu-token-chunks-weighted-v1")
        content = prepare()
        self.assertEqual(content, prepare())
        cache = SemanticCache.from_bytes(content, hashlib.sha256(content).hexdigest())
        self.assertGreater(cache.similarity(rows[0]["description"], query), cache.similarity(rows[1]["description"], query))


if __name__ == "__main__":
    unittest.main()
