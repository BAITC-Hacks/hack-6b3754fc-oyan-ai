"""Offline cosine matching of pinned embeddings and literal quote selection.

Runtime uses only the standard library. An encoder is supplied explicitly at
cache preparation time; loading a cache never downloads or runs model code.
The independent lexical excerpt helper is NOT neural semantic matching.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType

MATCHING_METHOD = "lexical_excerpt_v1"
_WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)
_BOUNDARIES = re.compile(r"[.!?]+(?=\s|$)")
_ABBREVIATIONS = {"г", "гг", "т", "д", "е", "п", "им", "ул", "пр", "др", "тыс", "млн", "млрд", "ч", "мин", "руб", "стр", "тел"}


def _tokens(text):
    return {word.casefold().replace("ё", "е") for word in _WORDS.findall(text)}


def _sentences(text):
    start = 0
    for boundary in _BOUNDARIES.finditer(text):
        words = _WORDS.findall(text[start:boundary.start()])
        if boundary.group() == "." and words:
            last = words[-1]
            if last.casefold() in _ABBREVIATIONS or (len(last) == 1 and last.isupper()):
                continue
        yield start, boundary.start()
        start = boundary.end()
    yield start, len(text)


def select_description_excerpt(description, query, max_chars=320):
    """Return source offsets and literal text, or None for an empty description.

    Exact-token overlap selects a complete short sentence; ties use source
    position. A zero overlap selects the first short sentence as context,
    without asserting semantic relevance. Long sentences are skipped, never
    clipped: a late negation or qualification must not be removed.
    """
    excerpts = _description_excerpts(description, query, max_chars)
    return min(excerpts, key=lambda item: (-len(item["matched_terms"]), item["start"])) if excerpts else None


def _description_excerpts(description, query, max_chars=320):
    if not isinstance(description, str):
        raise ValueError("description must be a string")
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    terms = _tokens(" ".join(str(query.get(k) or "") for k in ("event_type", "category")))
    excerpts = []
    for begin, stop in _sentences(description):
        raw = description[begin:stop]
        start = begin + len(raw) - len(raw.lstrip())
        end = stop - len(raw) + len(raw.rstrip())
        if end <= start or end - start > max_chars:
            continue
        text = description[start:end]
        if not _WORDS.search(text):
            continue
        matched = sorted(terms & _tokens(text))
        excerpts.append({"value": text, "start": start, "end": end,
                         "matched_terms": matched})
    return excerpts


TEXT_VERSION = "event-category-ru-excerpts320-v1"


class SemanticCacheError(ValueError):
    """Cache absent, stale, inconsistent or unable to cover the requested text."""


def query_text(query):
    """Only requested event/category affect meaning; eligibility is external."""
    for key in ("event_type", "category"):
        if not isinstance(query.get(key), str) or not query[key].strip():
            raise SemanticCacheError(f"{key} must be a nonempty string")
    return f"Тип мероприятия: {query['event_type']}. Категория подрядчика: {query['category']}."


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unit_vector(values, dimension):
    if not isinstance(values, (list, tuple)) or len(values) != dimension:
        raise SemanticCacheError("inconsistent embedding dimension")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise SemanticCacheError("embedding must contain finite numbers")
    # Scaling avoids overflow/underflow for otherwise valid finite vectors.
    scale = max(abs(x) for x in values)
    if scale == 0:
        raise SemanticCacheError("zero embedding is not a similarity measurement")
    scaled = tuple(x / scale for x in values)
    norm = math.sqrt(math.fsum(x * x for x in scaled))
    return tuple(x / norm for x in scaled)


def _json_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SemanticCacheError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class SemanticCache:
    """An immutable, verified snapshot; file changes cannot affect live ranking."""

    model: str
    revision: str
    sha256: str
    vectors: object

    @classmethod
    def from_bytes(cls, content, expected_sha256):
        digest = hashlib.sha256(content).hexdigest()
        if digest != expected_sha256:
            raise SemanticCacheError("semantic cache SHA-256 mismatch")
        try:
            data = json.loads(content, object_pairs_hook=_json_without_duplicates)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise SemanticCacheError("invalid semantic cache JSON") from exc
        if not isinstance(data, dict) or data.get("schema") != 1 or data.get("text_version") != TEXT_VERSION:
            raise SemanticCacheError("unsupported cache schema or text preparation")
        model, revision = data.get("model"), data.get("revision")
        if not isinstance(model, str) or not model.strip() or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise SemanticCacheError("cache requires a model name and full pinned revision")
        if not isinstance(data.get("encoder"), str) or not data["encoder"].strip():
            raise SemanticCacheError("cache requires an encoder version")
        dimension = data.get("dimension")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise SemanticCacheError("invalid embedding dimension")
        entries = data.get("entries")
        if not isinstance(entries, dict) or not entries:
            raise SemanticCacheError("cache has no embeddings")
        vectors = {}
        for key, entry in entries.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("text"), str) or _digest(entry["text"]) != key:
                raise SemanticCacheError("cache text/hash mismatch")
            vectors[key] = _unit_vector(entry.get("vector"), dimension)
        return cls(model, revision, digest, MappingProxyType(vectors))

    @classmethod
    def from_path(cls, path, expected_sha256):
        try:
            content = Path(path).read_bytes()
        except OSError as exc:
            raise SemanticCacheError("semantic cache cannot be read") from exc
        return cls.from_bytes(content, expected_sha256)

    def similarity(self, description, query):
        if not isinstance(description, str):
            raise SemanticCacheError("description must be a string")
        text = query_text(query)
        try:
            left, right = self.vectors[_digest(description)], self.vectors[_digest(text)]
        except KeyError as exc:
            raise SemanticCacheError("text missing from pinned cache; rebuild explicitly, no fallback") from exc
        return round(max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(left, right)))), 6)

    def evidence(self, description, query, similarity):
        return {"source": "profile", "field": "description", "value": description,
                "role": "semantic_scoring", "query_text": query_text(query),
                "similarity": similarity, "model": self.model, "revision": self.revision,
                "cache_sha256": self.sha256}

    def excerpt(self, description, query):
        excerpts = _description_excerpts(description, query)
        if not excerpts:
            return None
        for excerpt in excerpts:
            excerpt["similarity"] = self.similarity(excerpt["value"], query)
        best = min(excerpts, key=lambda item: (-item["similarity"], item["start"]))
        return {"source": "profile", "field": "description", "role": "semantic_excerpt", **best}


def prepare_semantic_cache(candidates, queries, encode, *, model, revision, encoder):
    """Return canonical UTF-8 JSON bytes; callers own storage and data loading.

    encode receives sorted unique texts once and returns one vector per text.
    Real models must be pinned and approved separately; tests use labelled fakes.
    Descriptions are passed verbatim, with no inferred wishes or extra profiles.
    """
    texts = {query_text(query) for query in queries}
    for candidate in candidates:
        description = candidate.get("description", "")
        if not isinstance(description, str):
            raise SemanticCacheError("description must be a string")
        texts.add(description)
        texts.update(excerpt["value"] for excerpt in _description_excerpts(description, {}))
    ordered = sorted(texts)
    if not ordered:
        raise SemanticCacheError("cannot prepare an empty cache")
    vectors = list(encode(ordered))
    if len(vectors) != len(ordered):
        raise SemanticCacheError("encoder returned a different number of vectors")
    dimension = len(vectors[0])
    data = {"schema": 1, "text_version": TEXT_VERSION, "model": model,
            "revision": revision, "encoder": encoder, "dimension": dimension,
            "entries": {_digest(text): {"text": text, "vector": list(vector)}
                        for text, vector in zip(ordered, vectors)}}
    try:
        content = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SemanticCacheError("encoder returned non-JSON or nonfinite vectors") from exc
    SemanticCache.from_bytes(content, hashlib.sha256(content).hexdigest())
    return content


def _text_chunks(text, token_count, limit):
    """Keep every character while respecting the actual tokenizer's limit."""
    if not text:
        return [""]
    chunks = []
    remaining = text
    while remaining:
        count = token_count(remaining)
        if count <= limit:
            chunks.append(remaining)
            break
        end = max(1, len(remaining) * limit // count)
        while token_count(remaining[:end]) > limit:
            if end == 1:
                raise SemanticCacheError("one character exceeds the encoder token limit")
            end = max(1, end * 3 // 4)
        # Prefer a word boundary, retaining whitespace in one of the chunks.
        boundaries = list(re.finditer(r"\s+", remaining[:end]))
        if boundaries and boundaries[-1].end() >= end // 2:
            end = boundaries[-1].end()
        chunk = remaining[:end]
        # Recheck: token counts are not guaranteed to be monotonic in length.
        if token_count(chunk) > limit:
            raise SemanticCacheError("chunk exceeds the encoder token limit")
        chunks.append(chunk)
        remaining = remaining[end:]
    return chunks


def encode_with_local_model(model, texts):
    """Prepare embeddings with a caller-supplied, pinned SentenceTransformer.

    No downloads/imports here. The caller configures CPU, eval, deterministic
    operations and the approved library versions. Each full text is split into
    lossless character chunks under max_seq_length, then token-count-weighted
    normalized embeddings are averaged. Empty text uses the model's own vector.
    Chunking is a fixed engineering heuristic, not a trained scoring weight.
    """
    limit = model.max_seq_length
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 4:
        raise SemanticCacheError("invalid model token limit")
    def token_count(text):
        return len(model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
    result = []
    for text in texts:
        chunks = _text_chunks(text, token_count, limit)
        encoded = model.encode(chunks, batch_size=1, precision="float32", normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        if len(encoded) != len(chunks):
            raise SemanticCacheError("encoder returned a different number of chunk vectors")
        vectors = [list(map(float, vector)) for vector in encoded]
        dimension = len(vectors[0])
        units = [_unit_vector(vector, dimension) for vector in vectors]
        weights = [max(1, token_count(chunk) - 2) for chunk in chunks]
        combined = [math.fsum(vector[i] * weight for vector, weight in zip(units, weights))
                    / sum(weights) for i in range(dimension)]
        result.append(list(_unit_vector(combined, dimension)))
    return result
