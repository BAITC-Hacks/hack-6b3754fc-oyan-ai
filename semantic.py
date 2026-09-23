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


TEXT_VERSION = "event-category-casefold-ru-excerpts320-v2"


class SemanticCacheError(ValueError):
    """Cache absent, stale, inconsistent or unable to cover the requested text."""


def query_text(query):
    """Match the core's strip/casefold equivalence without mutating the query.

    Only requested event/category affect meaning; eligibility is external.
    Source descriptions remain verbatim. Changing this preparation requires
    explicitly rebuilding embeddings under the new TEXT_VERSION.
    """
    for key in ("event_type", "category"):
        if not isinstance(query.get(key), str) or not query[key].strip():
            raise SemanticCacheError(f"{key} must be a nonempty string")
    event = query['event_type'].strip().casefold()
    category = query['category'].strip().casefold()
    return f"Тип мероприятия: {event}. Категория подрядчика: {category}."


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


# The preparation command is opt-in; importing runtime ranking needs no ML packages.
PINNED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PINNED_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
PREPARATION_PACKAGES = {"sentence-transformers": "6.1.0", "transformers": "5.17.0", "torch": "2.14.0"}
MODEL_FILES_SHA256 = {
    "1_Pooling/config.json": "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
    "config.json": "6300193cb75e01cf80c96decef7187dfb33094d97cc1490b7ead6ff134476e4e",
    "config_sentence_transformers.json": "b8c64b5cece00d8424b4896ea75b512b6008576088497609dfeb6bd63e6d36b8",
    "model.safetensors": "eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b",
    "modules.json": "8f4b264b80206c830bebbdcae377e137925650a433b689343a63bdc9b3145460",
    "sentence_bert_config.json": "70f4448f31320443fe3557cacea5abf2dcc4915dda8c80646bec9f3bb0aa5a1f",
    "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    "special_tokens_map.json": "378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d",
    "tokenizer.json": "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8",
    "tokenizer_config.json": "5036ea374ffedd706e3bef33e2e0d6953cb868ef8a490e76e32ba0faa37a6b9b",
    "unigram.json": "71b44701d7efd054205115acfa6ef126c5d2f84bd3affe0c59e48163674d19a6",
}


def catalog_queries(candidates):
    """Cover every selectable event/category pair, including cross-city pairs.

    Input is the validated catalog supplied by P1's loader. These are query
    texts for caching, not eligibility decisions or additional profiles.
    """
    events = sorted({value.strip().casefold() for row in candidates for value in row["event_formats"]})
    categories = sorted({value.strip().casefold() for row in candidates for value in row["categories"]})
    if not events or not categories:
        raise SemanticCacheError("catalog has no event/category pairs")
    return [{"event_type": event, "category": category} for event in events for category in categories]


def _verify_local_snapshot(directory):
    directory = Path(directory)
    harmless_files = {"README.md", "LICENSE", "NOTICE", ".gitattributes"}
    # Extra tokenizer/adapter files can alter inference despite unchanged weights.
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        if path.is_file() and relative.parts[0] != ".cache":
            if relative.as_posix() not in MODEL_FILES_SHA256 and relative.as_posix() not in harmless_files:
                raise SemanticCacheError(f"unverified local model file: {relative.as_posix()}")
    for name, expected in MODEL_FILES_SHA256.items():
        try:
            with (directory / name).open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
        except OSError as exc:
            raise SemanticCacheError(f"missing local model file: {name}") from exc
        if actual != expected:
            raise SemanticCacheError(f"local model SHA-256 mismatch: {name}")


def _publish_cache(content, output):
    """Publish complete bytes atomically, refusing to replace any existing path."""
    import os
    import tempfile

    output = Path(output)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".semantic-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink()


def _prepare_main(argv=None):
    import argparse
    import importlib.metadata
    import os

    parser = argparse.ArgumentParser(description="Prepare a pinned semantic cache offline using P1's dataset loader.")
    parser.add_argument("--dataset", required=True, type=Path, help="CSV/JSONL accepted by data_loader.load_contractors")
    parser.add_argument("--model-dir", required=True, type=Path, help=f"local snapshot of {PINNED_MODEL}@{PINNED_REVISION}")
    parser.add_argument("--output", required=True, type=Path, help="new JSON cache path; parent must exist; never overwritten")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise SemanticCacheError("output already exists; choose a new path")
        if not args.output.parent.is_dir():
            raise SemanticCacheError("output parent directory does not exist")
        # Do not create a second loader. Missing core/dependencies is an explicit error.
        from data_loader import load_contractors

        candidates = load_contractors(args.dataset)
        queries = catalog_queries(candidates)
        _verify_local_snapshot(args.model_dir)
        for package, expected in PREPARATION_PACKAGES.items():
            if importlib.metadata.version(package) != expected:
                raise SemanticCacheError(f"preparation requires {package}=={expected}")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                          HF_HUB_DISABLE_IMPLICIT_TOKEN="1", TOKENIZERS_PARALLELISM="false")
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        model = SentenceTransformer(str(args.model_dir.resolve()), device="cpu", local_files_only=True,
                                    trust_remote_code=False, model_kwargs={"use_safetensors": True})
        model.eval()
        encoder = ";".join(f"{key}={value}" for key, value in PREPARATION_PACKAGES.items())
        encoder += ";cpu;threads=1;batch=1;fp32;token-chunks-weighted-v1"
        content = prepare_semantic_cache(candidates, queries, lambda texts: encode_with_local_model(model, texts),
                                         model=PINNED_MODEL, revision=PINNED_REVISION, encoder=encoder)
        digest = hashlib.sha256(content).hexdigest()
        _publish_cache(content, args.output)
    except (ImportError, importlib.metadata.PackageNotFoundError, OSError, ValueError) as exc:
        parser.exit(2, f"Cannot prepare semantic cache: {exc}\n")
    print(json.dumps({"cache_path": str(args.output.resolve()), "sha256": digest,
                      "text_version": TEXT_VERSION, "model": PINNED_MODEL, "revision": PINNED_REVISION,
                      "profiles": len(candidates), "query_pairs": len(queries)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    _prepare_main()
