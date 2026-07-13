"""
Unstructured Data retrieval tool.

Design decision: uses a dependency-free TF-IDF + cosine similarity index over
the markdown documents rather than calling an embeddings API. This keeps the
"unstructured retrieval" sub-agent fully functional in MOCK mode (no API key)
and avoids a second network dependency. In a production build this module is
a drop-in swap for a real vector DB (see docs/design-decisions.md) — the
retrieve() function signature would not need to change.

Supports:
  - source citation (filename + doc_type + published date returned per hit)
  - metadata filtering (category, doc_type, tags, recency)
  - simple "supersedes" awareness: documents whose front-matter tags include
    'superseded' are down-weighted, and hybrid queries can be asked to prefer
    the most recent doc on a topic — this is what "document filtering using
    metadata, tags, and recency" and "transparent reporting of assumptions"
    exercise in the eval notebook (see the discount-policy question).
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "unstructured"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / ".cache"

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class _DocIndex:
    def __init__(self):
        self.docs: List[Dict[str, Any]] = []
        self._df: Counter = Counter()
        self._embeddings_by_fingerprint: Dict[str, List[float]] = {}
        self._embeddings_loaded = False
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        for path in sorted(DOCS_DIR.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            meta, body = self._parse_front_matter(raw)
            tokens = _tokenize(body)
            tf = Counter(tokens)
            self.docs.append({
                "filename": path.name,
                "meta": meta,
                "body": body,
                "tf": tf,
                "length": max(len(tokens), 1),
            })
            for term in set(tokens):
                self._df[term] += 1
        self._loaded = True

    @staticmethod
    def _fingerprint(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_file(self, model: str) -> Path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", model)
        return CACHE_DIR / f"doc_embeddings_{safe_model}.json"

    def load_embeddings_cache(self, model: str):
        if self._embeddings_loaded:
            return
        path = self._cache_file(model)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    self._embeddings_by_fingerprint = {
                        k: v for k, v in payload.items() if isinstance(k, str) and isinstance(v, list)
                    }
            except Exception:
                self._embeddings_by_fingerprint = {}
        self._embeddings_loaded = True

    def save_embeddings_cache(self, model: str):
        path = self._cache_file(model)
        path.write_text(json.dumps(self._embeddings_by_fingerprint), encoding="utf-8")

    @staticmethod
    def _parse_front_matter(raw: str):
        if raw.startswith("---"):
            end = raw.find("\n---", 3)
            fm_block = raw[3:end].strip()
            body = raw[end + 4:].strip()
            meta = {}
            for line in fm_block.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    v = v.strip()
                    if v.startswith("[") and v.endswith("]"):
                        v = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
                    meta[k.strip()] = v
            return meta, body
        return {}, raw

    def score(self, query: str) -> List[Dict[str, Any]]:
        self.load()
        n_docs = len(self.docs)
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        results = []
        for doc in self.docs:
            score = 0.0
            for term in set(q_tokens):
                tf = doc["tf"].get(term, 0)
                if tf == 0:
                    continue
                df = self._df.get(term, 0)
                idf = math.log((n_docs + 1) / (df + 1)) + 1
                score += (tf / doc["length"]) * idf
            if score > 0:
                # down-weight explicitly superseded documents but don't hide them
                tags = doc["meta"].get("tags", [])
                if isinstance(tags, list) and "superseded" in tags:
                    score *= 0.5
                results.append({"doc": doc, "score": score})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results


_INDEX = _DocIndex()


def retrieve(
    query: str,
    top_k: int = 4,
    category: Optional[str] = None,
    doc_type: Optional[str] = None,
    prefer_recent: bool = True,
) -> Dict[str, Any]:
    """Retrieve top_k relevant documents with metadata filters + citations."""
    mode = (os.environ.get("DOC_RETRIEVAL_MODE") or "hybrid").strip().lower()
    provider = (os.environ.get("DOC_VECTOR_PROVIDER") or "openai").strip().lower()

    retrieval_mode = "tfidf"
    vector_error = None
    scored: List[Dict[str, Any]] = []

    if mode in {"vector", "hybrid"} and provider == "openai":
        scored, vector_error = _vector_score_openai(query)
        if scored and mode == "vector":
            retrieval_mode = "vector"
        elif scored and mode == "hybrid":
            retrieval_mode = "hybrid"

    if not scored:
        scored = _INDEX.score(query)
        retrieval_mode = "tfidf"

    def matches_filters(doc):
        m = doc["meta"]
        if category and m.get("category", "").lower() != category.lower():
            return False
        if doc_type and m.get("doc_type", "").lower() != doc_type.lower():
            return False
        return True

    filtered = [r for r in scored if matches_filters(r["doc"])]
    if not filtered and (category or doc_type):
        # graceful fallback: filters too narrow, widen and flag it
        filtered = scored
        widened = True
    else:
        widened = False

    if prefer_recent:
        filtered.sort(key=lambda r: (r["score"], r["doc"]["meta"].get("published", "")), reverse=True)

    hits = filtered[:top_k]
    return {
        "query": query,
        "retrieval_mode": retrieval_mode,
        "vector_fallback_reason": vector_error,
        "filters_applied": {"category": category, "doc_type": doc_type},
        "filters_widened_due_to_no_match": widened,
        "results": [
            {
                "source": h["doc"]["filename"],
                "doc_type": h["doc"]["meta"].get("doc_type"),
                "category": h["doc"]["meta"].get("category"),
                "published": h["doc"]["meta"].get("published"),
                "tags": h["doc"]["meta"].get("tags"),
                "relevance_score": round(h["score"], 4),
                "excerpt": _excerpt(h["doc"]["body"], query),
            }
            for h in hits
        ],
    }


def _excerpt(body: str, query: str, window: int = 280) -> str:
    q_tokens = set(_tokenize(query))
    paragraphs = [p for p in body.split("\n\n") if p.strip()]
    best_p, best_hits = paragraphs[0] if paragraphs else body[:window], -1
    for p in paragraphs:
        hits = sum(1 for t in _tokenize(p) if t in q_tokens)
        if hits > best_hits:
            best_hits, best_p = hits, p
    return best_p[:window].strip()


def _vector_score_openai(query: str) -> tuple[List[Dict[str, Any]], Optional[str]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return [], "OPENAI_API_KEY not set for vector retrieval"

    _INDEX.load()
    model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    timeout = float(os.environ.get("DOC_VECTOR_TIMEOUT_SECONDS", "20"))

    try:
        _INDEX.load_embeddings_cache(model)

        doc_vectors: Dict[str, List[float]] = {}
        missing: List[tuple[str, str]] = []
        for d in _INDEX.docs:
            fp = _INDEX._fingerprint(d["body"])
            vec = _INDEX._embeddings_by_fingerprint.get(fp)
            if vec:
                doc_vectors[d["filename"]] = vec
            else:
                missing.append((d["filename"], d["body"]))

        if missing:
            texts = [b for _, b in missing]
            batch = _openai_embed(texts, model=model, api_key=api_key, timeout=timeout)
            for (filename, body), vec in zip(missing, batch):
                fp = _INDEX._fingerprint(body)
                _INDEX._embeddings_by_fingerprint[fp] = vec
                doc_vectors[filename] = vec
            _INDEX.save_embeddings_cache(model)

        q_vec = _openai_embed([query], model=model, api_key=api_key, timeout=timeout)[0]
        q_norm = _l2_norm(q_vec)

        rows = []
        for d in _INDEX.docs:
            d_vec = doc_vectors.get(d["filename"])
            if not d_vec:
                continue
            score = _cosine_from_normed(q_vec, q_norm, d_vec)
            rows.append({"doc": d, "score": score})

        rows.sort(key=lambda r: r["score"], reverse=True)

        if (os.environ.get("DOC_RETRIEVAL_MODE") or "hybrid").strip().lower() != "vector":
            tfidf_rows = _INDEX.score(query)
            tfidf_map = {r["doc"]["filename"]: r["score"] for r in tfidf_rows}
            if tfidf_map:
                max_tfidf = max(tfidf_map.values())
                if max_tfidf > 0:
                    for r in rows:
                        tfidf_norm = tfidf_map.get(r["doc"]["filename"], 0.0) / max_tfidf
                        r["score"] = 0.65 * r["score"] + 0.35 * tfidf_norm
                rows.sort(key=lambda r: r["score"], reverse=True)

        return rows, None
    except Exception as e:
        return [], f"vector retrieval unavailable: {e}"


def _openai_embed(texts: List[str], model: str, api_key: str, timeout: float) -> List[List[float]]:
    base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        url=f"{base_url}/embeddings",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    data = body.get("data", [])
    if len(data) != len(texts):
        raise RuntimeError("Unexpected embeddings response size")
    return [item["embedding"] for item in data]


def _l2_norm(vec: List[float]) -> float:
    return math.sqrt(sum(x * x for x in vec)) or 1.0


def _cosine_from_normed(a: List[float], a_norm: float, b: List[float]) -> float:
    b_norm = _l2_norm(b)
    return sum(x * y for x, y in zip(a, b)) / (a_norm * b_norm)


def list_metadata() -> Dict[str, Any]:
    _INDEX.load()
    categories = sorted({d["meta"].get("category", "") for d in _INDEX.docs})
    doc_types = sorted({d["meta"].get("doc_type", "") for d in _INDEX.docs})
    return {"categories": categories, "doc_types": doc_types, "document_count": len(_INDEX.docs)}
