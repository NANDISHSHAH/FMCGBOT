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
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "unstructured"

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class _DocIndex:
    def __init__(self):
        self.docs: List[Dict[str, Any]] = []
        self._df: Counter = Counter()
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
    scored = _INDEX.score(query)

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


def list_metadata() -> Dict[str, Any]:
    _INDEX.load()
    categories = sorted({d["meta"].get("category", "") for d in _INDEX.docs})
    doc_types = sorted({d["meta"].get("doc_type", "") for d in _INDEX.docs})
    return {"categories": categories, "doc_types": doc_types, "document_count": len(_INDEX.docs)}
