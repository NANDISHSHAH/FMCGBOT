"""
Front door of the pipeline: figures out what kind of turn this is BEFORE any
retrieval happens (this is the "intent validation before data retrieval"
requirement), and normalizes the query text (aliases/abbreviations/typos,
language) so downstream agents get a clean, canonical query.

Design decision: greetings and capability questions are caught with cheap
regex, not an LLM call. Full text ("hi", "hello", "what can you do") is a
tiny, closed set — spending a Gemini call to classify it would add ~500ms-1s
of pure latency+cost for zero accuracy benefit. Everything else goes through
LLM-based classification. This split is discussed in
docs/cost-latency-tradeoffs.md.
"""
from __future__ import annotations
import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.llm_client import LLMClient

GREETING_RE = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|namaste|yo)\b", re.I)
CAPABILITY_RE = re.compile(
    r"\b(what can you do|what do you do|help me understand|capabilit(y|ies)|what are you able to)\b", re.I
)
THANKS_BYE_RE = re.compile(r"^\s*(thanks|thank you|bye|goodbye|that'?s all)\b", re.I)

# ---------------------------------------------------------------------------
# Domain / entity dictionaries used for alias + typo correction (semantic
# understanding requirement). In production these would be generated from
# the warehouse's dimension tables at startup; hardcoded here for clarity.
# ---------------------------------------------------------------------------
BRAND_ALIASES = {
    "nutrioat": "NutriOat", "no": "NutriOat", "nutri oat": "NutriOat",
    "sunfresh": "SunFresh", "sf": "SunFresh", "sun fresh": "SunFresh",
    "crispco": "CrispCo", "cc": "CrispCo", "crisp co": "CrispCo",
    "purewave": "PureWave", "pw": "PureWave", "pure wave": "PureWave",
    "homeglow": "HomeGlow", "hg": "HomeGlow", "home glow": "HomeGlow",
}
REGION_ALIASES = {"north": "North", "south": "South", "east": "East", "west": "West",
                   "n": "North", "s": "South", "e": "East", "w": "West"}
KPI_ALIASES = {
    "revenue": "net_revenue_inr", "sales": "net_revenue_inr", "net revenue": "net_revenue_inr",
    "units": "units_sold", "volume": "units_sold", "qty": "units_sold", "quantity": "units_sold",
    "discount": "discount_inr", "stock": "closing_stock_units", "inventory": "closing_stock_units",
    "doc": "days_of_cover", "days of cover": "days_of_cover",
}
CANONICAL_TERMS = list(BRAND_ALIASES.values()) + list(REGION_ALIASES.values())

# Very small language cue set for "multilingual and mixed-language" detection.
# A production system would use a proper language-ID model; this is a
# lightweight heuristic sufficient for the demo (Hindi/Hinglish + English).
NON_ENGLISH_HINTS = re.compile(
    r"[\u0900-\u097F]"  # Devanagari script
    r"|\b(kitna|kitne|batao|kya|bikri|kripya|dikhao)\b", re.I
)

HINDI_EN_NORMALIZATION = {
    "kitna": "how much",
    "kitne": "how many",
    "kya": "what",
    "batao": "tell",
    "dikhao": "show",
    "bikri": "sales",
    "becha": "sold",
    "hua": "happened",
    "kitni": "how much",
    "rajasv": "revenue",
    "uttar": "north",
    "dakshin": "south",
    "purab": "east",
    "paschim": "west",
    "mahina": "month",
    "saal": "year",
    "ka": "of",
    "ki": "of",
    "ke": "of",
    "me": "in",
    "mein": "in",
}


@dataclass
class IntentResult:
    turn_type: str  # greeting | capability | out_of_scope | ambiguous | question | chitchat_close
    canonical_query: Optional[str] = None
    resolved_entities: Dict[str, Any] = field(default_factory=dict)
    clarification_question: Optional[str] = None
    detected_non_english: bool = False
    notes: List[str] = field(default_factory=list)


def _fuzzy_correct(token: str, vocab: Dict[str, str]) -> Optional[str]:
    # only single words of reasonable length are eligible; bigrams are
    # excluded to avoid spurious matches like "did nutrioat" -> NutriOat
    if len(token) < 4:
        return None
    matches = difflib.get_close_matches(token.lower(), vocab.keys(), n=1, cutoff=0.85)
    return vocab[matches[0]] if matches else None


KNOWN_REGIONS = {"north", "south", "east", "west"}
DATA_COVERAGE_YEARS = {2024, 2025}


def normalize_entities(text: str) -> Dict[str, Any]:
    """Alias + typo correction for brand/region/KPI mentions."""
    resolved: Dict[str, Any] = {}
    words = re.findall(r"[a-zA-Z]+", text.lower())
    bigrams = [f"{a} {b}" for a, b in zip(words, words[1:])]

    # exact multi-word aliases first (e.g. "nutri oat")
    for cand in bigrams:
        if cand in BRAND_ALIASES:
            resolved["brand"] = BRAND_ALIASES[cand]
        if cand in KPI_ALIASES:
            resolved["kpi"] = KPI_ALIASES[cand]

    matched_brands = []
    for cand in words:
        if cand in BRAND_ALIASES:
            resolved["brand"] = BRAND_ALIASES[cand]
            if BRAND_ALIASES[cand] not in matched_brands:
                matched_brands.append(BRAND_ALIASES[cand])
        elif "brand" not in resolved and (brand := _fuzzy_correct(cand, BRAND_ALIASES)):
            resolved["brand"] = brand
            resolved.setdefault("_typo_corrected", []).append((cand, brand))
        if cand in REGION_ALIASES and len(cand) > 1:  # avoid 1-letter false positives on common words
            resolved["region"] = REGION_ALIASES[cand]
        if cand in KPI_ALIASES:
            resolved["kpi"] = KPI_ALIASES[cand]
    if len(matched_brands) > 1:
        resolved["brands"] = matched_brands  # supports multi-brand comparisons

    # unsupported region: "<Word> region" where <Word> isn't one of the four
    # known regions (hierarchy-aware fallback requirement)
    region_mention = re.search(r"\b([a-zA-Z]+)\s+region\b", text, re.I)
    if region_mention and region_mention.group(1).lower() not in KNOWN_REGIONS:
        resolved["_unsupported_region"] = region_mention.group(1)

    month_match = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", text)
    if month_match:
        resolved["month"] = f"{month_match.group(1)}-{int(month_match.group(2)):02d}"

    # bare 4-digit year outside the warehouse's coverage window (2024-2025) —
    # flagged even without a month, e.g. "sales in 2020"
    for year_match in re.finditer(r"\b(19|20)\d{2}\b", text):
        year = int(year_match.group(0))
        if year not in DATA_COVERAGE_YEARS:
            resolved["_out_of_range_year"] = year
            break

    return resolved


def normalize_multilingual_query(text: str) -> str:
    """Map common Hindi/Hinglish business terms to canonical English hints."""
    normalized = text
    for src, tgt in HINDI_EN_NORMALIZATION.items():
        normalized = re.sub(rf"\b{re.escape(src)}\b", tgt, normalized, flags=re.I)
    return re.sub(r"\s+", " ", normalized).strip()


class IntentRouter:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def classify(self, text: str, memory) -> IntentResult:
        stripped = text.strip()
        canonical = normalize_multilingual_query(stripped)

        if GREETING_RE.match(canonical) and len(canonical.split()) <= 5:
            return IntentResult(turn_type="greeting")
        if THANKS_BYE_RE.match(canonical):
            return IntentResult(turn_type="chitchat_close")
        if CAPABILITY_RE.search(canonical):
            return IntentResult(turn_type="capability")

        non_english = bool(NON_ENGLISH_HINTS.search(stripped))
        entities = normalize_entities(canonical)

        # merge with last-resolved entities from memory for follow-up resolution
        merged = dict(memory.last_resolved_entities)
        merged.update(entities)

        # cheap ambiguity heuristic: very short, vague follow-ups with no
        # resolvable entity at all and no prior context to fall back on
        vague_followup = bool(re.match(r"^\s*(and (last|this) (month|year)|what about (it|that)|more)\W*$",
                                        canonical, re.I))
        if vague_followup and not merged:
            return IntentResult(
                turn_type="ambiguous",
                clarification_question="Could you clarify which brand, region, or KPI you'd like me to "
                                        "follow up on? I don't have enough context yet from this session.",
            )

        # domain scope check — keyword gate first (cheap), LLM fallback only if
        # the gate is inconclusive (keeps cost down; see cost-latency doc).
        # Checked primarily against THIS turn's own entities/keywords, not the
        # merged (memory-carried) entities — otherwise a brand-new, unrelated
        # question ("what's the capital of France?") would be treated as
        # in-scope just because a brand was mentioned earlier. The one
        # exception: a SHORT message (<=8 words) when the session already has
        # resolved context is very likely a genuine follow-up ("what about
        # October instead?") rather than a topic change, so it's allowed to
        # inherit scope from memory. Longer messages must stand on their own
        # keywords/entities regardless of prior context. This is a precision/
        # recall trade-off documented in docs/design-decisions.md.
        in_domain_hint = any(k in canonical.lower() for k in (
            "sale", "sku", "brand", "region", "revenue", "unit", "stock", "inventory",
            "campaign", "discount", "promo", "kpi", "quarter", "month", "growth", "market",
            "fmcg", "industry", "benchmark", "penetration", "competitor", "category"
        )) or bool(entities) or (bool(merged) and len(canonical.split()) <= 8)

        if not in_domain_hint:
            verdict = self._llm_scope_check(canonical)
            if verdict == "out_of_scope":
                return IntentResult(turn_type="out_of_scope")

        result = IntentResult(
            turn_type="question",
            canonical_query=canonical,
            resolved_entities=merged,
            detected_non_english=non_english,
        )
        if "_typo_corrected" in entities:
            result.notes.append(f"Auto-corrected: {entities['_typo_corrected']}")
        if "_unsupported_region" in entities:
            result.notes.append(
                f"'{entities['_unsupported_region']}' isn't a tracked region in this dataset "
                f"(supported: North, South, East, West) — showing all supported regions instead."
            )
        if "_out_of_range_year" in entities:
            result.notes.append(
                f"Requested year {entities['_out_of_range_year']} is outside this warehouse's data "
                f"coverage (2024-07 to 2025-06) — no data available for that period."
            )
        return result

    def _llm_scope_check(self, text: str) -> str:
        # NOTE: this LLM call is only reached when the cheap keyword gate
        # (see caller) found NO domain keywords and NO resolvable entities in
        # the question — i.e. it's already a fairly strong out-of-scope
        # signal. In mock mode there's no real classifier to fall back on, so
        # we default to 'out_of_scope' here rather than 'in_scope': it's the
        # answer a real classifier would almost always give given the gate
        # already failed, and it avoids the mock agent confidently answering
        # unrelated questions (e.g. "capital of France") with FMCG data.
        r = self.llm.complete(
            system=(
                "You classify whether a user question is about FMCG business analytics "
                "(sales, brands, SKUs, regions, campaigns, market reports) or something "
                "unrelated. Reply with exactly one word: 'in_scope' or 'out_of_scope'."
            ),
            messages=[{"role": "user", "content": text}],
            max_tokens=5,
            mock_responder=lambda s, m: "out_of_scope",
        )
        return "out_of_scope" if "out_of_scope" in r.text.lower() else "in_scope"
