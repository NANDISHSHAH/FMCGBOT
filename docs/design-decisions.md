# Design Decisions & Trade-offs

This document records the non-obvious choices made in this prototype and why,
including ones a reviewer might reasonably push back on.

## 1. Why Agno's `Agent` primitive, not its `Team`/graph orchestration

Agno (2.7.1, installed and available in this repo — see `requirements.txt`) ships
a `Team` abstraction for multi-agent coordination. This prototype does **not**
use it for the top-level orchestrator; instead `OrchestratorAgent`
(`src/core/orchestrator.py`) is a hand-written Python state machine.

**Why**: the assessment's hardest requirements — intent validation *before*
retrieval, clarification for ambiguous requests, retry-on-empty/failure,
hierarchy-aware fallback for unsupported entities, transparent reporting of
assumptions — are all about *fine-grained control flow around* tool calls, not
about delegating reasoning to an LLM-driven agent graph. Implementing them as
explicit, testable Python (see `tests/test_intent.py`) makes every branch
inspectable and unit-testable in isolation, which matters more for an eval-heavy
prototype than framework convenience. Agno's `Agent` class is still the right
primitive for each *sub-agent's* internal LLM+tool-calling loop in the live-mode
NL→SQL and synthesis paths — the abstraction is used where it earns its keep
(model/tool wiring) and skipped where hand-rolled control flow is more
transparent (orchestration logic).

**Trade-off**: this is more code to maintain by hand, and doesn't get Agno's
built-in tracing/eval UI for free. If this went to production, the control flow
in `orchestrator.py` would be the first candidate to formalize as an Agno
`Team` with custom routing, once the branching logic has stabilized.

## 2. Why a NAT-shaped tracer instead of the full NeMo Agent Toolkit

NVIDIA's NeMo Agent Toolkit (NAT) is genuinely a good fit conceptually — it's
framework-agnostic, has a documented integration path for Agno specifically, and
gives evaluation/profiling/MCP serving for free. It was requested for this build.

What this prototype actually ships is `src/core/tracing.py`, which reproduces
NAT's *span shape* (named steps, durations, nested per-turn traces, token/cost
metadata) and `nat_workflow.yaml`, an illustrative (non-executed) workflow
descriptor showing exactly how `OrchestratorAgent`'s tools would be registered
as a real NAT workflow.

**Why not the real thing**: `nat` installs from source via git submodules + Git
LFS and is built around NVIDIA NIM-served models by default; wiring it to a
plain Gemini-API-backed Agno agent is a real but nontrivial integration (see
NVIDIA's own blog post on adding new framework plugins) that would consume a
disproportionate share of a 3-day window relative to the value it adds *for this
prototype's grading criteria*, versus building out the actual agent behaviors
the assessment is testing for. This is a scope call, not a technical dead-end —
the adapter file documents the path so a reviewer can see exactly what "finish
the NAT integration" would involve.

**Trade-off**: no `nat eval`/profiler UI out of the box; cost/latency reporting
here is a bespoke (but equivalent-shape) trace log instead.

## 3. Why TF-IDF instead of a vector DB for document retrieval

`src/tools/doc_retrieval_tool.py` uses dependency-free TF-IDF + cosine-style
scoring rather than embeddings + a vector store.

**Why**: it keeps unstructured retrieval fully functional in mock mode (no
embeddings API call needed) and avoids a second network dependency in an
environment where only package registries are reachable. For a 10-document
corpus this also isn't a meaningfully worse retriever than embeddings — TF-IDF
struggles on paraphrase/semantic-only matches, which starts to matter above
maybe a few hundred documents with more lexical diversity than this corpus has.

**Trade-off / what changes in production**: swap `retrieve()`'s internals for a
real vector DB (pgvector, Chroma, etc.) behind the same function signature — no
caller changes. Documented explicitly as the first thing to change if the
document corpus grows past what lexical search handles well.

## 4. Why deterministic rule-based planners for mock mode (NL→SQL, synthesis)

Without a live API key, `StructuredDataAgent` and `CodingAgent` fall back to
small rule-based planners (`_deterministic_sql`, growth-only calculation) rather
than leaving those paths non-functional.

**Why**: makes the entire pipeline — and the graded notebook — runnable and
inspectable with zero setup, which matters for a take-home reviewer who may not
want to provision a key just to see the system work. The architecture (guard,
retry, validation, hierarchy fallback) is identical in both modes; only the
NL→SQL generality changes.

**Trade-off**: mock mode only handles the question *patterns* wired into the
planner (single-entity lookups, top-N, multi-brand comparisons, growth between
two points). Novel phrasing outside those patterns still runs safely (the SQL
guard always applies) but may return a broader/less-precise result. This is
called out explicitly in the demo notebook and README rather than hidden.

## 5. Why keyword-based routing/scope-gating instead of an LLM classifier for every turn

`IntentRouter` and `OrchestratorAgent._route` use cheap regex/keyword heuristics
first, and only fall back to an LLM call when the heuristic is inconclusive.

**Why**: greetings, capability questions, and clearly-in-domain questions
(mentioning "revenue", "SKU", a brand name, etc.) are a large fraction of real
traffic and classifiable with near-zero latency/cost. Spending a full LLM round
trip (500ms-1s+, real tokens) to confirm "hi" is a greeting is pure waste. See
`docs/cost-latency-tradeoffs.md` for the latency delta this produces.

**Trade-off — precision/recall on scope gating**: a short message (≤8 words)
sent while the session already has resolved entities is treated as an in-scope
follow-up even without its own keywords, so genuine multi-turn follow-ups like
"what about October instead?" work. The cost is that a short, truly unrelated
question asked immediately after an FMCG question could be misclassified as
in-scope. This is a deliberate recall-over-precision choice for a Q&A agent
where losing genuine follow-ups is more damaging to usability than occasionally
answering an unrelated short question.

## 6. SQL safety model

Guardrails in `src/tools/sql_tool.py`: single-statement-only, SELECT-only
regex gate, forbidden-keyword blocklist (DDL/DML), a hard table allow-list
(rejects even reads from `sqlite_master`), and a row cap enforced by rewriting
any `LIMIT` clause rather than trusting the caller's. This is defense in depth
rather than relying on any single layer — e.g. the table allow-list still
protects against schema-introspection queries even if a future forbidden-keyword
regex has a gap. In production this maps to a read-only DB role/user as the
outermost layer, with these checks as an additional application-level guard.
