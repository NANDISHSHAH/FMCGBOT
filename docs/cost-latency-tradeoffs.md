# Cost, Latency & Model Usage — Point of View

This is written from actual measured numbers in mock mode (this repo's committed
state) plus explicit, labeled *estimates* for live mode. The methodology is the
same either way: every LLM/tool call is wrapped in a `Trace` span (`src/core/tracing.py`)
that records duration and, for LLM calls, tokens + estimated USD cost using the
rates in `src/core/llm_client.py::PRICE_PER_MTOK`.

## 1. Model selection strategy

Two tiers, wired into `OrchestratorAgent.__init__`:

| Tier | Model | Used for |
|---|---|---|
| Fast/cheap | `gemini-3.5-flash` | intent scope-check fallback, NL→SQL planning, coding-snippet planning |
| Strong | `gemini-3.5-pro` | document summarization, final answer synthesis |

**Reasoning**: SQL/code planning and the binary in/out-of-scope check are
narrow, well-specified generation tasks (write one SQL statement against a
known 7-table schema; write one Python expression) where Haiku-tier quality is
sufficient and the latency/cost difference compounds — these are also the calls
most likely to fire multiple times per turn (plan + one retry). Synthesis and
document summarization are the user-facing quality bar (fluent prose, correct
grounding, no invented numbers), so they get the stronger model. This mirrors a
common production pattern: cheap model for structured/narrow sub-tasks, strong
model for the final user-facing generation.

## 2. Measured latency — mock mode (this repo's committed state)

Real numbers from running the demo notebook's queries end-to-end, no network
calls at all (deterministic planners bypass the LLM client entirely when
`mock=True`, rather than simulating a delay):

| Query type | Total orchestrator latency | Sub-agent calls |
|---|---|---|
| Greeting / capability / out-of-scope | ~0.1 ms | none (short-circuited before any sub-agent) |
| Simple structured lookup | ~2 ms | structured (plan + execute) |
| Hybrid (structured + unstructured) | ~2.4 ms | structured + unstructured |
| Structured + coding (growth calc) | ~1 ms | structured + coding |

Mock-mode latency is dominated by SQLite execution and Python control flow, not
generation — useful as a lower bound / sanity check on the orchestration
overhead itself, but **not** representative of live-mode latency since it
doesn't include any network round trip.

## 3. Estimated latency & cost — live mode

No API key was available in the build environment, so these are estimates built
from (a) typical Gemini API latency for short generations (~300ms-1.2s for
Flash, ~1-2.5s for Pro depending on output length) and (b) token counts
estimated from this repo's actual prompt text (e.g. the SQL schema description
passed to the structured agent's system prompt is ~170 tokens; see
`src/tools/sql_tool.py::SCHEMA_DESCRIPTION`).

| Turn type | LLM calls | Est. total tokens (in/out) | Est. latency | Est. cost/turn |
|---|---|---|---|---|
| Greeting/capability/out-of-scope | 0 | 0 | <5 ms | $0 |
| Simple structured lookup | 1 (Haiku: SQL plan) | ~250 / ~60 | ~0.6–1.0 s | ~$0.0004 |
| Hybrid (structured + unstructured + synth) | 1 Haiku (SQL plan) + 1 Sonnet (doc summary) + 1 Sonnet (final synth) | ~250/60 (Haiku) + ~500/250 + ~600/400 (Sonnet) | ~2.5–4.5 s (sequential) | ~$0.006 |
| Structured + coding + synth | 2 Haiku (SQL + code plan) + 1 Sonnet (synth) | ~250/60 ×2 (Haiku) + ~500/300 (Sonnet) | ~2–3.5 s | ~$0.003 |
| Retry path (failed/empty SQL) | +1 Haiku call | +~280/60 | +~0.6–1.0 s | +~$0.0004 |

At **10,000 turns/day** with a traffic mix skewed toward simple/hybrid queries
(roughly 40% simple, 40% hybrid, 20% other), the estimated blended cost is
**~$35–45/day** (~$1,050–1,350/month) at current published per-token rates —
small in absolute terms, but growing linearly with traffic and *particularly*
sensitive to the number of hybrid (multi-sub-agent) queries, since each one
triggers 2-3 LLM calls instead of 1.

**Where the latency actually goes** in live mode: sub-agent calls that don't
depend on each other's output (e.g. websearch + document retrieval for the same
question) currently run **sequentially** in `orchestrator.handle_turn` — the
single highest-leverage latency optimization not yet implemented is
parallelizing independent sub-agent calls (they don't share state until the
synthesis step). This is flagged as the top item in the "if I had another day"
list.

## 4. Cost/latency levers, in priority order

1. **Parallelize independent sub-agent calls** (biggest latency win, no cost
   change) — structured/unstructured/websearch don't depend on each other's
   output for a given turn; only the coding agent depends on structured's
   result.
2. **Skip the SQL-planning LLM call when the deterministic path is confident**
   — many queries (single entity + KPI, unambiguous) don't need a full NL→SQL
   generation; a template match could shortcut straight to guarded execution,
   saving a full LLM round trip on the most common query shape.
3. **Tune hybrid retrieval weights and cache policy** — unstructured retrieval
   now supports `tfidf`, `vector`, and `hybrid` modes (`DOC_RETRIEVAL_MODE`).
   The current hybrid blend favors vector similarity with TF-IDF as lexical
   grounding; production tuning should calibrate this by query type and monitor
   precision/recall drift.
4. **Batch/stream synthesis** — for long answers, streaming the final synthesis
   call improves *perceived* latency even though total cost/wall-time is
   unchanged.

## 5. What this build deliberately did NOT optimize for

Given the assessment's 3-day window, this prototype optimizes for **architectural
correctness and testability** (guardrails, validation/retry, cited hybrid
retrieval) over runtime performance. The two items above (parallel sub-agent
calls, template-shortcut for common SQL shapes) are the concrete next steps if
this moved toward a production latency budget.

## 6. Mock vs live multilingual behavior (updated)

Mock mode now performs deterministic query normalization for common Hindi/
Hinglish FMCG phrasing before intent routing (`src/core/intent.py`), e.g.
`kitna revenue hua SunFresh ka South me?` is normalized into an English
canonical query used by downstream planners. This closes the biggest parity gap
for intent/routing and structured retrieval in mixed-language turns.

Remaining gap: live models still outperform mock mode on open-ended multilingual
explanations and nuanced paraphrases because mock mode intentionally avoids full
translation/generation calls.
