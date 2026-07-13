# Architecture

## System overview

FMCGQABOT is a multi-agent Q&A system built on a hand-rolled orchestration layer
over four specialised sub-agents, with NAT (NeMo Agent Toolkit) providing the
runtime, tracing, and component-declaration surface.

---

## High-level component diagram

```mermaid
flowchart TD
    U[User / CLI / UI / NAT run] -->|question + session_id| Q[QNAAgent]
    Q --> O[OrchestratorAgent]

    subgraph Core
        O --> I[IntentRouter]
        I -->|greeting| R1[Greeting Response]
        I -->|capability| R2[Capability Response]
        I -->|out_of_scope| R3[Out of Scope Response]
        I -->|ambiguous| R4[Clarification Question]
        I -->|question| ROUTE[Routing Logic]
        ROUTE --> SA[StructuredDataAgent]
        ROUTE --> UA[UnstructuredDataAgent]
        ROUTE --> WA[WebSearchAgent]
        ROUTE --> CA[CodingAgent]
        SA --> V[Validation]
        UA --> V
        WA --> V
        CA --> V
        V --> SYN[Synthesis]
        SYN --> MEM[SessionMemory Update]
        MEM --> OUT[Final Answer + Sources + Suggestions]
    end

    subgraph ToolLayer
        SA --> SQL[sql_tool.py]
        UA --> DR[doc_retrieval_tool.py]
        WA --> WS[websearch_tool.py]
        CA --> CT[code_tool.py]
    end

    subgraph DataLayer
        SQL --> DB[(data/structured/fmcg.db)]
        DR --> DOC[(data/unstructured/*.md)]
        WS --> EXT[External search corpus or API]
    end

    subgraph NAT
        YAML[nat_workflow.yaml] --> PLUGIN[src/nat_plugin.py]
        PLUGIN --> Q
    end

    O --> T[Trace spans in src/core/tracing.py]
    T --> M[Latency + Tokens + Cost metrics]
```

---

## Request / response data flow

```mermaid
sequenceDiagram
    participant U as User
    participant Q as QNAAgent
    participant O as Orchestrator
    participant I as IntentRouter
    participant SA as StructuredAgent
    participant UA as UnstructuredAgent
    participant V as Validator
    participant S as Synthesizer
    participant M as SessionMemory

    U->>Q: chat(question, session_id)
    Q->>M: add_user_turn(question)
    Q->>O: handle_turn(question, memory)
    O->>I: classify(question, memory)
    I-->>O: IntentResult (turn_type, entities, canonical_query)
    alt not a business question
        O-->>U: greeting / capability / out_of_scope / clarification
    end
    O->>O: _route(intent) → plan
    par parallel-ready (currently sequential)
        O->>SA: run(query, entities, trace)
        SA-->>O: {ok, rows, row_count, sql}
    and
        O->>UA: run(query, entities, trace)
        UA-->>O: {sources, summary, filters_widened}
    end
    O->>V: _validate(sub_results)
    V-->>O: validation_notes
    O->>S: _synthesize(intent, sub_results, notes)
    S-->>O: answer_text
    O->>M: update_entities + maybe_compress
    O-->>Q: {answer, sources, suggestions, trace, validation_notes}
    Q-->>U: result dict
```

---

## ASCII overview diagram

```
                              ┌─────────────────────────┐
  User turn ──────────────▶  │   OrchestratorAgent       │
                              │  (src/core/orchestrator)  │
                              └────────────┬─────────────┘
                                           │
                       ┌───────────────────┼────────────────────┐
                       ▼                                        │
             ┌───────────────────┐                              │
             │   IntentRouter      │  (src/core/intent.py)      │
             │  - greeting/capability/out-of-scope/ambiguous     │
             │  - alias + typo correction (brand/region/KPI)     │
             │  - multilingual cue detection                     │
             │  - entity carry-over from SessionMemory            │
             └───────────────────┘                              │
                       │  turn_type == "question"                │
                       ▼                                        │
             ┌───────────────────┐                              │
             │      Router         │  keyword + entity based    │
             │  (orchestrator._route)                            │
             └─────────┬─────────┘                               │
        ┌───────────────┼───────────────┬───────────────┐        │
        ▼               ▼               ▼               ▼        │
 ┌─────────────┐ ┌───────────────┐ ┌────────────┐ ┌────────────┐ │
 │ Structured    │ │ Unstructured   │ │ WebSearch   │ │  Coding    │ │
 │ Data Agent    │ │ Data Agent     │ │ Agent       │ │  Agent     │ │
 │ NL→SQL +      │ │ TF-IDF +       │ │ (mock/live  │ │ sandboxed  │ │
 │ SQL guard     │ │ metadata filter│ │  provider)  │ │ Python exec│ │
 └──────┬────────┘ └───────┬───────┘ └──────┬──────┘ └─────┬──────┘ │
        │                  │                │              │        │
        ▼                  ▼                ▼              ▼        │
 fmcg.db (SQLite)   data/unstructured/*.md  curated corpus  numeric  │
                                                              context │
        └──────────────────┴────────────────┴──────────────┘        │
                                   │                                  │
                                   ▼                                  │
                    ┌───────────────────────────┐                    │
                    │  Validation                 │  empty results?  │
                    │  (orchestrator._validate)    │  failed query?   │
                    └──────────────┬────────────┘  narrowed filters? │
                                   ▼                                  │
                    ┌───────────────────────────┐                    │
                    │  Synthesis                  │  merge + cite +  │
                    │  (orchestrator._synthesize)  │  format + retry │
                    └──────────────┬────────────┘  hints             │
                                   ▼                                  │
                    ┌───────────────────────────┐                    │
                    │  SessionMemory update        │◀──────────────────┘
                    │  (entities + rolling summary)│
                    └──────────────┬────────────┘
                                   ▼
                              Final answer
                        (markdown, cited, with
                         follow-up suggestions)
```

Every step above is wrapped in a `Trace` span (`src/core/tracing.py`) capturing
duration, and — for LLM calls — token counts and estimated cost. The full span
list for a turn is returned alongside the answer (`result["trace"]`) and is what
the cost/latency analysis in `docs/cost-latency-tradeoffs.md` is built from.

## Turn lifecycle

1. **Intent gate** (before any retrieval, per the assessment's "intent validation
   before data retrieval" requirement): greetings, capability questions, and
   out-of-scope requests are resolved here and never reach a sub-agent. Ambiguous
   follow-ups with no resolvable entity and no session context trigger a
   clarification question instead of guessing.
2. **Entity normalization**: brand/region/KPI aliases and typos are resolved
   against small dictionaries (`BRAND_ALIASES`, `REGION_ALIASES`, `KPI_ALIASES`),
   with `difflib`-based fuzzy correction for misspellings. Unsupported regions and
   out-of-data-coverage years are flagged explicitly here so downstream retrieval
   doesn't silently produce a misleading answer.
3. **Routing**: a lightweight keyword+entity heuristic decides which of the four
   sub-agents are actually relevant — most non-trivial questions ("why did X
   happen") hit more than one, which is what exercises the "hybrid data retrieval"
   requirement.
4. **Sub-agent execution**: each sub-agent is tool-first — the structured agent
   never lets the LLM touch the database directly, it only ever gets to *propose*
   a query, which is then parsed and validated by `sql_tool.py` before execution,
   with one retry on failure or unexpected-empty-result.
5. **Validation**: checks each sub-agent's result for emptiness/failure/widened
   filters and turns them into explicit notes rather than silently dropping them.
6. **Synthesis**: assembles a single answer from whatever sub-agent output exists,
   in real mode asking the LLM to turn the structured sections into fluent prose
   *without inventing new numbers* (the prompt explicitly forbids it); in mock
   mode the sections are concatenated directly (still fully grounded, just less
   fluent — see cost-latency doc).
7. **Memory update**: the turn is added to `SessionMemory`; every 6th+ older turn
   is folded into a rolling summary so long sessions don't grow the context window
   unbounded.

## Data model

See `src/tools/sql_tool.py`'s `SCHEMA_DESCRIPTION` and
`src/data_gen/generate_data.py` for the full synthetic FMCG universe: 5 brands ×
10 SKUs × 4 regions × 4 channels × 12 months of sales/discount/inventory data,
plus 5 promotional campaigns, all cross-referenced by the 10 unstructured
documents (market reports, launch memos, promo playbooks, finance notes — including
one deliberately superseded finance note, to exercise recency-aware document
filtering).
