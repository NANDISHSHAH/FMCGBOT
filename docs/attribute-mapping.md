# Requirement → Implementation Mapping

Every bullet from the assessment prompt, mapped to the code that implements it and
the notebook cell that demonstrates it.

| # | Requirement | Implementation | Demo (notebooks/demo.ipynb) |
|---|---|---|---|
| 1 | Single-turn and multi-turn conversational interactions | `SessionMemory` + `QNAAgent.chat(session_id=...)` | Cells 5–6 (same session, follow-up) |
| 2 | Greeting, capability introduction, out-of-scope handling | `IntentRouter.classify` → `greeting`/`capability`/`out_of_scope` branches in `orchestrator.py` | Cells 1–3 |
| 3 | Intent validation before data retrieval | `IntentRouter.classify` runs and short-circuits *before* any sub-agent call in `handle_turn` | All question cells (routing happens after classification) |
| 4 | Clarification for ambiguous/incomplete requests | `IntentResult(turn_type="ambiguous", clarification_question=...)` in `intent.py` | Cell 4 |
| 5 | Contextual follow-up via conversation history | `SessionMemory.last_resolved_entities` merged into each turn's entity resolution | Cell 6 |
| 6 | Semantic understanding: aliases, abbreviations, typos | `BRAND_ALIASES`/`REGION_ALIASES`/`KPI_ALIASES` + `difflib`-based fuzzy correction | Cell 7 |
| 7 | Multilingual / mixed-language queries | `intent.normalize_multilingual_query()` maps common Hindi/Hinglish terms to canonical English query intent before routing + entity resolution (`NON_ENGLISH_HINTS` retained for detection) | Cell 8 |
| 8 | Conversation context preservation across interactions | `SessionMemory` persists turns/summary/entities to SQLite (`data/session_memory.sqlite`) by `session_id`, with in-memory fallback | Cells 5–6, 17 |
| 9 | Secure access with SQL safety controls | `src/tools/sql_tool.py`: SELECT-only, table allow-list, forbidden-keyword blocklist, row cap | `tests/test_sql_guard.py` (6 tests) |
| 10 | Structured + unstructured retrieval from multiple sources | `StructuredDataAgent` (SQLite) + `UnstructuredDataAgent` (markdown corpus) | Cell 5 (structured), Cell 10 (unstructured) |
| 11 | Document retrieval with source citations | `doc_retrieval_tool.retrieve()` returns `source`/`published`/`doc_type` per hit; surfaced in every answer that cites documents | Cells 9–10 |
| 12 | Hybrid data retrieval | `orchestrator._route` sends "why did X happen" style questions to structured **and** unstructured (**and** websearch when relevant) in the same turn | Cell 9 |
| 13 | Answer validation, retry mechanisms, response quality evaluation | `orchestrator._validate` + `StructuredDataAgent`'s one-retry-on-failure/empty-result logic | Cell 16 (out-of-range query surfaces a validation note instead of a guess) |
| 14 | Standardized formatting: markdown tables, unit-aware presentation | `src/core/formatting.py`: `rows_to_markdown_table`, `fmt_number` (₹, units, days) | Cell 5 |
| 15 | Temporal reasoning: current/historical/comparative periods | Month-grain SQL filtering + `CodingAgent` for period-over-period growth | Cell 12 |
| 16 | Context-aware follow-up suggestions within supported domains | `orchestrator._followups` generates suggestions from the entities actually resolved that turn | Cell 18 |
| 17 | Conversation memory optimization for long-running sessions | `SessionMemory.maybe_compress`: keeps last 6 turns verbatim, folds older turns into a rolling LLM summary | Cell 17 |
| 18 | Metadata queries | `sql_tool.get_metadata()` + `doc_retrieval_tool.list_metadata()` | Cell 13 |
| 19 | Multiple KPIs, entities, dimensions, hierarchical business structures | Schema: brand→SKU, region, channel, month, campaign; `KPI_ALIASES` covers 6 KPIs | `docs/architecture.md` §Data model |
| 20 | Analytical comparisons across KPIs, entities, periods, domains | Multi-brand `IN (...)` comparison path in `StructuredDataAgent._deterministic_sql` | Cell 11 |
| 21 | Hierarchy-aware fallback for unsupported entities/granularities | Unsupported-region detection in `intent.normalize_entities` → explicit fallback note + broadened query | Cell 14 |
| 22 | Metadata discovery for available KPIs/dimensions/periods/datasets | Same as #18 — `get_metadata()` returns coverage period, dimensions, KPI list | Cell 13 |
| 23 | Document filtering using metadata, tags, and recency | `doc_retrieval_tool.retrieve(category=..., doc_type=...)` + `superseded` tag down-weighting + recency sort | Cell 10 (superseded vs. current finance note) |
| 24 | Transparent reporting of assumptions, data availability, system limitations | `orchestrator._validate` notes surfaced in every answer; explicit "auto-corrected"/"unsupported region"/"out of range" notes from `intent.py` | Cells 7, 14, 16 |
| 25 | Graceful handling of unsupported/unavailable requests | Out-of-scope branch (#2) + out-of-range-year short-circuit in `orchestrator._route` + empty-result fallback notes | Cells 3, 16 |

Sub-agent requirement (structured / unstructured / internet search / coding) is
covered by `src/agents/structured_agent.py`, `unstructured_agent.py`,
`websearch_agent.py`, `coding_agent.py` respectively, all invoked from
`OrchestratorAgent._route` → `handle_turn`.
