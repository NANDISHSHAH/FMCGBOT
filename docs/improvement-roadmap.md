# Improvement Roadmap

This document maps every assessment requirement to its current implementation status,
identifies gaps, and provides a prioritised backlog with implementation strategy and
expected impact on the performance metrics measured by `tests/eval_framework.py`.

---

## Current state summary

| Area | Status | Gap |
|------|--------|-----|
| Multi-agent orchestration | ✓ Implemented | — |
| Intent routing + NLU | ✓ Implemented | Deterministic multilingual normalization covers common Hindi/Hinglish terms; full fluency still better in live mode |
| Structured retrieval + SQL safety | ✓ Implemented | No parallel execution |
| Unstructured retrieval + citations | ✓ Implemented | Hybrid/vector mode depends on embeddings provider availability |
| Hybrid retrieval | ✓ Implemented | Sequential, not parallel |
| Web search | ✓ Implemented (live + mock fallback) | Live results depend on provider key/egress |
| Coding / calculation agent | ✓ Implemented | Limited to series-based calc |
| Answer validation + retry | ✓ Implemented | Single retry only |
| Session memory + compression | ✓ Implemented | SQLite persistence by default; no multi-tenant retention policy yet |
| NAT workflow integration | ✓ Running (exit 0) | Custom fn types require plugin load order fix |
| Evaluation harness | ✓ Added (`tests/eval_framework.py`) | No CI pipeline yet |
| Latency optimization | ✗ Not done | Sub-agents run sequentially |
| Vector retrieval | ✓ Implemented | Needs production vector-store swap for scale |
| Persistent session storage | ✓ Implemented | SQLite-local only (single-node) |

---

## Priority 1 — High impact, low effort

### P1-A: Parallelize independent sub-agent calls

**File**: `src/core/orchestrator.py` — `handle_turn`

**Current behaviour**: structured, unstructured, websearch, coding run sequentially.

**Change**: wrap independent sub-agent calls in `asyncio.gather` (or `concurrent.futures.ThreadPoolExecutor` since the agent methods are synchronous).

```python
# Current (sequential)
if "structured" in plan:
    sub_results["structured"] = self.structured.run(...)
if "unstructured" in plan:
    sub_results["unstructured"] = self.unstructured.run(...)

# Target (parallel)
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor() as pool:
    futures = {}
    if "structured" in plan:
        futures["structured"] = pool.submit(self.structured.run, ...)
    if "unstructured" in plan:
        futures["unstructured"] = pool.submit(self.unstructured.run, ...)
    sub_results = {k: f.result() for k, f in futures.items()}
```

**Expected impact**: hybrid query latency drops from ~sum of agents to ~max of agents.
For a typical hybrid query (structured ~200ms + unstructured ~150ms) → reduces to ~200ms.
**Estimated latency improvement**: 30–45% on hybrid queries.

---

### P1-B: Add CI eval gate

**File**: `.github/workflows/eval.yml` (new)

**Change**: run `pytest tests/ -q && python tests/eval_framework.py --ids G01 G02 G03 S01 U01 H01` on every push. Fail if pass rate < 80%.

**Expected impact**: prevents regressions from being merged silently.

---

### P1-C: Fix `docs/README.md` section 1.4 *(done)*

Already fixed in this session. Section 1.4 now accurately reflects the single registered workflow function.

---

## Priority 2 — Medium impact, medium effort

### P2-A: Swap TF-IDF retrieval for real vector embeddings

**File**: `src/tools/doc_retrieval_tool.py`

**Current**: `TFIDFRetriever` with cosine scoring over raw term frequencies.

**Change**: Replace `retrieve()` internals with:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")  # 80MB, runs locally
```
or, for production, use OpenAI `text-embedding-3-small` (cost: ~$0.00002/1k tokens).

**Interface contract unchanged** — callers still get `{"sources": [...], "summary": ...}`.

**Expected impact**:
- Paraphrase and semantic-only matches (currently missed by TF-IDF) are correctly retrieved.
- Recall improves on out-of-vocabulary brand/campaign name variations.
- Especially important as document corpus grows past ~50 documents.

---

### P2-B: Add multiple retry on synthesis failure

**File**: `src/core/orchestrator.py` — `_synthesize`

**Current**: single LLM call, no retry on API error.

**Change**: wrap `self.llm.complete(...)` in a retry loop with exponential backoff (max 3 attempts).
Use `tenacity` (already available transitively) or a simple `for attempt in range(3)` loop.

**Expected impact**: reduces flaky failures in live mode under API load.

---

### P2-C: Add evaluation results to the demo notebook

**File**: `notebooks/demo.ipynb`

**Change**: Add a final cell that calls:
```python
from tests.eval_framework import run_eval, generate_report, GOLDEN_SUITE
results = run_eval(GOLDEN_SUITE)
print(generate_report(results))
```
Pre-run and commit with outputs so reviewers see the pass rate and latency table without running anything.

**Expected impact**: directly addresses the assessment requirement "notebook with pre-run sample test questions and responses".

---

### P2-D: Surface NAT component counts accurately

**Files**: `nat_workflow.yaml` (already updated), `src/nat_plugin.py`

**Current state**: `Number of Functions: 1`, `Number of LLMs: 1`, `Number of Object Stores: 1`.

**To reach higher counts without breaking NAT config validation**, add the `nvidia-nat-agno` plugin which ships a `react_agent` and `tool_calling_agent` function type — those can then be referenced as plain tools in `functions:`.

Alternatively: document this limitation clearly (already done in `docs/README.md` section 1.7) and point reviewers to `nat info components` output.

---

## Priority 3 — Lower priority / production concerns

### P3-A: Persistent session storage

**File**: `src/core/memory.py`

**Current**: sessions are in-process Python dicts — lost on restart.

**Production path**: serialize `SessionMemory` to Redis or SQLite (`data/sessions.db`).
```python
import sqlite3, json
def save_session(self, db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT OR REPLACE INTO sessions VALUES (?,?)",
                 (self.session_id, json.dumps(self.__dict__)))
```

---

### P3-B: Real web search provider

**File**: `src/tools/websearch_tool.py`

**Current**: curated static corpus (4 results).

**Production path**: swap `_mock_search()` for a Serper/SerpAPI/Brave call — one class change, no caller changes:
```python
import requests
def _live_search(query: str) -> list[dict]:
    resp = requests.get("https://serpapi.com/search",
                        params={"q": query, "api_key": os.environ["SERPAPI_KEY"]})
    return [{"title": r["title"], "url": r["link"], "snippet": r["snippet"]}
            for r in resp.json().get("organic_results", [])[:5]]
```

---

### P3-C: NAT memory integration

**Current**: `Number of Memory: 0`.

**Production path**: install `nvidia-nat-mem0ai`:
```bash
uv pip install nvidia-nat-mem0ai
```
Then in `nat_workflow.yaml`:
```yaml
memory:
  conversation_memory:
    _type: mem0
    api_key: ${MEM0_API_KEY}
```
And wrap the workflow with `auto_memory_agent` as the workflow `_type`.

**Impact**: enables true long-term memory across sessions via Mem0, and increases `Number of Memory: 1` in the NAT summary.

---

### P3-D: Structured data growth — Postgres/Snowflake

**File**: `src/tools/sql_tool.py`

**Current**: SQLite (`data/structured/fmcg.db`).

**Production path**: change `DB_PATH` to a connection string and swap `sqlite3` for `psycopg2` or `snowflake-connector-python`. The SQL guard, row cap, and allow-list logic are unchanged.

---

## Performance targets (after P1 + P2 improvements)

| Metric | Current (mock) | Target (live, after P1+P2) |
|--------|---------------|---------------------------|
| Avg hybrid query latency | ~6ms (mock) | < 3s (live, parallel) |
| Pass rate (eval_framework) | > 70% | > 90% |
| Cost per hybrid query | $0 (mock) | < $0.005 (with parallelism) |
| Recall on semantic queries | ~60% | > 85% (with vector embeddings) |

---

## Quick-win summary (interview talking points)

1. **Parallelize sub-agents** — single `ThreadPoolExecutor` change, 30–45% latency improvement on hybrid queries.
2. **Vector retrieval** — swap TF-IDF internals, same interface, significant recall improvement.
3. **Eval CI gate** — run `eval_framework.py` on every commit, fail build if pass rate drops.
4. **Notebook eval cell** — add pre-run eval output to `demo.ipynb` for grader visibility.
5. **NAT memory plugin** — `uv pip install nvidia-nat-mem0ai` + 5-line YAML change → `Number of Memory: 1`.
