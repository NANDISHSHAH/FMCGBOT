# FMCGQABOT — Enterprise FMCG Q&A Agent (Prototype)

A multi-agent Q&A system over synthetic FMCG business data: structured sales/inventory
data, unstructured market/finance documents, internet search, and a coding sub-agent,
coordinated by an orchestrator that does intent validation, clarification, hybrid
retrieval, answer validation/retry, and formatted, cited responses.

prompt). Every required capability is implemented and exercised in
[`notebooks/demo.ipynb`](notebooks/demo.ipynb) — see the mapping table in
[`docs/attribute-mapping.md`](docs/attribute-mapping.md).

## Quickstart

```bash
git clone <your-repo-url>
cd project-veer-qna
pip install -r requirements.txt

# 1. Generate the synthetic data (structured DB + documents)
python src/data_gen/generate_data.py

# 2. Chat with the agent
python -c "
from src.qna_agent import QNAAgent
agent = QNAAgent()
r = agent.chat('How did NutriOat Gold do in North during the festive campaign?')
print(r['answer'])
"

# 3. Run tests
pytest tests/ -q

# 4. Run the demo notebook (already pre-executed and committed with outputs)
jupyter notebook notebooks/demo.ipynb
```

**Live mode vs mock mode**: if a supported API key is set in the environment, the
agent uses real LLM calls for SQL planning, document summarization, and answer
synthesis. Without a key, it runs in a fully deterministic **mock mode** — same
architecture, same guardrails, but NL→SQL and synthesis are handled by small
rule-based planners instead of the LLM. Everything in this repo (tests, notebook)
was run and committed in mock mode; see
[`docs/cost-latency-tradeoffs.md`](docs/cost-latency-tradeoffs.md) for exactly what
degrades and what doesn't.

## Switching LLM providers

The LLM layer is adapter-based. You can use **Gemini**, **Anthropic Claude**, or
**OpenAI** without changing any agent code.

```python
from src.qna_agent import QNAAgent

# Gemini (default)
agent = QNAAgent(model="gemini-2.5-pro")

# Anthropic Claude
agent = QNAAgent(model="claude-3-5-sonnet-latest")

# OpenAI
agent = QNAAgent(model="gpt-4o")

# Or be explicit about the provider
agent = QNAAgent(model="gpt-4o", provider="openai")
```

Provider selection order:

1. Explicit `provider=` argument: `LLMClient(model="gpt-4o", provider="openai")` or `QNAAgent(model="gpt-4o", provider="openai")`
2. `LLM_PROVIDER` environment variable (`gemini`, `anthropic`, or `openai`)
3. Model-name prefix (`gemini-*`, `claude-*`, `gpt-*`, `o1*`, `o3*`)

Required API keys / packages:

| Provider | Env key(s) | Install |
|----------|------------|---------|
| Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | included in `requirements.txt` |
| Anthropic Claude | `ANTHROPIC_API_KEY` | `pip install anthropic` |
| OpenAI | `OPENAI_API_KEY` (optional `OPENAI_BASE_URL` for compatible endpoints) | `pip install openai` |

Install all provider SDKs at once with `pip install -e ".[all]"`.

## Repository layout

```
├── README.md
├── requirements.txt
├── nat_workflow.yaml            # illustrative NeMo Agent Toolkit workflow descriptor
├── data/
│   ├── structured/fmcg.db       # synthetic SQLite warehouse
│   └── unstructured/*.md        # synthetic market reports, memos, playbooks, finance notes
├── src/
│   ├── data_gen/generate_data.py
│   ├── core/
│   │   ├── llm_client.py        # unified LLM client + mock fallback + cost estimation
│   │   ├── llm_adapters.py      # swappable provider adapters (Gemini/Claude/OpenAI)
│   │   ├── tracing.py           # NAT-shaped span tracing (latency/cost/tokens per step)
│   │   ├── memory.py            # session memory + rolling summarization
│   │   ├── intent.py            # intent classification, clarification, alias/typo/lang handling
│   │   ├── orchestrator.py      # routing, synthesis, validation/retry, follow-ups
│   │   └── formatting.py        # markdown tables, unit-aware number formatting
│   ├── tools/
│   │   ├── sql_tool.py          # read-only, allow-listed, row-capped SQL execution
│   │   ├── doc_retrieval_tool.py # TF-IDF retrieval + metadata/recency filtering + citations
│   │   ├── websearch_tool.py    # pluggable web search (mock provider included)
│   │   └── code_tool.py         # sandboxed calculation execution
│   ├── agents/                  # one file per sub-agent, each wraps its tool(s)
│   └── qna_agent.py             # public entry point (QNAAgent.chat())
├── tests/                       # pytest unit tests for the guardrails + intent layer
├── notebooks/
│   ├── demo.ipynb               # pre-run test questions covering every required attribute
│   └── _build_notebook.py       # regenerates demo.ipynb deterministically
└── docs/
    ├── architecture.md
    ├── design-decisions.md
    ├── attribute-mapping.md
    └── cost-latency-tradeoffs.md
```

## Architecture (one paragraph)

A single `OrchestratorAgent` receives each user turn, classifies intent (greeting /
capability / out-of-scope / ambiguous / question) before doing any retrieval, resolves
aliases/typos/language cues, then routes to one or more of four sub-agents — structured
SQL retrieval, unstructured document retrieval, internet search, and a coding agent for
derived calculations — based on what the question actually needs (a "why did X happen"
question triggers both structured + unstructured; a pure lookup triggers only
structured). Sub-agent outputs are validated (empty results, failed queries, narrowed
document filters) and retried once if needed, then synthesized into one cited,
markdown-formatted answer with follow-up suggestions. See
[`docs/architecture.md`](docs/architecture.md) for the full diagram and data flow.

## Tech stack & why

- **Multi-provider LLM adapter** (`src/core/llm_client.py` + `src/core/llm_adapters.py`)
  supporting Gemini, Anthropic Claude, and OpenAI with a single unified interface.
- **Agno** installed and available as the per-agent LLM+tool-calling primitive
  (see `docs/design-decisions.md` for exactly how it's used vs. where the orchestration
  is hand-rolled, and why)
- **NeMo Agent Toolkit (NAT)**-shaped tracing (`src/core/tracing.py`) + an illustrative
  `nat_workflow.yaml` documenting the real integration path, without hard-depending on
  the full `nat` CLI/NIM stack in this prototype
- **SQLite** for the structured warehouse (trivially swappable for Postgres/Snowflake)
- **TF-IDF** (dependency-free) for document retrieval, standing in for a production
  vector DB — see trade-off discussion in `docs/design-decisions.md`

## Known limitations (mock mode)

- NL→SQL and document summarization use small rule-based planners instead of live LLM
  calls, so only the question patterns exercised in the demo notebook are guaranteed to
  work well; open-ended phrasing may need the real API key to get full generality.
- Web search uses a small curated corpus, not a live search API (no internet egress in
  the build environment; swapping in a real provider is a one-class change — see
  `src/tools/websearch_tool.py`).
- Multi-turn language translation isn't performed explicitly in mock mode (the live LLM
  handles this natively without a separate translation step).

## How to run

For detailed run instructions — including the **NeMo Agent Toolkit (NAT) workflow** and the **interactive Streamlit UI** — see [`docs/README.md`](docs/README.md).

### OpenAI quick run (NAT + notebook + UI)

```bash
cd /Users/nandishshah/Downloads/files/project-veer-qna

# 1) Install core deps + UI deps
uv pip install -e .
uv pip install -e ".[ui]"

# 2) Generate synthetic data once
uv run python src/data_gen/generate_data.py

# 3) Select OpenAI provider
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o
export OPENAI_API_KEY=sk-...

# 4) Sanity check provider wiring
uv run --no-env-file python -c "from src.qna_agent import QNAAgent; a=QNAAgent(); print(a.orchestrator.llm.provider, a.orchestrator.llm.model, a.mock_mode)"
```

Run each interface:

```bash
# Notebook
uv run jupyter notebook notebooks/demo.ipynb

# UI
uv run streamlit run src/ui.py

# NAT workflow (requires NAT packages installed; see docs/README.md)
uv run nat run --config_file nat_workflow.yaml --input "How did NutriOat Gold do in North during the festive campaign?"
```

### Notebook start (generic)

```bash
cd /Users/nandishshah/Downloads/files/project-veer-qna
set -a && source .env && set +a
uv run jupyter notebook notebooks/demo.ipynb
```