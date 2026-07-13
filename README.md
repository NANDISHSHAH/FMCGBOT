# FMCGQABOT: Enterprise FMCG Q&A Agent Prototype

FMCGQABOT is a multi-agent question-answering system over synthetic FMCG business data. It combines structured analytics, unstructured document retrieval, web context, and code-based calculations behind a single orchestration layer.

The implementation is designed to satisfy an enterprise-style assessment with explicit capabilities for intent validation, clarification, SQL safety, hybrid retrieval, citations, memory, and answer quality controls.

## What This System Does

- Answers business questions using a coordinated set of sub-agents:
  - Structured data retrieval (SQL over warehouse data)
  - Unstructured retrieval (document search with metadata and recency handling)
  - Internet search (mock/live pluggable provider)
  - Coding/calculation agent for derived metrics
- Supports both single-turn and multi-turn conversations
- Handles greeting, capability, out-of-scope, and ambiguous user requests
- Performs intent validation before retrieval
- Preserves context with session memory and rolling compression for long chats
- Produces grounded, formatted responses with citations and follow-up suggestions

## High-Level Implementation

Core implementation modules:

- Orchestration and control flow:
  - src/core/orchestrator.py
  - src/core/intent.py
  - src/core/memory.py
  - src/core/tracing.py
- Sub-agents:
  - src/agents/structured_agent.py
  - src/agents/unstructured_agent.py
  - src/agents/websearch_agent.py
  - src/agents/coding_agent.py
- Tools:
  - src/tools/sql_tool.py
  - src/tools/doc_retrieval_tool.py
  - src/tools/websearch_tool.py
  - src/tools/code_tool.py
- Entry point:
  - src/qna_agent.py

For full architecture and flow diagrams, see docs/architecture.md.

## Data Assets

The project includes overlapping entities and themes across structured and unstructured sources.

- Structured warehouse:
  - data/structured/fmcg.db
- Unstructured documents:
  - data/unstructured/*.md

These datasets are generated/curated to support hybrid questions such as:
- performance by brand/region/period
- "why" and campaign-effect analysis
- recency-sensitive policy guidance lookup

## Key Enterprise Capabilities Covered

Implemented capabilities include:

- Intent validation before retrieval
- Clarification for ambiguous requests
- Contextual follow-ups and conversation history usage
- Alias/abbreviation/typo normalization
- SQL safety controls (read-only, allow-list, row cap, statement checks)
- Structured and unstructured multi-source retrieval
- Source citations in responses
- Hybrid retrieval and synthesis
- Validation notes, fallback, and transparent limitations
- Unit-aware formatting and markdown table rendering
- Temporal and comparative analysis patterns
- Metadata discovery and document filtering by metadata/recency

For exact mapping from requirement to implementation, see docs/attribute-mapping.md.

## Getting Started

### Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd FMCGBOT

# With uv (recommended)
uv pip install -e .

# Or with pip
pip install -r requirements.txt
```

Set up environment variables (optional, see [.env.example](.env.example) for all options):

```bash
cp .env.example .env
# Edit .env with your API keys (or set env vars in your shell)
```

Generate synthetic FMCG data (one-time setup):

```bash
python src/data_gen/generate_data.py
```

This creates `data/structured/fmcg.db` and documents in `data/unstructured/`.

### Quick Verification

Run tests to ensure the environment is working:

```bash
pytest tests/ -q
```

Expected output: `12 passed`

### Run Modes

#### Mode 1: Mock Mode (Free, Instant)

No API key required. Uses deterministic planners for fast, reproducible responses.

```bash
# Python one-liner
python -c "
from src.qna_agent import QNAAgent
agent = QNAAgent()
result = agent.chat('How did NutriOat Gold do in North during the festive campaign?')
print(result['answer'])
"

# Or in a script
from src.qna_agent import QNAAgent
agent = QNAAgent()
result = agent.chat("How many units did SunFresh sell in North in November?", session_id="s1")
print(result['answer'])
```

#### Mode 2: Live Mode (OpenAI/Anthropic/Gemini)

Set your provider credentials, then run:

```bash
# OpenAI setup (recommended)
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...

# Then run the agent
python -c "
from src.qna_agent import QNAAgent
agent = QNAAgent()
result = agent.chat('Why did NutriOat Gold grow in North? What is the industry outlook?')
print(result['answer'])
"
```

Supported providers: `openai`, `anthropic`, `gemini`

#### Mode 3: Interactive Streamlit UI

The UI provides a chat interface with citations, sources, session management, and live tracing.

**Install UI dependencies:**

```bash
uv pip install -e ".[ui]"
```

**Quick start with .env file (Recommended):**

```bash
# Copy template and fill in your API keys
cp .env.example .env
# Edit .env: set LLM_PROVIDER, LLM_MODEL, and your API key (OPENAI_API_KEY, etc.)

# Launch UI (automatically loads .env)
streamlit run src/ui.py
```

The app opens at `http://localhost:8501`. The sidebar will show **Live (PROVIDER_NAME)** when connected.

**Alternative: Use environment variables:**

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...

streamlit run src/ui.py
```

**UI Features:**

- Automatic .env file loading (no manual export needed)
- Session sidebar: create, switch, and reset chat sessions
- Mock/live indicator badge showing active provider
- Citations and source references for each answer
- Suggested follow-up questions (clickable)
- Trace summary: latency, token usage, estimated cost
- Example question buttons for quick exploration

#### Mode 4: Notebook

Pre-run demonstrations of all capabilities:

```bash
uv run jupyter notebook notebooks/demo.ipynb
```

The notebook runs in the same environment as your CLI setup (picks up `LLM_PROVIDER` and `OPENAI_API_KEY` from shell).

#### Mode 5: NeMo Agent Toolkit (NAT)

Register the agent as a NAT workflow:

```bash
uv pip install -e ".[nat]"

uv run nat run --config_file nat_workflow.yaml --input "How did NutriOat Gold do in North during the festive campaign?"
```

For multi-turn sessions with NAT, edit `session_id` in `nat_workflow.yaml`.

### Evaluation

Run the comprehensive evaluation harness:

```bash
# Mock mode (free)
uv run python tests/eval_framework.py

# Specific test cases only
uv run python tests/eval_framework.py --ids H01 H02 S01

# Write report to file
uv run python tests/eval_framework.py --output eval_report.md
```

Output includes:
- Pass/fail by test case
- Latency and token usage per query
- Estimated cost breakdown
- Requirement coverage summary

## Optional Configuration

### Backend Providers

Configure optional backends for web search and retrieval:

```bash
# Live web search (optional; defaults to curated mock corpus)
export WEBSEARCH_PROVIDER=serpapi  # or tavily
export SERPAPI_API_KEY=...

# Unstructured retrieval mode
export DOC_RETRIEVAL_MODE=hybrid    # hybrid | vector | tfidf (default: hybrid)
export OPENAI_EMBED_MODEL=text-embedding-3-small

# Session memory backend
export SESSION_MEMORY_BACKEND=sqlite    # or inmemory (default: sqlite)
export SESSION_MEMORY_DB_PATH=data/session_memory.sqlite
```

### LLM Provider Selection

Precedence: explicit argument > `LLM_PROVIDER` env > model name prefix

**OpenAI:**

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...
# Optional: export OPENAI_BASE_URL=https://api.openai.com/v1
```

**Anthropic:**

```bash
export LLM_PROVIDER=anthropic
export LLM_MODEL=claude-3-5-haiku-latest
export ANTHROPIC_API_KEY=...
```

**Google Gemini:**

```bash
export LLM_PROVIDER=gemini
export LLM_MODEL=gemini-2.5-pro
export GOOGLE_API_KEY=...  # or GEMINI_API_KEY
```

## Evaluation and Metrics

Evaluation harness:

- tests/eval_framework.py

It runs a golden suite and reports:

- pass/fail by case
- latency per query and category
- token usage
- estimated cost per query and total
- requirement coverage summary

This is the primary artifact for prompt and system-level iteration.

## NAT Integration Notes

The project exposes a custom NAT workflow function via plugin entry point and runs successfully through nat run.

Workflow file:

- nat_workflow.yaml

Plugin wrapper:

- src/nat_plugin.py

Current declarative NAT config includes built-in component types and a custom workflow entry, with orchestration logic handled inside QNAAgent.

## Repository Layout

```text
FMCGBOT/
  README.md
  requirements.txt
  pyproject.toml
  nat_workflow.yaml
  data/
    structured/
    unstructured/
  src/
    qna_agent.py
    nat_plugin.py
    agents/
    core/
    tools/
    data_gen/
  tests/
    test_intent.py
    test_sql_guard.py
    eval_framework.py
  notebooks/
    demo.ipynb
    _build_notebook.py
  docs/
    README.md
    architecture.md
    attribute-mapping.md
    design-decisions.md
    cost-latency-tradeoffs.md
    improvement-roadmap.md
```

## Design and Trade-Off Documentation

- Architecture and data flow: docs/architecture.md
- Requirement-to-implementation mapping: docs/attribute-mapping.md
- Design rationale and trade-offs: docs/design-decisions.md
- Cost/latency/model trade-off analysis: docs/cost-latency-tradeoffs.md
- Prioritized optimization roadmap: docs/improvement-roadmap.md

## What's Implemented

### Core Capabilities

- ✓ Multi-turn conversations with contextual follow-ups
- ✓ Intent validation and clarification for ambiguous requests
- ✓ Structured retrieval with SQL safety controls (SELECT-only, allow-list, row cap)
- ✓ Unstructured retrieval with source citations and metadata filtering
- ✓ Hybrid retrieval (structured + unstructured + web search)
- ✓ Web search with live provider support (SerpAPI, Tavily) and mock fallback
- ✓ Persistent session memory (SQLite backend by default)
- ✓ Multi-language support (Hindi/Hinglish normalization)
- ✓ Answer validation and retry logic
- ✓ Markdown formatting with tables and unit awareness
- ✓ Comprehensive tracing (latency, tokens, cost estimation)

### Retrieval Modes

| Mode | Retrieval | Fallback | Best For |
|---|---|---|---|
| `tfidf` | TF-IDF lexical matching | — | Fast, deterministic, low resource |
| `vector` | OpenAI embeddings (semantic) | TF-IDF | High recall, semantic paraphrases |
| `hybrid` | Vector + TF-IDF blend (default) | Vector error → TF-IDF | Best of both |

### Session Memory

| Backend | Persistence | Fallback | Use Case |
|---|---|---|---|
| `sqlite` (default) | File-backed across restarts | — | Production, multi-process |
| `inmemory` | In-process only | — | Testing, development |

### Known Constraints (Prototype Scope)

- Mock mode uses deterministic planners for parts of NL-to-SQL and synthesis
- Web search uses live providers when configured (`WEBSEARCH_PROVIDER=serpapi|tavily`) and falls back to curated mock results on failures
- Session memory is persisted in SQLite by default (`data/session_memory.sqlite`) and can be switched to in-memory mode via `SESSION_MEMORY_BACKEND=inmemory`
- Unstructured retrieval supports `hybrid` / `vector` / `tfidf` modes via `DOC_RETRIEVAL_MODE` (OpenAI embeddings when available, graceful TF-IDF fallback)

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `No module named 'openai'` | OpenAI package not installed | `uv pip install openai` or `uv pip install -e ".[openai]"` |
| `OPENAI_API_KEY is not set` | Missing API key | Set `export OPENAI_API_KEY=sk-...` |
| `ImportError: Provider 'anthropic' requires...` | Provider SDK not installed | `uv pip install -e ".[anthropic]"` or `uv pip install -e ".[all]"` |
| Empty/generic responses in mock mode | Query phrasing not recognized | Rephrase using patterns from `notebooks/demo.ipynb` or switch to live mode |
| `data/structured/fmcg.db not found` | Synthetic data not generated | Run `python src/data_gen/generate_data.py` |
| Streamlit UI shows "MOCK" badge | No API key configured | Set `export OPENAI_API_KEY=sk-...` before `streamlit run src/ui.py` |
| Tests fail with `ModuleNotFoundError` | Dependencies incomplete | `uv pip install -e .` |

## Next Steps

1. **Explore with the notebook:**
   ```bash
   uv run jupyter notebook notebooks/demo.ipynb
   ```
   Run through all cells to see the 25 requirements in action.

2. **Check the evaluation report:**
   ```bash
   uv run python tests/eval_framework.py --output eval_report.md
   cat eval_report.md
   ```

3. **For interviews/demos:**
   - Mock mode: `python -c "from src.qna_agent import QNAAgent; print(QNAAgent().chat('...').get('answer'))"`
   - Live mode: Set OpenAI API key + `streamlit run src/ui.py` for interactive exploration
   - NAT workflow: `uv run nat run --config_file nat_workflow.yaml --input "..."`

4. **Deep dive documentation:**
   - Architecture: `docs/architecture.md`
   - Requirement mapping: `docs/attribute-mapping.md`
   - Design decisions: `docs/design-decisions.md`
   - Cost/latency analysis: `docs/cost-latency-tradeoffs.md`
   - Optimization roadmap: `docs/improvement-roadmap.md`
