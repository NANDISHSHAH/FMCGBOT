# Running FMCGQABOT — NAT Workflow & Interactive UI

This guide covers the two supported ways to run the FMCGQABOT Q&A agent:

1. **NeMo Agent Toolkit (NAT) workflow** — run the agent as a registered NAT function via the `nat` CLI.
2. **Interactive Streamlit UI** — chat with the agent in a browser, with citations, trace visibility, and session management.

For programmatic use (Python API, notebook, tests), see the top-level [`README.md`](../README.md).

---

## Prerequisites

All paths below assume you are in the repo root:

```bash
cd /Users/nandishshah/Downloads/files/project-veer-qna
```

Generate the synthetic data once (creates `data/structured/fmcg.db` and the documents under `data/unstructured/`):

```bash
python src/data_gen/generate_data.py
```

Set an API key if you want **live mode** (real LLM calls). Without it the agent runs in fully deterministic **mock mode**:

```bash
# Gemini (default)
export GEMINI_API_KEY=your_key_here
# or
export GOOGLE_API_KEY=your_key_here

# Anthropic Claude
export ANTHROPIC_API_KEY=your_key_here

# OpenAI
export OPENAI_API_KEY=your_key_here
# optional: point to an OpenAI-compatible endpoint
export OPENAI_BASE_URL=https://api.openai.com/v1
```

> **Mock vs live**: mock mode uses small rule-based planners for NL→SQL and synthesis. It is fast, free, and fully functional for the question patterns in [`notebooks/demo.ipynb`](../notebooks/demo.ipynb). Live mode uses the configured provider and handles open-ended phrasing better. See [`cost-latency-tradeoffs.md`](cost-latency-tradeoffs.md) for details.

### OpenAI setup (recommended quick path)

```bash
cd /Users/nandishshah/Downloads/files/project-veer-qna

# Install base package + UI (run once)
uv pip install -e .
uv pip install -e ".[ui]"

# Generate synthetic data (run once)
uv run python src/data_gen/generate_data.py

# Select OpenAI provider
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o
export OPENAI_API_KEY=sk-...

# Verify active provider/model
uv run --no-env-file python -c "from src.qna_agent import QNAAgent; a=QNAAgent(); print(a.orchestrator.llm.provider, a.orchestrator.llm.model, a.mock_mode)"
```

---

## 1. Run as a NAT workflow

The project registers `fmcgqabot_qna_workflow` as a NAT plugin via the `nat.plugins` entry point in [`pyproject.toml`](../pyproject.toml). The registration logic lives in [`src/nat_plugin.py`](../src/nat_plugin.py) and the workflow descriptor is [`nat_workflow.yaml`](../nat_workflow.yaml).

### 1.1 Install NAT dependencies

NAT is declared as an optional extra so the base install stays lightweight:

```bash
# with uv
uv pip install -e ".[nat]"

# with pip
pip install -e ".[nat]"
```

This installs `nvidia-nat`, `nvidia-nat-agno`, and `langchain-core`.

> **Note**: In this prototype the NAT workflow is **illustrative**. The `nvidia-nat` packages are not published on PyPI and must be installed from NVIDIA's source distribution (git submodules + Git LFS). See [`design-decisions.md`](design-decisions.md) for the scope call. The plugin code is still valid and will work once NAT is installed.

### 1.2 Run a single question

```bash
uv run nat run --config_file nat_workflow.yaml --input "How did NutriOat Gold do in North during the festive campaign?"
```

Expected output is a plain-text answer. The workflow function (`fmcgqabot_qna_workflow`) wraps `QNAAgent.chat()` and returns `result["answer"]`.

### 1.3 Run with OpenAI (or any supported provider)

The NAT plugin reads `LLM_PROVIDER` and `LLM_MODEL` from the environment, so you can switch providers without editing code:

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o
export OPENAI_API_KEY=sk-...

uv run nat run --config_file nat_workflow.yaml --input "How did NutriOat Gold do in North during the festive campaign?"
```

Supported `LLM_PROVIDER` values: `gemini`, `anthropic`, `openai`.

### 1.4 How the workflow is wired

- [`nat_workflow.yaml`](../nat_workflow.yaml) declares `_type: fmcgqabot_qna_workflow` and a `session_id`.
- [`src/nat_plugin.py`](../src/nat_plugin.py) defines `FMCGQABOTQNAWorkflowConfig` and the registered async function.
- The plugin lazy-imports `QNAAgent`, runs its synchronous `chat()` method in a thread pool, and surfaces errors gracefully.

### 1.5 Multi-turn sessions

The default `session_id` in `nat_workflow.yaml` is `nat_session`. NAT calls the workflow function fresh each time, but `QNAAgent` keeps an in-memory session store, so consecutive calls with the same `session_id` retain context:

```bash
uv run nat run --config_file nat_workflow.yaml --input "How is NutriOat Gold doing in North?"
uv run nat run --config_file nat_workflow.yaml --input "What about October instead?"
```

To start a fresh session, edit `session_id` in `nat_workflow.yaml` or pass a different config file.

### 1.6 NAT workflow troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'nat'` | NAT extras not installed | `uv pip install -e ".[nat]"` |
| `Plugin fmcgqabot not found` | Package not installed in editable mode | `uv pip install -e .` |
| `data/structured/fmcg.db not found` | Synthetic data not generated | `python src/data_gen/generate_data.py` |
| Empty or generic answer | Mock mode does not recognize the phrasing | Rephrase using the patterns in `notebooks/demo.ipynb`, or set a provider API key |
| OpenAI call fails with "model not found" | `LLM_MODEL` not set and default is a Gemini model | `export LLM_MODEL=gpt-4o` |

---

## 2. Run the interactive UI

A lightweight Streamlit UI is provided in [`src/ui.py`](../src/ui.py). It lets you chat with the agent, inspect citations/sources, view suggested follow-ups, and see a per-turn trace summary.

### 2.1 Install UI dependencies

```bash
# with uv
uv pip install -e ".[ui]"

# with pip
pip install -e ".[ui]"
```

Or install Streamlit directly:

```bash
pip install streamlit
```

### 2.2 Launch the UI

```bash
streamlit run src/ui.py
```

The app opens at `http://localhost:8501` by default.

### 2.3 Launch the UI with OpenAI (or any supported provider)

The UI reads `LLM_PROVIDER` and `LLM_MODEL` from the environment:

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o
export OPENAI_API_KEY=sk-...

streamlit run src/ui.py
```

The sidebar will show the active provider (e.g. "Live (OPENAI)").

### 2.4 UI features

- **Session sidebar**: create, switch, or reset chat sessions. Each session keeps its own memory and follow-up context.
- **Mock/live indicator**: a badge in the sidebar shows whether the agent is running in mock or live mode and which provider is active.
- **Citations & sources**: each answer displays structured sources and document references when available.
- **Suggested follow-ups**: clickable follow-up questions generated by the orchestrator.
- **Trace summary**: expand the "Trace" section to see latency, token usage, and estimated cost for the last turn.
- **Example questions**: buttons to populate the input with representative queries from the demo notebook.

### 2.5 UI architecture

```
┌─────────────────────────────────────────────┐
│  Streamlit UI (src/ui.py)                   │
│  - session state management                 │
│  - chat history rendering                   │
│  - trace/source expanders                   │
└─────────────────┬───────────────────────────┘
                  │  calls QNAAgent.chat()
                  ▼
┌─────────────────────────────────────────────┐
│  QNAAgent → OrchestratorAgent → sub-agents  │
│  (structured / unstructured / web / coding) │
└─────────────────────────────────────────────┘
```

The UI is stateless across restarts — sessions live in Streamlit's `st.session_state`. For a production deployment you would persist `SessionMemory` to Redis or a database.

---

## 3. Run the notebook with OpenAI

The notebook uses `QNAAgent()` and therefore picks up `LLM_PROVIDER` / `LLM_MODEL` from your shell environment.

```bash
cd /Users/nandishshah/Downloads/files/project-veer-qna
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o
export OPENAI_API_KEY=sk-...

uv run jupyter notebook notebooks/demo.ipynb
```

If you want to confirm from inside the notebook, run:

```python
from src.qna_agent import QNAAgent
agent = QNAAgent()
print(agent.orchestrator.llm.provider, agent.orchestrator.llm.model, agent.mock_mode)
```

---

## 4. Other ways to run

| Method | Command | Best for |
|---|---|---|
| Python one-liner | `python -c "from src.qna_agent import QNAAgent; print(QNAAgent().chat('...')['answer'])"` | Quick scripting |
| Notebook | `jupyter notebook notebooks/demo.ipynb` | Exploring all capabilities |
| Tests | `pytest tests/ -q` | Guardrails & intent validation |
| NAT CLI | `uv run nat run --config_file nat_workflow.yaml --input "..."` | NAT integration demo |
| Streamlit UI | `streamlit run src/ui.py` | Interactive exploration |

---

## 5. Environment variables

| Variable | Purpose | Required? |
|---|---|---|
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Enables live mode with Gemini | No — mock mode works without it |
| `ANTHROPIC_API_KEY` | Enables live mode with Claude | No — mock mode works without it |
| `OPENAI_API_KEY` | Enables live mode with OpenAI | No — mock mode works without it |
| `OPENAI_BASE_URL` | Optional endpoint override for OpenAI-compatible APIs | No |
| `LLM_PROVIDER` | Select provider: `gemini`, `anthropic`, `openai` | No — defaults to Gemini |
| `LLM_MODEL` | Override the model name for the selected provider | No — sensible default is chosen per provider |
| `PYTHONPATH` | Set to repo root if imports fail outside editable install | Usually not needed with `uv run` or `pip install -e .` |

---

## 6. Next steps

- See [`architecture.md`](architecture.md) for the full data-flow diagram.
- See [`design-decisions.md`](design-decisions.md) for why the orchestrator is hand-rolled and how NAT fits in.
- See [`attribute-mapping.md`](attribute-mapping.md) for which demo questions exercise each assessment requirement.
- See [`cost-latency-tradeoffs.md`](cost-latency-tradeoffs.md) for mock vs live performance and cost estimates.
