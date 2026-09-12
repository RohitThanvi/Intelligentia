# Enterprise GenAI Strategist — Multi-Agent System (Google ADK)

A hierarchical multi-agent system that behaves like an AI consulting firm:
given an organization's business context, it researches, evaluates GenAI vs.
non-GenAI alternatives, scores use cases, designs architecture, models ROI,
assesses risk, red-teams its own recommendation through a bounded loop, and
produces an executive report — all via real ADK orchestration primitives
(`SequentialAgent`, `ParallelAgent`, `LoopAgent`), not one big prompt.

Built and verified against `google-adk==2.8.0`.

## What's implemented now vs. what's next

**Implemented and verified in this pass:**
- Full 58-node agent hierarchy (see `agent.py`) matching the design doc's
  orchestration diagram: intake → research → RAG → use-case
  discovery/scoring → alternatives → architecture → finance → risk →
  red-team loop → synthesis.
- Deterministic tools for financial math (`tools/financial_tools.py`) and
  use-case scoring (`tools/scoring_tools.py`) — verified with real numbers,
  never left to LLM arithmetic.
- A working RAG subsystem (`tools/rag_tools.py`, TF-IDF-based) with two
  sample internal documents in `sample_documents/` — retrieval tested and
  returns correctly attributed chunks.
- Evidence provenance tooling (`tools/evidence_tools.py`) and web research
  via ADK's built-in `google_search` grounding tool.
- Bounded red-team/validator/revision loop with an `exit_loop` escalation
  tool, capped at `MAX_ITERATIONS` (default 3).

**Intentionally left for a follow-up pass** (tell me which to do next):
- Swapping the TF-IDF RAG for a production vector store (Vertex AI Vector
  Search / pgvector) — the tool interface (`rag_search`, `rerank_documents`)
  is already stable, so this is a drop-in swap.
- A custom web-scraping tool (`fetch_webpage`/`extract_web_content`) if you
  need scraping beyond what `google_search` grounding gives you.
- A richer automated eval harness (ADK's `.evalset.json` format) beyond the
  smoke-test runner in `evaluation/evaluation.py`.

## Project structure

```
enterprise_strategist/
├── agent.py                 # root_agent (ADK entry point)
├── agents/                  # one file per director/specialist group
├── tools/                   # deterministic + RAG + evidence tools
├── schemas/state.py         # shared state key constants + record shapes
├── config/settings.py       # env-driven config, no hardcoded secrets
├── sample_documents/        # RAG test corpus
├── evaluation/               # test prompts + local runner
└── requirements.txt
```

## Run it locally

```bash
cd enterprise_strategist
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # includes google-cloud-aiplatform for Vertex RAG

cp .env.example .env
# edit .env: set GOOGLE_CLOUD_PROJECT and RAG_CORPUS_RESOURCE_NAME (see below)
```

**Auth (all via env, no API key):**
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable aiplatform.googleapis.com
```
This creates local Application Default Credentials that both the Gemini
calls and the Vertex AI RAG Engine calls will pick up automatically via
`GOOGLE_GENAI_USE_VERTEXAI=TRUE` + `GOOGLE_CLOUD_PROJECT` in `.env`. Your
account (or the service account, if running non-interactively) needs at
minimum the `roles/aiplatform.user` IAM role.

**Create the RAG corpus once** — see "Switching to Vertex AI RAG Engine"
below — and put its resource name in `.env` as `RAG_CORPUS_RESOURCE_NAME`
before the first run.

**Option 1 — ADK's built-in dev UI (recommended for exploring the trace):**
```bash
adk web
```
This launches a local web UI where you can chat with `root_agent` and watch
every agent/tool call in the execution trace live, matching the trace shape
in Section 25 of the spec.

**Option 2 — CLI:**
```bash
adk run enterprise_strategist
```

**Option 3 — scripted eval run** against the demo scenario in
`evaluation/test_cases.py`:
```bash
python -m evaluation.evaluation
```

⚠️ A full run through all 58 nodes with the demo scenario will make dozens of
Gemini calls (5 parallel research + 6 parallel alternatives + 6 parallel
architecture + 5 parallel risk + up to 3 loop iterations × 3 agents each,
plus sequential steps). Expect it to take a few minutes and consume a
meaningful number of tokens — start with the `small_business_smoke_test`
scenario first.

## Hitting 429 RESOURCE_EXHAUSTED?

New/low-tier GCP projects often start with a very low Gemini
requests-per-minute quota. This pipeline fires several parallel specialist
groups (5 research domains, 6 alternatives evaluators, 6 architecture
specialists, 5 risk specialists), each a burst of simultaneous Gemini calls
-- easy to exceed a fresh project's default quota.

**The real fix:** request a quota increase. GCP Console -> IAM & Admin ->
Quotas -> filter for `generate_content_requests_per_minute` (or similar) for
your model/region, and request an increase. This can take anywhere from
instant to a day or two depending on the amount requested.

**Workaround while that's pending:** set `STRATEGIST_LOW_QUOTA_MODE=TRUE` in
`.env`. This converts every parallel specialist group in the tree into a
sequential one (same agents, same outputs, one call at a time instead of a
burst) -- noticeably slower end-to-end, but avoids concurrent-request bursts
entirely.

**Still hitting 429 even with that on?** `LOW_QUOTA_MODE` only removes
*bursts* -- it doesn't cap the overall *rate* of requests over time. A long
sequential run through this pipeline's ~47 LLM-calling agents can still
exceed a tight per-minute quota even one call at a time, just more slowly.
Set `STRATEGIST_MAX_RPM` in `.env` to your actual GCP quota's
requests-per-minute limit (check IAM & Admin -> Quotas) -- this enforces real
spacing between *every* Gemini call in the tree, correctly serialized even
across parallel branches if you ever turn `LOW_QUOTA_MODE` back off. 0/unset
disables it (default, zero overhead).

## Switching part of the pipeline to a non-Gemini model

You can route the **pure-reasoning agents** — `red_team_agent`,
`validator_agent`, `revision_agent`, `evidence_audit_agent`, and
`final_synthesis_agent` — through a different provider via LiteLLM. These
five were chosen deliberately: none of them use `google_search` or
`VertexAiRagRetrieval`.

**Why the rest of the pipeline can't move:** `google_search` and
`VertexAiRagRetrieval` are Gemini/Vertex-native built-in tools — their
execution happens inside Google's own model-serving stack (for
`google_search`) or is a Vertex-specific API client (`VertexAiRagRetrieval`).
Neither works through LiteLLM or any non-Google model provider. So the 5
research agents and `knowledge_retrieval_agent` stay on Gemini regardless of
this setting — only the tool-free reasoning agents are swappable.

Install the LiteLLM extra either way:
```bash
pip install "google-adk[extensions]"
```

### Option A: NVIDIA NIM (e.g. Nemotron 3 Ultra)

```
STRATEGIST_REASONING_PROVIDER=nvidia_nim
STRATEGIST_REASONING_MODEL_ID=nvidia/nemotron-3-ultra-550b-a55b
NVIDIA_NIM_API_KEY=nvapi-your-key-here
STRATEGIST_NVIDIA_MAX_RPM=20
```

**Read this before you rely on it:** NVIDIA's free tier is **hard-capped at
40 RPM with no official increase path** — confirmed directly on NVIDIA's own
developer forums, where moderators tell people to pay for a dedicated
deployment instead of asking for more free-tier headroom. The actual limit
you experience can be even lower than 40 during periods of high shared-tenant
load, since the free tier is explicitly described as "may vary based on the
number of concurrent users." `STRATEGIST_NVIDIA_MAX_RPM` throttles our own
requests below that ceiling, but it can't fix NVIDIA's side being busy. Also
note: published third-party benchmarks for Nemotron 3 Ultra 550B-A55B show it
competitive on some reasoning metrics but not uniformly ahead of Gemini 2.5
Pro — check current benchmarks for your use case rather than assuming bigger
parameter count wins.

### Option B: Local Ollama (zero external rate limit)

```
STRATEGIST_REASONING_PROVIDER=ollama
STRATEGIST_OLLAMA_MODEL_ID=llama3.1:8b
OLLAMA_API_BASE=http://localhost:11434
```

1. Install [Ollama](https://ollama.com) and make sure it's running.
2. Pull the model first: `ollama pull llama3.1:8b` (or whatever you set
   `STRATEGIST_OLLAMA_MODEL_ID` to).
3. Run as normal.

**Important:** the real Nemotron 3 Ultra is a 550-billion-parameter model —
even at aggressive 4-bit quantization that's roughly 275GB+ just for
weights, which needs enterprise multi-GPU hardware, not a typical dev
machine. It is genuinely not realistic to self-host that specific model via
Ollama. `llama3.1:8b` is the default here because it's a real,
commonly-available local model that fits on consumer hardware (~8GB VRAM at
Q4) and has solid tool-calling support — reasonable for critiquing
already-computed numbers in plain text, which is what these 5 agents
actually do. If you have serious hardware and want to try a larger local
model, check [ollama.com/library](https://ollama.com/library) for what's
genuinely available to pull locally (note: some listings tagged `-cloud` are
Ollama's own hosted cloud service, not local execution, and have their own
separate plan limits).

Local execution has **zero external rate limit** — no 429s from a vendor —
but is bounded by your own hardware's inference speed, so a "run" that took
minutes via cloud APIs may take considerably longer for the same total token
volume, especially without a capable GPU.

**Built-in robustness for this path** (verified with actual test servers,
not just written and assumed to work):

- **Pre-flight health check** — at startup, before any agent runs, we ping
  `OLLAMA_API_BASE` and confirm your configured model is actually pulled.
  If Ollama isn't running, or the model isn't there, you get a clear error
  immediately (e.g. `couldn't reach Ollama at http://localhost:11434 ... Is
  Ollama installed and running?`) instead of a cryptic failure minutes into
  a run.
- **Automatic fallback to Gemini per-call** — if any individual call to the
  local model fails for ANY reason (server crashes mid-run, a malformed
  tool-call the model couldn't format, a timeout, context overflow, etc.),
  that specific call is automatically retried against Gemini instead
  (`gemini-2.5-flash`, or `gemini-2.5-pro` for `final_synthesis_agent` —
  matching what each agent would use by default). You'll see a
  `[ollama-fallback]` line in the console when this happens. If Gemini
  *also* fails, you get the original Ollama error, not a confusing
  fallback-specific one.
- **Larger context window and longer timeout** than LiteLLM's defaults
  (`STRATEGIST_OLLAMA_NUM_CTX=16384`, `STRATEGIST_OLLAMA_TIMEOUT_SECONDS=300`)
  since this pipeline's agents pass fairly large state blobs and local
  inference on modest hardware is slower than a cloud API. Raise
  `STRATEGIST_OLLAMA_NUM_CTX` further if your hardware supports it and you
  still see truncated-looking output.

## Switching to Vertex AI RAG Engine (Google's managed RAG — default here)

`STRATEGIST_RAG_BACKEND=vertex` is the default. It requires project-based
Vertex auth (set up above) and a RAG corpus created ahead of time:

1. Create a corpus and upload your documents (one-time, via Python):
   ```python
   import vertexai
   from vertexai.preview import rag

   vertexai.init(project="YOUR_PROJECT_ID", location="us-central1")
   corpus = rag.create_corpus(display_name="enterprise-strategist-kb")
   rag.import_files(
       corpus.name,
       ["gs://YOUR_BUCKET/loan_processing_sop.md", "gs://YOUR_BUCKET/ai_governance_policy.md"],
   )
   print(corpus.name)  # e.g. projects/123/locations/us-central1/ragCorpora/456
   ```
3. In `.env`, set:
   ```
   GOOGLE_GENAI_USE_VERTEXAI=TRUE
   GOOGLE_CLOUD_PROJECT=your-gcp-project-id
   GOOGLE_CLOUD_LOCATION=us-central1
   STRATEGIST_RAG_BACKEND=vertex
   RAG_CORPUS_RESOURCE_NAME=projects/123/locations/us-central1/ragCorpora/456
   ```
4. Run as normal (`adk web` / `adk run`) — `knowledge_retrieval_agent` now
   queries the real corpus instead of the local TF-IDF index. No other code
   changes needed; the swap is fully config-driven.

Known ADK constraint (verified against the installed version): the
`VertexAiRagRetrieval` tool must be the *only* tool on its agent — it can't
be combined with other tools on the same `LlmAgent`, which is why the Vertex
path uses a dedicated tool list rather than reusing the local one.

⚠️ **Deprecation note** (surfaced when actually running this): the installed
`google-cloud-aiplatform` emits `vertexai.preview.rag module is deprecated,
migrate to the agentplatform client`. It still works today, but Google is
mid-migration to a new `agentplatform` client for RAG corpus management —
worth checking `google-cloud-aiplatform`'s changelog before you build much
on top of this, since the corpus-creation code in this README may need
updating to the new client sooner than the rest of the project.

Also note: installing `google-cloud-aiplatform` may downgrade/conflict with
`opentelemetry-api`'s pinned version from `google-adk` (pip will warn but it
still imports fine) — if you hit runtime tracing issues, pin
`opentelemetry-api<=1.42.1` explicitly after installing both packages.

## Deploying to Google Cloud (Vertex AI Agent Engine)

Verified against the actually-installed ADK CLI (`adk deploy agent_engine --help`)
rather than assumed — this surface changed recently: `--staging_bucket` is
now **deprecated/unused**, and there's no `--set_env_vars` flag. Instead,
`adk deploy` reads your project's `.env` file directly and passes those
key/values through as the deployed agent's runtime `env_vars` config (it does
not ship the raw file — confirmed by reading the CLI source).

1. Auth (same as local):
   ```bash
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```
2. Make sure `enterprise_strategist/.env` has your real
   `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and
   `RAG_CORPUS_RESOURCE_NAME` set (same file you use locally).
3. Deploy:
   ```bash
   # macOS/Linux/WSL/Git Bash:
   ./deploy.sh
   ```
   ```powershell
   # Windows PowerShell:
   .\deploy.ps1
   # If scripts are disabled, run once first:
   #   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```
   or directly:
   ```bash
   adk deploy agent_engine \
     --project=YOUR_PROJECT_ID \
     --region=YOUR_REGION \
     --display_name="enterprise-genai-strategist" \
     enterprise_strategist
   ```
4. `root_agent` exported from `enterprise_strategist/agent.py` is what gets
   deployed. Re-check `adk deploy agent_engine --help` against your own
   installed version before relying on any of this — deploy CLI flags are
   one of the more version-sensitive surfaces (Section 28), and this one
   changed between when this README was first drafted and now.
5. If anything beyond project/region/corpus-name needs to stay private,
   prefer Secret Manager over putting it in `.env` — the whole file's
   parsed contents become the deployed agent's env config either way.

## Notes on fidelity to the original design

- `human_process_evaluator` was added to `alternatives_director`'s parallel
  group even though the original diagram's text lists it under Section 10
  as a sixth agent alongside the five originally diagrammed — kept per the
  written spec.
- The financial/scoring "LLM interprets, tool computes" separation
  (Principle 1) is enforced structurally: the LLM agents are instructed to
  always call the tool, and the tools themselves ignore any LLM-supplied
  totals, recomputing everything from the raw inputs.
- Loop termination is via ADK's `tool_context.actions.escalate` pattern
  (the current supported mechanism for `LoopAgent` early-exit), bounded by
  `max_iterations=3` regardless of escalation.
#   I n t e l l i g e n t i a  
 