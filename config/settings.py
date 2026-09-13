"""
Central config. Do NOT hard-code secrets here (Section 27) -- everything
sensitive comes from environment variables, typically loaded from a local
.env file (kept OUT of any deployment package; see .env.example).
"""
import os

from schemas.state import DEFAULT_MAX_ITERATIONS

# --- Vertex AI / Gemini auth ---
# Either set GOOGLE_API_KEY (AI Studio key) for local dev,
# or set these three for Vertex AI project-based auth (recommended for prod):
GOOGLE_GENAI_USE_VERTEXAI = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# --- App-level settings ---
# NOTE: this is the ACTUAL knob that controls the red-team/validate/revise
# loop's bound (LoopAgent(max_iterations=...) in agents/critics/agent.py) and
# what final_synthesis_agent is told the cap is. Both of those used to read
# schemas.state.DEFAULT_MAX_ITERATIONS directly instead of this setting, which
# silently ignored STRATEGIST_MAX_ITERATIONS entirely -- fixed to read from
# here so the env var actually takes effect.
MAX_ITERATIONS = int(os.environ.get("STRATEGIST_MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS)))
DEFAULT_MODEL = os.environ.get("STRATEGIST_DEFAULT_MODEL", "gemini-2.5-flash")
SYNTHESIS_MODEL = os.environ.get("STRATEGIST_SYNTHESIS_MODEL", "gemini-2.5-pro")

# New/low-tier GCP projects often start with very low Gemini requests-per-minute
# quota. This pipeline fires several ParallelAgent groups (5 research domains,
# 6 alternatives evaluators, 6 architecture specialists, 5 risk specialists),
# each a burst of several simultaneous Gemini calls -- easy to hit
# 429 RESOURCE_EXHAUSTED on a fresh project. Setting this to true converts
# every ParallelAgent group in the tree to a SequentialAgent instead (same
# agents, same outputs, just one call at a time) -- slower, but avoids
# concurrent-request bursts entirely. The real fix is requesting a quota
# increase in the GCP console (IAM & Admin > Quotas); this is a workaround
# for testing while that's pending.
LOW_QUOTA_MODE = os.environ.get("STRATEGIST_LOW_QUOTA_MODE", "FALSE").upper() == "TRUE"

# De-parallelizing (LOW_QUOTA_MODE) only removes BURSTS -- it doesn't cap the
# overall RATE of requests over time. A long sequential run through this
# pipeline's ~47 LLM-calling agents can still exceed a tight per-minute quota
# even one request at a time, just more slowly (this is what hit 429 deep
# inside the red-team loop, which was never parallel to begin with). Set this
# to the actual requests-per-minute your GCP quota allows (check IAM & Admin
# > Quotas) to enforce real spacing between every single Gemini call in the
# tree, regardless of which agent or group it belongs to. 0/unset disables
# throttling entirely.
MAX_REQUESTS_PER_MINUTE = int(os.environ.get("STRATEGIST_MAX_RPM", "0"))

# --- Optional non-Gemini reasoning model (NVIDIA NIM / local Ollama) ---
# "gemini" (default): no change, uses each call site's own Gemini model.
# "nvidia_nim": routes red_team_agent/validator_agent/revision_agent/
#   evidence_audit_agent/final_synthesis_agent through LiteLLM -> NVIDIA NIM.
# "ollama": routes the same 5 agents through a LOCAL Ollama server instead --
#   no external rate limit at all, but only realistic for models that
#   actually fit your hardware (the real Nemotron 3 Ultra, 550B params, does
#   NOT -- see tools/agent_helpers.py docstring).
# These 5 agents were chosen because none of them use google_search or
# VertexAiRagRetrieval -- both are Gemini/Vertex-native built-ins that do NOT
# work through LiteLLM/any other provider, so agents that need them stay on
# Gemini regardless of this setting.
REASONING_PROVIDER = os.environ.get("STRATEGIST_REASONING_PROVIDER", "gemini")
REASONING_MODEL_ID = os.environ.get("STRATEGIST_REASONING_MODEL_ID", "nvidia/nemotron-3-ultra-550b-a55b")
# NVIDIA NIM free tier is hard-capped at 40 RPM with NO official increase
# path (confirmed on NVIDIA's own developer forums -- moderators redirect to
# paid dedicated deployment). Keep this comfortably under 40 to leave
# headroom for shared-tenant load, which can throttle you even below the
# stated cap. Separate from MAX_REQUESTS_PER_MINUTE above (that one's for
# Gemini/Vertex) since the two services have nothing to do with each other.
NVIDIA_MAX_RPM = int(os.environ.get("STRATEGIST_NVIDIA_MAX_RPM", "20"))

# Local Ollama server (only used if REASONING_PROVIDER=ollama). No rate
# limiting applied -- bounded only by your own hardware. Default model here
# is a small, genuinely local-friendly pick; override if your hardware can
# handle something bigger. It is NOT the real Nemotron 3 Ultra (550B) --
# that model is not realistic to self-host on consumer hardware.
OLLAMA_MODEL_ID = os.environ.get("STRATEGIST_OLLAMA_MODEL_ID", "llama3.1:8b")
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
# Local models are slower and this pipeline's agents pass fairly large state
# blobs (several prior agents' full outputs) -- a small default context
# window will silently truncate input on some models. Raise if your hardware
# and model support it; llama3.1 supports up to 128k but that needs a lot of
# RAM/VRAM, so 16384 is a safer practical default.
OLLAMA_NUM_CTX = int(os.environ.get("STRATEGIST_OLLAMA_NUM_CTX", "16384"))
# Local inference on modest hardware (no/weak GPU) can take minutes per call
# for an 8B model with a long context -- default timeout here is generous.
OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("STRATEGIST_OLLAMA_TIMEOUT_SECONDS", "300"))
# NVIDIA_NIM_API_KEY is read directly by litellm, not by this app -- set it
# in .env anyway so `adk deploy`/`adk web` pick it up the same way as every
# other credential in this project.

# --- RAG backend switch ---
# "vertex" (default here): Google's managed Vertex AI RAG Engine. Requires
#           GOOGLE_GENAI_USE_VERTEXAI=TRUE, a real GCP project with Vertex AI
#           enabled, Application Default Credentials, and an existing RAG
#           corpus resource name in RAG_CORPUS_RESOURCE_NAME.
# "local":  fallback TF-IDF over sample_documents/, works with just GOOGLE_API_KEY,
#           no GCP project needed. Useful if you ever want to test offline.
RAG_BACKEND = os.environ.get("STRATEGIST_RAG_BACKEND", "vertex")
RAG_CORPUS_RESOURCE_NAME = os.environ.get("RAG_CORPUS_RESOURCE_NAME", "")
