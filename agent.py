"""
Root Orchestrator (Section 3) for the Enterprise GenAI Strategist.

`root_agent` is exported here as required for deployment to Vertex AI Agent
Engine / `adk run` / `adk web` (Section 28-29).
"""
import os
import sys

# `adk web`/`adk run` load this project as a package named after its own
# folder (whatever you rename it to) and only add that folder's PARENT
# directory to sys.path -- not this folder itself. Since every internal
# module here uses absolute imports (`agents.x`, `tools.x`, `schemas.x`,
# `config.x`) rather than relative ones, we need this folder itself on
# sys.path too, or those imports fail with
# "ModuleNotFoundError: No module named 'agents'" regardless of what the
# project folder is named. This must run before any of the imports below.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from google.adk.agents import SequentialAgent

from agents.intake.agent import intake_director
from agents.research.agent import research_director
from agents.research.rag_agent import knowledge_retrieval_pipeline
from agents.use_cases.agent import use_case_discovery_agent
from agents.use_cases.scoring_agent import use_case_evaluation_pipeline
from agents.alternatives.agent import alternatives_director
from agents.architecture.agent import architecture_director
from agents.finance.agent import financial_director
from agents.risk.agent import risk_director
from agents.critics.agent import red_team_loop
from agents.synthesis.agent import synthesis_pipeline
from tools.rag_tools import build_index
from tools.agent_helpers import apply_rate_limit, apply_ollama_fallback, check_ollama_available
from config import settings

# Build the RAG index once at import time so knowledge_retrieval_agent has
# something to search from process start (Section 7).
build_index()

# Fail fast with a clear message if Ollama isn't actually reachable/pulled,
# rather than discovering this deep into a multi-minute pipeline run.
if settings.REASONING_PROVIDER == "ollama":
    check_ollama_available()

root_agent = SequentialAgent(
    name="enterprise_genai_strategist",
    description=(
        "Root orchestrator for enterprise GenAI strategy and use-case evaluation. "
        "Acts as the managing partner of an AI consulting engagement: delegates to "
        "intake, research, use-case discovery, alternatives, architecture, finance, "
        "risk, and red-team/critic directors, then synthesizes an executive report."
    ),
    sub_agents=[
        intake_director,
        research_director,
        knowledge_retrieval_pipeline,
        use_case_discovery_agent,
        alternatives_director,
        use_case_evaluation_pipeline,
        architecture_director,
        financial_director,
        risk_director,
        red_team_loop,
        synthesis_pipeline,
    ],
)

# Enforces real request-per-minute spacing across EVERY agent in the tree
# (not just within parallel groups) when STRATEGIST_MAX_RPM/NVIDIA_MAX_RPM is
# set -- see tools/agent_helpers.py. No-op, zero overhead, if both unset/0.
apply_rate_limit(root_agent)

# If REASONING_PROVIDER=ollama: automatically fall back to Gemini for any
# individual call that fails against the local model (server down, model
# not loaded, tool-call the model couldn't format, timeout, context
# overflow, etc.) -- keeps a single flaky local inference from killing the
# whole pipeline run. No-op if REASONING_PROVIDER isn't "ollama".
apply_ollama_fallback(root_agent)
