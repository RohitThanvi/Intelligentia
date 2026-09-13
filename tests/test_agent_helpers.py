"""
Unit tests for tools/agent_helpers.py -- specifically the Ollama-fallback
machinery, since that's hand-rolled glue code that only ever gets exercised
during a real (slow, flaky, hard-to-repro) local-model failure. These tests
mock the LLM calls entirely, so they run in milliseconds with no network, no
GCP credentials, and no Ollama server required.

Run with:
    pytest tests/test_agent_helpers.py -v
"""
import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GOOGLE_API_KEY", "test-placeholder")

from tools import agent_helpers  # noqa: E402
from config import settings  # noqa: E402


# ---------------------------------------------------------------------------
# _ollama_fallback_on_error
# ---------------------------------------------------------------------------

def _fake_generate_ok(text="ok"):
    async def _gen(self, request, stream=False):
        yield SimpleNamespace(text=text)
    return _gen


def _fake_generate_raises(exc):
    async def _gen(self, request, stream=False):
        raise exc
        yield  # pragma: no cover -- unreachable, keeps this an async generator
    return _gen


def test_fallback_overwrites_llm_request_model():
    """Regression test for the bug where the Gemini fallback call read
    llm_request.model (still the ORIGINAL "ollama_chat/..." string, stamped
    there by ADK before this callback runs) instead of the intended fallback
    model -- Gemini.generate_content_async ignores its own constructor's
    `model` kwarg and always uses llm_request.model. Without overwriting it,
    the "fallback" silently asks Vertex/Gemini for a model literally named
    "ollama_chat/llama3.1:8b", which fails with a confusing
    'Invalid Endpoint name' error instead of recovering.
    """
    llm_request = SimpleNamespace(model="ollama_chat/llama3.1:8b")
    llm_request._strategist_fallback_gemini_model = "gemini-2.5-flash"

    with patch("google.adk.models.google_llm.Gemini.generate_content_async",
               new=_fake_generate_ok()):
        result = asyncio.run(agent_helpers._ollama_fallback_on_error(
            llm_request=llm_request, error=TimeoutError("local model timed out"),
        ))

    assert llm_request.model == "gemini-2.5-flash"
    assert result is not None and result.text == "ok"


def test_fallback_defaults_to_settings_default_model_when_unstashed():
    """If apply_ollama_fallback never stashed a per-call-site fallback model
    (e.g. it's being invoked directly/unexpectedly), fall back to
    settings.DEFAULT_MODEL rather than crashing or using the Ollama name."""
    llm_request = SimpleNamespace(model="ollama_chat/llama3.1:8b")

    with patch("google.adk.models.google_llm.Gemini.generate_content_async",
               new=_fake_generate_ok()):
        asyncio.run(agent_helpers._ollama_fallback_on_error(
            llm_request=llm_request, error=RuntimeError("boom"),
        ))

    assert llm_request.model == settings.DEFAULT_MODEL


def test_fallback_returns_none_when_gemini_also_fails():
    """None => ADK re-raises the ORIGINAL Ollama error, not the fallback's.
    This is the "never silently hide a total failure" contract documented
    on the function -- if Gemini is also down, the caller must still see a
    failure, not a silently swallowed run."""
    llm_request = SimpleNamespace(model="ollama_chat/llama3.1:8b")
    llm_request._strategist_fallback_gemini_model = "gemini-2.5-flash"

    with patch("google.adk.models.google_llm.Gemini.generate_content_async",
               new=_fake_generate_raises(RuntimeError("gemini also down"))):
        result = asyncio.run(agent_helpers._ollama_fallback_on_error(
            llm_request=llm_request, error=TimeoutError("local model timed out"),
        ))

    assert result is None


# ---------------------------------------------------------------------------
# apply_ollama_fallback wiring
# ---------------------------------------------------------------------------

def _leaf(name, model):
    """Minimal stand-in for a leaf LlmAgent -- only the attributes
    apply_ollama_fallback / apply_rate_limit actually touch."""
    return SimpleNamespace(name=name, model=model, sub_agents=None,
                            before_model_callback=None, on_model_error_callback=None)


def test_apply_ollama_fallback_is_noop_when_provider_not_ollama(monkeypatch):
    monkeypatch.setattr(settings, "REASONING_PROVIDER", "gemini")
    agent = _leaf("a", "gemini-2.5-flash")

    agent_helpers.apply_ollama_fallback(agent)

    assert agent.before_model_callback is None
    assert agent.on_model_error_callback is None


def test_apply_ollama_fallback_only_wires_ollama_routed_leaves(monkeypatch):
    monkeypatch.setattr(settings, "REASONING_PROVIDER", "ollama")
    ollama_model = SimpleNamespace(model="ollama_chat/llama3.1:8b",
                                    _strategist_fallback_gemini_model="gemini-2.5-flash")
    gemini_agent = _leaf("gemini_agent", "gemini-2.5-flash")
    ollama_agent = _leaf("ollama_agent", ollama_model)
    tree = SimpleNamespace(sub_agents=[gemini_agent, ollama_agent])

    agent_helpers.apply_ollama_fallback(tree)

    assert gemini_agent.before_model_callback is None
    assert gemini_agent.on_model_error_callback is None
    assert ollama_agent.before_model_callback is not None
    assert ollama_agent.on_model_error_callback is not None


def test_apply_ollama_fallback_before_callback_stashes_correct_model(monkeypatch):
    """The before_model_callback's job is to stamp llm_request with the
    fallback model name so the (later) error callback can see it. Verify it
    stamps the SAME model apply_ollama_fallback read off the agent, not a
    stale/shared/default value."""
    monkeypatch.setattr(settings, "REASONING_PROVIDER", "ollama")
    ollama_model = SimpleNamespace(model="ollama_chat/llama3.1:8b",
                                    _strategist_fallback_gemini_model="gemini-2.5-pro")
    ollama_agent = _leaf("ollama_agent", ollama_model)

    agent_helpers.apply_ollama_fallback(ollama_agent)

    llm_request = SimpleNamespace()
    asyncio.run(ollama_agent.before_model_callback(llm_request=llm_request))

    assert llm_request._strategist_fallback_gemini_model == "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# STRATEGIST_MAX_ITERATIONS actually taking effect
# ---------------------------------------------------------------------------

def test_max_iterations_env_var_takes_effect(monkeypatch):
    """Regression test: this env var used to be defined in settings.py but
    never consumed anywhere -- agents/critics/agent.py and
    agents/synthesis/agent.py read a separate hardcoded constant
    (schemas.state.DEFAULT_MAX_ITERATIONS) instead, so setting this had no
    effect at all.

    Executes config/settings.py fresh via runpy rather than
    importlib.reload()-ing the already-imported `config.settings` module --
    reloading the real module would mutate shared global state that OTHER
    tests (and agents/critics/agent.py, agents/synthesis/agent.py, already
    imported elsewhere in this session) hold a live reference to, leaking
    across tests depending on run order.
    """
    import runpy
    monkeypatch.setenv("STRATEGIST_MAX_ITERATIONS", "7")
    settings_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "settings.py"
    )
    namespace = runpy.run_path(settings_path)
    assert namespace["MAX_ITERATIONS"] == 7


def test_critics_loop_reads_max_iterations_from_settings():
    from agents.critics.agent import red_team_loop
    from config import settings as settings_module
    assert red_team_loop.max_iterations == settings_module.MAX_ITERATIONS


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
