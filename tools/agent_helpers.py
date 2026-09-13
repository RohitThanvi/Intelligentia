"""Shared helpers for building the agent tree: parallel-vs-sequential
grouping (STRATEGIST_LOW_QUOTA_MODE), non-Gemini reasoning-model routing,
provider-aware request-rate throttling, and Ollama robustness (pre-flight
health check + automatic fallback to Gemini on any local failure)."""
import asyncio
import json
import time
import urllib.error
import urllib.request

from google.adk.agents import ParallelAgent, SequentialAgent

from config import settings


def parallel_or_sequential(name: str, description: str, sub_agents: list):
    """Build a ParallelAgent normally, or a SequentialAgent if LOW_QUOTA_MODE
    is enabled -- same sub_agents, same eventual state outputs, just traded
    concurrency for staying under a tight requests-per-minute quota."""
    cls = SequentialAgent if settings.LOW_QUOTA_MODE else ParallelAgent
    return cls(name=name, description=description, sub_agents=sub_agents)


# ---------------------------------------------------------------------------
# Reasoning-model provider routing (gemini / nvidia_nim / ollama)
# ---------------------------------------------------------------------------

def check_ollama_available():
    """Pre-flight check: is the Ollama server reachable, and is the
    configured model actually pulled? Raises a clear, actionable error
    immediately at startup rather than failing deep into a multi-minute
    pipeline run with a cryptic connection error.

    Called once, eagerly, from agent.py when REASONING_PROVIDER=ollama.
    """
    base = settings.OLLAMA_API_BASE.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"STRATEGIST_REASONING_PROVIDER=ollama but couldn't reach Ollama at "
            f"{base} ({e}). Is Ollama installed and running? Start it, or set "
            f"OLLAMA_API_BASE to the right address."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"STRATEGIST_REASONING_PROVIDER=ollama, but the response from {base}/api/tags "
            f"wasn't valid JSON ({e}). Confirm this is actually an Ollama server."
        ) from e

    available = {m.get("name", "") for m in data.get("models", [])}
    wanted = settings.OLLAMA_MODEL_ID
    # Ollama tags often include a version suffix (e.g. "llama3.1:8b" vs
    # "llama3.1:8b-instruct-q4_0") -- match on prefix, not just exact string.
    if not any(name == wanted or name.startswith(wanted + "-") for name in available):
        raise RuntimeError(
            f"STRATEGIST_REASONING_PROVIDER=ollama, but model '{wanted}' isn't pulled "
            f"yet. Available locally: {sorted(available) or '(none)'}. Run: "
            f"ollama pull {wanted}"
        )


def get_reasoning_model(default_gemini_model: str = None):
    """Returns the model to use for pure-reasoning agents that don't need
    google_search or VertexAiRagRetrieval (both Gemini/Vertex-only tools).
    Controlled by STRATEGIST_REASONING_PROVIDER in .env:

      "gemini" (default): returns default_gemini_model unchanged (falls back
        to STRATEGIST_DEFAULT_MODEL if the caller doesn't specify one) -- zero
        behavior change if you don't opt in.

      "nvidia_nim": LiteLlm-wrapped model on NVIDIA NIM (e.g. Nemotron 3
        Ultra), via NVIDIA_NIM_API_KEY. Free tier is hard-capped at 40 RPM
        with NO official increase path -- budget STRATEGIST_NVIDIA_MAX_RPM
        accordingly.

      "ollama": LiteLlm-wrapped model on a LOCAL Ollama server. No external
        rate limit at all. Passes a larger context window (num_ctx) and a
        generous timeout since local inference on modest hardware is much
        slower than a cloud API and this pipeline's agents pass fairly large
        state blobs -- both configurable, see config/settings.py.

    Args:
        default_gemini_model: the exact Gemini model string this call site
            used before opting into this helper -- pass this explicitly so
            switching providers never silently changes which Gemini model a
            given agent group uses when REASONING_PROVIDER stays "gemini",
            AND so the automatic Ollama->Gemini fallback (see
            apply_ollama_fallback below) knows which Gemini model to recover
            into if the local model fails.

    Only wire this into agents that have NO google_search or
    VertexAiRagRetrieval tool -- those cannot work through LiteLLM/any
    non-Gemini provider (see README).
    """
    provider = settings.REASONING_PROVIDER
    if provider in ("nvidia_nim", "ollama"):
        try:
            from google.adk.models.lite_llm import LiteLlm
        except ImportError as e:
            raise RuntimeError(
                f"STRATEGIST_REASONING_PROVIDER={provider} requires the litellm "
                "dependency. Install it with: pip install \"google-adk[extensions]\""
            ) from e
        if provider == "nvidia_nim":
            return LiteLlm(model=f"nvidia_nim/{settings.REASONING_MODEL_ID}")
        model = LiteLlm(
            model=f"ollama_chat/{settings.OLLAMA_MODEL_ID}",
            api_base=settings.OLLAMA_API_BASE,
            num_ctx=settings.OLLAMA_NUM_CTX,
            timeout=settings.OLLAMA_TIMEOUT_SECONDS,
        )
        # Stash the intended Gemini fallback model on the object itself so
        # apply_ollama_fallback can recover to the SAME model this call site
        # would have used with REASONING_PROVIDER=gemini, not a generic guess.
        model._strategist_fallback_gemini_model = default_gemini_model or settings.DEFAULT_MODEL
        return model
    return default_gemini_model or settings.DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Provider-aware request-rate throttling
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Async rate limiter enforcing minimum spacing between calls. A single
    asyncio.Lock serializes callers so concurrent branches still get
    correctly spaced release times, not just a per-branch delay."""

    def __init__(self, max_requests_per_minute: int):
        self._min_interval = 60.0 / max_requests_per_minute if max_requests_per_minute > 0 else 0.0
        self._lock = asyncio.Lock()
        self._last_call_at = 0.0

    async def wait_turn(self):
        if self._min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            remaining = self._min_interval - (now - self._last_call_at)
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last_call_at = time.monotonic()


# Separate budgets per provider -- these are DIFFERENT services with
# different limits (your Vertex/Gemini quota has nothing to do with NVIDIA
# NIM's 40 RPM cap, and a local Ollama server has no external limit at all).
_gemini_limiter = _RateLimiter(settings.MAX_REQUESTS_PER_MINUTE)
_nvidia_limiter = _RateLimiter(settings.NVIDIA_MAX_RPM)
# Ollama is local -- no rate limiter needed/applied.


def _model_string_of(agent) -> str | None:
    model = getattr(agent, "model", None)
    return getattr(model, "model", None) if not isinstance(model, str) else model


def _limiter_for_agent(agent) -> "_RateLimiter | None":
    model_name = _model_string_of(agent)
    if isinstance(model_name, str):
        if model_name.startswith("nvidia_nim/"):
            return _nvidia_limiter if settings.NVIDIA_MAX_RPM > 0 else None
        if model_name.startswith("ollama_chat/") or model_name.startswith("ollama/"):
            return None  # local, no throttling
    return _gemini_limiter if settings.MAX_REQUESTS_PER_MINUTE > 0 else None


def apply_rate_limit(agent_tree):
    """Walk an already-built agent tree and attach the correct provider-aware
    rate-limiting before_model_callback to every leaf LlmAgent. No-op for any
    agent whose provider has no configured/needed limit."""
    if settings.MAX_REQUESTS_PER_MINUTE <= 0 and settings.NVIDIA_MAX_RPM <= 0:
        return agent_tree

    def walk(a):
        sub_agents = getattr(a, "sub_agents", None)
        if sub_agents:
            for s in sub_agents:
                walk(s)
            return
        limiter = _limiter_for_agent(a)
        if limiter is None or not hasattr(a, "before_model_callback"):
            return

        async def _cb(callback_context=None, llm_request=None, _limiter=limiter, **_):
            await _limiter.wait_turn()
            return None

        _chain_before_model_callback(a, _cb)

    walk(agent_tree)
    return agent_tree


# ---------------------------------------------------------------------------
# Ollama robustness: automatic fallback to Gemini on ANY local failure
# ---------------------------------------------------------------------------

def _chain_before_model_callback(agent, new_callback):
    existing = agent.before_model_callback
    if existing is None:
        agent.before_model_callback = new_callback
    elif isinstance(existing, list):
        agent.before_model_callback = [new_callback] + list(existing)
    else:
        agent.before_model_callback = [new_callback, existing]


def _chain_on_model_error_callback(agent, new_callback):
    existing = agent.on_model_error_callback
    if existing is None:
        agent.on_model_error_callback = new_callback
    elif isinstance(existing, list):
        agent.on_model_error_callback = list(existing) + [new_callback]
    else:
        agent.on_model_error_callback = [existing, new_callback]


async def _ollama_fallback_on_error(callback_context=None, llm_request=None, error=None, **_):
    """on_model_error_callback: if the local Ollama call failed for ANY reason
    (server down, model not loaded, tool-call the model couldn't format,
    timeout on slow hardware, context overflow, etc.), retry the exact same
    request against the Gemini model this agent would have used by default.
    Returns the recovered LlmResponse, or None (re-raises the original error)
    if the fallback itself also fails -- never silently hides a total
    failure, but never lets a single flaky local inference kill the whole
    pipeline run either.
    """
    fallback_model_name = getattr(llm_request, "_strategist_fallback_gemini_model", None) \
        or settings.DEFAULT_MODEL
    print(
        f"[ollama-fallback] Local Ollama call failed ({error!r}); "
        f"retrying with Gemini ({fallback_model_name}) instead."
    )
    try:
        from google.adk.models.google_llm import Gemini
        fallback = Gemini(model=fallback_model_name)
        # Gemini.generate_content_async reads the model name to call from
        # llm_request.model, NOT from the Gemini instance's own .model field
        # -- ADK's base_llm_flow stamps llm_request.model with the ORIGINAL
        # agent's model string ("ollama_chat/llama3.1:8b") before this
        # callback ever runs, and that field is otherwise untouched here.
        # Without this line, the fallback silently sends the Ollama model
        # string to Vertex/Gemini, which (mis)parses the slash as a
        # publishers/{x}/models/{y} path and fails with an
        # "Invalid Endpoint name" error -- masking the ORIGINAL Ollama
        # failure and defeating the whole point of this fallback.
        llm_request.model = fallback_model_name
        last_response = None
        async for resp in fallback.generate_content_async(llm_request, stream=False):
            last_response = resp
        if last_response is not None:
            print("[ollama-fallback] Gemini fallback succeeded.")
        return last_response
    except Exception as fallback_error:
        print(f"[ollama-fallback] Gemini fallback ALSO failed: {fallback_error!r}. "
              "Raising the original Ollama error.")
        return None  # None => ADK re-raises the ORIGINAL error, not this one


def apply_ollama_fallback(agent_tree):
    """Walk an already-built agent tree and attach the Ollama->Gemini
    fallback to every leaf agent actually routed through Ollama. No-op
    (zero overhead) if REASONING_PROVIDER isn't "ollama"."""
    if settings.REASONING_PROVIDER != "ollama":
        return agent_tree

    def walk(a):
        sub_agents = getattr(a, "sub_agents", None)
        if sub_agents:
            for s in sub_agents:
                walk(s)
            return
        model_name = _model_string_of(a)
        if not (isinstance(model_name, str) and model_name.startswith("ollama_chat/")):
            return
        if not hasattr(a, "on_model_error_callback"):
            return

        # Stash the fallback model name where the request-level callback can
        # see it (llm_request is what _ollama_fallback_on_error receives).
        fallback_model_name = getattr(a.model, "_strategist_fallback_gemini_model", settings.DEFAULT_MODEL)

        async def _before_cb(callback_context=None, llm_request=None, _fb=fallback_model_name, **_):
            llm_request._strategist_fallback_gemini_model = _fb
            return None

        _chain_before_model_callback(a, _before_cb)
        _chain_on_model_error_callback(a, _ollama_fallback_on_error)

    walk(agent_tree)
    return agent_tree
