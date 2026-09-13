"""
Preflight self-check: builds the agent tree and asserts a handful of
structural invariants that, if silently wrong, only ever surface deep into a
multi-minute pipeline run (or not at all -- they can produce a wrong-but-
plausible-looking result, like a JSON block with doubled braces, or a
red-team loop capped at the wrong iteration count).

This is NOT an eval of output quality -- it never calls an LLM. It's a
structural/wiring check, meant to run in a few seconds, in CI or before
`adk deploy`, without needing GCP credentials, an Ollama server, or a
network connection.

Usage:
    python3 scripts/preflight_check.py

Exit code 0 = all checks passed. Non-zero = at least one check failed (see
printed output for which).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force the safest defaults for a no-credentials structural check: real
# Gemini provider (no Ollama server / NVIDIA key required to even import the
# agent tree), and local RAG backend (no GCP project / RAG corpus required).
os.environ.setdefault("GOOGLE_API_KEY", "preflight-check-placeholder")
os.environ.setdefault("STRATEGIST_RAG_BACKEND", "local")

FAILURES: list[str] = []


def check(description: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}")
    if not condition:
        FAILURES.append(f"{description}" + (f" -- {detail}" if detail else ""))


def _model_string_of(agent) -> str | None:
    model = getattr(agent, "model", None)
    return getattr(model, "model", None) if not isinstance(model, str) else model


def _walk_leaves(agent):
    sub_agents = getattr(agent, "sub_agents", None)
    if sub_agents:
        for s in sub_agents:
            yield from _walk_leaves(s)
    else:
        yield agent


def main() -> int:
    from config import settings

    # --- 1. The agent tree actually builds without raising. ---------------
    try:
        from agent import root_agent
        build_error = None
    except Exception as e:  # noqa: BLE001 -- we want to report ANY build failure
        build_error = e
    check("root_agent imports and builds", build_error is None,
          detail=repr(build_error) if build_error else "")
    if build_error is not None:
        # Nothing else below is meaningful if the tree didn't even build.
        _report_and_exit()
        return 1

    leaves = list(_walk_leaves(root_agent))
    check("agent tree has leaf LlmAgents", len(leaves) > 0)

    # --- 2. Every Ollama-routed leaf has a REAL (non-Ollama, non-NVIDIA) --
    #        Gemini fallback model stashed, and the fallback callback wired.
    ollama_leaves = [
        a for a in leaves
        if isinstance(_model_string_of(a), str) and _model_string_of(a).startswith("ollama_chat/")
    ]
    if settings.REASONING_PROVIDER == "ollama":
        check("at least one agent is actually routed through Ollama when "
              "STRATEGIST_REASONING_PROVIDER=ollama", len(ollama_leaves) > 0)
        for a in ollama_leaves:
            fb = getattr(a.model, "_strategist_fallback_gemini_model", None)
            valid_fb = isinstance(fb, str) and not fb.startswith(("ollama_chat/", "ollama/", "nvidia_nim/"))
            check(f"'{a.name}' has a real Gemini fallback model stashed", valid_fb,
                  detail=f"got {fb!r}")
            check(f"'{a.name}' has apply_ollama_fallback's on_model_error_callback wired",
                  getattr(a, "on_model_error_callback", None) is not None)
            check(f"'{a.name}' has apply_ollama_fallback's before_model_callback wired",
                  getattr(a, "before_model_callback", None) is not None)
    else:
        check("no agent is left dangling on ollama_chat/ when "
              "STRATEGIST_REASONING_PROVIDER != 'ollama'", len(ollama_leaves) == 0,
              detail=f"found: {[a.name for a in ollama_leaves]}")

    # --- 3. Single source of truth for the red-team loop bound. -----------
    # Regression guard for the bug where agents/critics/agent.py and
    # agents/synthesis/agent.py read a separate hardcoded constant
    # (schemas.state.DEFAULT_MAX_ITERATIONS) instead of settings.MAX_ITERATIONS,
    # silently ignoring STRATEGIST_MAX_ITERATIONS.
    from agents.critics.agent import red_team_loop
    check("red_team_loop.max_iterations matches settings.MAX_ITERATIONS "
          "(i.e. STRATEGIST_MAX_ITERATIONS actually takes effect)",
          red_team_loop.max_iterations == settings.MAX_ITERATIONS,
          detail=f"loop={red_team_loop.max_iterations} settings={settings.MAX_ITERATIONS}")

    # --- 4. The final report's JSON schema example is valid JSON braces. --
    # Regression guard for the quadruple-brace f-string bug that made the
    # LLM's instructed JSON example literally contain "{{" / "}}".
    import json
    from agents.synthesis.agent import final_synthesis_agent
    instruction = final_synthesis_agent.instruction
    fenced_start = instruction.find('```json')
    snippet = instruction[fenced_start:] if fenced_start != -1 else ""
    has_doubled_braces = "{{" in snippet or "}}" in snippet
    check("final_synthesis_agent's JSON schema example has no stray doubled "
          "braces (would mean the LLM is shown invalid JSON as its target shape)",
          fenced_start != -1 and not has_doubled_braces)

    return _report_and_exit()


def _report_and_exit() -> int:
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All preflight checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
