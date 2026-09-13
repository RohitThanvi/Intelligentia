"""
Post-run report generator: "what happened, and what came out of it."

Deliberately NOT an LLM step. An audit trail of what the pipeline actually
did (which stages ran, what the red-team loop found/changed each pass,
whether it hit its cap) is exactly the kind of output that must not be
paraphrased, summarized, or "helpfully" trimmed by a model that might drop a
detail. This reads state that's ALREADY been written by the real pipeline
and formats it -- no model call, no extra cost, no hallucination risk, and
it can never disagree with what actually happened.

Wired in as an `after_agent_callback` on `root_agent` (agent.py) and on
`_critic_loop_body` (agents/critics/agent.py) -- see those files for how the
callback plumbing surfaces this to the caller and accumulates iteration
history.
"""
import json
import re
from datetime import datetime, timezone

from schemas.state import (
    STATE_ALTERNATIVE_SOLUTIONS,
    STATE_ARCHITECTURE,
    STATE_BUSINESS_PROBLEM,
    STATE_CANDIDATE_USE_CASES,
    STATE_COMPANY_CONTEXT,
    STATE_EVIDENCE_AUDIT,
    STATE_EXTERNAL_EVIDENCE,
    STATE_FINAL_RECOMMENDATION,
    STATE_IMPACT_ANALYSIS,
    STATE_ITERATION_COUNT,
    STATE_ITERATION_HISTORY,
    STATE_MAX_ITERATIONS,
    STATE_PROCESS_MAP,
    STATE_RISK_SUMMARY,
    STATE_ROI_ANALYSIS,
    STATE_SENSITIVITY_ANALYSIS,
    STATE_USE_CASE_SCORES,
)

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _excerpt(value, limit: int = 400) -> str:
    """Never raises. Missing/empty values get an explicit, honest label
    rather than silently rendering a blank table cell or 'None'."""
    if value is None:
        return "_(not produced this run)_"
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    if not text:
        return "_(empty)_"
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"... _(truncated, {len(text)} chars total)_"


def _stage_row(label: str, key: str, state: dict) -> str:
    value = state.get(key)
    ran = value is not None and str(value).strip() != ""
    status = "done" if ran else "not run / no output"
    excerpt = _excerpt(value, 160).replace("\n", " ").replace("|", "\\|")
    return f"| {label} | {status} | {excerpt} |"


def extract_recommendation_json(final_recommendation) -> dict | None:
    """Best-effort pull of the fenced ```json block out of
    final_synthesis_agent's free-text output. Returns None (never raises) if
    it's missing or unparseable -- report generation must not crash a run
    just because the model formatted its JSON block oddly."""
    if not final_recommendation or not isinstance(final_recommendation, str):
        return None
    match = _JSON_BLOCK_RE.search(final_recommendation)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def generate_process_report(state: dict) -> str:
    """Build the full Markdown "process + outcome" report from session state.

    `state` only needs to support `.get()` -- works with a plain dict, ADK's
    `State` object, or `callback_context.state` directly.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = []

    lines.append("# Enterprise GenAI Strategist -- Run Report")
    lines.append(f"_Generated {now}_")
    lines.append("")
    lines.append(
        "This report documents what the pipeline actually did during this run "
        "(Process Summary, Red-Team Iteration History), followed by the final "
        "outcome it produced."
    )
    lines.append("")

    # --- 1. Intake -----------------------------------------------------
    lines.append("## 1. Intake")
    lines.append(f"- **Business problem:** {_excerpt(state.get(STATE_BUSINESS_PROBLEM), 300)}")
    lines.append(f"- **Company context:** {_excerpt(state.get(STATE_COMPANY_CONTEXT), 300)}")
    lines.append(f"- **Process map:** {_excerpt(state.get(STATE_PROCESS_MAP), 300)}")
    lines.append("")

    # --- 2. Process summary ---------------------------------------------
    lines.append("## 2. Process Summary")
    lines.append("| Stage | Status | Output (excerpt) |")
    lines.append("|---|---|---|")
    lines.append(_stage_row("External research / evidence fusion", STATE_EXTERNAL_EVIDENCE, state))
    lines.append(_stage_row("Candidate AI use cases", STATE_CANDIDATE_USE_CASES, state))
    lines.append(_stage_row("Alternatives comparison", STATE_ALTERNATIVE_SOLUTIONS, state))
    lines.append(_stage_row("Use case scoring", STATE_USE_CASE_SCORES, state))
    lines.append(_stage_row("Architecture design", STATE_ARCHITECTURE, state))
    lines.append(_stage_row("Financial impact analysis", STATE_IMPACT_ANALYSIS, state))
    lines.append(_stage_row("ROI analysis", STATE_ROI_ANALYSIS, state))
    lines.append(_stage_row("Sensitivity analysis", STATE_SENSITIVITY_ANALYSIS, state))
    lines.append(_stage_row("Risk assessment", STATE_RISK_SUMMARY, state))
    lines.append(_stage_row("Evidence audit", STATE_EVIDENCE_AUDIT, state))
    lines.append("")

    # --- 3. Red-team / validate / revise loop ---------------------------
    iteration_count = state.get(STATE_ITERATION_COUNT, 0) or 0
    max_iterations = state.get(STATE_MAX_ITERATIONS)
    lines.append("## 3. Red-Team / Validate / Revise Loop")
    cap_note = f" / {max_iterations} allowed" if max_iterations is not None else ""
    lines.append(f"- **Iterations run:** {iteration_count}{cap_note}")
    if max_iterations is not None and iteration_count >= max_iterations:
        lines.append(
            "- \u26a0\ufe0f **Loop hit its iteration cap.** The validator did not "
            "explicitly confirm every weakness was resolved -- treat any "
            "unresolved items disclosed in the final outcome below as "
            "genuinely open, not a formality."
        )
    elif iteration_count > 0:
        lines.append(
            "- Loop exited because the validator confirmed no material "
            "weaknesses remained (or judged further iteration wouldn't help)."
        )
    lines.append("")

    history = state.get(STATE_ITERATION_HISTORY) or []
    if history:
        for snap in history:
            lines.append(f"### Iteration {snap.get('iteration', '?')}")
            lines.append(f"**Red team findings:** {_excerpt(snap.get('red_team_findings'))}")
            lines.append("")
            lines.append(f"**Validator's judgment:** {_excerpt(snap.get('validation_result'))}")
            lines.append("")
            lines.append(f"**Revision made:** {_excerpt(snap.get('revision_notes'))}")
            lines.append("")
    else:
        lines.append(
            "_No per-iteration history recorded (the loop may not have run "
            "this pass)._"
        )
        lines.append("")

    # --- 4. Evidence audit -----------------------------------------------
    lines.append("## 4. Evidence Audit")
    lines.append(_excerpt(state.get(STATE_EVIDENCE_AUDIT), 1500))
    lines.append("")

    # --- 5. Final outcome -------------------------------------------------
    lines.append("## 5. Final Outcome")
    final_recommendation = state.get(STATE_FINAL_RECOMMENDATION)
    parsed = extract_recommendation_json(final_recommendation)
    if parsed:
        lines.append(f"- **Recommendation:** {parsed.get('recommendation', 'n/a')}")
        lines.append(f"- **Confidence:** {parsed.get('confidence', 'n/a')}")
        if parsed.get("top_use_cases"):
            lines.append(f"- **Top use cases:** {', '.join(map(str, parsed['top_use_cases']))}")
        if parsed.get("risks"):
            lines.append(f"- **Key risks:** {', '.join(map(str, parsed['risks']))}")
        lines.append("")
    else:
        lines.append(
            "_Could not extract a structured recommendation summary "
            "(missing, or the JSON block didn't parse) -- see the full report below._"
        )
        lines.append("")

    lines.append("### Full Executive Report")
    lines.append(
        final_recommendation.strip()
        if isinstance(final_recommendation, str) and final_recommendation.strip()
        else "_No final recommendation was produced this run._"
    )
    lines.append("")

    return "\n".join(lines)
