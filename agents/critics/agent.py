"""
Red Team / Validation / Iterative Loop (Sections 16-17, Principle 9: bounded loops).

ADK's LoopAgent runs its sub_agents repeatedly until max_iterations is hit OR
a sub-agent calls the `exit_loop` tool, which sets tool_context.actions.escalate
so the loop's parent breaks out (this is the current supported ADK escalation
pattern for loops). We increment/check STATE_ITERATION_COUNT ourselves so the
final report can disclose how many iterations ran (Section 17: "If maximum
iterations are reached, explicitly disclose that").
"""
from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent
from google.adk.tools import FunctionTool, ToolContext

from config import settings
from tools.agent_helpers import get_reasoning_model
from schemas.state import STATE_ITERATION_HISTORY

MODEL = get_reasoning_model("gemini-2.5-flash")  # same default as before; NVIDIA NIM/Nemotron if opted in via .env
# no google_search or VertexAiRagRetrieval used anywhere in this file --
# safe to route through a non-Gemini provider


def exit_loop(reason: str, tool_context: ToolContext) -> dict:
    """Signal that the red-team/validation loop should stop iterating.

    Call this ONLY when the validator has confirmed no material weaknesses
    remain (or genuinely believes further iteration won't help).

    Args:
        reason: Why the loop should stop now.
        tool_context: Injected automatically by ADK.

    Returns:
        dict confirming the escalation was signaled.
    """
    tool_context.actions.escalate = True
    return {"stopped": True, "reason": reason}


def increment_iteration(tool_context: ToolContext) -> dict:
    """Increment and return the current loop iteration count in session state.

    Call once at the start of each pass through the red-team/validator loop.
    """
    count = tool_context.state.get("iteration_count", 0) + 1
    tool_context.state["iteration_count"] = count
    return {"iteration_count": count, "max_iterations": settings.MAX_ITERATIONS}


exit_loop_tool = FunctionTool(exit_loop)
increment_iteration_tool = FunctionTool(increment_iteration)

red_team_agent = LlmAgent(
    name="red_team_agent",
    model=MODEL,
    description="Aggressively attacks the current recommendation before it's finalized.",
    instruction="""Call `increment_iteration` first.

Current state to attack:
Use case scores: {use_case_scores?}
Alternatives comparison: {alternative_solutions?}
Architecture: {architecture?}
Financial model -- impact: {impact_analysis?}, roi: {roi_analysis?}, sensitivity: {sensitivity_analysis?}
Risk summary: {risk_summary?}

You have no financial-calculation or scoring tools -- those numbers were
already computed upstream by specialists who do have them, and are given to
you above as-is. Critique them by reasoning about their INPUTS and
ASSUMPTIONS in plain text (e.g. "the hours-saved assumption looks
optimistic given X"). Do not attempt to invoke any function to recompute
them. Your only tool is `increment_iteration`.

Ask, and answer honestly: Is GenAI actually necessary? Are assumptions
unrealistic? Is the ROI overstated? Is the data sufficient? Are there
simpler alternatives that were underrated? Are important risks missing? Is
the architecture unnecessarily complex? Are external sources trustworthy?
Are RAG sources actually relevant? Are there contradictory sources? Could
the workflow be automated without an LLM? What happens when the model is
wrong? What happens when external services fail?

List concrete, material weaknesses (not nitpicks). If a prior iteration's
findings exist in {red_team_findings?}, check whether they were actually
addressed -- don't just repeat yourself. (On the first pass this will be
empty -- that's expected, just do the initial critique.)""",
    tools=[increment_iteration_tool],
    output_key="red_team_findings",
)

validator_agent = LlmAgent(
    name="validator_agent",
    model=MODEL,
    description="Independently judges whether the red team's criticisms are valid and material.",
    instruction="""Red team findings: {red_team_findings?}

You have no tools except `exit_loop` -- do not attempt to call any
calculation/scoring tool to verify numbers; reason about them in plain text.

Independently assess each red-team criticism: is it valid? Is it material
enough to require revising the recommendation, or is it a minor caveat that
can just be disclosed in the final report? You are not the red team's rubber
stamp -- push back on criticisms you think are overstated too.

If there are NO material weaknesses left (or you judge that further
iteration will not meaningfully improve the analysis), call `exit_loop` with
your reasoning. Otherwise, clearly state what must be revised and let the
loop continue.""",
    tools=[exit_loop_tool],
    output_key="validation_result",
)

revision_agent = LlmAgent(
    name="revision_agent",
    model=MODEL,
    description="Revises the affected parts of the analysis based on validated red-team findings.",
    instruction="""Validation result: {validation_result?}
Red team findings: {red_team_findings?}

You have no tools at all -- describe what should change in plain text; do
not attempt to call any calculation/scoring/financial tool. A later,
tool-equipped step will actually recompute anything that genuinely needs
recomputing.

If the validator identified material weaknesses requiring revision, revise
your view of the specific affected area(s) -- use case scores, financial
assumptions, architecture, or risk register -- and clearly state what
changed and why, referencing the specific criticism it addresses. If the
validator called exit_loop, briefly confirm no revision is needed this pass.""",
    output_key="revision_notes",
)

async def _record_iteration_snapshot(callback_context):
    """Runs once per pass through the loop body (red_team -> validator ->
    revision), BEFORE the next pass overwrites red_team_findings /
    validation_result / revision_notes with output_key. Deterministic --
    doesn't depend on any agent remembering to call a tool -- so the process
    report (tools/report_tools.py) always gets a full history, not just
    whatever survived from the last iteration.
    """
    state = callback_context.state
    history = list(state.get(STATE_ITERATION_HISTORY, []) or [])
    history.append({
        "iteration": state.get("iteration_count"),
        "red_team_findings": state.get("red_team_findings"),
        "validation_result": state.get("validation_result"),
        "revision_notes": state.get("revision_notes"),
    })
    state[STATE_ITERATION_HISTORY] = history
    return None


_critic_loop_body = SequentialAgent(
    name="critic_loop_body",
    description="One pass: red team attacks, validator judges, revision agent responds.",
    sub_agents=[red_team_agent, validator_agent, revision_agent],
    after_agent_callback=_record_iteration_snapshot,
)

red_team_loop = LoopAgent(
    name="red_team_loop",
    description="Repeats red-team/validate/revise until no material weaknesses remain, bounded by max_iterations.",
    sub_agents=[_critic_loop_body],
    max_iterations=settings.MAX_ITERATIONS,
)
