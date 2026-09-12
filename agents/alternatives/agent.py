"""
Non-GenAI alternative agents (Section 10) -- run in parallel since each
evaluates the same candidate use cases independently, then a sequential
comparison step consumes all five plus the deterministic compare_alternatives
tool (Principle 1: don't let the LLM eyeball a ranking it can compute).
"""
from google.adk.agents import LlmAgent, SequentialAgent

from tools.agent_helpers import parallel_or_sequential
from google.adk.tools import FunctionTool

from tools.scoring_tools import compare_alternatives

compare_alternatives_tool = FunctionTool(compare_alternatives)

MODEL = "gemini-2.5-flash"

_COMMON_INPUTS = """Candidate use cases: {candidate_use_cases?}
Process map: {process_map?}
"""

rpa_evaluator = LlmAgent(
    name="rpa_evaluator",
    model=MODEL,
    description="Evaluates whether robotic process automation is a better fit than GenAI.",
    instruction=_COMMON_INPUTS + """Evaluate whether RPA (rule-based, structured-data automation)
would be a better fit than GenAI for these use cases. RPA is strong when
inputs are structured/rule-based and rules are stable; weak when inputs are
unstructured text/documents or require judgment. Rate cost, speed,
flexibility, and maintenance on 0-100 for an "RPA" option. Be explicit about
which specific candidate use cases RPA would and would not suit.""",
    output_key="rpa_evaluation",
)

workflow_automation_evaluator = LlmAgent(
    name="workflow_automation_evaluator",
    model=MODEL,
    description="Evaluates conventional workflow/BPM automation as an alternative.",
    instruction=_COMMON_INPUTS + """Evaluate whether conventional workflow/BPM automation
(routing, orchestration, notifications, deterministic business rules) would
suffice instead of GenAI. Rate cost, speed, flexibility, maintenance 0-100.
Be explicit about which candidate use cases this suits.""",
    output_key="workflow_automation_evaluation",
)

traditional_software_evaluator = LlmAgent(
    name="traditional_software_evaluator",
    model=MODEL,
    description="Evaluates whether conventional (non-AI) software is more appropriate.",
    instruction=_COMMON_INPUTS + """Evaluate whether conventional software (forms, validation
rules, templated document generation, standard integrations -- no ML/AI at
all) would be more appropriate and cheaper than GenAI for these use cases.
Rate cost, speed, flexibility, maintenance 0-100.""",
    output_key="traditional_software_evaluation",
)

analytics_evaluator = LlmAgent(
    name="analytics_evaluator",
    model=MODEL,
    description="Evaluates traditional BI/statistical analytics as an alternative.",
    instruction=_COMMON_INPUTS + """Evaluate whether traditional BI/statistical analytics
(dashboards, rule-based flags, statistical models) would meet the need instead
of GenAI, particularly for any decision-support or reporting-style candidate
use cases. Rate cost, speed, flexibility, maintenance 0-100.""",
    output_key="analytics_evaluation",
)

ml_evaluator = LlmAgent(
    name="ml_evaluator",
    model=MODEL,
    description="Evaluates conventional (non-generative) machine learning as an alternative.",
    instruction=_COMMON_INPUTS + """Evaluate whether conventional supervised/unsupervised ML
(classification, regression, anomaly detection -- not generative/LLM-based)
would be sufficient, especially for prediction, scoring, or anomaly-flagging
style candidate use cases. Rate cost, speed, flexibility, maintenance 0-100.""",
    output_key="ml_evaluation",
)

human_process_evaluator = LlmAgent(
    name="human_process_evaluator",
    model=MODEL,
    description="Evaluates process redesign or staffing changes as an alternative to any technology.",
    instruction=_COMMON_INPUTS + """Evaluate whether the real fix is process redesign,
policy change, or additional/reallocated staffing rather than any new
technology at all. This is a legitimate alternative and should not be
dismissed by default. Rate cost, speed, flexibility, maintenance 0-100.""",
    output_key="human_process_evaluation",
)

_parallel_alternatives = parallel_or_sequential(
    name="parallel_alternatives",
    description="Evaluates all five non-GenAI alternatives -- concurrently, unless STRATEGIST_LOW_QUOTA_MODE is set.",
    sub_agents=[
        rpa_evaluator,
        workflow_automation_evaluator,
        traditional_software_evaluator,
        analytics_evaluator,
        ml_evaluator,
        human_process_evaluator,
    ],
)

alternatives_comparison_agent = LlmAgent(
    name="alternatives_comparison_agent",
    model=MODEL,
    description="Deterministically ranks alternatives and forces an explicit GenAI-vs-simpler-tech justification.",
    instruction="""You have five/six alternative evaluations:
RPA: {rpa_evaluation?}
Workflow automation: {workflow_automation_evaluation?}
Traditional software: {traditional_software_evaluation?}
Analytics: {analytics_evaluation?}
ML: {ml_evaluation?}
Human/process redesign: {human_process_evaluation?}
Candidate use cases: {candidate_use_cases?}

Call `compare_alternatives` with a "candidates" list built from each
evaluation's cost/speed/flexibility/maintenance scores (map speed->speed_score,
flexibility->flexibility_score, maintenance->maintenance_score, cost->cost_score)
to get a deterministic ranking, then also add a "GenAI" candidate using your
own best estimate of its scores.

Then, for EACH candidate use case, explicitly answer: "Why should this be
GenAI rather than a simpler technology?" If the answer is weak, say so
plainly and downgrade the GenAI recommendation for that use case
(Section 10). Output the ranking plus this per-use-case justification.""",
    tools=[compare_alternatives_tool],
    output_key="alternative_solutions",
)

alternatives_director = SequentialAgent(
    name="alternatives_director",
    description="Parallel evaluation of non-GenAI alternatives, then deterministic comparison and GenAI justification.",
    sub_agents=[_parallel_alternatives, alternatives_comparison_agent],
)
