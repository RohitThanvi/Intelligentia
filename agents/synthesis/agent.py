"""evidence_audit_agent + final_synthesis_agent (Sections 18, 23, 24)."""
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool

from tools.evidence_tools import validate_evidence
from schemas.state import DEFAULT_MAX_ITERATIONS
from tools.agent_helpers import get_reasoning_model

MODEL = get_reasoning_model("gemini-2.5-pro")  # same default as before; NVIDIA NIM/Nemotron if opted in via .env
# no google_search or VertexAiRagRetrieval used anywhere in this file --
# safe to route through a non-Gemini provider

validate_evidence_tool = FunctionTool(validate_evidence)

evidence_audit_agent = LlmAgent(
    name="evidence_audit_agent",
    model=MODEL,
    description="Final check that every major claim in the analysis is traceable to evidence.",
    instruction="""Review the full analysis assembled so far:
External evidence: {external_evidence?}
Internal evidence: {internal_evidence_validated?}
Use case scores: {use_case_scores?}
Financial model: {roi_analysis?}, {sensitivity_analysis?}
Risk summary: {risk_summary?}
Red team / validation history: {red_team_findings?}, {validation_result?}

List the key factual claims the final report is about to make, then call
`validate_evidence` against the evidence records to confirm coverage. Flag
any claim with no traceable evidence so the synthesis agent either sources it
properly, or labels it explicitly as an assumption/inference rather than a
fact (Principle 11: separate facts, assumptions, inference, recommendations).""",
    tools=[validate_evidence_tool],
    output_key="evidence_audit",
)

final_synthesis_agent = LlmAgent(
    name="final_synthesis_agent",
    model=MODEL,
    description="Produces the final executive-level strategy report and machine-readable summary.",
    instruction=f"""You are the managing partner delivering the final deliverable. Synthesize
everything gathered by the team into ONE executive-level strategy document.

Business context: {{company_context?}}
Business problem: {{business_problem?}}
Process map: {{process_map?}}
External research/evidence: {{external_evidence?}}
Internal evidence: {{internal_evidence_validated?}}
Candidate use cases: {{candidate_use_cases?}}
Use case scores: {{use_case_scores?}}
Alternatives comparison: {{alternative_solutions?}}
Architecture: {{architecture?}}
Financial -- impact: {{impact_analysis?}}, roi: {{roi_analysis?}}, sensitivity: {{sensitivity_analysis?}}
Risk summary: {{risk_summary?}}
Red team findings across iterations: {{red_team_findings?}}
Validation result: {{validation_result?}}
Revision notes: {{revision_notes?}}
Evidence audit: {{evidence_audit?}}
Iteration count: {{iteration_count?}} (max allowed: {DEFAULT_MAX_ITERATIONS} -- if this equals the max,
explicitly disclose that iteration was capped and some findings may be unresolved)

Where agents disagreed (e.g. an alternatives evaluator said a simpler
technology suffices, or red team challenged the ROI), do NOT just average
the views -- explain what the disagreement is, why it exists, which evidence
supports each side, which position is stronger and why, and what additional
information would resolve it (Section 19).

Produce the report with these sections: Executive Summary; Business Context;
Current-State Process; Identified Problems; Candidate AI Use Cases (as a
ranked markdown table: Rank | Use Case | Score | Value | Feasibility | Risk |
Recommendation); GenAI vs Alternatives; Recommended Architecture; Data
Requirements; Financial Model (implementation cost, operating cost, annual
benefit, net benefit, ROI, payback, conservative/base/optimistic scenarios);
Risk Assessment; Implementation Roadmap (Phase 0 Discovery through Phase 4
Scale); Governance; Success Metrics; Key Assumptions; Unresolved Questions;
Evidence and Sources.

Then output a fenced ```json block with:
{{{{"recommendation": "GO|PILOT|NO-GO", "confidence": 0.0, "top_use_cases": [],
"alternatives_considered": [], "roi": {{{{}}}}, "risks": [],
"critical_assumptions": [], "evidence": [], "implementation_phases": []}}}}
""",
    output_key="final_recommendation",
)

synthesis_pipeline = SequentialAgent(
    name="synthesis_pipeline",
    description="Audit evidence coverage, then produce the final executive report.",
    sub_agents=[evidence_audit_agent, final_synthesis_agent],
)
