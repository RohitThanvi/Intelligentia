"""data_readiness_agent (Section 12) + use_case_scoring_agent (Section 11)."""
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool

from tools.scoring_tools import calculate_use_case_score

score_tool = FunctionTool(calculate_use_case_score)

MODEL = "gemini-2.5-flash"

data_readiness_agent = LlmAgent(
    name="data_readiness_agent",
    model=MODEL,
    description="Assesses data availability, quality, and governance blockers for each candidate use case.",
    instruction="""Candidate use cases: {candidate_use_cases?}
Internal evidence: {internal_evidence_validated?}
Business context: {company_context?}

For each candidate use case, assess: data availability, quality, volume,
freshness, ownership, accessibility, sensitivity, label availability,
unstructured-data availability, integration complexity, and data governance.
Produce a 0-100 data-readiness score per use case and explicitly list
blockers. Where you're not certain, mark it as an assumption to validate
rather than asserting readiness.""",
    output_key="data_readiness",
)

use_case_scoring_agent = LlmAgent(
    name="use_case_scoring_agent",
    model=MODEL,
    description="Scores every candidate use case with the deterministic weighted scoring tool.",
    instruction="""Candidate use cases: {candidate_use_cases?}
Data readiness: {data_readiness?}
Alternative-solution analysis (for risk/feasibility signal): {alternative_solutions?}

For EVERY candidate use case, call `calculate_use_case_score` with your best
0-100 ratings for business_value, technical_feasibility (use data_readiness
and technology research as input), data_readiness, time_to_value,
scalability, and risk. You are NOT allowed to compute the weighted score or
classification yourself -- always call the tool. After scoring all use cases,
summarize them in a ranked table (name, final_score, classification).""",
    tools=[score_tool],
    output_key="use_case_scores",
)

use_case_evaluation_pipeline = SequentialAgent(
    name="use_case_evaluation_pipeline",
    description="Assess data readiness, then deterministically score each use case (scoring depends on readiness).",
    sub_agents=[data_readiness_agent, use_case_scoring_agent],
)
