"""Risk Director (Section 15): five specialists in parallel, then aggregation."""
from google.adk.agents import LlmAgent, SequentialAgent

from tools.agent_helpers import parallel_or_sequential

MODEL = "gemini-2.5-flash"

_CTX = """Architecture: {architecture?}
Top use cases: {use_case_scores?}
Regulatory research: {regulatory_research?}
"""

security_risk_agent = LlmAgent(
    name="security_risk_agent",
    model=MODEL,
    description="Assesses security risks: prompt injection, data leakage, tool abuse, etc.",
    instruction=_CTX + "Assess: prompt injection, data leakage, unauthorized access, "
    "tool abuse, identity risks, and supply-chain risks in the proposed "
    "architecture. Rate severity and note mitigations already present in the "
    "architecture vs. gaps.",
    output_key="security_risk",
)

privacy_risk_agent = LlmAgent(
    name="privacy_risk_agent",
    model=MODEL,
    description="Assesses privacy risks: PII, retention, residency, consent.",
    instruction=_CTX + "Internal governance evidence: {internal_evidence_validated?}\n"
    "Assess: PII exposure, sensitive information handling, data retention, "
    "data residency, consent, and data minimization. Cross-check against any "
    "internal governance policy evidence retrieved.",
    output_key="privacy_risk",
)

compliance_risk_agent = LlmAgent(
    name="compliance_risk_agent",
    model=MODEL,
    description="Assesses regulatory/compliance risk: applicable regulation, auditability, explainability.",
    instruction=_CTX + "Assess: applicable regulation, auditability, explainability "
    "requirements, record-keeping, and regulatory exposure. Be specific about "
    "which regulations apply given the researched regulatory context.",
    output_key="compliance_risk",
)

operational_risk_agent = LlmAgent(
    name="operational_risk_agent",
    model=MODEL,
    description="Assesses operational risk: reliability, hallucination, vendor dependency, cost volatility.",
    instruction=_CTX + "Assess: reliability, model failure modes, hallucination risk, "
    "vendor dependency, availability, and cost volatility (token/API cost "
    "exposure at the proposed scale).",
    output_key="operational_risk",
)

adoption_risk_agent = LlmAgent(
    name="adoption_risk_agent",
    model=MODEL,
    description="Assesses people-side risk: employee adoption, training, change management, trust.",
    instruction=_CTX + "Process map / current staffing: {process_map?}\n"
    "Assess: employee adoption risk, training needs, change management, "
    "trust, and workflow disruption for the affected staff.",
    output_key="adoption_risk",
)

_parallel_risk = parallel_or_sequential(
    name="parallel_risk",
    description="Runs all five risk specialists -- concurrently, unless STRATEGIST_LOW_QUOTA_MODE is set.",
    sub_agents=[
        security_risk_agent,
        privacy_risk_agent,
        compliance_risk_agent,
        operational_risk_agent,
        adoption_risk_agent,
    ],
)

risk_aggregator_agent = LlmAgent(
    name="risk_aggregator_agent",
    model=MODEL,
    description="Aggregates the five risk assessments into one prioritized risk register.",
    instruction="""Combine into one prioritized risk register (highest severity first):
Security: {security_risk?}
Privacy: {privacy_risk?}
Compliance: {compliance_risk?}
Operational: {operational_risk?}
Adoption: {adoption_risk?}

For each risk: severity, likelihood, affected use case(s), and recommended
mitigation. Flag any risk severe enough that it should block a "GO"
recommendation on its own.""",
    output_key="risk_summary",
)

risk_director = SequentialAgent(
    name="risk_director",
    description="Parallel risk specialists, then sequential aggregation into a prioritized register.",
    sub_agents=[_parallel_risk, risk_aggregator_agent],
)
