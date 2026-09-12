"""Architecture Director (Section 13): six specialists in parallel, then a
sequential integration step."""
from google.adk.agents import LlmAgent, SequentialAgent

from tools.agent_helpers import parallel_or_sequential

MODEL = "gemini-2.5-flash"

_CONTEXT = """Top-scoring use cases: {use_case_scores?}
Candidate use cases: {candidate_use_cases?}
Technology research: {technology_research?}
Company context: {company_context?}
"""

application_architecture_agent = LlmAgent(
    name="application_architecture_agent",
    model=MODEL,
    description="Designs the application layer: UIs, agent interfaces, APIs, services.",
    instruction=_CONTEXT + "Design the application-layer architecture: user-facing "
    "applications, UIs, agent interfaces, APIs, and services needed for the "
    "top-scoring use cases.",
    output_key="application_architecture",
)

ai_architecture_agent = LlmAgent(
    name="ai_architecture_agent",
    model=MODEL,
    description="Designs the AI/agent layer: models, prompting, tools, memory, RAG, guardrails.",
    instruction=_CONTEXT + "Design the AI-layer architecture: foundation model choice(s) "
    "and rationale, prompting strategy, agent architecture, tool use, memory, "
    "RAG components, and guardrails.",
    output_key="ai_architecture",
)

data_architecture_agent = LlmAgent(
    name="data_architecture_agent",
    model=MODEL,
    description="Designs data sources, pipelines, vector stores, and governance.",
    instruction=_CONTEXT + "Data readiness findings: {data_readiness?}\n"
    "Design the data architecture: sources, pipelines, vector stores/databases, "
    "and data governance controls needed.",
    output_key="data_architecture",
)

integration_architecture_agent = LlmAgent(
    name="integration_architecture_agent",
    model=MODEL,
    description="Designs integrations with enterprise systems (CRM, ERP, workflow, auth).",
    instruction=_CONTEXT + "Internal evidence on current systems: {internal_evidence_validated?}\n"
    "Design the integration architecture: APIs, connections to existing "
    "enterprise systems (CRM, ERP, workflow engines), and authentication.",
    output_key="integration_architecture",
)

security_architecture_agent = LlmAgent(
    name="security_architecture_agent",
    model=MODEL,
    description="Designs IAM, encryption, secrets, network controls, and audit logging.",
    instruction=_CONTEXT + "Design the security architecture: IAM, encryption, secrets "
    "management, network controls, access control, and audit logging.",
    output_key="security_architecture",
)

observability_agent = LlmAgent(
    name="observability_agent",
    model=MODEL,
    description="Designs logging, tracing, evaluation, and cost/latency/quality monitoring.",
    instruction=_CONTEXT + "Design the observability approach: logging, tracing, agent "
    "evaluation, cost monitoring, latency monitoring, quality monitoring, and "
    "hallucination monitoring.",
    output_key="observability_design",
)

_parallel_architecture = parallel_or_sequential(
    name="parallel_architecture",
    description="Runs all six architecture specialists -- concurrently, unless STRATEGIST_LOW_QUOTA_MODE is set.",
    sub_agents=[
        application_architecture_agent,
        ai_architecture_agent,
        data_architecture_agent,
        integration_architecture_agent,
        security_architecture_agent,
        observability_agent,
    ],
)

architecture_integration_agent = LlmAgent(
    name="architecture_integration_agent",
    model=MODEL,
    description="Integrates the six architecture specialist outputs into one coherent design.",
    instruction="""Combine these architecture specialist outputs into one coherent
end-to-end architecture (Users -> Application -> Agent Orchestrator ->
Specialized Agents -> RAG/Enterprise Data -> Enterprise Systems ->
Governance/Security/Observability):

Application: {application_architecture?}
AI: {ai_architecture?}
Data: {data_architecture?}
Integration: {integration_architecture?}
Security: {security_architecture?}
Observability: {observability_design?}

Note any conflicts between specialists (e.g. AI agent wants a capability the
security agent flags as risky) and how you resolved them.""",
    output_key="architecture",
)

architecture_director = SequentialAgent(
    name="architecture_director",
    description="Parallel architecture specialists, then sequential integration into one design.",
    sub_agents=[_parallel_architecture, architecture_integration_agent],
)
