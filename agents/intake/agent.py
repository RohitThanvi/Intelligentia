"""
Intake Director (Section 4): sequential pipeline because problem_analysis and
process_mapping genuinely depend on business_context having been extracted first.
"""
from google.adk.agents import LlmAgent, SequentialAgent

MODEL = "gemini-2.5-flash"

business_context_agent = LlmAgent(
    name="business_context_agent",
    model=MODEL,
    description="Extracts the organization's business profile from the user's request.",
    instruction="""You extract structured business context from the user's description
of their organization. Identify: company/organization profile, industry, geography,
business model, strategic objectives, revenue/cost drivers, workforce size and
composition, existing technology landscape, current AI maturity, and strategic
constraints. If information is missing, explicitly list it as an open question rather
than inventing it. Output a clear structured summary (use markdown headers/bullets).""",
    output_key="company_context",
)

problem_analysis_agent = LlmAgent(
    name="problem_analysis_agent",
    model=MODEL,
    description="Identifies business problems, root causes, and pain points.",
    instruction="""Using the business context already gathered:
{company_context?}

Identify: business problems, root causes (not just symptoms), process
bottlenecks, manual/repetitive work, decision bottlenecks, customer pain
points, employee pain points, revenue leakage, cost drivers, and risk
exposure. Distinguish symptoms from root causes explicitly. Output a
structured summary.""",
    output_key="business_problem",
)

process_mapping_agent = LlmAgent(
    name="process_mapping_agent",
    model=MODEL,
    description="Maps the relevant end-to-end workflow and flags AI opportunity points.",
    instruction="""Using the business context and problem analysis:

Business context: {company_context?}
Problem analysis: {business_problem?}

Map the relevant workflow as: Trigger -> Input -> Process steps -> Decision
points -> Human interactions -> Systems -> Outputs -> Exceptions. For each
stage, note where AI could plausibly augment or automate work, and be explicit
that this is a *candidate* opportunity, not a conclusion yet (that comes later
in the pipeline). Output a structured process map.""",
    output_key="process_map",
)

intake_director = SequentialAgent(
    name="intake_director",
    description="Understands the organization: business context -> problem analysis -> process map, in that order because each depends on the last.",
    sub_agents=[business_context_agent, problem_analysis_agent, process_mapping_agent],
)
