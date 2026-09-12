"""
Research Director (Section 5): five research domains run IN PARALLEL because
they are independent of each other, followed by a sequential fusion step that
depends on all five having finished (Principles 7 & 8).

IMPORTANT GEMINI/VERTEX CONSTRAINT (hit in production, verified against
Google's docs): a single LlmAgent call cannot mix the built-in `google_search`
tool with custom FunctionTools unless they're merged into one `Tool` object
with both `google_search` and `function_declarations` set -- which ADK's
plain `tools=[google_search, some_function_tool]` list does NOT do. Passing
them as two separate tools causes:
    400 INVALID_ARGUMENT: "Multiple tools are supported only when they are
    all search tools."
Fix: each research domain is split into two agents in sequence --
  1. a *_search_agent with ONLY google_search (finds and summarizes info)
  2. a *_evidence_agent with ONLY register_evidence (extracts structured,
     sourced claims from step 1's output)
This is more portable than depending on model/version-specific combined-tool
support, and keeps the "never state a fact without registering evidence"
requirement (Section 6, Principle 5) intact.
"""
from google.adk.agents import LlmAgent, SequentialAgent

from tools.agent_helpers import parallel_or_sequential
from google.adk.tools import google_search, FunctionTool

from tools.evidence_tools import register_evidence, detect_conflicting_claims

MODEL = "gemini-2.5-flash"

register_evidence_tool = FunctionTool(register_evidence)
detect_conflicts_tool = FunctionTool(detect_conflicting_claims)

_EVIDENCE_EXTRACTION_INSTRUCTION = """Raw research findings to process:
{findings}

For every distinct external factual claim in the findings above, call
`register_evidence` with the claim text, the source name, the URL,
publication date if known, and your confidence (0-1). Never register a claim
that outruns what the findings actually say. If the findings were thin or
search was unavailable, register nothing and say so explicitly rather than
inventing a source (Principle 4: never fabricate a source). After
registering everything, output the same findings again alongside a short
note on evidence coverage."""


def _research_domain(key: str, description: str, research_focus: str, extra_context: str = ""):
    """Build a (search_agent -> evidence_agent) SequentialAgent for one research domain.

    Keeps google_search and register_evidence on SEPARATE agent calls per the
    constraint documented above.
    """
    search_agent = LlmAgent(
        name=f"{key}_search_agent",
        model=MODEL,
        description=description,
        instruction=f"""Business context: {{company_context?}}
{extra_context}
{research_focus}
Use google_search for current information. Output a structured summary of
findings, and for each finding note the source/URL/date if the search result
provides one.""",
        tools=[google_search],
        output_key=f"{key}_findings_raw",
    )
    evidence_agent = LlmAgent(
        name=f"{key}_evidence_agent",
        model=MODEL,
        description=f"Extracts sourced evidence records from {key} research findings.",
        instruction=_EVIDENCE_EXTRACTION_INSTRUCTION.format(findings=f"{{{key}_findings_raw}}"),
        tools=[register_evidence_tool],
        output_key=key,
    )
    return SequentialAgent(
        name=f"{key}_pipeline",
        description=f"Search then extract evidence for {key} research.",
        sub_agents=[search_agent, evidence_agent],
    )


_DIFFERENTIATION_DIRECTIVE = """
STRICT SCOPE: stay in your lane. Do NOT restate the general case for "why
GenAI helps this business" -- that's covered elsewhere. If your findings
would read the same regardless of which specific company this is, you
haven't searched narrowly enough; run another, more specific query. Every
claim must be something the OTHER research agents in this pipeline would
NOT independently produce."""

market_research_pipeline = _research_domain(
    key="market_research",
    description="Researches market trends, AI adoption, and industry benchmarks via web search.",
    research_focus="Research ONLY: overall market size and growth-rate figures for GenAI adoption "
    "in this specific industry (cite the analyst firm and year for every number), "
    "current AI adoption rate percentages among comparable-sized companies, and "
    "which specific emerging technologies/models are gaining traction right now. "
    "Do not discuss this company's own workflow or use cases -- that's handled "
    "elsewhere." + _DIFFERENTIATION_DIRECTIVE,
)

industry_research_pipeline = _research_domain(
    key="industry_research",
    description="Researches industry-specific AI applications, benchmarks and regulatory developments.",
    research_focus="Research ONLY: name at least 2 specific, real companies or published case "
    "studies (not hypothetical examples) in this exact industry that have deployed "
    "AI/GenAI, with concrete before/after metrics if published. Identify industry-"
    "specific operational benchmarks (e.g. typical cost-per-ticket, typical "
    "automation rate) that are DIFFERENT from generic cross-industry averages." +
    _DIFFERENTIATION_DIRECTIVE,
)

competitor_research_pipeline = _research_domain(
    key="competitor_research",
    description="Researches competitor AI initiatives and comparable public deployments.",
    research_focus="Research ONLY: NAMED competitors or comparable companies (same industry, "
    "similar size/segment) and their PUBLICLY ANNOUNCED AI initiatives, with "
    "company names, product/vendor names, and dates. If you cannot find named "
    "competitors for a company this specific/small, say so explicitly -- do NOT "
    "fall back to generic industry-wide statements; that is the industry "
    "research agent's job, not yours." + _DIFFERENTIATION_DIRECTIVE,
)

technology_research_pipeline = _research_domain(
    key="technology_research",
    description="Researches relevant AI/GenAI technologies and architectures.",
    research_focus="Research ONLY: specific named technologies, foundation models (with version "
    "numbers), vendors, and products relevant to the process map below -- not "
    "generic categories like 'GenAI' or 'NLP'. Name actual products (e.g. "
    "specific helpdesk-AI vendors, specific foundation model families) and what "
    "makes each one fit or not fit this workflow." + _DIFFERENTIATION_DIRECTIVE,
    extra_context="Process map: {process_map?}",
)

regulatory_research_pipeline = _research_domain(
    key="regulatory_research",
    description="Researches applicable regulations, standards, and AI governance requirements.",
    research_focus="Research: relevant regulations, standards, government guidance, industry "
    "requirements, privacy requirements, data residency considerations, and AI "
    "governance requirements applicable to this organization's industry and "
    "geography. Regulation changes frequently -- do not rely on prior knowledge "
    "alone." + _DIFFERENTIATION_DIRECTIVE,
)

_parallel_research = parallel_or_sequential(
    name="parallel_research",
    description="Runs all five research-domain pipelines (search -> evidence extraction) -- concurrently, unless STRATEGIST_LOW_QUOTA_MODE is set.",
    sub_agents=[
        market_research_pipeline,
        industry_research_pipeline,
        competitor_research_pipeline,
        technology_research_pipeline,
        regulatory_research_pipeline,
    ],
)

knowledge_fusion_agent = LlmAgent(
    name="knowledge_fusion_agent",
    model=MODEL,
    description="Fuses all research streams into one evidence base and flags contradictions.",
    instruction="""You have five research streams:

Market: {market_research?}
Industry: {industry_research?}
Competitor: {competitor_research?}
Technology: {technology_research?}
Regulatory: {regulatory_research?}

Call `detect_conflicting_claims` against the evidence records gathered so far
if available. Then produce ONE unified, de-duplicated synthesis: for each major
conclusion, list supporting evidence, source, confidence, assumptions, and any
contradictory evidence. If sources disagree, explicitly flag the disagreement
-- do not silently pick a side (Section 8).""",
    tools=[detect_conflicts_tool],
    output_key="external_evidence",
)

research_director = SequentialAgent(
    name="research_director",
    description="Parallel research across 5 domains (each search-then-extract), then sequential fusion.",
    sub_agents=[_parallel_research, knowledge_fusion_agent],
)
