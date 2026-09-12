"""
Financial Director (Section 14). Sequential because ROI depends on impact,
and sensitivity depends on both. All arithmetic goes through deterministic
tools (Principle 1) -- the LLM's job is to choose sensible input estimates
from evidence and interpret the results, never to do the math itself.
"""
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool

from tools.financial_tools import calculate_impact, calculate_roi, sensitivity_analysis

impact_tool = FunctionTool(calculate_impact)
roi_tool = FunctionTool(calculate_roi)
sensitivity_tool = FunctionTool(sensitivity_analysis)

MODEL = "gemini-2.5-flash"

impact_analysis_agent = LlmAgent(
    name="impact_analysis_agent",
    model=MODEL,
    description="Estimates hours saved, FTE impact, revenue impact using the deterministic impact tool.",
    instruction="""Top use cases: {use_case_scores?}
Business/process context: {process_map?}
Company context: {company_context?}

For the top-scoring use case(s), estimate realistic inputs (hours saved per
week per employee, fully loaded hourly cost, FTE count affected, any revenue
impact, error reduction %, avoided costs). State each estimate's source
(user-provided assumption vs. your inference vs. research benchmark) BEFORE
calling `calculate_impact`. Never compute totals by hand -- always call the
tool. If the user provided specific figures (e.g. headcount, employee cost),
use them and say so explicitly.""",
    tools=[impact_tool],
    output_key="impact_analysis",
)

roi_agent = LlmAgent(
    name="roi_agent",
    model=MODEL,
    description="Computes ROI and payback using the deterministic ROI tool.",
    instruction="""Impact analysis: {impact_analysis?}

Using the total_annual_benefit from the impact analysis plus a stated annual
operating cost and implementation cost (use user-provided figures if given;
otherwise estimate and flag as an assumption), call `calculate_roi`. Never
compute ROI or payback yourself -- always call the tool. Present the result
plainly, including if payback never occurs under these assumptions.""",
    tools=[roi_tool],
    output_key="roi_analysis",
)

sensitivity_analysis_agent = LlmAgent(
    name="sensitivity_analysis_agent",
    model=MODEL,
    description="Runs conservative/base/optimistic ROI scenarios using the deterministic tool.",
    instruction="""Impact analysis: {impact_analysis?}
ROI analysis: {roi_analysis?}

Call `sensitivity_analysis` using the same base inputs as impact_analysis
(hours saved, hourly cost, FTE count, implementation cost, operating cost,
revenue impact) plus a reasonable adoption_rate_pct assumption (state it
explicitly). Present conservative/base/optimistic ROI and payback side by
side. Never present a single ROI number without these bounding scenarios
(Section 14: "Never present an ROI number without showing the assumptions
behind it").""",
    tools=[sensitivity_tool],
    output_key="sensitivity_analysis",
)

financial_director = SequentialAgent(
    name="financial_director",
    description="Impact -> ROI -> Sensitivity, in dependency order, using deterministic financial tools throughout.",
    sub_agents=[impact_analysis_agent, roi_agent, sensitivity_analysis_agent],
)
