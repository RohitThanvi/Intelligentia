"""use_case_discovery_agent (Section 9)."""
from google.adk.agents import LlmAgent

MODEL = "gemini-2.5-flash"

use_case_discovery_agent = LlmAgent(
    name="use_case_discovery_agent",
    model=MODEL,
    description="Generates a broad candidate set of AI/GenAI use cases before any selection.",
    instruction="""Inputs:
Business problem: {business_problem?}
Process map: {process_map?}
Internal evidence: {internal_evidence_validated?}
External evidence/research: {external_evidence?}

Generate a broad candidate set of AI/GenAI use cases relevant to this
organization (do not settle on one yet -- breadth first). For EACH candidate,
produce a JSON object with fields: name, problem, users, workflow,
ai_capability, inputs, outputs, human_role, expected_value, dependencies.
Ground each candidate in the business problem and process map -- don't
propose generic use cases disconnected from what was actually found.
Output the full list as JSON.""",
    output_key="candidate_use_cases",
)
