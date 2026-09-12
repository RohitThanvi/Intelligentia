"""
Shared state schema for the Enterprise GenAI Strategist multi-agent system.

ADK session state is a flat dict-like object (ctx.session.state / tool_context.state).
We namespace keys with a prefix so agents don't collide, and every agent writes
structured (JSON-serializable) data here instead of passing giant strings around.

Convention: state key names are ALL_CAPS constants below. Agents read their
inputs from state and write their outputs back to state under their own key.
"""

# --- Intake ---
STATE_COMPANY_CONTEXT = "company_context"
STATE_BUSINESS_PROBLEM = "business_problem"
STATE_PROCESS_MAP = "process_map"

# --- Research (external evidence) ---
STATE_MARKET_RESEARCH = "market_research"
STATE_INDUSTRY_RESEARCH = "industry_research"
STATE_COMPETITOR_RESEARCH = "competitor_research"
STATE_TECHNOLOGY_RESEARCH = "technology_research"
STATE_REGULATORY_RESEARCH = "regulatory_research"
STATE_EXTERNAL_EVIDENCE = "external_evidence"  # fused list of evidence dicts

# --- Use cases & alternatives ---
STATE_CANDIDATE_USE_CASES = "candidate_use_cases"
STATE_ALTERNATIVE_SOLUTIONS = "alternative_solutions"
STATE_USE_CASE_SCORES = "use_case_scores"
STATE_DATA_READINESS = "data_readiness"

# --- Architecture ---
STATE_ARCHITECTURE = "architecture"

# --- Financial ---
STATE_IMPACT_ANALYSIS = "impact_analysis"
STATE_ROI_ANALYSIS = "roi_analysis"
STATE_SENSITIVITY_ANALYSIS = "sensitivity_analysis"

# --- Risk ---
STATE_SECURITY_RISK = "security_risk"
STATE_PRIVACY_RISK = "privacy_risk"
STATE_COMPLIANCE_RISK = "compliance_risk"
STATE_OPERATIONAL_RISK = "operational_risk"
STATE_ADOPTION_RISK = "adoption_risk"
STATE_RISK_SUMMARY = "risk_summary"

# --- Red team / iteration ---
STATE_RED_TEAM_FINDINGS = "red_team_findings"
STATE_VALIDATION_RESULT = "validation_result"
STATE_ITERATION_COUNT = "iteration_count"
STATE_MAX_ITERATIONS = "max_iterations"
STATE_LOOP_SHOULD_STOP = "loop_should_stop"  # bool, checked by escalation tool

# --- Final ---
STATE_FINAL_RECOMMENDATION = "final_recommendation"

DEFAULT_MAX_ITERATIONS = 3


def evidence_item(claim: str, source: str, url: str = "", publication_date: str = "",
                   confidence: float = 0.5, evidence_type: str = "source_reported_claim") -> dict:
    """
    Canonical evidence record shape (Principle 5: preserve provenance).
    evidence_type in {"verified_fact","source_reported_claim","model_inference",
                       "assumption","recommendation"}
    """
    from datetime import datetime, timezone
    return {
        "claim": claim,
        "source": source,
        "url": url,
        "publication_date": publication_date,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "confidence": confidence,
        "evidence_type": evidence_type,
    }


def structured_conclusion(conclusion: str, evidence: list, assumptions: list,
                           uncertainties: list, confidence: float,
                           recommended_action: str) -> dict:
    """Canonical shape every analytical agent should emit (see Section 18)."""
    return {
        "conclusion": conclusion,
        "evidence": evidence,
        "assumptions": assumptions,
        "uncertainties": uncertainties,
        "confidence": confidence,
        "recommended_action": recommended_action,
    }
