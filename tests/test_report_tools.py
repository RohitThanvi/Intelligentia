"""
Unit tests for tools/report_tools.py -- the deterministic (no-LLM) post-run
report generator. Pure function of a state dict, so these run in
milliseconds with no mocking of any model or ADK machinery.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.report_tools import extract_recommendation_json, generate_process_report  # noqa: E402


def test_empty_state_does_not_crash_and_labels_everything_missing():
    """A report generator that crashes on partial state would take down a
    run that otherwise completed fine -- it must degrade gracefully."""
    report = generate_process_report({})
    assert "# Enterprise GenAI Strategist -- Run Report" in report
    assert "(not produced this run)" in report
    assert "No final recommendation was produced this run" in report


def test_full_state_renders_all_sections():
    state = {
        "business_problem": "Manual invoice reconciliation takes 3 days.",
        "company_context": "Mid-size logistics firm, 200 employees.",
        "process_map": "Ops -> Finance -> Approval -> Payment.",
        "external_evidence": "3 sources found on invoice automation ROI.",
        "candidate_use_cases": "1. Invoice OCR + matching\n2. Approval routing bot",
        "alternative_solutions": "RPA scored higher feasibility than full GenAI.",
        "use_case_scores": "Invoice OCR: 8.2/10",
        "architecture": "FastAPI + doc-AI + human-in-the-loop approval queue.",
        "impact_analysis": "Saves ~40 hours/month.",
        "roi_analysis": "18-month payback.",
        "sensitivity_analysis": "Payback ranges 12-24 months across scenarios.",
        "risk_summary": "Moderate: vendor lock-in, data privacy.",
        "evidence_audit": "All major claims traced to at least one source.",
        "iteration_count": 2,
        "max_iterations": 3,
        "iteration_history": [
            {"iteration": 1, "red_team_findings": "ROI looks optimistic.",
             "validation_result": "Valid, must revise.", "revision_notes": "Lowered hours-saved estimate."},
            {"iteration": 2, "red_team_findings": "No material weaknesses left.",
             "validation_result": "Confirmed, exiting.", "revision_notes": "None needed."},
        ],
        "final_recommendation": (
            "## Executive Summary\nProceed with a pilot.\n\n"
            "```json\n"
            '{"recommendation": "PILOT", "confidence": 0.7, '
            '"top_use_cases": ["Invoice OCR"], "risks": ["vendor lock-in"]}'
            "\n```"
        ),
    }
    report = generate_process_report(state)

    # process summary reflects real stage outputs, not placeholders
    assert "Manual invoice reconciliation" in report
    assert "| External research / evidence fusion | done |" in report
    # iteration history is fully present, not just the last pass
    assert "Iteration 1" in report and "Iteration 2" in report
    assert "Lowered hours-saved estimate." in report
    # cap disclosure logic: 2 of 3 run, should NOT claim the cap was hit
    assert "hit its iteration cap" not in report
    # final outcome summary extracted correctly from the fenced JSON block
    assert "**Recommendation:** PILOT" in report
    assert "**Confidence:** 0.7" in report
    assert "Invoice OCR" in report
    # the full executive report text is still included, not just the summary
    assert "Proceed with a pilot." in report


def test_iteration_cap_is_explicitly_disclosed():
    state = {"iteration_count": 3, "max_iterations": 3}
    report = generate_process_report(state)
    assert "hit its iteration cap" in report


def test_iteration_cap_not_falsely_claimed_when_under_limit():
    state = {"iteration_count": 1, "max_iterations": 3}
    report = generate_process_report(state)
    assert "hit its iteration cap" not in report


def test_extract_recommendation_json_handles_missing_and_malformed_input():
    assert extract_recommendation_json(None) is None
    assert extract_recommendation_json("") is None
    assert extract_recommendation_json("no json block here") is None
    assert extract_recommendation_json("```json\n{not valid json\n```") is None


def test_extract_recommendation_json_parses_valid_block():
    text = 'Report text.\n```json\n{"recommendation": "GO", "confidence": 0.9}\n```\n'
    parsed = extract_recommendation_json(text)
    assert parsed == {"recommendation": "GO", "confidence": 0.9}


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
