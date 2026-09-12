"""
Deterministic scoring tools (Section 11, Principle 1).
"""

DEFAULT_WEIGHTS = {
    "business_value": 0.30,
    "technical_feasibility": 0.20,
    "data_readiness": 0.15,
    "time_to_value": 0.15,
    "scalability": 0.10,
    "risk": 0.10,  # penalty, subtracted
}

DEFAULT_THRESHOLDS = [
    (0, 30, "Not Recommended"),
    (31, 50, "Explore / Pilot"),
    (51, 75, "Strategic Investment"),
    (76, 100, "Quick Win"),
]


def calculate_use_case_score(
    business_value: float,
    technical_feasibility: float,
    data_readiness: float,
    time_to_value: float,
    scalability: float,
    risk: float,
    weights: dict = None,
    thresholds: list = None,
) -> dict:
    """Compute the weighted use-case score from 0-100 component ratings.

    Args:
        business_value: 0-100 rating.
        technical_feasibility: 0-100 rating.
        data_readiness: 0-100 rating.
        time_to_value: 0-100 rating (higher = faster time to value).
        scalability: 0-100 rating.
        risk: 0-100 rating where higher = MORE risk (this component is subtracted).
        weights: optional override of DEFAULT_WEIGHTS.
        thresholds: optional override of DEFAULT_THRESHOLDS as list of (min,max,label).

    Returns:
        dict with component_scores, weighted_scores, final_score, classification, explanation.
    """
    w = weights or DEFAULT_WEIGHTS
    t = thresholds or DEFAULT_THRESHOLDS

    components = {
        "business_value": business_value,
        "technical_feasibility": technical_feasibility,
        "data_readiness": data_readiness,
        "time_to_value": time_to_value,
        "scalability": scalability,
        "risk": risk,
    }
    weighted = {
        k: (components[k] * w[k] if k != "risk" else -(components[k] * w[k]))
        for k in components
    }
    final_score = max(0.0, min(100.0, sum(weighted.values())))

    classification = "Unclassified"
    for lo, hi, label in t:
        if lo <= final_score <= hi:
            classification = label
            break

    explanation = (
        f"Final score {final_score:.1f} = "
        + " + ".join(f"{k}({components[k]:.0f}*{w[k]:.2f})" for k in components if k != "risk")
        + f" - risk({components['risk']:.0f}*{w['risk']:.2f}) => {classification}"
    )

    return {
        "component_scores": components,
        "weights_used": w,
        "weighted_scores": {k: round(v, 2) for k, v in weighted.items()},
        "final_score": round(final_score, 1),
        "classification": classification,
        "explanation": explanation,
    }


SOURCE_QUALITY_TIERS = {
    "government_regulatory": 1.0,
    "official_company_documentation": 0.9,
    "academic_research": 0.85,
    "industry_organization": 0.75,
    "consulting_research_firm": 0.7,
    "reputable_journalism": 0.6,
    "secondary_website": 0.4,
    "forum_social_media": 0.15,
}


def source_quality_score(source_type: str) -> dict:
    """Map a source-type label to a numeric quality/confidence weight (Section 6).

    Args:
        source_type: one of SOURCE_QUALITY_TIERS keys. Unknown values score 0.3.

    Returns:
        dict with source_type, quality_score, tier_rank.
    """
    score = SOURCE_QUALITY_TIERS.get(source_type, 0.3)
    ranked = sorted(SOURCE_QUALITY_TIERS.items(), key=lambda x: -x[1])
    rank = next((i + 1 for i, (k, _) in enumerate(ranked) if k == source_type), len(ranked) + 1)
    return {"source_type": source_type, "quality_score": score, "tier_rank": rank}


def compare_alternatives(candidates: list) -> dict:
    """Rank non-GenAI vs GenAI alternatives by a simple normalized composite score.

    Args:
        candidates: list of dicts like
            {"name": "RPA", "cost_score": 70, "speed_score": 80,
             "flexibility_score": 30, "maintenance_score": 60}
            All sub-scores are 0-100, higher is better.

    Returns:
        dict with ranked list (name, composite_score) sorted descending, and the raw inputs.
    """
    ranked = []
    for c in candidates:
        subs = [c.get("cost_score", 0), c.get("speed_score", 0),
                c.get("flexibility_score", 0), c.get("maintenance_score", 0)]
        composite = sum(subs) / len(subs) if subs else 0
        ranked.append({"name": c.get("name", "unnamed"), "composite_score": round(composite, 1),
                        "components": c})
    ranked.sort(key=lambda x: -x["composite_score"])
    return {"ranked": ranked}
