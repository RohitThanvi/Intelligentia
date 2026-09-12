"""
Evidence aggregation / validation tools (Sections 6, 8, 18, Principles 3-5).

Deterministic bookkeeping only — claim extraction and judgement of relevance
stays with the LLM agents; this module just gives them a structured, auditable
place to register and later validate evidence.
"""
from datetime import datetime, timezone


def register_evidence(claim: str, source: str, url: str = "", publication_date: str = "",
                       confidence: float = 0.5, evidence_type: str = "source_reported_claim") -> dict:
    """Create a canonical, timestamped evidence record.

    Args:
        claim: The specific factual claim being recorded, in plain text.
        source: Name/title of the source (e.g. publication, report, internal doc).
        url: URL of the source, if external. Empty string if internal/none.
        publication_date: Original publication date if known, else "".
        confidence: 0.0-1.0 confidence in this specific claim.
        evidence_type: one of verified_fact | source_reported_claim | model_inference |
                        assumption | recommendation.

    Returns:
        The evidence record dict, ready to append to shared state's evidence list.
    """
    return {
        "claim": claim,
        "source": source,
        "url": url,
        "publication_date": publication_date,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "confidence": max(0.0, min(1.0, confidence)),
        "evidence_type": evidence_type,
    }


def validate_evidence(claims: list, evidence_records: list) -> dict:
    """Check whether each claim has at least one backing evidence record.

    This is a structural/coverage check (does a record with a matching-enough
    'claim' text exist?), not a semantic proof — it exists to stop the system
    from silently asserting things with zero recorded provenance (Principle 4).

    Args:
        claims: list of claim strings the final report intends to assert.
        evidence_records: list of evidence dicts (as produced by register_evidence).

    Returns:
        dict with unsupported_claims (list) and coverage_ratio (0-1).
    """
    supported_texts = {e.get("claim", "").strip().lower() for e in evidence_records}
    unsupported = []
    for claim in claims:
        key = claim.strip().lower()
        # loose containment check rather than exact match
        if not any(key in s or s in key for s in supported_texts if s):
            unsupported.append(claim)
    coverage_ratio = 1.0 if not claims else round(1 - len(unsupported) / len(claims), 2)
    return {"unsupported_claims": unsupported, "coverage_ratio": coverage_ratio,
            "total_claims": len(claims), "supported_claims": len(claims) - len(unsupported)}


def detect_conflicting_claims(evidence_records: list) -> dict:
    """Flag pairs of evidence records that look like they might disagree.

    Heuristic only (shares no simple ground truth to check against): groups
    records by rough topic overlap via shared significant words, then flags
    groups with >1 distinct source as needing human/agent review rather than
    silently picking one (Section 8: 'If sources disagree, do not silently choose one').

    Args:
        evidence_records: list of evidence dicts.

    Returns:
        dict with potentially_conflicting_groups: list of lists of evidence claims.
    """
    import re
    from collections import defaultdict

    def sig_words(text):
        words = re.findall(r"[a-zA-Z]{5,}", text.lower())
        return set(words)

    groups = defaultdict(list)
    for e in evidence_records:
        words = sig_words(e.get("claim", ""))
        key = tuple(sorted(words))[:3]  # coarse bucket key
        groups[key].append(e)

    conflicting = [g for g in groups.values() if len({x.get("source") for x in g}) > 1]
    return {"potentially_conflicting_groups": [
        [{"claim": e["claim"], "source": e["source"]} for e in g] for g in conflicting
    ]}
