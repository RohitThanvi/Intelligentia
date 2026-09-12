"""
Deterministic financial tools (Section 14, Principle 1).

These are plain Python functions wrapped as ADK FunctionTools. The LLM calls
them with structured arguments and *interprets* the results in prose — it
never performs the arithmetic itself. Every function returns the inputs it
used alongside the outputs so the calculation is auditable/traceable
(Principle 12).
"""

from typing import Optional


def calculate_impact(
    hours_saved_per_week: float,
    fully_loaded_hourly_cost: float,
    fte_count_affected: int,
    revenue_impact_annual: float = 0.0,
    error_reduction_pct: float = 0.0,
    avoided_cost_annual: float = 0.0,
) -> dict:
    """Compute annual labor savings and total quantified impact.

    Args:
        hours_saved_per_week: Estimated hours saved per affected employee per week.
        fully_loaded_hourly_cost: Fully loaded cost per hour for affected employees.
        fte_count_affected: Number of FTEs whose work is affected.
        revenue_impact_annual: Estimated annual revenue impact (0 if none/unknown).
        error_reduction_pct: Estimated % reduction in error-driven rework (0-100).
        avoided_cost_annual: Other annual costs avoided (e.g. penalties, rework).

    Returns:
        dict with labor_savings_annual, total_annual_benefit, and all inputs echoed back.
    """
    weeks_per_year = 52
    labor_savings_annual = hours_saved_per_week * fully_loaded_hourly_cost * fte_count_affected * weeks_per_year
    total_annual_benefit = labor_savings_annual + revenue_impact_annual + avoided_cost_annual
    return {
        "inputs": {
            "hours_saved_per_week": hours_saved_per_week,
            "fully_loaded_hourly_cost": fully_loaded_hourly_cost,
            "fte_count_affected": fte_count_affected,
            "revenue_impact_annual": revenue_impact_annual,
            "error_reduction_pct": error_reduction_pct,
            "avoided_cost_annual": avoided_cost_annual,
        },
        "labor_savings_annual": round(labor_savings_annual, 2),
        "total_annual_benefit": round(total_annual_benefit, 2),
    }


def calculate_roi(
    annual_benefit: float,
    annual_operating_cost: float,
    implementation_cost: float,
) -> dict:
    """Compute net annual benefit, ROI %, and payback period in months.

    Args:
        annual_benefit: Total annual benefit (e.g. from calculate_impact).
        annual_operating_cost: Ongoing annual cost to run the solution.
        implementation_cost: One-time implementation/capex cost.

    Returns:
        dict with net_annual_benefit, roi_percent, payback_period_months.
        payback_period_months is None if net monthly benefit <= 0 (never pays back).
    """
    net_annual_benefit = annual_benefit - annual_operating_cost
    roi_percent = (net_annual_benefit / implementation_cost * 100) if implementation_cost else None
    monthly_net_benefit = net_annual_benefit / 12
    payback_period_months = (
        round(implementation_cost / monthly_net_benefit, 1)
        if monthly_net_benefit > 0 and implementation_cost > 0
        else None
    )
    return {
        "inputs": {
            "annual_benefit": annual_benefit,
            "annual_operating_cost": annual_operating_cost,
            "implementation_cost": implementation_cost,
        },
        "net_annual_benefit": round(net_annual_benefit, 2),
        "roi_percent": round(roi_percent, 1) if roi_percent is not None else None,
        "payback_period_months": payback_period_months,
        "payback_note": "Never pays back under these assumptions" if payback_period_months is None else None,
    }


def sensitivity_analysis(
    base_hours_saved_per_week: float,
    fully_loaded_hourly_cost: float,
    fte_count_affected: int,
    implementation_cost: float,
    annual_operating_cost: float,
    base_revenue_impact_annual: float = 0.0,
    adoption_rate_pct: float = 100.0,
) -> dict:
    """Run conservative / base / optimistic ROI scenarios by varying adoption and efficiency.

    Conservative = 60% of base adoption & benefit. Optimistic = 130% of base.
    This keeps the multiplier logic deterministic and visible rather than asking
    the LLM to invent scenario numbers.

    Args:
        base_hours_saved_per_week: Base-case hours saved per employee per week.
        fully_loaded_hourly_cost: Fully loaded hourly cost.
        fte_count_affected: FTEs affected.
        implementation_cost: One-time implementation cost.
        annual_operating_cost: Ongoing annual operating cost.
        base_revenue_impact_annual: Base-case annual revenue impact.
        adoption_rate_pct: Assumed adoption rate, 0-100, applied on top of scenario multiplier.

    Returns:
        dict with conservative/base/optimistic scenario results.
    """
    scenarios = {"conservative": 0.6, "base": 1.0, "optimistic": 1.3}
    adoption_factor = max(0.0, min(adoption_rate_pct, 100.0)) / 100.0
    results = {}
    for name, multiplier in scenarios.items():
        hours = base_hours_saved_per_week * multiplier * adoption_factor
        revenue = base_revenue_impact_annual * multiplier * adoption_factor
        impact = calculate_impact(
            hours_saved_per_week=hours,
            fully_loaded_hourly_cost=fully_loaded_hourly_cost,
            fte_count_affected=fte_count_affected,
            revenue_impact_annual=revenue,
        )
        roi = calculate_roi(
            annual_benefit=impact["total_annual_benefit"],
            annual_operating_cost=annual_operating_cost,
            implementation_cost=implementation_cost,
        )
        results[name] = {"impact": impact, "roi": roi}
    return {"adoption_rate_pct": adoption_rate_pct, "scenarios": results}
