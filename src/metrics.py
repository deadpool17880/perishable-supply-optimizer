"""Comprehensive Evaluation Metrics for Perishable Supply Chain Optimization."""

from typing import Dict, Any, List
from src.models import AllocationResult


def calculate_comparative_metrics(
    baseline_result: AllocationResult,
    proposed_result: AllocationResult,
) -> Dict[str, Any]:
    """Computes rigorous comparative performance metrics between baseline and proposed method.

    Definitions:
    - Spoilage Reduction % = (Baseline_Spoilage - Proposed_Spoilage) / Baseline_Spoilage * 100
    - Demand Fulfillment Diff = Proposed_Fulfillment% - Baseline_Fulfillment%
    - Cost Delta % = (Proposed_Cost - Baseline_Cost) / Baseline_Cost * 100
    - Risk Reduction % = (Baseline_Risk - Proposed_Risk) / Baseline_Risk * 100
    """
    b_spoil = baseline_result.total_spoilage
    p_spoil = proposed_result.total_spoilage
    spoil_reduction_pct = (
        ((b_spoil - p_spoil) / b_spoil * 100.0) if b_spoil > 1e-4 else 0.0
    )

    b_fulf = baseline_result.demand_fulfillment_percentage
    p_fulf = proposed_result.demand_fulfillment_percentage
    fulf_gain_pct_pts = p_fulf - b_fulf

    b_cost = baseline_result.total_transport_cost
    p_cost = proposed_result.total_transport_cost
    cost_change_pct = (
        ((p_cost - b_cost) / b_cost * 100.0) if b_cost > 1e-4 else 0.0
    )

    b_risk = baseline_result.total_risk_score
    p_risk = proposed_result.total_risk_score
    risk_reduction_pct = (
        ((b_risk - p_risk) / b_risk * 100.0) if b_risk > 1e-4 else 0.0
    )

    return {
        "baseline_method": baseline_result.method_name,
        "proposed_method": proposed_result.method_name,
        "baseline_spoilage_kg": round(b_spoil, 1),
        "proposed_spoilage_kg": round(p_spoil, 1),
        "spoilage_reduction_percentage": round(spoil_reduction_pct, 2),
        "baseline_fulfillment_pct": round(b_fulf, 2),
        "proposed_fulfillment_pct": round(p_fulf, 2),
        "fulfillment_gain_percentage_points": round(fulf_gain_pct_pts, 2),
        "baseline_cost": round(b_cost, 2),
        "proposed_cost": round(p_cost, 2),
        "cost_change_percentage": round(cost_change_pct, 2),
        "baseline_unmet_demand_kg": round(baseline_result.unmet_demand, 1),
        "proposed_unmet_demand_kg": round(proposed_result.unmet_demand, 1),
        "risk_reduction_percentage": round(risk_reduction_pct, 2),
        "runtime_speedup_factor": round(
            (baseline_result.runtime_seconds / max(1e-6, proposed_result.runtime_seconds)),
            2,
        ),
    }


def summarize_allocation_result(result: AllocationResult) -> Dict[str, Any]:
    """Creates a standardized dict summary for tables and reporting."""
    return {
        "Method": result.method_name,
        "Status": result.solver_status,
        "Dispatched (kg)": round(result.total_dispatched, 1),
        "Delivered (kg)": round(result.total_delivered, 1),
        "Spoilage (kg)": round(result.total_spoilage, 1),
        "Spoilage (%)": f"{result.spoilage_percentage:.2f}%",
        "Demand Fulfillment (%)": f"{result.demand_fulfillment_percentage:.2f}%",
        "Unmet Demand (kg)": round(result.unmet_demand, 1),
        "Transport Cost ($)": f"${result.total_transport_cost:,.2f}",
        "Risk Score": round(result.total_risk_score, 1),
        "Runtime (s)": f"{result.runtime_seconds:.5f}s",
    }
