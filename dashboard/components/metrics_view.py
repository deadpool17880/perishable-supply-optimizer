"""Dashboard Component: High-Impact Performance Metrics & Comparative Cards."""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
from src.models import AllocationResult
from src.metrics import calculate_comparative_metrics, summarize_allocation_result
from src.visualization import plot_comparison_bar_charts


def render_metric_cards(result: AllocationResult, baseline_result: Optional[AllocationResult] = None):
    """Renders high-visibility KPI cards with optional delta vs baseline."""
    c1, c2, c3, c4, c5 = st.columns(5)

    delta_spoil = None
    delta_fulf = None
    delta_cost = None
    if baseline_result:
        comp = calculate_comparative_metrics(baseline_result, result)
        delta_spoil = f"-{comp['spoilage_reduction_percentage']:.1f}% vs {baseline_result.method_name}"
        delta_fulf = f"+{comp['fulfillment_gain_percentage_points']:.1f}% pts"
        delta_cost = f"{comp['cost_change_percentage']:+.1f}%"

    with c1:
        st.metric(
            label="Spoilage Rate",
            value=f"{result.spoilage_percentage:.2f}%",
            delta=delta_spoil,
            delta_color="inverse",
            help="Total spoiled mass divided by total mass dispatched.",
        )
    with c2:
        st.metric(
            label="Demand Fulfillment",
            value=f"{result.demand_fulfillment_percentage:.1f}%",
            delta=delta_fulf,
            help="Usable fresh produce successfully delivered vs total demand.",
        )
    with c3:
        st.metric(
            label="Total Transport Cost",
            value=f"${result.total_transport_cost:,.2f}",
            delta=delta_cost,
            delta_color="inverse",
            help="Transportation and routing expenditure.",
        )
    with c4:
        st.metric(
            label="Dispatched Produce",
            value=f"{result.total_dispatched:,.1f} kg",
            help="Total inventory dispatched from farm gates.",
        )
    with c5:
        st.metric(
            label="HiGHS Runtime",
            value=f"{result.runtime_seconds * 1000:.2f} ms",
            help="Computational solver execution time.",
        )


def render_comparison_table(results: List[AllocationResult]):
    """Renders formatted comparison table across multiple optimization methods."""
    data = [summarize_allocation_result(r) for r in results]
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    fig = plot_comparison_bar_charts(results)
    st.plotly_chart(fig, use_container_width=True)
