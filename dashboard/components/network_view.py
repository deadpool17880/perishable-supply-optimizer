"""Dashboard Component: Network Topology & Flow Visualization."""

import streamlit as st
from typing import Optional
from src.models import SupplyChainState, AllocationResult
from src.visualization import plot_network_graph


def render_network_view(
    state: SupplyChainState,
    result: Optional[AllocationResult] = None,
    title: str = "Supply-Chain Network Topology & Flow Allocation",
):
    """Renders the interactive Plotly network graph."""
    fig = plot_network_graph(state=state, result=result, title=title)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ Network Legend & Coordinate Mapping"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("🟢 **Farms (Supply Sinks)**: Origin points where produce is harvested with initial shelf life.")
        with c2:
            st.markdown("🟦 **Cold Storages (Hubs)**: Temperature-controlled facilities (4°C) mitigating spoilage.")
        with c3:
            st.markdown("🟠 **Markets (Demand Sinks)**: Urban centers with target demand and delivery priority.")
        st.markdown(
            "- **Green Solid Lines**: Active transportation flows (thickness proportional to dispatched volume).\n"
            "- **Red Dashed Lines**: Disrupted / severed corridors unavailable for routing.\n"
            "- **Gray Dotted Lines**: Inactive but operational routes."
        )
