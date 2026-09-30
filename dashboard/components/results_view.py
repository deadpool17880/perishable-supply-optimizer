"""Dashboard Component: Allocation Results & Recovery Inspection."""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from src.models import AllocationResult
from src.visualization import plot_allocation_sankey


def render_allocation_details(result: AllocationResult, title: str = "Batch Allocation Details"):
    """Displays detailed per-batch routing, spoilage, and delivered quantities."""
    st.subheader(f"📋 {title}")

    rows = []
    for a in result.allocations:
        via = f"Storage {a.storage_id}" if a.storage_id else "Direct Route"
        rows.append({
            "Batch ID": a.batch_id,
            "Origin Farm": a.farm_id,
            "Transit Path": via,
            "Destination Market": a.market_id,
            "Dispatched (kg)": round(a.quantity, 1),
            "Spoiled (kg)": round(a.spoiled_qty, 1),
            "Delivered Fresh (kg)": round(a.delivered_qty, 1),
            "Spoilage %": f"{a.spoilage_fraction * 100:.2f}%",
            "Transport Time (h)": f"{a.travel_time_hours:.1f}h",
            "Cost ($)": f"${a.transport_cost:.2f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("🌊 Flow Distribution (Sankey Diagram)")
    fig_sankey = plot_allocation_sankey(result)
    st.plotly_chart(fig_sankey, use_container_width=True)


def render_recovery_panel(recovery_stats: Dict[str, Any]):
    """Renders the recovery summary card after dynamic re-optimization."""
    st.subheader("🛡️ Dynamic Re-Optimization & Recovery Metrics")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Recovery Decision Time", f"{recovery_stats['recovery_time_seconds'] * 1000:.2f} ms")
    with c2:
        st.metric("Affected Batches", f"{recovery_stats['affected_batch_count']}")
    with c3:
        st.metric("Stranded Volume", f"{recovery_stats['stranded_quantity_kg']:,.1f} kg")
    with c4:
        st.metric(
            "Rerouted Volume",
            f"{recovery_stats['successfully_rerouted_kg']:,.1f} kg",
            delta=f"{recovery_stats['reroute_success_rate_pct']:.1f}% recovered",
        )

    st.markdown(
        f"**Rerouting Summary**: Disrupted corridors affected {recovery_stats['affected_batch_count']} batches "
        f"(`{', '.join(recovery_stats['affected_batches'])}`). The dynamic optimizer preserved operational in-flight flows "
        f"and rerouted **{recovery_stats['successfully_rerouted_kg']:,.1f} kg** of stranded inventory to alternate hubs and markets, "
        f"limiting spoilage change to **{recovery_stats['spoilage_delta_kg']:+.1f} kg** and cost delta to **${recovery_stats['cost_delta']:+,.2f}**."
    )
