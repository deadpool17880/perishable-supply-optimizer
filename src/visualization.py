"""Visualization Engine for DPSRO.

Builds publication-grade Plotly interactive figures:
1. Supply-chain network topology graph with flow weights and failure alerts
2. Multi-algorithm comparative bar charts (Spoilage %, Demand Fulfillment %, Cost)
3. Scalability curve (Problem Size vs HiGHS Solver Runtime)
4. Dynamic allocation Sankey diagram (Farms -> Storage Hubs -> Market Sinks)
5. Monte Carlo robustness distribution boxplots
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, Optional, List
from src.models import SupplyChainState, AllocationResult


def plot_network_graph(
    state: SupplyChainState,
    result: Optional[AllocationResult] = None,
    title: str = "Supply-Chain Network Topology & Flow Allocation",
) -> go.Figure:
    """Renders interactive 2D node-link network diagram with flows, node types, and disruption highlights."""
    fig = go.Figure()

    # Track active flows on routes if result is provided
    route_flow: Dict[str, float] = {}
    if result:
        for alloc in result.allocations:
            for rid in alloc.route_ids:
                route_flow[rid] = route_flow.get(rid, 0.0) + alloc.quantity

    max_flow = max(route_flow.values()) if route_flow else 1.0

    # 1. Draw Route Edges
    for rid, route in state.routes.items():
        src_pos = None
        dst_pos = None

        # Resolve coordinates
        if route.source in state.farms:
            src_pos = state.farms[route.source].location
        elif route.source in state.storage_facilities:
            src_pos = state.storage_facilities[route.source].location

        if route.destination in state.storage_facilities:
            dst_pos = state.storage_facilities[route.destination].location
        elif route.destination in state.markets:
            dst_pos = state.markets[route.destination].location

        if not src_pos or not dst_pos:
            continue

        flow = route_flow.get(rid, 0.0)

        # Style edge based on status
        if not route.available:
            edge_color = "rgba(239, 68, 68, 0.85)"  # Red dashed (disrupted)
            edge_width = 3.0
            dash_style = "dash"
            hover_text = f"ROUTE CLOSED: {rid}<br>Source: {route.source} -> Dest: {route.destination}<br>Corridor Offline"
        elif flow > 0:
            # Active transport flow
            edge_color = "rgba(16, 185, 129, 0.9)"  # Emerald green
            edge_width = max(2.5, min(8.0, 2.5 + (flow / max_flow) * 5.5))
            dash_style = "solid"
            hover_text = f"Route: {rid}<br>Flow: {flow:,.1f} kg<br>Distance: {route.distance_km:.1f} km<br>Travel Time: {route.travel_time_hours:.1f}h"
        else:
            # Operational but zero flow
            edge_color = "rgba(148, 163, 184, 0.35)"  # Slate faint
            edge_width = 1.2
            dash_style = "dot"
            hover_text = f"Route: {rid}<br>Flow: 0 kg (Idle)<br>Distance: {route.distance_km:.1f} km"

        fig.add_trace(
            go.Scatter(
                x=[src_pos[0], dst_pos[0]],
                y=[src_pos[1], dst_pos[1]],
                mode="lines",
                line=dict(color=edge_color, width=edge_width, dash=dash_style),
                hoverinfo="text",
                text=hover_text,
                showlegend=False,
            )
        )

    # 2. Draw Nodes (Farms, Storages, Markets)
    # (a) Farms
    farm_x = [f.location[0] for f in state.farms.values()]
    farm_y = [f.location[1] for f in state.farms.values()]
    farm_hover = [
        f"Farm: {f.farm_id}<br>Available Produce: {f.available_quantity:,.0f} kg"
        for f in state.farms.values()
    ]
    farm_labels = [f.farm_id for f in state.farms.values()]

    fig.add_trace(
        go.Scatter(
            x=farm_x,
            y=farm_y,
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=18, color="#10B981", line=dict(width=1.5, color="#064E3B")),
            text=farm_labels,
            textposition="top center",
            hovertext=farm_hover,
            hoverinfo="text",
            name="Farms (Supply)",
        )
    )

    # (b) Storage Facilities
    st_x, st_y, st_hover, st_labels, st_colors = [], [], [], [], []
    for s in state.storage_facilities.values():
        st_x.append(s.location[0])
        st_y.append(s.location[1])
        st_labels.append(s.storage_id)
        if not s.available:
            st_colors.append("#EF4444")  # Red for outage
            st_hover.append(f"COLD STORAGE OUTAGE: {s.storage_id}<br>Status: OFFLINE (Power Failure)")
        else:
            st_colors.append("#3B82F6")  # Blue
            st_hover.append(
                f"Cold Storage: {s.storage_id}<br>Capacity: {s.capacity:,.0f} kg<br>Current Load: {s.current_load:,.0f} kg<br>Temp: {s.temperature_capacity}°C"
            )

    fig.add_trace(
        go.Scatter(
            x=st_x,
            y=st_y,
            mode="markers+text",
            marker=dict(symbol="square", size=20, color=st_colors, line=dict(width=1.5, color="#1E3A8A")),
            text=st_labels,
            textposition="top center",
            hovertext=st_hover,
            hoverinfo="text",
            name="Cold Storage Hubs",
        )
    )

    # (c) Markets
    m_x = [m.location[0] for m in state.markets.values()]
    m_y = [m.location[1] for m in state.markets.values()]
    m_hover = [
        f"Market: {m.market_id}<br>Demand: {m.demand:,.0f} kg<br>Priority: {m.priority}x"
        for m in state.markets.values()
    ]
    m_labels = [m.market_id for m in state.markets.values()]

    fig.add_trace(
        go.Scatter(
            x=m_x,
            y=m_y,
            mode="markers+text",
            marker=dict(symbol="circle", size=18, color="#F59E0B", line=dict(width=1.5, color="#78350F")),
            text=m_labels,
            textposition="top center",
            hovertext=m_hover,
            hoverinfo="text",
            name="Markets (Demand)",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Inter, sans-serif")),
        xaxis=dict(title="X Coordinate (km)", showgrid=True, zeroline=False),
        yaxis=dict(title="Y Coordinate (km)", showgrid=True, zeroline=False),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#F8FAFC"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=520,
    )
    return fig


def plot_comparison_bar_charts(results: List[AllocationResult]) -> go.Figure:
    """Generates grouped bar chart comparing Spoilage %, Demand Fulfillment %, and Cost."""
    methods = [r.method_name for r in results]
    spoilage = [r.spoilage_percentage for r in results]
    fulfillment = [r.demand_fulfillment_percentage for r in results]
    costs = [r.total_transport_cost for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Spoilage % (Lower is better)",
        x=methods,
        y=spoilage,
        marker_color="#EF4444",
        text=[f"{v:.2f}%" for v in spoilage],
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        name="Demand Fulfillment % (Higher is better)",
        x=methods,
        y=fulfillment,
        marker_color="#10B981",
        text=[f"{v:.1f}%" for v in fulfillment],
        textposition="auto",
    ))

    fig.update_layout(
        barmode="group",
        title="Comparative Optimization Performance Across Methods",
        yaxis=dict(title="Percentage (%)", range=[0, 115]),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#F8FAFC"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )
    return fig


def plot_scalability_curve(df_scalability: pd.DataFrame) -> go.Figure:
    """Plots problem size vs runtime scaling curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_scalability["num_batches"],
        y=df_scalability["runtime_dpsro_seconds"],
        mode="lines+markers",
        name="Proposed DPSRO (SciPy HiGHS)",
        line=dict(color="#38BDF8", width=3),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=df_scalability["num_batches"],
        y=df_scalability["runtime_cheapest_seconds"],
        mode="lines+markers",
        name="Cheapest Baseline (LP)",
        line=dict(color="#F59E0B", width=2, dash="dash"),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=df_scalability["num_batches"],
        y=df_scalability["runtime_nearest_seconds"],
        mode="lines+markers",
        name="Nearest Baseline (Greedy)",
        line=dict(color="#94A3B8", width=2, dash="dot"),
        marker=dict(size=6),
    ))

    fig.update_layout(
        title="Scalability Analysis: Problem Size vs Solver Execution Time",
        xaxis=dict(title="Number of Perishable Batches (Problem Scale)", type="linear"),
        yaxis=dict(title="Runtime (Seconds)", type="log"),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#F8FAFC"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )
    return fig


def plot_allocation_sankey(result: AllocationResult) -> go.Figure:
    """Builds interactive Sankey diagram of allocation flows: Farm -> Hub / Direct -> Market."""
    nodes = []
    node_map = {}

    def get_node_idx(name: str, color: str = "#3B82F6") -> int:
        if name not in node_map:
            node_map[name] = len(nodes)
            nodes.append({"label": name, "color": color})
        return node_map[name]

    sources = []
    targets = []
    values = []
    link_colors = []

    for alloc in result.allocations:
        if alloc.quantity <= 0.1:
            continue

        farm_node = get_node_idx(f"Farm {alloc.farm_id}", "#10B981")
        market_node = get_node_idx(f"Market {alloc.market_id}", "#F59E0B")

        if alloc.storage_id:
            storage_node = get_node_idx(f"Storage {alloc.storage_id}", "#38BDF8")
            # Farm -> Storage
            sources.append(farm_node)
            targets.append(storage_node)
            values.append(alloc.quantity)
            link_colors.append("rgba(56, 189, 248, 0.4)")

            # Storage -> Market
            sources.append(storage_node)
            targets.append(market_node)
            values.append(alloc.delivered_qty)
            link_colors.append("rgba(16, 185, 129, 0.4)")
        else:
            # Direct Farm -> Market
            sources.append(farm_node)
            targets.append(market_node)
            values.append(alloc.delivered_qty)
            link_colors.append("rgba(245, 158, 11, 0.4)")

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[n["label"] for n in nodes],
                color=[n["color"] for n in nodes],
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
            ),
        )
    )

    fig.update_layout(
        title="Perishable Flow Distribution: Farms → Hubs / Direct → Markets",
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#F8FAFC", size=12),
        height=420,
    )
    return fig


def plot_robustness_boxplots(df_experiments: pd.DataFrame) -> go.Figure:
    """Generates box plots of spoilage percentage distributions across 100 Monte Carlo scenarios."""
    post_df = df_experiments[df_experiments["stage"] == "Post-Disruption"]
    fig = px.box(
        post_df,
        x="method",
        y="spoilage_pct",
        color="method",
        title="Robustness Distribution: Post-Disruption Spoilage % (100 Monte Carlo Scenarios)",
        labels={"spoilage_pct": "Spoilage Percentage (%)", "method": "Optimization Algorithm"},
        color_discrete_map={
            "Nearest Market Baseline": "#EF4444",
            "Cheapest Route Baseline": "#F59E0B",
            "Dynamic Re-Optimization (DPSRO)": "#10B981",
        },
    )
    fig.update_layout(
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#F8FAFC"),
        showlegend=False,
        height=400,
    )
    return fig
