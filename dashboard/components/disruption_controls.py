"""Dashboard Component: Interactive Disruption Injection Controls."""

import streamlit as st
from typing import Tuple, Optional
from src.models import SupplyChainState
from src.disruption import (
    Disruption,
    TruckFailureDisruption,
    StorageOutageDisruption,
    DemandShockDisruption,
    TemperatureShockDisruption,
    RouteClosureDisruption,
    CombinedDisruption,
)


def render_disruption_controls(state: SupplyChainState) -> Tuple[Optional[Disruption], bool]:
    """Renders controls to inject disruptions into the supply chain network."""
    st.subheader("⚠️ Disruption Injection Console")

    disruption_type = st.selectbox(
        "Select Disruption Class",
        [
            "Truck Breakdown (Corridor Interruption)",
            "Cold Storage Outage (Hub Shutdown)",
            "Demand Shock (Sudden Market Volatility)",
            "Heatwave Shock (Thermal Deterioration)",
            "Route Closure (Highway Blockage)",
            "Combined Multi-Vector Crisis",
        ],
        index=0,
    )

    disruption: Optional[Disruption] = None

    if "Truck" in disruption_type:
        st.info("Simulates a vehicle breakdown en route, closing a primary transportation corridor.")
        avail_trucks = [tid for tid, t in state.trucks.items() if t.available]
        chosen_trk = st.selectbox("Select Truck to Fail", avail_trucks if avail_trucks else ["Auto-select"])
        truck_id = chosen_trk if chosen_trk != "Auto-select" else None
        disruption = TruckFailureDisruption(truck_id=truck_id)

    elif "Storage" in disruption_type:
        st.warning("Simulates a complete electrical or cooling outage at a cold-storage hub, reducing capacity to 0.")
        avail_storages = [sid for sid, s in state.storage_facilities.items() if s.available]
        chosen_st = st.selectbox("Select Storage Hub to Fail", avail_storages if avail_storages else ["Auto-select"])
        st_id = chosen_st if chosen_st != "Auto-select" else None
        disruption = StorageOutageDisruption(storage_id=st_id)

    elif "Demand" in disruption_type:
        st.info("Simulates macroeconomic or local retail volatility shifting demand across markets.")
        mult = st.slider("Demand Multiplier (1.0 = Normal, 1.4 = +40% Surge, 0.7 = -30% Drop)", 0.5, 2.0, 1.35, 0.05)
        target_m = st.selectbox("Target Market", ["All Markets"] + list(state.markets.keys()))
        m_id = None if target_m == "All Markets" else target_m
        disruption = DemandShockDisruption(market_id=m_id, multiplier=mult)

    elif "Heatwave" in disruption_type:
        st.error("Simulates an acute heatwave elevating temperatures and accelerating biological spoilage kinetics.")
        temp_delta = st.slider("Temperature Elevation (+°C)", 2.0, 14.0, 8.0, 1.0)
        disruption = TemperatureShockDisruption(delta_temp_c=temp_delta)

    elif "Route" in disruption_type:
        st.warning("Simulates roadway obstruction, landslide, or congestion on a specific route.")
        avail_routes = [rid for rid, r in state.routes.items() if r.available]
        chosen_r = st.selectbox("Select Route Corridor", avail_routes if avail_routes else ["Auto-select"])
        r_id = chosen_r if chosen_r != "Auto-select" else None
        disruption = RouteClosureDisruption(route_id=r_id)

    elif "Combined" in disruption_type:
        st.error("Simulates a compound catastrophe: Simultaneous storage outage, truck breakdown, and regional heatwave.")
        disruption = CombinedDisruption()

    trigger_clicked = st.button("🚨 INJECT DISRUPTION", type="primary", use_container_width=True)
    return disruption, trigger_clicked
