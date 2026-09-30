"""Disruption Engine for Perishable Supply Chains.

Simulates dynamic operational disruptions across network assets:
1. Truck / Transport Asset Breakdown (corridor or vehicle failure)
2. Cold Storage Facility Outage (complete hub shutdown)
3. Market Demand Shocks (Surges / Collapses)
4. Regional Temperature Shocks (Heatwaves)
5. Route / Corridor Closures
6. Combined Multi-Vector Disruptions
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from src.models import SupplyChainState


class Disruption(ABC):
    """Abstract base class for supply-chain disruptions."""

    @abstractmethod
    def apply(self, state: SupplyChainState) -> SupplyChainState:
        """Applies disruption to state and returns the updated state."""
        pass

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of disruption."""
        pass


class TruckFailureDisruption(Disruption):
    """Simulates mechanical breakdown or transit failure of a truck on its corridor."""

    def __init__(self, truck_id: Optional[str] = None, target_route_id: Optional[str] = None):
        self.truck_id = truck_id
        self.target_route_id = target_route_id

    def apply(self, state: SupplyChainState) -> SupplyChainState:
        target_tid = self.truck_id
        if not target_tid:
            for tid, trk in state.trucks.items():
                if trk.available:
                    target_tid = tid
                    break

        if target_tid and target_tid in state.trucks:
            state.trucks[target_tid].available = False
            loc = state.trucks[target_tid].current_location

            # Disable the primary active route from this location (leaving alternatives open)
            disabled_route = None
            if self.target_route_id and self.target_route_id in state.routes:
                disabled_route = self.target_route_id
                state.routes[disabled_route].available = False
            else:
                for rid, r in state.routes.items():
                    if (r.source == loc or r.destination == loc) and r.available:
                        r.available = False
                        disabled_route = rid
                        break

            state.disruptions_applied.append({
                "type": "TRUCK_FAILURE",
                "target_id": target_tid,
                "affected_routes": [disabled_route] if disabled_route else [],
            })
        else:
            # Fallback: disable first available hub route
            first_avail = next((r for r in state.routes.values() if r.available and "S" in r.destination), None)
            if not first_avail:
                first_avail = next((r for r in state.routes.values() if r.available), None)
            if first_avail:
                first_avail.available = False
                state.disruptions_applied.append({
                    "type": "TRUCK_FAILURE",
                    "target_id": first_avail.route_id,
                    "affected_routes": [first_avail.route_id],
                })
        return state

    def describe(self) -> str:
        return f"Truck Failure: Vehicle {self.truck_id or 'T1'} broke down in transit, blocking primary corridor."


class StorageOutageDisruption(Disruption):
    """Simulates cooling failure or complete operational shutdown of a cold storage facility."""

    def __init__(self, storage_id: Optional[str] = None):
        self.storage_id = storage_id

    def apply(self, state: SupplyChainState) -> SupplyChainState:
        target_id = self.storage_id
        if not target_id:
            target_id = next(
                (sid for sid, s in state.storage_facilities.items() if s.available), None
            )

        if target_id and target_id in state.storage_facilities:
            st = state.storage_facilities[target_id]
            st.available = False
            st.current_load = 0.0  # Cannot receive or store inventory
            # All routes into and out of this storage facility become unavailable
            closed_routes = []
            for r_id, r in state.routes.items():
                if (r.source == target_id or r.destination == target_id) and r.available:
                    r.available = False
                    closed_routes.append(r_id)

            state.disruptions_applied.append({
                "type": "STORAGE_OUTAGE",
                "target_id": target_id,
                "closed_routes": closed_routes,
            })
        return state

    def describe(self) -> str:
        return f"Cold Storage Outage: Facility {self.storage_id or 'Auto-selected'} suffered total power failure."


class DemandShockDisruption(Disruption):
    """Simulates sudden market demand shift (+30% surge or -30% drop)."""

    def __init__(self, market_id: Optional[str] = None, multiplier: float = 1.35):
        self.market_id = market_id
        self.multiplier = float(multiplier)

    def apply(self, state: SupplyChainState) -> SupplyChainState:
        target_id = self.market_id
        affected = []
        if target_id and target_id in state.markets:
            m = state.markets[target_id]
            m.demand = round(m.demand * self.multiplier, 1)
            affected.append(target_id)
        else:
            for mid, m in state.markets.items():
                m.demand = round(m.demand * self.multiplier, 1)
                affected.append(mid)

        state.disruptions_applied.append({
            "type": "DEMAND_SHOCK",
            "multiplier": self.multiplier,
            "affected_markets": affected,
        })
        return state

    def describe(self) -> str:
        pct = int((self.multiplier - 1.0) * 100)
        sign = "+" if pct > 0 else ""
        return f"Demand Shock: Market demand shifted by {sign}{pct}%."


class TemperatureShockDisruption(Disruption):
    """Simulates acute heatwave elevating temperatures across ambient and uncooled stages."""

    def __init__(self, delta_temp_c: float = 8.0):
        self.delta_temp_c = float(delta_temp_c)

    def apply(self, state: SupplyChainState) -> SupplyChainState:
        for b_id, b in state.batches.items():
            b.temperature = round(b.temperature + self.delta_temp_c, 1)
        for r_id, r in state.routes.items():
            r.risk_factor = min(1.0, round(r.risk_factor + 0.15, 2))

        state.disruptions_applied.append({
            "type": "TEMPERATURE_SHOCK",
            "delta_temp": self.delta_temp_c,
        })
        return state

    def describe(self) -> str:
        return f"Heatwave: Ambient temperatures elevated by +{self.delta_temp_c}°C across network."


class RouteClosureDisruption(Disruption):
    """Simulates roadway closure, severe congestion, or roadblock."""

    def __init__(self, route_id: Optional[str] = None):
        self.route_id = route_id

    def apply(self, state: SupplyChainState) -> SupplyChainState:
        target_id = self.route_id
        if not target_id:
            target_id = next(
                (rid for rid, r in state.routes.items() if r.available), None
            )

        if target_id and target_id in state.routes:
            state.routes[target_id].available = False
            state.disruptions_applied.append({
                "type": "ROUTE_CLOSURE",
                "route_id": target_id,
            })
        return state

    def describe(self) -> str:
        return f"Route Closure: Transport corridor {self.route_id or 'Auto-selected'} closed to traffic."


class CombinedDisruption(Disruption):
    """Simulates severe compounded multi-hazard disruption."""

    def __init__(self):
        self.disruptions = [
            StorageOutageDisruption(),
            TruckFailureDisruption(),
            TemperatureShockDisruption(delta_temp_c=6.0),
        ]

    def apply(self, state: SupplyChainState) -> SupplyChainState:
        for d in self.disruptions:
            state = d.apply(state)
        return state

    def describe(self) -> str:
        return "Compound Crisis: Simultaneous storage outage, truck breakdown, and regional heatwave."
