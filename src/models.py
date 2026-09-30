"""Data models for DPSRO supply-chain entities, states, and optimization decisions."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import json


@dataclass
class Farm:
    farm_id: str
    location: Tuple[float, float]  # (x, y) coordinates or (lat, lon)
    available_quantity: float
    harvest_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "farm_id": self.farm_id,
            "location_x": self.location[0],
            "location_y": self.location[1],
            "available_quantity": self.available_quantity,
            "harvest_time": self.harvest_time,
        }


@dataclass
class Batch:
    batch_id: str
    farm_id: str
    commodity: str
    quantity: float
    harvest_time: float
    initial_shelf_life_hours: float
    remaining_shelf_life_hours: float
    temperature: float  # Current storage/ambient temperature in °C
    quality_level: float = 1.0  # Normalized quality score 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StorageFacility:
    storage_id: str
    location: Tuple[float, float]
    capacity: float
    current_load: float = 0.0
    temperature_capacity: float = 4.0  # Operating temperature °C
    available: bool = True

    def available_capacity(self) -> float:
        if not self.available:
            return 0.0
        return max(0.0, self.capacity - self.current_load)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "storage_id": self.storage_id,
            "location_x": self.location[0],
            "location_y": self.location[1],
            "capacity": self.capacity,
            "current_load": self.current_load,
            "temperature_capacity": self.temperature_capacity,
            "available": self.available,
        }


@dataclass
class Market:
    market_id: str
    location: Tuple[float, float]
    demand: float
    priority: float = 1.0  # Multiplier for high-priority fulfillment

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market_id,
            "location_x": self.location[0],
            "location_y": self.location[1],
            "demand": self.demand,
            "priority": self.priority,
        }


@dataclass
class TransportRoute:
    route_id: str
    source: str
    destination: str
    distance_km: float
    travel_time_hours: float
    transport_cost_per_unit: float
    available: bool = True
    risk_factor: float = 0.1  # Inherent risk/congestion 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Truck:
    truck_id: str
    capacity: float
    available: bool = True
    current_location: str = ""
    travel_time_factor: float = 1.0  # Speed modifier (1.0 = normal, 1.5 = slow/traffic)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AllocationDecision:
    batch_id: str
    farm_id: str
    storage_id: Optional[str]  # None if routed direct farm -> market
    market_id: str
    quantity: float
    spoilage_fraction: float
    spoiled_qty: float
    delivered_qty: float
    transport_cost: float
    travel_time_hours: float
    risk_score: float
    route_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AllocationResult:
    method_name: str
    allocations: List[AllocationDecision]
    total_dispatched: float
    total_delivered: float
    total_spoilage: float
    spoilage_percentage: float
    demand_fulfillment_percentage: float
    unmet_demand: float
    total_transport_cost: float
    total_risk_score: float
    runtime_seconds: float
    solver_status: str
    is_feasible: bool
    objective_value: float = 0.0
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["allocations"] = [a.to_dict() for a in self.allocations]
        return data


@dataclass
class SupplyChainState:
    farms: Dict[str, Farm] = field(default_factory=dict)
    storage_facilities: Dict[str, StorageFacility] = field(default_factory=dict)
    markets: Dict[str, Market] = field(default_factory=dict)
    routes: Dict[str, TransportRoute] = field(default_factory=dict)
    batches: Dict[str, Batch] = field(default_factory=dict)
    trucks: Dict[str, Truck] = field(default_factory=dict)
    current_time_hours: float = 0.0
    disruptions_applied: List[Dict[str, Any]] = field(default_factory=list)

    def get_route(self, src: str, dst: str) -> Optional[TransportRoute]:
        for r in self.routes.values():
            if r.source == src and r.destination == dst:
                return r
        return None

    def clone(self) -> "SupplyChainState":
        """Deep copy state for what-if simulation and disruption evaluation."""
        import copy
        return copy.deepcopy(self)
