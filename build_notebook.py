import json
import os

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def md(text):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        }

    def code(text):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        }

    # Cell 0: Header
    nb["cells"].append(md("""# Dynamic Perishable Supply-Chain Resilience Optimizer (DPSRO)
## Complete Computational Intelligence Pipeline in a Single Standalone Notebook

This notebook consolidates the entire mathematical and computational engine:
1. **Domain Models**: Data structures for supply chain entities, batches, and allocation decisions
2. **Biophysical Spoilage Model**: Arrhenius-type kinetic deterioration & thermal acceleration
3. **Synthetic Generator**: Realistic spatial network topologies, batches, and routes
4. **Baseline Solvers**: Greedy Nearest-Market & Pure Cheapest-Route Linear Program
5. **Proposed DPSRO Engine**: Multi-objective perishability-aware LP using SciPy HiGHS
6. **Disruption Engine**: Dynamic corridor failures, hub outages, and demand shocks
7. **Dynamic Stateful Re-Optimizer**: Residual graph reformulation, elapsed shelf-life aging, and flow merging
8. **Interactive Visualizations**: Network topology, flow allocations, Sankey diagrams, and benchmark charts"""))

    # Cell 1: Imports
    nb["cells"].append(md("### 1. Dependencies and Environment Setup"))
    nb["cells"].append(code("""import math
import time
import copy
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import plotly.graph_objects as go
import plotly.express as px

print("All dependencies successfully loaded.")"""))

    # Cell 2: Domain Models
    nb["cells"].append(md("### 2. Supply-Chain Entity and Allocation Models"))
    nb["cells"].append(code("""@dataclass
class Farm:
    farm_id: str
    location: Tuple[float, float]
    available_quantity: float
    harvest_time: float = 0.0

@dataclass
class Batch:
    batch_id: str
    farm_id: str
    commodity: str
    quantity: float
    harvest_time: float
    initial_shelf_life_hours: float
    remaining_shelf_life_hours: float
    temperature: float
    quality_level: float = 1.0

@dataclass
class StorageFacility:
    storage_id: str
    location: Tuple[float, float]
    capacity: float
    current_load: float = 0.0
    temperature_capacity: float = 4.0
    available: bool = True

    def available_capacity(self) -> float:
        if not self.available:
            return 0.0
        return max(0.0, self.capacity - self.current_load)

@dataclass
class Market:
    market_id: str
    location: Tuple[float, float]
    demand: float
    priority: float = 1.0

@dataclass
class TransportRoute:
    route_id: str
    source: str
    destination: str
    distance_km: float
    travel_time_hours: float
    transport_cost_per_unit: float
    available: bool = True
    risk_factor: float = 0.1

@dataclass
class Truck:
    truck_id: str
    capacity: float
    available: bool = True
    current_location: str = ""
    travel_time_factor: float = 1.0

@dataclass
class AllocationDecision:
    batch_id: str
    farm_id: str
    storage_id: Optional[str]
    market_id: str
    quantity: float
    spoilage_fraction: float
    spoiled_qty: float
    delivered_qty: float
    transport_cost: float
    travel_time_hours: float
    risk_score: float
    route_ids: List[str] = field(default_factory=list)

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
        return copy.deepcopy(self)"""))

    # Cell 3: Spoilage Kinetics
    nb["cells"].append(md("### 3. Biophysical Spoilage Kinetics Engine"))
    nb["cells"].append(code("""class SpoilageModel:
    \"\"\"Arrhenius-inspired biophysical shelf life and spoilage kinetics model.\"\"\"

    def __init__(
        self,
        alpha: float = 2.5,
        beta: float = 0.8,
        gamma: float = 0.5,
        reference_temperature: float = 4.0,
        spoilage_threshold: float = 0.85,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.reference_temperature = float(reference_temperature)
        self.spoilage_threshold = float(spoilage_threshold)

    def calculate_temperature_penalty(self, actual_temp: float) -> float:
        temp_diff = actual_temp - self.reference_temperature
        if temp_diff <= 0.0:
            return 0.0
        return temp_diff / max(1.0, self.reference_temperature)

    def calculate_spoilage_risk(
        self,
        batch: Batch,
        transit_time_hours: float,
        route_temp: Optional[float] = None,
        delay_hours: float = 0.0,
    ) -> float:
        effective_temp = route_temp if route_temp is not None else batch.temperature
        temp_penalty = self.calculate_temperature_penalty(effective_temp)
        remaining_shelf = max(0.01, batch.remaining_shelf_life_hours)
        total_time = transit_time_hours + delay_hours
        time_ratio = total_time / remaining_shelf
        delay_factor = delay_hours / (transit_time_hours + 1.0) if transit_time_hours > 0 else 0.0

        logit = self.alpha * (time_ratio - 0.7) + self.beta * temp_penalty + self.gamma * delay_factor
        if logit >= 20.0:
            return 1.0
        elif logit <= -20.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-logit))

    def calculate_spoilage_fraction(
        self,
        batch: Batch,
        transit_time_hours: float,
        route_temp: Optional[float] = None,
        delay_hours: float = 0.0,
    ) -> float:
        remaining_shelf = batch.remaining_shelf_life_hours
        total_time = transit_time_hours + delay_hours
        if total_time >= remaining_shelf:
            return 1.0

        effective_temp = route_temp if route_temp is not None else batch.temperature
        temp_penalty = self.calculate_temperature_penalty(effective_temp)
        effective_elapsed = total_time * (1.0 + self.beta * temp_penalty)

        if effective_elapsed >= remaining_shelf:
            return 1.0
        fraction = (effective_elapsed / remaining_shelf) ** 1.8
        return min(1.0, max(0.0, fraction))

    def update_batch_shelf_life(
        self,
        batch: Batch,
        elapsed_hours: float,
        storage_temp: Optional[float] = None,
    ) -> Batch:
        effective_temp = storage_temp if storage_temp is not None else batch.temperature
        temp_penalty = self.calculate_temperature_penalty(effective_temp)
        shelf_loss = elapsed_hours * (1.0 + self.beta * temp_penalty)
        new_remaining = max(0.0, batch.remaining_shelf_life_hours - shelf_loss)
        new_quality = max(0.0, batch.quality_level * (new_remaining / max(0.1, batch.initial_shelf_life_hours)))

        return Batch(
            batch_id=batch.batch_id,
            farm_id=batch.farm_id,
            commodity=batch.commodity,
            quantity=batch.quantity,
            harvest_time=batch.harvest_time,
            initial_shelf_life_hours=batch.initial_shelf_life_hours,
            remaining_shelf_life_hours=new_remaining,
            temperature=effective_temp,
            quality_level=new_quality,
        )"""))

    # Cell 4: Synthetic Data Generator
    nb["cells"].append(md("### 4. Synthetic Network & Perishable Batch Generator"))
    nb["cells"].append(code("""COMMODITY_PROFILES = {
    "Berries": {"shelf_life_min": 24.0, "shelf_life_max": 48.0, "ideal_temp": 2.0},
    "Leafy Greens": {"shelf_life_min": 36.0, "shelf_life_max": 72.0, "ideal_temp": 4.0},
    "Tomatoes": {"shelf_life_min": 72.0, "shelf_life_max": 144.0, "ideal_temp": 12.0},
    "Peaches": {"shelf_life_min": 48.0, "shelf_life_max": 96.0, "ideal_temp": 4.0},
    "Apples": {"shelf_life_min": 168.0, "shelf_life_max": 336.0, "ideal_temp": 4.0},
}

def euclidean_distance(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
    return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

def generate_supply_chain(
    num_farms: int = 3,
    num_storage: int = 2,
    num_markets: int = 3,
    num_batches: int = 10,
    num_trucks: int = 5,
    seed: int = 42,
    area_size_km: float = 300.0,
    avg_speed_kmh: float = 50.0,
) -> SupplyChainState:
    rng = random.Random(seed)
    farms: Dict[str, Farm] = {}
    for i in range(num_farms):
        fid = f"F{i+1}"
        loc = (rng.uniform(10.0, area_size_km * 0.35), rng.uniform(20.0, area_size_km * 0.9))
        farms[fid] = Farm(farm_id=fid, location=loc, available_quantity=0.0)

    storages: Dict[str, StorageFacility] = {}
    for j in range(num_storage):
        sid = f"S{j+1}"
        loc = (rng.uniform(area_size_km * 0.4, area_size_km * 0.65), rng.uniform(area_size_km * 0.25, area_size_km * 0.75))
        storages[sid] = StorageFacility(
            storage_id=sid,
            location=loc,
            capacity=round(rng.uniform(8000.0, 20000.0), -2),
            temperature_capacity=4.0,
        )

    markets: Dict[str, Market] = {}
    total_est_supply = num_batches * 1200.0
    demand_per_market = round(total_est_supply / max(1, num_markets), -1)
    for k in range(num_markets):
        mid = f"M{k+1}"
        loc = (rng.uniform(area_size_km * 0.7, area_size_km * 0.95), rng.uniform(20.0, area_size_km * 0.9))
        m_demand = max(500.0, round(rng.uniform(0.7, 1.3) * demand_per_market, -1))
        markets[mid] = Market(market_id=mid, location=loc, demand=m_demand, priority=round(rng.choice([1.0, 1.2, 1.5]), 1))

    batches: Dict[str, Batch] = {}
    farm_ids = list(farms.keys())
    commodities = list(COMMODITY_PROFILES.keys())

    for b in range(num_batches):
        bid = f"B{b+1:02d}"
        fid = farm_ids[b % len(farm_ids)]
        comm = rng.choice(commodities)
        prof = COMMODITY_PROFILES[comm]
        initial_shelf = rng.uniform(prof["shelf_life_min"], prof["shelf_life_max"])
        age_hours = rng.uniform(0.0, initial_shelf * 0.4)
        rem_shelf = max(4.0, initial_shelf - age_hours)
        qty = round(rng.uniform(500.0, 2500.0), -1)
        ambient_temp = round(rng.uniform(18.0, 30.0), 1)

        batches[bid] = Batch(
            batch_id=bid,
            farm_id=fid,
            commodity=comm,
            quantity=qty,
            harvest_time=round(age_hours, 1),
            initial_shelf_life_hours=round(initial_shelf, 1),
            remaining_shelf_life_hours=round(rem_shelf, 1),
            temperature=ambient_temp,
            quality_level=round(rem_shelf / initial_shelf, 2),
        )
        farms[fid].available_quantity += qty

    routes: Dict[str, TransportRoute] = {}
    route_counter = 1

    # Farm -> Storage routes
    for fid, f in farms.items():
        for sid, s in storages.items():
            dist = round(euclidean_distance(f.location, s.location), 1)
            time_h = round(dist / avg_speed_kmh, 1)
            cost_u = round(0.015 * dist + rng.uniform(0.5, 1.5), 2)
            rid = f"R{route_counter:03d}"
            routes[rid] = TransportRoute(route_id=rid, source=fid, destination=sid, distance_km=dist, travel_time_hours=time_h, transport_cost_per_unit=cost_u, risk_factor=round(rng.uniform(0.05, 0.2), 2))
            route_counter += 1

    # Storage -> Market routes
    for sid, s in storages.items():
        for mid, m in markets.items():
            dist = round(euclidean_distance(s.location, m.location), 1)
            time_h = round(dist / avg_speed_kmh, 1)
            cost_u = round(0.015 * dist + rng.uniform(0.5, 1.5), 2)
            rid = f"R{route_counter:03d}"
            routes[rid] = TransportRoute(route_id=rid, source=sid, destination=mid, distance_km=dist, travel_time_hours=time_h, transport_cost_per_unit=cost_u, risk_factor=round(rng.uniform(0.05, 0.2), 2))
            route_counter += 1

    # Direct Farm -> Market express routes
    for fid, f in farms.items():
        for mid, m in markets.items():
            dist = round(euclidean_distance(f.location, m.location), 1)
            time_h = round(dist / avg_speed_kmh, 1)
            cost_u = round(0.022 * dist + rng.uniform(1.5, 3.0), 2)
            rid = f"R{route_counter:03d}"
            routes[rid] = TransportRoute(route_id=rid, source=fid, destination=mid, distance_km=dist, travel_time_hours=time_h, transport_cost_per_unit=cost_u, risk_factor=round(rng.uniform(0.1, 0.35), 2))
            route_counter += 1

    trucks: Dict[str, Truck] = {}
    for t in range(num_trucks):
        tid = f"T{t+1}"
        trucks[tid] = Truck(truck_id=tid, capacity=5000.0, available=True, current_location=farm_ids[t % len(farm_ids)])

    return SupplyChainState(
        farms=farms,
        storage_facilities=storages,
        markets=markets,
        routes=routes,
        batches=batches,
        trucks=trucks,
    )"""))

    # Cell 5: Baseline 1 - Nearest Market
    nb["cells"].append(md("### 5. Baseline 1: Nearest-Market Greedy Allocation"))
    nb["cells"].append(code("""def solve_nearest_market(
    state: SupplyChainState,
    spoilage_model: SpoilageModel,
) -> AllocationResult:
    start_time = time.perf_counter()
    remaining_demand = {m_id: m.demand for m_id, m in state.markets.items()}
    remaining_storage_cap = {s_id: s.available_capacity() for s_id, s in state.storage_facilities.items()}

    allocations: List[AllocationDecision] = []
    total_dispatched = total_delivered = total_spoilage = total_cost = total_risk = 0.0

    batch_list = sorted(state.batches.values(), key=lambda b: b.harvest_time)

    for batch in batch_list:
        batch_qty_left = batch.quantity
        if batch_qty_left <= 0:
            continue
        farm = state.farms.get(batch.farm_id)
        if not farm:
            continue

        candidate_paths = []
        for m_id, market in state.markets.items():
            if remaining_demand[m_id] <= 0:
                continue
            dir_route = state.get_route(farm.farm_id, m_id)
            if dir_route and dir_route.available:
                candidate_paths.append(("direct", m_id, None, dir_route.distance_km, dir_route.travel_time_hours, dir_route.transport_cost_per_unit, dir_route.risk_factor, [dir_route.route_id]))

        for s_id, storage in state.storage_facilities.items():
            if not storage.available or remaining_storage_cap[s_id] <= 0:
                continue
            r1 = state.get_route(farm.farm_id, s_id)
            if not r1 or not r1.available:
                continue
            for m_id, market in state.markets.items():
                if remaining_demand[m_id] <= 0:
                    continue
                r2 = state.get_route(s_id, m_id)
                if not r2 or not r2.available:
                    continue
                candidate_paths.append(("hub", m_id, s_id, r1.distance_km + r2.distance_km, r1.travel_time_hours + r2.travel_time_hours, r1.transport_cost_per_unit + r2.transport_cost_per_unit, (r1.risk_factor + r2.risk_factor)/2.0, [r1.route_id, r2.route_id]))

        candidate_paths.sort(key=lambda p: p[3])

        for p_type, m_id, s_id, dist, travel_t, unit_cost, risk_val, r_ids in candidate_paths:
            if batch_qty_left <= 0:
                break
            if remaining_demand[m_id] <= 0:
                continue

            max_possible = min(batch_qty_left, remaining_demand[m_id])
            if s_id:
                max_possible = min(max_possible, remaining_storage_cap[s_id])
            if max_possible <= 1e-4:
                continue

            alloc_qty = max_possible
            batch_qty_left -= alloc_qty
            remaining_demand[m_id] -= alloc_qty
            if s_id:
                remaining_storage_cap[s_id] -= alloc_qty

            spoil_fraction = spoilage_model.calculate_spoilage_fraction(batch, travel_t, route_temp=4.0 if s_id else batch.temperature)
            spoiled_qty = alloc_qty * spoil_fraction
            delivered_qty = alloc_qty - spoiled_qty
            route_cost = alloc_qty * unit_cost
            risk_score = alloc_qty * risk_val

            allocations.append(AllocationDecision(
                batch_id=batch.batch_id,
                farm_id=farm.farm_id,
                storage_id=s_id,
                market_id=m_id,
                quantity=alloc_qty,
                spoilage_fraction=spoil_fraction,
                spoiled_qty=spoiled_qty,
                delivered_qty=delivered_qty,
                transport_cost=route_cost,
                travel_time_hours=travel_t,
                risk_score=risk_score,
                route_ids=r_ids,
            ))
            total_dispatched += alloc_qty
            total_delivered += delivered_qty
            total_spoilage += spoiled_qty
            total_cost += route_cost
            total_risk += risk_score

    runtime = time.perf_counter() - start_time
    total_demand = sum(m.demand for m in state.markets.values())
    unmet = max(0.0, total_demand - total_delivered)
    spoil_pct = (total_spoilage / total_dispatched * 100.0) if total_dispatched > 0 else 0.0
    fulf_pct = (total_delivered / total_demand * 100.0) if total_demand > 0 else 0.0

    return AllocationResult(
        method_name="Nearest Market Baseline",
        allocations=allocations,
        total_dispatched=round(total_dispatched, 1),
        total_delivered=round(total_delivered, 1),
        total_spoilage=round(total_spoilage, 1),
        spoilage_percentage=round(spoil_pct, 2),
        demand_fulfillment_percentage=round(fulf_pct, 2),
        unmet_demand=round(unmet, 1),
        total_transport_cost=round(total_cost, 2),
        total_risk_score=round(total_risk, 1),
        runtime_seconds=round(runtime, 5),
        solver_status="OPTIMAL",
        is_feasible=True,
    )"""))

    # Cell 6: Baseline 2 - Cheapest Route
    nb["cells"].append(md("### 6. Baseline 2: Cheapest-Route Linear Program"))
    nb["cells"].append(code("""def solve_cheapest_route(
    state: SupplyChainState,
    spoilage_model: SpoilageModel,
) -> AllocationResult:
    start_time = time.perf_counter()
    candidate_paths = []

    for b_id, batch in state.batches.items():
        farm = state.farms.get(batch.farm_id)
        if not farm or batch.quantity <= 0:
            continue
        for m_id, market in state.markets.items():
            r_dir = state.get_route(farm.farm_id, m_id)
            if r_dir and r_dir.available:
                candidate_paths.append({
                    "batch_id": b_id, "farm_id": farm.farm_id, "storage_id": None, "market_id": m_id,
                    "travel_time": r_dir.travel_time_hours, "unit_cost": r_dir.transport_cost_per_unit,
                    "risk_factor": r_dir.risk_factor, "route_ids": [r_dir.route_id],
                })
        for s_id, storage in state.storage_facilities.items():
            if not storage.available or storage.available_capacity() <= 0:
                continue
            r1 = state.get_route(farm.farm_id, s_id)
            if not r1 or not r1.available:
                continue
            for m_id, market in state.markets.items():
                r2 = state.get_route(s_id, m_id)
                if not r2 or not r2.available:
                    continue
                candidate_paths.append({
                    "batch_id": b_id, "farm_id": farm.farm_id, "storage_id": s_id, "market_id": m_id,
                    "travel_time": r1.travel_time_hours + r2.travel_time_hours,
                    "unit_cost": r1.transport_cost_per_unit + r2.transport_cost_per_unit,
                    "risk_factor": (r1.risk_factor + r2.risk_factor)/2.0,
                    "route_ids": [r1.route_id, r2.route_id],
                })

    num_paths = len(candidate_paths)
    market_ids = list(state.markets.keys())
    batch_ids = list(state.batches.keys())
    storage_ids = list(state.storage_facilities.keys())
    market_idx = {m_id: i for i, m_id in enumerate(market_ids)}
    batch_idx = {b_id: i for i, b_id in enumerate(batch_ids)}
    storage_idx = {s_id: i for i, s_id in enumerate(storage_ids)}

    total_vars = num_paths + len(market_ids)
    if num_paths == 0:
        return AllocationResult("Cheapest Route Baseline", [], 0.0, 0.0, 0.0, 0.0, 0.0, sum(m.demand for m in state.markets.values()), 0.0, 0.0, round(time.perf_counter()-start_time, 5), "NO_PATHS", False)

    c = np.zeros(total_vars)
    for p_i, p in enumerate(candidate_paths):
        c[p_i] = p["unit_cost"]
    for m_i in range(len(market_ids)):
        c[num_paths + m_i] = 50.0

    num_ub = len(batch_ids) + len(storage_ids)
    A_ub = np.zeros((num_ub, total_vars))
    b_ub = np.zeros(num_ub)
    for b_i, b_id in enumerate(batch_ids):
        b_ub[b_i] = state.batches[b_id].quantity
    for s_i, s_id in enumerate(storage_ids):
        b_ub[len(batch_ids) + s_i] = state.storage_facilities[s_id].available_capacity()

    for p_i, p in enumerate(candidate_paths):
        A_ub[batch_idx[p["batch_id"]], p_i] = 1.0
        if p["storage_id"]:
            A_ub[len(batch_ids) + storage_idx[p["storage_id"]], p_i] = 1.0

    A_eq = np.zeros((len(market_ids), total_vars))
    b_eq = np.zeros(len(market_ids))
    for m_i, m_id in enumerate(market_ids):
        b_eq[m_i] = state.markets[m_id].demand
        A_eq[m_i, num_paths + m_i] = 1.0

    for p_i, p in enumerate(candidate_paths):
        A_eq[market_idx[p["market_id"]], p_i] = 1.0

    bounds = [(0.0, None) for _ in range(total_vars)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    runtime = time.perf_counter() - start_time

    allocations = []
    total_dispatched = total_delivered = total_spoilage = total_cost = total_risk = 0.0

    if res.success:
        x_vals = res.x[:num_paths]
        for p_i, flow in enumerate(x_vals):
            if flow < 1e-4:
                continue
            p = candidate_paths[p_i]
            batch = state.batches[p["batch_id"]]
            spoil_frac = spoilage_model.calculate_spoilage_fraction(batch, p["travel_time"], route_temp=4.0 if p["storage_id"] else batch.temperature)
            spoiled_qty = flow * spoil_frac
            deliv_qty = flow - spoiled_qty
            cost = flow * p["unit_cost"]
            risk = flow * p["risk_factor"]

            allocations.append(AllocationDecision(
                p["batch_id"], p["farm_id"], p["storage_id"], p["market_id"],
                round(flow, 2), spoil_frac, round(spoiled_qty, 2), round(deliv_qty, 2),
                round(cost, 2), p["travel_time"], round(risk, 2), p["route_ids"]
            ))
            total_dispatched += flow
            total_delivered += deliv_qty
            total_spoilage += spoiled_qty
            total_cost += cost
            total_risk += risk

    total_demand = sum(m.demand for m in state.markets.values())
    unmet = max(0.0, total_demand - total_delivered)
    spoil_pct = (total_spoilage / total_dispatched * 100.0) if total_dispatched > 0 else 0.0
    fulf_pct = (total_delivered / total_demand * 100.0) if total_demand > 0 else 0.0

    return AllocationResult(
        method_name="Cheapest Route Baseline",
        allocations=allocations,
        total_dispatched=round(total_dispatched, 1),
        total_delivered=round(total_delivered, 1),
        total_spoilage=round(total_spoilage, 1),
        spoilage_percentage=round(spoil_pct, 2),
        demand_fulfillment_percentage=round(fulf_pct, 2),
        unmet_demand=round(unmet, 1),
        total_transport_cost=round(total_cost, 2),
        total_risk_score=round(total_risk, 1),
        runtime_seconds=round(runtime, 5),
        solver_status=res.message if hasattr(res, "message") else "COMPLETED",
        is_feasible=res.success,
        objective_value=float(res.fun) if res.success else 0.0,
    )"""))

    # Cell 7: Proposed DPSRO Optimizer
    nb["cells"].append(md("### 7. Proposed Method: Dynamic Perishability-Aware Optimizer (DPSRO)"))
    nb["cells"].append(code("""class DynamicPerishableOptimizer:
    \"\"\"Exact multi-objective mathematical program using SciPy HiGHS.\"\"\"

    def __init__(
        self,
        spoilage_model: SpoilageModel,
        lambda_spoilage: float = 10.0,
        lambda_cost: float = 0.15,
        lambda_unmet_demand: float = 15.0,
        lambda_risk: float = 5.0,
    ):
        self.spoilage_model = spoilage_model
        self.lambda_spoilage = float(lambda_spoilage)
        self.lambda_cost = float(lambda_cost)
        self.lambda_unmet_demand = float(lambda_unmet_demand)
        self.lambda_risk = float(lambda_risk)

    def solve(self, state: SupplyChainState, allow_expired_routing: bool = False) -> AllocationResult:
        start_time = time.perf_counter()
        candidate_paths = []

        for b_id, batch in state.batches.items():
            if batch.quantity <= 0:
                continue
            farm = state.farms.get(batch.farm_id)
            if not farm:
                continue

            for m_id, market in state.markets.items():
                r_dir = state.get_route(farm.farm_id, m_id)
                if not r_dir or not r_dir.available:
                    continue
                if not allow_expired_routing and r_dir.travel_time_hours > batch.remaining_shelf_life_hours:
                    continue

                spoil_frac = self.spoilage_model.calculate_spoilage_fraction(batch, r_dir.travel_time_hours, batch.temperature)
                spoil_risk = self.spoilage_model.calculate_spoilage_risk(batch, r_dir.travel_time_hours, batch.temperature)
                candidate_paths.append({
                    "batch_id": b_id, "farm_id": farm.farm_id, "storage_id": None, "market_id": m_id,
                    "travel_time": r_dir.travel_time_hours, "unit_cost": r_dir.transport_cost_per_unit,
                    "spoil_frac": spoil_frac, "risk_factor": (r_dir.risk_factor + spoil_risk) / 2.0,
                    "route_ids": [r_dir.route_id],
                })

            for s_id, storage in state.storage_facilities.items():
                if not storage.available or storage.available_capacity() <= 0:
                    continue
                r1 = state.get_route(farm.farm_id, s_id)
                if not r1 or not r1.available:
                    continue
                for m_id, market in state.markets.items():
                    r2 = state.get_route(s_id, m_id)
                    if not r2 or not r2.available:
                        continue
                    total_t = r1.travel_time_hours + r2.travel_time_hours
                    if not allow_expired_routing and total_t > batch.remaining_shelf_life_hours:
                        continue
                    spoil_frac = self.spoilage_model.calculate_spoilage_fraction(batch, total_t, storage.temperature_capacity)
                    spoil_risk = self.spoilage_model.calculate_spoilage_risk(batch, total_t, storage.temperature_capacity)
                    candidate_paths.append({
                        "batch_id": b_id, "farm_id": farm.farm_id, "storage_id": s_id, "market_id": m_id,
                        "travel_time": total_t, "unit_cost": r1.transport_cost_per_unit + r2.transport_cost_per_unit,
                        "spoil_frac": spoil_frac, "risk_factor": (((r1.risk_factor + r2.risk_factor)/2.0) + spoil_risk) / 2.0,
                        "route_ids": [r1.route_id, r2.route_id],
                    })

        num_paths = len(candidate_paths)
        market_ids = list(state.markets.keys())
        batch_ids = list(state.batches.keys())
        storage_ids = list(state.storage_facilities.keys())
        market_idx = {m_id: i for i, m_id in enumerate(market_ids)}
        batch_idx = {b_id: i for i, b_id in enumerate(batch_ids)}
        storage_idx = {s_id: i for i, s_id in enumerate(storage_ids)}

        total_vars = num_paths + len(market_ids)
        if num_paths == 0:
            return AllocationResult("Dynamic Perishability Optimizer (DPSRO)", [], 0.0, 0.0, 0.0, 0.0, 0.0, sum(m.demand for m in state.markets.values()), 0.0, 0.0, round(time.perf_counter()-start_time, 5), "NO_PATHS", False)

        c = np.zeros(total_vars)
        for p_i, p in enumerate(candidate_paths):
            c[p_i] = (
                self.lambda_spoilage * p["spoil_frac"]
                + self.lambda_cost * p["unit_cost"]
                + self.lambda_risk * p["risk_factor"]
            )
        for m_i, m_id in enumerate(market_ids):
            c[num_paths + m_i] = self.lambda_unmet_demand * state.markets[m_id].priority

        num_ub = len(batch_ids) + len(storage_ids)
        A_ub = np.zeros((num_ub, total_vars))
        b_ub = np.zeros(num_ub)
        for b_i, b_id in enumerate(batch_ids):
            b_ub[b_i] = state.batches[b_id].quantity
        for s_i, s_id in enumerate(storage_ids):
            b_ub[len(batch_ids) + s_i] = state.storage_facilities[s_id].available_capacity()

        for p_i, p in enumerate(candidate_paths):
            A_ub[batch_idx[p["batch_id"]], p_i] = 1.0
            if p["storage_id"]:
                A_ub[len(batch_ids) + storage_idx[p["storage_id"]], p_i] = 1.0

        A_eq = np.zeros((len(market_ids), total_vars))
        b_eq = np.zeros(len(market_ids))
        for m_i, m_id in enumerate(market_ids):
            b_eq[m_i] = state.markets[m_id].demand
            A_eq[m_i, num_paths + m_i] = 1.0

        for p_i, p in enumerate(candidate_paths):
            A_eq[market_idx[p["market_id"]], p_i] = (1.0 - p["spoil_frac"])

        bounds = [(0.0, None) for _ in range(total_vars)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        runtime = time.perf_counter() - start_time

        allocations = []
        total_dispatched = total_delivered = total_spoilage = total_cost = total_risk = 0.0

        if res.success:
            x_vals = res.x[:num_paths]
            for p_i, flow in enumerate(x_vals):
                if flow < 1e-4:
                    continue
                p = candidate_paths[p_i]
                spoiled_qty = flow * p["spoil_frac"]
                deliv_qty = flow - spoiled_qty
                cost = flow * p["unit_cost"]
                risk = flow * p["risk_factor"]

                allocations.append(AllocationDecision(
                    p["batch_id"], p["farm_id"], p["storage_id"], p["market_id"],
                    round(flow, 2), p["spoil_frac"], round(spoiled_qty, 2), round(deliv_qty, 2),
                    round(cost, 2), p["travel_time"], round(risk, 2), p["route_ids"]
                ))
                total_dispatched += flow
                total_delivered += deliv_qty
                total_spoilage += spoiled_qty
                total_cost += cost
                total_risk += risk

        total_demand = sum(m.demand for m in state.markets.values())
        unmet = max(0.0, total_demand - total_delivered)
        spoil_pct = (total_spoilage / total_dispatched * 100.0) if total_dispatched > 0 else 0.0
        fulf_pct = (total_delivered / total_demand * 100.0) if total_demand > 0 else 0.0

        return AllocationResult(
            method_name="Dynamic Perishability Optimizer (DPSRO)",
            allocations=allocations,
            total_dispatched=round(total_dispatched, 1),
            total_delivered=round(total_delivered, 1),
            total_spoilage=round(total_spoilage, 1),
            spoilage_percentage=round(spoil_pct, 2),
            demand_fulfillment_percentage=round(fulf_pct, 2),
            unmet_demand=round(unmet, 1),
            total_transport_cost=round(total_cost, 2),
            total_risk_score=round(total_risk, 1),
            runtime_seconds=round(runtime, 5),
            solver_status=res.message if hasattr(res, "message") else "COMPLETED",
            is_feasible=res.success,
            objective_value=float(res.fun) if res.success else 0.0,
        )"""))

    # Cell 8: Disruption Engine
    nb["cells"].append(md("### 8. Disruption Engine"))
    nb["cells"].append(code("""class Disruption(ABC):
    @abstractmethod
    def apply(self, state: SupplyChainState) -> SupplyChainState:
        pass

    @abstractmethod
    def describe(self) -> str:
        pass

class TruckFailureDisruption(Disruption):
    def __init__(self, truck_id: Optional[str] = None):
        self.truck_id = truck_id

    def apply(self, state: SupplyChainState) -> SupplyChainState:
        target_tid = self.truck_id or next((tid for tid, t in state.trucks.items() if t.available), None)
        if target_tid and target_tid in state.trucks:
            state.trucks[target_tid].available = False
            loc = state.trucks[target_tid].current_location
            disabled_route = None
            for rid, r in state.routes.items():
                if (r.source == loc or r.destination == loc) and r.available:
                    r.available = False
                    disabled_route = rid
                    break
            state.disruptions_applied.append({"type": "TRUCK_FAILURE", "target_id": target_tid, "affected_routes": [disabled_route] if disabled_route else []})
        return state

    def describe(self) -> str:
        return f"Truck Failure: Vehicle {self.truck_id or 'T1'} breakdown in transit corridor."

class StorageOutageDisruption(Disruption):
    def __init__(self, storage_id: Optional[str] = None):
        self.storage_id = storage_id

    def apply(self, state: SupplyChainState) -> SupplyChainState:
        target_id = self.storage_id or next((sid for sid, s in state.storage_facilities.items() if s.available), None)
        if target_id and target_id in state.storage_facilities:
            st = state.storage_facilities[target_id]
            st.available = False
            st.current_load = 0.0
            closed = []
            for rid, r in state.routes.items():
                if (r.source == target_id or r.destination == target_id) and r.available:
                    r.available = False
                    closed.append(rid)
            state.disruptions_applied.append({"type": "STORAGE_OUTAGE", "target_id": target_id, "closed_routes": closed})
        return state

    def describe(self) -> str:
        return f"Cold Storage Outage: Facility {self.storage_id or 'Hub'} power failure."

class DemandShockDisruption(Disruption):
    def __init__(self, market_id: Optional[str] = None, multiplier: float = 1.35):
        self.market_id = market_id
        self.multiplier = float(multiplier)

    def apply(self, state: SupplyChainState) -> SupplyChainState:
        for mid, m in state.markets.items():
            if not self.market_id or mid == self.market_id:
                m.demand = round(m.demand * self.multiplier, 1)
        state.disruptions_applied.append({"type": "DEMAND_SHOCK", "multiplier": self.multiplier})
        return state

    def describe(self) -> str:
        return f"Demand Shock: Market consumption shifted by {self.multiplier}x." """))

    # Cell 9: Dynamic Stateful Re-Optimizer
    nb["cells"].append(md("### 9. Dynamic Stateful Re-Optimization Engine"))
    nb["cells"].append(code("""class DynamicReoptimizer:
    \"\"\"Stateful receding-horizon rerouting engine.\"\"\"

    def __init__(self, optimizer: DynamicPerishableOptimizer):
        self.optimizer = optimizer
        self.spoilage_model = optimizer.spoilage_model

    def reoptimize_after_disruption(
        self,
        post_disruption_state: SupplyChainState,
        prior_result: AllocationResult,
        elapsed_hours_during_disruption: float = 2.0,
    ) -> Tuple[AllocationResult, Dict[str, Any]]:
        recovery_start = time.perf_counter()
        preserved_allocations: List[AllocationDecision] = []
        stranded_batches_qty: Dict[str, float] = {}
        affected_batch_ids = set()

        for alloc in prior_result.allocations:
            is_valid = True
            for rid in alloc.route_ids:
                route = post_disruption_state.routes.get(rid)
                if not route or not route.available:
                    is_valid = False
            if alloc.storage_id:
                st = post_disruption_state.storage_facilities.get(alloc.storage_id)
                if not st or not st.available:
                    is_valid = False

            if is_valid:
                preserved_allocations.append(alloc)
            else:
                affected_batch_ids.add(alloc.batch_id)
                stranded_batches_qty[alloc.batch_id] = stranded_batches_qty.get(alloc.batch_id, 0.0) + alloc.quantity

        residual_state = post_disruption_state.clone()
        for bid, batch in residual_state.batches.items():
            aged_batch = self.spoilage_model.update_batch_shelf_life(batch, elapsed_hours_during_disruption, batch.temperature)
            aged_batch.quantity = stranded_batches_qty.get(bid, 0.0)
            residual_state.batches[bid] = aged_batch

        for alloc in preserved_allocations:
            if alloc.market_id in residual_state.markets:
                m = residual_state.markets[alloc.market_id]
                m.demand = max(0.0, m.demand - alloc.delivered_qty)
            if alloc.storage_id and alloc.storage_id in residual_state.storage_facilities:
                st = residual_state.storage_facilities[alloc.storage_id]
                st.current_load = min(st.capacity, st.current_load + alloc.quantity)

        residual_result = self.optimizer.solve(residual_state)
        recovery_time = time.perf_counter() - recovery_start

        merged_allocations = list(preserved_allocations) + list(residual_result.allocations)
        total_dispatched = sum(a.quantity for a in merged_allocations)
        total_delivered = sum(a.delivered_qty for a in merged_allocations)
        total_spoilage = sum(a.spoiled_qty for a in merged_allocations)
        total_cost = sum(a.transport_cost for a in merged_allocations)
        total_risk = sum(a.risk_score for a in merged_allocations)

        total_original_demand = sum(m.demand for m in post_disruption_state.markets.values())
        unmet_demand = max(0.0, total_original_demand - total_delivered)
        spoilage_pct = (total_spoilage / total_dispatched * 100.0) if total_dispatched > 0 else 0.0
        fulfillment_pct = (total_delivered / total_original_demand * 100.0) if total_original_demand > 0 else 0.0

        rerouted_qty = sum(a.quantity for a in residual_result.allocations)
        total_stranded_qty = sum(stranded_batches_qty.values())

        recovery_summary = {
            "recovery_time_seconds": round(recovery_time, 5),
            "affected_batch_count": len(affected_batch_ids),
            "affected_batches": list(affected_batch_ids),
            "stranded_quantity_kg": round(total_stranded_qty, 1),
            "successfully_rerouted_kg": round(rerouted_qty, 1),
            "reroute_success_rate_pct": round((rerouted_qty / total_stranded_qty * 100.0) if total_stranded_qty > 0 else 100.0, 2),
            "cost_delta": round(total_cost - prior_result.total_transport_cost, 2),
            "spoilage_delta_kg": round(total_spoilage - prior_result.total_spoilage, 1),
            "unmet_demand_delta_kg": round(unmet_demand - prior_result.unmet_demand, 1),
        }

        reoptimized_result = AllocationResult(
            method_name="Dynamic Re-Optimization (DPSRO)",
            allocations=merged_allocations,
            total_dispatched=round(total_dispatched, 1),
            total_delivered=round(total_delivered, 1),
            total_spoilage=round(total_spoilage, 1),
            spoilage_percentage=round(spoilage_pct, 2),
            demand_fulfillment_percentage=round(fulfillment_pct, 2),
            unmet_demand=round(unmet_demand, 1),
            total_transport_cost=round(total_cost, 2),
            total_risk_score=round(total_risk, 1),
            runtime_seconds=round(prior_result.runtime_seconds + recovery_time, 5),
            solver_status="OPTIMAL_RECOVERY",
            is_feasible=True,
            extra_metrics=recovery_summary,
        )
        return reoptimized_result, recovery_summary"""))

    # Cell 10: Comparative Metrics
    nb["cells"].append(md("### 10. Metrics & Evaluation Functions"))
    nb["cells"].append(code("""def calculate_comparative_metrics(b_res: AllocationResult, p_res: AllocationResult) -> Dict[str, Any]:
    b_spoil, p_spoil = b_res.total_spoilage, p_res.total_spoilage
    spoil_red = ((b_spoil - p_spoil) / b_spoil * 100.0) if b_spoil > 1e-4 else 0.0
    fulf_gain = p_res.demand_fulfillment_percentage - b_res.demand_fulfillment_percentage
    cost_diff = ((p_res.total_transport_cost - b_res.total_transport_cost) / b_res.total_transport_cost * 100.0) if b_res.total_transport_cost > 1e-4 else 0.0

    return {
        "baseline_method": b_res.method_name,
        "proposed_method": p_res.method_name,
        "spoilage_reduction_pct": round(spoil_red, 2),
        "fulfillment_gain_pts": round(fulf_gain, 2),
        "cost_change_pct": round(cost_diff, 2),
        "spoilage_saved_kg": round(b_spoil - p_spoil, 1),
        "extra_delivered_kg": round(p_res.total_delivered - b_res.total_delivered, 1),
    }"""))

    # Cell 11: Visualizations
    nb["cells"].append(md("### 11. Interactive Visualizations (Plotly)"))
    nb["cells"].append(code("""def plot_network_graph(state: SupplyChainState, result: Optional[AllocationResult] = None, title: str = "Supply-Chain Network & Active Allocations"):
    fig = go.Figure()
    route_flow = {}
    if result:
        for a in result.allocations:
            for rid in a.route_ids:
                route_flow[rid] = route_flow.get(rid, 0.0) + a.quantity

    max_f = max(route_flow.values()) if route_flow else 1.0

    for rid, route in state.routes.items():
        src_pos = state.farms[route.source].location if route.source in state.farms else state.storage_facilities[route.source].location
        dst_pos = state.storage_facilities[route.destination].location if route.destination in state.storage_facilities else state.markets[route.destination].location
        flow = route_flow.get(rid, 0.0)

        if not route.available:
            color, width, dash = "rgba(239, 68, 68, 0.85)", 3.0, "dash"
            h_text = f"ROUTE CLOSED: {rid}<br>{route.source} -> {route.destination}"
        elif flow > 0:
            color, width, dash = "rgba(16, 185, 129, 0.9)", max(2.5, min(8.0, 2.5 + (flow/max_f)*5.5)), "solid"
            h_text = f"Active Flow: {flow:,.1f} kg<br>{route.source} -> {route.destination}"
        else:
            color, width, dash = "rgba(148, 163, 184, 0.35)", 1.2, "dot"
            h_text = f"Idle: {rid} ({route.distance_km:.1f} km)"

        fig.add_trace(go.Scatter(x=[src_pos[0], dst_pos[0]], y=[src_pos[1], dst_pos[1]], mode="lines", line=dict(color=color, width=width, dash=dash), hoverinfo="text", text=h_text, showlegend=False))

    fig.add_trace(go.Scatter(x=[f.location[0] for f in state.farms.values()], y=[f.location[1] for f in state.farms.values()], mode="markers+text", marker=dict(symbol="triangle-up", size=18, color="#10B981"), text=list(state.farms.keys()), textposition="top center", name="Farms"))
    fig.add_trace(go.Scatter(x=[s.location[0] for s in state.storage_facilities.values()], y=[s.location[1] for s in state.storage_facilities.values()], mode="markers+text", marker=dict(symbol="square", size=20, color=["#EF4444" if not s.available else "#3B82F6" for s in state.storage_facilities.values()]), text=list(state.storage_facilities.keys()), textposition="top center", name="Cold Storages"))
    fig.add_trace(go.Scatter(x=[m.location[0] for m in state.markets.values()], y=[m.location[1] for m in state.markets.values()], mode="markers+text", marker=dict(symbol="circle", size=18, color="#F59E0B"), text=list(state.markets.keys()), textposition="top center", name="Markets"))

    fig.update_layout(title=title, xaxis_title="X Coordinates (km)", yaxis_title="Y Coordinates (km)", template="plotly_dark", height=500, margin=dict(l=40, r=40, t=50, b=40))
    return fig

def plot_comparative_barchart(results: List[AllocationResult]):
    data = []
    for r in results:
        data.extend([
            {"Method": r.method_name, "Metric": "Spoilage %", "Value": r.spoilage_percentage},
            {"Method": r.method_name, "Metric": "Demand Fulfilled %", "Value": r.demand_fulfillment_percentage},
        ])
    df = pd.DataFrame(data)
    fig = px.bar(df, x="Method", y="Value", color="Metric", barmode="group", title="Comparative Performance Benchmarks", color_discrete_map={"Spoilage %": "#EF4444", "Demand Fulfilled %": "#10B981"})
    fig.update_layout(template="plotly_dark", height=400)
    return fig"""))

    # Cell 12: End-to-End Pipeline Execution
    nb["cells"].append(md("### 12. Full Pipeline Execution: Pre-Disruption vs. Dynamic Post-Disruption Recovery"))
    nb["cells"].append(code("""# 1. Initialize Supply Chain Network (Farms, Storages, Markets, Perishable Batches)
print("=== STEP 1: GENERATING NETWORK INSTANCE ===")
state = generate_supply_chain(num_farms=3, num_storage=2, num_markets=3, num_batches=10, seed=42)
spoilage = SpoilageModel(alpha=2.5, beta=0.8, reference_temperature=4.0)
optimizer = DynamicPerishableOptimizer(spoilage_model=spoilage, lambda_spoilage=10.0, lambda_cost=0.15, lambda_unmet_demand=15.0, lambda_risk=5.0)

print(f"Network Assets: {len(state.farms)} Farms | {len(state.storage_facilities)} Cold Storages | {len(state.markets)} Markets")
print(f"Perishable Batches: {len(state.batches)} | Total Harvest: {sum(b.quantity for b in state.batches.values()):,.1f} kg")
print(f"Market Demand: {sum(m.demand for m in state.markets.values()):,.1f} kg")

# 2. Solve Baselines & Proposed Optimizer
print("\\n=== STEP 2: PRE-DISRUPTION BENCHMARK SOLVING ===")
res_nearest = solve_nearest_market(state, spoilage)
res_cheapest = solve_cheapest_route(state, spoilage)
res_proposed = optimizer.solve(state)

summary_df = pd.DataFrame([
    {"Method": r.method_name, "Dispatched (kg)": r.total_dispatched, "Delivered (kg)": r.total_delivered, "Spoiled (kg)": r.total_spoilage, "Spoilage %": f"{r.spoilage_percentage:.2f}%", "Fulfillment %": f"{r.demand_fulfillment_percentage:.2f}%", "Cost ($)": f"${r.total_transport_cost:,.2f}", "Runtime": f"{r.runtime_seconds:.5f}s"}
    for r in [res_nearest, res_cheapest, res_proposed]
])
display(summary_df)

comp_near = calculate_comparative_metrics(res_nearest, res_proposed)
comp_cheap = calculate_comparative_metrics(res_cheapest, res_proposed)
print(f"\\n>> DPSRO vs Nearest: Spoilage reduced by {comp_near['spoilage_reduction_pct']}% | Extra Delivered: {comp_near['extra_delivered_kg']} kg")
print(f">> DPSRO vs Cheapest: Spoilage reduced by {comp_cheap['spoilage_reduction_pct']}% | Extra Delivered: {comp_cheap['extra_delivered_kg']} kg")

# 3. Inject Disruption: Operational Truck Breakdown
print("\\n=== STEP 3: INJECTING DISRUPTION ===")
disrupted_state = state.clone()
disruption = TruckFailureDisruption()
disrupted_state = disruption.apply(disrupted_state)
d_info = disrupted_state.disruptions_applied[-1]
print(f"Disruption Event: {disruption.describe()}")
print(f"Corridors Closed: {d_info['affected_routes']}")

# 4. Dynamic Stateful Re-Optimization
print("\\n=== STEP 4: DYNAMIC STATEFUL RE-OPTIMIZATION ===")
reoptimizer = DynamicReoptimizer(optimizer=optimizer)
res_reopt, rec_stats = reoptimizer.reoptimize_after_disruption(
    post_disruption_state=disrupted_state,
    prior_result=res_proposed,
    elapsed_hours_during_disruption=2.5
)

print(f"Recovery Time: {rec_stats['recovery_time_seconds']}s")
print(f"Affected Batches: {rec_stats['affected_batch_count']} ({rec_stats['affected_batches']})")
print(f"Stranded Harvest: {rec_stats['stranded_quantity_kg']} kg | Successfully Rerouted: {rec_stats['successfully_rerouted_kg']} kg ({rec_stats['reroute_success_rate_pct']}%)")
print(f"Final Post-Disruption Delivered: {res_reopt.total_delivered:,.1f} kg (Spoilage: {res_reopt.spoilage_percentage:.2f}%)")

# 5. Display Interactive Charts
fig_net = plot_network_graph(disrupted_state, res_reopt, "Post-Disruption Dynamic Rerouting Flows")
fig_net.show()

fig_bar = plot_comparative_barchart([res_nearest, res_cheapest, res_proposed, res_reopt])
fig_bar.show()"""))

    # Cell 13: Novel CIS x EMBS Hybrid Orchestrator & Security Layer
    nb["cells"].append(md("""### 13. Novel Track Innovation: CIS × EMBS Evolutionary Hybrid Orchestrator & Security Layer

This section highlights the direct fusion between **Computational Intelligence (CIS)** and **Engineering in Medicine & Biology (EMBS)**:
- **Biomedical & Biophysical Kinetics (EMBS)**: Arrhenius reaction kinetics $k(T) = A \\cdot \\exp(-E_a / (R \\cdot T))$ with $Q_{10}=2.0$ thermal acceleration modeling enzymatic cellular degradation of produce.
- **Evolutionary Computation (CIS)**: A $(1+\\lambda)$-Evolution Strategy that tunes the multi-objective loss landscape weights $(\\lambda_{\\text{spoilage}}, \\lambda_{\\text{cost}}, \\lambda_{\\text{unmet}}, \\lambda_{\\text{risk}})$ across generations to map the Pareto frontier.
- **Input Sanitization & Security**: Production-grade parameter validation against malformed inputs, log injections, and numerical anomalies."""))

    nb["cells"].append(code("""from src.hybrid_orchestrator import HybridCISEMBSOrchestrator, ArrheniusEMBSModel
from src.security import validate_quantity, validate_shelf_life, validate_temperature, validate_node_id

print("=== STEP 5: NOVEL CIS × EMBS HYBRID EVOLUTIONARY TUNING ===")

# 1. Biophysical Arrhenius Kinetics (EMBS)
embs_model = ArrheniusEMBSModel(activation_energy_j_mol=75000.0, reference_temp_celsius=4.0, q10=2.0)
for temp in [4.0, 10.0, 14.0, 24.0]:
    rate = embs_model.relative_rate(temp)
    print(f"Biophysical degradation acceleration at {temp:4.1f}°C: {rate:5.2f}x reference rate")

# 2. Hybrid Evolutionary Orchestrator (CIS + EMBS)
orchestrator = HybridCISEMBSOrchestrator(
    embs_model=embs_model,
    generations=6,
    offspring_per_gen=3,
    mutation_std=0.25,
    random_seed=42,
)

print("\\nExecuting (1+λ)-ES Evolutionary Weight Optimization on Network State...")
hybrid_res, pareto_pts = orchestrator.solve_hybrid(state, pareto_samples=5)

print(f"Optimized Hybrid Spoilage: {hybrid_res.spoilage_percentage:.2f}% | Fulfillment: {hybrid_res.demand_fulfillment_percentage:.2f}%")
print(f"Evolution Generations: {len(orchestrator.evolution_log)} iterations logged.")

# 3. Pareto Frontier Overview
pareto_df = pd.DataFrame([
    {
        "Weights (Spoil/Cost/Unmet/Risk)": f"({p['weights'].lambda_spoilage:.1f}, {p['weights'].lambda_cost:.2f}, {p['weights'].lambda_unmet:.1f}, {p['weights'].lambda_risk:.1f})",
        "Spoilage %": f"{p['spoilage_pct']:.2f}%",
        "Cost ($)": f"${p['transport_cost']:,.2f}",
        "Delivered (kg)": f"{p['delivered_kg']:,.1f}",
        "Composite Fitness": f"{p['fitness']:.4f}",
    }
    for p in pareto_pts
])
print("\\nPareto Frontier Approximate Points:")
display(pareto_df)

# 4. Security Bounds & Input Sanitization
print("\\nVerifying Input Sanitization Layer:")
val_q = validate_quantity(1500.0, "farm_harvest")
val_t = validate_temperature(6.5, "reefer_temp")
val_node = validate_node_id("storage_01", "node_validation")
print(f"Sanitization Validated: Harvest={val_q} kg, Temp={val_t}°C, NodeID='{val_node}'")
print("All security guards, biophysical kinetics, and evolutionary loops operational.")"""))

    return nb


if __name__ == "__main__":
    nb = create_notebook()
    target_path = "/Users/parvatapuramrevanth/.gemini/antigravity/scratch/perishable_supply_optimizer/DPSRO_Complete_Optimization_Pipeline.ipynb"
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Master Notebook written to {target_path}")

    # Also save to notebooks/
    notebooks_target = "/Users/parvatapuramrevanth/.gemini/antigravity/scratch/perishable_supply_optimizer/notebooks/DPSRO_Complete_Optimization_Pipeline.ipynb"
    with open(notebooks_target, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Master Notebook also written to {notebooks_target}")
