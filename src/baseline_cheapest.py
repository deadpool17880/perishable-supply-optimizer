"""Baseline 2: Cheapest-Route Allocation.

Solves a minimum-cost flow LP formulation that minimizes transportation expenditure
while satisfying supply, storage capacity, and demand constraints, but entirely ignores
shelf-life decay and perishability risk.
"""

import time
import numpy as np
from scipy.optimize import linprog
from typing import List, Dict, Optional, Tuple, Any
from src.models import (
    SupplyChainState,
    AllocationDecision,
    AllocationResult,
    TransportRoute,
)
from src.spoilage_model import SpoilageModel


def solve_cheapest_route(
    state: SupplyChainState,
    spoilage_model: SpoilageModel,
) -> AllocationResult:
    """Solves allocation minimizing pure transportation cost using SciPy HiGHS LP.

    Objective:
        Min sum(cost_p * x_p) + M * sum(u_m)

    Subject to:
        sum_{p in Paths(b)} x_p <= quantity(b)   for each batch b
        sum_{p via s} x_p <= capacity(s)          for each storage facility s
        sum_{p to m} x_p + u_m = demand(m)        for each market m
        x_p >= 0, u_m >= 0
    """
    start_time = time.perf_counter()

    # 1. Enumerate candidate feasible paths
    # Each path: (batch_id, farm_id, storage_id, market_id, travel_time, unit_cost, risk_factor, route_ids)
    candidate_paths: List[Dict[str, Any]] = []

    for b_id, batch in state.batches.items():
        farm = state.farms.get(batch.farm_id)
        if not farm or batch.quantity <= 0:
            continue

        # Direct paths Farm -> Market
        for m_id, market in state.markets.items():
            r_dir = state.get_route(farm.farm_id, m_id)
            if r_dir and r_dir.available:
                candidate_paths.append({
                    "batch_id": b_id,
                    "farm_id": farm.farm_id,
                    "storage_id": None,
                    "market_id": m_id,
                    "travel_time": r_dir.travel_time_hours,
                    "unit_cost": r_dir.transport_cost_per_unit,
                    "risk_factor": r_dir.risk_factor,
                    "route_ids": [r_dir.route_id],
                })

        # Hub paths Farm -> Storage -> Market
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
                total_c = r1.transport_cost_per_unit + r2.transport_cost_per_unit
                avg_risk = (r1.risk_factor + r2.risk_factor) / 2.0

                candidate_paths.append({
                    "batch_id": b_id,
                    "farm_id": farm.farm_id,
                    "storage_id": s_id,
                    "market_id": m_id,
                    "travel_time": total_t,
                    "unit_cost": total_c,
                    "risk_factor": avg_risk,
                    "route_ids": [r1.route_id, r2.route_id],
                })

    num_paths = len(candidate_paths)
    market_ids = list(state.markets.keys())
    num_markets = len(market_ids)
    market_idx = {m_id: i for i, m_id in enumerate(market_ids)}

    batch_ids = list(state.batches.keys())
    num_batches = len(batch_ids)
    batch_idx = {b_id: i for i, b_id in enumerate(batch_ids)}

    storage_ids = list(state.storage_facilities.keys())
    num_storage = len(storage_ids)
    storage_idx = {s_id: i for i, s_id in enumerate(storage_ids)}

    # Total decision variables: N_paths (x_p) + N_markets (u_m slack for unmet demand)
    total_vars = num_paths + num_markets

    if num_paths == 0:
        # Edge case: No feasible paths available
        runtime = time.perf_counter() - start_time
        total_demand = sum(m.demand for m in state.markets.values())
        return AllocationResult(
            method_name="Cheapest Route Baseline",
            allocations=[],
            total_dispatched=0.0,
            total_delivered=0.0,
            total_spoilage=0.0,
            spoilage_percentage=0.0,
            demand_fulfillment_percentage=0.0,
            unmet_demand=total_demand,
            total_transport_cost=0.0,
            total_risk_score=0.0,
            runtime_seconds=round(runtime, 5),
            solver_status="NO_FEASIBLE_PATHS",
            is_feasible=False,
        )

    # Objective vector c:
    # Minimize sum(unit_cost_p * x_p) + unmet_penalty * sum(u_m)
    c = np.zeros(total_vars)
    for p_i, p in enumerate(candidate_paths):
        c[p_i] = p["unit_cost"]

    UNMET_PENALTY = 50.0  # High penalty ensuring demand is satisfied whenever feasible
    for m_i in range(num_markets):
        c[num_paths + m_i] = UNMET_PENALTY

    # Constraints:
    # 1. Batch supply: sum_{p in Paths(b)} x_p <= quantity(b)  (Inequality <=)
    # 2. Storage capacity: sum_{p via s} x_p <= capacity(s)    (Inequality <=)
    num_ub = num_batches + num_storage
    A_ub = np.zeros((num_ub, total_vars))
    b_ub = np.zeros(num_ub)

    for p_i, p in enumerate(candidate_paths):
        b_i = batch_idx[p["batch_id"]]
        A_ub[b_i, p_i] = 1.0
        if p["storage_id"] is not None:
            s_i = storage_idx[p["storage_id"]]
            A_ub[num_batches + s_i, p_i] = 1.0

    for b_id, b_i in batch_idx.items():
        b_ub[b_i] = state.batches[b_id].quantity

    for s_id, s_i in storage_idx.items():
        st = state.storage_facilities[s_id]
        b_ub[num_batches + s_i] = st.available_capacity() if st.available else 0.0

    # 3. Market demand: sum_{p to m} x_p + u_m = demand(m)  (Equality ==)
    A_eq = np.zeros((num_markets, total_vars))
    b_eq = np.zeros(num_markets)

    for p_i, p in enumerate(candidate_paths):
        m_i = market_idx[p["market_id"]]
        A_eq[m_i, p_i] = 1.0

    for m_id, m_i in market_idx.items():
        A_eq[m_i, num_paths + m_i] = 1.0  # slack variable u_m
        b_eq[m_i] = state.markets[m_id].demand

    # Bounds: all variables >= 0
    bounds = [(0, None) for _ in range(total_vars)]

    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    runtime = time.perf_counter() - start_time

    if not res.success:
        total_demand = sum(m.demand for m in state.markets.values())
        return AllocationResult(
            method_name="Cheapest Route Baseline",
            allocations=[],
            total_dispatched=0.0,
            total_delivered=0.0,
            total_spoilage=0.0,
            spoilage_percentage=0.0,
            demand_fulfillment_percentage=0.0,
            unmet_demand=total_demand,
            total_transport_cost=0.0,
            total_risk_score=0.0,
            runtime_seconds=round(runtime, 5),
            solver_status=res.message,
            is_feasible=False,
        )

    x_sol = res.x
    allocations: List[AllocationDecision] = []
    total_dispatched = 0.0
    total_delivered = 0.0
    total_spoilage = 0.0
    total_cost = 0.0
    total_risk = 0.0

    for p_i, p in enumerate(candidate_paths):
        flow_qty = x_sol[p_i]
        if flow_qty < 1e-3:
            continue

        batch = state.batches[p["batch_id"]]
        # Evaluate physical spoilage on this path
        spoil_fraction = spoilage_model.calculate_spoilage_fraction(
            batch=batch,
            transit_time_hours=p["travel_time"],
            route_temp=4.0 if p["storage_id"] else batch.temperature,
        )
        spoiled_qty = flow_qty * spoil_fraction
        delivered_qty = flow_qty - spoiled_qty
        cost_val = flow_qty * p["unit_cost"]
        risk_val = flow_qty * p["risk_factor"]

        allocations.append(
            AllocationDecision(
                batch_id=batch.batch_id,
                farm_id=p["farm_id"],
                storage_id=p["storage_id"],
                market_id=p["market_id"],
                quantity=flow_qty,
                spoilage_fraction=spoil_fraction,
                spoiled_qty=spoiled_qty,
                delivered_qty=delivered_qty,
                transport_cost=cost_val,
                travel_time_hours=p["travel_time"],
                risk_score=risk_val,
                route_ids=p["route_ids"],
            )
        )

        total_dispatched += flow_qty
        total_delivered += delivered_qty
        total_spoilage += spoiled_qty
        total_cost += cost_val
        total_risk += risk_val

    total_market_demand = sum(m.demand for m in state.markets.values())
    unmet_demand = sum(x_sol[num_paths + m_i] for m_i in range(num_markets))
    fulfillment_pct = (
        (total_delivered / total_market_demand * 100.0) if total_market_demand > 0 else 0.0
    )
    spoilage_pct = (
        (total_spoilage / total_dispatched * 100.0) if total_dispatched > 0 else 0.0
    )

    return AllocationResult(
        method_name="Cheapest Route Baseline",
        allocations=allocations,
        total_dispatched=round(total_dispatched, 1),
        total_delivered=round(total_delivered, 1),
        total_spoilage=round(total_spoilage, 1),
        spoilage_percentage=round(spoilage_pct, 2),
        demand_fulfillment_percentage=round(fulfillment_pct, 2),
        unmet_demand=round(unmet_demand, 1),
        total_transport_cost=round(total_cost, 2),
        total_risk_score=round(total_risk, 2),
        runtime_seconds=round(runtime, 5),
        solver_status="OPTIMAL",
        is_feasible=True,
        objective_value=round(float(res.fun), 2),
    )
