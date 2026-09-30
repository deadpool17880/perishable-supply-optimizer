"""Baseline 1: Nearest-Market Greedy Allocation.

Implements an intuitive heuristic that dispatches inventory to the geographically nearest
accessible market until demand or capacity is exhausted, without considering perishability risk.
"""

import time
from typing import List, Dict, Optional, Tuple
from src.models import (
    SupplyChainState,
    AllocationDecision,
    AllocationResult,
    TransportRoute,
)
from src.spoilage_model import SpoilageModel


def solve_nearest_market(
    state: SupplyChainState,
    spoilage_model: SpoilageModel,
) -> AllocationResult:
    """Solves allocation using Nearest-Market greedy algorithm.

    Algorithm:
    For each batch:
      1. Enumerate feasible routes to markets (direct or via available storage).
      2. Select the route with the minimum total distance.
      3. Allocate quantity min(batch_quantity, market_remaining_demand, storage_capacity).
      4. Repeat until batch is fully allocated or no feasible market remains.
    """
    start_time = time.perf_counter()

    # Track remaining market demands and storage capacities
    remaining_demand: Dict[str, float] = {
        m_id: m.demand for m_id, m in state.markets.items()
    }
    remaining_storage_cap: Dict[str, float] = {
        s_id: s.available_capacity() for s_id, s in state.storage_facilities.items()
    }

    allocations: List[AllocationDecision] = []
    total_dispatched = 0.0
    total_delivered = 0.0
    total_spoilage = 0.0
    total_cost = 0.0
    total_risk = 0.0

    # Sort batches by harvest time / batch ID
    batch_list = sorted(state.batches.values(), key=lambda b: b.harvest_time)

    for batch in batch_list:
        batch_qty_left = batch.quantity
        if batch_qty_left <= 0:
            continue

        farm = state.farms.get(batch.farm_id)
        if not farm:
            continue

        # Enumerate candidate paths: (path_type, target_market, storage_id, total_dist, total_time, cost_per_kg, risk, route_ids)
        candidate_paths = []

        # 1. Direct routes Farm -> Market
        for m_id, market in state.markets.items():
            if remaining_demand[m_id] <= 0:
                continue
            dir_route = state.get_route(farm.farm_id, m_id)
            if dir_route and dir_route.available:
                candidate_paths.append((
                    "direct",
                    m_id,
                    None,
                    dir_route.distance_km,
                    dir_route.travel_time_hours,
                    dir_route.transport_cost_per_unit,
                    dir_route.risk_factor,
                    [dir_route.route_id],
                ))

        # 2. Via Storage Hub: Farm -> Storage -> Market
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

                total_d = r1.distance_km + r2.distance_km
                total_t = r1.travel_time_hours + r2.travel_time_hours
                total_c = r1.transport_cost_per_unit + r2.transport_cost_per_unit
                avg_risk = (r1.risk_factor + r2.risk_factor) / 2.0

                candidate_paths.append((
                    "hub",
                    m_id,
                    s_id,
                    total_d,
                    total_t,
                    total_c,
                    avg_risk,
                    [r1.route_id, r2.route_id],
                ))

        # Sort candidate paths by minimum distance (Nearest first)
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

            # Spoilage physics evaluation
            spoil_fraction = spoilage_model.calculate_spoilage_fraction(
                batch=batch,
                transit_time_hours=travel_t,
                route_temp=4.0 if s_id else batch.temperature,
            )
            spoiled_qty = alloc_qty * spoil_fraction
            delivered_qty = alloc_qty - spoiled_qty
            trans_cost = alloc_qty * unit_cost
            risk_score = alloc_qty * risk_val

            allocations.append(
                AllocationDecision(
                    batch_id=batch.batch_id,
                    farm_id=batch.farm_id,
                    storage_id=s_id,
                    market_id=m_id,
                    quantity=alloc_qty,
                    spoilage_fraction=spoil_fraction,
                    spoiled_qty=spoiled_qty,
                    delivered_qty=delivered_qty,
                    transport_cost=trans_cost,
                    travel_time_hours=travel_t,
                    risk_score=risk_score,
                    route_ids=r_ids,
                )
            )

            total_dispatched += alloc_qty
            total_delivered += delivered_qty
            total_spoilage += spoiled_qty
            total_cost += trans_cost
            total_risk += risk_score

    runtime = time.perf_counter() - start_time
    total_market_demand = sum(m.demand for m in state.markets.values())
    unmet = sum(max(0.0, d) for d in remaining_demand.values())
    fulfillment_pct = (
        (total_delivered / total_market_demand * 100.0) if total_market_demand > 0 else 0.0
    )
    spoilage_pct = (
        (total_spoilage / total_dispatched * 100.0) if total_dispatched > 0 else 0.0
    )

    return AllocationResult(
        method_name="Nearest Market Baseline",
        allocations=allocations,
        total_dispatched=total_dispatched,
        total_delivered=total_delivered,
        total_spoilage=total_spoilage,
        spoilage_percentage=round(spoilage_pct, 2),
        demand_fulfillment_percentage=round(fulfillment_pct, 2),
        unmet_demand=round(unmet, 1),
        total_transport_cost=round(total_cost, 2),
        total_risk_score=round(total_risk, 2),
        runtime_seconds=round(runtime, 5),
        solver_status="SUCCESS",
        is_feasible=True,
    )
