"""Proposed Dynamic Perishability-Aware Optimizer (DPSRO).

Implements a multi-objective mathematical program that jointly optimizes:
1. Spoilage minimization via biophysical kinetics
2. Demand fulfillment prioritization
3. Transportation cost efficiency
4. Disruption & infrastructure risk avoidance
"""

import time
import numpy as np
from scipy.optimize import linprog
from typing import List, Dict, Optional, Tuple, Any
from src.models import (
    SupplyChainState,
    AllocationDecision,
    AllocationResult,
)
from src.spoilage_model import SpoilageModel


class DynamicPerishableOptimizer:
    """Exact multi-objective optimizer using SciPy HiGHS."""

    def __init__(
        self,
        spoilage_model: SpoilageModel,
        lambda_spoilage: float = 10.0,
        lambda_cost: float = 0.15,
        lambda_unmet_demand: float = 15.0,
        lambda_risk: float = 5.0,
        time_limit_seconds: float = 30.0,
    ):
        self.spoilage_model = spoilage_model
        self.lambda_spoilage = float(lambda_spoilage)
        self.lambda_cost = float(lambda_cost)
        self.lambda_unmet_demand = float(lambda_unmet_demand)
        self.lambda_risk = float(lambda_risk)
        self.time_limit_seconds = float(time_limit_seconds)

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], spoilage_model: SpoilageModel
    ) -> "DynamicPerishableOptimizer":
        weights = config.get("weights", {})
        solver_cfg = config.get("solver", {})
        return cls(
            spoilage_model=spoilage_model,
            lambda_spoilage=weights.get("lambda_spoilage", 10.0),
            lambda_cost=weights.get("lambda_cost", 0.15),
            lambda_unmet_demand=weights.get("lambda_unmet_demand", 15.0),
            lambda_risk=weights.get("lambda_risk", 5.0),
            time_limit_seconds=solver_cfg.get("time_limit_seconds", 30.0),
        )

    def solve(
        self,
        state: SupplyChainState,
        allow_expired_routing: bool = False,
    ) -> AllocationResult:
        """Formulates and solves the perishability-weighted linear program.

        Variables:
            x_p: Quantity dispatched along path p
            u_m: Unmet demand slack variable at market m

        Objective:
            Min sum_p [ (lambda_spoilage * spoil_frac_p + lambda_cost * cost_p + lambda_risk * risk_p) * x_p ]
                + sum_m [ lambda_unmet_demand * priority_m * u_m ]

        Constraints:
            1. Batch availability: sum_{p in Paths(b)} x_p <= quantity(b)
            2. Storage capacity:   sum_{p via s} x_p <= capacity(s)
            3. Delivered demand:   sum_{p to m} (1 - spoil_frac_p) * x_p + u_m = demand(m)
            4. Non-negativity:     x_p >= 0, u_m >= 0
        """
        start_time = time.perf_counter()

        candidate_paths: List[Dict[str, Any]] = []

        # 1. Enumerate candidate paths respecting current network state
        for b_id, batch in state.batches.items():
            if batch.quantity <= 0:
                continue
            farm = state.farms.get(batch.farm_id)
            if not farm:
                continue

            # (a) Direct paths: Farm -> Market
            for m_id, market in state.markets.items():
                r_dir = state.get_route(farm.farm_id, m_id)
                if not r_dir or not r_dir.available:
                    continue

                travel_time = r_dir.travel_time_hours
                if not allow_expired_routing and travel_time > batch.remaining_shelf_life_hours:
                    # Physically cannot arrive fresh
                    continue

                spoil_frac = self.spoilage_model.calculate_spoilage_fraction(
                    batch=batch,
                    transit_time_hours=travel_time,
                    route_temp=batch.temperature,
                )
                spoil_risk = self.spoilage_model.calculate_spoilage_risk(
                    batch=batch,
                    transit_time_hours=travel_time,
                    route_temp=batch.temperature,
                )
                combined_risk = (r_dir.risk_factor + spoil_risk) / 2.0

                candidate_paths.append({
                    "batch_id": b_id,
                    "farm_id": farm.farm_id,
                    "storage_id": None,
                    "market_id": m_id,
                    "travel_time": travel_time,
                    "unit_cost": r_dir.transport_cost_per_unit,
                    "spoil_frac": spoil_frac,
                    "risk_factor": combined_risk,
                    "route_ids": [r_dir.route_id],
                })

            # (b) Hub paths: Farm -> Storage -> Market
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

                    # Cold storage protects produce between legs
                    spoil_frac = self.spoilage_model.calculate_spoilage_fraction(
                        batch=batch,
                        transit_time_hours=total_t,
                        route_temp=storage.temperature_capacity,
                    )
                    spoil_risk = self.spoilage_model.calculate_spoilage_risk(
                        batch=batch,
                        transit_time_hours=total_t,
                        route_temp=storage.temperature_capacity,
                    )
                    route_risk = (r1.risk_factor + r2.risk_factor) / 2.0
                    combined_risk = (route_risk + spoil_risk) / 2.0

                    total_c = r1.transport_cost_per_unit + r2.transport_cost_per_unit

                    candidate_paths.append({
                        "batch_id": b_id,
                        "farm_id": farm.farm_id,
                        "storage_id": s_id,
                        "market_id": m_id,
                        "travel_time": total_t,
                        "unit_cost": total_c,
                        "spoil_frac": spoil_frac,
                        "risk_factor": combined_risk,
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

        total_vars = num_paths + num_markets

        if num_paths == 0:
            runtime = time.perf_counter() - start_time
            total_demand = sum(m.demand for m in state.markets.values())
            return AllocationResult(
                method_name="Dynamic Perishable Optimizer",
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

        # 2. Objective coefficients
        c = np.zeros(total_vars)
        for p_i, p in enumerate(candidate_paths):
            c[p_i] = (
                self.lambda_spoilage * p["spoil_frac"]
                + self.lambda_cost * p["unit_cost"]
                + self.lambda_risk * p["risk_factor"]
            )

        for m_id, m_i in market_idx.items():
            market = state.markets[m_id]
            c[num_paths + m_i] = self.lambda_unmet_demand * market.priority

        # 3. Inequality Constraints (Batch quantity & Storage capacity)
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

        # 4. Equality Constraints (Market demand satisfied by delivered usable produce)
        A_eq = np.zeros((num_markets, total_vars))
        b_eq = np.zeros(num_markets)

        for p_i, p in enumerate(candidate_paths):
            m_i = market_idx[p["market_id"]]
            # Each unit dispatched delivers (1 - spoil_frac) units of fresh produce
            A_eq[m_i, p_i] = max(0.01, 1.0 - p["spoil_frac"])

        for m_id, m_i in market_idx.items():
            A_eq[m_i, num_paths + m_i] = 1.0  # slack u_m
            b_eq[m_i] = state.markets[m_id].demand

        bounds = [(0, None) for _ in range(total_vars)]

        res = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"time_limit": self.time_limit_seconds},
        )

        runtime = time.perf_counter() - start_time

        if not res.success:
            total_demand = sum(m.demand for m in state.markets.values())
            return AllocationResult(
                method_name="Dynamic Perishable Optimizer",
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
            spoil_fraction = p["spoil_frac"]
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
                    quantity=round(flow_qty, 2),
                    spoilage_fraction=round(spoil_fraction, 4),
                    spoiled_qty=round(spoiled_qty, 2),
                    delivered_qty=round(delivered_qty, 2),
                    transport_cost=round(cost_val, 2),
                    travel_time_hours=p["travel_time"],
                    risk_score=round(risk_val, 2),
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
            method_name="Dynamic Perishable Optimizer",
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
