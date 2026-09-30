"""Dynamic Re-Optimization & State-Preserving Recovery Engine.

Manages dynamic state transitions:
    State(t) -> Initial Allocation -> Disruption Event -> State(t+1) -> Dynamic Re-Optimization

Identifies invalidated flows, preserves operational flows where feasible, reroutes stranded
perishable batches, and computes recovery metrics (recovery time, rerouted quantities, delta costs).
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from src.models import (
    SupplyChainState,
    AllocationDecision,
    AllocationResult,
    Batch,
)
from src.spoilage_model import SpoilageModel
from src.optimizer import DynamicPerishableOptimizer


class DynamicReoptimizer:
    """Orchestrates dynamic re-allocation of supply chains undergoing disruptions."""

    def __init__(self, optimizer: DynamicPerishableOptimizer):
        self.optimizer = optimizer
        self.spoilage_model = optimizer.spoilage_model

    def reoptimize_after_disruption(
        self,
        post_disruption_state: SupplyChainState,
        prior_result: AllocationResult,
        elapsed_hours_during_disruption: float = 2.0,
    ) -> Tuple[AllocationResult, Dict[str, Any]]:
        """Re-optimizes an ongoing supply-chain plan after a disruption.

        Steps:
        1. Identify which allocations in `prior_result` became invalid:
           - Traverse an unavailable route
           - Rely on an unavailable storage facility
        2. Keep intact allocations that remain fully valid and operational.
        3. Age remaining batches by `elapsed_hours_during_disruption`.
        4. Create a residual optimization problem for the stranded inventory and unfulfilled demands.
        5. Solve with DynamicPerishableOptimizer on the updated residual network.
        6. Merge preserved allocations and new rerouted allocations into a unified plan.
        7. Compute recovery metrics:
           - Recovery time
           - Rerouted quantity & batch count
           - Spoilage difference
           - Delta cost
        """
        recovery_start = time.perf_counter()

        preserved_allocations: List[AllocationDecision] = []
        stranded_batches_qty: Dict[str, float] = {}
        affected_batch_ids = set()
        affected_route_ids = set()
        affected_storage_ids = set()

        # Check each prior allocation for viability in post_disruption_state
        for alloc in prior_result.allocations:
            is_valid = True

            # Check routes
            for rid in alloc.route_ids:
                route = post_disruption_state.routes.get(rid)
                if not route or not route.available:
                    is_valid = False
                    affected_route_ids.add(rid)

            # Check storage
            if alloc.storage_id:
                st = post_disruption_state.storage_facilities.get(alloc.storage_id)
                if not st or not st.available:
                    is_valid = False
                    affected_storage_ids.add(alloc.storage_id)

            if is_valid:
                # Flow can proceed unaffected
                preserved_allocations.append(alloc)
            else:
                # Allocation is stranded and needs dynamic rerouting
                affected_batch_ids.add(alloc.batch_id)
                stranded_batches_qty[alloc.batch_id] = (
                    stranded_batches_qty.get(alloc.batch_id, 0.0) + alloc.quantity
                )

        # Build residual supply chain state
        residual_state = post_disruption_state.clone()

        # 1. Update batch shelf lives for elapsed time during disruption incident
        for bid, batch in residual_state.batches.items():
            aged_batch = self.spoilage_model.update_batch_shelf_life(
                batch=batch,
                elapsed_hours=elapsed_hours_during_disruption,
                storage_temp=batch.temperature,
            )
            residual_state.batches[bid] = aged_batch

        # 2. Set batch quantities to stranded quantities (or 0 if already safely dispatched)
        for bid, batch in residual_state.batches.items():
            batch.quantity = stranded_batches_qty.get(bid, 0.0)

        # 3. Deduct preserved deliveries from market demands
        for alloc in preserved_allocations:
            if alloc.market_id in residual_state.markets:
                m = residual_state.markets[alloc.market_id]
                m.demand = max(0.0, m.demand - alloc.delivered_qty)

        # 4. Deduct preserved flows through storage facilities
        for alloc in preserved_allocations:
            if alloc.storage_id and alloc.storage_id in residual_state.storage_facilities:
                st = residual_state.storage_facilities[alloc.storage_id]
                st.current_load = min(st.capacity, st.current_load + alloc.quantity)

        # 5. Solve residual optimization problem
        residual_result = self.optimizer.solve(residual_state)
        recovery_time = time.perf_counter() - recovery_start

        # 6. Merge preserved and rerouted allocations
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
            "affected_routes": list(affected_route_ids),
            "affected_storages": list(affected_storage_ids),
            "stranded_quantity_kg": round(total_stranded_qty, 1),
            "successfully_rerouted_kg": round(rerouted_qty, 1),
            "reroute_success_rate_pct": round(
                (rerouted_qty / total_stranded_qty * 100.0) if total_stranded_qty > 0 else 100.0,
                2,
            ),
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
            total_risk_score=round(total_risk, 2),
            runtime_seconds=round(recovery_time, 5),
            solver_status=residual_result.solver_status,
            is_feasible=residual_result.is_feasible,
            objective_value=round(residual_result.objective_value, 2),
            extra_metrics=recovery_summary,
        )

        return reoptimized_result, recovery_summary
