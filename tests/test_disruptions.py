"""Unit tests for Disruption Engine and Dynamic Stateful Re-Optimization."""

import pytest
from src.data_generator import generate_supply_chain
from src.spoilage_model import SpoilageModel
from src.optimizer import DynamicPerishableOptimizer
from src.dynamic_optimizer import DynamicReoptimizer
from src.disruption import (
    TruckFailureDisruption,
    StorageOutageDisruption,
    DemandShockDisruption,
    TemperatureShockDisruption,
    CombinedDisruption,
)


def test_truck_failure_disruption():
    state = generate_supply_chain(num_batches=10, seed=42)
    disruption = TruckFailureDisruption()
    disrupted_state = disruption.apply(state)

    assert any(not t.available for t in disrupted_state.trucks.values()) or any(
        not r.available for r in disrupted_state.routes.values()
    )
    assert len(disrupted_state.disruptions_applied) > 0


def test_storage_outage_disruption():
    state = generate_supply_chain(num_batches=10, seed=42)
    s1 = list(state.storage_facilities.keys())[0]
    disruption = StorageOutageDisruption(storage_id=s1)
    disrupted_state = disruption.apply(state)

    assert not disrupted_state.storage_facilities[s1].available
    assert disrupted_state.storage_facilities[s1].available_capacity() == 0.0


def test_demand_shock_disruption():
    state = generate_supply_chain(num_batches=10, seed=42)
    original_demand = sum(m.demand for m in state.markets.values())

    disruption = DemandShockDisruption(multiplier=1.3)
    disrupted_state = disruption.apply(state)

    new_demand = sum(m.demand for m in disrupted_state.markets.values())
    assert abs(new_demand - original_demand * 1.3) < 5.0


def test_dynamic_reoptimization_preserves_valid_and_recovers():
    state = generate_supply_chain(num_batches=12, seed=42)
    spoilage = SpoilageModel()
    optimizer = DynamicPerishableOptimizer(spoilage_model=spoilage)
    reoptimizer = DynamicReoptimizer(optimizer=optimizer)

    # 1. Initial solve
    initial_res = optimizer.solve(state)
    assert initial_res.is_feasible

    # 2. Inject truck breakdown on an active route
    disrupted_state = state.clone()
    TruckFailureDisruption().apply(disrupted_state)

    # 3. Dynamic Re-optimize
    reopt_res, stats = reoptimizer.reoptimize_after_disruption(
        post_disruption_state=disrupted_state,
        prior_result=initial_res,
        elapsed_hours_during_disruption=1.5,
    )

    assert reopt_res.is_feasible
    assert stats["recovery_time_seconds"] < 1.0
    assert reopt_res.total_delivered > 0.0
