"""Unit tests for Dynamic Perishable Optimizer and Baseline comparison."""

import pytest
from src.data_generator import generate_supply_chain
from src.spoilage_model import SpoilageModel
from src.baseline_nearest import solve_nearest_market
from src.baseline_cheapest import solve_cheapest_route
from src.optimizer import DynamicPerishableOptimizer


def test_optimizer_feasibility_and_optimality():
    state = generate_supply_chain(num_batches=12, seed=42)
    spoilage = SpoilageModel()
    optimizer = DynamicPerishableOptimizer(spoilage_model=spoilage)
    result = optimizer.solve(state)

    assert result.is_feasible
    assert result.solver_status == "OPTIMAL"
    assert result.total_delivered > 0.0
    assert result.runtime_seconds < 1.0


def test_proposed_outperforms_nearest_on_spoilage():
    state = generate_supply_chain(num_batches=15, seed=42)
    spoilage = SpoilageModel()
    res_near = solve_nearest_market(state, spoilage)
    res_prop = DynamicPerishableOptimizer(spoilage_model=spoilage).solve(state)

    assert res_prop.spoilage_percentage < res_near.spoilage_percentage
    assert res_prop.total_delivered >= res_near.total_delivered


def test_edge_case_zero_demand():
    state = generate_supply_chain(num_batches=5, seed=42)
    for m in state.markets.values():
        m.demand = 0.0

    spoilage = SpoilageModel()
    res = DynamicPerishableOptimizer(spoilage_model=spoilage).solve(state)
    assert res.is_feasible
    assert res.total_dispatched == 0.0
    assert res.total_delivered == 0.0


def test_edge_case_zero_storage_capacity():
    state = generate_supply_chain(num_batches=5, seed=42)
    for s in state.storage_facilities.values():
        s.capacity = 0.0

    spoilage = SpoilageModel()
    res = DynamicPerishableOptimizer(spoilage_model=spoilage).solve(state)
    assert res.is_feasible
    # All routed flows must be direct (storage_id is None)
    for a in res.allocations:
        assert a.storage_id is None


def test_expired_batches_excluded():
    state = generate_supply_chain(num_batches=5, seed=42)
    # Set shelf life very low (0.1 hours)
    for b in state.batches.values():
        b.remaining_shelf_life_hours = 0.1

    spoilage = SpoilageModel()
    res = DynamicPerishableOptimizer(spoilage_model=spoilage).solve(state, allow_expired_routing=False)
    # Cannot deliver any expired produce fresh
    assert res.total_delivered == 0.0
