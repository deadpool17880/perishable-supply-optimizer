"""Unit tests for supply-chain constraint enforcement."""

import pytest
from src.data_generator import generate_supply_chain
from src.spoilage_model import SpoilageModel
from src.optimizer import DynamicPerishableOptimizer


@pytest.fixture
def solved_network():
    state = generate_supply_chain(num_farms=3, num_storage=2, num_markets=3, num_batches=15, seed=42)
    spoilage = SpoilageModel()
    optimizer = DynamicPerishableOptimizer(spoilage_model=spoilage)
    result = optimizer.solve(state)
    return state, result


def test_no_negative_flows(solved_network):
    state, result = solved_network
    for alloc in result.allocations:
        assert alloc.quantity >= 0.0
        assert alloc.delivered_qty >= 0.0
        assert alloc.spoiled_qty >= 0.0
        assert alloc.transport_cost >= 0.0


def test_batch_capacity_not_exceeded(solved_network):
    state, result = solved_network
    batch_allocated = {}
    for alloc in result.allocations:
        batch_allocated[alloc.batch_id] = batch_allocated.get(alloc.batch_id, 0.0) + alloc.quantity

    for b_id, alloc_qty in batch_allocated.items():
        original_qty = state.batches[b_id].quantity
        # Allow minor float precision tolerance of 0.1 kg
        assert alloc_qty <= original_qty + 0.1, f"Batch {b_id} exceeded supply: {alloc_qty} > {original_qty}"


def test_storage_capacity_not_exceeded(solved_network):
    state, result = solved_network
    storage_allocated = {}
    for alloc in result.allocations:
        if alloc.storage_id:
            storage_allocated[alloc.storage_id] = storage_allocated.get(alloc.storage_id, 0.0) + alloc.quantity

    for s_id, alloc_qty in storage_allocated.items():
        cap = state.storage_facilities[s_id].capacity
        assert alloc_qty <= cap + 0.1, f"Storage {s_id} exceeded capacity: {alloc_qty} > {cap}"


def test_market_demand_conservation(solved_network):
    state, result = solved_network
    market_delivered = {}
    for alloc in result.allocations:
        market_delivered[alloc.market_id] = market_delivered.get(alloc.market_id, 0.0) + alloc.delivered_qty

    total_market_demand = sum(m.demand for m in state.markets.values())
    total_delivered = sum(market_delivered.values())
    # Numerical tolerance for rounded allocation sums
    assert total_delivered <= total_market_demand + 0.1
    assert abs(total_delivered + result.unmet_demand - total_market_demand) < 1.0


def test_no_routing_over_unavailable_assets():
    state = generate_supply_chain(num_farms=3, num_storage=2, num_markets=3, num_batches=10, seed=42)
    # Mark Storage S1 as unavailable
    state.storage_facilities["S1"].available = False

    spoilage = SpoilageModel()
    optimizer = DynamicPerishableOptimizer(spoilage_model=spoilage)
    result = optimizer.solve(state)

    for alloc in result.allocations:
        assert alloc.storage_id != "S1", "Allocated through offline storage S1!"
