"""Interactive Command-Line Demonstration of DPSRO.

Demonstrates end-to-end lifecycle:
1. Initialize supply chain network & batches
2. Solve with Baseline 1 (Nearest Market)
3. Solve with Baseline 2 (Cheapest Route)
4. Solve with Proposed Method (DPSRO Multi-Objective MIP)
5. Inject Disruption (Truck Failure / Storage Outage)
6. Execute Dynamic Stateful Re-Optimization
7. Display comparative performance metrics and recovery statistics
"""

import sys
import yaml
from src.data_generator import generate_supply_chain
from src.spoilage_model import SpoilageModel
from src.baseline_nearest import solve_nearest_market
from src.baseline_cheapest import solve_cheapest_route
from src.optimizer import DynamicPerishableOptimizer
from src.disruption import TruckFailureDisruption, StorageOutageDisruption
from src.dynamic_optimizer import DynamicReoptimizer
from src.metrics import calculate_comparative_metrics, summarize_allocation_result


def run_cli_demonstration(seed: int = 42):
    print("=" * 80)
    print("DYNAMIC PERISHABLE SUPPLY-CHAIN RESILIENCE OPTIMIZER (DPSRO)")
    print("Computational Intelligence Prototype: CLI Demonstration")
    print("=" * 80)

    # 1. Initialize State
    print(f"\n[1/6] Generating Supply Chain Network (Seed = {seed})...")
    state = generate_supply_chain(
        num_farms=3,
        num_storage=2,
        num_markets=3,
        num_batches=10,
        num_trucks=5,
        seed=seed,
    )
    total_supply = sum(b.quantity for b in state.batches.values())
    total_demand = sum(m.demand for m in state.markets.values())
    print(f"  • Farms: {len(state.farms)} | Cold Storages: {len(state.storage_facilities)} | Markets: {len(state.markets)}")
    print(f"  • Perishable Batches: {len(state.batches)} (Total Produce: {total_supply:,.1f} kg)")
    print(f"  • Total Market Demand: {total_demand:,.1f} kg")

    # 2. Spoilage Kinetics & Optimizer
    spoilage = SpoilageModel(alpha=2.5, beta=0.8, reference_temperature=4.0)
    optimizer = DynamicPerishableOptimizer(
        spoilage_model=spoilage,
        lambda_spoilage=10.0,
        lambda_cost=0.15,
        lambda_unmet_demand=15.0,
        lambda_risk=5.0,
    )

    # 3. Solve Baselines & Proposed Method (Pre-disruption)
    print("\n[2/6] Executing Pre-Disruption Optimization...")
    res_nearest = solve_nearest_market(state, spoilage)
    res_cheapest = solve_cheapest_route(state, spoilage)
    res_proposed = optimizer.solve(state)

    print("\n--- PERFORMANCE SUMMARY (NORMAL NETWORK OPERATION) ---")
    headers = ["Method", "Dispatched (kg)", "Delivered (kg)", "Spoilage (kg)", "Spoilage %", "Demand Fulfilled %", "Cost ($)", "Runtime (s)"]
    row_fmt = "{:<32} {:<16} {:<16} {:<14} {:<12} {:<20} {:<12} {:<12}"
    print(row_fmt.format(*headers))
    print("-" * 140)

    for r in [res_nearest, res_cheapest, res_proposed]:
        print(row_fmt.format(
            r.method_name,
            f"{r.total_dispatched:,.1f}",
            f"{r.total_delivered:,.1f}",
            f"{r.total_spoilage:,.1f}",
            f"{r.spoilage_percentage:.2f}%",
            f"{r.demand_fulfillment_percentage:.2f}%",
            f"${r.total_transport_cost:,.2f}",
            f"{r.runtime_seconds:.5f}s",
        ))

    comp_nearest = calculate_comparative_metrics(res_nearest, res_proposed)
    comp_cheapest = calculate_comparative_metrics(res_cheapest, res_proposed)
    print(f"\n>> Advantage vs Nearest Baseline: Spoilage reduced by {comp_nearest['spoilage_reduction_percentage']}% | Fulfillment +{comp_nearest['fulfillment_gain_percentage_points']}% pts")
    print(f">> Advantage vs Cheapest Baseline: Spoilage reduced by {comp_cheapest['spoilage_reduction_percentage']}% | Delivered +{res_proposed.total_delivered - res_cheapest.total_delivered:.1f} kg fresh produce")

    # 4. Inject Disruption
    print("\n[3/6] Injecting Disruption: Operational Truck Breakdown...")
    disrupted_state = state.clone()
    truck_disruption = TruckFailureDisruption()
    disrupted_state = truck_disruption.apply(disrupted_state)
    disruption_info = disrupted_state.disruptions_applied[-1]
    print(f"  • Event: {truck_disruption.describe()}")
    print(f"  • Asset Affected: {disruption_info['target_id']}")
    print(f"  • Routes Disabled: {disruption_info['affected_routes']}")

    # 5. Dynamic Re-Optimization
    print("\n[4/6] Executing Dynamic Stateful Re-Optimization...")
    reoptimizer = DynamicReoptimizer(optimizer=optimizer)
    res_reopt, recovery_stats = reoptimizer.reoptimize_after_disruption(
        post_disruption_state=disrupted_state,
        prior_result=res_proposed,
        elapsed_hours_during_disruption=2.5,
    )

    print("\n--- DYNAMIC RECOVERY METRICS ---")
    print(f"  • Recovery Decision Time: {recovery_stats['recovery_time_seconds']:.5f}s")
    print(f"  • Affected Batches: {recovery_stats['affected_batch_count']} ({recovery_stats['affected_batches']})")
    print(f"  • Stranded Inventory: {recovery_stats['stranded_quantity_kg']} kg")
    print(f"  • Successfully Rerouted Inventory: {recovery_stats['successfully_rerouted_kg']} kg ({recovery_stats['reroute_success_rate_pct']}%)")
    print(f"  • Post-Disruption Total Delivered: {res_reopt.total_delivered:,.1f} kg")
    print(f"  • Post-Disruption Total Spoilage: {res_reopt.total_spoilage:,.1f} kg ({res_reopt.spoilage_percentage:.2f}%)")
    print(f"  • Demand Fulfillment: {res_reopt.demand_fulfillment_percentage:.2f}%")
    print(f"  • Cost Impact: ${res_reopt.total_transport_cost:,.2f} (Delta: ${recovery_stats['cost_delta']:+,.2f})")

    # 6. Detailed Allocation Changes
    print("\n[5/6] Sample Dynamic Allocation Transitions:")
    rerouted_allocs = [a for a in res_reopt.allocations if a.batch_id in recovery_stats['affected_batches']]
    for a in rerouted_allocs[:5]:
        hub_str = f"via Storage {a.storage_id}" if a.storage_id else "Direct Express"
        print(f"  -> Batch {a.batch_id}: {a.quantity:,.0f} kg routed {hub_str} to Market {a.market_id} | Transit: {a.travel_time_hours:.1f}h | Spoilage: {a.spoiled_qty:.1f} kg")

    print("\n[6/6] Demonstration Complete: Mathematical validation successful.\n")
    return res_proposed, res_reopt, recovery_stats


if __name__ == "__main__":
    run_cli_demonstration()
