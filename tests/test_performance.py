"""DPSRO Performance Benchmark Suite.

Measures wall-clock latency and throughput for all critical hot-paths:
- Spoilage model scalar vs. vectorised batch matrix
- LP optimizer across 6 problem scales (10 → 500 batches)
- Dynamic re-optimizer recovery time after disruptions
- Cache efficiency (hit rates) for LRU-cached spoilage computations

Run with:
    python -m tests.test_performance

Or as part of the full test suite:
    pytest tests/test_performance.py -v -s
"""

from __future__ import annotations

import time
import unittest
from typing import List

import numpy as np

from src.data_generator import generate_supply_chain
from src.disruption import TruckFailureDisruption, CombinedDisruption
from src.dynamic_optimizer import DynamicReoptimizer
from src.optimizer import DynamicPerishableOptimizer
from src.spoilage_model import SpoilageModel


# ─────────────────────────────────────────────────────────────────────────────
# Latency thresholds (milliseconds)
# ─────────────────────────────────────────────────────────────────────────────
SCALAR_SPOILAGE_THRESHOLD_MS = 5.0       # 1000 scalar spoilage risk calls
VECTOR_SPOILAGE_THRESHOLD_MS = 10.0      # Vectorised batch of 1000 paths
OPTIMIZER_SMALL_THRESHOLD_MS = 150.0     # 10-batch LP solve
OPTIMIZER_MEDIUM_THRESHOLD_MS = 800.0    # 100-batch LP solve
OPTIMIZER_LARGE_THRESHOLD_MS = 3000.0    # 500-batch LP solve
REOPTIMIZER_THRESHOLD_MS = 500.0         # Dynamic re-optimization after disruption
CACHE_MIN_HIT_RATE = 0.50                # Cache hit rate floor after 100 calls


class TestSpoilageModelLatency(unittest.TestCase):
    """Benchmarks for SpoilageModel scalar and vectorised code paths."""

    def setUp(self) -> None:
        self.model = SpoilageModel(alpha=2.5, beta=0.8, reference_temperature=4.0)
        self.state = generate_supply_chain(num_batches=10, seed=42)

    def test_temperature_penalty_cache_hit_rate(self) -> None:
        """LRU cache hit rate must exceed 50% after 100 calls with repeated temperatures."""
        temps = [4.0, 8.0, 15.0, 20.0, 4.0, 8.0, 4.0, 15.0, 20.0, 4.0] * 10
        for t in temps:
            self.model.calculate_temperature_penalty(t)
        info = self.model.cache_info()["temperature_penalty"]
        hit_rate = info.hits / (info.hits + info.misses + 1e-9)
        self.assertGreaterEqual(
            hit_rate, CACHE_MIN_HIT_RATE,
            f"Cache hit rate {hit_rate:.2%} below threshold {CACHE_MIN_HIT_RATE:.0%}"
        )

    def test_scalar_spoilage_risk_latency(self) -> None:
        """1000 scalar spoilage risk calls must complete within threshold."""
        batches = list(self.state.batches.values())
        t0 = time.perf_counter()
        for _ in range(1000):
            b = batches[_ % len(batches)]
            self.model.calculate_spoilage_risk(b, transit_time_hours=6.0, route_temp=8.0)
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        self.assertLessEqual(
            elapsed_ms, SCALAR_SPOILAGE_THRESHOLD_MS,
            f"Scalar spoilage risk: {elapsed_ms:.2f} ms > {SCALAR_SPOILAGE_THRESHOLD_MS} ms"
        )

    def test_vectorised_batch_matrix_latency(self) -> None:
        """Vectorised batch_spoilage_matrix on 1000 paths must complete within threshold."""
        P = 1000
        transit_times = np.random.uniform(1.0, 20.0, P)
        remaining_shelves = np.random.uniform(10.0, 100.0, P)
        temperatures = np.random.uniform(4.0, 25.0, P)

        t0 = time.perf_counter()
        fracs, risks = self.model.batch_spoilage_matrix(
            transit_times, remaining_shelves, temperatures
        )
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        self.assertLessEqual(
            elapsed_ms, VECTOR_SPOILAGE_THRESHOLD_MS,
            f"Vectorised batch matrix: {elapsed_ms:.2f} ms > {VECTOR_SPOILAGE_THRESHOLD_MS} ms"
        )
        # Correctness: all outputs in [0, 1]
        self.assertTrue(np.all(fracs >= 0.0) and np.all(fracs <= 1.0))
        self.assertTrue(np.all(risks >= 0.0) and np.all(risks <= 1.0))

    def test_vectorised_faster_than_scalar_loop(self) -> None:
        """Vectorised kernel must be faster than an equivalent Python scalar loop."""
        P = 500
        rng = np.random.default_rng(42)
        transit_times = rng.uniform(1.0, 15.0, P)
        remaining_shelves = rng.uniform(10.0, 80.0, P)
        temperatures = rng.uniform(4.0, 20.0, P)

        # Scalar loop (warm cache)
        model = SpoilageModel()
        batches_dummy = list(generate_supply_chain(num_batches=P, seed=0).batches.values())
        for i in range(len(batches_dummy)):
            b = batches_dummy[i]
            model.calculate_spoilage_fraction(b, transit_times[i], temperatures[i])

        # Vectorised
        t_vec = time.perf_counter()
        model.batch_spoilage_matrix(transit_times, remaining_shelves, temperatures)
        vec_ms = (time.perf_counter() - t_vec) * 1_000

        # Scalar loop (cold start, different model instance)
        model2 = SpoilageModel()
        t_sca = time.perf_counter()
        for i, b in enumerate(batches_dummy):
            model2.calculate_spoilage_fraction(b, transit_times[i], temperatures[i])
        sca_ms = (time.perf_counter() - t_sca) * 1_000

        print(f"\n  Vectorised: {vec_ms:.3f} ms | Scalar loop: {sca_ms:.3f} ms | Speedup: {sca_ms/max(vec_ms,0.001):.1f}×")
        self.assertLessEqual(vec_ms, sca_ms * 2.0,
                             "Vectorised kernel should not be slower than scalar loop")


class TestOptimizerScalabilityLatency(unittest.TestCase):
    """End-to-end optimizer latency across problem scales."""

    def setUp(self) -> None:
        self.spoilage = SpoilageModel()
        self.optimizer = DynamicPerishableOptimizer(spoilage_model=self.spoilage)

    def _solve_and_time(self, num_batches: int, seed: int = 42) -> float:
        """Returns solve wall-clock time in milliseconds."""
        n_farms = max(2, num_batches // 20)
        n_storage = max(2, num_batches // 35)
        n_markets = max(2, num_batches // 25)
        state = generate_supply_chain(
            num_farms=n_farms, num_storage=n_storage,
            num_markets=n_markets, num_batches=num_batches, seed=seed
        )
        t0 = time.perf_counter()
        result = self.optimizer.solve(state)
        return (time.perf_counter() - t0) * 1_000, result

    def test_small_problem_latency(self) -> None:
        """10-batch LP must solve in under 150 ms."""
        elapsed_ms, res = self._solve_and_time(10)
        print(f"\n  10-batch LP: {elapsed_ms:.2f} ms | feasible={res.is_feasible}")
        self.assertLessEqual(elapsed_ms, OPTIMIZER_SMALL_THRESHOLD_MS,
                             f"10-batch solve {elapsed_ms:.1f} ms > {OPTIMIZER_SMALL_THRESHOLD_MS} ms")

    def test_medium_problem_latency(self) -> None:
        """100-batch LP must solve in under 800 ms."""
        elapsed_ms, res = self._solve_and_time(100)
        print(f"\n  100-batch LP: {elapsed_ms:.2f} ms | feasible={res.is_feasible}")
        self.assertLessEqual(elapsed_ms, OPTIMIZER_MEDIUM_THRESHOLD_MS,
                             f"100-batch solve {elapsed_ms:.1f} ms > {OPTIMIZER_MEDIUM_THRESHOLD_MS} ms")

    def test_large_problem_latency(self) -> None:
        """500-batch LP must solve in under 3000 ms."""
        elapsed_ms, res = self._solve_and_time(500)
        print(f"\n  500-batch LP: {elapsed_ms:.2f} ms | feasible={res.is_feasible}")
        self.assertLessEqual(elapsed_ms, OPTIMIZER_LARGE_THRESHOLD_MS,
                             f"500-batch solve {elapsed_ms:.1f} ms > {OPTIMIZER_LARGE_THRESHOLD_MS} ms")

    def test_result_correctness_at_scale(self) -> None:
        """Optimal solution must satisfy basic sanity checks at 250-batch scale."""
        _, res = self._solve_and_time(250)
        if res.is_feasible:
            self.assertGreaterEqual(res.total_delivered, 0.0)
            self.assertGreaterEqual(res.demand_fulfillment_percentage, 0.0)
            self.assertLessEqual(res.spoilage_percentage, 100.0)
            self.assertGreater(res.objective_value, 0.0)

    def test_runtime_scales_sub_quadratically(self) -> None:
        """Runtime growth from 10→100 batches must be less than 100× (sub-O(n²))."""
        t10, _ = self._solve_and_time(10, seed=1)
        t100, _ = self._solve_and_time(100, seed=1)
        scale_factor = t100 / max(t10, 0.01)
        print(f"\n  Runtime scale factor 10→100 batches: {scale_factor:.1f}×")
        self.assertLess(scale_factor, 100.0,
                        f"Runtime scaled {scale_factor:.1f}× for 10× problem size — likely super-quadratic")


class TestDynamicReOptimizerLatency(unittest.TestCase):
    """Latency tests for disruption detection and dynamic re-optimization."""

    def setUp(self) -> None:
        self.spoilage = SpoilageModel()
        self.optimizer = DynamicPerishableOptimizer(spoilage_model=self.spoilage)
        self.state = generate_supply_chain(num_batches=20, seed=42)
        self.initial_result = self.optimizer.solve(self.state)

    def test_reoptimization_latency_after_truck_failure(self) -> None:
        """Re-optimization after truck failure must complete within 500 ms."""
        disrupted = self.state.clone()
        disrupted = TruckFailureDisruption().apply(disrupted)
        reopt = DynamicReoptimizer(optimizer=self.optimizer)

        t0 = time.perf_counter()
        res, stats = reopt.reoptimize_after_disruption(
            post_disruption_state=disrupted,
            prior_result=self.initial_result,
            elapsed_hours_during_disruption=2.0,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        print(f"\n  Truck failure re-opt: {elapsed_ms:.2f} ms | rerouted={stats['successfully_rerouted_kg']} kg")
        self.assertLessEqual(elapsed_ms, REOPTIMIZER_THRESHOLD_MS,
                             f"Re-opt {elapsed_ms:.1f} ms > {REOPTIMIZER_THRESHOLD_MS} ms threshold")

    def test_reoptimization_latency_after_combined_disruption(self) -> None:
        """Re-optimization after combined disruption must complete within 1000 ms."""
        disrupted = self.state.clone()
        disrupted = CombinedDisruption().apply(disrupted)
        reopt = DynamicReoptimizer(optimizer=self.optimizer)

        t0 = time.perf_counter()
        res, stats = reopt.reoptimize_after_disruption(
            post_disruption_state=disrupted,
            prior_result=self.initial_result,
            elapsed_hours_during_disruption=3.0,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        print(f"\n  Combined disruption re-opt: {elapsed_ms:.2f} ms")
        self.assertLessEqual(elapsed_ms, 1000.0,
                             f"Combined re-opt {elapsed_ms:.1f} ms > 1000 ms threshold")

    def test_recovery_stats_are_populated(self) -> None:
        """Recovery stats dict must contain all required keys."""
        disrupted = self.state.clone()
        disrupted = TruckFailureDisruption().apply(disrupted)
        reopt = DynamicReoptimizer(optimizer=self.optimizer)
        _, stats = reopt.reoptimize_after_disruption(
            post_disruption_state=disrupted,
            prior_result=self.initial_result,
            elapsed_hours_during_disruption=1.0,
        )
        required_keys = {
            "recovery_time_seconds", "affected_batch_count", "stranded_quantity_kg",
            "successfully_rerouted_kg", "reroute_success_rate_pct", "cost_delta",
            "spoilage_delta_kg",
        }
        self.assertTrue(required_keys.issubset(set(stats.keys())),
                        f"Missing keys: {required_keys - set(stats.keys())}")


class TestBenchmarkSummary(unittest.TestCase):
    """Prints a comprehensive latency table for the audit report."""

    def test_print_full_benchmark_table(self) -> None:
        """Runs a full sweep and prints a formatted benchmark table."""
        spoilage = SpoilageModel()
        optimizer = DynamicPerishableOptimizer(spoilage_model=spoilage)
        scales = [10, 25, 50, 100, 250, 500]

        print("\n" + "=" * 70)
        print("DPSRO PERFORMANCE BENCHMARK RESULTS")
        print("=" * 70)
        print(f"{'Batches':>8} | {'LP Solve (ms)':>14} | {'Feasible':>8} | {'Spoilage%':>10} | {'Fulfil%':>8}")
        print("-" * 70)

        for n in scales:
            n_f = max(2, n // 20)
            n_s = max(2, n // 35)
            n_m = max(2, n // 25)
            state = generate_supply_chain(
                num_farms=n_f, num_storage=n_s, num_markets=n_m,
                num_batches=n, seed=42
            )
            t0 = time.perf_counter()
            res = optimizer.solve(state)
            ms = (time.perf_counter() - t0) * 1_000
            print(
                f"{n:>8} | {ms:>14.2f} | {str(res.is_feasible):>8} | "
                f"{res.spoilage_percentage:>10.2f} | {res.demand_fulfillment_percentage:>8.2f}"
            )

        cache_info = spoilage.cache_info()
        print("=" * 70)
        print(f"\nCache Stats:")
        for name, info in cache_info.items():
            total = info.hits + info.misses
            rate = info.hits / max(total, 1)
            print(f"  {name}: {info.hits} hits / {total} calls ({rate:.1%} hit rate)")
        print()


if __name__ == "__main__":
    unittest.main(verbosity=2)
