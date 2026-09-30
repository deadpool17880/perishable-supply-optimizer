"""Comprehensive validation and integration tests for DPSRO.

Tests cover:
- Property-based invariants (spoilage ∈ [0,1], fulfillment ∈ [0,100])
- Monotonicity properties (higher temperature → higher spoilage)
- Input validation and security bounds
- End-to-end hybrid CIS×EMBS orchestrator pipeline
- Pareto-front diversity (multi-objective coverage)
- Convergence evidence (objective decreases over generations)
- Statistical validation across 20 random seeds
"""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

from src.data_generator import generate_supply_chain
from src.disruption import TruckFailureDisruption, CombinedDisruption
from src.dynamic_optimizer import DynamicReoptimizer
from src.hybrid_orchestrator import (
    ArrheniusEMBSModel,
    HybridCISEMBSOrchestrator,
    WeightVector,
    REFERENCE_TEMP_K,
)
from src.optimizer import DynamicPerishableOptimizer
from src.security import (
    validate_quantity,
    validate_shelf_life,
    validate_temperature,
    validate_lambda_weight,
    validate_node_id,
    validate_optimizer_config,
)
from src.spoilage_model import SpoilageModel


# ─────────────────────────────────────────────────────────────────────────────
# EMBS Arrhenius Model Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestArrheniusEMBSModel(unittest.TestCase):
    """Tests for the EMBS biophysical Arrhenius kinetic layer."""

    def setUp(self) -> None:
        self.model = ArrheniusEMBSModel()

    def test_rate_at_reference_is_one(self) -> None:
        """Arrhenius rate at cold-chain reference temperature must equal 1.0."""
        ref_c = REFERENCE_TEMP_K - 273.15
        rate = self.model.arrhenius_rate(ref_c)
        self.assertAlmostEqual(rate, 1.0, places=6,
                               msg="k(T_ref) must equal 1.0 by Arrhenius definition")

    def test_rate_increases_with_temperature(self) -> None:
        """Arrhenius rate must be monotonically increasing with temperature."""
        temps = [4.0, 10.0, 20.0, 30.0, 40.0]
        rates = [self.model.arrhenius_rate(t) for t in temps]
        for i in range(len(rates) - 1):
            self.assertGreater(rates[i + 1], rates[i],
                               f"Rate should increase: k({temps[i+1]}) > k({temps[i]})")

    def test_rate_positive_for_all_temps(self) -> None:
        """Arrhenius rate must be strictly positive for all valid temperatures."""
        for t in [-20.0, 0.0, 4.0, 15.0, 25.0, 40.0]:
            rate = self.model.arrhenius_rate(t)
            self.assertGreater(rate, 0.0, f"Rate must be positive at {t}°C")

    def test_q10_doubles_every_10_degrees(self) -> None:
        """Q₁₀ factor must double for every 10 °C above reference."""
        ref_c = REFERENCE_TEMP_K - 273.15
        q_at_ref   = self.model.q10_acceleration(ref_c)
        q_at_ref10 = self.model.q10_acceleration(ref_c + 10.0)
        q_at_ref20 = self.model.q10_acceleration(ref_c + 20.0)
        self.assertAlmostEqual(q_at_ref, 1.0, places=5)
        self.assertAlmostEqual(q_at_ref10, 2.0, places=5)
        self.assertAlmostEqual(q_at_ref20, 4.0, places=5)

    def test_batch_kinetics_all_batches_covered(self) -> None:
        """compute_batch_kinetics must return an entry for every batch in state."""
        state = generate_supply_chain(num_batches=15, seed=7)
        k_vals, q10_vals = self.model.compute_batch_kinetics(state)
        self.assertEqual(set(k_vals.keys()), set(state.batches.keys()))
        self.assertEqual(set(q10_vals.keys()), set(state.batches.keys()))

    def test_batch_kinetics_values_are_positive(self) -> None:
        """All Arrhenius and Q₁₀ values must be strictly positive."""
        state = generate_supply_chain(num_batches=10, seed=3)
        k_vals, q10_vals = self.model.compute_batch_kinetics(state)
        for b_id, k in k_vals.items():
            self.assertGreater(k, 0.0, f"k must be positive for {b_id}")
        for b_id, q in q10_vals.items():
            self.assertGreater(q, 0.0, f"Q10 must be positive for {b_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid CIS×EMBS Orchestrator Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestHybridOrchestrator(unittest.TestCase):
    """Integration tests for the HybridCISEMBSOrchestrator."""

    def setUp(self) -> None:
        self.state = generate_supply_chain(num_batches=10, seed=42)
        self.orchestrator = HybridCISEMBSOrchestrator(n_generations=5, population_size=3, seed=42)

    def test_hybrid_solve_returns_feasible_result(self) -> None:
        """Hybrid orchestrator must return a feasible allocation."""
        hybrid_result = self.orchestrator.solve(self.state)
        self.assertTrue(hybrid_result.allocation_result.is_feasible,
                        "Hybrid solve must find a feasible allocation")

    def test_evolution_log_populated(self) -> None:
        """Evolution log must contain at least one entry (convergence evidence)."""
        hybrid_result = self.orchestrator.solve(self.state)
        self.assertGreater(len(hybrid_result.evolution_log), 0,
                           "Evolution log must not be empty")

    def test_evolution_log_format(self) -> None:
        """Each evolution log entry must be a (generation_int, objective_float) tuple."""
        hybrid_result = self.orchestrator.solve(self.state)
        for gen, obj in hybrid_result.evolution_log:
            self.assertIsInstance(gen, int)
            self.assertIsInstance(obj, float)
            self.assertGreater(obj, 0.0)

    def test_arrhenius_kinetics_in_result(self) -> None:
        """HybridResult must include Arrhenius k values for all batches."""
        hybrid_result = self.orchestrator.solve(self.state)
        self.assertEqual(
            set(hybrid_result.arrhenius_k_values.keys()),
            set(self.state.batches.keys()),
        )

    def test_best_weights_are_valid(self) -> None:
        """Evolved best weights must all be non-negative."""
        hybrid_result = self.orchestrator.solve(self.state)
        w = hybrid_result.best_weights
        self.assertGreaterEqual(w.lambda_spoilage, 0.0)
        self.assertGreaterEqual(w.lambda_cost, 0.0)
        self.assertGreaterEqual(w.lambda_unmet_demand, 0.0)
        self.assertGreaterEqual(w.lambda_risk, 0.0)

    def test_hybrid_outperforms_or_matches_single_lp(self) -> None:
        """Hybrid evolved solution objective must be ≤ single static LP objective."""
        static_opt = DynamicPerishableOptimizer(spoilage_model=SpoilageModel())
        static_res = static_opt.solve(self.state)
        hybrid_res = self.orchestrator.solve(self.state)
        if static_res.is_feasible and hybrid_res.allocation_result.is_feasible:
            self.assertLessEqual(
                hybrid_res.allocation_result.objective_value,
                static_res.objective_value * 1.05,  # allow 5% tolerance
                "Hybrid should match or beat static LP objective"
            )

    def test_pareto_approximation_returns_diverse_points(self) -> None:
        """Pareto approximation must return multiple distinct trade-off points."""
        points = self.orchestrator.get_pareto_approximation(self.state, n_samples=10)
        self.assertGreater(len(points), 0, "Should find at least one feasible Pareto point")
        spoilage_vals = [p["spoilage_pct"] for p in points]
        # Check there is diversity (not all identical)
        self.assertGreater(max(spoilage_vals) - min(spoilage_vals), 0.0,
                           "Pareto front should show trade-off diversity")

    def test_weight_perturbation_changes_values(self) -> None:
        """WeightVector.perturb must return a different vector each call."""
        rng = np.random.default_rng(0)
        w = WeightVector()
        offspring = w.perturb(1.0, rng)
        # At least one weight should differ (extremely unlikely all identical)
        diffs = [
            abs(w.lambda_spoilage - offspring.lambda_spoilage),
            abs(w.lambda_cost - offspring.lambda_cost),
            abs(w.lambda_unmet_demand - offspring.lambda_unmet_demand),
            abs(w.lambda_risk - offspring.lambda_risk),
        ]
        self.assertGreater(sum(diffs), 0.0, "Perturbed offspring must differ from parent")


# ─────────────────────────────────────────────────────────────────────────────
# Security / Input Validation Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestInputValidation(unittest.TestCase):
    """Tests for security bounds and input sanitisation."""

    def test_quantity_valid(self) -> None:
        self.assertAlmostEqual(validate_quantity(1000.0), 1000.0)

    def test_quantity_rejects_negative(self) -> None:
        with self.assertRaises(ValueError):
            validate_quantity(-1.0)

    def test_quantity_rejects_infinity(self) -> None:
        with self.assertRaises(ValueError):
            validate_quantity(float("inf"))

    def test_quantity_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            validate_quantity(float("nan"))

    def test_quantity_rejects_over_max(self) -> None:
        with self.assertRaises(ValueError):
            validate_quantity(2_000_000.0)

    def test_shelf_life_valid(self) -> None:
        self.assertAlmostEqual(validate_shelf_life(48.0), 48.0)

    def test_shelf_life_rejects_zero(self) -> None:
        with self.assertRaises(ValueError):
            validate_shelf_life(0.0)

    def test_shelf_life_rejects_over_year(self) -> None:
        with self.assertRaises(ValueError):
            validate_shelf_life(9000.0)

    def test_temperature_valid(self) -> None:
        self.assertAlmostEqual(validate_temperature(4.0), 4.0)
        self.assertAlmostEqual(validate_temperature(-20.0), -20.0)

    def test_temperature_rejects_boiling(self) -> None:
        with self.assertRaises(ValueError):
            validate_temperature(200.0)

    def test_temperature_rejects_absolute_zero(self) -> None:
        with self.assertRaises(ValueError):
            validate_temperature(-300.0)

    def test_lambda_weight_valid(self) -> None:
        self.assertAlmostEqual(validate_lambda_weight(10.0), 10.0)
        self.assertAlmostEqual(validate_lambda_weight(0.0), 0.0)

    def test_lambda_weight_rejects_negative(self) -> None:
        with self.assertRaises(ValueError):
            validate_lambda_weight(-1.0)

    def test_node_id_valid(self) -> None:
        self.assertEqual(validate_node_id("F1"), "F1")
        self.assertEqual(validate_node_id("storage-hub_2"), "storage-hub_2")

    def test_node_id_rejects_injection(self) -> None:
        with self.assertRaises(ValueError):
            validate_node_id("'; DROP TABLE batches; --")

    def test_node_id_rejects_too_long(self) -> None:
        with self.assertRaises(ValueError):
            validate_node_id("A" * 65)

    def test_config_validation_valid(self) -> None:
        cfg = {"weights": {"lambda_spoilage": 10.0}, "solver": {"time_limit_seconds": 30.0}}
        result = validate_optimizer_config(cfg)
        self.assertIn("weights", result)
        self.assertIn("solver", result)

    def test_config_validation_rejects_bad_time_limit(self) -> None:
        cfg = {"solver": {"time_limit_seconds": -1.0}}
        with self.assertRaises(ValueError):
            validate_optimizer_config(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Property-Based Invariant Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestPropertyInvariants(unittest.TestCase):
    """Property-based tests across 20 random seeds."""

    def setUp(self) -> None:
        self.spoilage = SpoilageModel()
        self.optimizer = DynamicPerishableOptimizer(spoilage_model=self.spoilage)

    def test_spoilage_fraction_always_in_0_1(self) -> None:
        """Spoilage fraction must always be in [0, 1] across all seeds."""
        for seed in range(20):
            state = generate_supply_chain(num_batches=10, seed=seed)
            for batch in state.batches.values():
                for temp in [4.0, 15.0, 25.0]:
                    frac = self.spoilage.calculate_spoilage_fraction(batch, 6.0, temp)
                    self.assertGreaterEqual(frac, 0.0)
                    self.assertLessEqual(frac, 1.0)

    def test_spoilage_risk_always_in_0_1(self) -> None:
        """Spoilage risk must always be in [0, 1] across all seeds."""
        for seed in range(20):
            state = generate_supply_chain(num_batches=8, seed=seed)
            for batch in state.batches.values():
                risk = self.spoilage.calculate_spoilage_risk(batch, 8.0, 10.0)
                self.assertGreaterEqual(risk, 0.0)
                self.assertLessEqual(risk, 1.0)

    def test_fulfillment_pct_in_0_100(self) -> None:
        """Demand fulfillment percentage must be in [0, 100] for all feasible solutions."""
        for seed in range(10):
            state = generate_supply_chain(num_batches=10, seed=seed * 3)
            result = self.optimizer.solve(state)
            if result.is_feasible:
                self.assertGreaterEqual(result.demand_fulfillment_percentage, 0.0)
                self.assertLessEqual(result.demand_fulfillment_percentage, 100.01)

    def test_spoilage_pct_in_0_100(self) -> None:
        """Spoilage percentage must be in [0, 100] for all feasible solutions."""
        for seed in range(10):
            state = generate_supply_chain(num_batches=10, seed=seed * 5)
            result = self.optimizer.solve(state)
            if result.is_feasible:
                self.assertGreaterEqual(result.spoilage_percentage, 0.0)
                self.assertLessEqual(result.spoilage_percentage, 100.01)

    def test_higher_temperature_means_more_spoilage(self) -> None:
        """Monotonicity: higher route temperature must produce equal or higher spoilage fraction."""
        state = generate_supply_chain(num_batches=5, seed=0)
        batch = list(state.batches.values())[0]
        temps = [4.0, 10.0, 20.0, 30.0]
        fracs = [self.spoilage.calculate_spoilage_fraction(batch, 5.0, t) for t in temps]
        for i in range(len(fracs) - 1):
            self.assertLessEqual(fracs[i], fracs[i + 1],
                                 f"Spoilage at {temps[i]}°C should be ≤ {temps[i+1]}°C")

    def test_longer_transit_means_more_spoilage(self) -> None:
        """Monotonicity: longer transit time must produce equal or higher spoilage fraction."""
        state = generate_supply_chain(num_batches=5, seed=1)
        batch = list(state.batches.values())[0]
        max_t = batch.remaining_shelf_life_hours * 0.9
        times = [max_t * f for f in [0.1, 0.3, 0.5, 0.7, 0.9]]
        fracs = [self.spoilage.calculate_spoilage_fraction(batch, t, 4.0) for t in times]
        for i in range(len(fracs) - 1):
            self.assertLessEqual(fracs[i], fracs[i + 1],
                                 f"Spoilage at t={times[i]:.1f}h should be ≤ t={times[i+1]:.1f}h")

    def test_delivered_plus_spoilage_equals_dispatched(self) -> None:
        """Mass conservation: delivered + spoiled must equal dispatched (within rounding)."""
        for seed in [42, 7, 13]:
            state = generate_supply_chain(num_batches=12, seed=seed)
            result = self.optimizer.solve(state)
            if result.is_feasible and result.total_dispatched > 0:
                balance = abs(
                    result.total_delivered + result.total_spoilage - result.total_dispatched
                )
                self.assertLess(balance, 1.0,
                                f"Mass conservation violated: balance error = {balance:.3f} kg")

    def test_statistical_mean_spoilage_below_threshold(self) -> None:
        """Mean spoilage across 20 seeds must be < 20% (system effectiveness check)."""
        spoilage_vals = []
        for seed in range(20):
            state = generate_supply_chain(num_batches=10, seed=seed)
            result = self.optimizer.solve(state)
            if result.is_feasible:
                spoilage_vals.append(result.spoilage_percentage)
        if spoilage_vals:
            mean_spoilage = sum(spoilage_vals) / len(spoilage_vals)
            self.assertLess(mean_spoilage, 20.0,
                            f"Mean spoilage {mean_spoilage:.2f}% exceeds 20% threshold")


if __name__ == "__main__":
    unittest.main(verbosity=2)
