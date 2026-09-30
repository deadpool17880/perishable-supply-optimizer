"""Mathematical Spoilage & Perishability Kinetics Engine.

Implements biophysical deterioration kinetics as a function of:
1. Remaining shelf life vs transit duration
2. Temperature deviations from optimal cold-chain reference
3. Unanticipated transit delays / congestion
"""

import math
from typing import Dict, Any, Optional
from src.models import Batch


class SpoilageModel:
    """Interpretable kinetic spoilage and perishability risk estimator."""

    def __init__(
        self,
        alpha: float = 2.5,
        beta: float = 0.8,
        gamma: float = 0.5,
        reference_temperature: float = 4.0,
        spoilage_threshold: float = 0.85,
    ):
        """
        Args:
            alpha: Shelf life decay sensitivity factor.
            beta: Temperature penalty coefficient.
            gamma: Transit delay penalty coefficient.
            reference_temperature: Baseline cold-chain temperature (°C).
            spoilage_threshold: Quality loss threshold for classifying unsellable spoilage.
        """
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.reference_temperature = float(reference_temperature)
        self.spoilage_threshold = float(spoilage_threshold)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SpoilageModel":
        cfg = config.get("spoilage", {})
        return cls(
            alpha=cfg.get("alpha", 2.5),
            beta=cfg.get("beta", 0.8),
            gamma=cfg.get("gamma", 0.5),
            reference_temperature=cfg.get("reference_temperature", 4.0),
            spoilage_threshold=cfg.get("spoilage_threshold", 0.85),
        )

    def calculate_temperature_penalty(self, actual_temp: float) -> float:
        """Penalty for temperature elevation above reference cold-chain conditions."""
        temp_diff = actual_temp - self.reference_temperature
        if temp_diff <= 0.0:
            return 0.0
        # Normalized degree elevation
        ref = max(1.0, self.reference_temperature)
        return temp_diff / ref

    def calculate_spoilage_risk(
        self,
        batch: Batch,
        transit_time_hours: float,
        route_temp: Optional[float] = None,
        delay_hours: float = 0.0,
    ) -> float:
        """Computes normalized spoilage risk score in [0.0, 1.0].

        Formulation:
        risk = sigmoid(alpha * (total_transit / remaining_shelf_life)
                       + beta * temp_penalty
                       + gamma * (delay / (transit_time + 1e-3)) - bias)
        """
        effective_temp = route_temp if route_temp is not None else batch.temperature
        temp_penalty = self.calculate_temperature_penalty(effective_temp)

        remaining_shelf = max(0.01, batch.remaining_shelf_life_hours)
        total_time = transit_time_hours + delay_hours

        # Ratio of time required to remaining shelf life
        time_ratio = total_time / remaining_shelf

        # Delay factor relative to nominal time
        delay_factor = delay_hours / (transit_time_hours + 1.0) if transit_time_hours > 0 else 0.0

        # Logit calculation
        # Centered such that when time_ratio = 1.0 and temp is at reference, risk is high (~0.85)
        logit = (
            self.alpha * (time_ratio - 0.7)
            + self.beta * temp_penalty
            + self.gamma * delay_factor
        )

        # Numerically stable sigmoid
        if logit >= 20.0:
            return 1.0
        elif logit <= -20.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-logit))

    def calculate_spoilage_fraction(
        self,
        batch: Batch,
        transit_time_hours: float,
        route_temp: Optional[float] = None,
        delay_hours: float = 0.0,
    ) -> float:
        """Calculates expected physical quantity fraction lost to spoilage [0.0, 1.0].

        If transit duration exceeds remaining shelf life, 100% of produce spoils.
        Otherwise, spoilage scales with accelerated kinetic decay.
        """
        remaining_shelf = batch.remaining_shelf_life_hours
        total_time = transit_time_hours + delay_hours

        if total_time >= remaining_shelf:
            return 1.0

        effective_temp = route_temp if route_temp is not None else batch.temperature
        temp_penalty = self.calculate_temperature_penalty(effective_temp)

        # Arrhenius-inspired accelerated deterioration: (1 + beta * temp_penalty)
        thermal_acceleration = 1.0 + self.beta * temp_penalty
        effective_elapsed = total_time * thermal_acceleration

        if effective_elapsed >= remaining_shelf:
            return 1.0

        # Base fractional loss: power curve reflecting shelf-life exhaustion
        fraction = (effective_elapsed / remaining_shelf) ** 1.8
        return min(1.0, max(0.0, fraction))

    def update_batch_shelf_life(
        self,
        batch: Batch,
        elapsed_hours: float,
        storage_temp: Optional[float] = None,
    ) -> Batch:
        """Returns a cloned Batch updated for elapsed storage/transport time."""
        effective_temp = storage_temp if storage_temp is not None else batch.temperature
        temp_penalty = self.calculate_temperature_penalty(effective_temp)
        acceleration = 1.0 + self.beta * temp_penalty

        shelf_loss = elapsed_hours * acceleration
        new_remaining = max(0.0, batch.remaining_shelf_life_hours - shelf_loss)

        # Quality degrades proportionally
        new_quality = max(0.0, batch.quality_level * (new_remaining / max(0.1, batch.initial_shelf_life_hours)))

        return Batch(
            batch_id=batch.batch_id,
            farm_id=batch.farm_id,
            commodity=batch.commodity,
            quantity=batch.quantity,
            harvest_time=batch.harvest_time,
            initial_shelf_life_hours=batch.initial_shelf_life_hours,
            remaining_shelf_life_hours=new_remaining,
            temperature=effective_temp,
            quality_level=new_quality,
        )
