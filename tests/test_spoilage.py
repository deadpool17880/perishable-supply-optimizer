"""Unit tests for Spoilage Kinetics Model."""

import pytest
from src.models import Batch
from src.spoilage_model import SpoilageModel


@pytest.fixture
def default_spoilage_model():
    return SpoilageModel(
        alpha=2.5,
        beta=0.8,
        gamma=0.5,
        reference_temperature=4.0,
        spoilage_threshold=0.85,
    )


@pytest.fixture
def fresh_batch():
    return Batch(
        batch_id="B_TEST",
        farm_id="F1",
        commodity="Strawberries",
        quantity=1000.0,
        harvest_time=0.0,
        initial_shelf_life_hours=48.0,
        remaining_shelf_life_hours=48.0,
        temperature=4.0,
        quality_level=1.0,
    )


def test_temperature_penalty_zero_at_or_below_reference(default_spoilage_model):
    assert default_spoilage_model.calculate_temperature_penalty(4.0) == 0.0
    assert default_spoilage_model.calculate_temperature_penalty(2.0) == 0.0


def test_temperature_penalty_positive_above_reference(default_spoilage_model):
    penalty = default_spoilage_model.calculate_temperature_penalty(12.0)
    assert penalty == (12.0 - 4.0) / 4.0  # 2.0


def test_spoilage_fraction_one_when_transit_exceeds_shelf_life(default_spoilage_model, fresh_batch):
    # Remaining shelf life is 48h, transit is 50h
    frac = default_spoilage_model.calculate_spoilage_fraction(fresh_batch, transit_time_hours=50.0)
    assert frac == 1.0


def test_spoilage_fraction_low_for_short_transit_at_ref_temp(default_spoilage_model, fresh_batch):
    # 2 hours transit on a 48h batch at 4°C
    frac = default_spoilage_model.calculate_spoilage_fraction(fresh_batch, transit_time_hours=2.0, route_temp=4.0)
    assert 0.0 <= frac < 0.05


def test_higher_temperature_accelerates_spoilage(default_spoilage_model, fresh_batch):
    frac_cold = default_spoilage_model.calculate_spoilage_fraction(fresh_batch, transit_time_hours=12.0, route_temp=4.0)
    frac_hot = default_spoilage_model.calculate_spoilage_fraction(fresh_batch, transit_time_hours=12.0, route_temp=28.0)
    assert frac_hot > frac_cold


def test_update_batch_shelf_life(default_spoilage_model, fresh_batch):
    updated = default_spoilage_model.update_batch_shelf_life(fresh_batch, elapsed_hours=10.0, storage_temp=4.0)
    assert updated.remaining_shelf_life_hours == 38.0
    assert updated.quality_level < fresh_batch.quality_level
