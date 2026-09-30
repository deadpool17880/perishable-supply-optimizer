"""Synthetic Supply-Chain Network and Batch Data Generator.

Generates reproducible, realistic perishable logistics instances across arbitrary scales
(10 to 500+ batches) with realistic spatial geometry, transit times, shelf lives, and capacities.
"""

import math
import random
from typing import Dict, List, Tuple, Optional
from src.models import (
    Farm,
    Batch,
    StorageFacility,
    Market,
    TransportRoute,
    Truck,
    SupplyChainState,
)


COMMODITY_PROFILES = {
    "Berries": {"shelf_life_min": 24.0, "shelf_life_max": 48.0, "ideal_temp": 2.0},
    "Leafy Greens": {"shelf_life_min": 36.0, "shelf_life_max": 72.0, "ideal_temp": 4.0},
    "Tomatoes": {"shelf_life_min": 72.0, "shelf_life_max": 144.0, "ideal_temp": 12.0},
    "Peaches": {"shelf_life_min": 48.0, "shelf_life_max": 96.0, "ideal_temp": 4.0},
    "Apples": {"shelf_life_min": 168.0, "shelf_life_max": 336.0, "ideal_temp": 4.0},
}


def euclidean_distance(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
    return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)


def generate_supply_chain(
    num_farms: int = 3,
    num_storage: int = 2,
    num_markets: int = 3,
    num_batches: int = 10,
    num_trucks: int = 5,
    seed: int = 42,
    area_size_km: float = 300.0,
    avg_speed_kmh: float = 50.0,
) -> SupplyChainState:
    """Generates a complete, consistent supply chain network and batch state.

    Spatial layout:
    - Farms generated primarily on the western/northern perimeter (rural source)
    - Cold storage facilities situated strategically in intermediate hub zones
    - Markets situated towards urban centers (eastern/southern zones)
    """
    rng = random.Random(seed)

    # 1. Generate Farms
    farms: Dict[str, Farm] = {}
    for i in range(num_farms):
        fid = f"F{i+1}"
        loc = (rng.uniform(10.0, area_size_km * 0.35), rng.uniform(20.0, area_size_km * 0.9))
        farms[fid] = Farm(
            farm_id=fid,
            location=loc,
            available_quantity=0.0,  # Will be aggregated from batches
            harvest_time=0.0,
        )

    # 2. Generate Storage Facilities (Hubs)
    storages: Dict[str, StorageFacility] = {}
    for j in range(num_storage):
        sid = f"S{j+1}"
        loc = (
            rng.uniform(area_size_km * 0.4, area_size_km * 0.65),
            rng.uniform(area_size_km * 0.25, area_size_km * 0.75),
        )
        storages[sid] = StorageFacility(
            storage_id=sid,
            location=loc,
            capacity=round(rng.uniform(8000.0, 20000.0), -2),
            current_load=0.0,
            temperature_capacity=4.0,
            available=True,
        )

    # 3. Generate Markets (Demand sinks)
    markets: Dict[str, Market] = {}
    total_est_supply = num_batches * 1200.0  # approximate scale
    demand_per_market = round(total_est_supply / max(1, num_markets), -1)

    for k in range(num_markets):
        mid = f"M{k+1}"
        loc = (rng.uniform(area_size_km * 0.7, area_size_km * 0.95), rng.uniform(20.0, area_size_km * 0.9))
        # Add variance to demand
        m_demand = max(500.0, round(rng.uniform(0.7, 1.3) * demand_per_market, -1))
        priority = round(rng.choice([1.0, 1.2, 1.5]), 1)
        markets[mid] = Market(
            market_id=mid,
            location=loc,
            demand=m_demand,
            priority=priority,
        )

    # 4. Generate Batches
    batches: Dict[str, Batch] = {}
    farm_ids = list(farms.keys())
    commodities = list(COMMODITY_PROFILES.keys())

    for b in range(num_batches):
        bid = f"B{b+1:02d}"
        fid = farm_ids[b % len(farm_ids)]
        comm = rng.choice(commodities)
        prof = COMMODITY_PROFILES[comm]

        initial_shelf = rng.uniform(prof["shelf_life_min"], prof["shelf_life_max"])
        # Some batches are freshly harvested, others have spent some hours at farm
        age_hours = rng.uniform(0.0, initial_shelf * 0.4)
        rem_shelf = max(4.0, initial_shelf - age_hours)

        qty = round(rng.uniform(500.0, 2500.0), -1)
        ambient_temp = round(rng.uniform(18.0, 30.0), 1)

        batches[bid] = Batch(
            batch_id=bid,
            farm_id=fid,
            commodity=comm,
            quantity=qty,
            harvest_time=round(age_hours, 1),
            initial_shelf_life_hours=round(initial_shelf, 1),
            remaining_shelf_life_hours=round(rem_shelf, 1),
            temperature=ambient_temp,
            quality_level=round(rem_shelf / initial_shelf, 2),
        )
        farms[fid].available_quantity += qty

    # 5. Generate Transport Routes
    # Routes exist between:
    # (a) Farms -> Storage facilities
    # (b) Storage facilities -> Markets
    # (c) Direct Farms -> Markets (express / high-cost routes)
    routes: Dict[str, TransportRoute] = {}
    route_counter = 1

    # (a) Farm -> Storage
    for fid, farm in farms.items():
        for sid, storage in storages.items():
            dist = euclidean_distance(farm.location, storage.location)
            tt = dist / avg_speed_kmh
            cost_per_kg = round(0.0008 * dist + rng.uniform(0.02, 0.05), 4)
            rid = f"R_F{fid[-1]}_S{sid[-1]}"
            routes[rid] = TransportRoute(
                route_id=rid,
                source=fid,
                destination=sid,
                distance_km=round(dist, 1),
                travel_time_hours=round(tt, 2),
                transport_cost_per_unit=cost_per_kg,
                available=True,
                risk_factor=round(rng.uniform(0.05, 0.20), 2),
            )
            route_counter += 1

    # (b) Storage -> Market
    for sid, storage in storages.items():
        for mid, market in markets.items():
            dist = euclidean_distance(storage.location, market.location)
            tt = dist / avg_speed_kmh
            cost_per_kg = round(0.0008 * dist + rng.uniform(0.02, 0.05), 4)
            rid = f"R_S{sid[-1]}_M{mid[-1]}"
            routes[rid] = TransportRoute(
                route_id=rid,
                source=sid,
                destination=mid,
                distance_km=round(dist, 1),
                travel_time_hours=round(tt, 2),
                transport_cost_per_unit=cost_per_kg,
                available=True,
                risk_factor=round(rng.uniform(0.05, 0.20), 2),
            )
            route_counter += 1

    # (c) Direct Farm -> Market (Express routes, ~1.5x cost, slightly higher road risk)
    for fid, farm in farms.items():
        for mid, market in markets.items():
            dist = euclidean_distance(farm.location, market.location)
            tt = dist / (avg_speed_kmh * 1.1)  # Faster express transit
            cost_per_kg = round(0.0012 * dist + rng.uniform(0.05, 0.10), 4)
            rid = f"R_DIR_F{fid[-1]}_M{mid[-1]}"
            routes[rid] = TransportRoute(
                route_id=rid,
                source=fid,
                destination=mid,
                distance_km=round(dist, 1),
                travel_time_hours=round(tt, 2),
                transport_cost_per_unit=cost_per_kg,
                available=True,
                risk_factor=round(rng.uniform(0.12, 0.35), 2),
            )
            route_counter += 1

    # 6. Generate Trucks / Fleet
    trucks: Dict[str, Truck] = {}
    for t in range(num_trucks):
        tid = f"T{t+1}"
        trucks[tid] = Truck(
            truck_id=tid,
            capacity=round(rng.choice([4000.0, 6000.0, 8000.0, 10000.0]), -2),
            available=True,
            current_location=rng.choice(list(farms.keys()) + list(storages.keys())),
            travel_time_factor=round(rng.choice([1.0, 1.0, 1.0, 1.15]), 2),
        )

    return SupplyChainState(
        farms=farms,
        storage_facilities=storages,
        markets=markets,
        routes=routes,
        batches=batches,
        trucks=trucks,
        current_time_hours=0.0,
    )
