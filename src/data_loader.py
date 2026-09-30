"""CSV Data Loader and Exporter for DPSRO."""

import os
import csv
from typing import Optional, Dict
from src.models import (
    Farm,
    Batch,
    StorageFacility,
    Market,
    TransportRoute,
    Truck,
    SupplyChainState,
)


def export_state_to_csv(state: SupplyChainState, output_dir: str) -> None:
    """Exports state entities to standard CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Farms
    with open(os.path.join(output_dir, "farms.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["farm_id", "location_x", "location_y", "available_quantity", "harvest_time"]
        )
        writer.writeheader()
        for farm in state.farms.values():
            writer.writerow(farm.to_dict())

    # 2. Storage
    with open(os.path.join(output_dir, "storage.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "storage_id",
                "location_x",
                "location_y",
                "capacity",
                "current_load",
                "temperature_capacity",
                "available",
            ],
        )
        writer.writeheader()
        for st in state.storage_facilities.values():
            writer.writerow(st.to_dict())

    # 3. Markets
    with open(os.path.join(output_dir, "markets.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["market_id", "location_x", "location_y", "demand", "priority"]
        )
        writer.writeheader()
        for m in state.markets.values():
            writer.writerow(m.to_dict())

    # 4. Batches
    with open(os.path.join(output_dir, "batches.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch_id",
                "farm_id",
                "commodity",
                "quantity",
                "harvest_time",
                "initial_shelf_life_hours",
                "remaining_shelf_life_hours",
                "temperature",
                "quality_level",
            ],
        )
        writer.writeheader()
        for b in state.batches.values():
            writer.writerow(b.to_dict())

    # 5. Routes
    with open(os.path.join(output_dir, "routes.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "route_id",
                "source",
                "destination",
                "distance_km",
                "travel_time_hours",
                "transport_cost_per_unit",
                "available",
                "risk_factor",
            ],
        )
        writer.writeheader()
        for r in state.routes.values():
            writer.writerow(r.to_dict())


def load_state_from_csv(data_dir: str) -> SupplyChainState:
    """Loads supply chain entities from a directory of CSV files."""
    farms: Dict[str, Farm] = {}
    with open(os.path.join(data_dir, "farms.csv"), "r") as f:
        for row in csv.DictReader(f):
            farms[row["farm_id"]] = Farm(
                farm_id=row["farm_id"],
                location=(float(row["location_x"]), float(row["location_y"])),
                available_quantity=float(row["available_quantity"]),
                harvest_time=float(row.get("harvest_time", 0.0)),
            )

    storage: Dict[str, StorageFacility] = {}
    with open(os.path.join(data_dir, "storage.csv"), "r") as f:
        for row in csv.DictReader(f):
            storage[row["storage_id"]] = StorageFacility(
                storage_id=row["storage_id"],
                location=(float(row["location_x"]), float(row["location_y"])),
                capacity=float(row["capacity"]),
                current_load=float(row.get("current_load", 0.0)),
                temperature_capacity=float(row.get("temperature_capacity", 4.0)),
                available=row.get("available", "True").lower() == "true",
            )

    markets: Dict[str, Market] = {}
    with open(os.path.join(data_dir, "markets.csv"), "r") as f:
        for row in csv.DictReader(f):
            markets[row["market_id"]] = Market(
                market_id=row["market_id"],
                location=(float(row["location_x"]), float(row["location_y"])),
                demand=float(row["demand"]),
                priority=float(row.get("priority", 1.0)),
            )

    batches: Dict[str, Batch] = {}
    with open(os.path.join(data_dir, "batches.csv"), "r") as f:
        for row in csv.DictReader(f):
            batches[row["batch_id"]] = Batch(
                batch_id=row["batch_id"],
                farm_id=row["farm_id"],
                commodity=row["commodity"],
                quantity=float(row["quantity"]),
                harvest_time=float(row["harvest_time"]),
                initial_shelf_life_hours=float(row["initial_shelf_life_hours"]),
                remaining_shelf_life_hours=float(row["remaining_shelf_life_hours"]),
                temperature=float(row["temperature"]),
                quality_level=float(row.get("quality_level", 1.0)),
            )

    routes: Dict[str, TransportRoute] = {}
    with open(os.path.join(data_dir, "routes.csv"), "r") as f:
        for row in csv.DictReader(f):
            routes[row["route_id"]] = TransportRoute(
                route_id=row["route_id"],
                source=row["source"],
                destination=row["destination"],
                distance_km=float(row["distance_km"]),
                travel_time_hours=float(row["travel_time_hours"]),
                transport_cost_per_unit=float(row["transport_cost_per_unit"]),
                available=row.get("available", "True").lower() == "true",
                risk_factor=float(row.get("risk_factor", 0.1)),
            )

    return SupplyChainState(
        farms=farms,
        storage_facilities=storage,
        markets=markets,
        routes=routes,
        batches=batches,
        trucks={},
        current_time_hours=0.0,
    )
