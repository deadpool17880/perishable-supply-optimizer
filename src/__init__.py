"""DPSRO — Dynamic Perishable Supply-Chain Resilience Optimizer.

Public package interface. Exposes the complete CIS×EMBS hybrid framework
as a clean, importable API so that downstream scripts, notebooks, and the
Streamlit dashboard can import from a single namespace.

Modules
-------
models
    Dataclasses for supply-chain entities (Farm, Batch, StorageFacility,
    Market, TransportRoute, Truck, SupplyChainState, AllocationResult).

data_generator
    Synthetic supply-chain network generator with realistic spatial geometry
    and commodity-specific shelf-life profiles.

spoilage_model
    **EMBS component** — Arrhenius-inspired biophysical kinetic spoilage model.
    Provides LRU-cached scalar methods and a vectorised NumPy batch matrix kernel.

optimizer
    **CIS component** — Multi-objective Linear Program using SciPy HiGHS.
    Consumes EMBS spoilage coefficients as time-varying cost weights.

dynamic_optimizer
    Stateful re-optimizer that reformulates the residual graph after disruptions,
    ages batch shelf lives by elapsed time, and merges preserved flows.

disruption
    Disruption event library: TruckFailure, StorageOutage, DemandShock,
    TemperatureShock, RouteClosure, CombinedDisruption.

baseline_nearest
    Greedy Nearest-Market baseline allocation (Baseline 1).

baseline_cheapest
    Cheapest-Route Linear Program baseline (Baseline 2).

metrics
    Comparative metric computation and allocation result summarisation.

visualization
    Plotly-based interactive visualisations: network graph, Sankey flow diagram,
    scalability curves, and robustness box plots.

simulation
    CLI demonstration runner — end-to-end lifecycle from network generation
    through disruption injection to dynamic re-optimization.

experiments
    Automated experimental framework: Monte Carlo robustness study (100 seeds),
    scalability analysis (10–500 batches), and ablation study.

Usage
-----
Quick start (Python REPL or Jupyter):

    >>> from src.data_generator import generate_supply_chain
    >>> from src.spoilage_model import SpoilageModel
    >>> from src.optimizer import DynamicPerishableOptimizer

    >>> state = generate_supply_chain(num_batches=15, seed=42)
    >>> spoilage = SpoilageModel(alpha=2.5, beta=0.8, reference_temperature=4.0)
    >>> optimizer = DynamicPerishableOptimizer(spoilage_model=spoilage)
    >>> result = optimizer.solve(state)
    >>> print(f"Spoilage: {result.spoilage_percentage:.2f}% | Fulfilled: {result.demand_fulfillment_percentage:.2f}%")

License
-------
Open-source prototype — IEEE EMBS × CIS Hackathon submission.
"""

# Convenience re-exports for cleaner downstream imports
from src.models import (  # noqa: F401
    AllocationDecision,
    AllocationResult,
    Batch,
    Farm,
    Market,
    StorageFacility,
    SupplyChainState,
    TransportRoute,
    Truck,
)
from src.spoilage_model import SpoilageModel  # noqa: F401
from src.optimizer import DynamicPerishableOptimizer  # noqa: F401
from src.data_generator import generate_supply_chain  # noqa: F401

__version__ = "1.0.0"
__author__ = "DPSRO Team — IEEE EMBS × CIS Hackathon"
