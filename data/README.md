# Supply-Chain Dataset Documentation

## Real-World Data Policy & Provenance
In adherence to the project's zero-fabrication and scientific integrity rules:
- **Synthetic Network Instances**: The supply network topologies, coordinates, route transit times, and batch sizes in this prototype are generated deterministically using reproducible pseudorandom generators (fixed seed `42`).
- **Contextual Calibration**: Biophysical shelf life ranges, initial deterioration rates, and ideal storage temperatures are calibrated against empirical post-harvest literature from the **FAO (Food and Agriculture Organization)** and **USDA Agricultural Handbook No. 66 (The Commercial Storage of Fruits, Vegetables, and Florist and Nursery Stocks)**:
  - *Strawberries / Berries*: 24–48 hours ambient, 2°C ideal cold storage.
  - *Leafy Greens*: 36–72 hours ambient, 4°C ideal cold storage.
  - *Tomatoes*: 72–144 hours ambient, 10–12°C ideal cold storage.
  - *Stone Fruits / Peaches*: 48–96 hours ambient, 4°C ideal cold storage.
  - *Apples*: 168–336 hours ambient, 4°C ideal cold storage.

## Files Description
1. `farms.csv`: Agricultural production origins with geographic coordinates, available crop quantities, and harvest timestamps.
2. `storage.csv`: Intermediate cold-storage consolidation hubs, specifying total physical holding capacities, current loads, operating temperatures (default 4°C), and operational availability.
3. `markets.csv`: Urban consumption sinks, specifying target demand volume, geographic coordinates, and demand priority multipliers.
4. `routes.csv`: Transportation corridors (direct express and hub-feeder arcs) containing distance in km, average travel duration in hours, cost per unit mass, operational status, and baseline risk factors.
5. `batches.csv`: Perishable production lots tagged by origin farm, commodity class, mass in kg, initial shelf life, remaining shelf life, current ambient temperature, and quality index.
