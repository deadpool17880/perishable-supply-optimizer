# 🌾 FreshRoute — Smart Crop Delivery Planner
### Built on the Dynamic Perishable Supply-Chain Resilience Optimizer (DPSRO)

---

## What Is This App?

**FreshRoute** is a smart computer system that helps farmers, agricultural co-operatives, and food-logistics managers get the most produce to market in the freshest possible condition — and quickly re-plan when something goes wrong on the way.

Think of it as a **GPS for your harvest**. Instead of just finding the shortest road, FreshRoute considers:

- ⏰ How much freshness each batch of produce has left
- ❄️ Whether there is a cold storage warehouse on the route that can slow down spoilage
- 💰 How much the journey costs
- 🏪 Which markets need produce the most urgently
- 🚨 What happens if a truck breaks down, a warehouse loses power, or a road is blocked

The system finds the **mathematically optimal plan** for all your crops at once — in less than a second — and instantly re-plans if a disruption occurs.

---

## Who Is This For?

| Audience | Why FreshRoute Helps |
|---|---|
| 🌾 **Smallholder Farmers** | Get more of your harvest to market without wasting time figuring out routes manually |
| 🤝 **Farming Co-operatives** | Coordinate multiple farms' produce deliveries across shared cold stores and markets |
| 🚛 **Cold-Chain Logistics Operators** | Optimise truck utilisation, cold storage use, and route selection under real constraints |
| 🏛️ **Agricultural Extension Officers** | Run scenario simulations ("what if a heatwave hits?") to plan community resilience |
| 🔬 **Agri-Tech Researchers** | Validate supply chain algorithms against an experimentally rigorous platform with real objective metrics |
| 🏪 **Food Market Managers** | Ensure consistent supply from multiple farms with clear delivery planning |

---

## How Does the App Work? (Plain Language)

### Step 1: Tell the System About Your Farm Network
The system creates a map of:
- **Farms** — where your produce is harvested
- **Crop batches** — each harvest load (e.g., 800 kg of tomatoes from Farm 1)
- **Cold storage warehouses** — where produce can be temporarily kept cool to stay fresh longer
- **Markets** — the buyers and destinations for your produce
- **Trucks & Routes** — which vehicles can carry which loads over which roads

### Step 2: The Smart Plan Is Calculated
The optimizer considers **every possible combination** of paths and picks the one that:
1. Minimises food wasted to spoilage
2. Maximises how much fresh produce reaches markets
3. Keeps delivery costs reasonable
4. Avoids routes that would take too long given freshness remaining

> **In the default demo:** 10 crop batches, 3 farms, 2 cold stores, 3 markets, 5 trucks  
> **Result:** 100% demand fulfilled, spoilage under 1%, in < 10 milliseconds

### Step 3: Simulate a Disruption
You can test real-world problems:
- A **truck breaking down** in transit
- A **cold storage warehouse losing power**
- A **sudden demand surge** at one market
- A **heatwave** causing produce to spoil faster
- A **road being blocked** due to flooding or an accident
- **Multiple problems at once**

### Step 4: Watch the System Fix the Plan
FreshRoute **keeps all deliveries already safely moving**, and only re-routes the blocked or stranded batches. This targeted re-planning preserves efficiency and responds in under 5 milliseconds.

### Step 5: See Your Reports
View how many attempts were made, how the objective score improved, how much food was saved across 100 different test scenarios, and how fast the system performs with larger farm networks.

---

## Step-by-Step User Instructions

### First-Time Setup

1. **Open a terminal (Command Prompt on Windows, Terminal on Mac)**
2. Navigate to the project folder:
   ```bash
   cd perishable_supply_optimizer
   ```
3. Install all required software (one time only):
   ```bash
   python -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   venv/bin/streamlit run dashboard/app.py --server.port 8501
   ```
5. Open your web browser and go to: **http://localhost:8501**

---

### Using the App — Screen by Screen

#### 🏠 My Farm Dashboard (Home Screen)

When you first open the app, you see:
- **A summary of all your crop batches** — sorted by urgency (batches about to expire are shown in red)
- **Your farm network map** — showing farms, cold stores, and markets as connected nodes
- **Four key numbers** at the top: total crops, market demand, average freshness remaining, and batches at risk
- **How the app works** — an explainer for first-time users

✅ **What to do:** Review your crops. If you see red "URGENT" batches, go to "Plan My Delivery" immediately.

---

#### 🚛 Plan My Delivery

This is the main action page.

1. **Click the large green button** — *"🟢 Find Best Delivery Plan"*
2. Wait a moment (usually less than 1 second)
3. You will see:
   - **4 key results**: Crops to deliver, how much arrives fresh, delivery cost, and how fast the plan was made
   - **A flow map** (called a "Sankey diagram") showing exactly which crops go to which markets via which cold stores
   - **A comparison** showing how much better the smart plan is vs. just sending crops to the nearest market
   - **A full batch-by-batch table** with every route, transit time, spoilage estimate, and cost

**Changing the number of batches:**
Use the sidebar slider — *"How many crop batches do you have today?"* — then click *"Start Fresh / Change My Farm"*.

**Starting a new farm setup:**
Change the *"Scenario Number"* and click *"Start Fresh / Change My Farm"*. Each number gives a different randomised farm network.

---

#### ⚠️ What If Something Goes Wrong?

Use this to test what would happen if a disaster struck your supply chain.

1. **Select a problem** from the dropdown:
   - *Truck Breaks Down* — most common real-world problem
   - *Cold Storage Loses Power* — second most common
   - *Market Demand Jumps* — useful for harvest festivals, food drives
   - *Heatwave* — set the temperature increase with the slider
   - *Road Blocked* — simulate flooding or accidents
   - *Multiple Problems at Once* — stress test

2. **Set any extra options** (e.g., which truck breaks down, how hot the heatwave is)

3. **Click "🚨 Simulate This Problem"**

4. See a **before vs. after map** showing what routes broke

5. Then go to the next page to see the fix.

---

#### 🔁 Fix the Plan Fast

1. If you've already simulated a problem (previous page), click **"🟢 Fix My Delivery Plan Now!"**
2. In milliseconds, you'll see:
   - **Recovery report**: how many batches were affected, how many kg were rerouted, how much food was saved
   - **New delivery map** — updated routes around the blocked areas
   - **Before vs. after comparison table** — see the exact cost and spoilage impact of the disruption

---

#### 📊 Results & Reports

Four report tabs:

| Tab | What You See |
|---|---|
| **Optimization Attempt Log** | Every plan that was computed — label, what changed, objective score, spoilage %, delivery %, runtime |
| **100-Run Study** | Box plots of performance across 100 randomised scenarios — proves reliability |
| **Speed Benchmarks** | How fast the system runs with 10, 50, 100, 200, 500 crop batches |
| **Component Analysis** | What happens when parts of the algorithm are switched off (ablation study) |

> **To generate the 100-run study data**, run this in your terminal:
> ```bash
> venv/bin/python -m src.experiments
> ```

---

#### 🔬 For Researchers & Experts

This page contains the full mathematical and algorithmic documentation, intended for judges, technical reviewers, and researchers. It includes:

- The LP formulation with LaTeX notation
- The EMBS-CIS hybrid algorithm architecture
- Full parameter configuration table with justifications
- Jury Q&A answers with references

---

## Running the CLI Demo (No Browser Needed)

For a quick text-based demo:
```bash
source venv/bin/activate
python -m src.simulation
```

This runs the full pipeline from data generation through optimisation through disruption and recovery, printing all results to your terminal.

---

## Running Unit Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

Expected: **20 tests pass in < 3 seconds**.

---

## Project File Structure

```
perishable_supply_optimizer/
├── dashboard/                  ← Web app (Streamlit)
│   ├── app.py                  ← Main app — this is what you open in browser
│   └── components/             ← Reusable UI panels
├── src/                        ← Core algorithm engine
│   ├── models.py               ← Data structures (Farm, Batch, Market, etc.)
│   ├── spoilage_model.py       ← Arrhenius thermal spoilage kinetics
│   ├── data_generator.py       ← Synthetic farm network generator
│   ├── optimizer.py            ← Multi-objective LP solver (main algorithm)
│   ├── dynamic_optimizer.py    ← Disruption re-planner (stateful recovery)
│   ├── disruption.py           ← Disruption types (truck failure, etc.)
│   ├── baseline_nearest.py     ← Simple nearest-market baseline
│   ├── baseline_cheapest.py    ← Cost-minimising LP baseline
│   ├── simulation.py           ← CLI demo runner
│   ├── experiments.py          ← 100-scenario Monte Carlo + benchmarks
│   ├── metrics.py              ← Performance metrics calculation
│   └── visualization.py        ← Plotly chart generators
├── tests/                      ← 20 automated unit tests
├── results/                    ← Experiment output CSV files
├── config/config.yaml          ← Algorithm parameter settings
├── requirements.txt            ← Python package list
├── README.md                   ← Technical overview
├── APPLICATION_GUIDE.md        ← This file
└── ALGORITHMIC_ARCHITECTURE.md ← Algorithm design document
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| App doesn't open in browser | Make sure you see `You can now view your Streamlit app` in the terminal. Then go to `http://localhost:8501` |
| `ModuleNotFoundError` | Make sure you activated the venv: `source venv/bin/activate` |
| Port already in use | Stop the old Streamlit: `pkill -f streamlit` then restart |
| Optimization takes too long | Reduce batch count in sidebar slider to under 20 |
| Missing results/CSV files | Run `venv/bin/python -m src.experiments` first |

---

## Technology Used

| Component | Technology |
|---|---|
| User Interface | Streamlit 1.x (Python web app framework) |
| Optimisation Solver | SciPy HiGHS (dual-simplex LP) |
| Graphs & Charts | Plotly Express + Plotly Graph Objects |
| Network Modelling | NetworkX |
| Numerical Core | NumPy + Pandas |
| Unit Tests | Pytest |
| Language | Python 3.11+ |

---

*FreshRoute — Dynamic Perishable Supply-Chain Resilience Optimizer (DPSRO)*  
*Built for the Agri-Tech Hackathon 2024*
