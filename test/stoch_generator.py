# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:34:49 2026

@author: aless
"""

# -*- coding: utf-8 -*-
"""
Batch runner for PyPSA-Eur -> stochastic PyPSA network -> optional PyPSA solve -> optional SMS++.

Main features:
- Loads a PyPSA-Eur .nc network directly.
- Cleans the network with the same utilities used in pypsa2smspp debug runners.
- Builds stochastic scenarios with bounded Latin Hypercube multipliers.
- Runs all combinations of:
    - reduced snapshots
    - number of scenarios
- Exports stochastic PyPSA networks.
- Optionally solves with PyPSA.
- Optionally runs SMS++ transformation/optimization.

Edit only the INPUT PARAMETERS section in normal use.
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import pypsa


# =============================================================================
# INPUT PARAMETERS
# =============================================================================

# Input PyPSA-Eur network
NETWORK_NC = Path(
    r"C:\Users\aless\sms\transformation_pypsa_smspp\test\networks\network_small_fewsectors.nc"
)

# Output root
OUT_ROOT = Path("output/develop/test_tssb_complete")

# Grid settings
# If a target is None or >= current number of snapshots, the network is not reduced.
# Therefore, if the network has 8760 snapshots, target=8760 means "do not cut".
REDUCE_SNAPSHOTS_TARGETS = [100]
N_SCENARIOS_LIST = [3]

# Stochastic parameters to activate
STOCHASTIC_PARAMETERS = [
    "demand",
    "renewable_maxpower",
    #"renewable_marginal_cost",
    "hydro_inflow"
]

# Scenario generation
SEED = 123
INCLUDE_BASE_SCENARIO = False

# Use bounded scenario-level multipliers instead of Gaussian noise.
# These ranges are intentionally stress-test ranges.
# For a more realistic paper case, I would probably reduce demand to (0.75, 1.35).
MULTIPLIER_RANGES = {
    "demand": (0.60, 1.60),
    "renewable_maxpower": (0.70, 1.20),
    "renewable_marginal_cost": (0.50, 1.80),
}

# If True, high demand scenarios are paired with low renewable availability.
# This creates more stressful and more clearly separated scenarios.
COUPLE_DEMAND_AND_RENEWABLE_STRESS = True

# Optional mild elementwise perturbation around the scenario multiplier.
# Keep this small. The scenario-level multiplier is what creates real scenario diversity.
ADD_ELEMENTWISE_NOISE = False
ELEMENTWISE_SIGMA = {
    "demand": 0.03,
    "renewable_maxpower": 0.03,
    "renewable_marginal_cost": 0.00,
}

# Renewable carriers affected by renewable_maxpower and renewable_marginal_cost
RENEWABLE_CARRIERS = {
    "solar",
    "solar-hsat",
    "onwind",
    "offwind-ac",
    "offwind-dc",
    "offwind-float",
    "ror",
}

# PyPSA solve
RUN_PYPSA_SOLVE = True
EXPORT_PYPSA_LP = False
SOLVER_NAME = "gurobi"
SOLVER_OPTIONS = {
    "Threads": 32,
    "Method": 2,
    "Crossover": 0,
    "Seed": 123,
    "AggFill": 0,
    "PreDual": 0,
}

# SMS++ run
CREATE_SMSPP_MODEL = True
OPTIMIZE_SMSPP = True
RETRIEVE_SMSPP_SOLUTION = True

# Transformation toggles
CAPACITY_EXPANSION_UCBLOCK = True
ENABLE_THERMAL_UNITS = False
INTERMITTENT_CARRIERS = None
MERGE_LINKS = False
MERGE_SELECTOR = None

# For stochastic TSSBlock
CONFIGFILE = "TSSBlock/TSSBSCfg_grb.txt"
PYSMSSP_OPTIONS = {"logging": True}

# SMS++ artifacts
FP_TEMP = "smspp_{name}_temp.nc"
FP_LOG = "smspp_{name}_log.txt"
FP_SOLUTION = "smspp_{name}_solution.nc"

# Cleaning toggles
DO_CLEAN_E_SUM = False
DO_CLEAN_CICLICITY_STORAGE = False
DO_ADD_SLACK_UNIT = True
DO_CLEAN_STORAGE_UNITS = False
DO_CLEAN_STORES = False
REMOVE_STORE_BUSES = False
REMOVE_GENERATORS_ON_REMOVED_BUSES = False
DO_CLEAN_GLOBAL_CONSTRAINTS = True
DO_MEAN_EFFICIENCIES = False

# Exports
EXPORT_STOCHASTIC_PYPSA_NC = True
EXPORT_SOLVED_PYPSA_NC = True
EXPORT_SMSPP_MODEL_NC = True
EXPORT_SMSPP_REPOPULATED_NC = True
EXPORT_STATISTICS_CSV = True

# Verbosity
VERBOSE = True


# =============================================================================
# PATH SETUP
# =============================================================================

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
print(">>> FORCED CWD:", Path.cwd())

REPO_ROOT = HERE.parent
SCRIPTS = (REPO_ROOT / "scripts").resolve()
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

from pypsa2smspp.transformation import Transformation
from pypsa2smspp.network_correction import (
    clean_e_sum,
    clean_ciclicity_storage,
    add_slack_unit,
    reduce_snapshots_and_scale_costs,
    clean_storage_units,
    clean_stores,
    clean_global_constraints,
)

from pypsa2smspp.utils import preprocess_dynamic_link_parameters_to_static_means


# =============================================================================
# UTILITIES
# =============================================================================

def ensure_dir(path: Path) -> None:
    """Create a directory if missing."""
    path.mkdir(parents=True, exist_ok=True)


def safe_remove(path: Path) -> None:
    """Remove a file if it exists."""
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def pypsa_reference_objective(network: pypsa.Network) -> Optional[float]:
    """Return the PyPSA objective, including the constant if available."""
    try:
        return float(network.objective + network.objective_constant)
    except Exception:
        try:
            return float(network.objective)
        except Exception:
            return None


def maybe_reduce_snapshots(network: pypsa.Network, target: Optional[int]) -> pypsa.Network:
    """Reduce snapshots only when target is smaller than the current number of snapshots."""
    if target is None:
        return network

    n_snapshots = len(network.snapshots)

    if int(target) >= n_snapshots:
        print(f"[INFO] target={target} >= current snapshots={n_snapshots}. No snapshot reduction.")
        return network

    print(f"[INFO] Reducing snapshots from {n_snapshots} to {target}.")
    return reduce_snapshots_and_scale_costs(
        network,
        target=int(target),
        scale_capital_costs=False,
    )


def clean_network(network: pypsa.Network) -> pypsa.Network:
    """Apply selected pypsa2smspp cleaning steps."""
    if DO_CLEAN_E_SUM:
        network = clean_e_sum(network)

    if DO_CLEAN_CICLICITY_STORAGE:
        network = clean_ciclicity_storage(network)

    if DO_ADD_SLACK_UNIT:
        network = add_slack_unit(network)

    if DO_CLEAN_STORAGE_UNITS:
        network = clean_storage_units(network)

    if DO_CLEAN_STORES:
        network = clean_stores(
            network,
            remove_store_buses=REMOVE_STORE_BUSES,
            remove_generators_on_removed_buses=REMOVE_GENERATORS_ON_REMOVED_BUSES,
        )

    if DO_CLEAN_GLOBAL_CONSTRAINTS:
        network = clean_global_constraints(network)

    if DO_MEAN_EFFICIENCIES:
        network = preprocess_dynamic_link_parameters_to_static_means(
            network,
            drop_dynamic=True,
        )

    return network


def lhs_uniform(
    n: int,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate bounded Latin Hypercube samples in [low, high].

    This guarantees that the full interval is explored more evenly than pure random sampling.
    """
    if n <= 0:
        return np.array([], dtype=float)

    edges = np.linspace(0.0, 1.0, n + 1)
    points = rng.uniform(edges[:-1], edges[1:])
    rng.shuffle(points)
    return low + points * (high - low)


def build_scenario_table(
    n_scenarios: int,
    seed: int,
) -> pd.DataFrame:
    """Build scenario multipliers."""
    rng = np.random.default_rng(seed)

    scenario_names = [f"scenario_{i + 1:03d}" for i in range(n_scenarios)]

    rows = []

    if INCLUDE_BASE_SCENARIO:
        rows.append(
            {
                "scenario": "base",
                "probability": 1.0 / n_scenarios,
                "demand": 1.0,
                "renewable_maxpower": 1.0,
                "renewable_marginal_cost": 1.0,
            }
        )
        n_random = n_scenarios - 1
        random_names = scenario_names[:n_random]
    else:
        n_random = n_scenarios
        random_names = scenario_names

    if n_random < 0:
        raise ValueError("n_scenarios must be at least 1.")

    demand_low, demand_high = MULTIPLIER_RANGES["demand"]
    ren_low, ren_high = MULTIPLIER_RANGES["renewable_maxpower"]
    cost_low, cost_high = MULTIPLIER_RANGES["renewable_marginal_cost"]

    demand = lhs_uniform(n_random, demand_low, demand_high, rng)
    renewable_maxpower = lhs_uniform(n_random, ren_low, ren_high, rng)
    renewable_marginal_cost = lhs_uniform(n_random, cost_low, cost_high, rng)

    if COUPLE_DEMAND_AND_RENEWABLE_STRESS:
        demand = np.sort(demand)
        renewable_maxpower = np.sort(renewable_maxpower)[::-1]
        renewable_marginal_cost = np.sort(renewable_marginal_cost)

    for name, d_mult, r_mult, c_mult in zip(
        random_names,
        demand,
        renewable_maxpower,
        renewable_marginal_cost,
    ):
        rows.append(
            {
                "scenario": name,
                "probability": 1.0 / n_scenarios,
                "demand": float(d_mult),
                "renewable_maxpower": float(r_mult),
                "renewable_marginal_cost": float(c_mult),
            }
        )

    table = pd.DataFrame(rows)

    if len(table) != n_scenarios:
        raise RuntimeError(
            f"Scenario table has {len(table)} rows, expected {n_scenarios}."
        )

    return table


def elementwise_multiplier(
    base: pd.DataFrame | pd.Series,
    center: float,
    sigma: float,
    rng: np.random.Generator,
    lower: float,
    upper: Optional[float] = None,
) -> pd.DataFrame | pd.Series:
    """Build a mild elementwise multiplier around a scenario-level center."""
    values = rng.normal(loc=center, scale=sigma, size=base.shape)
    values = np.maximum(values, lower)

    if upper is not None:
        values = np.minimum(values, upper)

    if isinstance(base, pd.DataFrame):
        return pd.DataFrame(values, index=base.index, columns=base.columns)

    if isinstance(base, pd.Series):
        return pd.Series(values, index=base.index)

    raise TypeError(f"Unsupported base type: {type(base)}")


def stochasticify_network(
    network: pypsa.Network,
    n_scenarios: int,
    seed: int,
    case_dir: Path,
) -> tuple[pypsa.Network, pd.DataFrame]:
    """Activate PyPSA stochastic scenarios and assign scenario-dependent values."""
    scenario_table = build_scenario_table(n_scenarios=n_scenarios, seed=seed)
    scenario_table.to_csv(case_dir / "scenario_multipliers.csv", index=False)

    scenarios = scenario_table["scenario"].tolist()
    probabilities = dict(zip(scenario_table["scenario"], scenario_table["probability"]))

    base_load = network.loads_t.p_set.copy()
    base_pmaxpu = network.generators_t.p_max_pu.copy()
    base_marginal = network.generators.marginal_cost.copy()

    renewable_generators = network.generators.index[
        network.generators.carrier.isin(RENEWABLE_CARRIERS)
    ]

    missing_renewables = sorted(set(RENEWABLE_CARRIERS) - set(network.generators.carrier.unique()))
    if VERBOSE and missing_renewables:
        print("[INFO] Some renewable carriers were not found in this network:")
        print("       ", missing_renewables)

    network.set_scenarios(probabilities)

    rng = np.random.default_rng(seed + 10_000)

    for _, row in scenario_table.iterrows():
        scenario = row["scenario"]

        if "demand" in STOCHASTIC_PARAMETERS:
            d_mult = float(row["demand"])

            if ADD_ELEMENTWISE_NOISE:
                mult = elementwise_multiplier(
                    base_load,
                    center=d_mult,
                    sigma=ELEMENTWISE_SIGMA["demand"],
                    rng=rng,
                    lower=0.0,
                )
            else:
                mult = d_mult

            network.loads_t.p_set[scenario] = (base_load * mult).clip(lower=0.0)

        if "renewable_maxpower" in STOCHASTIC_PARAMETERS:
            r_mult = float(row["renewable_maxpower"])
            scenario_pmaxpu = base_pmaxpu.copy()

            if len(renewable_generators) > 0:
                base_ren_pmaxpu = base_pmaxpu.loc[:, renewable_generators]

                if ADD_ELEMENTWISE_NOISE:
                    mult = elementwise_multiplier(
                        base_ren_pmaxpu,
                        center=r_mult,
                        sigma=ELEMENTWISE_SIGMA["renewable_maxpower"],
                        rng=rng,
                        lower=0.0,
                    )
                else:
                    mult = r_mult

                scenario_pmaxpu.loc[:, renewable_generators] = (
                    base_ren_pmaxpu * mult
                ).clip(lower=0.0, upper=1.0)

            network.generators_t.p_max_pu[scenario] = scenario_pmaxpu

        if "renewable_marginal_cost" in STOCHASTIC_PARAMETERS:
            c_mult = float(row["renewable_marginal_cost"])
            scenario_marginal = base_marginal.copy()

            if len(renewable_generators) > 0:
                base_ren_marginal = base_marginal.loc[renewable_generators]

                if ADD_ELEMENTWISE_NOISE:
                    mult = elementwise_multiplier(
                        base_ren_marginal,
                        center=c_mult,
                        sigma=ELEMENTWISE_SIGMA["renewable_marginal_cost"],
                        rng=rng,
                        lower=0.0,
                    )
                else:
                    mult = c_mult

                scenario_marginal.loc[renewable_generators] = (
                    base_ren_marginal * mult
                ).clip(lower=0.0)

            network.generators.marginal_cost[scenario] = scenario_marginal

    return network, scenario_table


def export_statistics(network: pypsa.Network, path: Path) -> None:
    """Export PyPSA statistics if possible."""
    try:
        stats = network.statistics()
        if hasattr(stats, "to_frame"):
            stats = stats.to_frame(name="value")
        stats.to_csv(path)
    except Exception as e:
        print(f"[WARN] Could not export statistics to {path}: {e}")


def print_case_header(case_name: str, case_dir: Path) -> None:
    """Print a readable case header."""
    print("\n" + "=" * 80)
    print(f"CASE: {case_name}")
    print(f"DIR : {case_dir}")
    print("=" * 80)


# =============================================================================
# BATCH PIPELINE
# =============================================================================

ensure_dir(OUT_ROOT)

summary_rows = []

base_name = NETWORK_NC.stem

for target_snapshots in REDUCE_SNAPSHOTS_TARGETS:
    for n_scenarios in N_SCENARIOS_LIST:

        case_name = f"{base_name}__snap{target_snapshots}__scen{n_scenarios}"
        case_dir = OUT_ROOT / case_name
        ensure_dir(case_dir)

        print_case_header(case_name, case_dir)

        metrics: Dict[str, Any] = {
            "case": case_name,
            "input_file": str(NETWORK_NC),
            "target_snapshots": target_snapshots,
            "n_scenarios": n_scenarios,
            "status": "OK",
            "error_msg": "",
            "n_snapshots_final": None,
            "Obj_PyPSA": None,
            "Obj_SMSpp": None,
            "Obj_rel_error_pct": None,
        }

        stochastic_network = None
        solved_pypsa_network = None
        smspp_network = None
        transformation = None

        try:
            # -----------------------------------------------------------------
            # Load deterministic network
            # -----------------------------------------------------------------
            network = pypsa.Network(str(NETWORK_NC))

            # -----------------------------------------------------------------
            # Reduce snapshots before stochasticification
            # -----------------------------------------------------------------
            network = maybe_reduce_snapshots(network, target_snapshots)

            # -----------------------------------------------------------------
            # Clean network
            # -----------------------------------------------------------------
            network = clean_network(network)

            metrics["n_snapshots_final"] = len(network.snapshots)

            # -----------------------------------------------------------------
            # Stochasticify
            # -----------------------------------------------------------------
            stochastic_network, scenario_table = stochasticify_network(
                network=network,
                n_scenarios=n_scenarios,
                seed=SEED + int(n_scenarios) + int(target_snapshots),
                case_dir=case_dir,
            )

            if EXPORT_STOCHASTIC_PYPSA_NC:
                stochastic_nc = case_dir / f"network_stochastic_{case_name}.nc"
                safe_remove(stochastic_nc)
                stochastic_network.export_to_netcdf(str(stochastic_nc))

            # -----------------------------------------------------------------
            # Optional PyPSA solve
            # -----------------------------------------------------------------
            if RUN_PYPSA_SOLVE:
                solved_pypsa_network = stochastic_network.copy()

                solved_pypsa_network.optimize(
                    solver_name=SOLVER_NAME,
                    solver_options=SOLVER_OPTIONS,
                )

                obj_pypsa = pypsa_reference_objective(solved_pypsa_network)
                metrics["Obj_PyPSA"] = obj_pypsa

                if EXPORT_PYPSA_LP:
                    pypsa_lp = case_dir / f"pypsa_{case_name}.lp"
                    try:
                        solved_pypsa_network.model.to_file(fn=str(pypsa_lp))
                    except Exception as e:
                        print(f"[WARN] Could not export PyPSA LP: {e}")

                if EXPORT_SOLVED_PYPSA_NC:
                    solved_nc = case_dir / f"network_pypsa_solved_{case_name}.nc"
                    safe_remove(solved_nc)
                    solved_pypsa_network.export_to_netcdf(str(solved_nc))

                if EXPORT_STATISTICS_CSV:
                    export_statistics(
                        solved_pypsa_network,
                        case_dir / f"stats_pypsa_{case_name}.csv",
                    )

            # -----------------------------------------------------------------
            # Optional SMS++ model creation / optimization / inverse transform
            # -----------------------------------------------------------------
            if CREATE_SMSPP_MODEL:
                transformation_kwargs = {
                    "capacity_expansion_ucblock": CAPACITY_EXPANSION_UCBLOCK,
                    "enable_thermal_units": ENABLE_THERMAL_UNITS,
                    "intermittent_carriers": INTERMITTENT_CARRIERS,
                    "merge_links": MERGE_LINKS,
                    "merge_selector": MERGE_SELECTOR,
                    "workdir": case_dir,
                    "name": case_name,
                    "overwrite": True,
                    "fp_temp": FP_TEMP,
                    "fp_log": FP_LOG,
                    "fp_solution": FP_SOLUTION,
                    "configfile": CONFIGFILE,
                    "pysmspp_options": PYSMSSP_OPTIONS,
                    "stochastic_parameters": {
                        "stochastic_type": "tssb",
                        "parameters": STOCHASTIC_PARAMETERS,
                    },
                }

                transformation = Transformation(**transformation_kwargs)

                # Step 1: create SMS++ model only.
                smspp_model = transformation.create_model(
                    stochastic_network,
                    verbose=VERBOSE,
                )

                metrics["SMSpp_model_created"] = True
                metrics["SMSpp_optimized"] = False
                metrics["SMSpp_solution_retrieved"] = False
                metrics["SMSpp_model_file"] = None

                if EXPORT_SMSPP_MODEL_NC:
                    smspp_model_nc = case_dir / f"network_smspp_model_{case_name}.nc"
                    safe_remove(smspp_model_nc)
                    smspp_model.to_netcdf(str(smspp_model_nc), force=True)
                    metrics["SMSpp_model_file"] = str(smspp_model_nc)

                # Step 2: optionally optimize SMS++.
                if OPTIMIZE_SMSPP:
                    transformation.optimize(verbose=VERBOSE)

                    metrics["SMSpp_optimized"] = True

                    obj_smspp = float(transformation.result.objective_value)
                    metrics["Obj_SMSpp"] = obj_smspp

                    obj_pypsa = metrics["Obj_PyPSA"]
                    if obj_pypsa is not None and obj_pypsa != 0.0:
                        metrics["Obj_rel_error_pct"] = (
                            (obj_smspp - obj_pypsa) / obj_pypsa * 100.0
                        )

                    # Step 3: optionally retrieve SMS++ solution into PyPSA network.
                    if RETRIEVE_SMSPP_SOLUTION:
                        smspp_network = transformation.retrieve_solution(
                            stochastic_network,
                            verbose=VERBOSE,
                        )

                        metrics["SMSpp_solution_retrieved"] = True

                        if EXPORT_SMSPP_REPOPULATED_NC:
                            smspp_repopulated_nc = case_dir / f"network_smspp_repopulated_{case_name}.nc"
                            safe_remove(smspp_repopulated_nc)
                            smspp_network.export_to_netcdf(str(smspp_repopulated_nc))

                        if EXPORT_STATISTICS_CSV:
                            export_statistics(
                                smspp_network,
                                case_dir / f"stats_smspp_{case_name}.csv",
                            )

                timer_rows = getattr(getattr(transformation, "timer", None), "rows", None) or []
                if timer_rows:
                    pd.DataFrame(timer_rows).to_csv(
                        case_dir / f"timings_{case_name}.csv",
                        index=False,
                    )

                    metrics["SMSpp_total_s"] = sum(
                        float(r.get("elapsed_s", 0.0))
                        for r in timer_rows
                        if r.get("elapsed_s") is not None
                    )

                timer_rows = getattr(getattr(transformation, "timer", None), "rows", None) or []
                if timer_rows:
                    pd.DataFrame(timer_rows).to_csv(
                        case_dir / f"timings_{case_name}.csv",
                        index=False,
                    )
                    metrics["SMSpp_total_s"] = sum(
                        float(r.get("elapsed_s", 0.0))
                        for r in timer_rows
                        if r.get("elapsed_s") is not None
                    )

            print("[OK] Finished case.")
            print(f"     snapshots              : {metrics['n_snapshots_final']}")
            print(f"     scenarios              : {metrics['n_scenarios']}")
            print(f"     Obj_PyPSA              : {metrics['Obj_PyPSA']}")
            print(f"     Obj_SMS++              : {metrics['Obj_SMSpp']}")
            print(f"     rel err %              : {metrics['Obj_rel_error_pct']}")
            print(f"     SMS++ model created    : {metrics.get('SMSpp_model_created')}")
            print(f"     SMS++ optimized        : {metrics.get('SMSpp_optimized')}")
            print(f"     SMS++ solution retrieved: {metrics.get('SMSpp_solution_retrieved')}")
            print(f"     SMS++ model file       : {metrics.get('SMSpp_model_file')}")

        except Exception as e:
            metrics["status"] = "FAIL"
            metrics["error_msg"] = f"{type(e).__name__}: {e}"
            print(f"[FAIL] {metrics['error_msg']}")
            traceback.print_exc()

        finally:
            pd.DataFrame([metrics]).to_csv(
                case_dir / "summary_row.csv",
                index=False,
            )
            summary_rows.append(metrics)

            pd.DataFrame(summary_rows).to_csv(
                OUT_ROOT / "summary_all_cases.csv",
                index=False,
            )

print("\nDONE.")
print(f"Summary written to: {OUT_ROOT / 'summary_all_cases.csv'}")