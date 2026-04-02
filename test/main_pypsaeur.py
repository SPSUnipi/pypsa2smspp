# -*- coding: utf-8 -*-
"""
Single-case debug runner for PyPSA -> SMS++ (via pypsa2smspp).

Linear script style:
- All inputs/toggles at the top
- No main(), no wrapper functions returning only a subset of stuff
- Variables remain in global scope so you can inspect everything in a debugger

Artifacts per-case in output/debug/<case_name>/.
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pypsa


# =============================================================================
# INPUT PARAMETERS (EDIT HERE)
# =============================================================================

# Inputs
# NETWORK_NC = Path(r"/home/pampado/sector-coupled/pypsa-eur-smspp/resources/unit_commitment_smspp_italy/networks/base_s_5_elec_.nc")
NETWORK_NC = Path(
     r"/home/pampado/sector-coupled/pypsa-eur-smspp/resources/smspp_italy_very_small_fewsectors/networks/base_s_2___2050.nc"
)
# NETWORK_NC = Path(r"C:\Users\aless\sms\transformation_pypsa_smspp\test\networks\network_giga_small.nc")


# Output
OUT_ROOT = Path("output/develop")
CASE_NAME = None  # if None -> derived from NETWORK_NC.stem

# PyPSA reference solve
SOLVER_NAME = "gurobi"
SOLVER_OPTIONS = {
    "Threads": 32,
    "Method": 2,       # barrier
    "Crossover": 0,
    "BarConvTol": 1e-5,
    "Seed": 123,
    "AggFill": 0,
    "PreDual": 0,
}

# Transformation toggles
CAPACITY_EXPANSION_UCBLOCK = True      # True -> UCBlock, False -> InvestmentBlock
ENABLE_THERMAL_UNITS = False            # False -> everything (except slack) treated as intermittent
INTERMITTENT_CARRIERS = None           # None -> default renewable_carriers; list/str -> override
MERGE_LINKS = False                    # False / True / ["tes","battery","h2", ...]
MERGE_SELECTOR = None                  # optional callable (only needed for custom merge tags)

# SMS++ artifacts (rendered with {name} and placed in CASE_DIR)
FP_TEMP = "smspp_{name}_temp.nc"
FP_LOG = "smspp_{name}_log.txt"
FP_SOLUTION = "smspp_{name}_solution.nc"

# SMS++ config selection (no YAML)
# - "auto" chooses a default template based on CAPACITY_EXPANSION_UCBLOCK inside Transformation.optimize()
# - otherwise pass a template path or a pysmspp.SMSConfig (depending on your implementation)
CONFIGFILE = "UCBlock/uc_solverconfig_grb.txt"

# Optional: pass-through options for pySMSpp (all optional; defaults exist in pySMSpp)
# Examples:
#   PYSMSSP_OPTIONS = {"inner_block_name": "Block_0", "smspp_solver": "auto"}
PYSMSSP_OPTIONS = {"logging": True}

# Cleaning toggles
DO_CLEAN_E_SUM = False
DO_CLEAN_CICLICITY_STORAGE = False
DO_ADD_SLACK_UNIT = True
DO_REDUCE_SNAPSHOTS = True
REDUCE_SNAPSHOTS_TO = 24
DO_CLEAN_STORAGE_UNITS = True  # optional, kept off by default
DO_CLEAN_STORES = False         # optional, kept off by default
REMOVE_STORE_BUSES = True
REMOVE_GENERATORS_ON_REMOVED_BUSES = True
DO_CLEAN_GLOBAL_CONSTRAINTS = True

# Debug artifacts
EXPORT_PYPSA_LP = True
EXPORT_PYPSA_NC = True
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
# PROJECT IMPORTS (after sys.path adjustment)
# =============================================================================

from pypsa2smspp.transformation import Transformation
from pypsa2smspp.network_correction import (
    clean_e_sum,
    clean_ciclicity_storage,
    add_slack_unit,
    reduce_snapshots_and_scale_costs,
    clean_storage_units,
    clean_stores,
    clean_global_constraints
)


# =============================================================================
# SMALL UTILITIES (kept as functions, but no "main runner" function)
# =============================================================================

def safe_remove(path: Path) -> None:
    """Remove a file if it exists."""
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def ensure_dir(path: Path) -> None:
    """Create a directory if missing."""
    path.mkdir(parents=True, exist_ok=True)


def print_kv(d: Dict[str, Any], title: str = "") -> None:
    """Pretty print small dict."""
    if title:
        print(title)
    for k, v in d.items():
        print(f"  - {k}: {v}")


def pypsa_reference_objective(network: "pypsa.Network") -> float:
    """Robust reference objective extraction."""
    try:
        return float(network.objective + getattr(network, "objective_constant", 0.0))
    except Exception:
        return float(network.objective)


# =============================================================================
# DERIVED SETTINGS & OUTPUT PATHS
# =============================================================================

if CASE_NAME is None:
    CASE_NAME = NETWORK_NC.stem

CASE_DIR = OUT_ROOT / CASE_NAME
ensure_dir(CASE_DIR)

PYPSA_LP = CASE_DIR / f"pypsa_{CASE_NAME}.lp"
PYPSA_OUT_NC = CASE_DIR / f"network_pypsa_{CASE_NAME}.nc"
SMSPP_REPOP_NC = CASE_DIR / f"network_smspp_{CASE_NAME}.nc"
DEBUG_SUMMARY_CSV = CASE_DIR / "debug_summary_row.csv"
STATS_PYPSA_CSV = CASE_DIR / f"stats_pypsa_{CASE_NAME}.csv"
STATS_SMSPP_CSV = CASE_DIR / f"stats_smspp_{CASE_NAME}.csv"
TIMINGS_CSV = CASE_DIR / f"timings_{CASE_NAME}.csv"

# Clean stale files
for p in (
    PYPSA_LP,
    PYPSA_OUT_NC,
    SMSPP_REPOP_NC,
    DEBUG_SUMMARY_CSV,
    STATS_PYPSA_CSV,
    STATS_SMSPP_CSV,
    TIMINGS_CSV,
):
    safe_remove(p)

# Also clean SMS++ artifacts (since overwrite behaviour is inside Transformation.optimize)
safe_remove(CASE_DIR / FP_TEMP.format(name=CASE_NAME))
safe_remove(CASE_DIR / FP_LOG.format(name=CASE_NAME))
safe_remove(CASE_DIR / FP_SOLUTION.format(name=CASE_NAME))

print(f"\n=== DEBUG RUN: {CASE_NAME} ===")
print(f"Input network: {NETWORK_NC}")
print(f"Output folder: {CASE_DIR}")

metrics: Dict[str, Any] = {
    "case": CASE_NAME,
    "input_file": str(NETWORK_NC),
    "pypsa_solver": SOLVER_NAME,
    "capacity_expansion_ucblock": CAPACITY_EXPANSION_UCBLOCK,
    "status": "OK",
    "error_msg": "",
    "Obj_PyPSA": None,
    "Obj_SMSpp": None,
    "Obj_rel_error_pct": None,
}


# =============================================================================
# PIPELINE (LINEAR)
# =============================================================================

# Keep these in global scope so you can inspect them in the debugger even if it fails mid-way
n_smspp = None
network = None
transformation = None
df_metrics = None
timer_rows = []

try:
    # -------- Load network --------
    n_smspp = pypsa.Network(str(NETWORK_NC))

    if DO_CLEAN_E_SUM:
        n_smspp = clean_e_sum(n_smspp)

    if DO_CLEAN_CICLICITY_STORAGE:
        n_smspp = clean_ciclicity_storage(n_smspp)

    if DO_REDUCE_SNAPSHOTS:
        n_smspp = reduce_snapshots_and_scale_costs(n_smspp, target=REDUCE_SNAPSHOTS_TO, scale_capital_costs=False)

    if DO_ADD_SLACK_UNIT:
        n_smspp = add_slack_unit(n_smspp)

    # Optional cleanups (uncomment if needed)
    if DO_CLEAN_STORAGE_UNITS:
        n_smspp = clean_storage_units(n_smspp)
    if DO_CLEAN_STORES:
        n_smspp = clean_stores(n_smspp, remove_store_buses=REMOVE_STORE_BUSES, remove_generators_on_removed_buses=REMOVE_GENERATORS_ON_REMOVED_BUSES)
    
    if DO_CLEAN_GLOBAL_CONSTRAINTS:
# %%
        n_smspp = clean_global_constraints(n_smspp)

    # -------- PyPSA optimization (reference) --------
    network = n_smspp.copy()
    network.optimize(
        solver_name=SOLVER_NAME,
        solver_options=SOLVER_OPTIONS,
        # linearized_unit_commitment=True,
    )

    if EXPORT_PYPSA_LP:
        try:
            network.model.to_file(fn=str(PYPSA_LP))
        except Exception:
            pass

    if EXPORT_PYPSA_NC:
        try:
            network.export_to_netcdf(str(PYPSA_OUT_NC))
        except Exception:
            pass

    obj_pypsa = pypsa_reference_objective(network)
    metrics["Obj_PyPSA"] = obj_pypsa

    # -------- SMS++ pipeline (ONE CALL) --------
    transformation = Transformation(
        capacity_expansion_ucblock=CAPACITY_EXPANSION_UCBLOCK,
        enable_thermal_units=ENABLE_THERMAL_UNITS,
        intermittent_carriers=INTERMITTENT_CARRIERS,
        merge_links=MERGE_LINKS,
        merge_selector=MERGE_SELECTOR,
        workdir=CASE_DIR,
        name=CASE_NAME,
        overwrite=True,
        fp_temp=FP_TEMP,
        fp_log=FP_LOG,
        fp_solution=FP_SOLUTION,
        configfile=CONFIGFILE,
        pysmspp_options=PYSMSSP_OPTIONS,
    )

    n_smspp = transformation.run(n_smspp, verbose=VERBOSE)

    obj_smspp = float(transformation.result.objective_value)
    metrics["Obj_SMSpp"] = obj_smspp
    if obj_pypsa != 0.0:
        metrics["Obj_rel_error_pct"] = (obj_pypsa - obj_smspp) / obj_pypsa * 100.0

    # -------- Print quick sanity info --------
    print("\n[DEBUG] Network sizes after SMS++ inverse transformation:")
    print("  buses:", len(getattr(n_smspp, "buses", [])))
    print("  generators:", len(getattr(n_smspp, "generators", [])))
    print("  loads:", len(getattr(n_smspp, "loads", [])))
    print("  lines:", len(getattr(n_smspp, "lines", [])))
    print("  links:", len(getattr(n_smspp, "links", [])))
    print("  stores:", len(getattr(n_smspp, "stores", [])))
    print("  storage_units:", len(getattr(n_smspp, "storage_units", [])))

    # -------- Export statistics --------
    if EXPORT_STATISTICS_CSV:
        try:
            stats_pypsa = network.statistics()
            if hasattr(stats_pypsa, "to_frame"):
                stats_pypsa = stats_pypsa.to_frame(name="value")
            stats_pypsa.to_csv(STATS_PYPSA_CSV)
        except Exception as e:
            print("[WARN] Could not export PyPSA statistics:", e)

        try:
            stats_smspp = n_smspp.statistics()
            if hasattr(stats_smspp, "to_frame"):
                stats_smspp = stats_smspp.to_frame(name="value")
            stats_smspp.to_csv(STATS_SMSPP_CSV)
        except Exception as e:
            print("[WARN] Could not export SMS++ statistics:", e)

    # -------- Export repopulated network --------
    if EXPORT_SMSPP_REPOPULATED_NC:
        try:
            n_smspp.export_to_netcdf(str(SMSPP_REPOP_NC))
        except Exception:
            pass

    # -------- Timings (from Transformation.timer) --------
    timer_rows = getattr(getattr(transformation, "timer", None), "rows", None) or []

    try:
        if timer_rows:
            pd.DataFrame(timer_rows).to_csv(TIMINGS_CSV, index=False)
    except Exception:
        pass

    for r in timer_rows:
        step_name = r.get("step", "unknown")
        elapsed_s = r.get("elapsed_s", None)
        metrics[f"time__{step_name}"] = elapsed_s

    metrics["SMSpp_total_s"] = sum(
        float(r.get("elapsed_s", 0.0)) for r in timer_rows if r.get("elapsed_s") is not None
    )

    if timer_rows:
        print("\nTimings (from Transformation.timer):")
        for r in timer_rows:
            try:
                print(f"  - {r.get('step')}: {round(float(r.get('elapsed_s', 0.0)), 6)} s")
            except Exception:
                print(f"  - {r.get('step')}: {r.get('elapsed_s')} s")
        print(f"  - TOTAL: {round(metrics['SMSpp_total_s'], 6)} s")

    print_kv(
        {
            "Obj_PyPSA": metrics["Obj_PyPSA"],
            "Obj_SMSpp": metrics["Obj_SMSpp"],
            "Obj_rel_error_pct": metrics["Obj_rel_error_pct"],
            "SMSpp_total_s": metrics.get("SMSpp_total_s"),
        },
        title="\nMetrics:",
    )

except Exception as e:
    metrics["status"] = "FAIL"
    metrics["error_msg"] = f"{type(e).__name__}: {e}"
    print(f"\n!!! FAILED: {metrics['error_msg']}")
    traceback.print_exc()

finally:
    # Always build the single-row DF and save, even on failure
    df_metrics = pd.DataFrame([metrics])
    try:
        df_metrics.to_csv(DEBUG_SUMMARY_CSV, index=False)
    except Exception:
        pass

print("\n>>> Wrote per-case artifacts to:", CASE_DIR)