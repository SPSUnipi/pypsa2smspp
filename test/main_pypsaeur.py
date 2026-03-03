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
NETWORK_NC = Path(r"C:\Users\aless\sms\transformation_pypsa_smspp\test\networks\base_s_5___2050.nc")
CONFIG_YAML = Path(r"../pypsa2smspp/data/config_default.yaml")

# Output
OUT_ROOT = Path("output/debug")
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

# Cleaning toggles
DO_CLEAN_E_SUM = False
DO_CLEAN_CICLICITY_STORAGE = True
DO_ADD_SLACK_UNIT = False
DO_REDUCE_SNAPSHOTS = True
REDUCE_SNAPSHOTS_TO = 24
# DO_CLEAN_STORAGE_UNITS = False  # optional, kept off by default
# DO_CLEAN_STORES = False         # optional, kept off by default

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


print(f"\n=== DEBUG RUN: {CASE_NAME} ===")
print(f"Input network: {NETWORK_NC}")
print(f"Config YAML:   {CONFIG_YAML}")
print(f"Output folder: {CASE_DIR}")

metrics: Dict[str, Any] = {
    "case": CASE_NAME,
    "input_file": str(NETWORK_NC),
    "config_yaml": str(CONFIG_YAML),
    "pypsa_solver": SOLVER_NAME,
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
        n_smspp = reduce_snapshots_and_scale_costs(n_smspp, REDUCE_SNAPSHOTS_TO)

    if DO_ADD_SLACK_UNIT:
        n_smspp = add_slack_unit(n_smspp)

    # Optional cleanups (uncomment if needed)
    # if DO_CLEAN_STORAGE_UNITS:
    #     n_smspp = clean_storage_units(n_smspp)
    # if DO_CLEAN_STORES:
    #     n_smspp = clean_stores(n_smspp)

    # -------- PyPSA optimization (reference) --------
    network = n_smspp.copy()
    network.optimize(
        solver_name=SOLVER_NAME,
        solver_options=SOLVER_OPTIONS,
    )

    if EXPORT_PYPSA_LP:
        # Exports LP/MPS depending on PyPSA version; to_file uses the path suffix
        network.model.to_file(fn=str(PYPSA_LP))

    if EXPORT_PYPSA_NC:
        try:
            network.export_to_netcdf(str(PYPSA_OUT_NC))
        except Exception:
            pass

    obj_pypsa = pypsa_reference_objective(network)
    metrics["Obj_PyPSA"] = obj_pypsa

    # -------- SMS++ pipeline (ONE CALL) --------
    transformation = Transformation(name=CASE_NAME, workdir="output/test_pypsaeur")
    n_smspp = transformation.run(network, verbose=VERBOSE)

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
            print(f"  - {r.get('step')}: {round(float(r.get('elapsed_s', 0.0)), 6)} s")
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