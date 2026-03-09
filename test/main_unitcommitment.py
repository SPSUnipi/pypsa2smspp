# -*- coding: utf-8 -*-
"""
Single-case debug runner for PyPSA -> SMS++ (via pypsa2smspp), with double PyPSA solve:
- PyPSA optimize twice:
    (1) linearized_unit_commitment = True
    (2) linearized_unit_commitment = False
- Save PyPSA .nc and .lp in two separate subfolders:
    output/unit_commitment/<case_name>/linearized_unit_commitment/
    output/unit_commitment/<case_name>/unit_commitment/

Then run the SMS++ pipeline (one call) using the non-linearized UC solution by default.

Linear script style:
- All inputs/toggles at the top
- No main(), no wrapper functions returning only a subset of stuff
- Variables remain in global scope so you can inspect everything in a debugger
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
NETWORK_NC = Path(
    r"/home/pampado/sector-coupled/pypsa-eur-smspp/resources/unit_commitment_no_linearized_smspp_italy/networks/base_s_5_elec_.nc"
)

# Output (NEW ROOT)
OUT_ROOT = Path("output/unit_commitment")
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

# Which PyPSA variant should be used as input to SMS++ pipeline?
# - "unit_commitment" -> linearized_unit_commitment=False (default)
# - "linearized_unit_commitment" -> linearized_unit_commitment=True
SMSPP_INPUT_VARIANT = "unit_commitment"

# Transformation toggles
CAPACITY_EXPANSION_UCBLOCK = True      # True -> UCBlock, False -> InvestmentBlock
ENABLE_THERMAL_UNITS = True            # False -> everything (except slack) treated as intermittent
INTERMITTENT_CARRIERS = None           # None -> default renewable_carriers; list/str -> override
MERGE_LINKS = True                     # False / True / ["tes","battery","h2", ...]
MERGE_SELECTOR = None                  # optional callable (only needed for custom merge tags)

# SMS++ artifacts (rendered with {name} and placed in CASE_DIR)
FP_TEMP = "smspp_{name}_temp.nc"
FP_LOG = "smspp_{name}_log.txt"
FP_SOLUTION = "smspp_{name}_solution.nc"

# SMS++ config selection (no YAML)
CONFIGFILE = "auto"

# Optional: pass-through options for pySMSpp (all optional; defaults exist in pySMSpp)
PYSMSSP_OPTIONS = {}

# Cleaning toggles
DO_CLEAN_E_SUM = False
DO_CLEAN_CICLICITY_STORAGE = True
DO_ADD_SLACK_UNIT = False
DO_REDUCE_SNAPSHOTS = False
REDUCE_SNAPSHOTS_TO = 24

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
)


# =============================================================================
# SMALL UTILITIES
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


def pypsa_optimize_with_linearized_uc(
    network: "pypsa.Network",
    *,
    solver_name: str,
    solver_options: Dict[str, Any],
    linearized_unit_commitment: bool,
) -> None:
    """
    Try to call PyPSA optimize with linearized_unit_commitment flag.
    If unsupported by the installed PyPSA version, fallback to optimize without the flag.
    """
    try:
        network.optimize(
            solver_name=solver_name,
            solver_options=solver_options,
            linearized_unit_commitment=linearized_unit_commitment,
        )
    except TypeError as e:
        msg = str(e)
        if "linearized_unit_commitment" in msg:
            print(
                "[WARN] Your PyPSA version does not accept `linearized_unit_commitment` in optimize(). "
                "Falling back to optimize() without that flag."
            )
            network.optimize(
                solver_name=solver_name,
                solver_options=solver_options,
            )
        else:
            raise


# =============================================================================
# DERIVED SETTINGS & OUTPUT PATHS
# =============================================================================

if CASE_NAME is None:
    CASE_NAME = NETWORK_NC.stem

CASE_DIR = OUT_ROOT / CASE_NAME
ensure_dir(CASE_DIR)

# Subfolders for the two PyPSA variants
DIR_LIN = CASE_DIR / "linearized_unit_commitment"
DIR_UC = CASE_DIR / "unit_commitment"
ensure_dir(DIR_LIN)
ensure_dir(DIR_UC)

# PyPSA artifacts (two variants)
PYPSA_LP_LIN = DIR_LIN / f"pypsa_{CASE_NAME}.lp"
PYPSA_NC_LIN = DIR_LIN / f"network_pypsa_{CASE_NAME}.nc"

PYPSA_LP_UC = DIR_UC / f"pypsa_{CASE_NAME}.lp"
PYPSA_NC_UC = DIR_UC / f"network_pypsa_{CASE_NAME}.nc"

# SMS++ / debug artifacts (shared at CASE_DIR)
SMSPP_REPOP_NC = CASE_DIR / f"network_smspp_{CASE_NAME}.nc"
DEBUG_SUMMARY_CSV = CASE_DIR / "debug_summary_row.csv"
STATS_PYPSA_CSV_LIN = DIR_LIN / f"stats_pypsa_{CASE_NAME}.csv"
STATS_PYPSA_CSV_UC = DIR_UC / f"stats_pypsa_{CASE_NAME}.csv"
STATS_SMSPP_CSV = CASE_DIR / f"stats_smspp_{CASE_NAME}.csv"
TIMINGS_CSV = CASE_DIR / f"timings_{CASE_NAME}.csv"

# Clean stale files
for p in (
    PYPSA_LP_LIN, PYPSA_NC_LIN,
    PYPSA_LP_UC, PYPSA_NC_UC,
    SMSPP_REPOP_NC,
    DEBUG_SUMMARY_CSV,
    STATS_PYPSA_CSV_LIN,
    STATS_PYPSA_CSV_UC,
    STATS_SMSPP_CSV,
    TIMINGS_CSV,
):
    safe_remove(p)

# Also clean SMS++ artifacts (since overwrite behaviour is inside Transformation.optimize)
safe_remove(CASE_DIR / FP_TEMP.format(name=CASE_NAME))
safe_remove(CASE_DIR / FP_LOG.format(name=CASE_NAME))
safe_remove(CASE_DIR / FP_SOLUTION.format(name=CASE_NAME))

print(f"\n=== DEBUG RUN (DOUBLE PyPSA UC): {CASE_NAME} ===")
print(f"Input network: {NETWORK_NC}")
print(f"Output folder: {CASE_DIR}")
print(f"PyPSA variants: {DIR_LIN.name} / {DIR_UC.name}")

metrics: Dict[str, Any] = {
    "case": CASE_NAME,
    "input_file": str(NETWORK_NC),
    "pypsa_solver": SOLVER_NAME,
    "status": "OK",
    "error_msg": "",
    "Obj_PyPSA_linearized": None,
    "Obj_PyPSA_unit_commitment": None,
    "Obj_SMSpp": None,
    "Obj_rel_error_pct__vs_unit_commitment": None,
    "Obj_rel_error_pct__vs_linearized": None,
    "smspp_input_variant": SMSPP_INPUT_VARIANT,
}


# =============================================================================
# PIPELINE (LINEAR)
# =============================================================================

n_base = None
network_lin = None
network_uc = None
network_for_smspp = None
n_smspp = None
transformation = None
df_metrics = None
timer_rows = []

try:
    # -------- Load base network --------
    n_base = pypsa.Network(str(NETWORK_NC))

    if DO_CLEAN_E_SUM:
        n_base = clean_e_sum(n_base)

    if DO_CLEAN_CICLICITY_STORAGE:
        n_base = clean_ciclicity_storage(n_base)

    if DO_REDUCE_SNAPSHOTS:
        n_base = reduce_snapshots_and_scale_costs(n_base, REDUCE_SNAPSHOTS_TO)

    if DO_ADD_SLACK_UNIT:
        n_base = add_slack_unit(n_base)

    # -------- PyPSA optimization #1: linearized_unit_commitment=True --------
    network_lin = n_base.copy()
    pypsa_optimize_with_linearized_uc(
        network_lin,
        solver_name=SOLVER_NAME,
        solver_options=SOLVER_OPTIONS,
        linearized_unit_commitment=True,
    )

    if EXPORT_PYPSA_LP:
        try:
            network_lin.model.to_file(fn=str(PYPSA_LP_LIN))
        except Exception:
            pass

    if EXPORT_PYPSA_NC:
        try:
            network_lin.export_to_netcdf(str(PYPSA_NC_LIN))
        except Exception:
            pass

    metrics["Obj_PyPSA_linearized"] = pypsa_reference_objective(network_lin)

    if EXPORT_STATISTICS_CSV:
        try:
            stats_lin = network_lin.statistics()
            if hasattr(stats_lin, "to_frame"):
                stats_lin = stats_lin.to_frame(name="value")
            stats_lin.to_csv(STATS_PYPSA_CSV_LIN)
        except Exception as e:
            print("[WARN] Could not export PyPSA statistics (linearized):", e)

    # -------- PyPSA optimization #2: linearized_unit_commitment=False --------
    network_uc = n_base.copy()
    pypsa_optimize_with_linearized_uc(
        network_uc,
        solver_name=SOLVER_NAME,
        solver_options=SOLVER_OPTIONS,
        linearized_unit_commitment=False,
    )

    if EXPORT_PYPSA_LP:
        try:
            network_uc.model.to_file(fn=str(PYPSA_LP_UC))
        except Exception:
            pass

    if EXPORT_PYPSA_NC:
        try:
            network_uc.export_to_netcdf(str(PYPSA_NC_UC))
        except Exception:
            pass

    metrics["Obj_PyPSA_unit_commitment"] = pypsa_reference_objective(network_uc)

    if EXPORT_STATISTICS_CSV:
        try:
            stats_uc = network_uc.statistics()
            if hasattr(stats_uc, "to_frame"):
                stats_uc = stats_uc.to_frame(name="value")
            stats_uc.to_csv(STATS_PYPSA_CSV_UC)
        except Exception as e:
            print("[WARN] Could not export PyPSA statistics (unit_commitment):", e)

    # -------- Choose which PyPSA variant feeds SMS++ --------
    if SMSPP_INPUT_VARIANT == "linearized_unit_commitment":
        network_for_smspp = network_lin
    else:
        network_for_smspp = network_uc

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

    n_smspp = transformation.run(network_for_smspp, verbose=VERBOSE)

    obj_smspp = float(transformation.result.objective_value)
    metrics["Obj_SMSpp"] = obj_smspp

    obj_uc = metrics["Obj_PyPSA_unit_commitment"]
    obj_lin = metrics["Obj_PyPSA_linearized"]

    if obj_uc not in (None, 0.0):
        metrics["Obj_rel_error_pct__vs_unit_commitment"] = (obj_uc - obj_smspp) / obj_uc * 100.0
    if obj_lin not in (None, 0.0):
        metrics["Obj_rel_error_pct__vs_linearized"] = (obj_lin - obj_smspp) / obj_lin * 100.0

    # -------- Export SMS++ repopulated network --------
    if EXPORT_SMSPP_REPOPULATED_NC:
        try:
            n_smspp.export_to_netcdf(str(SMSPP_REPOP_NC))
        except Exception:
            pass

    # -------- Export SMS++ statistics --------
    if EXPORT_STATISTICS_CSV:
        try:
            stats_smspp = n_smspp.statistics()
            if hasattr(stats_smspp, "to_frame"):
                stats_smspp = stats_smspp.to_frame(name="value")
            stats_smspp.to_csv(STATS_SMSPP_CSV)
        except Exception as e:
            print("[WARN] Could not export SMS++ statistics:", e)

    # -------- Timings (from Transformation.timer) --------
    timer_rows = getattr(getattr(transformation, "timer", None), "rows", None) or []
    try:
        if timer_rows:
            pd.DataFrame(timer_rows).to_csv(TIMINGS_CSV, index=False)
    except Exception:
        pass

    for r in timer_rows:
        step_name = r.get("step", "unknown")
        metrics[f"time__{step_name}"] = r.get("elapsed_s", None)

    metrics["SMSpp_total_s"] = sum(
        float(r.get("elapsed_s", 0.0)) for r in timer_rows if r.get("elapsed_s") is not None
    )

    # -------- Console summary --------
    print("\n[DEBUG] Objectives:")
    print("  PyPSA linearized UC:", metrics["Obj_PyPSA_linearized"])
    print("  PyPSA unit commitment:", metrics["Obj_PyPSA_unit_commitment"])
    print("  SMS++:", metrics["Obj_SMSpp"])
    print("  Err% vs UC:", metrics["Obj_rel_error_pct__vs_unit_commitment"])
    print("  Err% vs LIN:", metrics["Obj_rel_error_pct__vs_linearized"])

    if timer_rows:
        print("\nTimings (from Transformation.timer):")
        for r in timer_rows:
            try:
                print(f"  - {r.get('step')}: {round(float(r.get('elapsed_s', 0.0)), 6)} s")
            except Exception:
                print(f"  - {r.get('step')}: {r.get('elapsed_s')} s")
        print(f"  - TOTAL: {round(metrics.get('SMSpp_total_s', 0.0), 6)} s")

except Exception as e:
    metrics["status"] = "FAIL"
    metrics["error_msg"] = f"{type(e).__name__}: {e}"
    print(f"\n!!! FAILED: {metrics['error_msg']}")
    traceback.print_exc()

finally:
    df_metrics = pd.DataFrame([metrics])
    try:
        df_metrics.to_csv(DEBUG_SUMMARY_CSV, index=False)
    except Exception:
        pass

print("\n>>> Wrote per-case artifacts to:", CASE_DIR)
print(">>> PyPSA folders:", DIR_LIN, "and", DIR_UC)