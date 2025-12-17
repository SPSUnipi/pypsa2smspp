# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:23:55 2025

@author: aless
"""

# -*- coding: utf-8 -*-
"""
Batch SMS++ benchmark runner over multiple PyPSA networks (.nc).

Scans networks/bench/*.nc and runs the full pipeline:
PyPSA optimization -> Transformation -> SMS++ solve -> inverse transform,
measuring timings and collecting objective gap vs PyPSA.

Outputs:
- Per-case artifacts in output/test/
- CSV summary at output/test/bench_summary_nc.csv

Author: aless (adapted)
"""

import os, sys, traceback
from pathlib import Path
from datetime import datetime
import time
import pandas as pd
import pypsa
import pysmspp
import re

# --- Force working directory to this file's folder and build robust paths ---
HERE = Path(__file__).resolve().parent            # .../pypsa2smspp/test
os.chdir(HERE)                                    # force CWD regardless of VSCode
print(">>> FORCED CWD:", Path.cwd())

# Ensure PYTHONPATH for imports from repo root (e.g., scripts/)
REPO_ROOT = HERE.parent                           # .../pypsa2smspp
SCRIPTS = (REPO_ROOT / "scripts").resolve()
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Safe output dirs
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)
OUT_TEST = OUT / "test_pypsaeur"
OUT_TEST.mkdir(parents=True, exist_ok=True)

# --- Domain imports (after PYTHONPATH is set) ---
from pypsa2smspp.transformation import Transformation

from pypsa2smspp.network_correction import (
    clean_global_constraints,
    clean_e_sum,
    clean_ciclicity_storage,
    clean_stores,
    parse_txt_file,
    add_slack_unit,
    # clean_storage_units,  # add if you want
    # reduce_snapshots_and_scale_costs,
    # compare_networks,
)

# ---------- Utilities ----------
def t_now() -> float:
    """High-resolution time in seconds."""
    return time.perf_counter()

def delta_s(start: float) -> float:
    """Seconds elapsed since start() using perf_counter."""
    return time.perf_counter() - start

def safe_remove(p: Path):
    """Remove path if exists."""
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


# ---------- Core runner for a single .nc network ----------
def run_single_nc(nc_path: Path,
                  solver_name: str = "gurobi",
                  uc_template: str = "UCBlock/uc_solverconfig_grb",       # adjust to *_grb if you want
                  inv_template: str = "InvestmentBlock/BSPar.txt",
                  merge_links: bool = True,
                  expansion_ucblock: bool = True) -> dict:
    """
    Run the full flow for a given PyPSA network (.nc file).

    Returns a dict with timings, status, and metrics to be appended to summary CSV.
    """
    case_name = nc_path.stem   # e.g., base_s_5_elec_1h
    print(f"\n=== Running NC case: {case_name} ({nc_path.name}) ===")

    # Build per-case output paths
    prefix = f"{case_name}"
    network_nc = OUT_TEST / f"network_{prefix}.nc"       # PyPSA network after optimization
    smspp_tmp_nc = OUT_TEST / f"tmp_smspp_{prefix}.nc"   # temp write during optimize
    smspp_solution_nc = OUT_TEST / f"solution_{prefix}.nc"
    smspp_log_txt = OUT_TEST / f"log_{prefix}.txt"
    pypsa_lp = OUT_TEST / f"pypsa_{prefix}.lp"

    # Clean any stale files
    for p in (network_nc, smspp_tmp_nc, smspp_solution_nc, smspp_log_txt, pypsa_lp):
        safe_remove(p)

    summary = {
        "case": case_name,
        "input_file": str(nc_path),
        "status": "OK",
        "error_msg": "",
        "PyPSA_opt_s": None,
        "SMSpp_total_s": None,
        "Transform_direct_s": None,
        "PySMSpp_convert_s": None,
        "SMSpp_solver_write_s": None,
        "SMSpp_solver_s": None,
        "SMSpp_write_s": None,
        "Inverse_transform_s": None,
        "Obj_PyPSA": None,
        "Obj_SMSpp": None,
        "Obj_rel_error_pct": None,
        "Investment_mode": None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        # -------- Load network --------
        n_smspp = pypsa.Network(str(nc_path))

        # Optional cleanups (adapt as in your old script)
        n_smspp = clean_ciclicity_storage(n_smspp)

        # Keep a working copy for PyPSA optimization (do not mutate original)
        network = n_smspp.copy()

        # ---- (1) PyPSA optimization ----
        t0 = t_now()
        network.optimize(solver_name=solver_name)
        summary["PyPSA_opt_s"] = round(delta_s(t0), 6)

        # Optional: export LP for debugging
        try:
            network.model.to_file(fn=str(pypsa_lp))
        except Exception:
            pass

        # ---- (2) Direct transformation ----
        t0 = t_now()
        transformation = Transformation(network,
                                        merge_links=merge_links,
                                        expansion_ucblock=expansion_ucblock)
        summary["Transform_direct_s"] = round(delta_s(t0), 6)

        # ---- (3) Conversion to SMS++ blocks (PySMSpp object graph) ----
        t0 = t_now()
        tran = transformation.convert_to_blocks()
        summary["PySMSpp_convert_s"] = round(delta_s(t0), 6)

        # Determine which block to run (same logic as Excel batch)
        is_investment = (not transformation.expansion_ucblock) and \
                        (transformation.dimensions['InvestmentBlock']['NumAssets'] > 0)
        summary["Investment_mode"] = bool(is_investment)

        # ---- (4) SMS++ optimization ----
        if not is_investment:
            # UCBlock configuration
            configfile = pysmspp.SMSConfig(template=uc_template)
            tmp_nc = str(smspp_tmp_nc)
            out_txt = str(smspp_log_txt)
            sol_nc = str(smspp_solution_nc)

            t0 = t_now()
            result = tran.optimize(configfile, tmp_nc, out_txt, sol_nc, log_executable_call=True)
            total_smspp = delta_s(t0)
            summary["SMSpp_solver_write_s"] = round(total_smspp, 6)

            # Parse solver time from log (if available)
            try:
                d = parse_txt_file(out_txt)
                solver_s = float(d.get("elapsed_time", 0.0))
                summary["SMSpp_solver_s"] = round(solver_s, 6)
                summary["SMSpp_write_s"] = round(total_smspp - solver_s, 6)
            except Exception:
                summary["SMSpp_solver_s"] = None
                summary["SMSpp_write_s"] = None

            # Objectives & error
            try:
                obj_pypsa = float(network.objective + network.objective_constant)
            except Exception:
                obj_pypsa = float(network.objective)

            obj_smspp = float(result.objective_value)
            summary["Obj_PyPSA"] = obj_pypsa
            summary["Obj_SMSpp"] = obj_smspp
            if obj_pypsa != 0.0:
                summary["Obj_rel_error_pct"] = round((obj_pypsa - obj_smspp) / obj_pypsa * 100.0, 8)

            # Inverse transform
            t0 = t_now()
            _ = transformation.parse_solution_to_unitblocks(result.solution, n_smspp)
            transformation.inverse_transformation(n_smspp)
            summary["Inverse_transform_s"] = round(delta_s(t0), 6)

        else:
            # InvestmentBlock configuration
            configfile = pysmspp.SMSConfig(template=inv_template)
            tmp_nc = str(smspp_tmp_nc)
            out_txt = str(smspp_log_txt)
            sol_nc = str(smspp_solution_nc)

            t0 = t_now()
            result = tran.optimize(configfile, tmp_nc, out_txt, sol_nc,
                                   inner_block_name='InvestmentBlock',
                                   log_executable_call=True,
                                   logging=False)
            total_smspp = delta_s(t0)
            summary["SMSpp_solver_write_s"] = round(total_smspp, 6)

            # Try to split solver vs writing from log
            try:
                d = parse_txt_file(out_txt)
                solver_s = float(d.get("elapsed_time", 0.0))
                summary["SMSpp_solver_s"] = round(solver_s, 6)
                summary["SMSpp_write_s"] = round(total_smspp - solver_s, 6)
            except Exception:
                pass

            obj_pypsa = float(network.objective)   # objective_constant usually 0 here
            obj_smspp = float(result.objective_value)
            summary["Obj_PyPSA"] = obj_pypsa
            summary["Obj_SMSpp"] = obj_smspp
            if obj_pypsa != 0.0:
                summary["Obj_rel_error_pct"] = round((obj_pypsa - obj_smspp) / obj_pypsa * 100.0, 8)

            # Inverse transform
            t0 = t_now()
            _ = transformation.parse_solution_to_unitblocks(result.solution, n_smspp)
            transformation.inverse_transformation(n_smspp)
            summary["Inverse_transform_s"] = round(delta_s(t0), 6)

        # ---- (5) Save PyPSA network after optimization ----
        try:
            network.export_to_netcdf(str(network_nc))
        except Exception:
            pass

        # Compute total SMS++ pipeline time (excluding PyPSA)
        if summary["Transform_direct_s"] is not None and \
           summary["PySMSpp_convert_s"] is not None and \
           summary["SMSpp_solver_write_s"] is not None and \
           summary["Inverse_transform_s"] is not None:
            summary["SMSpp_total_s"] = round(
                summary["Transform_direct_s"]
                + summary["PySMSpp_convert_s"]
                + summary["SMSpp_solver_write_s"]
                + summary["Inverse_transform_s"],
                6,
            )

        print(f"=== Done NC: {case_name} | Obj_err%: {summary['Obj_rel_error_pct']} | SMS++ total s: {summary['SMSpp_total_s']} ===")
        return summary

    except Exception as e:
        summary["status"] = "FAIL"
        summary["error_msg"] = f"{type(e).__name__}: {e}"
        print(f"!!! FAILED NC {case_name}: {summary['error_msg']}")
        traceback.print_exc()
        return summary


def main():
    # Discover inputs (.nc networks)
    # You can change this folder as you like, e.g. HERE / "networks"
    # inputs_dir = Path("/home/pampado/sector-coupled/pypsa-eur/resources/smspp_electricity_only_italy/networks")
    inputs_dir = Path("/home/pampado/sector-coupled/pypsa-eur-smspp/resources/smspp/networks")
    nc_files = sorted(inputs_dir.glob("*h*.nc"))

    # Regex per catturare il numero dopo s_
    #pattern = re.compile(r"s_(\d+)")

    #def extract_cluster_number(path):
    #    m = pattern.search(path.name)
    #    return int(m.group(1)) if m else float("inf")  # se manca, mettilo in fondo

    #nc_files = sorted(
    #    inputs_dir.glob("*h*.nc"),
    #    key=extract_cluster_number
    #)

    if not nc_files:
        print(f"No .nc file in {inputs_dir}")
        return

    rows = []
    for nc in nc_files:
        # Same logic as Excel script: filenames containing "inv" are investment cases
        expansion_ucblock = False if "inv" in nc.name else True
        row = run_single_nc(nc, expansion_ucblock=expansion_ucblock)
        rows.append(row)

    # Build DataFrame and save CSV
    df = pd.DataFrame(rows)
    csv_path = OUT_TEST / "bench_summary_pypsaeur.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n>>> Summary (NC) written to: {csv_path}")
    try:
        with pd.option_context('display.max_columns', None, 'display.width', 140):
            print(df)
    except Exception:
        pass


if __name__ == "__main__":
    main()
