# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 14:41:15 2025

@author: aless
"""

# -*- coding: utf-8 -*-
"""
Batch SMS++ benchmark runner over multiple Excel inputs.

Scans data/test/*.xlsx and runs the full pipeline:
PyPSA optimization -> Transformation -> SMS++ solve -> inverse transform,
measuring timings and collecting objective gap vs PyPSA.

Outputs:
- Per-case artifacts in output/test/
- CSV summary at output/test/bench_summary.csv

Author: aless (adapted)
"""

import os, sys, traceback
from pathlib import Path
from datetime import datetime
import time
import pandas as pd
import pypsa
import pysmspp

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
OUT_TEST = OUT / "test"
OUT_TEST.mkdir(parents=True, exist_ok=True)

# --- Domain imports (after PYTHONPATH is set) ---
from configs.test_config import TestConfig
from network_definition import NetworkDefinition
from pypsa2smspp.transformation import Transformation

from pypsa2smspp.network_correction import (
    clean_global_constraints,
    clean_e_sum,
    clean_ciclicity_storage,
    clean_stores,
    parse_txt_file,
    add_slack_unit,
    compare_networks,  # optional: not used in timings but kept for debugging
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

# ---------- Core runner for a single Excel input ----------
def run_single_case(xlsx_path: Path,
                    solver_name: str = "gurobi",
                    uc_template: str = "UCBlock/uc_solverconfig_grb",
                    inv_template: str = "InvestmentBlock/BSPar.txt",
                    merge_links: bool = True,
                    expansion_ucblock: bool = True) -> dict:
    """
    Run the full flow for a given Excel components file.

    Returns a dict with timings, status, and metrics to be appended to summary CSV.
    """
    case_name = xlsx_path.stem  # e.g., "components_caseA"
    print(f"\n=== Running case: {case_name} ({xlsx_path.name}) ===")

    # Build per-case output paths
    prefix = f"{case_name}"
    network_nc = OUT_TEST / f"network_{prefix}.nc"
    smspp_tmp_nc = OUT_TEST / f"tmp_smspp_{prefix}.nc"            # temp write during optimize
    smspp_solution_nc = OUT_TEST / f"solution_{prefix}.nc"
    smspp_log_txt = OUT_TEST / f"log_{prefix}.txt"
    pypsa_lp = OUT_TEST / f"pypsa_{prefix}.lp"

    # Clean any stale files
    for p in (network_nc, smspp_tmp_nc, smspp_solution_nc, smspp_log_txt, pypsa_lp):
        safe_remove(p)

    summary = {
        "case": case_name,
        "input_file": str(xlsx_path),
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
        # ---- Build config overriding parser to point at this Excel ----
        # We "monkey-patch" the parser fields so NetworkDefinition reads from our file.
        parser = TestConfig()
        parser.input_data_path = str(xlsx_path.parent)       # folder of the excel
        parser.input_name_components = xlsx_path.name        # excel filename

        if "sector" in xlsx_path.name:
            parser.load_sign = -1
        # ---- Build network from Excel via your NetworkDefinition pipeline ----
        nd = NetworkDefinition(parser)

        # Optional: clean-ups consistent with your other scripts
        n = nd.n
        n = clean_ciclicity_storage(n)
        if "sector" not in xlsx_path.name:
            n = add_slack_unit(n)
        

        # Keep a working copy for PyPSA optimization (do not mutate original)
        network = n.copy()

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

        # Determine which block to run
        is_investment = (not transformation.expansion_ucblock) and \
                        (transformation.dimensions['InvestmentBlock']['NumAssets'] > 0)
        summary["Investment_mode"] = bool(is_investment)

        # ---- (4) SMS++ optimization ----
        if not is_investment:
            # UCBlock configuration
            configfile = pysmspp.SMSConfig(template=uc_template)
            # The temporary eBlock file to write/solve
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
                # Fallback if parsing fails
                summary["SMSpp_solver_s"] = None
                summary["SMSpp_write_s"] = None

            # ---- Objective & error ----
            try:
                # PyPSA objective
                obj_pypsa = float(network.objective + network.objective_constant)
            except Exception:
                # In some versions, objective_constant may be zero or missing
                obj_pypsa = float(network.objective)

            obj_smspp = float(result.objective_value)
            summary["Obj_PyPSA"] = obj_pypsa
            summary["Obj_SMSpp"] = obj_smspp
            if obj_pypsa != 0.0:
                summary["Obj_rel_error_pct"] = round((obj_pypsa - obj_smspp) / obj_pypsa * 100.0, 8)

            # ---- (5) Parse solution & inverse transform ----
            t0 = t_now()
            _ = transformation.parse_solution_to_unitblocks(result.solution, n)
            transformation.inverse_transformation(result.objective_value, n)
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
                                   log_executable_call=True)
            total_smspp = delta_s(t0)
            summary["SMSpp_solver_write_s"] = round(total_smspp, 6)

            # No robust solver-time split here (template dependent), but try:
            try:
                d = parse_txt_file(out_txt)
                solver_s = float(d.get("elapsed_time", 0.0))
                summary["SMSpp_solver_s"] = round(solver_s, 6)
                summary["SMSpp_write_s"] = round(total_smspp - solver_s, 6)
            except Exception:
                pass

            # Objectives
            obj_pypsa = float(network.objective)  # often objective_constant already included or 0
            obj_smspp = float(result.objective_value)
            summary["Obj_PyPSA"] = obj_pypsa
            summary["Obj_SMSpp"] = obj_smspp
            if obj_pypsa != 0.0:
                summary["Obj_rel_error_pct"] = round((obj_pypsa - obj_smspp) / obj_pypsa * 100.0, 8)

            # Inverse
            t0 = t_now()
            _ = transformation.parse_solution_to_unitblocks(result.solution, n)
            transformation.inverse_transformation(result.objective_value, n)
            summary["Inverse_transform_s"] = round(delta_s(t0), 6)

        # ---- (6) Save final networks if needed ----
        try:
            # PyPSA network after optimization
            network.export_to_netcdf(str(network_nc))
            # The inverse-transformed network n could be exported similarly:
            # (Uncomment if you want a dedicated file)
            # (OUT_TEST / f"network_smspp_{prefix}.nc")
        except Exception:
            pass
        
        
        summary["SMSpp_total_s"] = summary["Transform_direct_s"] + summary["PySMSpp_convert_s"] + summary["SMSpp_solver_write_s"] + summary["Inverse_transform_s"]
        print(f"=== Done: {case_name} | Obj_err%: {summary['Obj_rel_error_pct']} | SMS++ total s: {summary['SMSpp_total_s']} ===")
        return summary

    except Exception as e:
        summary["status"] = "FAIL"
        summary["error_msg"] = f"{type(e).__name__}: {e}"
        print(f"!!! FAILED {case_name}: {summary['error_msg']}")
        traceback.print_exc()
        return summary


def main():
    # Discover inputs
    inputs_dir = HERE / "configs" / "data" / "test"
    xlsx_files = sorted(inputs_dir.glob("*.xlsx"))

    if not xlsx_files:
        print(f"Nessun .xlsx trovato in {inputs_dir}")
        return

    rows = []
    for x in xlsx_files:
        expansion_ucblock = False if "inv" in x.name else True
        row = run_single_case(x, expansion_ucblock=expansion_ucblock)
        rows.append(row)

    # Build DataFrame and save CSV
    df = pd.DataFrame(rows)
    csv_path = OUT_TEST / "bench_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n>>> Summary written to: {csv_path}")
    # Optional: pretty print
    try:
        # Simple terminal preview
        with pd.option_context('display.max_columns', None, 'display.width', 140):
            print(df)
    except Exception:
        pass


if __name__ == "__main__":
    main()
