# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 18:18:41 2025

@author: aless
"""

"""
Batch tests for pypsa2smspp on multiple networks.

- For each provided .nc network path:
  * Detect "investment" mode if filename contains 'inv' (case-insensitive).
  * Optimize in PyPSA.
  * Transform to SMS++ blocks and optimize with UCBlock or InvestmentBlock.
  * Build output filenames in test/output/ with informative stems.
  * Collect timings and key metrics in a nested dict (defaults to NA).
  * At the end, dump a CSV of timings per network.

Comments are in English. Script output is concise.
"""

# --- Force working directory to this file's folder and build robust paths ---
import os, sys, json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # .../pypsa2smspp/test
os.chdir(HERE)                                  # force CWD regardless of VSCode
print(">>> FORCED CWD:", Path.cwd())

# Ensure PYTHONPATH for imports from repo root (e.g., scripts/)
REPO_ROOT = HERE.parent                          # .../pypsa2smspp
SCRIPTS = (REPO_ROOT / "scripts").resolve()
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Safe output directory (create if missing, check writability)
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)
if not os.access(OUT, os.W_OK):
    raise PermissionError(f"Output dir not writable: {OUT} (check owner/perms)")

# --- Imports that depend on repo environment ---
import pypsa
import pysmspp
from pypsa2smspp.transformation import Transformation

from pypsa2smspp.network_correction import (
    clean_marginal_cost,
    clean_global_constraints,
    clean_e_sum,
    clean_efficiency_link,
    clean_ciclicity_storage,
    clean_marginal_cost_intermittent,
    clean_storage_units,
    clean_stores,
    reduce_snapshots_and_scale_costs,
    parse_txt_file,
    compare_networks,
    add_slack_unit,
    from_investment_to_uc
)

NA = np.nan  # default NA for fields

def safe_stem(p: Path) -> str:
    """Return a clean stem to embed into filenames (no spaces or odd chars)."""
    s = p.stem
    return (
        s.replace(" ", "_")
         .replace("/", "_")
         .replace("\\", "_")
         .replace(":", "_")
    )

def default_result_dict() -> Dict[str, Any]:
    """Base structure with NA defaults; will be overwritten when known."""
    return {
        "mode": NA,                       # "UCBlock" | "InvestmentBlock"
        "is_investment_name": NA,         # bool by filename heuristic
        "objective_pypsa": NA,
        "objective_smspp": NA,
        "error_pct": NA,                  # (pypsa - smspp)/pypsa * 100
        "files": {
            "temporary_smspp_file": NA,
            "solution_file": NA,
            "log_txt": NA,
        },
        "times": {                        # seconds
            "PyPSA": NA,
            "Direct transformation": NA,
            "PySMSpp conversion": NA,
            "SMS++ (solver+writing)": NA,
            "SMS++ solver": NA,
            "SMS++ writing": NA,
            "Inverse transformation": NA,
        },
    }

def clean_network_for_experiment(n: pypsa.Network, investment_bool: bool) -> pypsa.Network:
    """Apply standard cleanups and add slack; mirror your current single-test logic."""
    n = clean_global_constraints(n)
    n = clean_e_sum(n)
    n = clean_ciclicity_storage(n)
    # n = reduce_snapshots_and_scale_costs(n, 240)
    # n = clean_storage_units(n)
    n = clean_stores(n)

    if investment_bool:
        n = add_slack_unit(n)
        n.links.p_nom_extendable = True
    else:
        n.generators.p_nom_extendable = False
        n.lines.s_nom_extendable = False
        n.lines.s_nom *= 1.5
        n = add_slack_unit(n)

    return n

def run_one_network(network_path: Path, merge_links: bool = True, expansion_ucblock: bool = False) -> Dict[str, Any]:
    """Run the full pipeline on one network file and return a result dict."""
    result = default_result_dict()
    name_stem = safe_stem(network_path)
    is_inv = ("inv" in name_stem.lower())
    result["is_investment_name"] = bool(is_inv)

    # Load and clean network
    n_smspp = pypsa.Network(str(network_path))
    investment_bool = is_inv  # temporary heuristic as requested
    n_smspp = clean_network_for_experiment(n_smspp, investment_bool=investment_bool)
    network = n_smspp.copy()

    # --- PyPSA optimize ---
    then = datetime.now()
    network.optimize(solver_name="gurobi")
    result["times"]["PyPSA"] = (datetime.now() - then).total_seconds()

    # --- Direct transform + blocks conversion ---
    then = datetime.now()
    transformation = Transformation(network, merge_links=merge_links, expansion_ucblock=expansion_ucblock)
    result["times"]["Direct transformation"] = (datetime.now() - then).total_seconds()

    then = datetime.now()
    tran = transformation.convert_to_blocks()
    result["times"]["PySMSpp conversion"] = (datetime.now() - then).total_seconds()

    # Decide UC vs Investment based on dimensions or flag (mirrors your logic)
    use_ucblock = (
        transformation.dimensions['InvestmentBlock']['NumAssets'] == 0 
        or transformation.expansion_ucblock
    )

    if use_ucblock:
        # --- UCBlock configuration ---
        result["mode"] = "UCBlock"
        configfile = pysmspp.SMSConfig(template="UCBlock/uc_solverconfig")
        temporary_smspp_file = OUT / f"network_ucblock_{name_stem}.nc"
        solution_file = OUT / f"solution_ucblock_{name_stem}.nc"
        log_txt = OUT / f"log_ucblock_{name_stem}.txt"
    else:
        # --- InvestmentBlock configuration ---
        result["mode"] = "InvestmentBlock"
        configfile = pysmspp.SMSConfig(template="InvestmentBlock/BSPar.txt")
        temporary_smspp_file = OUT / f"network_inv_{name_stem}.nc"
        solution_file = OUT / f"solution_inv_{name_stem}.nc"
        log_txt = OUT / f"log_investment_{name_stem}.txt"

    # Ensure old solution is removed
    if solution_file.exists():
        solution_file.unlink()

    # Record filenames
    result["files"]["temporary_smspp_file"] = str(temporary_smspp_file)
    result["files"]["solution_file"] = str(solution_file)
    result["files"]["log_txt"] = str(log_txt)

    # --- Optimize with SMS++ ---
    then = datetime.now()
    smspp_result = tran.optimize(
        configfile,
        str(temporary_smspp_file),
        str(log_txt),
        str(solution_file),
        inner_block_name=('InvestmentBlock' if result["mode"] == "InvestmentBlock" else None),
        log_executable_call=True
    )
    result["times"]["SMS++ (solver+writing)"] = (datetime.now() - then).total_seconds()

    # --- Objective comparison ---
    if result["mode"] == "UCBlock":
        stats = network.statistics()
        op_cost = stats['Operational Expenditure'].sum()
        obj_pypsa = float(op_cost)
        obj_smspp = float(smspp_result.objective_value)
    else:
        obj_pypsa = float(network.objective)
        obj_smspp = float(smspp_result.objective_value)

    result["objective_pypsa"] = obj_pypsa
    result["objective_smspp"] = obj_smspp
    result["error_pct"] = (obj_pypsa - obj_smspp) / obj_pypsa * 100 if obj_pypsa != 0 else NA

    # --- Parse solver time from log, split writing vs solver ---
    try:
        data_dict = parse_txt_file(str(log_txt))
        solver_time = float(data_dict.get("elapsed_time", NA))
        result["times"]["SMS++ solver"] = solver_time
        total = result["times"]["SMS++ (solver+writing)"]
        result["times"]["SMS++ writing"] = (total - solver_time) if (isinstance(total, (int, float)) and isinstance(solver_time, (int, float))) else NA
    except Exception as e:
        # Keep NA if parsing fails
        pass

    # --- Parse solution back and inverse transform ---
    then = datetime.now()
    _ = transformation.parse_solution_to_unitblocks(smspp_result.solution, n_smspp)
    transformation.inverse_transformation(n_smspp)
    result["times"]["Inverse transformation"] = (datetime.now() - then).total_seconds()

    return result

def flatten_times_for_csv(batch_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Build a tidy DataFrame with one row per network and columns for timings & metadata."""
    rows = []
    for net_key, res in batch_results.items():
        row = {
            "network": net_key,
            "mode": res.get("mode", NA),
            "is_investment_name": res.get("is_investment_name", NA),
            "objective_pypsa": res.get("objective_pypsa", NA),
            "objective_smspp": res.get("objective_smspp", NA),
            "error_pct": res.get("error_pct", NA),
            "temporary_smspp_file": res.get("files", {}).get("temporary_smspp_file", NA),
            "solution_file": res.get("files", {}).get("solution_file", NA),
            "log_txt": res.get("files", {}).get("log_txt", NA),
        }
        # Merge timing fields
        for k, v in res.get("times", {}).items():
            row[f"time_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)

def run_batch(network_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """Run many networks; return nested dict keyed by network stem or short name."""
    results: Dict[str, Dict[str, Any]] = {}
    for path_str in network_paths:
        p = Path(path_str).resolve()
        if not p.exists():
            print(f"[WARN] Network not found: {p}")
            # Keep a placeholder row with NA values
            key = safe_stem(p)
            results[key] = default_result_dict()
            results[key]["files"]["temporary_smspp_file"] = str(OUT / f"missing_{key}.nc")
            continue

        print(f"\n=== Running: {p.name} ===")
        res = run_one_network(p)
        key = safe_stem(p)
        results[key] = res
        print(f"  -> mode: {res['mode']} | error: {res['error_pct']}%")
    return results

if __name__ == "__main__":
    # --- Example input list (edit as needed) --------------------------------
    # Provide absolute or relative paths to your .nc networks here.
    NETWORKS = [
        "networks/base_s_2_elec_1h_inv.nc",
        # "networks/another_network_uc.nc",
        # "networks/demo_inv_240_bat.nc",
    ]
    # ------------------------------------------------------------------------

    batch_results = run_batch(NETWORKS)

    # Save CSV with flattened timings/metadata
    df = flatten_times_for_csv(batch_results)
    csv_path = OUT / "batch_times.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n>>> Wrote timings CSV: {csv_path}")

    # Optionally save the full nested dict to JSON for later inspection
    json_path = OUT / "batch_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(batch_results, f, indent=2)
    print(f">>> Wrote full results JSON: {json_path}")
