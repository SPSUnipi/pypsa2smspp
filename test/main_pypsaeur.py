# -*- coding: utf-8 -*-
"""
Created on Wed May 28 16:25:30 2025

@author: aless
"""

"""
Single-case debug runner for PyPSA -> SMS++ (via pypsa2smspp).

Loads one PyPSA network (.nc), optionally applies cleaning steps,
runs PyPSA optimization, transforms to SMS++ blocks, solves with SMS++,
parses solution back, and (optionally) runs inverse transformation.

Outputs per-case artifacts into output/debug/<case_name>/.

Comments in English by request.
"""

import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import pypsa
import pysmspp

# --- Force working directory to this file's folder and build robust paths ---
HERE = Path(__file__).resolve().parent
os.chdir(HERE)
print(">>> FORCED CWD:", Path.cwd())

# Ensure PYTHONPATH for imports from repo root (e.g., scripts/)
REPO_ROOT = HERE.parent
SCRIPTS = (REPO_ROOT / "scripts").resolve()
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# --- Project imports (after sys.path adjustment) ---
from pypsa2smspp.transformation import Transformation
from pypsa2smspp.network_correction import (
    clean_global_constraints,
    clean_e_sum,
    clean_ciclicity_storage,
    clean_stores,
    parse_txt_file,
    add_slack_unit,
    reduce_snapshots_and_scale_costs,
)

# ----------------- Helpers -----------------
def t_now() -> float:
    """High-resolution time in seconds."""
    return time.perf_counter()

def dt(start: float) -> float:
    """Elapsed seconds since start."""
    return time.perf_counter() - start

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

# ----------------- Config -----------------
@dataclass
class DebugRunConfig:
    # Input
    network_nc: Path = Path("networks/base_s_5_elec_1h.nc")

    # Solve / mode
    solver_name: str = "gurobi"

    # pypsa2smspp options
    expansion_ucblock: bool = True          # True -> UCBlock path preferred
    merge_links: bool = True

    # Templates
    uc_template: str = "UCBlock/uc_solverconfig"        # or UCBlock/uc_solverconfig_grb
    inv_template: str = "InvestmentBlock/BSPar.txt"

    # Cleaning toggles
    do_clean_e_sum: bool = True
    do_clean_ciclicity_storage: bool = True
    do_add_slack_unit: bool = True
    do_reduce_snapshots: bool = False
    reduce_snapshots_to: int = 240

    # Debug toggles
    export_pypsa_lp: bool = True
    export_pypsa_nc: bool = True
    export_pypsa_smspp_nc: bool = True

    # Output
    out_root: Path = Path("output/pypsaeur")
    case_name: Optional[str] = None  # if None -> derived from network_nc.stem


def run_debug(cfg: DebugRunConfig) -> pd.DataFrame:
    """
    Run a single debug pipeline and return a DataFrame with timings/metrics.

    The function writes artifacts under cfg.out_root/case_name/.
    """
    if cfg.case_name is None:
        cfg.case_name = cfg.network_nc.stem

    case_dir = cfg.out_root / cfg.case_name
    ensure_dir(case_dir)

    # Artifact paths
    pypsa_lp = case_dir / f"pypsa_{cfg.case_name}.lp"
    pypsa_out_nc = case_dir / f"network_pypsa_{cfg.case_name}.nc"

    smspp_tmp_nc = case_dir / f"tmp_smspp_{cfg.case_name}.nc"
    smspp_solution_nc = case_dir / f"solution_{cfg.case_name}.nc"
    smspp_log_txt = case_dir / f"log_{cfg.case_name}.txt"

    pypsa_smspp_nc = case_dir / f"network_smspp_{cfg.case_name}.nc"

    # Clean stale files
    for p in (pypsa_lp, pypsa_out_nc, smspp_tmp_nc, smspp_solution_nc, smspp_log_txt, pypsa_smspp_nc):
        safe_remove(p)

    print(f"\n=== DEBUG RUN: {cfg.case_name} ===")
    print(f"Input network: {cfg.network_nc}")
    print(f"Output folder: {case_dir}")

    times: Dict[str, float] = {}
    metrics: Dict[str, Any] = {
        "case": cfg.case_name,
        "input_file": str(cfg.network_nc),
        "solver": cfg.solver_name,
        "merge_links": cfg.merge_links,
        "expansion_ucblock": cfg.expansion_ucblock,
        "status": "OK",
        "error_msg": "",
        "Obj_PyPSA": None,
        "Obj_SMSpp": None,
        "Obj_rel_error_pct": None,
        "SMSpp_solver_s_from_log": None,
    }

    try:
        # -------- Load network --------
        t0 = t_now()
        n_smspp = pypsa.Network(str(cfg.network_nc))
        times["Load network"] = dt(t0)

        # -------- Optional cleaning --------
        if cfg.do_clean_e_sum:
            t0 = t_now()
            n_smspp = clean_e_sum(n_smspp)
            times["clean_e_sum"] = dt(t0)

        if cfg.do_clean_ciclicity_storage:
            t0 = t_now()
            n_smspp = clean_ciclicity_storage(n_smspp)
            times["clean_ciclicity_storage"] = dt(t0)

        if cfg.do_reduce_snapshots:
            t0 = t_now()
            n_smspp = reduce_snapshots_and_scale_costs(n_smspp, cfg.reduce_snapshots_to)
            times["reduce_snapshots_and_scale_costs"] = dt(t0)

        if cfg.do_add_slack_unit:
            t0 = t_now()
            n_smspp = add_slack_unit(n_smspp)
            times["add_slack_unit"] = dt(t0)

        # -------- PyPSA optimization --------
        network = n_smspp.copy()
        t0 = t_now()
        network.optimize(solver_name=cfg.solver_name)
        times["PyPSA optimize"] = dt(t0)

        # Export LP (debug)
        if cfg.export_pypsa_lp:
            try:
                network.model.to_file(fn=str(pypsa_lp))
            except Exception:
                pass

        # Export optimized PyPSA network
        if cfg.export_pypsa_nc:
            try:
                network.export_to_netcdf(str(pypsa_out_nc))
            except Exception:
                pass

        # Objective (PyPSA)
        try:
            obj_pypsa = float(network.objective + network.objective_constant)
        except Exception:
            obj_pypsa = float(network.objective)
        metrics["Obj_PyPSA"] = obj_pypsa

        # -------- Transformation --------
        t0 = t_now()
        transformation = Transformation(
            network,
            merge_links=cfg.merge_links,
            expansion_ucblock=cfg.expansion_ucblock,
        )
        times["Direct transformation"] = dt(t0)

        t0 = t_now()
        tran = transformation.convert_to_blocks()
        times["PySMSpp conversion"] = dt(t0)

        # Decide mode (same logic you used)
        is_investment = (not transformation.expansion_ucblock) and \
                        (transformation.dimensions["InvestmentBlock"]["NumAssets"] > 0)
        metrics["Investment_mode"] = bool(is_investment)

        # -------- SMS++ optimization --------
        if not is_investment:
            configfile = pysmspp.SMSConfig(template=cfg.uc_template)
            inner_block_name = None
        else:
            configfile = pysmspp.SMSConfig(template=cfg.inv_template)
            inner_block_name = "InvestmentBlock"

        safe_remove(smspp_solution_nc)

        t0 = t_now()
        if inner_block_name is None:
            result = tran.optimize(
                configfile,
                str(smspp_tmp_nc),
                str(smspp_log_txt),
                str(smspp_solution_nc),
                log_executable_call=True,
            )
        else:
            result = tran.optimize(
                configfile,
                str(smspp_tmp_nc),
                str(smspp_log_txt),
                str(smspp_solution_nc),
                inner_block_name=inner_block_name,
                log_executable_call=True,
            )
        times["SMS++ (solver+writing)"] = dt(t0)

        # Parse solver time from log (if possible)
        try:
            d = parse_txt_file(str(smspp_log_txt))
            metrics["SMSpp_solver_s_from_log"] = float(d.get("elapsed_time", 0.0))
        except Exception:
            metrics["SMSpp_solver_s_from_log"] = None

        # Objective (SMS++)
        obj_smspp = float(result.objective_value)
        metrics["Obj_SMSpp"] = obj_smspp
        if obj_pypsa != 0.0:
            metrics["Obj_rel_error_pct"] = (obj_pypsa - obj_smspp) / obj_pypsa * 100.0

        # -------- Parse solution & inverse transformation --------
        t0 = t_now()
        _ = transformation.parse_solution_to_unitblocks(result.solution, n_smspp)
        times["parse_solution_to_unitblocks"] = dt(t0)

        t0 = t_now()
        transformation.inverse_transformation(result.objective_value,n_smspp)
        times["Inverse transformation"] = dt(t0)

        # ---- Quick sanity checks before exporting ----
        print("\n[DEBUG] Network sizes before export:")
        print("  buses:", len(getattr(n_smspp, "buses", [])))
        print("  generators:", len(getattr(n_smspp, "generators", [])))
        print("  loads:", len(getattr(n_smspp, "loads", [])))
        print("  lines:", len(getattr(n_smspp, "lines", [])))
        print("  links:", len(getattr(n_smspp, "links", [])))
        print("  stores:", len(getattr(n_smspp, "stores", [])))
        print("  storage_units:", len(getattr(n_smspp, "storage_units", [])))

        # ---- Export statistics to CSV (easy to inspect) ----
        try:
            stats_pypsa = network.statistics()
            # statistics() often returns a Series; make it a 1-column DataFrame
            if hasattr(stats_pypsa, "to_frame"):
                stats_pypsa = stats_pypsa.to_frame(name="value")
            stats_pypsa.to_csv(case_dir / f"stats_pypsa_{cfg.case_name}.csv")
        except Exception as e:
            print("[WARN] Could not export PyPSA statistics:", e)

        try:
            stats_smspp = n_smspp.statistics()
            if hasattr(stats_smspp, "to_frame"):
                stats_smspp = stats_smspp.to_frame(name="value")
            stats_smspp.to_csv(case_dir / f"stats_smspp_{cfg.case_name}.csv")
        except Exception as e:
            print("[WARN] Could not export SMS++-repopulated statistics:", e)
            
        if cfg.export_pypsa_smspp_nc:
            try:
                n_smspp.export_to_netcdf(str(pypsa_smspp_nc))
            except Exception:
                pass

        # Total SMS++ pipeline time (excluding PyPSA optimize)
        metrics["SMSpp_total_s"] = (
            times.get("Direct transformation", 0.0)
            + times.get("PySMSpp conversion", 0.0)
            + times.get("SMS++ (solver+writing)", 0.0)
            + times.get("parse_solution_to_unitblocks", 0.0)
            + times.get("Inverse transformation", 0.0)
        )

        # Print a compact recap
        print_kv({k: round(v, 6) for k, v in times.items()}, title="\nTimings (s):")
        print_kv(
            {
                "Investment_mode": metrics["Investment_mode"],
                "Obj_PyPSA": metrics["Obj_PyPSA"],
                "Obj_SMSpp": metrics["Obj_SMSpp"],
                "Obj_rel_error_pct": metrics["Obj_rel_error_pct"],
                "SMSpp_solver_s_from_log": metrics["SMSpp_solver_s_from_log"],
                "SMSpp_total_s": metrics["SMSpp_total_s"],
            },
            title="\nMetrics:",
        )

    except Exception as e:
        metrics["status"] = "FAIL"
        metrics["error_msg"] = f"{type(e).__name__}: {e}"
        print(f"\n!!! FAILED: {metrics['error_msg']}")
        traceback.print_exc()

    # Build a single-row DataFrame: merge metrics + times columns
    row = {**metrics, **{f"time__{k}": v for k, v in times.items()}}
    df = pd.DataFrame([row])

    # Save a per-case CSV snapshot (handy for quick comparisons)
    df.to_csv(case_dir / "debug_summary_row.csv", index=False)

    return df


def main():
    # ---- Edit these defaults for quick debugging ----
    cfg = DebugRunConfig(
        network_nc=Path("/home/pampado/sector-coupled/pypsa-eur/resources/smspp_electricity_only_italy/networks/base_s_20_elec_1h.nc"),
        solver_name="gurobi",
        merge_links=True,
        # If your file name contains "inv" you can flip this quickly:
        expansion_ucblock=True,
        uc_template="UCBlock/uc_solverconfig_grb",   # or "UCBlock/uc_solverconfig_grb"
        inv_template="InvestmentBlock/BSPar.txt",
        do_clean_e_sum=False,
        do_clean_ciclicity_storage=True,
        do_add_slack_unit=False,
        do_reduce_snapshots=False,
        reduce_snapshots_to=240,
        export_pypsa_lp=True,
        export_pypsa_nc=True,
        export_pypsa_smspp_nc=True,
        out_root=Path("output/debug"),
    )

    df = run_debug(cfg)
    print("\n>>> Wrote per-case artifacts to:", (cfg.out_root / (cfg.case_name or cfg.network_nc.stem)))


if __name__ == "__main__":
    main()

