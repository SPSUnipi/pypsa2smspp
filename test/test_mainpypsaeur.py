# -*- coding: utf-8 -*-
"""
Batch SMS++ benchmark runner over multiple PyPSA networks (.nc).

New style:
- PyPSA optimize (reference)
- SMS++ pipeline with ONE call: Transformation(cfg).run(network)
- Timings are taken from Transformation.timer

Outputs:
- Per-case artifacts in output/test_pypsaeur/
- CSV summary at output/test_pypsaeur/bench_summary_pypsaeur.csv
"""

import os
import sys
import traceback
from pathlib import Path
from datetime import datetime
import re

import pandas as pd
import pypsa

# --- Force working directory to this file's folder and build robust paths ---
HERE = Path(__file__).resolve().parent
os.chdir(HERE)
print(">>> FORCED CWD:", Path.cwd())

# Ensure PYTHONPATH for imports from repo root (e.g., scripts/)
REPO_ROOT = HERE.parent
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
from pypsa2smspp.pip_utils import load_yaml_config  # to override io.name per case

from pypsa2smspp.network_correction import (
    clean_ciclicity_storage,
    # clean_e_sum,
    # add_slack_unit,
    # clean_stores,
)

# ---------- Utilities ----------

def safe_remove(p: Path) -> None:
    """Remove file if it exists."""
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def pypsa_reference_objective(network: "pypsa.Network") -> float:
    """Robust reference objective extraction."""
    try:
        return float(network.objective + getattr(network, "objective_constant", 0.0))
    except Exception:
        return float(network.objective)


def make_case_cfg(config_yaml: Path, case_name: str):
    """
    Load YAML config and override io.name so each case writes unique artifacts.
    Prevents collisions in batch runs.
    """
    cfg = load_yaml_config(config_yaml)
    if not hasattr(cfg, "io"):
        cfg.io = {}
    cfg.io.name = f"bench_{case_name}"
    return cfg


# ---------- Core runner for a single .nc network ----------

def run_single_nc(
    nc_path: Path,
    *,
    config_yaml: Path,
    solver_name: str = "gurobi",
    do_clean_ciclicity_storage: bool = True,
    export_pypsa_lp: bool = True,
    export_pypsa_nc: bool = True,
    export_smspp_repopulated_nc: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Run the full flow for a given PyPSA network (.nc file).

    Returns a dict with timings, status, and metrics to be appended to summary CSV.
    """
    case_name = nc_path.stem
    print(f"\n=== Running NC case: {case_name} ({nc_path.name}) ===")

    # Per-case artifacts (PyPSA-side)
    pypsa_lp = OUT_TEST / f"pypsa_{case_name}.lp"
    pypsa_out_nc = OUT_TEST / f"network_pypsa_{case_name}.nc"
    smspp_repop_nc = OUT_TEST / f"network_smspp_{case_name}.nc"
    timings_csv = OUT_TEST / f"timings_{case_name}.csv"

    for p in (pypsa_lp, pypsa_out_nc, smspp_repop_nc, timings_csv):
        safe_remove(p)

    summary = {
        "case": case_name,
        "input_file": str(nc_path),
        "config_yaml": str(config_yaml),
        "status": "OK",
        "error_msg": "",
        "PyPSA_opt_s": None,          # NOTE: we do not time PyPSA via StepTimer; do it manually if needed
        "SMSpp_total_s": None,        # from Transformation.timer
        "Obj_PyPSA": None,
        "Obj_SMSpp": None,
        "Obj_rel_error_pct": None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        # -------- Load network --------
        n_raw = pypsa.Network(str(nc_path))

        # -------- Optional cleanups --------
        n_clean = n_raw
        if do_clean_ciclicity_storage:
            n_clean = clean_ciclicity_storage(n_clean)

        # -------- PyPSA optimization (reference) --------
        network = n_clean.copy()

        # Time PyPSA optimize (optional manual timing)
        t0 = pd.Timestamp.now()
        network.optimize(solver_name=solver_name)
        summary["PyPSA_opt_s"] = float((pd.Timestamp.now() - t0).total_seconds())

        # Export LP for debugging (best effort)
        if export_pypsa_lp:
            try:
                network.model.to_file(fn=str(pypsa_lp))
            except Exception:
                pass

        # Export optimized PyPSA network
        if export_pypsa_nc:
            try:
                network.export_to_netcdf(str(pypsa_out_nc))
            except Exception:
                pass

        obj_pypsa = pypsa_reference_objective(network)
        summary["Obj_PyPSA"] = obj_pypsa

        # -------- SMS++ pipeline (ONE CALL) --------
        cfg = make_case_cfg(config_yaml, case_name)
        transformation = Transformation(cfg)  # requires Transformation to accept dict/AttrDict
        n_smspp = transformation.run(network, verbose=verbose)

        obj_smspp = float(transformation.result.objective_value)
        summary["Obj_SMSpp"] = obj_smspp
        if obj_pypsa != 0.0:
            summary["Obj_rel_error_pct"] = (obj_pypsa - obj_smspp) / obj_pypsa * 100.0

        # -------- Timings from Transformation.timer --------
        timer_rows = getattr(getattr(transformation, "timer", None), "rows", None) or []
        if timer_rows:
            # Save long-format timings table
            try:
                pd.DataFrame(timer_rows).to_csv(timings_csv, index=False)
            except Exception:
                pass

            # Add wide-format columns to summary
            for r in timer_rows:
                step_name = r.get("step", "unknown")
                summary[f"time__{step_name}"] = r.get("elapsed_s", None)

            summary["SMSpp_total_s"] = sum(
                float(r.get("elapsed_s", 0.0)) for r in timer_rows if r.get("elapsed_s") is not None
            )

        # -------- Export repopulated network (after inverse) --------
        if export_smspp_repopulated_nc:
            try:
                n_smspp.export_to_netcdf(str(smspp_repop_nc))
            except Exception:
                pass

        print(
            f"=== Done NC: {case_name} | Obj_err%: {summary['Obj_rel_error_pct']} "
            f"| SMS++ total s: {summary['SMSpp_total_s']} ==="
        )
        return summary

    except Exception as e:
        summary["status"] = "FAIL"
        summary["error_msg"] = f"{type(e).__name__}: {e}"
        print(f"!!! FAILED NC {case_name}: {summary['error_msg']}")
        traceback.print_exc()
        return summary


def main():
    # ---- Choose your config YAML ----
    # Use a benchmark-specific YAML if you want, otherwise default is fine.
    config_yaml = Path(__file__).resolve().parents[1] / "pypsa2smspp" / "data" / "config_default.yaml"
    if not config_yaml.exists():
        raise FileNotFoundError(config_yaml)

    # ---- Discover inputs (.nc networks) ----
    inputs_dir = Path("/home/pampado/sector-coupled/pypsa-eur/resources/smspp_electricity_only_italy/networks")

    # Sort by cluster number after s_ (optional)
    pattern = re.compile(r"s_(\d+)")

    def extract_cluster_number(path: Path):
        m = pattern.search(path.name)
        return int(m.group(1)) if m else float("inf")

    nc_files = sorted(inputs_dir.glob("*h*.nc"), key=extract_cluster_number)

    if not nc_files:
        print(f"No .nc file in {inputs_dir}")
        return

    rows = []
    for nc in nc_files:
        row = run_single_nc(
            nc,
            config_yaml=config_yaml,
            solver_name="gurobi",
            do_clean_ciclicity_storage=True,
            export_pypsa_lp=True,
            export_pypsa_nc=True,
            export_smspp_repopulated_nc=True,
            verbose=False,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = OUT_TEST / "bench_summary_pypsaeur.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n>>> Summary (NC) written to: {csv_path}")
    try:
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(df)
    except Exception:
        pass


if __name__ == "__main__":
    main()
