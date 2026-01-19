# -*- coding: utf-8 -*-
"""
Single-case debug runner for PyPSA -> SMS++ (via pypsa2smspp).

New style:
- Load one PyPSA network (.nc)
- Optional cleaning steps
- Run PyPSA optimization (reference)
- Run SMS++ pipeline with ONE call: Transformation(config_yaml).run(network)

Artifacts per-case in output/debug/<case_name>/.
"""

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

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

# --- Project imports (after sys.path adjustment) ---
from pypsa2smspp.transformation import Transformation
from pypsa2smspp.network_correction import (
    clean_e_sum,
    clean_ciclicity_storage,
    add_slack_unit,
    reduce_snapshots_and_scale_costs,
)

# ----------------- Helpers -----------------

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

# ----------------- Config -----------------

@dataclass
class DebugRunConfig:
    # Inputs
    network_nc: Path = Path("networks/base_s_5_elec_1h.nc")
    config_yaml: Path = Path("pypsa2smspp/data/config_default.yaml")

    # PyPSA reference solve
    solver_name: str = "gurobi"

    # Cleaning toggles
    do_clean_e_sum: bool = False
    do_clean_ciclicity_storage: bool = True
    do_add_slack_unit: bool = False
    do_reduce_snapshots: bool = False
    reduce_snapshots_to: int = 240

    # Debug artifacts
    export_pypsa_lp: bool = True
    export_pypsa_nc: bool = True
    export_smspp_repopulated_nc: bool = True
    export_statistics_csv: bool = True

    # Output
    out_root: Path = Path("output/debug")
    case_name: Optional[str] = None  # if None -> derived from network_nc.stem

    # Verbosity
    verbose: bool = True


def run_debug(cfg: DebugRunConfig) -> Tuple[pd.DataFrame, pypsa.Network, pypsa.Network]:
    """
    Run a single debug pipeline and return a 1-row DataFrame with timings/metrics.
    Writes artifacts under cfg.out_root/case_name/.
    """
    if cfg.case_name is None:
        cfg.case_name = cfg.network_nc.stem

    case_dir = cfg.out_root / cfg.case_name
    ensure_dir(case_dir)

    # Artifact paths
    pypsa_lp = case_dir / f"pypsa_{cfg.case_name}.lp"
    pypsa_out_nc = case_dir / f"network_pypsa_{cfg.case_name}.nc"
    smspp_repop_nc = case_dir / f"network_smspp_{cfg.case_name}.nc"
    debug_summary_csv = case_dir / "debug_summary_row.csv"
    stats_pypsa_csv = case_dir / f"stats_pypsa_{cfg.case_name}.csv"
    stats_smspp_csv = case_dir / f"stats_smspp_{cfg.case_name}.csv"
    timings_csv = case_dir / f"timings_{cfg.case_name}.csv"

    # Clean stale files
    for p in (
        pypsa_lp, pypsa_out_nc, smspp_repop_nc,
        debug_summary_csv, stats_pypsa_csv, stats_smspp_csv, timings_csv
    ):
        safe_remove(p)

    print(f"\n=== DEBUG RUN: {cfg.case_name} ===")
    print(f"Input network: {cfg.network_nc}")
    print(f"Config YAML:   {cfg.config_yaml}")
    print(f"Output folder: {case_dir}")

    metrics: Dict[str, Any] = {
        "case": cfg.case_name,
        "input_file": str(cfg.network_nc),
        "config_yaml": str(cfg.config_yaml),
        "pypsa_solver": cfg.solver_name,
        "status": "OK",
        "error_msg": "",
        "Obj_PyPSA": None,
        "Obj_SMSpp": None,
        "Obj_rel_error_pct": None,
    }

    try:
        # -------- Load network --------
        n_raw = pypsa.Network(str(cfg.network_nc))

        # -------- Optional cleaning --------
        n_clean = n_raw

        if cfg.do_clean_e_sum:
            n_clean = clean_e_sum(n_clean)

        if cfg.do_clean_ciclicity_storage:
            n_clean = clean_ciclicity_storage(n_clean)

        if cfg.do_reduce_snapshots:
            n_clean = reduce_snapshots_and_scale_costs(n_clean, cfg.reduce_snapshots_to)

        if cfg.do_add_slack_unit:
            n_clean = add_slack_unit(n_clean)

        # -------- PyPSA optimization (reference) --------
        network = n_clean.copy()
        network.optimize(solver_name=cfg.solver_name)

        # Export LP (debug)
        if cfg.export_pypsa_lp:
            network.model.to_file(fn=str(pypsa_lp))

        # Export optimized PyPSA network
        if cfg.export_pypsa_nc:
            try:
                network.export_to_netcdf(str(pypsa_out_nc))
            except Exception:
                pass

        obj_pypsa = pypsa_reference_objective(network)
        metrics["Obj_PyPSA"] = obj_pypsa

        # -------- SMS++ pipeline (ONE CALL) --------
        transformation = Transformation(str(cfg.config_yaml))
        n_smspp = transformation.run(network, verbose=cfg.verbose)

        # Objective (SMS++)
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

        # -------- Export statistics to CSV --------
        if cfg.export_statistics_csv:
            try:
                stats_pypsa = network.statistics()
                if hasattr(stats_pypsa, "to_frame"):
                    stats_pypsa = stats_pypsa.to_frame(name="value")
                stats_pypsa.to_csv(stats_pypsa_csv)
            except Exception as e:
                print("[WARN] Could not export PyPSA statistics:", e)

            try:
                stats_smspp = n_smspp.statistics()
                if hasattr(stats_smspp, "to_frame"):
                    stats_smspp = stats_smspp.to_frame(name="value")
                stats_smspp.to_csv(stats_smspp_csv)
            except Exception as e:
                print("[WARN] Could not export SMS++ statistics:", e)

        # -------- Export repopulated network --------
        if cfg.export_smspp_repopulated_nc:
            try:
                n_smspp.export_to_netcdf(str(smspp_repop_nc))
            except Exception:
                pass

        # -------- Timings (from Transformation.timer) --------
        # Expected: transformation.timer.rows is a list of dicts with "step" and "elapsed_s".
        timer_rows = getattr(getattr(transformation, "timer", None), "rows", None)
        if timer_rows is None:
            timer_rows = []

        # Save detailed timings table
        try:
            if timer_rows:
                pd.DataFrame(timer_rows).to_csv(timings_csv, index=False)
        except Exception:
            pass

        # Add step timings to metrics (wide format)
        for r in timer_rows:
            step_name = r.get("step", "unknown")
            elapsed_s = r.get("elapsed_s", None)
            metrics[f"time__{step_name}"] = elapsed_s

        # Total SMS++ pipeline time from timer rows
        metrics["SMSpp_total_s"] = sum(
            float(r.get("elapsed_s", 0.0)) for r in timer_rows if r.get("elapsed_s") is not None
        )

        # Print recap
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

    # Build a single-row DataFrame
    df = pd.DataFrame([metrics])

    # Save a per-case CSV snapshot
    try:
        df.to_csv(debug_summary_csv, index=False)
    except Exception:
        pass

    return df, n_smspp, network


def main():
    # ---- Edit these defaults for quick debugging ----
    cfg = DebugRunConfig(
        network_nc=Path(
            "/home/pampado/sector-coupled/pypsa-eur/resources/smspp_electricity_only_italy/networks/base_s_20_elec_1h.nc"
        ),
        config_yaml=Path("pypsa2smspp/data/config_default.yaml"),
        solver_name="gurobi",
        do_clean_e_sum=False,
        do_clean_ciclicity_storage=True,
        do_add_slack_unit=False,
        do_reduce_snapshots=False,
        reduce_snapshots_to=240,
        export_pypsa_lp=True,
        export_pypsa_nc=True,
        export_smspp_repopulated_nc=True,
        export_statistics_csv=True,
        out_root=Path("output/debug"),
        verbose=True,
    )

    df, n_smspp, network = run_debug(cfg)
    print("\n>>> Wrote per-case artifacts to:", (cfg.out_root / (cfg.case_name or cfg.network_nc.stem)))
    return df, n_smspp, network
    

if __name__ == "__main__":
    df, n_smspp,network = main()

