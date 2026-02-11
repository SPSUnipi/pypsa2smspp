# -*- coding: utf-8 -*-
"""
Batch SMS++ benchmark runner over multiple Excel inputs.

YAML-driven, single-call pipeline:
    nd.n = Transformation(cfg).run(nd.n)

All timings are collected internally by Transformation (StepTimer).

Outputs (per case, in output/test/):
- pypsa_<case>.nc
- pypsa_<case>.lp
- smspp_<case>_trm.nc
- smspp_<case>_solution.nc
- smspp_<case>_log.txt
- smspp_<case>_network.nc

Author: aless
"""

import os
import sys
import traceback
from pathlib import Path
from datetime import datetime
import pandas as pd
import pypsa
import shutil
import time

# ---------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
print(">>> FORCED CWD:", Path.cwd())

REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT = HERE / "output" / "test"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Domain imports
# ---------------------------------------------------------------------

from configs.test_config import TestConfig
from network_definition import NetworkDefinition
from pypsa2smspp.transformation import Transformation
from pypsa2smspp.network_correction import (
    clean_ciclicity_storage,
    add_slack_unit,
)
from pypsa2smspp.pip_utils import load_yaml_config

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def extract_step_time(timer_rows, step_name):
    for r in timer_rows:
        if r["step"] == step_name:
            return round(r["elapsed_s"], 6)
    return None


def copy_if_exists(src: Path, dst: Path):
    try:
        if src.exists():
            shutil.copy(src, dst)
    except Exception:
        pass
    

def make_case_cfg(config_yaml: Path, case_name: str, out_dir: Path):
    """
    Load YAML config and override io.name so all outputs are unique
    but written into the same directory.
    """
    cfg = load_yaml_config(config_yaml)

    if not hasattr(cfg, "io"):
        cfg.io = {}

    cfg.io.workdir = str(out_dir)   # output/test
    cfg.io.name = case_name         # UNIQUE PER NETWORK

    return cfg


# ---------------------------------------------------------------------
# Single-case runner
# ---------------------------------------------------------------------

def run_single_case(xlsx_path: Path, config_yaml: Path) -> dict:
    case_name = xlsx_path.stem
    print(f"\n=== Running case: {case_name} ===")
    
    cfg = make_case_cfg(
        config_yaml=config_yaml,
        case_name=case_name,
        out_dir=OUT,
    )

    # ---------------- Paths ----------------

    pypsa_nc = OUT / f"pypsa_{case_name}.nc"
    pypsa_lp = OUT / f"pypsa_{case_name}.lp"

    smspp_trm_nc = OUT / f"smspp_{case_name}_trm.nc"
    smspp_solution_nc = OUT / f"smspp_{case_name}_solution.nc"
    smspp_log_txt = OUT / f"smspp_{case_name}_log.txt"
    smspp_network_nc = OUT / f"smspp_{case_name}_network.nc"

    summary = {
        "case": case_name,
        "input_file": str(xlsx_path),
        "status": "OK",
        "error_msg": "",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        # timings
        "PyPSA_opt_s": None,
        "Transform_direct_s": None,
        "PySMSpp_convert_s": None,
        "SMSpp_optimize_s": None,
        "Inverse_transform_s": None,
        "SMSpp_total_s": None,
        # objectives
        "Obj_PyPSA": None,
        "Obj_SMSpp": None,
        "Obj_rel_error_pct": None,
    }

    try:
        # --------------------------------------------------------------
        # Build PyPSA network from Excel
        # --------------------------------------------------------------

        parser = TestConfig()
        parser.input_data_path = str(xlsx_path.parent)
        parser.input_name_components = xlsx_path.name

        if "sector" in xlsx_path.name:
            parser.load_sign = -1

        nd = NetworkDefinition(parser)

        n = nd.n
        n = clean_ciclicity_storage(n)
        if "sector" not in xlsx_path.name:
            n = add_slack_unit(n)

        # --------------------------------------------------------------
        # PyPSA optimization (reference)
        # --------------------------------------------------------------

        network = n.copy()

        t0 = time.perf_counter()
        network.optimize(solver_name="gurobi")
        summary["PyPSA_opt_s"] = round(time.perf_counter() - t0, 6)

        try:
            obj_pypsa = float(network.objective + getattr(network, "objective_constant", 0.0))
        except Exception:
            obj_pypsa = float(network.objective)

        summary["Obj_PyPSA"] = obj_pypsa

        # Save PyPSA artifacts
        try:
            network.export_to_netcdf(pypsa_nc)
        except Exception:
            pass

        try:
            network.model.to_file(fn=str(pypsa_lp))
        except Exception:
            pass

        # --------------------------------------------------------------
        # SMS++ pipeline (ONE CALL)
        # --------------------------------------------------------------

        transformation = Transformation(cfg)
        nd.n = transformation.run(network, verbose=True)

        # --------------------------------------------------------------
        # Save SMS++ artifacts
        # --------------------------------------------------------------

        # transformed network (TRM)
        try:
            transformation.sms_network.export_to_netcdf(smspp_trm_nc)
        except Exception:
            pass

        workdir = Path(transformation.cfg.io.workdir)

        copy_if_exists(workdir / "solution.nc", smspp_solution_nc)
        copy_if_exists(workdir / "log.txt", smspp_log_txt)

        # final PyPSA network after inverse
        try:
            nd.n.export_to_netcdf(smspp_network_nc)
        except Exception:
            pass

        # --------------------------------------------------------------
        # Objectives
        # --------------------------------------------------------------

        obj_smspp = float(transformation.result.objective_value)
        summary["Obj_SMSpp"] = obj_smspp

        if obj_pypsa != 0.0:
            summary["Obj_rel_error_pct"] = round(
                (obj_pypsa - obj_smspp) / obj_pypsa * 100.0, 8
            )

        # --------------------------------------------------------------
        # Timings (from Transformation.timer)
        # --------------------------------------------------------------

        rows = transformation.timer.rows

        summary["Transform_direct_s"] = extract_step_time(rows, "direct")
        summary["PySMSpp_convert_s"] = extract_step_time(rows, "convert_to_blocks")
        summary["SMSpp_optimize_s"] = extract_step_time(rows, "optimize")
        summary["Inverse_transform_s"] = extract_step_time(rows, "inverse_transformation")

        summary["SMSpp_total_s"] = round(
            sum(
                v for v in [
                    summary["Transform_direct_s"],
                    summary["PySMSpp_convert_s"],
                    summary["SMSpp_optimize_s"],
                    summary["Inverse_transform_s"],
                ]
                if v is not None
            ),
            6,
        )

        print(
            f"=== Done: {case_name} | "
            f"Obj_err%: {summary['Obj_rel_error_pct']} | "
            f"SMS++ total s: {summary['SMSpp_total_s']}"
        )

        return summary

    except Exception as e:
        summary["status"] = "FAIL"
        summary["error_msg"] = f"{type(e).__name__}: {e}"
        print(f"!!! FAILED {case_name}: {summary['error_msg']}")
        traceback.print_exc()
        return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    inputs_dir = HERE / "configs" / "data" / "test"
    xlsx_files = sorted(inputs_dir.glob("*.xlsx"))

    if not xlsx_files:
        print(f"Nessun .xlsx trovato in {inputs_dir}")
        return

    rows = []
    for xlsx in xlsx_files:
        if "inv" in xlsx.name:
            cfg = REPO_ROOT / "test" / "configs" / "config_test_investment.yaml"
        else:
            cfg = REPO_ROOT / "pypsa2smspp" / "data" / "config_default.yaml"

        rows.append(run_single_case(xlsx, cfg))

    df = pd.DataFrame(rows)
    csv_path = OUT / "bench_summary.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n>>> Summary written to: {csv_path}")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df)


if __name__ == "__main__":
    main()
