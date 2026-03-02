# -*- coding: utf-8 -*-
"""
Batch SMS++ benchmark runner over multiple Excel inputs.

New style (no YAML):
    nd.n = Transformation(...).run(network)

Per case outputs (in output/test/):
- pypsa_<case>.nc
- pypsa_<case>.lp
- smspp_<case>_trm.nc              (SMSNetwork exported after conversion, if available)
- smspp_<case>_temp.nc             (fp_temp used by pySMSpp during optimization)
- smspp_<case>_solution.nc         (fp_solution produced by the solver)
- smspp_<case>_log.txt             (fp_log produced by the solver)
- smspp_<case>_network.nc          (final PyPSA network after inverse transformation)

Author: aless (adapted)
"""

import os
import sys
import traceback
from pathlib import Path
from datetime import datetime
import shutil
import time

import pandas as pd
import pypsa

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

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def extract_step_time(timer_rows, step_name):
    for r in timer_rows:
        if r.get("step") == step_name:
            try:
                return round(float(r.get("elapsed_s", 0.0)), 6)
            except Exception:
                return None
    return None


def safe_unlink(path: Path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------
# Single-case runner
# ---------------------------------------------------------------------

def run_single_case(xlsx_path: Path) -> dict:
    case_name = xlsx_path.stem
    print(f"\n=== Running case: {case_name} ===")

    is_inv = "inv" in xlsx_path.name.lower()
    capacity_expansion_ucblock = (not is_inv)

    # ---------------- Paths ----------------

    pypsa_nc = OUT / f"pypsa_{case_name}.nc"
    pypsa_lp = OUT / f"pypsa_{case_name}.lp"

    smspp_trm_nc = OUT / f"smspp_{case_name}_trm.nc"
    smspp_temp_nc = OUT / f"smspp_{case_name}_temp.nc"
    smspp_solution_nc = OUT / f"smspp_{case_name}_solution.nc"
    smspp_log_txt = OUT / f"smspp_{case_name}_log.txt"
    smspp_network_nc = OUT / f"smspp_{case_name}_network.nc"

    summary = {
        "case": case_name,
        "input_file": str(xlsx_path),
        "capacity_expansion_ucblock": capacity_expansion_ucblock,
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
        # artifacts
        "fp_temp": str(smspp_temp_nc),
        "fp_log": str(smspp_log_txt),
        "fp_solution": str(smspp_solution_nc),
    }

    try:
        # --------------------------------------------------------------
        # Build PyPSA network from Excel
        # --------------------------------------------------------------

        parser = TestConfig()
        parser.input_data_path = str(xlsx_path.parent)
        parser.input_name_components = xlsx_path.name

        if "sector" in xlsx_path.name.lower():
            parser.load_sign = -1

        nd = NetworkDefinition(parser)

        n = nd.n
        n = clean_ciclicity_storage(n)
        if "sector" not in xlsx_path.name.lower():
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

        # Optional: clean artifacts if overwrite-like behavior is needed also outside
        # (Transformation already does overwrite for fp_* according to your new optimize())
        safe_unlink(smspp_trm_nc)

        transformation = Transformation(
            # mode selection
            capacity_expansion_ucblock=capacity_expansion_ucblock,

            # IO / naming
            workdir=OUT,
            name=case_name,
            overwrite=True,

            # Explicit artifacts (using {name} -> case_name)
            fp_temp="smspp_{name}_temp.nc",
            fp_log="smspp_{name}_log.txt",
            fp_solution="smspp_{name}_solution.nc",

            # SMS++ config selection inside optimize via "auto"
            configfile="auto",

            # Solver options (optional; keep empty to use pySMSpp defaults)
            pysmspp_options={},
        )

        nd.n = transformation.run(network, verbose=True)

        # --------------------------------------------------------------
        # Save SMS++ artifacts
        # --------------------------------------------------------------

        # Export transformed SMS network if available (not strictly required, but useful)
        try:
            if getattr(transformation, "sms_network", None) is not None:
                transformation.sms_network.export_to_netcdf(smspp_trm_nc)
        except Exception:
            pass

        # Final PyPSA network after inverse
        try:
            nd.n.export_to_netcdf(smspp_network_nc)
        except Exception:
            pass

        # --------------------------------------------------------------
        # Objectives
        # --------------------------------------------------------------

        try:
            obj_smspp = float(transformation.result.objective_value)
        except Exception:
            # fallback if result structure changes
            obj_smspp = float(getattr(transformation.result, "objective", float("nan")))

        summary["Obj_SMSpp"] = obj_smspp

        if obj_pypsa != 0.0 and pd.notna(obj_smspp):
            summary["Obj_rel_error_pct"] = round((obj_pypsa - obj_smspp) / obj_pypsa * 100.0, 8)

        # --------------------------------------------------------------
        # Timings (from Transformation.timer)
        # --------------------------------------------------------------

        rows = getattr(transformation, "timer", None)
        rows = rows.rows if rows is not None else []

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
        rows.append(run_single_case(xlsx))

    df = pd.DataFrame(rows)
    csv_path = OUT / "bench_summary.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n>>> Summary written to: {csv_path}")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df)


if __name__ == "__main__":
    main()