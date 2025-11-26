import os, sys, traceback
from pathlib import Path
from datetime import datetime
import time
import pandas as pd
import pypsa
import pysmspp
import pytest

# pytest -p no:warnings

REL_TOL = 1e-5   # relative tolerance for objective comparison
ABS_TOL = 1e-3   # absolute tolerance for objective comparison

# --- Force working directory to this file's folder and build robust paths ---
HERE = Path(__file__).resolve().parent            # .../pypsa2smspp/test

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

def create_test_config(xlsx_path: Path) -> TestConfig:
    """Create a TestConfig object pointing to the given Excel file."""
    parser = TestConfig()
    parser.input_data_path = str(xlsx_path.parent)       # folder of the excel
    parser.input_name_components = xlsx_path.name        # excel filename
    if "sector" in xlsx_path.name:
        parser.load_sign = -1
    return parser

def create_summary_dict(xlsx_path: Path) -> dict:
    """Create an empty summary dict for the given case."""
    return {
        "case": xlsx_path.stem,
        "input_file": str(xlsx_path),
        "status": "OK",
        "error_msg": "",
        "PyPSA_opt_s": None,
        "Transform_direct_s": None,
        "PySMSpp_convert_s": None,
        "SMSpp_total_s": None,
        "SMSpp_solver_s": None,
        "SMSpp_write_s": None,
        "Inverse_transform_s": None,
        "Obj_PyPSA": None,
        "Obj_SMSpp": None,
        "Obj_rel_error_pct": None,
        "Investment_mode": None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

def run_dispatch(xlsx_path: Path,
                    solver_name: str = "gurobi",
                    uc_template: str = "UCBlock/uc_solverconfig_grb",
                    inv_template: str = "InvestmentBlock/BSPar.txt",
                    merge_links: bool = True,
                    expansion_ucblock: bool = True) -> dict:
    
    summary = create_summary_dict(xlsx_path)
    case_name = summary["case"]  # e.g., "components_caseA"

    expansion_ucblock = False if "inv" in xlsx_path.name else True

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

    parser = create_test_config(xlsx_path)

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

    # Export LP for debugging
    network.model.to_file(fn=str(pypsa_lp))

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

    # Skip for investment mode (not implemented here)
    if is_investment:
        pytest.skip("Investment mode not implemented in this test.")

    # ---- (4) SMS++ optimization ----
    # UCBlock configuration
    configfile = pysmspp.SMSConfig(template=uc_template)
    # The temporary eBlock file to write/solve
    tmp_nc = str(smspp_tmp_nc)
    out_txt = str(smspp_log_txt)
    sol_nc = str(smspp_solution_nc)

    t0 = t_now()
    result = tran.optimize(configfile, tmp_nc, out_txt, sol_nc, log_executable_call=True)
    total_smspp = delta_s(t0)
    summary["SMSpp_total_s"] = round(total_smspp, 6)

    # Parse solver time from log (if available)
    d = parse_txt_file(out_txt)
    solver_s = float(d.get("elapsed_time", 0.0))
    summary["SMSpp_solver_s"] = round(solver_s, 6)
    summary["SMSpp_write_s"] = round(total_smspp - solver_s, 6)

    # ---- Objective & error ----
    try:
        obj_pypsa = float(network.objective + network.objective_constant)
    except Exception:
        obj_pypsa = float(network.objective)

    obj_smspp = float(result.objective_value)
    summary["Obj_PyPSA"] = obj_pypsa
    summary["Obj_SMSpp"] = obj_smspp
    if obj_pypsa != 0.0:
        summary["Obj_rel_error_pct"] = round((obj_pypsa - obj_smspp) / obj_pypsa * 100.0, 8)
    
    assert obj_smspp == pytest.approx(obj_pypsa, rel=REL_TOL, abs=ABS_TOL)

    # ---- (5) Parse solution & inverse transform ----
    t0 = t_now()
    _ = transformation.parse_solution_to_unitblocks(result.solution, n)
    transformation.inverse_transformation(n)
    summary["Inverse_transform_s"] = round(delta_s(t0), 6)

    network.export_to_netcdf(str(network_nc))

    print(f"=== Done: {case_name} | Obj_err%: {summary['Obj_rel_error_pct']} | SMS++ total s: {summary['SMSpp_total_s']} ===")


# def main():
#     # Discover inputs
#     inputs_dir = HERE / "configs" / "data" / "test"
#     xlsx_files = sorted(inputs_dir.glob("*.xlsx"))

#     if not xlsx_files:
#         print(f"Nessun .xlsx trovato in {inputs_dir}")
#         return

#     rows = []
#     for x in xlsx_files:
#         expansion_ucblock = False if "inv" in x.name else True
#         row = run_single_case(x, expansion_ucblock=expansion_ucblock)
#         rows.append(row)

#     # Build DataFrame and save CSV
#     df = pd.DataFrame(rows)
#     csv_path = OUT_TEST / "bench_summary.csv"
#     df.to_csv(csv_path, index=False)
#     print(f"\n>>> Summary written to: {csv_path}")
#     # Optional: pretty print
#     try:
#         # Simple terminal preview
#         with pd.option_context('display.max_columns', None, 'display.width', 140):
#             print(df)
#     except Exception:
#         pass

def get_test_cases():
    inputs_dir = HERE / "configs" / "data" / "test"
    files = list(sorted(inputs_dir.glob("*.xlsx")))
    names = [f"{i}: {f.name}" for (i,f) in enumerate(files)]
    return files, names

test_cases, test_names = get_test_cases()

# def pytest_generate_tests(metafunc):
#     if "test_case_xlsx" in metafunc.fixturenames:
#         # use the filename as ID for better test names
#         metafunc.parametrize("test_case_xlsx", test_cases)

@pytest.mark.parametrize("test_case_xlsx", test_cases, ids=test_names)
def test_dispatch(test_case_xlsx):
    run_dispatch(test_case_xlsx)

if __name__ == "__main__":
    test_dispatch(test_case_xlsx=test_cases[0])