import os, sys, traceback
from pathlib import Path
from datetime import datetime
import time
import pandas as pd
import pypsa
import pysmspp
import pytest

REL_TOL = 1e-1   # relative tolerance for objective comparison. TODO: tighten tolerance
ABS_TOL = 1e-2   # absolute tolerance for objective comparison

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

def run_dispatch(xlsx_path: Path,
                    solver_name: str = "highs",
                    uc_template: str = "UCBlock/uc_solverconfig",
                    inv_template: str = "InvestmentBlock/BSPar.txt",
                    merge_links: bool = True) -> dict:
    
    case_name = xlsx_path.name  # e.g., "components_caseA"

    expansion_ucblock = False if "inv" in xlsx_path.name else True

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
    network.optimize(solver_name=solver_name)

    # Export LP for debugging
    network.model.to_file(fn=str(pypsa_lp))

    # ---- (2) Direct transformation ----
    transformation = Transformation(network,
                                    merge_links=merge_links,
                                    expansion_ucblock=expansion_ucblock)

    # ---- (3) Conversion to SMS++ blocks (PySMSpp object graph) ----
    tran = transformation.convert_to_blocks()

    # Determine which block to run
    is_investment = (not transformation.expansion_ucblock) and \
                    (transformation.dimensions['InvestmentBlock']['NumAssets'] > 0)

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

    result = tran.optimize(configfile, tmp_nc, out_txt, sol_nc, log_executable_call=True)

    # Parse solver time from log (if available)
    d = parse_txt_file(out_txt)

    # ---- Objective & error ----
    try:
        obj_pypsa = float(network.objective + network.objective_constant)
    except Exception:
        obj_pypsa = float(network.objective)

    obj_smspp = float(result.objective_value)
    
    assert obj_smspp == pytest.approx(obj_pypsa, rel=REL_TOL, abs=ABS_TOL)

    # ---- (5) Parse solution & inverse transform ----
    _ = transformation.parse_solution_to_unitblocks(result.solution, n)
    transformation.inverse_transformation(n)

    network.export_to_netcdf(str(network_nc))

def run_investment(xlsx_path: Path,
                    solver_name: str = "highs",
                    uc_template: str = "UCBlock/uc_solverconfig",
                    inv_template: str = "InvestmentBlock/BSPar.txt",
                    merge_links: bool = True) -> dict:
    
    case_name = xlsx_path.name  # e.g., "components_caseA"

    expansion_ucblock = False if "inv" in xlsx_path.name else True

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
    network.optimize(solver_name=solver_name)

    # Export LP for debugging
    network.model.to_file(fn=str(pypsa_lp))

    # ---- (2) Direct transformation ----
    transformation = Transformation(network,
                                    merge_links=merge_links,
                                    expansion_ucblock=expansion_ucblock)

    # ---- (3) Conversion to SMS++ blocks (PySMSpp object graph) ----
    tran = transformation.convert_to_blocks()

    # Determine which block to run
    is_investment = (not transformation.expansion_ucblock) and \
                    (transformation.dimensions['InvestmentBlock']['NumAssets'] > 0)

    # Skip for dispatch mode (not implemented here)
    if not is_investment:
        pytest.skip("Dispatch mode not implemented in this test.")

    # ---- (4) SMS++ optimization ----
    # InvestmentBlock configuration
    configfile = pysmspp.SMSConfig(template=inv_template)
    tmp_nc = str(smspp_tmp_nc)
    out_txt = str(smspp_log_txt)
    sol_nc = str(smspp_solution_nc)

    result = tran.optimize(configfile, tmp_nc, out_txt, sol_nc,
                            inner_block_name='InvestmentBlock',
                            log_executable_call=True)

    # No robust solver-time split here (template dependent), but try:
    try:
        d = parse_txt_file(out_txt)
    except Exception:
        pass

    # Objectives
    obj_pypsa = float(network.objective)  # often objective_constant already included or 0
    obj_smspp = float(result.objective_value)
    
    assert obj_smspp == pytest.approx(obj_pypsa, rel=REL_TOL, abs=ABS_TOL)

    # Inverse
    _ = transformation.parse_solution_to_unitblocks(result.solution, n)
    transformation.inverse_transformation(n)

    network.export_to_netcdf(str(network_nc))


def get_test_cases(inputs_dir = HERE / "configs" / "data" / "test"):
    """
    Get all test case Excel files and their names for parametrization.
    """
    files = list(sorted(inputs_dir.glob("*.xlsx")))
    names = [f"{i}: {f.name}" for (i,f) in enumerate(files)]
    return files, names

test_cases, test_names = get_test_cases()

@pytest.mark.parametrize("test_case_xlsx", test_cases, ids=test_names)
def test_dispatch(test_case_xlsx):
    run_dispatch(test_case_xlsx)

@pytest.mark.parametrize("test_case_xlsx", test_cases, ids=test_names)
def test_investment(test_case_xlsx):
    run_investment(test_case_xlsx)

if __name__ == "__main__":
    run_investment(test_cases[5])