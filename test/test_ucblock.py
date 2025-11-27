from pathlib import Path
import pysmspp
import pytest

# --- Domain imports (after PYTHONPATH is set) ---
from conftest import (
    create_test_config,
    safe_remove,
    test_cases,
    REL_TOL,
    ABS_TOL,
    OUT_TEST,
)
from configs.test_config import TestConfig
from network_definition import NetworkDefinition
from pypsa2smspp.transformation import Transformation

from pypsa2smspp.network_correction import (
    clean_ciclicity_storage,
    parse_txt_file,
    add_slack_unit,
    compare_networks,  # optional: not used in timings but kept for debugging
)

def run_ucblock(xlsx_path: Path,
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
@pytest.mark.parametrize(
        "test_case_xlsx", test_cases["xlsx_paths"], ids=test_cases["ids"]
)
def test_ucblock(test_case_xlsx):
    run_ucblock(test_case_xlsx)

if __name__ == "__main__":
    run_ucblock(test_cases["xlsx_paths"][5])