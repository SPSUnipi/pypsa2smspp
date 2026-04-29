# -*- coding: utf-8 -*-
from pathlib import Path
import csv
import pytest

from conftest import (
    create_test_config,
    safe_remove,
    test_cases,
    REL_TOL,
    ABS_TOL,
    OUT_TEST,
)

from network_definition import NetworkDefinition
from pypsa2smspp.transformation import Transformation

from pypsa2smspp.network_correction import (
    clean_ciclicity_storage,
    add_slack_unit,
)


OBJECTIVES_CSV = OUT_TEST / "investment_objectives_report.csv"
_OBJECTIVES_HEADER_WRITTEN = False


def _ensure_objectives_csv() -> None:
    """
    Create the CSV report with header if it does not exist yet.
    """
    global _OBJECTIVES_HEADER_WRITTEN

    if OBJECTIVES_CSV.exists() and OBJECTIVES_CSV.stat().st_size > 0:
        _OBJECTIVES_HEADER_WRITTEN = True
        return

    OBJECTIVES_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OBJECTIVES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_name",
                "xlsx_path",
                "solver_name",
                "objective_pypsa",
                "objective_smspp",
                "absolute_difference",
                "relative_difference",
            ]
        )

    _OBJECTIVES_HEADER_WRITTEN = True


def _append_objective_row(
    case_name: str,
    xlsx_path: Path,
    solver_name: str,
    obj_pypsa: float,
    obj_smspp: float,
) -> None:
    """
    Append one comparison row to the objectives CSV report.
    """
    _ensure_objectives_csv()

    abs_diff = abs(obj_smspp - obj_pypsa)

    if abs(obj_pypsa) > 0:
        rel_diff = abs_diff / abs(obj_pypsa)
    else:
        rel_diff = 0.0 if abs_diff == 0.0 else float("inf")

    with OBJECTIVES_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                case_name,
                str(xlsx_path),
                solver_name,
                obj_pypsa,
                obj_smspp,
                abs_diff,
                rel_diff,
            ]
        )


def run_investment_block(xlsx_path: Path) -> None:
    """
    InvestmentBlock regression test:
    - build network from Excel
    - solve reference with PyPSA
    - run full SMS++ pipeline in one call (no YAML)
    - compare objectives
    - save PyPSA/SMS++ objectives to CSV
    """
    case_name = xlsx_path.stem

    # Artifacts (optional)
    network_nc = OUT_TEST / f"network_{case_name}.nc"
    pypsa_lp = OUT_TEST / f"pypsa_{case_name}.lp"

    for p in (network_nc, pypsa_lp):
        safe_remove(p)

    # ---- Build network from Excel ----
    parser = create_test_config(xlsx_path)
    nd = NetworkDefinition(parser)

    n = nd.n
    n = clean_ciclicity_storage(n)
    if "sector" not in xlsx_path.name:
        n = add_slack_unit(n)

    # Reference solve on a copy
    network = n.copy()

    # ---- (1) PyPSA optimization (reference) ----
    solver_name = getattr(parser, "solver_name", "highs")
    network.optimize(solver_name=solver_name)

    # Export LP for debugging (best effort)
    try:
        network.model.to_file(fn=str(pypsa_lp))
    except Exception:
        pass

    try:
        obj_pypsa = float(network.objective + getattr(network, "objective_constant", 0.0))
    except Exception:
        obj_pypsa = float(network.objective)

    # ---- (2) SMS++ pipeline (ONE CALL) ----
    transformation = Transformation(
        capacity_expansion_ucblock=False,  # InvestmentBlock
        workdir=OUT_TEST,
        name=case_name,
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
        fp_log="smspp_{name}_log.txt",
        fp_solution="smspp_{name}_solution.nc",
        configfile="auto",
        pysmspp_options={},  # keep pySMSpp defaults
    )

    n = transformation.run(n, verbose=False)

    obj_smspp = float(transformation.result.objective_value)

    # ---- (3) Save comparison row ----
    _append_objective_row(
        case_name=case_name,
        xlsx_path=xlsx_path,
        solver_name=solver_name,
        obj_pypsa=obj_pypsa,
        obj_smspp=obj_smspp,
    )

    # ---- (4) Compare objectives ----
    assert obj_smspp == pytest.approx(obj_pypsa, rel=REL_TOL, abs=ABS_TOL)

    # ---- (5) Optional export ----
    try:
        n.export_to_netcdf(str(network_nc))
    except Exception:
        pass


@pytest.mark.parametrize("test_case_xlsx", test_cases["xlsx_paths"], ids=test_cases["ids"])
def test_investment(test_case_xlsx):
    name_l = test_case_xlsx.name.lower()
    if "ml" in name_l or "sector" in name_l or "ext" not in name_l:
        pytest.skip("Skipping case for investment block")

    run_investment_block(test_case_xlsx)


if __name__ == "__main__":
    safe_remove(OBJECTIVES_CSV)
    run_investment_block(test_cases["xlsx_paths"][6])
    print(f"Objective report written to: {OBJECTIVES_CSV}")