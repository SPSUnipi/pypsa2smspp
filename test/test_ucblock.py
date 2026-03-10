# -*- coding: utf-8 -*-
from pathlib import Path
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


def run_ucblock(xlsx_path: Path) -> None:
    """
    UCBlock regression test:
    - build network from Excel
    - solve reference with PyPSA
    - run full SMS++ pipeline in one call (no YAML)
    - compare objectives
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

    # Work on a copy for reference solve
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
        capacity_expansion_ucblock=True,  # UCBlock
        workdir=OUT_TEST,
        name=case_name,
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
        fp_log="smspp_{name}_log.txt",
        fp_solution="smspp_{name}_solution.nc",
        configfile="auto",
        pysmspp_options={},  # keep pySMSpp defaults
    )

    n = transformation.run(network, verbose=False)

    obj_smspp = float(transformation.result.objective_value)

    assert obj_smspp == pytest.approx(obj_pypsa, rel=REL_TOL, abs=ABS_TOL)

    # ---- (3) Optional export ----
    try:
        n.export_to_netcdf(str(network_nc))
    except Exception:
        pass


@pytest.mark.parametrize("test_case_xlsx", test_cases["xlsx_paths"], ids=test_cases["ids"])
def test_ucblock(test_case_xlsx):
    run_ucblock(test_case_xlsx)


if __name__ == "__main__":
    run_ucblock(test_cases["xlsx_paths"][5])