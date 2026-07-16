# -*- coding: utf-8 -*-
from pathlib import Path
import math
import pytest
import pypsa

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

NETWORK_TEST_DIR = Path(__file__).resolve().parent / "networks" / "test_ucblock"
UCBLOCK_NETWORK_CASES = sorted(NETWORK_TEST_DIR.glob("test*.nc"))
UCBLOCK_NETWORK_IDS = [p.stem for p in UCBLOCK_NETWORK_CASES]


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


def run_ucblock_network(nc_path: Path) -> None:
    """
    UCBlock regression test for a PyPSA network fixture.
    """
    case_name = f"network__{nc_path.stem}"
    workdir = OUT_TEST / "ucblock_networks" / case_name
    workdir.mkdir(parents=True, exist_ok=True)

    network_nc = workdir / f"network_{case_name}.nc"
    pypsa_lp = workdir / f"pypsa_{case_name}.lp"

    for p in (network_nc, pypsa_lp):
        safe_remove(p)

    n = pypsa.Network(str(nc_path))
    n = clean_ciclicity_storage(n)

    if "sector" not in nc_path.name.lower():
        n = add_slack_unit(n)

    network = n.copy()
    network.optimize(solver_name="highs")

    try:
        network.model.to_file(fn=str(pypsa_lp))
    except Exception:
        pass

    try:
        obj_pypsa = float(network.objective + getattr(network, "objective_constant", 0.0))
    except Exception:
        obj_pypsa = float(network.objective)

    transformation = Transformation(
        capacity_expansion_ucblock=True,
        workdir=workdir,
        name=case_name,
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
        fp_log="smspp_{name}_log.txt",
        fp_solution="smspp_{name}_solution.nc",
        configfile="auto",
        pysmspp_options={},
    )

    n = transformation.run(network, verbose=False)

    obj_smspp = float(transformation.result.objective_value)

    assert transformation.result is not None
    assert math.isfinite(obj_smspp)
    assert obj_smspp == pytest.approx(obj_pypsa, rel=REL_TOL, abs=ABS_TOL)

    try:
        n.export_to_netcdf(str(network_nc))
    except Exception:
        pass


@pytest.mark.parametrize("test_case_xlsx", test_cases["xlsx_paths"], ids=test_cases["ids"])
def test_ucblock(test_case_xlsx):
    run_ucblock(test_case_xlsx)


@pytest.mark.parametrize("nc_path", UCBLOCK_NETWORK_CASES, ids=UCBLOCK_NETWORK_IDS)
def test_ucblock_network(nc_path):
    run_ucblock_network(nc_path)


if __name__ == "__main__":
    run_ucblock(test_cases["xlsx_paths"][13])
