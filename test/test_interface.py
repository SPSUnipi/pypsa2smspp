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

def test_split(xlsx_path: Path = test_cases["xlsx_paths"][0]) -> None:
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

    # run call
    transformation = Transformation(
        capacity_expansion_ucblock=True,  # UCBlock
        workdir=OUT_TEST,
        name=case_name,
        overwrite=True,
        fp_temp="smspp_{name}_temp_split.nc",
        fp_log="smspp_{name}_log_split.txt",
        fp_solution="smspp_{name}_solution_split.nc",
        configfile="auto",
        pysmspp_options={},  # keep pySMSpp defaults
    )

    n = transformation.run(network, verbose=False)

    obj_smspp = float(transformation.result.objective_value)

    try:
        obj_pypsa = float(network.objective + getattr(network, "objective_constant", 0.0))
    except Exception:
        obj_pypsa = float(network.objective)

    assert obj_smspp == pytest.approx(obj_pypsa, rel=REL_TOL, abs=ABS_TOL)