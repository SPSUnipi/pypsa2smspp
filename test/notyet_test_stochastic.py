# -*- coding: utf-8 -*-
from pathlib import Path
import pytest
import pypsa

from conftest import (
    create_test_config,
    safe_remove,
    test_cases,
    REL_TOL,
    ABS_TOL,
    OUT_TEST,
    get_network,
)

from network_definition import NetworkDefinition
from pypsa2smspp.transformation import Transformation

from pypsa2smspp.network_correction import (
    clean_ciclicity_storage,
    add_slack_unit,
)

def run_tssb(fp) -> None:
    """
    UCBlock regression test:
    - build network from Excel
    - solve reference with PyPSA
    - run full SMS++ pipeline in one call (config-driven)
    - compare objectives
    """

    n = pypsa.Network(fp)

    # Work on a copy for reference solve
    network = n.copy()

    # ---- (1) PyPSA optimization (reference) ----
    network.optimize(solver_name="highs")

    try:
        obj_pypsa = float(network.objective + network.objective_constant)
    except Exception:
        obj_pypsa = float(network.objective)

    # ---- (2) SMS++ pipeline (ONE CALL) ----
    transformation = Transformation()
    n = transformation.run(n, verbose=False)

    obj_smspp = float(transformation.result.objective_value)

    # If you want to ensure UCBlock is used, either:
    # (a) enforce run.mode: ucblock in the YAML used here, OR
    # (b) if you expose transformation.last_mode_used, check it:
    #
    # if hasattr(transformation, "last_mode_used") and transformation.last_mode_used != "ucblock":
    #     pytest.skip(f"Not UCBlock mode for this case (mode={transformation.last_mode_used}).")

    assert obj_smspp == pytest.approx(obj_pypsa, rel=REL_TOL, abs=ABS_TOL)


def test_stochastic_network(fp=get_network("pypsa_stoch_load.nc")):
    """
    Uses a dedicated YAML config that forces UCBlock mode (recommended).
    """
    run_tssb(fp)


if __name__ == "__main__":
    test_stochastic_network()
