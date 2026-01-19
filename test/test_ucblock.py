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

def run_ucblock(xlsx_path: Path, config_yaml: Path) -> None:
    """
    UCBlock regression test:
    - build network from Excel
    - solve reference with PyPSA
    - run full SMS++ pipeline in one call (config-driven)
    - compare objectives
    """

    case_name = xlsx_path.stem  # cleaner than .name
    prefix = case_name

    # Artifacts (optional)
    network_nc = OUT_TEST / f"network_{prefix}.nc"
    pypsa_lp = OUT_TEST / f"pypsa_{prefix}.lp"

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
    network.model.to_file(fn=str(pypsa_lp))

    try:
        obj_pypsa = float(network.objective + getattr(network, "objective_constant", 0.0))
    except Exception:
        obj_pypsa = float(network.objective)

    # ---- (2) SMS++ pipeline (ONE CALL) ----
    transformation = Transformation(str(config_yaml))
    n = transformation.run(network, verbose=False)

    obj_smspp = float(transformation.result.objective_value)

    # If you want to ensure UCBlock is used, either:
    # (a) enforce run.mode: ucblock in the YAML used here, OR
    # (b) if you expose transformation.last_mode_used, check it:
    #
    # if hasattr(transformation, "last_mode_used") and transformation.last_mode_used != "ucblock":
    #     pytest.skip(f"Not UCBlock mode for this case (mode={transformation.last_mode_used}).")

    assert obj_smspp == pytest.approx(obj_pypsa, rel=REL_TOL, abs=ABS_TOL)

    # ---- (3) Optional export ----
    try:
        network.export_to_netcdf(str(network_nc))
    except Exception:
        pass


@pytest.mark.parametrize("test_case_xlsx", test_cases["xlsx_paths"], ids=test_cases["ids"])
def test_ucblock(test_case_xlsx):
    """
    Uses a dedicated YAML config that forces UCBlock mode (recommended).
    """
    # Put a test YAML here (recommended):
    # pypsa2smspp/data/config_test_ucblock.yaml
    config_yaml = Path(__file__).resolve().parents[1] / "test" / "configs" / "config_test_ucblock.yaml"
    if not config_yaml.exists():
        pytest.skip(f"Missing UCBlock test config: {config_yaml}")

    run_ucblock(test_case_xlsx, config_yaml)


if __name__ == "__main__":
    config_yaml = Path(__file__).resolve().parents[1] / "test" / "configs" / "config_test_ucblock.yaml"
    run_ucblock(test_cases["xlsx_paths"][5], config_yaml)
