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


def run_investment_block(xlsx_path: Path, config_yaml: Path) -> None:
    """
    InvestmentBlock regression test:
    - build network from Excel
    - solve reference with PyPSA
    - run full SMS++ pipeline in one call (config-driven)
    - compare objectives
    """

    case_name = xlsx_path.stem
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

    # Reference solve on a copy
    network = n.copy()

    # ---- (1) PyPSA optimization (reference) ----
    solver_name = getattr(parser, "solver_name", "highs")
    network.optimize(solver_name=solver_name)

    # Export LP for debugging (best effort)
    network.model.to_file(fn=str(pypsa_lp))

    try:
        obj_pypsa = float(network.objective + network.objective_constant)
    except Exception:
        obj_pypsa = float(network.objective)

    # ---- (2) SMS++ pipeline (ONE CALL) ----
    transformation = Transformation(str(config_yaml))
    n = transformation.run(network, verbose=False)

    obj_smspp = float(transformation.result.objective_value)

    # Optional: if you expose the used mode, you can enforce it here
    # if hasattr(transformation, "last_mode_used") and transformation.last_mode_used != "investmentblock":
    #     pytest.skip(f"Not InvestmentBlock mode for this case (mode={transformation.last_mode_used}).")

    assert obj_smspp == pytest.approx(obj_pypsa, rel=REL_TOL, abs=ABS_TOL)

    # ---- (3) Optional export ----
    try:
        network.export_to_netcdf(str(network_nc))
    except Exception:
        pass


@pytest.mark.parametrize("test_case_xlsx", test_cases["xlsx_paths"], ids=test_cases["ids"])
def test_investment(test_case_xlsx):
    """
    Uses a dedicated YAML config that forces InvestmentBlock mode (recommended).
    """
    config_yaml = Path(__file__).resolve().parents[1] / "test" / "configs" / "config_test_investment.yaml"
    if not config_yaml.exists():
        pytest.skip(f"Missing InvestmentBlock test config: {config_yaml}")
    name_l = test_case_xlsx.name.lower()
    if "ml" in name_l or "sector" in name_l:
        pytest.skip("Skipping case for investment block")

    run_investment_block(test_case_xlsx, config_yaml)


if __name__ == "__main__":
    config_yaml = Path(__file__).resolve().parents[1] / "test" / "configs" / "config_test_investment.yaml"
    run_investment_block(test_cases["xlsx_paths"][7], config_yaml)
