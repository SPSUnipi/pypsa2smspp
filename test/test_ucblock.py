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
from pypsa2smspp.constants import nuclear_rules_default
from pypsa2smspp.transformation import Transformation

from pypsa2smspp.network_correction import (
    clean_ciclicity_storage,
    add_slack_unit,
)
from pypsa2smspp.utils import preprocess_zero_capital_cost_extendable_generators


# the rules of a nuclear unit switched off, the modulation ramps being the full
# thermal ones: a NuclearUnitBlock that behaves as a ThermalUnitBlock
NON_BINDING_NUCLEAR_RULES = {key: None for key in nuclear_rules_default}
NON_BINDING_NUCLEAR_RULES.update(modulation_ramp_fraction=1.0,
                                 modulation_time=2.0)


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
    network.optimize(solver_name=solver_name, include_objective_constant=True)

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
        # a committable generator is a ThermalUnitBlock
        enable_thermal_units=bool(network.generators.committable.any()),
        # the nuclear units are NuclearUnitBlocks, with rules that do not bind
        # so that the optimum is the one of PyPSA
        nuclear_units=({"nuclear": NON_BINDING_NUCLEAR_RULES}
                       if "nuclear" in set(network.generators.carrier) else None),
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



@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_free_capacity_with_minimum_output_is_refused(dynamic):
    """
    An extendable generator whose capacity costs nothing and whose output has
    a positive minimum is forced to produce more the more capacity it has:
    its best capacity is not known in advance, and fixing it at its maximum
    would force that output, hence it is refused, whether the minimum is
    static or in time.
    """
    import pypsa

    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[10.0, 20.0])
    n.add("Generator", "forced", bus="bus", p_nom_extendable=True,
          p_nom_max=50.0, capital_cost=0.0,
          p_min_pu=[0.0, 0.5] if dynamic else 0.5)

    with pytest.raises(ValueError, match="forced"):
        preprocess_zero_capital_cost_extendable_generators(n)


if __name__ == "__main__":
    run_ucblock(test_cases["xlsx_paths"][13])