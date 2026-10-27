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


def storage_network(component):
    """
    One bus with a load of 50 over 24 hours, a generator at 10 and a storage
    half full at 5, cheaper than the generator: PyPSA empties the storage,
    paying 5 * 500 + 10 * 700 = 9500.
    """
    import pandas as pd
    import pypsa

    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(24))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=50.0)
    n.add("Generator", "gen", bus="bus", p_nom=100.0, marginal_cost=10.0)
    if component == "Store":
        n.add("Store", "store", bus="bus", e_nom=1000.0, e_initial=500.0,
              e_cyclic=False, marginal_cost=5.0)
    else:
        n.add("StorageUnit", "battery", bus="bus", p_nom=100.0, max_hours=10.0,
              state_of_charge_initial=500.0, cyclic_state_of_charge=False,
              marginal_cost=5.0)
    return n


def storage_transformation(name):
    return Transformation(
        capacity_expansion_ucblock=True,
        workdir=OUT_TEST,
        name=name,
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
        fp_log="smspp_{name}_log.txt",
        fp_solution="smspp_{name}_solution.nc",
        configfile="auto",
        pysmspp_options={},
    )


def test_store_marginal_cost():
    """
    The marginal cost of a Store is paid on the energy it delivers, as in
    PyPSA, and not earned.
    """
    network = storage_network("Store")
    network.optimize(solver_name="highs")
    assert float(network.objective) == pytest.approx(9500.0, rel=1e-6)

    transformation = storage_transformation("store_marginal_cost")
    transformation.run(network, verbose=False)
    assert float(transformation.result.objective_value) == \
        pytest.approx(9500.0, rel=1e-6)


def test_storage_unit_marginal_cost_refused():
    """
    A StorageUnit that can store and has a marginal cost is refused, since
    PyPSA charges that cost on the dispatch only.
    """
    network = storage_network("StorageUnit")
    transformation = storage_transformation("storage_unit_marginal_cost")
    with pytest.raises(ValueError, match="marginal cost"):
        transformation.run(network, verbose=False)


if __name__ == "__main__":
    run_ucblock(test_cases["xlsx_paths"][13])