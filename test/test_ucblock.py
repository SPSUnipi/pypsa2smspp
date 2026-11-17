# -*- coding: utf-8 -*-
import re
import shutil
from pathlib import Path
import numpy as np
import pysmspp
import pytest

from conftest import (
    create_test_config,
    safe_remove,
    test_cases,
    REL_TOL,
    ABS_TOL,
    OUT_TEST,
    BUS_FORMS,
    BUS_FORM_IDS,
    skip_carrier_form,
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


def run_ucblock(xlsx_path: Path, multi_carrier: bool = False,
                hydro_system: bool = False, configfile="auto"):
    """
    UCBlock regression test:
    - build network from Excel
    - solve reference with PyPSA
    - run full SMS++ pipeline in one call (no YAML), with the buses of each
      carrier in a carrier of the UCBlock if multi_carrier
    - compare objectives

    Returns the Transformation and the network it wrote the solution in.
    """
    case_name = (xlsx_path.stem + ("_carriers" if multi_carrier else "")
                 + ("_hsub" if hydro_system else ""))

    # Artifacts (optional)
    network_nc = OUT_TEST / f"network_{case_name}.nc"
    pypsa_lp = OUT_TEST / f"pypsa_{case_name}.lp"

    for p in (network_nc, pypsa_lp):
        safe_remove(p)

    # ---- Build network from Excel ----
    parser = create_test_config(xlsx_path)
    nd = NetworkDefinition(parser)

    n = nd.n
    skip_carrier_form(n, multi_carrier)
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
        hydro_system=hydro_system,
        # a committable link is a ConversionUnitBlock
        enable_conversion_units=bool(network.links.committable.any()),
        multi_carrier=multi_carrier,
        workdir=OUT_TEST,
        name=case_name,
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
        fp_log="smspp_{name}_log.txt",
        fp_solution="smspp_{name}_solution.nc",
        configfile=configfile,
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

    return transformation, n


@pytest.mark.parametrize("multi_carrier", BUS_FORMS, ids=BUS_FORM_IDS)
@pytest.mark.parametrize("test_case_xlsx", test_cases["xlsx_paths"], ids=test_cases["ids"])
def test_ucblock(test_case_xlsx, multi_carrier):
    run_ucblock(test_case_xlsx, multi_carrier)


def ucblock_solver_config():
    """
    The BlockSolverConfig of the installed smspp_ucblock_solver, None if there
    is none. Its directory holds the InnerBCfg.txt that the tool reads by
    default, which asks for the "linearized" PolyhedralFunctionBlock that a
    MILPSolver needs.
    """
    solver = shutil.which("smspp_ucblock_solver")
    if solver is None:
        return None
    config = (Path(solver).resolve().parent.parent / "share" / "SMS++_tools"
              / "ucblock_solver" / "config")
    inner = config / "InnerBCfg.txt"
    if not (inner.exists() and (config / "BSCfg.txt").exists()
            and "PolyhedralFunctionBlock" in inner.read_text()):
        return None
    return pysmspp.SMSConfig(fp=config / "BSCfg.txt")


# the networks with a hydro StorageUnit
hydro_cases = [(path, name)
               for path, name in zip(test_cases["xlsx_paths"], test_cases["ids"])
               if re.search(r"_\d+h(_|$)", path.stem)]


@pytest.mark.parametrize("test_case_xlsx",
                         [path for path, _ in hydro_cases],
                         ids=[name for _, name in hydro_cases])
def test_ucblock_hydro_system(test_case_xlsx):
    """
    The hydro units in a HydroSystemUnitBlock, whose future cost of the water
    has no cut: the optimum is the one of PyPSA and of the HydroUnitBlocks in
    the UCBlock, and the solution of the hydro units is read back.
    """
    config = ucblock_solver_config()
    if config is None:
        pytest.skip("no installed smspp_ucblock_solver linearizing a "
                    "PolyhedralFunctionBlock")

    # the profiles that the xlsx does not give are drawn at random: the two
    # networks are drawn alike
    np.random.seed(0)
    plain, n_plain = run_ucblock(test_case_xlsx)
    np.random.seed(0)
    system, n = run_ucblock(test_case_xlsx, hydro_system=True,
                            configfile=config)

    assert (float(system.result.objective_value)
            == pytest.approx(float(plain.result.objective_value),
                             rel=REL_TOL, abs=ABS_TOL))

    hydro = n.storage_units.index[
        n.storage_units.carrier.isin(["hydro", "PHS"])]
    ucblock = system.sms_network.blocks["Block_0"]
    units = [ucblock.blocks[key] for key in ucblock.blocks
             if key.startswith("UnitBlock_")]
    assert ucblock.dimensions["NumberUnits"].value == len(units)
    hsub = units[-1]
    assert hsub.block_type == "HydroSystemUnitBlock"
    assert hsub.dimensions["NumberHydroUnits"].value == len(hydro)
    assert [hsub.blocks[f"HydroUnitBlock_{j}"].attributes["name"].value
            for j in range(len(hydro))] == list(hydro)
    polyhedral = hsub.blocks["PolyhedralFunctionBlock"]
    assert polyhedral.dimensions["PolyFunction_NumVar"].value == len(hydro)

    # the hydro units have their solution, the one of the HydroUnitBlocks in
    # the UCBlock
    for attr in ("p_dispatch", "p_store", "state_of_charge"):
        assert not n.storage_units_t[attr][hydro].isna().any().any()
        np.testing.assert_allclose(n.storage_units_t[attr][hydro],
                                   n_plain.storage_units_t[attr][hydro],
                                   rtol=REL_TOL, atol=ABS_TOL)


def storage_unit_network():
    """
    One bus with a load of 50 over 24 hours, a generator at 10 and a
    StorageUnit half full that can store and has a marginal cost of 5.
    """
    import pandas as pd
    import pypsa

    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(24))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=50.0)
    n.add("Generator", "gen", bus="bus", p_nom=100.0, marginal_cost=10.0)
    n.add("StorageUnit", "battery", bus="bus", p_nom=100.0, max_hours=10.0,
          state_of_charge_initial=500.0, cyclic_state_of_charge=False,
          marginal_cost=5.0)
    return n


def test_storage_unit_marginal_cost_refused():
    """
    A StorageUnit that can store and has a marginal cost is refused, since
    PyPSA charges that cost on the dispatch only.
    """
    network = storage_unit_network()
    transformation = Transformation(
        capacity_expansion_ucblock=True,
        workdir=OUT_TEST,
        name="storage_unit_marginal_cost",
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
        configfile="auto",
    )
    with pytest.raises(ValueError, match="marginal cost"):
        transformation.run(network, verbose=False)

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


def test_network_cost_not_one_value_is_refused() -> None:
    """
    The cost of the flow on a link is one value for the whole horizon in
    SMS++, the marginal cost times the weighting of the snapshots (the case
    2n_1c_1g_1lcost.xlsx checks it against PyPSA): a marginal cost that varies
    over the snapshots, or weightings that do, are refused instead of written
    as those of the first snapshot.
    """
    import pypsa
    from pypsa2smspp.utils import resolve_param_value

    def network():
        n = pypsa.Network()
        n.set_snapshots(range(3))
        n.add("Bus", ["a", "b"])
        n.add("Link", "ab", bus0="a", bus1="b", p_nom=10, marginal_cost=2.0)
        return n

    sizes = {"Links": {"Size": {"NetworkCost": "[Li]"}}}

    def cost(n):
        return resolve_param_value("marginal_cost", sizes, "Links_parameters",
                                   "NetworkCost", n.links, n.links_t, n,
                                   "Link", None)

    n = network()
    n.snapshot_weightings.loc[:, "objective"] = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        cost(n)

    n = network()
    n.links_t.marginal_cost["ab"] = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        cost(n)


if __name__ == "__main__":
    run_ucblock(test_cases["xlsx_paths"][13])