# -*- coding: utf-8 -*-
"""
The capacity expansion of a network with the UCBlock path, on small networks
built here, against PyPSA:

- an extendable asset whose capacity costs nothing is fixed at its finite
  maximum, and at a large capacity only when it has none or when a minimum
  output makes capacity cost something;
- fom_cost is counted with capital_cost, as PyPSA does;
- an extendable StorageUnit has its design Variable, which sizes power and
  energy together, and gives the objective and the capacity of PyPSA;
- a stochastic Block names in its paths only the design Variable that the
  units have, i.e., those whose investment cost is not zero.
"""
from __future__ import annotations

import math
import shutil

import numpy as np
import pandas as pd
import pypsa
import pytest

from conftest import OUT_TEST

from pypsa2smspp.transformation import Transformation
from pypsa2smspp.utils import (
    fold_fom_cost_into_capital_cost,
    preprocess_zero_capital_cost_extendable_generators,
    preprocess_zero_capital_cost_extendable_lines_links,
)


def two_buses(snapshots=4):
    """Two buses joined by a link, a load on the second and a cheap
    generator on the first, an expensive one on the second."""
    n = pypsa.Network()
    n.set_snapshots(range(snapshots))
    n.add("Bus", ["a", "b"])
    n.add("Load", "load", bus="b", p_set=[10.0, 30.0, 20.0, 40.0][:snapshots])
    n.add("Generator", "cheap", bus="a", p_nom=100.0, marginal_cost=1.0)
    n.add("Generator", "dear", bus="b", p_nom=100.0, marginal_cost=50.0)
    return n


def test_zero_cost_assets_are_fixed_at_their_maximum():
    n = two_buses()
    n.add("Link", "capped", bus0="a", bus1="b", p_nom=10.0, p_min_pu=-1,
          p_nom_extendable=True, p_nom_max=20.0, capital_cost=0.0)
    n.add("Link", "uncapped", bus0="a", bus1="b", p_nom=10.0, p_min_pu=-1,
          p_nom_extendable=True, capital_cost=0.0)
    n.add("Generator", "free", bus="a", p_nom=5.0, p_nom_extendable=True,
          p_nom_max=8.0, capital_cost=0.0)
    n.add("Generator", "forced", bus="a", p_nom=5.0, p_nom_extendable=True,
          p_nom_max=8.0, p_min_pu=0.5, capital_cost=0.0)

    preprocess_zero_capital_cost_extendable_lines_links(n, fixed_capacity=1e9)
    preprocess_zero_capital_cost_extendable_generators(n, fixed_capacity=1e9)

    assert n.links.at["capped", "p_nom"] == 20.0
    assert n.links.at["uncapped", "p_nom"] == 1e9
    assert n.generators.at["free", "p_nom"] == 8.0
    # a minimum output makes capacity cost something: not at the maximum
    assert n.generators.at["forced", "p_nom"] == 1e9
    assert not n.links.p_nom_extendable.any()


def test_fom_cost_is_folded_into_capital_cost():
    n = two_buses()
    n.add("Generator", "new", bus="b", p_nom_extendable=True,
          capital_cost=100.0, fom_cost=7.0)
    n.add("StorageUnit", "su", bus="b", p_nom_extendable=True,
          capital_cost=0.0, fom_cost=3.0)

    fold_fom_cost_into_capital_cost(n)

    assert n.generators.at["new", "capital_cost"] == 107.0
    assert n.storage_units.at["su", "capital_cost"] == 3.0
    assert not n.generators.fom_cost.any()
    assert not n.storage_units.fom_cost.any()


def storage_network():
    """The two buses with an extendable StorageUnit on the second, whose
    capacity has a cost and a fixed cost and a minimum, and which is worth
    building above it: it moves cheap energy to the dear snapshots."""
    n = two_buses()
    n.add("Link", "line", bus0="a", bus1="b", p_nom=25.0, p_min_pu=-1)
    n.add("StorageUnit", "su", bus="b", p_nom=2.0, p_nom_extendable=True,
          p_nom_min=2.0, p_nom_max=50.0, max_hours=3.0, capital_cost=4.0,
          fom_cost=1.0, efficiency_store=0.9, efficiency_dispatch=0.9,
          cyclic_state_of_charge=True)
    return n


@pytest.mark.skipif(
    shutil.which("smspp_ucblock_solver") is None
    and shutil.which("ucblock_solver") is None,
    reason="no smspp_ucblock_solver on PATH")
def test_extendable_storage_unit_matches_pypsa():
    p = storage_network()
    p.optimize(solver_name="highs")
    pypsa_value = p.objective + p.objective_constant

    s = storage_network()
    transformation = Transformation(
        name="storage_expansion",
        configfile="auto",
        capacity_expansion_ucblock=True,
        workdir=str(OUT_TEST / "storage_expansion"),
        overwrite=True,
    )
    s = transformation.run(s, verbose=False)
    smspp_value = float(transformation.result.objective_value)

    assert math.isclose(smspp_value, pypsa_value, rel_tol=1e-6)
    assert math.isclose(s.storage_units.at["su", "p_nom_opt"],
                        p.storage_units.at["su", "p_nom_opt"],
                        rel_tol=1e-6, abs_tol=1e-6)
    # it is built above its minimum, else the test proves little
    assert p.storage_units.at["su", "p_nom_opt"] > 2.0 + 1e-6


def test_tssb_paths_only_to_design_variables():
    n = storage_network()
    n.add("StorageUnit", "free", bus="a", p_nom=1.0, p_nom_extendable=True,
          p_nom_max=5.0, max_hours=2.0, capital_cost=0.0)
    n.set_scenarios({"low": 0.5, "high": 0.5})
    loads = pd.DataFrame(
        index=n.snapshots,
        columns=pd.MultiIndex.from_product([n.scenarios, ["load"]],
                                           names=["scenario", "name"]))
    for sc, f in (("low", 0.8), ("high", 1.2)):
        loads[(sc, "load")] = f * np.array([10.0, 30.0, 20.0, 40.0])
    n.c.loads.dynamic.p_set = loads

    transformation = Transformation(
        name="tssb_design_paths",
        configfile="TSSBlock/TSSBSCfg.txt",
        capacity_expansion_ucblock=True,
        workdir=str(OUT_TEST / "tssb" / "design_paths"),
        stochastic_parameters={"stochastic_type": "tssb",
                               "parameters": ["demand"]},
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
    )
    transformation.create_model(n, verbose=False)

    names = {(d["block_index"], d["var_name"])
             for d in transformation.unitblock_design_data}
    # the StorageUnit with a cost has its x_battery and no x_converter, the
    # one without has none at all
    assert sum(v == "x_battery" for _, v in names) == 1
    assert not any(v == "x_converter" for _, v in names)


def test_stochastic_storage_unit_is_a_storage_unit():
    """The inverse of a BatteryUnitBlock of a stochastic network goes to its
    StorageUnit, whose name is in the index of the storage units together
    with the scenario."""
    from pypsa2smspp.inverse import component_definition

    n = storage_network()
    n.add("Store", "store", bus="a", e_nom=10.0)
    n.set_scenarios({"low": 0.5, "high": 0.5})

    assert component_definition(
        n, {"block": "BatteryUnitBlock", "name": "su"}) == "StorageUnit"
    assert component_definition(
        n, {"block": "BatteryUnitBlock", "name": "store"}) == "Store"
