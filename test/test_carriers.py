# -*- coding: utf-8 -*-
"""
Regression tests for the carrier form of the conversion
(Transformation(multi_carrier=True)) on a network that the Excel cases do
not have: two regions, each with an electric and a hydrogen bus, joined by an
AC line and by an extendable hydrogen pipeline, a link with efficiency 1
that can go both ways. The pipeline is a line of the network of the hydrogen
with a design, a DesignNetworkBlock of that carrier in the UCBlock form and
an asset of the InvestmentBlock in the other, while the electrolyser, the
fuel cell and the gas turbine are units with an output on each carrier of
their buses; both forms are held to the optimum of PyPSA.

It also has a network with a pump, a link that draws water from one node
and electricity and delivers the water to another node of the same carrier,
which a unit with one generator cannot express, since a generator has one
node on each carrier: it is a ConversionUnitBlock, one generator per port,
held to the optimum of PyPSA with and without a minimum flow, which keeps
the unit on throughout the horizon.
"""
from __future__ import annotations

import pandas as pd
import pypsa
import pytest

from conftest import (
    ABS_TOL,
    OUT_TEST,
    REL_TOL,
    skip_carrier_form,
    solver_reads_conversion_unit,
)

from pypsa2smspp.transformation import Transformation


T = 4
WIND = [0.9, 0.2, 0.6, 0.1]
LOAD = [60.0, 80.0, 40.0, 90.0]


def build_network() -> pypsa.Network:
    """The two regions, the wind and the electrolyser in A, the load and the
    fuel cell in B, the hydrogen stored in B."""
    n = pypsa.Network()
    n.set_snapshots(range(T))
    for carrier in ("AC", "H2", "gas"):
        n.add("Carrier", carrier)
    for region in ("A", "B"):
        n.add("Bus", f"el {region}", carrier="AC")
        n.add("Bus", f"h2 {region}", carrier="H2")
    n.add("Bus", "gas", carrier="gas")
    n.add("Generator", "wind", bus="el A", p_nom=150,
          p_max_pu=pd.Series(WIND, n.snapshots), marginal_cost=0.0)
    n.add("Generator", "gas supply", bus="gas", p_nom=1000,
          marginal_cost=20.0)
    n.add("Load", "load", bus="el B", p_set=pd.Series(LOAD, n.snapshots))
    n.add("Line", "AB", bus0="el A", bus1="el B", x=0.1, s_nom=60)
    n.add("Link", "electrolyser", bus0="el A", bus1="h2 A", efficiency=0.7,
          p_nom=80, marginal_cost=1.0)
    n.add("Link", "fuel cell", bus0="h2 B", bus1="el B", efficiency=0.5,
          p_nom=50, marginal_cost=2.0)
    n.add("Link", "gas turbine", bus0="gas", bus1="el B", efficiency=0.4,
          p_nom=200, marginal_cost=3.0)
    n.add("Store", "h2 store", bus="h2 B", e_nom=200)
    # the pipeline, whose capacity is built
    n.add("Link", "pipeline", bus0="h2 A", bus1="h2 B", efficiency=1.0,
          p_min_pu=-1, p_nom_extendable=True, p_nom_max=100,
          capital_cost=5.0)
    return n


@pytest.mark.parametrize("ucblock", [True, False],
                         ids=["ucblock", "investmentblock"])
def test_carrier_design_line(ucblock):
    n = build_network()
    skip_carrier_form(n, True)

    reference = n.copy()
    reference.optimize(solver_name="highs")
    expected = float(reference.objective
                     + getattr(reference, "objective_constant", 0.0))

    name = "carrier_design_line_" + ("ucblock" if ucblock else "investment")
    transformation = Transformation(
        capacity_expansion_ucblock=ucblock,
        multi_carrier=True,
        workdir=OUT_TEST,
        name=name,
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
        fp_log="smspp_{name}_log.txt",
        fp_solution="smspp_{name}_solution.nc",
        configfile="auto",
        pysmspp_options={},
    )
    n = transformation.run(n, verbose=False)
    obtained = float(transformation.result.objective_value)

    assert obtained == pytest.approx(expected, rel=REL_TOL, abs=ABS_TOL)
    # the capacity of the pipeline read back is the one PyPSA builds
    assert n.links.at["pipeline", "p_nom_opt"] == pytest.approx(
        reference.links.at["pipeline", "p_nom_opt"], rel=1e-4, abs=1e-3)


def build_pump_network(p_min_pu: float) -> pypsa.Network:
    """The water of the source A pumped to the users at B by a pump that
    draws 0.3 MWh of electricity per unit of water, the electricity from a
    generator dearer at the second snapshot, a tank at B taking what the
    users do not; with p_min_pu the pump has a minimum flow at every
    snapshot, which binds when the users take nothing, so that a unit that
    could stop would cost less."""
    n = pypsa.Network()
    n.set_snapshots(range(T))
    for carrier in ("AC", "water"):
        n.add("Carrier", carrier)
    n.add("Bus", "el", carrier="AC")
    n.add("Bus", "water A", carrier="water")
    n.add("Bus", "water B", carrier="water")
    n.add("Generator", "plant", bus="el", p_nom=200,
          marginal_cost=pd.Series([10.0, 30.0, 10.0, 10.0], n.snapshots))
    n.add("Load", "load", bus="el", p_set=pd.Series(LOAD, n.snapshots))
    n.add("Generator", "source", bus="water A", p_nom=200,
          marginal_cost=1.0)
    n.add("Load", "users", bus="water B",
          p_set=pd.Series([30.0, 0.0, 50.0, 20.0], n.snapshots))
    n.add("Store", "tank", bus="water B", e_nom=100)
    n.add("Link", "pump", bus0="water A", bus1="water B", bus2="el",
          efficiency=1.0, efficiency2=-0.3, p_nom=60, p_min_pu=p_min_pu)
    return n


@pytest.mark.parametrize("p_min_pu", [0.0, 0.2], ids=["free", "min flow"])
def test_pump_between_two_nodes_of_a_carrier(p_min_pu):
    n = build_pump_network(p_min_pu)
    skip_carrier_form(n, True)
    if not solver_reads_conversion_unit():
        pytest.skip("the SMS++ solver does not read a ConversionUnitBlock")

    reference = n.copy()
    reference.optimize(solver_name="highs")
    expected = float(reference.objective
                     + getattr(reference, "objective_constant", 0.0))

    name = "carrier_pump_" + ("min_flow" if p_min_pu else "free")
    transformation = Transformation(
        enable_conversion_units=True,
        multi_carrier=True,
        workdir=OUT_TEST,
        name=name,
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
        fp_log="smspp_{name}_log.txt",
        fp_solution="smspp_{name}_solution.nc",
        configfile="auto",
        pysmspp_options={},
    )
    n = transformation.run(n, verbose=False)
    obtained = float(transformation.result.objective_value)

    assert obtained == pytest.approx(expected, rel=REL_TOL, abs=ABS_TOL)
    # the pump is a ConversionUnitBlock with a generator per port
    pump = [u for u in transformation.unitblocks.values()
            if u["name"] == "pump"]
    assert len(pump) == 1 and pump[0]["block"] == "ConversionUnitBlock"
    assert pump[0]["dimensions"]["NumberGenerators"] == 3
