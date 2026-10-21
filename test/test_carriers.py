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
