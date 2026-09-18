# -*- coding: utf-8 -*-
"""
What the weighting of a snapshot multiplies, and what it does not.

PyPSA charges a start-up once, whatever the weighting of the snapshot, while
it charges the cost of the energy by the weighting: in `define_objective()`
the unit commitment term is `(var * cost).sum(...)` with no weight, whereas
the marginal and the stand-by costs carry one. Here both are read back from
the netCDF file on a network whose weighting is not 1, which needs no SMS++.
"""
import netCDF4 as nc
import numpy as np
import pypsa

from conftest import safe_remove, OUT_TEST

from pypsa2smspp.transformation import Transformation


SNAPSHOTS = 8
WEIGHTING = 3.0
MARGINAL_COST = 40.0
START_UP_COST = 700.0


def committable_network() -> pypsa.Network:
    """One bus, one committable generator, a load, and a weighting of 3."""
    n = pypsa.Network()
    n.set_snapshots(range(SNAPSHOTS))
    n.snapshot_weightings.loc[:, :] = WEIGHTING
    n.add("Bus", "bus")
    n.add("Carrier", ["ocgt", "slack"])
    n.add("Load", "load", bus="bus",
          p_set=600.0 + 200.0 * np.cos(2.0 * np.pi * np.arange(SNAPSHOTS) / 4.0))
    n.add("Generator", "unit", bus="bus", carrier="ocgt", p_nom=1000.0,
          marginal_cost=MARGINAL_COST, p_min_pu=0.3, committable=True,
          up_time_before=1, start_up_cost=START_UP_COST)
    n.add("Generator", "shedding", bus="bus", carrier="slack", p_nom=1e5,
          marginal_cost=3000.0)
    return n


def unit_variables() -> dict:
    """The variables of the ThermalUnitBlock of the committable generator."""
    temp_nc = OUT_TEST / "start_up_cost.nc"
    safe_remove(temp_nc)

    transformation = Transformation(capacity_expansion_ucblock=True,
                                    enable_thermal_units=True,
                                    intermittent_carriers=[])
    transformation.create_model(committable_network(), verbose=False)
    transformation.sms_network.to_netcdf(temp_nc, force=True)

    with nc.Dataset(temp_nc) as dataset:
        for group in dataset["Block_0"].groups.values():
            if getattr(group, "type", None) != "ThermalUnitBlock":
                continue
            if float(np.ravel(group["MaxPower"][...])[0]) != 1000.0:
                continue   # the slack unit, not the one under test
            return {name: np.ravel(variable[...])
                    for name, variable in group.variables.items()}

    raise AssertionError("no ThermalUnitBlock of the generator in the file")


def test_the_start_up_is_charged_once():
    assert np.all(unit_variables()["StartUpCost"] == START_UP_COST)


def test_the_energy_is_charged_by_the_weighting():
    assert np.all(unit_variables()["LinearTerm"] == MARGINAL_COST * WEIGHTING)
