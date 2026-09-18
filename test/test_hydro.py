# -*- coding: utf-8 -*-
"""
The spillway of a storage unit translated into a `HydroUnitBlock`.

A `HydroUnitBlock` whose turbine is the only arc out of its reservoir states
the relation between the flow of the turbine and its power as an inequality,
which lets the water through without producing and so stands in for a spill;
PyPSA instead spills through a variable of its own, which exists only where
the inflow is positive and is bounded by it. The unit is therefore given a
third arc, the spillway, and here what the conversion writes for it is read
back from the netCDF file, which needs no SMS++.
"""
import netCDF4 as nc
import numpy as np
import pypsa

from conftest import safe_remove, OUT_TEST

from pypsa2smspp.transformation import Transformation


SNAPSHOTS = 12


def network_with_storage(carrier: str, inflow=None) -> pypsa.Network:
    """One bus, a load, a generator that covers it, and one storage unit."""
    n = pypsa.Network()
    n.set_snapshots(range(SNAPSHOTS))
    n.add("Bus", "bus")
    n.add("Carrier", ["ocgt", "slack", carrier])
    n.add("Load", "load", bus="bus",
          p_set=600.0 + 200.0 * np.cos(2.0 * np.pi * np.arange(SNAPSHOTS) / 6.0))
    n.add("Generator", "ocgt", bus="bus", carrier="ocgt", p_nom=2000.0,
          marginal_cost=80.0)
    n.add("StorageUnit", "storage", bus="bus", carrier=carrier, p_nom=100.0,
          max_hours=6.0, efficiency_store=0.87, efficiency_dispatch=0.87,
          cyclic_state_of_charge=True, p_min_pu=-1.0,
          **({} if inflow is None else {"inflow": inflow}))
    return n


def hydro_block(case_name: str, network: pypsa.Network):
    """The HydroUnitBlock the conversion writes for that network."""
    temp_nc = OUT_TEST / f"hydro_{case_name}.nc"
    safe_remove(temp_nc)

    transformation = Transformation(capacity_expansion_ucblock=True)
    transformation.create_model(network, verbose=False)
    transformation.sms_network.to_netcdf(temp_nc, force=True)

    with nc.Dataset(temp_nc) as dataset:
        blocks = [group for group in dataset["Block_0"].groups.values()
                  if getattr(group, "type", None) == "HydroUnitBlock"]
        assert len(blocks) == 1
        unit = blocks[0]
        yield unit


def test_the_spillway_is_the_third_arc():
    for unit in hydro_block("phs", network_with_storage("PHS")):
        # the turbine, the pump and the spillway, the last one with no power
        assert len(unit.dimensions["NumberArcs"]) == 3
        assert np.ravel(unit["LinearTerm"][...])[2] == 0.0
        assert not np.any(np.asarray(unit["MaxPower"][...])[:, 2])


def test_a_unit_with_no_inflow_cannot_spill():
    for unit in hydro_block("phs_flow", network_with_storage("PHS")):
        # a pumped storage unit has no inflow, hence nothing to spill
        assert not np.any(np.asarray(unit["MaxFlow"][...])[:, 2])


def test_the_spillway_carries_what_flows_in():
    inflow = 30.0 + np.arange(SNAPSHOTS, dtype=float)
    network = network_with_storage("hydro", inflow=inflow)
    for unit in hydro_block("hydro", network):
        spillway = np.asarray(unit["MaxFlow"][...])[:, 2]
        assert np.allclose(spillway, inflow)
