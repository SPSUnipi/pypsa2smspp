# -*- coding: utf-8 -*-
"""
The global constraints of a PyPSA network on the dispatch, translated into the
pollutant budget constraints of the UCBlock: primary energy limits with each
of the three senses, an operational limit, and a primary energy limit charging
a non-cyclic storage unit for the change of its state of charge. The network
is small and deterministic, and the objective of SMS++ is held to that of
PyPSA.
"""
import numpy as np
import pandas as pd
import pypsa
import pytest

from conftest import REL_TOL, ABS_TOL, OUT_TEST
from test_pollutant_budget import solver_reads_pollutant_budget
from pypsa2smspp.transformation import Transformation


def build_network():
    """One bus with coal, gas, wind, a gas storage unit and a slack."""
    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(6))
    n.add("Carrier", "AC")
    n.add("Carrier", "coal", co2_emissions=1.0)
    n.add("Carrier", "gas", co2_emissions=0.2)
    n.add("Carrier", "wind")
    n.add("Carrier", "slack")
    n.add("Bus", "bus", carrier="AC")
    n.add("Load", "load", bus="bus", p_set=[60, 80, 100, 90, 70, 50])
    n.add("Generator", "coal", bus="bus", carrier="coal", p_nom=100,
          efficiency=0.4, marginal_cost=20)
    n.add("Generator", "gas", bus="bus", carrier="gas", p_nom=100,
          efficiency=0.5, marginal_cost=40)
    n.add("Generator", "wind", bus="bus", carrier="wind", p_nom=40,
          p_max_pu=[0.9, 0.2, 0.5, 0.1, 0.7, 1.0], marginal_cost=0)
    n.add("Generator", "slack", bus="bus", carrier="slack", p_nom=1000,
          marginal_cost=1000)
    return n


def add_storage(n):
    """A non-cyclic storage unit of gas, which starts half full."""
    n.add("StorageUnit", "gas storage", bus="bus", carrier="gas", p_nom=20,
          max_hours=4, state_of_charge_initial=40,
          cyclic_state_of_charge=False, marginal_cost=0)


def emissions(n, attribute):
    """The emissions of the dispatch of an optimized network, generators only."""
    efficiency = n.get_switchable_as_dense("Generator", "efficiency")
    total = 0.0
    for gen, carrier in n.generators.carrier.items():
        rate = n.carriers.at[carrier, attribute]
        if rate:
            total += rate * (n.generators_t.p[gen] / efficiency[gen]).sum()
    return total


# name: (storage, GlobalConstraint type, carrier attribute, sense, fraction of
#        the value of the unconstrained dispatch, or of 100 if that is 0)
CASES = {
    "co2_le": (False, "primary_energy", "co2_emissions", "<=", 0.5),
    "co2_ge": (False, "primary_energy", "co2_emissions", ">=", 1.2),
    "co2_eq": (False, "primary_energy", "co2_emissions", "==", 0.8),
    "gas_ge": (False, "operational_limit", "gas", ">=", 1.0),
    "co2_storage": (True, "primary_energy", "co2_emissions", "<=", 0.5),
}


def run_case(name):
    storage, gc_type, attribute, sense, fraction = CASES[name]
    n = build_network()
    if storage:
        add_storage(n)

    free = n.copy()
    free.optimize(solver_name="highs")
    if gc_type == "primary_energy":
        value = emissions(free, attribute)
    else:
        value = free.generators_t.p[free.generators.index[
            free.generators.carrier == attribute]].sum().sum()
    if value == 0:
        value = 100.0  # the unconstrained dispatch burns no gas at all

    n.add("GlobalConstraint", "limit", type=gc_type, carrier_attribute=attribute,
          sense=sense, constant=fraction * value)

    network = n.copy()
    network.optimize(solver_name="highs")
    # every case is built so that the limit is binding
    assert network.global_constraints.mu["limit"] != 0
    obj_pypsa = float(network.objective + getattr(network, "objective_constant", 0.0))

    transformation = Transformation(
        capacity_expansion_ucblock=True,
        workdir=OUT_TEST,
        name=f"global_constraint_{name}",
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
        fp_log="smspp_{name}_log.txt",
        fp_solution="smspp_{name}_solution.nc",
        configfile="auto",
        pysmspp_options={},
    )
    transformation.run(network, verbose=False)
    obj_smspp = float(transformation.result.objective_value)

    assert obj_smspp == pytest.approx(obj_pypsa, rel=REL_TOL, abs=ABS_TOL)
    return obj_pypsa, obj_smspp


@pytest.mark.skipif(
    not solver_reads_pollutant_budget(),
    reason="the SMS++ ucblock_solver on PATH does not read pollutant budgets",
)
@pytest.mark.parametrize("name", list(CASES))
def test_global_constraints(name):
    run_case(name)


if __name__ == "__main__":
    for case in CASES:
        print(case, run_case(case))
