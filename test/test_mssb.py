# -*- coding: utf-8 -*-
"""
Regression tests for the MultiStageStochasticBlock conversion.

They build a genuine two-level scenario tree out of the network shipped with
the repository: the outer stage is the climate year, which scales the
availability of a renewable generator, and the inner stage is the demand,
whose realizations depend on the branch they hang from, in value and in
probability. Since the only here-and-now variables are the design ones, the
tree has the same extensive form as the flat network with one scenario per
leaf, which is what makes the optimum of the two directly comparable.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

from conftest import OUT_TEST

from pypsa2smspp.network_correction import clean_dispatch_setpoints
from pypsa2smspp.stochastic_utils import normalize_scenario_tree
from pypsa2smspp.transformation import Transformation


HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "networks" / "pypsa_stoch_load.nc"

STOCHASTIC_PARAMETERS = ["demand", "renewable_maxpower"]
MSSB_CONFIGFILE = "TSSBlock/TSSBSCfg.txt"

# the outer stage: how available the renewable is in a climate year
CLIMATE = {"dry": (0.70, 0.3), "normal": (1.00, 0.4), "wet": (1.30, 0.3)}

# the inner stage: the demand, conditional on the climate year. Both the
# levels and the probabilities depend on the branch
DEMAND = {
    "dry": [("low", 0.97, 0.2), ("mid", 1.05, 0.3), ("high", 1.13, 0.5)],
    "normal": [("low", 0.94, 0.3), ("mid", 1.00, 0.4), ("high", 1.06, 0.3)],
    "wet": [("low", 0.90, 0.5), ("mid", 0.96, 0.3), ("high", 1.02, 0.2)],
}

# low enough for the renewable to be worth building: over this horizon a MW of
# it saves about 5.3 MWh of fuel, so that the climate is not inert
SOLAR_CAPEX = 8.0


def _deterministic_base() -> tuple[pypsa.Network, pd.DataFrame]:
    """The shipped network without its scenarios, plus a renewable generator."""
    src = pypsa.Network(str(FIXTURE))
    reference = src.scenarios[0]        # static data is replicated across them

    n = pypsa.Network()
    n.set_snapshots(src.snapshots)

    for bus in src.buses.loc[reference].index:
        n.add("Bus", bus)

    for carrier in ("diesel", "slack", "solar"):
        n.add("Carrier", carrier)

    for name, generator in src.generators.loc[reference].iterrows():
        n.add(
            "Generator",
            name,
            bus=generator.bus,
            carrier=generator.carrier,
            p_nom=generator.p_nom,
            p_nom_extendable=bool(generator.p_nom_extendable),
            p_nom_max=generator.p_nom_max,
            marginal_cost=generator.marginal_cost,
            capital_cost=generator.capital_cost,
        )

    for name, load in src.loads.loc[reference].iterrows():
        n.add("Load", name, bus=load.bus)

    # the generator the climate year acts on, extendable so that its capacity
    # is a genuine first-stage here-and-now decision
    n.add(
        "Generator",
        "solar",
        bus=n.buses.index[0],
        carrier="solar",
        p_nom=0.0,
        p_nom_extendable=True,
        p_nom_max=1e6,
        marginal_cost=0.0,
        capital_cost=SOLAR_CAPEX,
    )

    base_load = src.loads_t.p_set[reference].copy()
    base_load.index = n.snapshots

    return n, base_load


def _solar_profile(snapshots) -> pd.Series:
    """A plain daily profile, peaking at noon."""
    hour = np.arange(len(snapshots)) % 24
    return pd.Series(
        np.clip(np.sin(np.pi * (hour - 6) / 12.0), 0.0, None), index=snapshots
    )


def build_two_level_network() -> tuple[pypsa.Network, dict]:
    """
    Build the flat network of the leaves, and the tree they are the leaves of.
    """
    n, base_load = _deterministic_base()
    base_pmaxpu = _solar_profile(n.snapshots)
    load_name = n.loads.index[0]

    leaves, groups = [], {}

    for climate, (availability, probability) in CLIMATE.items():
        p_max_pu = (base_pmaxpu * availability).clip(upper=1.0)

        groups[climate] = {
            "probability": probability,
            "scenarios": {
                f"{climate}_{demand}": conditional
                for demand, _, conditional in DEMAND[climate]
            },
        }

        for demand, multiplier, conditional in DEMAND[climate]:
            leaves.append(
                {
                    "scenario": f"{climate}_{demand}",
                    "joint": probability * conditional,
                    "load": (base_load * multiplier).clip(lower=0.0),
                    "p_max_pu": p_max_pu,
                }
            )

    n.set_scenarios({leaf["scenario"]: leaf["joint"] for leaf in leaves})

    for leaf in leaves:
        n.loads_t.p_set[leaf["scenario"], load_name] = \
            leaf["load"].values.ravel()
        n.generators_t.p_max_pu[leaf["scenario"], "solar"] = \
            leaf["p_max_pu"].values

    return n, {"groups": groups}


def _transformation(tree: dict, name: str) -> Transformation:
    workdir = OUT_TEST / "mssb" / name
    workdir.mkdir(parents=True, exist_ok=True)

    return Transformation(
        name=name,
        configfile=MSSB_CONFIGFILE,
        enable_thermal_units=False,
        workdir=str(workdir),
        stochastic_parameters={
            "stochastic_type": "mssb",
            "parameters": STOCHASTIC_PARAMETERS,
            "tree": tree,
        },
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
        fp_log="smspp_{name}_log.txt",
        fp_solution="smspp_{name}_solution.nc",
        pysmspp_options={"B": "TSSBCfg.txt"},
    )


# ---------------------------------------------------------------------------
# the scenario tree
# ---------------------------------------------------------------------------

def test_normalize_scenario_tree_reads_both_forms():
    """The mapping and the list of groups describe the same tree."""
    as_mapping = normalize_scenario_tree(
        {"groups": {"a": {"probability": 0.4,
                          "scenarios": {"a0": 0.25, "a1": 0.75}},
                    "b": {"probability": 0.6,
                          "scenarios": {"b0": 1.0}}}}
    )
    as_list = normalize_scenario_tree(
        {"groups": [{"name": "a", "probability": 0.4,
                     "scenarios": [("a0", 0.25), ("a1", 0.75)]},
                    {"name": "b", "probability": 0.6,
                     "scenarios": [("b0", 1.0)]}]}
    )

    assert as_mapping == as_list
    assert [group["name"] for group in as_mapping["groups"]] == ["a", "b"]


@pytest.mark.parametrize(
    "tree",
    [
        # the outer probabilities do not sum to one
        {"groups": {"a": {"probability": 0.4, "scenarios": {"a0": 1.0}}}},
        # the conditional ones do not either
        {"groups": {"a": {"probability": 1.0,
                          "scenarios": {"a0": 0.25, "a1": 0.25}}}},
    ],
)
def test_normalize_scenario_tree_rejects_bad_probabilities(tree):
    with pytest.raises(ValueError):
        normalize_scenario_tree(tree)


def test_scenario_tree_leaves_must_be_the_scenarios():
    """A tree that does not cover the scenarios of the network is refused."""
    n, tree = build_two_level_network()
    scenarios = tree["groups"]["dry"]["scenarios"]
    scenarios["dry_mild"] = scenarios.pop("dry_low")

    with pytest.raises(ValueError, match="leaves of the scenario tree"):
        _transformation(tree, "mssb_bad_tree").consistency_check(n)


# ---------------------------------------------------------------------------
# the emitted block
# ---------------------------------------------------------------------------

def test_mssb_block_structure():
    """
    The conversion emits a MultiStageStochasticBlock holding the whole tree
    and one inner TwoStageStochasticBlock per outer scenario, none of which
    carries a DiscreteScenarioSet of its own.
    """
    n, tree = build_two_level_network()

    transformation = _transformation(tree, "mssb_structure")
    transformation.create_model(n, verbose=False)

    mssb = transformation.sms_network.blocks["Block_0"]
    assert mssb.attributes["type"] == "MultiStageStochasticBlock"
    assert mssb.dimensions["NumberSubBlocks"].value == len(CLIMATE)

    generator = mssb.blocks["ScenarioGenerator"]
    assert generator.attributes["type"] == "MultiStageDiscreteScenarioSet"
    assert generator.dimensions["NumberStages"].value == 3

    number_leaves = sum(len(demands) for demands in DEMAND.values())
    assert generator.dimensions["NumberNodes"].value == \
        1 + len(CLIMATE) + number_leaves

    stages = np.asarray(generator.variables["NodeStage"].data)
    parents = np.asarray(generator.variables["NodeParent"].data)
    probabilities = np.asarray(generator.variables["NodeProbability"].data)

    # node 0 is the only root, and every other node hangs from a node of the
    # stage before its own
    assert stages[0] == 0
    assert [node for node, parent in enumerate(parents)
            if parent >= stages.size] == [0]
    for node in range(1, stages.size):
        assert stages[parents[node]] == stages[node] - 1

    # only the leaves carry data
    assert list(generator.variables["StageScenarioSize"].data)[:2] == [0, 0]

    # the children of the root are the outer scenarios, with their probability
    outer_nodes = [node for node, parent in enumerate(parents) if parent == 0]
    assert len(outer_nodes) == len(CLIMATE)
    assert list(probabilities[outer_nodes]) == pytest.approx(
        [probability for _, probability in CLIMATE.values()]
    )

    # and each of them has its own leaves, whose probabilities are conditional
    for node, demands in zip(outer_nodes, DEMAND.values()):
        leaves = [leaf for leaf, parent in enumerate(parents) if parent == node]
        assert list(probabilities[leaves]) == pytest.approx(
            [conditional for _, _, conditional in demands]
        )
        assert sum(probabilities[leaves]) == pytest.approx(1.0)

    for outer in range(len(CLIMATE)):
        inner = mssb.blocks[f"Block_{outer}"]
        assert inner.attributes["type"] == "TwoStageStochasticBlock"
        assert "DiscreteScenarioSet" not in inner.blocks
        assert "StochasticBlock" in inner.blocks
        assert "StaticAbstractPath" in inner.blocks


# ---------------------------------------------------------------------------
# where the investment is stated
# ---------------------------------------------------------------------------

def test_investment_outside_wraps_the_stochastic_block():
    """
    With investment_outside the investment is stated once, in an
    InvestmentBlock wrapping the stochastic Block, instead of being replicated
    in every scenario.
    """
    n, tree = build_two_level_network()

    transformation = Transformation(
        name="mssb_investment_outside",
        configfile=MSSB_CONFIGFILE,
        enable_thermal_units=False,
        capacity_expansion_ucblock=False,
        workdir=str(OUT_TEST / "mssb" / "investment_outside"),
        stochastic_parameters={
            "stochastic_type": "mssb",
            "parameters": STOCHASTIC_PARAMETERS,
            "tree": tree,
            "investment_outside": True,
        },
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
    )
    transformation.create_model(n, verbose=False)

    root = transformation.sms_network.blocks["Block_0"]
    assert root.attributes["type"] == "InvestmentBlock"

    inner = root.blocks["InnerBlock"]
    assert inner.attributes["type"] == "MultiStageStochasticBlock"
    assert "ScenarioGenerator" in inner.blocks


def test_investment_outside_needs_an_investment_block():
    """It is refused when the investment goes through the UCBlock instead."""
    n, tree = build_two_level_network()

    transformation = Transformation(
        name="mssb_investment_outside_refused",
        configfile=MSSB_CONFIGFILE,
        enable_thermal_units=False,
        workdir=str(OUT_TEST / "mssb" / "investment_outside_refused"),
        stochastic_parameters={
            "stochastic_type": "mssb",
            "parameters": STOCHASTIC_PARAMETERS,
            "tree": tree,
            "investment_outside": True,
        },
        overwrite=True,
    )

    with pytest.raises(ValueError, match="capacity_expansion_ucblock"):
        transformation.consistency_check(n)


# ---------------------------------------------------------------------------
# the optimum
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    shutil.which("mssb_solver") is None,
    reason="the SMS++ mssb_solver tool is not on PATH",
)
def test_mssb_matches_the_flat_optimum():
    """
    The multi-stage block and the flat network with one scenario per leaf have
    the same extensive form, hence the same optimum.
    """
    n, tree = build_two_level_network()

    reference = n.copy()
    # a finite p_set on a dispatchable component is enforced as a fixed
    # dispatch, which would freeze it and inflate the reference
    clean_dispatch_setpoints(reference)
    reference.optimize(solver_name="highs")
    expected = float(reference.objective + reference.objective_constant)

    transformation = _transformation(tree, "mssb_optimum")
    transformation.run(n, verbose=False)
    obtained = float(transformation.result.objective_value)

    # the objective is read back from the log of the solver, which prints it
    # with six digits
    assert obtained == pytest.approx(expected, rel=1e-5)
