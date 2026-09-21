"""Multi-stage (tree) counterpart of the resilient TSSB instances.

The instances under UCBlock/data/nc4/resilient-data are two-stage: one flat
list of scenarios, drawn from a PyPSA-Eur network by test/stoch_generator.py,
in which demand, renewable availability and hydro inflow are all perturbed at
once. This builds their multi-stage counterpart out of the same network, by
splitting that single axis in two:

  - the OUTER stage is the climate year, which scales the availability of the
    renewables and the hydro inflow: what one commits to before knowing it is
    the design;
  - the INNER stage is the demand, whose realizations are drawn *conditional
    on* the climate year, so that both their values and their probabilities
    depend on the branch they hang from.

That dependence on the history is the whole point: a two-stage block over the
same leaves cannot express it, and neither can the baked multi-stage format,
which would duplicate the scenarios branch by branch and lose the tree. Hence
the output is the shared-tree format, one MultiStageDiscreteScenarioSet read
through views by the inner blocks.

Like the two-stage generator, the multipliers are drawn by bounded Latin
Hypercube sampling, and the stress is coupled: a dry, low-availability year is
also the one whose demand leans high.

    python gen_resilient_tree.py --network <pypsa-eur.nc> \
        --climates 4 --demands 3 --snapshots 100

It writes <name>_flat.nc, the equivalent flat network to be used as the
reference, and <name>_tree.json, the tree to hand to the MSSB converter.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "output" / "mssb_tree"
DATA.mkdir(parents=True, exist_ok=True)
TEST = HERE.parents[1]
sys.path.insert(0, str(TEST))

from pypsa2smspp.network_correction import (            # noqa: E402
    add_slack_unit,
    clean_ciclicity_storage,
    clean_e_sum,
    clean_global_constraints,
    clean_storage_units,
    reduce_snapshots_and_scale_costs,
    )

# the same ranges the two-stage generator stresses the network with, split
# over the two stages: the climate one is the availability, the demand one is
# what is left to the short term
CLIMATE_RANGE = (0.70, 1.20)         # renewable availability and hydro inflow
DEMAND_RANGE = (0.60, 1.60)          # demand

# the renewables the climate year acts upon
RENEWABLE_CARRIERS = {"solar", "solar-hsat", "onwind", "offwind-ac",
                      "offwind-dc", "offwind-float", "ror"}

# how much the demand of a branch is shifted by its climate: a dry year is a
# hot one, hence its demand is both higher and more skewed towards its high
# realizations
STRESS_COUPLING = 0.15


def latin_hypercube(number, low, high, rng):
    """Bounded Latin Hypercube samples in [low, high], as in the two-stage
    generator: the interval is explored more evenly than by plain sampling."""
    edges = np.linspace(0.0, 1.0, number + 1)
    points = rng.uniform(edges[:-1], edges[1:])
    rng.shuffle(points)
    return low + points * (high - low)


def prepare_network(path, snapshots, drop_storage):
    """Read the PyPSA-Eur network and clean it the way the two-stage
    generator does, so that the two families are comparable.

    Note that clean_storage_units() *removes* every storage unit, hydro
    included, so it is off by default here: the inflow of the hydro units is
    half of what the climate year acts upon, and dropping them would leave
    the outer stage with the renewables alone.
    """
    n = pypsa.Network(str(path))

    n = clean_e_sum(n)
    n = clean_ciclicity_storage(n)
    n = add_slack_unit(n)
    if drop_storage:
        n = clean_storage_units(n)
    n = clean_global_constraints(n)

    if snapshots and snapshots < len(n.snapshots):
        n = reduce_snapshots_and_scale_costs(n, target=int(snapshots),
                                             scale_capital_costs=False)
    return n


def renewables(n):
    """The generators the climate year acts upon."""
    carriers = n.generators.carrier
    return n.generators.index[carriers.isin(RENEWABLE_CARRIERS)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True,
                        help="the PyPSA-Eur network to draw the tree from")
    parser.add_argument("--climates", type=int, default=4)
    parser.add_argument("--demands", type=int, default=3)
    parser.add_argument("--snapshots", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--climate-acts-on", default="both",
                        choices=["renewables", "hydro", "both"],
                        help="what the climate year scales: the availability "
                             "of the renewables, the hydro inflow, or both. "
                             "The two-stage family has one instance per axis, "
                             "and these are their multi-stage counterparts")
    parser.add_argument("--drop-storage-units", action="store_true",
                        help="remove the storage units, as the two-stage "
                             "generator optionally does; this also removes "
                             "the hydro inflow from the outer stage")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    source = Path(args.network)
    if not source.exists():
        raise SystemExit(
            f"network not found: {source}\n"
            "This is the PyPSA-Eur network the resilient instances are drawn "
            "from; it is not part of the repository."
            )

    name = args.name or (
        f"{source.stem}__snap{args.snapshots}"
        f"__c{args.climates}_d{args.demands}_{args.climate_acts_on}"
        )

    n = prepare_network(source, args.snapshots, args.drop_storage_units)
    rng = np.random.default_rng(args.seed)

    base_load = n.loads_t.p_set.copy()
    base_pmaxpu = n.generators_t.p_max_pu.copy()
    base_inflow = (n.storage_units_t.inflow.copy()
                   if not n.storage_units_t.inflow.empty else None)
    renewable = renewables(n)

    # the outer stage: the climate years, from the driest to the wettest, so
    # that the coupling below is monotone in the branch
    climate = np.sort(latin_hypercube(args.climates, * CLIMATE_RANGE, rng))
    climate_probability = np.full(args.climates, 1.0 / args.climates)

    nodes = [{"stage": 0, "parent": None, "probability": 1.0, "name": "root"}]
    groups, leaves = {}, []

    for index, (availability, probability) in enumerate(
            zip(climate, climate_probability)):
        climate_name = f"climate_{index + 1:03d}"
        outer = len(nodes)
        nodes.append({"stage": 1, "parent": 0, "probability": float(probability),
                      "name": climate_name})

        # the demand of this branch: the scarcer the year, the higher the
        # levels and the heavier the weight of the high ones
        stress = STRESS_COUPLING * (1.0 - 2.0 * index / max(args.climates - 1, 1))
        demand = np.sort(latin_hypercube(args.demands, * DEMAND_RANGE, rng)) \
            * (1.0 + stress)
        weights = np.clip(1.0 + np.linspace(-1.0, 1.0, args.demands) * stress
                          * 4.0, 0.05, None)
        weights = weights / weights.sum()

        groups[climate_name] = {"probability": float(probability),
                                "scenarios": {}}

        for inner, (multiplier, conditional) in enumerate(zip(demand, weights)):
            scenario = f"{climate_name}_demand_{inner + 1:03d}"
            nodes.append({"stage": 2, "parent": outer,
                          "probability": float(conditional),
                          "name": scenario})
            groups[climate_name]["scenarios"][scenario] = float(conditional)
            leaves.append({"scenario": scenario,
                           "joint": float(probability * conditional),
                           "demand": float(multiplier),
                           "availability": float(availability)})

    total = sum(leaf["joint"] for leaf in leaves)
    assert abs(total - 1.0) < 1e-9, total

    # the equivalent flat network: one scenario per leaf, with the joint
    # probabilities. As long as the only here-and-now variables are the design
    # ones this has the same extensive form as the tree, hence it is an exact
    # reference
    flat = n.copy()
    flat.set_scenarios({leaf["scenario"]: leaf["joint"] for leaf in leaves})

    for leaf in leaves:
        scenario = leaf["scenario"]
        flat.loads_t.p_set[scenario] = base_load * leaf["demand"]

        if (args.climate_acts_on in ("renewables", "both")
                and len(renewable) and not base_pmaxpu.empty):
            available = base_pmaxpu.columns.intersection(renewable)
            profile = base_pmaxpu.copy()
            profile[available] = (profile[available] * leaf["availability"]
                                  ).clip(upper=1.0)
            flat.generators_t.p_max_pu[scenario] = profile

        if (args.climate_acts_on in ("hydro", "both")
                and base_inflow is not None):
            flat.storage_units_t.inflow[scenario] = \
                base_inflow * leaf["availability"]

    flat.export_to_netcdf(str(DATA / f"{name}_flat.nc"))

    with open(DATA / f"{name}_tree.json", "w") as f:
        json.dump({"name": name, "stages": 3, "groups": groups,
                   "nodes": nodes}, f, indent=1)

    print(f"{name}: {args.climates} climate years x {args.demands} demands = "
          f"{len(leaves)} leaves, {len(nodes)} tree nodes, "
          f"{len(n.snapshots)} snapshots, {len(n.buses)} buses, "
          f"{len(n.generators)} generators, {len(renewable)} of them "
          f"renewable, the climate acting on {args.climate_acts_on}, "
          f"joint probability {total:.12f}")


if __name__ == "__main__":
    main()
