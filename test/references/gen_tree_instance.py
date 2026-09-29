"""Two-level (climate, demand) scenario-tree generator for MSSB instances.

Builds, from the deterministic skeleton of the shipped stochastic fixture, a
genuine two-level scenario tree: the outer stage is the climate year, which
scales the availability of the renewable generators, and the inner stage is
the demand, whose realizations *depend on the outer branch*, both in value and
in conditional probability. That dependence is what makes this a tree and not
a fishbone, i.e. what a MultiStageStochasticBlock is for.

Every dimension is a parameter, so that the same recipe gives both the toy
instance one validates on and the big one one measures on:

    python gen_tree_instance.py --climates 8 --demands 6 --buses 4 --days 7

It writes two things:
  - <name>_flat.nc : the equivalent flat PyPSA network, with one scenario per
    leaf and the joint probabilities P(c) * P(d|c). As long as the only
    here-and-now variables are the design ones, this has the very same
    extensive form as the tree, hence it is an exact reference;
  - <name>_tree.json : the tree itself (groups, nodes, parents, conditional
    probabilities, per-node data), to be handed to the MSSB writer.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "output" / "mssb_tree"
DATA.mkdir(parents=True, exist_ok=True)
FIXTURE = HERE.parent / "networks" / "pypsa_stoch_load.nc"

# annualized cost of the renewable, low enough for it to be worth building:
# with a daily profile a MW of it saves some 5.3 MWh of fuel a day
SOLAR_CAPEX = 8.0


def climate_axis(number):
    """The outer stage: availability multipliers with their probabilities."""
    multipliers = np.linspace(0.70, 1.30, number)
    # more weight on the middle years, less on the extreme ones
    weights = 1.0 + np.cos(np.linspace(-np.pi, np.pi, number + 2)[1:-1])
    return multipliers, weights / weights.sum()


def demand_axis(number, climate_index, number_climates):
    """
    The inner stage, conditional on the climate: the drier the year, the
    higher the demand and the more it leans towards its high realizations.
    """
    shift = 0.05 * (1.0 - 2.0 * climate_index / max(number_climates - 1, 1))
    multipliers = np.linspace(0.92, 1.08, number) + shift

    # a skew that changes sign along the climate axis, so that no two outer
    # branches share the same conditional distribution
    skew = np.linspace(-1.0, 1.0, number) * shift * 10.0
    weights = np.clip(1.0 + skew, 0.05, None)
    return multipliers, weights / weights.sum()


def deterministic_base(number_buses, number_days):
    """The fixture without its scenario axis, widened and lengthened."""
    src = pypsa.Network(FIXTURE)
    reference = src.scenarios[0]        # static data is replicated over them
    template_generators = src.generators.loc[reference]
    base_bus = src.buses.loc[reference].index[0]
    base_load = src.loads_t.p_set[reference].iloc[:, 0].to_numpy()

    snapshots = pd.RangeIndex(len(base_load) * number_days)

    n = pypsa.Network()
    n.set_snapshots(snapshots)
    for carrier in ("diesel", "slack", "solar"):
        n.add("Carrier", carrier)

    # one day of the fixture repeated, with a smooth seasonal factor on top,
    # so that the horizon is longer without being periodic
    season = 1.0 + 0.15 * np.sin(np.linspace(0.0, np.pi, len(snapshots)))
    load_profile = np.tile(base_load, number_days) * season

    for bus in range(number_buses):
        bus_name = f"bus{bus}"
        n.add("Bus", bus_name)

        for name, generator in template_generators.iterrows():
            # only the first bus has the dispatchable unit, so that the
            # network has to carry power around instead of sitting idle
            if (generator.carrier == "diesel") and (bus > 0):
                continue

            n.add(
                "Generator", f"{bus_name} {generator.carrier}",
                bus=bus_name, carrier=generator.carrier,
                p_nom=generator.p_nom,
                p_nom_extendable=bool(generator.p_nom_extendable),
                p_nom_max=generator.p_nom_max,
                marginal_cost=generator.marginal_cost,
                capital_cost=generator.capital_cost,
                )

        n.add("Generator", f"{bus_name} solar", bus=bus_name, carrier="solar",
              p_nom=0.0, p_nom_extendable=True, p_nom_max=1e6,
              marginal_cost=0.0, capital_cost=SOLAR_CAPEX)

        n.add("Load", f"{bus_name} load", bus=bus_name)

        # a radial network: a multi-node network with no line at all is not a
        # well-formed DCNetworkBlock, and a single bus would hide the network
        # from the model altogether
        if bus > 0:
            n.add("Link", f"line{bus - 1}", bus0=f"bus{bus - 1}",
                  bus1=bus_name, length=1.0, p_nom=5e5, p_min_pu=-1.0,
                  p_nom_extendable=False, capital_cost=0.0)

    # the buses differ by a fixed factor, so that the model is not symmetric
    loads = pd.DataFrame(
        {f"bus{bus} load": load_profile * (1.0 + 0.1 * bus)
         for bus in range(number_buses)},
        index=snapshots,
        )

    return n, loads


def solar_profile(snapshots):
    """A plain daily profile, peaking at noon."""
    hour = np.arange(len(snapshots)) % 24
    return pd.Series(np.clip(np.sin(np.pi * (hour - 6) / 12.0), 0.0, None),
                     index=snapshots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--climates", type=int, default=3)
    parser.add_argument("--demands", type=int, default=3)
    parser.add_argument("--buses", type=int, default=1)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    name = args.name or (
        f"pypsa_tree_c{args.climates}_d{args.demands}"
        f"_b{args.buses}_t{args.days}"
        )

    n, loads = deterministic_base(args.buses, args.days)
    base_pmaxpu = solar_profile(n.snapshots)
    solar = [g for g in n.generators.index if g.endswith("solar")]

    climate_multipliers, climate_weights = climate_axis(args.climates)

    nodes = [{"stage": 0, "parent": None, "probability": 1.0,
              "name": "root"}]
    leaves, groups = [], {}

    for climate, (availability, probability) in enumerate(
            zip(climate_multipliers, climate_weights)):
        climate_name = f"c{climate}"
        p_max_pu = (base_pmaxpu * availability).clip(upper=1.0)

        outer = len(nodes)
        nodes.append({"stage": 1, "parent": 0, "probability": float(probability),
                      "name": climate_name})

        demand_multipliers, demand_weights = demand_axis(
            args.demands, climate, args.climates)

        groups[climate_name] = {"probability": float(probability),
                                "scenarios": {}}

        for demand, (multiplier, conditional) in enumerate(
                zip(demand_multipliers, demand_weights)):
            scenario = f"{climate_name}_d{demand}"
            nodes.append({"stage": 2, "parent": outer,
                          "probability": float(conditional),
                          "name": scenario})
            groups[climate_name]["scenarios"][scenario] = float(conditional)
            leaves.append({"scenario": scenario,
                           "joint": float(probability * conditional),
                           "load": loads * multiplier,
                           "p_max_pu": p_max_pu})

    total = sum(leaf["joint"] for leaf in leaves)
    assert abs(total - 1.0) < 1e-9, total

    flat = n.copy()
    flat.set_scenarios({leaf["scenario"]: leaf["joint"] for leaf in leaves})
    for leaf in leaves:
        for load in loads.columns:
            flat.loads_t.p_set[leaf["scenario"], load] = \
                leaf["load"][load].to_numpy()
        for generator in solar:
            flat.generators_t.p_max_pu[leaf["scenario"], generator] = \
                leaf["p_max_pu"].to_numpy()

    flat.export_to_netcdf(str(DATA / f"{name}_flat.nc"))

    with open(DATA / f"{name}_tree.json", "w") as f:
        json.dump({"name": name, "stages": 3, "groups": groups,
                   "nodes": nodes}, f, indent=1)

    print(f"{name}: {args.climates} outer x {args.demands} inner = "
          f"{len(leaves)} leaves, {len(nodes)} tree nodes, "
          f"{args.buses} buses, {len(n.snapshots)} snapshots, "
          f"joint probability {total:.12f}")


if __name__ == "__main__":
    main()
