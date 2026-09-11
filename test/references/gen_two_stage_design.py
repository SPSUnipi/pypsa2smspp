"""Scenario tree with a SECOND decision stage: the design is corrected once
the climate year is known.

The tree is the same as gen_tree_instance.py builds, climate outside and
demand inside, but the capacity of every expandable technology is split in
two: a part decided at the root, before anything is known, and a part decided
once the climate branch is known and shared by the demand realizations hanging
from it. That is what a multi-stage problem buys and a two-stage one over the
product of the scenarios cannot say: the second part depends on the history,
not on the leaf.

In SMS++ the two parts are two parallel Generator on the same bus, hence two
design Variable per technology in every leaf; which of them is tied where is
what the two levels of here-and-now paths state, the outer one naming the root
part alone and the inner one naming both.

Waiting is not free: the part decided later carries a premium, otherwise one
would always wait and the root part would be pointless.

    python gen_two_stage_design.py --climates 3 --demands 3 --late-premium 1.25

It writes:
  - <name>_flat.nc      the flat network with the LATE part removed, whose
                        optimum is the value of deciding everything at the
                        root: an upper bound on the tree's optimum;
  - <name>_c<k>.nc      one network per climate branch, with its own demand
                        scenarios and everything decided inside the branch:
                        their weighted sum is the wait-and-see value, a lower
                        bound on the tree's optimum;
  - <name>_tree.json    the tree, plus the "stages" section saying which
                        design Variable are decided where.
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
import sys
sys.path.insert(0, str(HERE))

from gen_tree_instance import (climate_axis, demand_axis, deterministic_base,
                               solar_profile, SOLAR_CAPEX)

LATE = "late"          # the suffix of the technology decided after the climate


def split_expandable(n, premium):
    """Give every expandable Generator a twin, to be decided one stage later."""
    root, late = [], []
    for name, generator in list(n.generators.iterrows()):
        if not bool(generator.p_nom_extendable):
            continue
        twin = f"{name} {LATE}"
        n.add("Generator", twin,
              bus=generator.bus, carrier=generator.carrier,
              p_nom=0.0, p_nom_extendable=True,
              p_nom_max=generator.p_nom_max,
              marginal_cost=generator.marginal_cost,
              capital_cost=generator.capital_cost * premium)
        root.append(name)
        late.append(twin)
    return root, late


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--climates", type=int, default=3)
    parser.add_argument("--demands", type=int, default=3)
    parser.add_argument("--buses", type=int, default=1)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--late-premium", type=float, default=1.25,
                        help="what a MW decided after the climate costs, "
                             "relative to one decided at the root")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    name = args.name or (f"pypsa_2stage_c{args.climates}_d{args.demands}"
                         f"_b{args.buses}_t{args.days}")

    n, loads = deterministic_base(args.buses, args.days)
    base_pmaxpu = solar_profile(n.snapshots)
    root_design, late_design = split_expandable(n, args.late_premium)
    # taken here, before any set_scenarios: on a stochastic network the index
    # is a MultiIndex, so g is a tuple and "solar" in g stops being a
    # substring test and becomes an equality one, giving an empty list
    solar = [g for g in n.generators.index if "solar" in g]

    climate_multipliers, climate_weights = climate_axis(args.climates)

    nodes = [{"stage": 0, "parent": None, "probability": 1.0, "name": "root"}]
    leaves, groups, branches = [], {}, []

    for climate, (availability, probability) in enumerate(
            zip(climate_multipliers, climate_weights)):
        climate_name = f"c{climate}"
        p_max_pu = (base_pmaxpu * availability).clip(upper=1.0)

        outer = len(nodes)
        nodes.append({"stage": 1, "parent": 0,
                      "probability": float(probability), "name": climate_name})

        demand_multipliers, demand_weights = demand_axis(
            args.demands, climate, args.climates)

        groups[climate_name] = {"probability": float(probability),
                                "scenarios": {}}
        branch = []

        for demand, (multiplier, conditional) in enumerate(
                zip(demand_multipliers, demand_weights)):
            scenario = f"{climate_name}_d{demand}"
            nodes.append({"stage": 2, "parent": outer,
                          "probability": float(conditional), "name": scenario})
            groups[climate_name]["scenarios"][scenario] = float(conditional)
            leaf = {"scenario": scenario,
                    "joint": float(probability * conditional),
                    "conditional": float(conditional),
                    "load": loads * multiplier,
                    "p_max_pu": p_max_pu}
            leaves.append(leaf)
            branch.append(leaf)

        branches.append({"name": climate_name,
                         "probability": float(probability),
                         "leaves": branch})

    total = sum(leaf["joint"] for leaf in leaves)
    assert abs(total - 1.0) < 1e-9, total

    def scenarios_of(net, chosen, weights):
        net.set_scenarios(dict(zip(chosen, weights)))
        for leaf, scenario in zip(chosen, chosen):
            pass
        return net

    # the root-only network: no late twin at all, everything decided once
    root_only = n.copy()
    root_only.remove("Generator", late_design)
    # the twins are gone from this one, and writing a profile for a Generator
    # that is not there leaves an orphan column behind
    root_solar = [g for g in solar if g in root_only.generators.index]
    root_only.set_scenarios({leaf["scenario"]: leaf["joint"]
                             for leaf in leaves})
    for leaf in leaves:
        for load in loads.columns:
            root_only.loads_t.p_set[leaf["scenario"], load] = \
                leaf["load"][load].to_numpy()
        for generator in root_solar:
            root_only.generators_t.p_max_pu[leaf["scenario"], generator] = \
                leaf["p_max_pu"].to_numpy()
    root_only.export_to_netcdf(str(DATA / f"{name}_flat.nc"))

    # the whole thing: both families of Generator and all the leaves, which is
    # what becomes the tree. Which of the two is decided where is not in the
    # network, it is in the design_stages section of the tree
    full = n.copy()
    full.set_scenarios({leaf["scenario"]: leaf["joint"] for leaf in leaves})
    for leaf in leaves:
        for load in loads.columns:
            full.loads_t.p_set[leaf["scenario"], load] = \
                leaf["load"][load].to_numpy()
        for generator in solar:
            full.generators_t.p_max_pu[leaf["scenario"], generator] = \
                leaf["p_max_pu"].to_numpy()
    full.export_to_netcdf(str(DATA / f"{name}_full.nc"))

    # one network per branch: everything decided knowing the climate
    for branch in branches:
        inner = n.copy()
        inner.remove("Generator", root_design)   # only the late part is left,
        inner.generators.loc[late_design, "capital_cost"] /= args.late_premium
        inner_solar = [g for g in solar if g in inner.generators.index]
        weights = [leaf["conditional"] for leaf in branch["leaves"]]
        names = [leaf["scenario"] for leaf in branch["leaves"]]
        inner.set_scenarios(dict(zip(names, weights)))
        for leaf in branch["leaves"]:
            for load in loads.columns:
                inner.loads_t.p_set[leaf["scenario"], load] = \
                    leaf["load"][load].to_numpy()
            for generator in inner_solar:
                inner.generators_t.p_max_pu[leaf["scenario"], generator] = \
                    leaf["p_max_pu"].to_numpy()
        inner.export_to_netcdf(str(DATA / f"{name}_{branch['name']}.nc"))

    with open(DATA / f"{name}_tree.json", "w") as f:
        json.dump({"name": name, "stages": 3, "groups": groups,
                   "nodes": nodes,
                   "design_stages": {"root": root_design, "branch": late_design},
                   "late_premium": args.late_premium},
                  f, indent=1)

    print(f"{name}: {args.climates} rami climatici x {args.demands} domande = "
          f"{len(leaves)} foglie; design alla radice {len(root_design)}, "
          f"dopo il clima {len(late_design)}, premio {args.late_premium}")


if __name__ == "__main__":
    main()
