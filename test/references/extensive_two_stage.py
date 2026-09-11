"""The exact optimum of the two-decision tree, written as one flat program.

The two bounds of bounds_two_stage.py say where the optimum has to sit; this
says what it is. The tree is written out in full: one copy of the operational
model per leaf, on its own buses so that nothing flows between leaves, and the
design tied by hand, the root part shared by every leaf and the late part
shared inside a branch alone. That sharing is the whole content of the tree,
and stating it here by hand is what makes this an independent reference for
what the converter emits.

Each copy carries its own probability as the weight of its snapshots, so the
operating cost is an expectation; the same weight multiplies its capital cost,
so that a design shared by a set of copies is paid once, the weights of a set
summing to its probability.
"""
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
sys.path.insert(0, str(HERE))

from gen_tree_instance import climate_axis, demand_axis, deterministic_base

LATE = "late"


def build(climates, demands, buses, days, premium):
    """One network holding every leaf, and who shares what with whom."""
    base, loads = deterministic_base(buses, days)
    horizon = len(base.snapshots)

    # the design of the base network: the extendable Generator are the root
    # part, each of which gets a twin decided one stage later
    root_design = [name for name, g in base.generators.iterrows()
                   if g.p_nom_extendable]

    climate_multipliers, climate_weights = climate_axis(climates)

    leaves = []
    for climate, (availability, probability) in enumerate(
            zip(climate_multipliers, climate_weights)):
        # the fixture carries no availability profile: the climate acts on
        # the demand alone, exactly as the networks the bounds are read from
        multipliers, conditionals = demand_axis(demands, climate, climates)
        for demand, (multiplier, conditional) in enumerate(
                zip(multipliers, conditionals)):
            leaves.append({
                "branch": f"c{climate}",
                "name": f"c{climate}_d{demand}",
                "weight": float(probability * conditional),
                "load": loads * multiplier,
                })

    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(horizon * len(leaves)))
    for carrier in base.carriers.index:
        n.add("Carrier", carrier)

    weights = pd.Series(0.0, index=n.snapshots)
    # name -> the copies of it that have to hold the same value
    shared = {}

    for index, leaf in enumerate(leaves):
        window = n.snapshots[index * horizon:(index + 1) * horizon]
        weights.loc[window] = leaf["weight"]
        tag = leaf["name"]

        for bus in base.buses.index:
            n.add("Bus", f"{bus} {tag}")

        for name, generator in base.generators.iterrows():
            for late in (False, True) if name in root_design else (False,):
                copy = f"{name} {LATE} {tag}" if late else f"{name} {tag}"
                cost = generator.capital_cost * (premium if late else 1.0)
                n.add("Generator", copy,
                      bus=f"{generator.bus} {tag}",
                      carrier=generator.carrier,
                      p_nom=0.0 if generator.p_nom_extendable else generator.p_nom,
                      p_nom_extendable=bool(generator.p_nom_extendable),
                      p_nom_max=generator.p_nom_max,
                      marginal_cost=generator.marginal_cost,
                      # paid once by the set of copies that share it
                      capital_cost=cost * leaf["weight"])

                if generator.p_nom_extendable:
                    # the root part is one decision for the whole tree, the
                    # late part one decision per branch
                    key = (f"{name} {LATE}" , leaf["branch"]) if late \
                          else (name , "root")
                    shared.setdefault(key, []).append(copy)

        for bus in base.buses.index:
            copy = f"{bus} load {tag}"
            n.add("Load", copy, bus=f"{bus} {tag}")
            series = pd.Series(0.0, index=n.snapshots)
            series.loc[window] = leaf["load"][f"{bus} load"].to_numpy()
            n.loads_t.p_set[copy] = series

        for name, link in base.links.iterrows():
            n.add("Link", f"{name} {tag}",
                  bus0=f"{link.bus0} {tag}", bus1=f"{link.bus1} {tag}",
                  length=link.length, p_nom=link.p_nom, p_min_pu=link.p_min_pu,
                  p_nom_extendable=False, capital_cost=0.0)

    n.snapshot_weightings.loc[:, :] = 0.0
    n.snapshot_weightings["objective"] = weights
    n.snapshot_weightings["generators"] = 1.0
    n.snapshot_weightings["stores"] = 1.0

    return n, shared


def tie(n, snapshots, shared):
    """Every set of copies of one decision holds one value."""
    capacity = n.model["Generator-p_nom"]
    for copies in shared.values():
        first = copies[0]
        for other in copies[1:]:
            n.model.add_constraints(
                capacity.loc[first] - capacity.loc[other] == 0,
                name=f"tie {first} {other}")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "pypsa_2stage_c3_d3_b1_t1"
    tree = json.load(open(DATA / f"{name}_tree.json"))

    climates = len(tree["groups"])
    demands = len(next(iter(tree["groups"].values()))["scenarios"])
    premium = float(tree["late_premium"])

    n, shared = build(climates, demands, 1, 1, premium)
    print(f"[albero ] {climates} climi x {demands} domande, "
          f"premio {premium}, {len(n.snapshots)} istanti")
    print(f"[legami ] {len(shared)} decisioni condivise: "
          + ", ".join(f"{k[0]} ({k[1]}, {len(v)} copie)"
                      for k, v in shared.items()))

    n.optimize(solver_name="gurobi", extra_functionality=
               lambda net, sns: tie(net, sns, shared))

    value = float(n.objective + n.objective_constant)
    print(f"forma estensiva esatta   = {value:.6f}")
    return value


if __name__ == "__main__":
    main()
