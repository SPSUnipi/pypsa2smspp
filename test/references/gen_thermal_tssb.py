"""Two-stage instances whose scenario sub-problem is a real unit-commitment
problem, i.e. the family on which a Lagrangian relaxation of the units has
something to say.

The instances that came out of the energy-community pipeline carry one
thermal unit out of fifteen, so their deterministic equivalent is an LP with a
handful of binaries and a MILP solver closes it before a decomposition has
finished reading the file; here the fleet is the point, and its size, the
length of the horizon and the number of scenarios are all parameters:

    python gen_thermal_tssb.py --units 60 --snapshots 168 --scenarios 5

The uncertainty is the usual pair (demand, availability of the renewables),
the here-and-now variables are the capacities of the expandable technologies
(solar, and a battery when asked for), and everything else, the fleet above
all, is decided scenario by scenario. The fleet is heterogeneous on purpose:
minimum up and down times of four to twelve periods, minimum powers between
a third and a half of the maximum one, start-up costs proportional to the
size, and a reserve margin tight enough (`--margin`) that the commitment is
not decided by the demand alone.

It writes `<name>_flat.nc`, the PyPSA network carrying the scenario axis,
which `emit_thermal_tssb.py` turns into a TwoStageStochasticBlock. PyPSA 1.2.3
cannot optimize a network that is at once stochastic and committable (its
committability constraints do not know about the scenario axis), so the
reference value of a `--no-design` instance is the weighted sum of the
per-scenario optima, which `crosscheck_thermal_tssb.py` computes one leaf at a
time.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "output" / "thermal_tssb"

# the four technologies the fleet is drawn from, in merit order: cost per MWh,
# minimum power as a fraction of the maximum one, minimum up and down times,
# ramping rate per period, start-up cost per MW, and the share of the fleet
TECHNOLOGIES = (
    # name      mc    p_min_pu  up  down  ramp  su_cost  share
    ("nuclear", 9.0, 0.50, 12, 12, 0.20, 180.0, 0.15),
    ("coal", 24.0, 0.40, 8, 8, 0.30, 100.0, 0.25),
    ("ccgt", 42.0, 0.35, 4, 4, 0.50, 50.0, 0.35),
    ("ocgt", 78.0, 0.30, 2, 2, 0.80, 20.0, 0.25),
)

SLACK_COST = 3000.0        # value of the energy not served
SOLAR_CAPEX = 1800.0       # annualized, per MW of capacity, over the horizon
BATTERY_CAPEX = 900.0


def fleet(number_units, peak_load, rng):
    """The thermal fleet: sizes drawn around the share of each technology."""
    units = []
    shares = np.array([t[-1] for t in TECHNOLOGIES])
    exact = number_units * shares
    counts = np.floor(exact).astype(int)
    # the largest remainders take the units the rounding left over, and every
    # technology gets one as long as there are units to go round
    for position in np.argsort(exact - counts)[::-1]:
        if counts.sum() >= number_units:
            break
        counts[position] += 1

    capacity = peak_load / max(sum(counts), 1)
    for technology, count in zip(TECHNOLOGIES, counts):
        name, cost, p_min_pu, up, down, ramp, start_up, _ = technology
        for unit in range(count):
            size = capacity * rng.uniform(0.7, 1.4)
            units.append({
                "name": f"{name}{unit}",
                "carrier": name,
                "p_nom": float(size),
                "p_min_pu": p_min_pu,
                "marginal_cost": cost * rng.uniform(0.9, 1.1),
                "start_up_cost": start_up * size,
                "min_up_time": up,
                "min_down_time": down,
                "ramp_limit_up": ramp,
                "ramp_limit_down": ramp,
                "ramp_limit_start_up": max(p_min_pu, ramp),
                "ramp_limit_shut_down": max(p_min_pu, ramp),
                })
    return units


def load_profile(number_snapshots, rng):
    """Two peaks a day, a weekly modulation, and a bit of noise on top."""
    hour = np.arange(number_snapshots) % 24
    daily = (0.78
             + 0.13 * np.exp(-0.5 * ((hour - 9.0) / 2.5) ** 2)
             + 0.22 * np.exp(-0.5 * ((hour - 19.0) / 2.0) ** 2))
    weekly = 1.0 - 0.08 * (((np.arange(number_snapshots) // 24) % 7) >= 5)
    noise = 1.0 + 0.02 * rng.standard_normal(number_snapshots)
    return daily * weekly * noise


def solar_profile(number_snapshots):
    hour = np.arange(number_snapshots) % 24
    return np.clip(np.sin(np.pi * (hour - 6.0) / 12.0), 0.0, None)


def wind_profile(number_snapshots, rng):
    """A slow random walk, so the renewable is not a copy of the demand."""
    steps = rng.standard_normal(number_snapshots).cumsum()
    steps = steps / (np.abs(steps).max() + 1e-9)
    return np.clip(0.45 + 0.35 * steps, 0.02, 0.95)


def scenario_axis(number):
    """Demand and availability multipliers, with their probabilities."""
    if number == 1:
        return np.array([1.0]), np.array([1.0]), np.array([1.0])
    demand = np.linspace(0.92, 1.10, number)
    availability = np.linspace(1.20, 0.75, number)   # the dry year is the busy one
    weights = 1.0 + np.cos(np.linspace(-np.pi, np.pi, number + 2)[1:-1])
    return demand, availability, weights / weights.sum()


def build(args):
    rng = np.random.default_rng(args.seed)
    snapshots = pd.RangeIndex(args.snapshots)

    n = pypsa.Network()
    n.set_snapshots(snapshots)
    for carrier in [t[0] for t in TECHNOLOGIES] + ["solar", "wind", "slack",
                                                   "battery"]:
        n.add("Carrier", carrier)

    shape = load_profile(args.snapshots, rng)
    peak = args.peak_load
    units = fleet(args.units, peak * args.margin, rng)

    for bus in range(args.buses):
        bus_name = f"bus{bus}"
        n.add("Bus", bus_name)
        if bus > 0:
            # the link has to be able to bind, otherwise a radial chain with no
            # losses moves power for free and the optimum is the one of a single
            # bus carrying the whole demand
            n.add("Link", f"link{bus - 1}", bus0=f"bus{bus - 1}", bus1=bus_name,
                  p_nom=args.link_capacity * peak, p_min_pu=-1.0,
                  p_nom_extendable=False)

    for index, unit in enumerate(units):
        # the fleet is laid out along the chain in merit order, the cheap and
        # slow units on the first bus and the peakers on the last one, so that
        # the cheap power has to travel and the links have something to say;
        # spreading the technologies evenly over the buses would make every bus
        # self-sufficient and the network transparent
        bus_name = f"bus{min(args.buses - 1, index * args.buses // len(units))}"
        initial = 1 if index % 3 else 0       # two units out of three start on
        n.add("Generator", f"{bus_name} {unit['name']}",
              bus=bus_name, carrier=unit["carrier"], committable=True,
              p_nom=unit["p_nom"], p_min_pu=unit["p_min_pu"],
              marginal_cost=unit["marginal_cost"],
              start_up_cost=unit["start_up_cost"],
              min_up_time=unit["min_up_time"],
              min_down_time=unit["min_down_time"],
              ramp_limit_up=unit["ramp_limit_up"],
              ramp_limit_down=unit["ramp_limit_down"],
              ramp_limit_start_up=unit["ramp_limit_start_up"],
              ramp_limit_shut_down=unit["ramp_limit_shut_down"],
              up_time_before=initial * unit["min_up_time"],
              down_time_before=(1 - initial) * unit["min_down_time"],
              # the converter writes InitialPower = p_nom for a unit that
              # starts on, and PyPSA leaves the ramp constraint of the first
              # period out altogether when p_init is not given, so the two
              # models only say the same thing if it is
              p_init=initial * unit["p_nom"])

    # the renewables: the wind farm is there, the solar one is what the first
    # stage decides, and the slack unit prices the energy not served
    for bus in range(args.buses):
        bus_name = f"bus{bus}"
        n.add("Generator", f"{bus_name} wind", bus=bus_name, carrier="wind",
              p_nom=0.25 * peak / args.buses, p_nom_extendable=False,
              marginal_cost=0.0)
        n.add("Generator", f"{bus_name} solar", bus=bus_name, carrier="solar",
              p_nom=0.0, p_nom_extendable=not args.no_design, p_nom_max=peak,
              marginal_cost=0.0, capital_cost=SOLAR_CAPEX)
        n.add("Generator", f"{bus_name} slack", bus=bus_name, carrier="slack",
              p_nom=peak, marginal_cost=SLACK_COST)
        n.add("Load", f"{bus_name} load", bus=bus_name)
        if args.battery and not args.no_design:
            n.add("StorageUnit", f"{bus_name} battery", bus=bus_name,
                  carrier="battery", p_nom=0.0, p_nom_extendable=True,
                  max_hours=4.0, capital_cost=BATTERY_CAPEX,
                  efficiency_store=0.95, efficiency_dispatch=0.95,
                  cyclic_state_of_charge=True)

    # the buses carry different shares of the same total demand, so that the
    # network has something to do and the peak stays where the fleet was sized
    share = np.array([1.0 + 0.15 * bus for bus in range(args.buses)])
    share /= share.sum()
    loads = pd.DataFrame(
        {f"bus{bus} load": shape * peak * share[bus]
         for bus in range(args.buses)},
        index=snapshots,
        )
    base_solar = pd.Series(solar_profile(args.snapshots), index=snapshots)
    base_wind = pd.Series(wind_profile(args.snapshots, rng), index=snapshots)

    demand, availability, probability = scenario_axis(args.scenarios)
    names = [f"s{index}" for index in range(args.scenarios)]

    n.set_scenarios(dict(zip(names, probability.astype(float))))
    for scenario, multiplier, factor in zip(names, demand, availability):
        for column in loads.columns:
            n.loads_t.p_set[scenario, column] = \
                (loads[column] * multiplier).to_numpy()
        for bus in range(args.buses):
            n.generators_t.p_max_pu[scenario, f"bus{bus} solar"] = \
                (base_solar * factor).clip(upper=1.0).to_numpy()
            n.generators_t.p_max_pu[scenario, f"bus{bus} wind"] = \
                (base_wind * factor).clip(upper=1.0).to_numpy()

    return n, names, probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=20)
    parser.add_argument("--snapshots", type=int, default=48)
    parser.add_argument("--scenarios", type=int, default=3)
    parser.add_argument("--buses", type=int, default=1)
    parser.add_argument("--peak-load", type=float, default=10000.0)
    parser.add_argument("--margin", type=float, default=1.25,
                        help="thermal capacity over peak load")
    parser.add_argument("--link-capacity", type=float, default=0.10,
                        help="capacity of each link, as a share of the peak")
    parser.add_argument("--battery", action="store_true")
    parser.add_argument("--no-design", action="store_true",
                        help="no expandable capacity: the scenarios decouple, "
                             "and the instance has a per-scenario reference")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    name = args.name or (
        f"tuc_u{args.units}_t{args.snapshots}_s{args.scenarios}"
        f"_b{args.buses}" + ("_nd" if args.no_design else "")
        )

    DATA.mkdir(parents=True, exist_ok=True)
    n, names, probability = build(args)
    n.export_to_netcdf(str(DATA / f"{name}_flat.nc"))

    thermal = int((~n.generators.loc[names[0]].committable.eq(False)).sum())
    print(f"{name}: {thermal} committable units, {args.snapshots} snapshots, "
          f"{args.scenarios} scenarios, {args.buses} buses, "
          f"probability {probability.sum():.12f}")
    print(f"          {DATA / (name + '_flat.nc')}")


if __name__ == "__main__":
    main()
