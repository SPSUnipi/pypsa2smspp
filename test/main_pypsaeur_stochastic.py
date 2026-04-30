# -*- coding: utf-8 -*-
"""
Load an already-stochastic PyPSA network and test the pypsa2smspp TSSB pipeline.

This script does NOT call n.set_scenarios(...) and does NOT overwrite stochastic
time series. It assumes the input network already contains its scenario structure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pypsa

from configs.test_config import TestConfig
from pypsa2smspp.transformation import Transformation
from pypsa2smspp.network_correction import (
    clean_marginal_cost,
    clean_marginal_cost_intermittent,
    add_slack_unit,
)


# =============================================================================
# Paths
# =============================================================================

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

print(">>> FORCED CWD:", Path.cwd())

REPO_ROOT = HERE.parent
SCRIPTS = (REPO_ROOT / "scripts").resolve()
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

if not os.access(OUT, os.W_OK):
    raise PermissionError(f"Output dir not writable: {OUT}.")


# =============================================================================
# User settings
# =============================================================================

# NETWORK_PATH = "/home/pampado/stochastic/pypsa-eur/results/stochastic_eth/test_europe/networks/base_s_stoch_adm___2050.nc"
NETWORK_PATH = r"C:\Users\aless\sms\transformation_pypsa_smspp\test\networks\base_s_stoch_adm___2050.nc"

NAME = "stochastic_network"
FOLDER = "develop/tssb_loaded"

WORKDIR = OUT / FOLDER
WORKDIR.mkdir(parents=True, exist_ok=True)

CONFIG_FP = "application_stochastic.ini"
SMSPP_CONFIGFILE = "TSSBlock/TSSBSCfg_grb.txt"

RUN_PYPSA_REFERENCE = False
EXPORT_PYPSA_LP = True
EXPORT_NETCDF = True

ADD_SLACK = True

# If None, the script infers stochastic parameters from the network.
# Otherwise use, for example:
# STOCHASTIC_PARAMETERS_OVERRIDE = ["demand"]
# STOCHASTIC_PARAMETERS_OVERRIDE = ["demand", "renewables"]
STOCHASTIC_PARAMETERS_OVERRIDE = ["demand"]

SOLVER_NAME = "gurobi"
SOLVER_OPTIONS = {
    "Threads": 32,
    "Method": 2,
    "Crossover": 0,
    "Seed": 123,
    "AggFill": 0,
    "PreDual": 0,
}

APPLY_CLEANERS = True
ZERO_RENEWABLE_MARGINAL_COST = False
REMOVE_GLOBAL_CONSTRAINTS = True
CLEAN_LINK_AND_STORAGE_MARGINAL_COSTS = False


# =============================================================================
# Stochastic-safe helpers
# =============================================================================

def _is_multiindex_columns(df: pd.DataFrame) -> bool:
    """Return True if dataframe columns are a pandas MultiIndex."""
    return isinstance(df.columns, pd.MultiIndex)


def _last_column_level_values(columns: pd.Index | pd.MultiIndex) -> pd.Index:
    """
    Return asset names from dataframe columns.

    For deterministic tables, columns are asset names.
    For stochastic tables, the last MultiIndex level is assumed to be asset names.
    """
    if isinstance(columns, pd.MultiIndex):
        return pd.Index(columns.get_level_values(-1))
    return pd.Index(columns)


def _columns_for_assets(
    columns: pd.Index | pd.MultiIndex,
    assets: Iterable[str],
) -> pd.Index | pd.MultiIndex:
    """
    Select deterministic or stochastic dataframe columns belonging to given assets.
    """
    assets = set(map(str, assets))

    if isinstance(columns, pd.MultiIndex):
        mask = pd.Index(columns.get_level_values(-1)).map(str).isin(assets)
        return columns[mask]

    mask = pd.Index(columns).map(str).isin(assets)
    return columns[mask]


def drop_time_dependent_columns_for_assets(panel, assets: Iterable[str]) -> None:
    """
    Drop columns associated with assets from a PyPSA *_t panel.

    Works both for deterministic columns:
        Index(["gen1", "gen2"])

    and stochastic columns:
        MultiIndex([("low", "gen1"), ("high", "gen1")])
    """
    for key, df in panel.items():
        if not isinstance(df, pd.DataFrame):
            continue

        cols_to_drop = _columns_for_assets(df.columns, assets)

        if len(cols_to_drop) > 0:
            df.drop(columns=cols_to_drop, inplace=True)


def get_scenario_names(n: pypsa.Network) -> list[str]:
    """
    Infer scenario names from a stochastic PyPSA network.

    Priority:
    1. n.scenario_weightings.index
    2. first MultiIndex level in time-dependent tables
    """
    if hasattr(n, "scenario_weightings"):
        sw = getattr(n, "scenario_weightings")
        if isinstance(sw, pd.DataFrame) and len(sw.index) > 0:
            return list(map(str, sw.index))

    for panel_name in [
        "loads_t",
        "generators_t",
        "links_t",
        "stores_t",
        "storage_units_t",
    ]:
        if not hasattr(n, panel_name):
            continue

        panel = getattr(n, panel_name)

        for _, df in panel.items():
            if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
                return list(map(str, df.columns.get_level_values(0).unique()))

    return []


def infer_stochastic_parameters(n: pypsa.Network) -> list[str]:
    """
    Infer which stochastic parameters are present in the network.

    This is intentionally conservative and only checks the tables currently
    supported by your TSSB builder.
    """
    params = []

    if hasattr(n, "loads_t"):
        df = n.loads_t.get("p_set", pd.DataFrame())
        if isinstance(df, pd.DataFrame) and _is_multiindex_columns(df):
            params.append("demand")

    if hasattr(n, "generators_t"):
        df = n.generators_t.get("p_max_pu", pd.DataFrame())
        if isinstance(df, pd.DataFrame) and _is_multiindex_columns(df):
            params.append("renewables")

    # Keep this only if your pypsa2smspp side already supports stochastic prices.
    if hasattr(n, "generators_t"):
        df = n.generators_t.get("marginal_cost", pd.DataFrame())
        if isinstance(df, pd.DataFrame) and _is_multiindex_columns(df):
            params.append("price")

    return params


def total_load_timeseries(n: pypsa.Network) -> pd.Series | pd.DataFrame:
    """
    Return total load over all load assets.

    Deterministic output:
        Series indexed by snapshots.

    Stochastic output:
        DataFrame indexed by snapshots, columns=scenarios.
    """
    if not hasattr(n, "loads_t") or "p_set" not in n.loads_t:
        return pd.Series(0.0, index=n.snapshots)

    p_set = n.loads_t.p_set

    if p_set.empty:
        return pd.Series(0.0, index=n.snapshots)

    if isinstance(p_set.columns, pd.MultiIndex):
        scenario_level = 0
        return p_set.groupby(level=scenario_level, axis=1).sum()

    return p_set.sum(axis=1)

def clean_global_constraints(n, inplace=True):
    """
    Remove all GlobalConstraint components from the network.

    This version is safe for stochastic PyPSA networks where
    n.global_constraints.index may be a MultiIndex, e.g.
    (scenario, global_constraint_name).

    Notes
    -----
    Avoid n.remove("GlobalConstraint", n.global_constraints.index) here,
    because PyPSA/pandas may misinterpret MultiIndex tuple labels.
    """
    net = n if inplace else n.copy()

    if hasattr(net, "global_constraints") and not net.global_constraints.empty:
        net.global_constraints = net.global_constraints.iloc[0:0].copy()

    return net

def add_slack_unit_stochastic_safe(
    n: pypsa.Network,
    exclude_suffixes=("H2", "battery"),
    carrier: str = "slack",
    cost: float = 10000.0,
    capacity_multiplier: float = 10.0,
) -> pypsa.Network:
    """
    Add high-cost slack generators to load buses.

    Unlike the older add_slack_unit, this handles stochastic load tables with
    MultiIndex columns by summing demand scenario-wise.
    """
    total_load = total_load_timeseries(n)

    if isinstance(total_load, pd.DataFrame):
        max_total_demand = float(total_load.max().max())
        min_total_demand = float(total_load.min().min())
    else:
        max_total_demand = float(total_load.max())
        min_total_demand = float(total_load.min())

    def _is_excluded(bus_name: str) -> bool:
        bn = str(bus_name).strip().lower()
        return any(bn.endswith(sfx.lower()) for sfx in exclude_suffixes)

    load_buses = set(n.loads.bus.dropna().astype(str))

    if max_total_demand > 0:
        p_nom = capacity_multiplier * max_total_demand
        p_max_pu = 1.0
        p_min_pu = 0.0
        marginal_cost = cost
    else:
        p_nom = max(abs(min_total_demand), 1.0)
        p_max_pu = 0.0
        p_min_pu = -1.0
        marginal_cost = -cost

    for bus in n.buses.index:
        bus_str = str(bus)

        if bus_str not in load_buses:
            continue

        if _is_excluded(bus_str):
            continue

        slack_name = f"slack_unit {bus_str}"

        if slack_name in n.generators.index:
            continue

        n.add(
            "Generator",
            name=slack_name,
            carrier=carrier,
            bus=bus,
            p_nom=p_nom,
            p_max_pu=p_max_pu,
            p_min_pu=p_min_pu,
            marginal_cost=marginal_cost,
            capital_cost=0.0,
            p_nom_extendable=False,
        )

    return n


def clean_e_sum_safe(n: pypsa.Network) -> pypsa.Network:
    """
    Disable generator energy-sum constraints if present.

    The original version has a likely typo:
        n.generators_e_sum_min = ...
    instead of:
        n.generators.e_sum_min = ...
    """
    for col, value in [
        ("e_sum_max", np.inf),
        ("e_sum_min", -np.inf),
    ]:
        if hasattr(n, "generators") and col in n.generators.columns:
            n.generators[col] = value

    return n


def clean_cyclicity_storage_safe(n: pypsa.Network) -> pypsa.Network:
    """
    Disable cyclic storage constraints.

    This is mostly static and works also for stochastic networks.
    """
    if hasattr(n, "storage_units") and not n.storage_units.empty:
        if "cyclic_state_of_charge" in n.storage_units.columns:
            n.storage_units["cyclic_state_of_charge"] = False

        if "cyclic_state_of_charge_per_period" in n.storage_units.columns:
            n.storage_units["cyclic_state_of_charge_per_period"] = False

        if {"max_hours", "p_nom"}.issubset(n.storage_units.columns):
            n.storage_units["state_of_charge_initial"] = (
                n.storage_units["max_hours"] * n.storage_units["p_nom"]
            )

    if hasattr(n, "stores") and not n.stores.empty:
        if "e_cyclic" in n.stores.columns:
            n.stores["e_cyclic"] = False

        # Typo-safe: PyPSA uses e_cyclic_per_period, not e_cycic_per_period.
        if "e_cyclic_per_period" in n.stores.columns:
            n.stores["e_cyclic_per_period"] = False

    return n


def clean_storage_units_stochastic_safe(n: pypsa.Network) -> pypsa.Network:
    """
    Remove all StorageUnit components and their time-dependent columns.

    Works for deterministic and stochastic *_t tables.
    """
    if not hasattr(n, "storage_units"):
        return n

    storage_units_to_drop = n.storage_units.index.copy()

    if len(storage_units_to_drop) == 0:
        return n

    n.storage_units.drop(storage_units_to_drop, inplace=True)

    if hasattr(n, "storage_units_t"):
        drop_time_dependent_columns_for_assets(n.storage_units_t, storage_units_to_drop)

    return n


def clean_stores_stochastic_safe(
    n: pypsa.Network,
    carriers: Optional[list[str]] = None,
    *,
    remove_store_buses: bool = True,
    remove_generators_on_removed_buses: bool = False,
) -> pypsa.Network:
    """
    Remove Stores and incident Links in a stochastic-safe way.

    The important difference from the older version is that time-dependent columns
    are dropped also when they are MultiIndex columns of the form:
        (scenario, asset)
    """
    if not hasattr(n, "stores") or n.stores.empty:
        return n

    if carriers is None:
        store_idx = n.stores.index.copy()
    else:
        store_idx = n.stores.index[n.stores["carrier"].isin(carriers)].copy()

    if len(store_idx) == 0:
        return n

    store_buses = pd.Index(n.stores.loc[store_idx, "bus"].dropna().unique())

    link_idx = pd.Index([])

    if hasattr(n, "links") and not n.links.empty:
        bus_cols = [col for col in n.links.columns if col.startswith("bus")]

        if bus_cols:
            mask_links = pd.Series(False, index=n.links.index)

            for col in bus_cols:
                mask_links |= n.links[col].isin(store_buses)

            link_idx = n.links.index[mask_links].copy()

            if len(link_idx) > 0:
                n.links.drop(link_idx, inplace=True)

                if hasattr(n, "links_t"):
                    drop_time_dependent_columns_for_assets(n.links_t, link_idx)

    n.stores.drop(store_idx, inplace=True)

    if hasattr(n, "stores_t"):
        drop_time_dependent_columns_for_assets(n.stores_t, store_idx)

    if remove_store_buses and remove_generators_on_removed_buses:
        if hasattr(n, "generators") and not n.generators.empty:
            gen_idx = n.generators.index[n.generators["bus"].isin(store_buses)].copy()

            if len(gen_idx) > 0:
                n.generators.drop(gen_idx, inplace=True)

                if hasattr(n, "generators_t"):
                    drop_time_dependent_columns_for_assets(n.generators_t, gen_idx)

    if remove_store_buses:
        n.buses.drop(store_buses, errors="ignore", inplace=True)

    return n


def apply_network_corrections(n: pypsa.Network) -> pypsa.Network:
    """
    Apply only corrections that are safe or explicitly adapted for stochastic
    networks.
    """
    # if CLEAN_LINK_AND_STORAGE_MARGINAL_COSTS:
    #    n = clean_marginal_cost(n)

    if REMOVE_GLOBAL_CONSTRAINTS:
        n = clean_global_constraints(n)

    # n = clean_e_sum_safe(n)
    # n = clean_cyclicity_storage_safe(n)

    # if ZERO_RENEWABLE_MARGINAL_COST:
    #    n = clean_marginal_cost_intermittent(n)

    if ADD_SLACK:
        n = add_slack_unit_stochastic_safe(n)

    return n


def print_stochastic_summary(n: pypsa.Network) -> None:
    """Print a compact summary of the stochastic structure."""
    scenarios = get_scenario_names(n)
    inferred_params = infer_stochastic_parameters(n)

    print("\n>>> Stochastic network summary")
    print(f"    snapshots: {len(n.snapshots)}")
    print(f"    scenarios: {scenarios if scenarios else 'not detected'}")
    print(f"    inferred stochastic parameters: {inferred_params if inferred_params else 'none'}")

    if hasattr(n, "scenario_weightings"):
        print("    scenario_weightings:")
        print(n.scenario_weightings)

    for panel_name, attrs in {
        "loads_t": ["p_set"],
        "generators_t": ["p_max_pu", "marginal_cost"],
        "links_t": ["p_max_pu", "p_min_pu", "efficiency"],
    }.items():
        if not hasattr(n, panel_name):
            continue

        panel = getattr(n, panel_name)

        for attr in attrs:
            if attr not in panel:
                continue

            df = panel[attr]

            if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
                print(
                    f"    {panel_name}.{attr}: MultiIndex columns "
                    f"levels={df.columns.names}, ncols={len(df.columns)}"
                )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    config = TestConfig(fp=CONFIG_FP)

    print(f">>> Loading network: {NETWORK_PATH}")
    n = pypsa.Network(NETWORK_PATH)

    print_stochastic_summary(n)

    if APPLY_CLEANERS:
        print("\n>>> Applying stochastic-safe corrections")
        n = apply_network_corrections(n)

    if STOCHASTIC_PARAMETERS_OVERRIDE is None:
        stochastic_parameters = infer_stochastic_parameters(n)
    else:
        stochastic_parameters = list(STOCHASTIC_PARAMETERS_OVERRIDE)

    print(f"\n>>> Stochastic parameters passed to pypsa2smspp: {stochastic_parameters}")

    n_pypsa = n.copy()

    obj_pypsa = None
    statistics_pypsa = None

    if RUN_PYPSA_REFERENCE:
        print("\n>>> Solving PyPSA reference")
        n_pypsa.optimize(
            solver_name=SOLVER_NAME,
            solver_options=SOLVER_OPTIONS,
        )

        obj_pypsa = n_pypsa.objective + n_pypsa.objective_constant
        statistics_pypsa = n_pypsa.statistics()

        print(f">>> PyPSA objective: {obj_pypsa}")

    print("\n>>> Running pypsa2smspp transformation")
    transformation = Transformation(
        name=NAME,
        configfile=SMSPP_CONFIGFILE,
        enable_thermal_units=False,
        workdir=str(WORKDIR),
        stochastic_parameters={
            "stochastic_type": "tssb",
            "parameters": stochastic_parameters,
        },
    )

    n_smspp = transformation.run(n)

    statistics_smspp = n_smspp.statistics()
    obj_smspp = n_smspp.objective

    print(f"\n>>> SMS++ objective: {obj_smspp}")

    if obj_pypsa is not None:
        error = (obj_smspp - obj_pypsa) / obj_pypsa * 100
        print(f">>> Error PyPSA-SMS++: {error:.8f}%")

    if EXPORT_NETCDF:
        if RUN_PYPSA_REFERENCE:
            n_pypsa.export_to_netcdf(WORKDIR / f"pypsa_{NAME}.nc")

        n_smspp.export_to_netcdf(WORKDIR / f"smspp_{NAME}.nc")

    if EXPORT_PYPSA_LP and RUN_PYPSA_REFERENCE:
        n_pypsa.model.to_file(fn=WORKDIR / f"pypsa_{NAME}.lp")

    print("\n>>> Done")