# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:59:51 2025

@author: aless
"""


import pypsa
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Mapping, Optional, Union, Sequence
import math

from pypsa2smspp.constants import nominal_attrs

def clean_marginal_cost(n):
    n.links.marginal_cost = 0
    n.storage_units.marginal_cost = 0
    # n.stores.marginal_cost = 0
    return n
    
def clean_global_constraints(n, inplace=True):
    """
    Remove all GlobalConstraint components from the network.
    """
    net = n if inplace else n.copy()

    if not net.global_constraints.empty:
        net.remove("GlobalConstraint", net.global_constraints.index)

    return net

def clean_dispatch_setpoints(n, inplace=True):
    """
    Drop the dispatch set-points (p_set, e_set) of dispatchable components.

    On Generator/Link/Store/StorageUnit a finite p_set (or e_set) is a
    power-flow set-point, not a LOPF input; recent PyPSA enforces it as a
    fixed-dispatch equality constraint, which freezes those components (e.g. a
    p_set=0 on a heat-supply link forces the optimizer to shed the heat demand
    at VOLL instead of serving it). SMS++ ignores these set-points, so leaving
    them in only inflates the PyPSA reference. Load p_set is the demand and is
    kept. Networks built from Excel already drop it in add_component; this
    covers networks loaded from netCDF or built outside that path.
    """
    net = n if inplace else n.copy()

    # component (plural table) -> its dispatch set-point attributes
    dispatchable = {
        "generators": ("p_set",),
        "links": ("p_set",),
        "stores": ("p_set", "e_set"),
        "storage_units": ("p_set",),
    }
    for plural, attrs in dispatchable.items():
        static = getattr(net, plural)
        dynamic = getattr(net, plural + "_t")
        for attr in attrs:
            if attr in static.columns:
                static[attr] = np.nan
            if attr in dynamic and not dynamic[attr].empty:
                dynamic[attr] = dynamic[attr].iloc[:, :0]

    return net

def clean_e_sum(n):
    n.generators.e_sum_max = float('inf')
    n.generators_e_sum_min = float('-inf')
    return n

def clean_efficiency_link(n):
    n.links.efficiency = 1
    return n

def clean_p_min_pu(n):
    n.storage_units.p_min_pu = 0
    return n

def clean_ciclicity_storage(n):
    n.storage_units.cyclic_state_of_charge = False
    n.storage_units.cyclic_state_of_charge_per_period = False
    n.storage_units.state_of_charge_initial = n.storage_units.max_hours * n.storage_units.p_nom
    n.stores.e_cyclic = False
    n.stores.e_cycic_per_period = False
    return n

def clean_marginal_cost_intermittent(n):
    renewable_carriers = ['solar', 'solar-hsat', 'onwind', 'offwind-ac', 'offwind-dc', 'offwind-float', 'PV', 'wind', 'ror']
    renewable_mask = n.generators.carrier.isin(renewable_carriers)
    n.generators.loc[renewable_mask, 'marginal_cost'] = 0.0
    return n

def clean_storage_units(n):
    n.storage_units.drop(n.storage_units.index, inplace=True)
    for key in n.storage_units_t.keys():
        n.storage_units_t[key].drop(columns=n.storage_units_t[key].columns, inplace=True)
    return n

# def clean_stores(n):
#     n.stores.drop(n.stores.index, inplace=True)
#     for key in n.stores_t.keys():
#         n.stores_t[key].drop(columns=n.stores_t[key].columns, inplace=True)
#     return n

def clean_stores(
    n,
    carriers=None,
    *,
    remove_store_buses=True,
    remove_generators_on_removed_buses=False,
):
    """
    Remove Stores (optionally filtered by carrier) and all incident charge/discharge
    Links from a PyPSA network. Optionally also remove the dedicated store buses and
    generators connected to those buses.

    Parameters
    ----------
    n : pypsa.Network
        The network to clean.
    carriers : list[str] or None
        If provided, only stores with carrier in this list are removed
        (e.g. ["battery"], ["H2"]). If None, remove all stores.
    remove_store_buses : bool, optional
        If True, also remove the buses associated with the removed stores.
    remove_generators_on_removed_buses : bool, optional
        If True, also remove generators connected to buses that are removed.
        This flag is only relevant if remove_store_buses=True.

    Returns
    -------
    pypsa.Network
        The modified network (mutated in place and also returned).
    """
    # --- Select stores to remove
    if carriers is None:
        store_idx = n.stores.index
    else:
        store_idx = n.stores.index[n.stores["carrier"].isin(carriers)]

    if len(store_idx) == 0:
        return n

    # --- Buses associated with those stores
    store_buses = pd.Index(n.stores.loc[store_idx, "bus"].unique())

    # --- Remove links incident to any store bus
    if hasattr(n, "links") and not n.links.empty:
        mask_links = n.links["bus0"].isin(store_buses) | n.links["bus1"].isin(store_buses)
        link_idx = n.links.index[mask_links]

        if len(link_idx):
            n.links.drop(link_idx, inplace=True)
            for key, df in n.links_t.items():
                cols = df.columns.intersection(link_idx)
                if len(cols):
                    df.drop(columns=cols, inplace=True)

    # --- Remove stores and store time series
    n.stores.drop(store_idx, inplace=True)
    for key, df in n.stores_t.items():
        cols = df.columns.intersection(store_idx)
        if len(cols):
            df.drop(columns=cols, inplace=True)

    # --- Optionally remove generators connected to removed store buses
    if remove_store_buses and remove_generators_on_removed_buses:
        if hasattr(n, "generators") and not n.generators.empty:
            gen_idx = n.generators.index[n.generators["bus"].isin(store_buses)]
            if len(gen_idx):
                n.generators.drop(gen_idx, inplace=True)
                for key, df in n.generators_t.items():
                    cols = df.columns.intersection(gen_idx)
                    if len(cols):
                        df.drop(columns=cols, inplace=True)

    # --- Optionally remove store buses
    if remove_store_buses:
        n.buses.drop(store_buses, errors="ignore", inplace=True)

    return n


def one_bus_network(n):
    # Delete lines
    n.lines.drop(n.lines.index, inplace=True)
    for key in n.lines_t.keys():
        n.lines_t[key].drop(columns=n.lines_t[key].columns, inplace=True)
        
    # Delete links
    n.links.drop(n.links.index, inplace=True)
    for key in n.links_t.keys():
        n.links_t[key].drop(columns=n.links_t[key].columns, inplace=True)
        
    
    n.buses = n.buses.iloc[[0]]
    n.loads = n.loads.iloc[[0]]

    n.generators['bus'] = n.buses.index[0]
    n.storage_units['bus'] = n.buses.index[0]
    n.stores['bus'] = n.buses.index[0]
    n.loads['bus'] = n.buses.index[0]
    
    n.loads_t.p_set = pd.DataFrame(n.loads_t.p_set.sum(axis=1), index=n.loads_t.p_set.index, columns=[n.buses.index[0]])
    
    return n


def reduce_snapshots_and_scale_costs(
    n,
    target: Union[int, Sequence, pd.Index],
    *,
    drop_renewables: bool = False,
    renewable_carriers: Sequence[str] = (
        "solar",
        "solar-hsat",
        "onwind",
        "offwind-ac",
        "offwind-dc",
        "offwind-float",
        "PV",
        "wind",
        "ror",
    ),
    adjust_snapshot_weightings: bool = False,
    evenly_spaced: bool = False,
    scale_capital_costs: bool = True,
    inplace: bool = False,
    stochastic: Optional[bool] = None,
):
    """
    Reduce the number of snapshots in a PyPSA network and optionally scale
    capital costs.

    This function supports both deterministic and stochastic PyPSA-like networks.
    In the stochastic case, time-dependent tables may have MultiIndex columns,
    typically of the form (scenario, asset). The function reduces only the
    temporal index and keeps all scenario columns.

    Parameters
    ----------
    n : pypsa.Network
        The network to modify.

    target : int | Sequence | pd.Index
        If int:
            - keep that many snapshots;
            - if `evenly_spaced=True`, pick them evenly spaced across the
              original horizon;
            - otherwise, keep the first snapshots.
        If a sequence/index of snapshot labels, keep exactly those snapshots.

    drop_renewables : bool, optional
        If True, drop generators whose carrier is in `renewable_carriers`.

    renewable_carriers : Sequence[str], optional
        Carriers to drop if `drop_renewables` is True.

    adjust_snapshot_weightings : bool, optional
        If True, scale n.snapshot_weightings["objective"] by old_len / new_len
        on the retained snapshots.

    evenly_spaced : bool, optional
        If True and `target` is int, select snapshots evenly spaced instead of
        the first ones.

    scale_capital_costs : bool, optional
        If True, scale component capital_cost by new_len / old_len.

    inplace : bool, optional
        If False, operate on a copy and return it. If True, modify `n` in place.

    stochastic : bool | None, optional
        Optional flag for readability. The implementation is automatic and does
        not require this flag. It is kept only to make calls explicit if desired.

    Returns
    -------
    pypsa.Network
        The modified network, or the same object if `inplace=True`.
    """

    def _slice_rows_by_snapshots(obj, snapshots: pd.Index):
        """
        Slice an object on its time index.

        This supports:
        - simple Index matching snapshots directly;
        - MultiIndex where one level contains the snapshot labels.
        """
        if not hasattr(obj, "loc"):
            return obj

        if not isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj

        idx = obj.index

        if isinstance(idx, pd.MultiIndex):
            for level in range(idx.nlevels):
                level_values = idx.get_level_values(level)
                if snapshots.isin(level_values).all():
                    mask = level_values.isin(snapshots)
                    return obj.loc[mask]

            raise ValueError(
                "Could not slice a time-dependent table with MultiIndex rows: "
                "none of the index levels contains all requested snapshots."
            )

        return obj.loc[snapshots]

    def _drop_assets_from_temporal_df(df: pd.DataFrame, assets: pd.Index) -> pd.DataFrame:
        """
        Drop asset columns from a deterministic or stochastic time-dependent table.

        In deterministic tables, columns are usually asset names.

        In stochastic tables, columns may be a MultiIndex such as:
            (scenario, asset)

        The function drops columns whose last level matches the selected assets.
        If that fails, it drops columns whose any level matches the selected assets.
        """
        if not isinstance(df, pd.DataFrame):
            return df

        if len(assets) == 0:
            return df

        if isinstance(df.columns, pd.MultiIndex):
            cols = df.columns

            # Prefer the last level, because stochastic columns are usually
            # structured as (scenario, asset).
            last_level = cols.get_level_values(cols.nlevels - 1)
            mask = last_level.isin(assets)

            # Fallback: drop columns where any level matches an asset name.
            if not mask.any():
                mask = np.zeros(len(cols), dtype=bool)
                for level in range(cols.nlevels):
                    mask |= cols.get_level_values(level).isin(assets)

            return df.loc[:, ~mask]

        return df.drop(columns=assets, errors="ignore")

    def _asset_names_from_component_index(index: pd.Index) -> pd.Index:
        """
        Return physical asset names from a component index.

        In deterministic networks, the index is usually directly the asset name.

        In stochastic networks, the index may be a MultiIndex such as:
            (scenario, asset)

        In that case, the physical asset name is assumed to be the last level.
        """
        if isinstance(index, pd.MultiIndex):
            return pd.Index(index.get_level_values(index.nlevels - 1).unique())

        return pd.Index(index)

    net = n if inplace else n.copy()

    old_snapshots = pd.Index(net.snapshots.copy())
    old_len = len(old_snapshots)

    if old_len == 0:
        raise ValueError("Network has no snapshots to reduce.")

    if isinstance(target, int):
        if target <= 0:
            raise ValueError("target (int) must be > 0.")

        if target > old_len:
            raise ValueError(
                f"target ({target}) cannot exceed original snapshots ({old_len})."
            )

        if evenly_spaced:
            positions = np.linspace(0, old_len - 1, target)
            positions = np.rint(positions).astype(int)
            positions = np.clip(positions, 0, old_len - 1)

            # Preserve order while removing possible duplicates.
            positions = pd.Index(positions).drop_duplicates().to_numpy()

            new_snapshots = old_snapshots[positions]
        else:
            new_snapshots = old_snapshots[:target]

    else:
        new_snapshots = pd.Index(target)

        missing = new_snapshots.difference(old_snapshots)
        if len(missing) > 0:
            raise ValueError(
                "Some requested snapshots are not in the network: "
                f"{missing[:5].tolist()} ..."
            )

    new_len = len(new_snapshots)

    if new_len == 0:
        raise ValueError("Resulting snapshot set is empty.")

    # Set network snapshots.
    net.snapshots = new_snapshots

    # Slice snapshot weightings if present.
    if hasattr(net, "snapshot_weightings"):
        sw = net.snapshot_weightings

        if isinstance(sw, pd.DataFrame):
            net.snapshot_weightings = _slice_rows_by_snapshots(sw, new_snapshots)

            if adjust_snapshot_weightings and "objective" in net.snapshot_weightings.columns:
                net.snapshot_weightings.loc[:, "objective"] = (
                    net.snapshot_weightings["objective"] * (old_len / new_len)
                )

    # Slice all time-dependent component tables.
    for attr in dir(net):
        if not attr.endswith("_t"):
            continue

        df_dict = getattr(net, attr)

        if not isinstance(df_dict, dict):
            continue

        for key, df in list(df_dict.items()):
            if isinstance(df, (pd.DataFrame, pd.Series)):
                df_dict[key] = _slice_rows_by_snapshots(df, new_snapshots)

    # Optionally drop renewable generators.
    if drop_renewables and hasattr(net, "generators"):
        generators = net.generators

        if isinstance(generators, pd.DataFrame) and "carrier" in generators.columns:
            renewable_rows = generators["carrier"].isin(renewable_carriers)

            rows_to_drop = generators.index[renewable_rows]
            assets_to_drop = _asset_names_from_component_index(rows_to_drop)

            if len(rows_to_drop) > 0:
                net.generators.drop(index=rows_to_drop, inplace=True)

            if hasattr(net, "generators_t"):
                for key, df in list(net.generators_t.items()):
                    net.generators_t[key] = _drop_assets_from_temporal_df(
                        df,
                        assets_to_drop,
                    )

    # Optionally scale capital costs.
    if scale_capital_costs:
        scale_capex = new_len / old_len

        capex_tables = [
            "generators",
            "links",
            "stores",
            "storage_units",
            "lines",
            "transformers",
        ]

        for tab in capex_tables:
            if not hasattr(net, tab):
                continue

            df = getattr(net, tab)

            if isinstance(df, pd.DataFrame) and "capital_cost" in df.columns:
                df.loc[:, "capital_cost"] = df["capital_cost"] * scale_capex

    return net


def split_traditional_generators_into_modules(
    n,
    module_sizes: Mapping[str, float],
    n_modules: Union[int, Mapping[str, int]],
    *,
    randomize: Optional[Mapping[str, Union[float, Mapping[str, float]]]] = None,
    seed: Optional[int] = None,
    round_mode: str = "nearest",
    split_extendable: bool = False,
    scale_startup_like_costs: bool = True,
    startup_like_cost_attrs: tuple[str, ...] = (
        "start_up_cost",
        "shut_down_cost",
        "stand_by_cost",
    ),
    suffix_template: str = "{name}__mod{idx:02d}",
    inplace: bool = True,
    verbose: bool = True,
    max_capacity_deviation: float = 0.10
):
    """
    Split selected conventional PyPSA generators into several modular generators.

    Parameters
    ----------
    n : pypsa.Network
        Input PyPSA network.

    module_sizes : mapping
        Mapping carrier -> base module size in MW.
        Example:
            {
                "coal": 195,
                "diesel": 230,
                "gas": 500,
            }

    n_modules : int | mapping
        Number of modules created for each selected generator.
        If an integer is provided, the same value is used for all carriers.
        If a mapping is provided, it is interpreted as carrier -> number of modules.

    randomize : mapping, optional
        Mapping attribute -> randomization settings.

        Simple form:
            {"marginal_cost": 0.05}

        means normal multiplicative perturbation with std = 5%.

        Extended form:
            {
                "marginal_cost": {"std_pct": 0.05, "mean_pct": 0.0, "clip_pct": 0.15},
                "start_up_cost": {"std_pct": 0.10, "clip_pct": 0.30},
            }

        The multiplier is:
            1 + Normal(mean_pct, std_pct)

        If clip_pct is provided, the multiplier is clipped to:
            [1 - clip_pct, 1 + clip_pct]

        Values above 1 are interpreted as percentages.
        For example 5 means 5%, while 0.05 also means 5%.

    seed : int, optional
        Random seed for reproducible perturbations.

    round_mode : {"nearest", "floor", "ceil"}
        Rule used to round each module capacity to a multiple of the carrier base size.

    split_extendable : bool
        If False, generators with p_nom_extendable=True are skipped.

    scale_startup_like_costs : bool
        If True, startup-like costs are divided by the number of modules before
        random perturbations are applied. This approximately preserves the total
        startup-like cost when all modules are started.

    startup_like_cost_attrs : tuple of str
        Static attributes to divide by the number of modules.

    suffix_template : str
        Template used to name the new module generators.

    inplace : bool
        If True, modify the input network in place.
        If False, work on a copy.

    verbose : bool
        Print a compact summary.

    Returns
    -------
    pypsa.Network
        Network with selected generators split into modules.
    """
    net = n if inplace else n.copy()

    if round_mode not in {"nearest", "floor", "ceil"}:
        raise ValueError("round_mode must be one of: 'nearest', 'floor', 'ceil'.")

    if randomize is None:
        randomize = {}

    rng = np.random.default_rng(seed)

    module_sizes = dict(module_sizes)
    target_carriers = set(module_sizes)

    if isinstance(n_modules, Mapping):
        modules_by_carrier = dict(n_modules)
    else:
        modules_by_carrier = {carrier: int(n_modules) for carrier in target_carriers}

    def _as_fraction(value: float) -> float:
        """Interpret both 5 and 0.05 as 5%."""
        value = float(value)
        return value / 100.0 if abs(value) > 1.0 else value

    def _parse_random_settings(settings: Union[float, Mapping[str, float]]) -> dict[str, Optional[float]]:
        """Normalize randomization settings."""
        if isinstance(settings, Mapping):
            mean_pct = _as_fraction(settings.get("mean_pct", 0.0))
            std_pct = _as_fraction(settings.get("std_pct", 0.0))
            clip_pct = settings.get("clip_pct", None)
            clip_pct = None if clip_pct is None else _as_fraction(clip_pct)
        else:
            mean_pct = 0.0
            std_pct = _as_fraction(settings)
            clip_pct = None

        return {
            "mean_pct": mean_pct,
            "std_pct": std_pct,
            "clip_pct": clip_pct,
        }

    def _draw_multiplier(attr: str) -> float:
        """Draw one multiplicative random factor for one attribute."""
        if attr not in randomize:
            return 1.0

        settings = _parse_random_settings(randomize[attr])
        multiplier = 1.0 + rng.normal(settings["mean_pct"], settings["std_pct"])

        clip_pct = settings["clip_pct"]
        if clip_pct is not None:
            multiplier = float(np.clip(multiplier, 1.0 - clip_pct, 1.0 + clip_pct))

        return float(multiplier)

    def _round_to_module_size(value: float, base_size: float) -> float:
        """Round a capacity to a positive multiple of base_size."""
        ratio = float(value) / float(base_size)

        if round_mode == "nearest":
            multiple = round(ratio)
        elif round_mode == "floor":
            multiple = math.floor(ratio)
        elif round_mode == "ceil":
            multiple = math.ceil(ratio)
        else:
            raise RuntimeError("Unexpected round_mode.")

        multiple = max(1, int(multiple))
        return float(multiple * base_size)

    selected = net.generators.index[
        net.generators["carrier"].isin(target_carriers)
    ].tolist()

    if not split_extendable and "p_nom_extendable" in net.generators.columns:
        selected = [
            gen
            for gen in selected
            if not bool(net.generators.at[gen, "p_nom_extendable"])
        ]

    if not selected:
        if verbose:
            print("[split_generators] No generators selected.")
        return net

    new_rows = []
    old_names_to_drop = []
    dynamic_columns_to_add: dict[str, dict[str, pd.Series]] = {}
    report_rows = []

    for old_name in selected:
        old_row = net.generators.loc[old_name].copy()
        carrier = old_row["carrier"]

        if carrier not in modules_by_carrier:
            continue

        requested_k = int(modules_by_carrier[carrier])
        if requested_k <= 0:
            raise ValueError(f"Invalid number of modules for carrier {carrier}: {requested_k}")

        base_size = float(module_sizes[carrier])
        old_p_nom = float(old_row.get("p_nom", 0.0))

        if old_p_nom <= 0:
            if verbose:
                print(
                    f"[WARN] Generator '{old_name}' has non-positive p_nom={old_p_nom:.6g}. "
                    "Skipping split."
                )
            continue

        # If the generator is already smaller than the base module size, keep it unchanged.
        if old_p_nom < base_size:
            if verbose:
                print(
                    f"[split_generators] Generator '{old_name}' with carrier '{carrier}' has "
                    f"p_nom={old_p_nom:.6g} MW < base module size {base_size:.6g} MW. "
                    "Keeping it unchanged."
                )
            continue

        # In this version, the base size is the actual size of each module.
        # We choose the number of modules that best approximates the original capacity,
        # without exceeding the requested number of modules.
        candidates = []

        for candidate_k in range(1, requested_k + 1):
            candidate_total_p_nom = candidate_k * base_size
            candidate_deviation = (candidate_total_p_nom - old_p_nom) / old_p_nom

            candidates.append(
                {
                    "k": candidate_k,
                    "module_p_nom": base_size,
                    "total_p_nom": candidate_total_p_nom,
                    "deviation": candidate_deviation,
                    "abs_deviation": abs(candidate_deviation),
                }
            )

        best = min(candidates, key=lambda x: x["abs_deviation"])

        k = int(best["k"])
        module_p_nom = float(best["module_p_nom"])

        requested_total_p_nom = requested_k * base_size
        requested_deviation = (requested_total_p_nom - old_p_nom) / old_p_nom

        if k != requested_k and verbose:
            print(
                f"[WARN] Generator '{old_name}' with carrier '{carrier}' has "
                f"p_nom={old_p_nom:.6g} MW. Requested {requested_k} modules of "
                f"{base_size:.6g} MW would give {requested_total_p_nom:.6g} MW total "
                f"({requested_deviation * 100:.2f}% deviation). "
                f"Using {k} modules of {base_size:.6g} MW instead "
                f"({k * base_size:.6g} MW total, {best['deviation'] * 100:.2f}% deviation)."
            )

        if abs(best["deviation"]) > max_capacity_deviation and verbose:
            print(
                f"[WARN] Closest modular split for generator '{old_name}' still exceeds "
                f"the allowed ±{max_capacity_deviation * 100:.2f}% capacity deviation. "
                "The closest available split is used anyway."
            )

        module_names = [
            suffix_template.format(name=old_name, idx=i + 1)
            for i in range(k)
        ]

        # Draw one multiplier per module and randomized attribute.
        multipliers_by_module = {
            module_name: {
                attr: _draw_multiplier(attr)
                for attr in randomize
            }
            for module_name in module_names
        }

        for module_name in module_names:
            new_row = old_row.copy()
            new_row.name = module_name

            # Split nominal capacity.
            if "p_nom" in new_row.index:
                new_row["p_nom"] = module_p_nom

            # Scale extendable capacity bounds if present.
            for attr in ("p_nom_min", "p_nom_max"):
                if attr in new_row.index and pd.notna(new_row[attr]):
                    new_row[attr] = float(new_row[attr]) / k

            # Divide startup-like costs by the number of modules.
            if scale_startup_like_costs:
                for attr in startup_like_cost_attrs:
                    if attr in new_row.index and pd.notna(new_row[attr]):
                        new_row[attr] = float(new_row[attr]) / k

            # Apply random static perturbations.
            for attr, multiplier in multipliers_by_module[module_name].items():
                if attr in new_row.index and pd.notna(new_row[attr]):
                    new_row[attr] = float(new_row[attr]) * multiplier

            new_rows.append(new_row)

        # Duplicate dynamic generator time series.
        for attr, df in net.generators_t.items():
            if old_name not in df.columns:
                continue

            if attr not in dynamic_columns_to_add:
                dynamic_columns_to_add[attr] = {}

            for module_name in module_names:
                series = df[old_name].copy()

                # Apply random dynamic perturbations, if the randomized attribute
                # is time-dependent.
                if attr in randomize:
                    series = series * multipliers_by_module[module_name][attr]

                dynamic_columns_to_add[attr][module_name] = series

        old_names_to_drop.append(old_name)

        report_rows.append(
            {
                "generator": old_name,
                "carrier": carrier,
                "old_p_nom": old_p_nom,
                "n_modules": k,
                "base_module_size": base_size,
                "new_module_p_nom": module_p_nom,
                "new_total_p_nom": k * module_p_nom,
                "delta_p_nom": k * module_p_nom - old_p_nom,
            }
        )

    if not new_rows:
        if verbose:
            print("[split_generators] No new module rows created.")
        return net

    new_generators = pd.DataFrame(new_rows)
    net.generators = pd.concat([net.generators, new_generators], axis=0)

    # Add dynamic columns for the new module generators.
    for attr, columns in dynamic_columns_to_add.items():
        for module_name, series in columns.items():
            net.generators_t[attr][module_name] = series

    # Remove original generators and their dynamic columns.
    net.generators = net.generators.drop(index=old_names_to_drop)

    for attr, df in net.generators_t.items():
        cols_to_drop = [name for name in old_names_to_drop if name in df.columns]
        if cols_to_drop:
            net.generators_t[attr] = df.drop(columns=cols_to_drop)

    if verbose:
        report = pd.DataFrame(report_rows)
        print("[split_generators] Split summary:")
        print(report.to_string(index=False))

    return net

def add_slack_unit(n, exclude_suffixes=("H2", "battery")):
    """
    Add a high-cost slack generator only to buses that actually host at least one load,
    excluding buses whose names end with specific suffixes (e.g. H2, battery).

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to which slack units will be added.
    exclude_suffixes : tuple of str, optional
        Bus-name suffixes (case-insensitive) to exclude.
        Default: ("H2", "battery")

    Returns
    -------
    n : pypsa.Network
        The network with slack generators added.
    """
    # Compute the maximum and minimum total demand over all time steps
    total_demand = n.loads_t.p_set.sum(axis=1)
    max_total_demand = total_demand.max()
    min_total_demand = total_demand.min()

    # Helper to decide if a bus should be excluded
    def _is_excluded(bus_name: str) -> bool:
        bn = str(bus_name).strip().lower()
        return any(bn.endswith(sfx.lower()) for sfx in exclude_suffixes)

    # Keep only buses that actually have at least one load attached
    load_buses = set(n.loads.bus.dropna().astype(str))

    # Iterate only over buses with load and not excluded
    for bus in n.buses.index:
        if str(bus) not in load_buses:
            continue
        if _is_excluded(bus):
            continue

        n.add(
            "Generator",
            name=f"slack_unit {bus}",
            carrier="slack",
            bus=bus,
            p_nom=10 * max_total_demand if max_total_demand > 0 else -min_total_demand,
            p_max_pu=1 if max_total_demand > 0 else 0,
            p_min_pu=0 if max_total_demand > 0 else -1,
            marginal_cost=10000 if max_total_demand > 0 else -10000,
            capital_cost=0,
            p_nom_extendable=False,
        )

    return n



def from_investment_to_uc(n):
    """
    For each investment component in the network, set the nominal capacity equal to
    the optimized value (if available) and turn off extendability.

    Notes:
    - Uses 'nominal_attrs' to map component class -> nominal attribute (e.g., p_nom/s_nom/e_nom).
    - Handles irregular plural names like 'storage_units'.
    - Keeps existing nominal values where *_opt is NaN or missing.
    """
    # Map irregular (or just explicit) plural attribute names on the Network
    component_df_name = {
        "Generator": "generators",
        "Line": "lines",
        "Transformer": "transformers",
        "Link": "links",
        "Store": "stores",
        "StorageUnit": "storage_units",  # <- the tricky one
    }

    for comp, nominal_attr in nominal_attrs.items():
        df_name = component_df_name.get(comp, comp.lower() + "s")
        if not hasattr(n, df_name):
            continue  # component not present in this network

        df = getattr(n, df_name)
        opt_attr = f"{nominal_attr}_opt"
        extend_attr = f"{nominal_attr}_extendable"

        # If optimized values exist, copy them where available, otherwise keep current
        if opt_attr in df.columns:
            # Fill only where *_opt is not NaN; keep original elsewhere
            df[nominal_attr] = df[opt_attr].fillna(df[nominal_attr])

        # Disable extendability if the flag exists
        if extend_attr in df.columns:
            df[extend_attr] = False
    return n

def parse_txt_file(file_path):
    data = {'DCNetworkBlock': {'PowerFlow': []}}
    current_block = None  

    with open(file_path, "r") as file:
        for line in file:
            match_time = re.search(r"Elapsed time:\s*([\deE\+\.-]+)\s*s", line)
            if match_time:
                elapsed_time = float(match_time.group(1))
                data['elapsed_time'] = elapsed_time
                continue 
            
            block_match = re.search(r"(ThermalUnitBlock|BatteryUnitBlock|IntermittentUnitBlock|HydroUnitBlock|DCNetworkBlock)\s*(\d*)", line)
            if block_match:
                base_block = block_match.group(1)
                block_number = block_match.group(2) or "0"  # Se non c'è numero, usa "0"

                block_key = f"{base_block}_{block_number}"  # Nome univoco del blocco

                if base_block != 'DCNetworkBlock':
                    if base_block not in data:
                        data[base_block] = {}  # Ora un dizionario invece di una lista
                    data[base_block][block_key] = {}  # Crea il nuovo blocco
                    current_block = block_key
                else:
                    current_block = 'DCNetworkBlock'
                continue  

            match = re.match(r"([\w\s]+?)(?:\s*\[(\d+)\])?\s+=\s+\[([^\]]*)\]", line)
            if match and current_block:
                key_base, sub_index, values = match.groups()
                key_base = key_base.strip()
            
                if current_block == 'DCNetworkBlock':
                    data[current_block]['PowerFlow'].extend([float(x) for x in values.split()])
                else:
                    base_block = current_block.split("_")[0]
                    block_data = data[base_block][current_block]
            
                    if sub_index is not None:
                        # Se la chiave esiste ed è già un array, va trasformata in un dizionario
                        if key_base in block_data and not isinstance(block_data[key_base], dict):
                            existing_value = block_data[key_base]
                            block_data[key_base] = {0: existing_value}  # Sposta il valore precedente come indice 0
                    
                        if key_base not in block_data:
                            block_data[key_base] = {}
                    
                        block_data[key_base][int(sub_index)] = np.array([float(x) for x in values.split()])
                    else:
                        # Se è già stato salvato come dizionario con indici, inseriamo in 0
                        if key_base in block_data and isinstance(block_data[key_base], dict):
                            block_data[key_base][0] = np.array([float(x) for x in values.split()])
                        else:
                            block_data[key_base] = np.array([float(x) for x in values.split()])

    if data['DCNetworkBlock']['PowerFlow']:
        data['DCNetworkBlock']['PowerFlow'] = np.array(data['DCNetworkBlock']['PowerFlow'])

    return data

import pandas as pd


def _normalize_index(idx):
    """Return index as strings for safer comparison after Excel roundtrip."""
    return pd.Index(idx).map(str)


def _prepare_df(df):
    """Return a sorted copy with stringified index/columns."""
    out = df.copy()
    out.index = _normalize_index(out.index)
    out = out.sort_index()

    out.columns = pd.Index(out.columns).map(str)
    out = out.reindex(sorted(out.columns), axis=1)

    return out


def _is_numeric_series(s):
    """Check whether a Series is numeric-like."""
    return pd.api.types.is_numeric_dtype(s)


def compare_dataframes(
    df1,
    df2,
    name="dataframe",
    rtol=1e-9,
    atol=1e-12,
    compare_dtypes=False,
):
    """
    Compare two DataFrames robustly.

    Returns a list of dictionaries describing differences.
    """
    diffs = []

    df1 = _prepare_df(df1)
    df2 = _prepare_df(df2)

    all_index = df1.index.union(df2.index)
    all_cols = df1.columns.union(df2.columns)

    df1 = df1.reindex(index=all_index, columns=all_cols)
    df2 = df2.reindex(index=all_index, columns=all_cols)

    for col in all_cols:
        s1 = df1[col]
        s2 = df2[col]

        if compare_dtypes and s1.dtype != s2.dtype:
            diffs.append(
                {
                    "where": name,
                    "kind": "dtype_mismatch",
                    "row": None,
                    "column": col,
                    "value_1": str(s1.dtype),
                    "value_2": str(s2.dtype),
                }
            )

        if _is_numeric_series(s1) and _is_numeric_series(s2):
            a1 = pd.to_numeric(s1, errors="coerce")
            a2 = pd.to_numeric(s2, errors="coerce")

            both_nan = a1.isna() & a2.isna()
            close = np.isclose(a1.fillna(0.0), a2.fillna(0.0), rtol=rtol, atol=atol)
            equal_mask = both_nan | close

            bad_rows = equal_mask.index[~equal_mask]
            for row in bad_rows:
                diffs.append(
                    {
                        "where": name,
                        "kind": "value_mismatch",
                        "row": row,
                        "column": col,
                        "value_1": a1.loc[row],
                        "value_2": a2.loc[row],
                    }
                )
        else:
            v1 = s1.astype("object")
            v2 = s2.astype("object")

            equal_mask = (v1 == v2) | (pd.isna(v1) & pd.isna(v2))
            bad_rows = equal_mask.index[~equal_mask.fillna(False)]

            for row in bad_rows:
                diffs.append(
                    {
                        "where": name,
                        "kind": "value_mismatch",
                        "row": row,
                        "column": col,
                        "value_1": v1.loc[row],
                        "value_2": v2.loc[row],
                    }
                )

    return diffs


def compare_time_dependent_panel(
    panel1,
    panel2,
    comp_name,
    rtol=1e-9,
    atol=1e-12,
    compare_dtypes=False,
):
    """
    Compare a PyPSA *_t container (e.g. net.generators_t, net.links_t).
    """
    diffs = []

    attrs1 = set(panel1.keys())
    attrs2 = set(panel2.keys())
    all_attrs = sorted(attrs1 | attrs2)

    for attr in all_attrs:
        if attr not in attrs1:
            diffs.append(
                {
                    "where": f"{comp_name}_t",
                    "kind": "missing_attribute",
                    "row": None,
                    "column": attr,
                    "value_1": None,
                    "value_2": "present_only_in_net2",
                }
            )
            continue

        if attr not in attrs2:
            diffs.append(
                {
                    "where": f"{comp_name}_t",
                    "kind": "missing_attribute",
                    "row": None,
                    "column": attr,
                    "value_1": "present_only_in_net1",
                    "value_2": None,
                }
            )
            continue

        df1 = panel1[attr]
        df2 = panel2[attr]

        diffs.extend(
            compare_dataframes(
                df1,
                df2,
                name=f"{comp_name}_t.{attr}",
                rtol=rtol,
                atol=atol,
                compare_dtypes=compare_dtypes,
            )
        )

    return diffs


def compare_networks(
    net1,
    net2,
    static_components=None,
    compare_dynamic=True,
    rtol=1e-9,
    atol=1e-12,
    compare_dtypes=False,
    max_diffs_per_block=None,
):
    """
    Robust comparison between two PyPSA networks.

    Parameters
    ----------
    net1, net2 : pypsa.Network
        Networks to compare.
    static_components : list[str] | None
        Components to compare statically. If None, a default broad set is used.
    compare_dynamic : bool
        Whether to compare *_t tables.
    rtol, atol : float
        Numeric tolerances.
    compare_dtypes : bool
        Whether to also report dtype differences.
    max_diffs_per_block : int | None
        If given, truncates reported differences for each block.

    Returns
    -------
    pandas.DataFrame
        Table of differences.
    """
    if static_components is None:
        static_components = [
            "buses",
            "carriers",
            "loads",
            "generators",
            "lines",
            "links",
            "transformers",
            "shunt_impedances",
            "storage_units",
            "stores",
            "global_constraints",
        ]

    all_diffs = []

    # Compare snapshots explicitly
    s1 = pd.DataFrame(index=_normalize_index(net1.snapshots))
    s2 = pd.DataFrame(index=_normalize_index(net2.snapshots))
    all_diffs.extend(
        compare_dataframes(
            s1,
            s2,
            name="snapshots",
            rtol=rtol,
            atol=atol,
            compare_dtypes=compare_dtypes,
        )
    )

    # Compare snapshot weightings
    all_diffs.extend(
        compare_dataframes(
            net1.snapshot_weightings,
            net2.snapshot_weightings,
            name="snapshot_weightings",
            rtol=rtol,
            atol=atol,
            compare_dtypes=compare_dtypes,
        )
    )

    # Compare static components
    for comp in static_components:
        if not hasattr(net1, comp) or not hasattr(net2, comp):
            all_diffs.append(
                {
                    "where": comp,
                    "kind": "missing_component_table",
                    "row": None,
                    "column": None,
                    "value_1": hasattr(net1, comp),
                    "value_2": hasattr(net2, comp),
                }
            )
            continue

        df1 = getattr(net1, comp)
        df2 = getattr(net2, comp)

        all_diffs.extend(
            compare_dataframes(
                df1,
                df2,
                name=comp,
                rtol=rtol,
                atol=atol,
                compare_dtypes=compare_dtypes,
            )
        )

    # Compare dynamic components
    if compare_dynamic:
        for comp in static_components:
            panel_name = f"{comp}_t"
            if hasattr(net1, panel_name) and hasattr(net2, panel_name):
                panel1 = getattr(net1, panel_name)
                panel2 = getattr(net2, panel_name)
                all_diffs.extend(
                    compare_time_dependent_panel(
                        panel1,
                        panel2,
                        comp_name=comp,
                        rtol=rtol,
                        atol=atol,
                        compare_dtypes=compare_dtypes,
                    )
                )

    df = pd.DataFrame(
        all_diffs,
        columns=["where", "kind", "row", "column", "value_1", "value_2"],
    )

    if max_diffs_per_block is not None and not df.empty:
        df = (
            df.groupby("where", group_keys=False)
            .head(max_diffs_per_block)
            .reset_index(drop=True)
        )

    return df

def check_all_storages_balance(sut, n, name):
    """
    Checks energy balance for all storage units in sut dataframe.
    inflows: dict of inflow series per storage name
    """
    storages = sut['state_of_charge'].columns
    inflows = sut['inflow']

    for s in storages:
        soc = sut['state_of_charge'][s]
        p = sut['p'][s]
        
        eta_charge = n.storage_units.efficiency_store.loc[s] if n.storage_units.efficiency_store.loc[s] != 0 else 1
        eta_discharge = n.storage_units.efficiency_dispatch.loc[s]
        
        inflow = 0
        if isinstance(inflows, pd.DataFrame) and s in inflows:
            inflow = inflows[s]
        
        if isinstance(inflow, pd.Series):
            inflow = inflow
        
        delta_soc = soc.diff().fillna(0).iloc[1:]
        expected_delta_soc = (
            eta_charge * (-p.clip(upper=0))
            - p.clip(lower=0) / eta_discharge
            + inflow
        ).iloc[1:]
        
        mismatch = delta_soc - expected_delta_soc
        
        # set mismatch very small values to zero
        mismatch = np.where(np.abs(mismatch) < 1e-1, 0, mismatch)
        
        plt.figure()
        plt.plot(mismatch, label=f"{s}")
        plt.title(f"Energy balance mismatch for {s}-{name}")
        plt.ylabel("MWh")
        plt.xlabel("Timestep")
        plt.legend()
        plt.grid()
        plt.show()

        print(f"{s} - {name} - Max mismatch: {np.abs(mismatch).max():.3f} MWh")



#%% Network definition with PyPSA

if __name__ == '__main__':
    network_name = "base_s_5_elec_lvopt_1h"
    network = pypsa.Network(f"../test/networks/{network_name}.nc")
    
    network = clean_marginal_cost(network)
    network = clean_global_constraints(network)
    network = clean_e_sum(network)
    
    network = one_bus_network(network)
    network.optimize(solver_name='gurobi')



