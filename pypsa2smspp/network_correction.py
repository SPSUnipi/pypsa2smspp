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
from typing import Union, Sequence

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
):
    """
    Reduce the number of snapshots in a PyPSA network and optionally scale capital costs.

    Parameters
    ----------
    n : pypsa.Network
        The network to modify.
    target : int | Sequence | pd.Index
        If int:
            - keep that many snapshots (first ones, by default);
            - if `evenly_spaced=True`, pick them evenly spaced across horizon.
        If a sequence/index of snapshot labels, keep exactly those.
    drop_renewables : bool, optional
        If True, drop generators whose 'carrier' is in `renewable_carriers`.
    renewable_carriers : Sequence[str], optional
        Carriers to drop if `drop_renewables` is True.
    adjust_snapshot_weightings : bool, optional
        If True, scale n.snapshot_weightings['objective'] by (old_len / new_len).
    evenly_spaced : bool, optional
        If True and `target` is int, select snapshots evenly spaced instead of the first ones.
    scale_capital_costs : bool, optional
        If True, scale component capital_cost by (new_len / old_len).
    inplace : bool, optional
        If False (default), operate on a copy and return it. If True, modify `n` in place.

    Returns
    -------
    pypsa.Network
        The modified network (or the same object if `inplace=True`).
    """

    net = n if inplace else n.copy()

    old_snapshots = net.snapshots.copy()
    old_len = len(old_snapshots)
    if old_len == 0:
        raise ValueError("Network has no snapshots to reduce.")

    if isinstance(target, int):
        if target <= 0:
            raise ValueError("target (int) must be > 0.")
        if target > old_len:
            raise ValueError(f"target ({target}) cannot exceed original snapshots ({old_len}).")

        if evenly_spaced:
            # Equally spaced selection
            idx = np.linspace(0, old_len - 1, target, dtype=int)
            idx = np.unique(idx)
            new_snapshots = old_snapshots[idx]
        else:
            # Just take the first N snapshots
            new_snapshots = old_snapshots[:target]

    else:
        # Sequence of labels
        new_snapshots = pd.Index(target)
        missing = new_snapshots.difference(old_snapshots)
        if len(missing) > 0:
            raise ValueError(
                f"Some requested snapshots are not in the network: {missing[:5].tolist()} ..."
            )

    new_len = len(new_snapshots)
    if new_len == 0:
        raise ValueError("Resulting snapshot set is empty.")

    # Apply new snapshots
    net.snapshots = new_snapshots

    # Slice all *_t DataFrames
    for attr in dir(net):
        if attr.endswith("_t"):
            df_dict = getattr(net, attr)
            for key, df in list(df_dict.items()):
                if hasattr(df, "loc"):
                    df_dict[key] = df.loc[new_snapshots]

    # Optionally drop renewable generators
    if drop_renewables and hasattr(net, "generators"):
        to_drop = net.generators.loc[
            net.generators["carrier"].isin(renewable_carriers)
        ].index
        if len(to_drop) > 0:
            net.generators.drop(index=to_drop, inplace=True)
            if hasattr(net, "generators_t"):
                for key, df in list(net.generators_t.items()):
                    if hasattr(df, "drop"):
                        df.drop(columns=to_drop, inplace=True, errors="ignore")

    # Optionally scale capital_costs
    if scale_capital_costs:
        scale_capex = new_len / old_len
        capex_tables = ["generators", "links", "stores", "storage_units", "lines", "transformers"]
        for tab in capex_tables:
            if hasattr(net, tab):
                df = getattr(net, tab)
                if isinstance(df, pd.DataFrame) and "capital_cost" in df.columns:
                    df["capital_cost"] = df["capital_cost"] * scale_capex

    # Optionally adjust snapshot_weightings
    if adjust_snapshot_weightings and hasattr(net, "snapshot_weightings"):
        sw = net.snapshot_weightings
        if "objective" in sw.columns:
            sw.loc[new_snapshots, "objective"] = (
                sw.loc[new_snapshots, "objective"] * (old_len / new_len)
            )

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
            p_nom=5 * max_total_demand if max_total_demand > 0 else -min_total_demand,
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



