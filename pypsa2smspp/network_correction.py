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
    
def clean_global_constraints(n):
    n.global_constraints = pd.DataFrame(columns=n.global_constraints.columns)
    n.global_constraints_t = dict()
    n.links.p_nom_extendable = False
    return n
    
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

def clean_stores(n, carriers=None):
    """
    Remove Stores (optionally filtered by carrier) and all incident charge/discharge
    Links from a PyPSA network. Also drops the dedicated store buses.

    Parameters
    ----------
    n : pypsa.Network
        The network to clean.
    carriers : list[str] or None
        If provided, only stores with carrier in this list are removed
        (e.g., ["battery"], ["H2"]). If None, remove all stores.

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
        return n  # nothing to do

    # --- Buses dedicated to those stores
    store_buses = n.stores.loc[store_idx, "bus"].unique()

    # --- Remove links incident to any store bus (charge/discharge, electrolysis/FC, etc.)
    mask_links = n.links["bus0"].isin(store_buses) | n.links["bus1"].isin(store_buses)
    link_idx = n.links.index[mask_links]

    if len(link_idx):
        n.links.drop(link_idx, inplace=True)
        # Drop link time-series columns safely if present
        for key, df in n.links_t.items():
            cols = df.columns.intersection(link_idx)
            if len(cols):
                df.drop(columns=cols, inplace=True)

    # --- Remove stores and their time-series
    n.stores.drop(store_idx, inplace=True)
    for key, df in n.stores_t.items():
        cols = df.columns.intersection(store_idx)
        if len(cols):
            df.drop(columns=cols, inplace=True)

    # --- Remove the now-unneeded store buses
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
    renewable_carriers: Sequence[str] = ('solar', 'solar-hsat', 'onwind', 'offwind-ac', 'offwind-dc', 'offwind-float', 'PV', 'wind', 'ror'),
    adjust_snapshot_weightings: bool = False,
    evenly_spaced: bool = False,
    inplace: bool = False,
):
    """
    Reduce the number of snapshots in a PyPSA network and scale capital_costs.

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
            raise ValueError(f"Some requested snapshots are not in the network: {missing[:5].tolist()} ...")

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
        to_drop = net.generators.loc[net.generators["carrier"].isin(renewable_carriers)].index
        if len(to_drop) > 0:
            net.generators.drop(index=to_drop, inplace=True)
            if hasattr(net, "generators_t"):
                for key, df in list(net.generators_t.items()):
                    if hasattr(df, "drop"):
                        df.drop(columns=to_drop, inplace=True, errors="ignore")

    # Scale capital_costs
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
            sw.loc[new_snapshots, "objective"] = sw.loc[new_snapshots, "objective"] * (old_len / new_len)

    return net



def add_slack_unit(n, exclude_suffixes=("H2", "battery")):
    """
    Adds a high-cost slack generator to each bus in the network,
    excluding buses whose names end with specific suffixes (e.g., H2, battery).

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to which slack units will be added.
    exclude_suffixes : tuple of str, optional
        Bus-name suffixes (case-insensitive) to exclude. Default: ("H2", "battery")
    """
    # Compute the maximum total demand over all time steps
    max_total_demand = n.loads_t.p_set.sum(axis=1).max()
    min_total_demand = n.loads_t.p_set.sum(axis=1).min()
    
    # Helper to decide if a bus should be excluded
    def _is_excluded(bus_name: str) -> bool:
        bn = str(bus_name).strip().lower()
        return any(bn.endswith(sfx.lower()) for sfx in exclude_suffixes)

    # Iterate over eligible buses only
    for bus in n.buses.index:
        if _is_excluded(bus):
            continue

        n.add(
            "Generator",
            name=f"slack_unit {bus}",
            carrier="slack",
            bus=bus,
            p_nom=max_total_demand if max_total_demand > 0 else -min_total_demand,
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


def compare_static_components(comp1, comp2, comp_name):
    differences = []
    common_indices = comp1.index.intersection(comp2.index)
    all_indices = comp1.index.union(comp2.index)
    
    for idx in all_indices:
        if idx not in common_indices:
            differences.append((comp_name, idx, "Missing in one network", None, None))
            continue
        
        row1 = comp1.loc[idx]
        row2 = comp2.loc[idx]
        for col in comp1.columns.union(comp2.columns):
            val1 = row1[col] if col in row1 else np.nan
            val2 = row2[col] if col in row2 else np.nan

            if pd.isna(val1) and pd.isna(val2):
                continue  # Both NaN, consider equal
            try:
                if isinstance(val1, (int, float, np.number)) and isinstance(val2, (int, float, np.number)):
                    if not np.isclose(val1, val2, equal_nan=True):
                        differences.append((comp_name, idx, col, val1, val2))
                else:
                    if val1 != val2:
                        differences.append((comp_name, idx, col, val1, val2))
            except Exception:
                if val1 != val2:
                    differences.append((comp_name, idx, col, val1, val2))
    return differences


def compare_dynamic_components(comp1_t, comp2_t, comp_name):
    differences = []
    for attr in comp1_t.keys() | comp2_t.keys():
        if attr not in comp1_t or attr not in comp2_t:
            differences.append((f"{comp_name}_t", attr, "Missing attribute", None, None))
            continue
        df1 = comp1_t[attr]
        df2 = comp2_t[attr]
        all_columns = df1.columns.union(df2.columns)
        for col in all_columns:
            if col not in df1 or col not in df2:
                differences.append((f"{comp_name}_t", attr, col, "Missing", "Missing"))
                continue
            vals1 = df1[col]
            vals2 = df2[col]
            if not vals1.equals(vals2):
                differences.append((f"{comp_name}_t", attr, col, vals1.values[:5], vals2.values[:5]))
    return differences

def compare_networks(net1, net2, components_to_check=["loads", "generators", "lines", "storage_units"]):
    all_differences = []
    for comp in components_to_check:
        comp_df1 = getattr(net1, comp)
        comp_df2 = getattr(net2, comp)
        static_diff = compare_static_components(comp_df1, comp_df2, comp)
        all_differences.extend(static_diff)
        
        comp_t_df1 = getattr(net1, f"{comp}_t")
        comp_t_df2 = getattr(net2, f"{comp}_t")
        dynamic_diff = compare_dynamic_components(comp_t_df1, comp_t_df2, comp)
        all_differences.extend(dynamic_diff)
    
    df_diff = pd.DataFrame(all_differences, columns=["Component", "Element", "Attribute", "Network1", "Network2"])
    return df_diff

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



