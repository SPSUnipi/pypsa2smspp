# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 15:52:57 2025

@author: aless
"""

"""
utils.py

This module contains utility functions used throughout the Transformation 
process. These are stateless helper functions that operate on standard
data structures such as DataFrames, Series, or Networks, and can be reused 
across multiple components.

They are typically imported and used within the Transformation class.
"""

import numpy as np
import pandas as pd
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
import re
from pypsa2smspp import logger

#%%
################################################################################################
########################## Utilities for PyPSA network values ##################################
################################################################################################

def get_param_as_dense(n, component, field, weights=True):
    """
    Get the parameters of a component as a dense DataFrame.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network.
    component : str
        The component to get the parameters from (e.g., 'Generator').
    field : str
        The field/attribute to extract.
    weights : bool, default=True
        Whether to weight time-dependent values by snapshot weights.

    Returns
    -------
    pd.DataFrame
        A dense DataFrame of parameter values across snapshots.
    """
    sns = n.snapshots

    if not n.investment_period_weightings.empty:
        periods = sns.unique("period")
        period_weighting = n.investment_period_weightings.objective[periods]
    weighting = n.snapshot_weightings.objective
    if not n.investment_period_weightings.empty:
        weighting = weighting.mul(period_weighting, level=0).loc[sns]
    else:
        weighting = weighting.loc[sns]

    if field in n.static(component).columns:
        field_val = get_as_dense(n, component, field, sns)
    else:
        field_val = n.dynamic(component)[field]

    if weights:
        field_val = field_val.mul(weighting, axis=0)
    return field_val

         
def remove_zero_p_nom_opt_components(n, nominal_attrs):
    # Lista dei componenti che hanno l'attributo p_nom_opt
    components_with_p_nom_opt = ["Generator", "Link", "Store", "StorageUnit", "Line", "Transformer"]
    
    for components in n.iterate_components(["Line", "Generator", "Link", "Store", "StorageUnit"]):
        components_df = components.df
        components_df = components_df[components_df[f"{nominal_attrs[components.name]}_opt"] > 0]
        setattr(n, components.list_name, components_df)


def is_extendable(component_df, component_type, nominal_attrs):
    """
    Returns the boolean Series indicating which components are extendable.

    Parameters
    ----------
    component_df : pd.DataFrame
        The component DataFrame (e.g., n.generators).
    component_type : str
        The PyPSA component type (e.g., "Generator").
    nominal_attrs : dict
        Dictionary mapping component types to nominal attribute names.

    Returns
    -------
    pd.Series
        Boolean Series where True indicates an extendable component.
    """
    attr = nominal_attrs.get(component_type)
    extendable_attr = f"{attr}_extendable"
    return component_df[extendable_attr].values


def filter_extendable_components(components_df, component_type, nominal_attrs):
    """
    Filters a component DataFrame to retain only extendable components.

    Parameters
    ----------
    components_df : pd.DataFrame
        DataFrame of a PyPSA component (e.g., n.generators).
    component_type : str
        Component type (capitalized singular, e.g., "Generator").
    nominal_attrs : dict
        Mapping from component types to their nominal attributes.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only extendable entries.
    """
    attr = nominal_attrs.get(component_type)
    if not attr:
        return components_df

    extendable_attr = f"{attr}_extendable"
    if extendable_attr in components_df.columns:
        return components_df[components_df[extendable_attr]]
    return components_df


def get_bus_idx(n, components_df, bus_series, column_name, dtype="uint32"):
    """
    Maps one or multiple bus series to their integer indices in n.buses and
    stores them as new columns in the components_df.

    Parameters
    ----------
    n : pypsa.Network
        The network.
    components_df : pd.DataFrame
        DataFrame of the component to update.
    bus_series : pd.Series or list of pd.Series
        Series (or list of Series) of bus names (e.g., generators.bus, lines.bus0).
    column_name : str or list of str
        Name(s) of the new column(s) to store numeric indices.
    dtype : str, optional
        Data type of the index column(s) (default: "uint32").

    Returns
    -------
    None
    """
    if isinstance(bus_series, list):
        for series, col in zip(bus_series, column_name):
            components_df[col] = series.map(n.buses.index.get_loc).astype(dtype).values
    else:
        components_df[column_name] = bus_series.map(n.buses.index.get_loc).astype(dtype).values



def get_nominal_aliases(component_type, nominal_attrs):
    """
    Creates aliases for nominal attributes used in the investment block.

    Parameters
    ----------
    component_type : str
        PyPSA component type (e.g., 'Generator').
    nominal_attrs : dict
        Dictionary of nominal attributes.

    Returns
    -------
    dict
        Aliases for the nominal attribute, min, and max.
    """
    base = nominal_attrs[component_type]
    return {
        base: "p_nom",
        base + "_min": "p_nom_min",
        base + "_max": "p_nom_max",
    }

#%%
#################################################################################################
############################### Dimensions for SMS++ ############################################
#################################################################################################

def ucblock_dimensions(n):
    """
    Computes the dimensions of the UCBlock from the PyPSA network.
    """
    if len(n.snapshots) == 0:
        raise ValueError("No snapshots defined in the network.")

    components = {
        "NumberUnits": ["generators", "storage_units", "stores"],
        "NumberElectricalGenerators": ["generators", "storage_units", "stores"],
        "NumberNodes": ["buses"],
        "NumberLines": ["lines", "links"],
    }

    dimensions = {
        "TimeHorizon": len(n.snapshots),
        **{
            name: sum(len(getattr(n, comp)) for comp in comps)
            for name, comps in components.items()
        }
    }
    return dimensions


def networkblock_dimensions(n):
    """
    Computes NetworkBlock dimensions from a PyPSA network `n`.
    Returns a dict with:
      - Lines, Links, combined (physical objects)
      - NumberLines, NumberBranches
      - HyperMode (True iff NumberBranches > NumberLines)
    Notes:
      - Multi-link detection is based on bus2, bus3, ... columns
        that are present AND have non-empty values.
    """
    # --- physical counts (each physical link counts 1 even if multi-output) ---
    lines_count = len(getattr(n, "lines", []))
    links_count = len(getattr(n, "links", []))
    combined_count = lines_count + links_count

    # --- detect extra outputs from multi-links to build branches ---
    extra_outputs = 0
    if links_count > 0:
        link_df = n.links
        # iterate bus2, bus3, ... only while column exists
        k = 2
        while f"bus{k}" in link_df.columns:
            s = link_df[f"bus{k}"]
            # count non-empty entries: notna and not just whitespace
            valid = s.notna() & (s.astype(str).str.strip() != "")
            extra_outputs += int(valid.sum())
            k += 1

    # For branches: each physical line contributes 1 branch.
    # Each physical link contributes 1 branch for bus1 (the first output),
    # plus one branch for every additional non-empty bus{k>=2}.
    number_lines = combined_count
    number_branches = lines_count + links_count + extra_outputs

    return {
        "Lines": lines_count,
        "Links": links_count,
        "combined": combined_count,
        "NumberLines": number_lines,
        "NumberBranches": number_branches,
    }



def investmentblock_dimensions(n, nominal_attrs):
    """
    Computes the dimensions of the InvestmentBlock from the PyPSA network.
    """
    investment_components = ['generators', 'storage_units', 'stores', 'lines', 'links']
    num_assets = 0
    for comp in investment_components:
        df = getattr(n, comp)
        comp_type = comp[:-1].capitalize() if comp != "storage_units" else "StorageUnit"
        attr = nominal_attrs.get(comp_type)
        if attr and f"{attr}_extendable" in df.columns:
            num_assets += df[f"{attr}_extendable"].sum()

    return {"NumAssets": int(num_assets)}


def hydroblock_dimensions():
    """
    Computes the static dimensions for a HydroUnitBlock (assuming one reservoir).
    """
    dimensions = dict()
    dimensions["NumberReservoirs"] = 1
    dimensions["NumberArcs"] = 2 * dimensions["NumberReservoirs"]
    dimensions["TotalNumberPieces"] = 2
    return dimensions

# -------------------------------- Correction --------------------------------------

def correct_dimensions(dimensions, stores_df, links_merged_df, n):
    dimensions['NetworkBlock']['Links'] -= len(stores_df)
    dimensions['NetworkBlock']['combined'] -= len(stores_df)
    dimensions['UCBlock']['NumberLines'] -= len(stores_df)
    dimensions['InvestmentBlock']['NumAssets'] -= len(stores_df[stores_df['e_nom_extendable'] == True])
    
    if "NumberBranches" in dimensions['NetworkBlock']:
        # To understand if all of these are needed
        dimensions['NetworkBlock']['NumberBranches'] -= len(stores_df)
        # dimensions['NetworkBlock']['NumberLines'] = dimensions['NetworkBlock']['combined']



#%%
###############################################################################################
############################### Direct transformation #########################################
###############################################################################################

def get_attr_name(component_type: str, carrier: str | None = None, renewable_carriers: list[str] = []) -> str:
    """
    Maps a PyPSA component type and its carrier to the corresponding
    UnitBlock attribute name to be used in the Transformation.

    Parameters
    ----------
    component_type : str
        The PyPSA component type (e.g., 'Generator', 'Store', 'StorageUnit', 'Line', 'Link')
    carrier : str or None
        The carrier name if available (e.g., 'solar', 'hydro', 'slack')

    Returns
    -------
    str
        The attribute name for the Transformation block parameters.
    """

    # normalize for case-insensitive match
    if carrier:
        carrier = carrier.lower()

    # Generators
    if component_type == "Generator":
        if carrier in renewable_carriers:
            return "IntermittentUnitBlock_parameters"
        elif carrier == "slack":
            return "SlackUnitBlock_parameters"
        else:
            return "IntermittentUnitBlock_parameters"
            # return "ThermalUnitBlock_parameters"

    # StorageUnit
    if component_type == "StorageUnit":
        if carrier in ["hydro", "phs"]:
            return "HydroUnitBlock_parameters"
        else:
            return "BatteryUnitBlock_parameters"

    # Store
    if component_type == "Store":
        return "BatteryUnitBlock_store_parameters"

    # Lines
    if component_type == "Line":
        return "Lines_parameters"

    # Links
    if component_type == "Link":
        return "Links_parameters"

    raise ValueError(f"Component type {component_type} with carrier {carrier} not recognized.")

# ------------------------ Pre-processing functions --------------------------------

def build_store_and_merged_links(n, merge_links=False, logger=print):
    """
    Build enriched stores_df (adds efficiency_store/efficiency_dispatch)
    and links_merged_df (replaces per-store charge/discharge link pair with
    a single merged link with eta=1 and summed capital_cost). Keeps a mapping
    for perfect inverse transformation.

    Returns
    -------
    stores_df : pd.DataFrame
        Copy of n.stores with extra columns:
        - efficiency_store (eta_ch)
        - efficiency_dispatch (eta_dis)

    links_merged_df : pd.DataFrame
        Copy of n.links where the store-related charge/discharge rows are
        replaced by a single merged link per store. The merged link has:
        - efficiency = 1.0
        - marginal_cost = 0.0
        - capital_cost = capex_ch + capex_dis
        - p_nom = chosen from originals (see note below)
        - p_nom_extendable = common value (assert both equal)
        - name includes both original names for traceability

    store_link_map : dict
        Mapping with all info needed for inverse reconstruction:
        {store_name: {
            'bus_elec': str,
            'bus_store': str,
            'name_link_ch': str or None,
            'name_link_dis': str or None,
            'eta_ch': float,
            'eta_dis': float,
            'p_nom_ch': float, 'p_nom_dis': float,
            'p_nom_extendable': bool,
            'capex_ch': float, 'capex_dis': float,
            'merged_name': str
        }}

    Notes
    -----
    - Charge link is detected as (bus0 == bus_elec) & (bus1 == bus_store).
      Discharge link as (bus0 == bus_store) & (bus1 == bus_elec).
    - If only one of the two links exists, we still merge using the available
      one and set missing values to defaults; a warning is emitted.
    - On p_nom: PyPSA-Eur ties charger/discharger sizes in practice. We:
        * assert ~equal within tolerance, otherwise pick min and warn.
      This avoids over-stating capability if data are slightly inconsistent.
    """

    stores_df = n.stores.copy()
    links_merged_df = n.links.copy()

    # Add the two new columns with safe defaults
    for col in ["efficiency_store", "efficiency_dispatch"]: # Serve questo passaggio?
        if col not in stores_df.columns:
            stores_df[col] = 1.0


    if not merge_links or links_merged_df.empty or stores_df.empty:
        return stores_df, links_merged_df

    # We will collect rows to drop and rows to append
    rows_to_drop = []
    rows_to_append = []

    # Tolerance for p_nom equality check
    PNOM_TOL = 1e-6

    # Build a fast index by bus to find links connected to a given store bus
    # We'll just filter per store; clarity over micro-optimization.
    for store_name, srow in stores_df.iterrows():
        bus_store = srow["bus"]
        # Heuristic: the links are the ones connected to the bus
        mask_ch = (links_merged_df["bus0"] == bus_store) | (links_merged_df["bus1"] == bus_store)
        cand = links_merged_df[mask_ch]
        if cand.empty:
            # No links connected to this store -> nothing to merge
            continue
        
        charge_row = cand[cand['bus1'] == bus_store].iloc[0]
        discharge_row = cand[cand['bus0'] == bus_store].iloc[0]
        bus_elec = charge_row['bus0'] if charge_row['bus0'] == discharge_row['bus1'] else None

        if bus_elec is None:
            # Could not determine the paired electrical bus; skip merge
            continue

        # Extract params with defaults
        # Charge (elec -> store
        eta_ch = charge_row.efficiency
        p_nom_ch = charge_row.p_nom
        capex_ch = charge_row.capital_cost
        ext_ch = charge_row.p_nom_extendable
        name_ch = charge_row.name


        # Discharge (store -> elec)
        eta_dis = discharge_row.efficiency
        p_nom_dis = discharge_row.p_nom
        capex_dis = discharge_row.capital_cost
        ext_dis = discharge_row.p_nom_extendable
        name_dis = discharge_row.name


        # Extendability must match (as per your assumption)
        if ext_ch != ext_dis:
            logger(f"[merge] Warning: extendability mismatch for store '{store_name}' "
                   f"(charge={ext_ch}, discharge={ext_dis}). Using logical AND.")
        pnom_extendable = bool(ext_ch and ext_dis)

        # Choose p_nom for the merged link
        # In PyPSA-Eur they should be equal; we assert/clip to min to be safe.
        if abs(p_nom_ch - p_nom_dis) > PNOM_TOL:
            logger(f"[merge] Warning: p_nom mismatch for store '{store_name}' "
                   f"(ch={p_nom_ch}, dis={p_nom_dis}). Using min().")
        p_nom_merged = float(min(p_nom_ch, p_nom_dis))

        # Capital cost is the SUM (two converters of same size)
        capex_merged = float(capex_ch + capex_dis)

        # Update store efficiencies
        stores_df.at[store_name, "efficiency_store"] = float(eta_ch)
        stores_df.at[store_name, "efficiency_dispatch"] = float(eta_dis)

        # Prepare merged link row:
        # We clone one of the originals to inherit optional columns, then override.
        new_row = charge_row if charge_row is not None else discharge_row

        merged_name = f"{name_ch or 'NA'}__{name_dis or 'NA'}"

        # Override key fields
        new_row.name = merged_name
        new_row["bus0"] = bus_elec
        new_row["bus1"] = bus_store
        new_row["efficiency"] = 1.0
        new_row["marginal_cost"] = 0.0
        new_row["capital_cost"] = capex_merged
        new_row["p_nom"] = p_nom_merged
        new_row["p_nom_extendable"] = pnom_extendable
        new_row["p_min_pu"] = -eta_dis # Correction to account limit of perspective
        # If you don't add this, the store is gonna produce too much (to have -1, they can discharge 1/eta_dis)
        # To correct if there are problems with sector coupling

        # If there are p_nom_min/max columns, keep them consistent (safe defaults)
        for col in ["p_nom_min", "p_nom_max"]:
            if col in new_row.index and pd.isna(new_row[col]):
                # set permissive bounds
                new_row[col] = 0.0 if col.endswith("_min") else np.inf

        # Mark rows to drop (original charge/discharge)
        if name_ch is not None:
            rows_to_drop.append(name_ch)
        if name_dis is not None:
            rows_to_drop.append(name_dis)

        rows_to_append.append(new_row)


    # Apply drops/appends
    if rows_to_drop:
        links_merged_df = links_merged_df.drop(index=[r for r in rows_to_drop if r in links_merged_df.index])
    if rows_to_append:
        links_merged_df = pd.concat([links_merged_df, pd.DataFrame(rows_to_append)], axis=0)

    return stores_df, links_merged_df



def explode_multilinks_into_branches(links_merged_df: pd.DataFrame, hyper_id, logger=print):
    """
    Split multi-output links in `links_merged_df` into one row per output branch.
    Does NOT touch `n.links`. Returns a new DataFrame where:
      - Each physical link row produces N rows (one per non-empty output) if it is a multilink;
        otherwise produces exactly 1 row identical in name to the original.
      - Column 'hyper' identifies the original physical link for all its branches.
      - 'name' is suffixed as f"{original}__to_{busX}" ONLY for true multilinks.
      - bus2/bus3/... and efficiency2/efficiency3/... columns are DROPPED.
    NOTE: we keep the function signature; if you need the next hyper_id, you can return it too.
    """
    if links_merged_df.empty:
        return links_merged_df.copy()

    df = links_merged_df.copy()

    # Identify extra bus/eff columns dynamically
    bus_extra_cols = []
    k = 2
    while f"bus{k}" in df.columns:
        bus_extra_cols.append(f"bus{k}")
        k += 1

    # If no bus2+ columns exist at all, nothing to explode; just attach hyper & is_primary_branch
    if not bus_extra_cols:
        out = df.copy()
        out["hyper"] = np.arange(hyper_id, hyper_id + len(out), dtype=int)
        out["is_primary_branch"] = True
        return out

    def _non_empty(val) -> bool:
        return pd.notna(val) and str(val).strip() != ""

    new_rows = []

    for link_name, row in df.iterrows():
        # Gather valid extra outputs for THIS row
        extra_outputs = []
        for idx, bcol in enumerate(bus_extra_cols, start=2):
            bval = row.get(bcol, np.nan)
            if _non_empty(bval):
                ecol = f"efficiency{idx}"
                if ecol not in df.columns or pd.isna(row.get(ecol, np.nan)):
                    raise ValueError(f"Multi-link '{link_name}' has '{bcol}={bval}' but missing '{ecol}'.")
                extra_outputs.append((bcol, ecol))

        is_multilink = len(extra_outputs) > 0

        if not is_multilink:
            # Single-output link: keep as-is (no renaming)
            out_row = row.copy()
            out_row["hyper"] = hyper_id
            out_row["is_primary_branch"] = True
            new_rows.append(out_row)
            hyper_id += 1
            continue

        # True multilink: create one row per output (bus1 + extras) and rename
        # primary branch = bus1
        primary_bus = row["bus1"]
        primary_eff = row["efficiency"]

        # primary
        pr = row.copy()
        pr["bus1"] = primary_bus
        pr["efficiency"] = float(primary_eff)
        pr.name = f"{link_name}__to_{primary_bus}"
        pr["hyper"] = hyper_id
        pr["is_primary_branch"] = True
        new_rows.append(pr)

        # extras
        for bcol, ecol in extra_outputs:
            child = row.copy()
            child["bus1"] = row[bcol]
            child["efficiency"] = float(row[ecol])
            child.name = f"{link_name}__to_{child['bus1']}"
            child["hyper"] = hyper_id
            child["is_primary_branch"] = False
            new_rows.append(child)

        hyper_id += 1

    exploded = pd.DataFrame(new_rows)

    # Drop the extra bus/eff columns to avoid ambiguity downstream
    cols_to_drop = [c for c in exploded.columns
                    if (c.startswith("bus") and c not in ("bus0", "bus1"))
                    or (c.startswith("efficiency") and c != "efficiency")]
    exploded = exploded.drop(columns=cols_to_drop, errors="ignore")

    # Log
    if callable(logger):
        n_phys = len(df)
        number_branches = len(exploded)
        extra = number_branches - n_phys
        logger(f"[multilink] Exploded {n_phys} physical links into {number_branches} branches (+{extra}).")

    return exploded
    # If you need the next hyper_id, you can instead: `return exploded, hyper_id`




# Translate into generic once the ucblock\investmentblock general use is defined  
def add_hyperarcid_to_parameters(Lines_parameters, Links_parameters):
    """
    Add a HyperArcID entry to Lines_parameters and Links_parameters.
    For now it uses a dummy lambda that simply returns the Hyper column if present.
    You can later customize the logic.
    """

    # Default lambda: looks for a Series/array called 'Hyper' and returns its values
    hyper_def = lambda hyper: hyper.values
    
    # For lines
    if "HyperArcID" not in Lines_parameters:
        Lines_parameters["HyperArcID"] = hyper_def

    # For links
    if "HyperArcID" not in Links_parameters:
        Links_parameters["HyperArcID"] = hyper_def
        
    Links_parameters.update({
    "MaxPowerFlow": lambda p_nom, p_max_pu, p_nom_extendable, is_primary_branch:
        (p_nom[is_primary_branch] * p_max_pu[is_primary_branch]).where(
            ~p_nom_extendable[is_primary_branch], p_max_pu[is_primary_branch]
        ).values,
    "MinPowerFlow": lambda p_nom, p_min_pu, p_nom_extendable, is_primary_branch:
        (p_nom[is_primary_branch] * p_min_pu[is_primary_branch]).where(
            ~p_nom_extendable[is_primary_branch], p_min_pu[is_primary_branch]
        ).values,
    "LineSusceptance": lambda p_nom, is_primary_branch:
        np.zeros_like(p_nom[is_primary_branch].values)})

# ------------------------------------------

def process_dcnetworkblock(
    components_df,
    components_name,
    investment_meta,
    unitblock_index,
    lines_index,
    df_investment,
    nominal_attrs,
):
    """
    Updates investment_meta for lines or links after adding the unit block.

    Parameters
    ----------
    components_df : pd.DataFrame
        DataFrame of the components (lines or links).
    components_name : str
        Component name, e.g., 'Line' or 'Link'.
    investment_meta : dict
        Shared investment metadata dictionary to update.
    unitblock_index : int
        Current block index.
    df_investment : pd.DataFrame
        The investment dataframe for the component.
    renewable_carriers : list
        Renewable carriers list.
    nominal_attrs : dict
        Nominal attributes dictionary.

    Returns
    -------
    next_index : int
        Updated block index after processing.
    """

    extendable_mask = is_extendable(components_df, components_name, nominal_attrs)

    for idx in components_df[extendable_mask].index:
        investment_meta["Blocks"].append(f"DCNetworkBlock_{unitblock_index}")
        investment_meta["index_extendable"].append(lines_index)  
        lines_index += 1
        unitblock_index += 1
    

    investment_meta["asset_type"].extend([1] * len(df_investment))
    

    return unitblock_index, lines_index


def parse_unitblock_parameters(
    attr_name,
    unitblock_parameters,
    smspp_parameters,
    dimensions,
    conversion_dict,
    components_df,
    components_t,
    n,
    components_type,
    component
):

    """
    Parse the parameters for a unit block.

    Parameters
    ----------
    attr_name : str
        The attribute name of the block (e.g. ThermalUnitBlock_parameters)
    unitblock_parameters : dict
        Dictionary of functions or values for each variable.
    smspp_parameters : dict
        Excel-read parameters describing sizes and types.
    components_df : pd.DataFrame
        The static data of the component.
    components_t : pd.DataFrame
        The dynamic data (time series) of the component.
    n : pypsa.Network
        The PyPSA network object.
    components_type : str
        The component type name (e.g. "Generator").
    component : str or None
        Single component name, or None.

    Returns
    -------
    converted_dict : dict
        A dictionary with keys as variable names and values as
        dictionaries describing 'value', 'type', and 'size'
    """
    converted_dict = {}

    for key, func in unitblock_parameters.items():
        if callable(func):
            param_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            args = [
                resolve_param_value(
                    param,
                    smspp_parameters,
                    attr_name,
                    key,
                    components_df,
                    components_t,
                    n,
                    components_type,
                    component
                )
                for param in param_names
            ]

            value = func(*args)
            # force consistent type
            if isinstance(value, pd.DataFrame) and component is not None:
                value = value[[component]].values
            elif isinstance(value, pd.Series):
                value = value.tolist()
                
            variable_type, variable_size = determine_size_type(
                smspp_parameters,
                dimensions,
                conversion_dict,
                attr_name,
                key,
                value
            )

            converted_dict[key] = {
                "value": value,
                "type": variable_type,
                "size": variable_size
            }
        else:
            # fixed value
            logger.debug(f"[parse_unitblock_parameters] Using fixed value for {key}")
            variable_type, variable_size = determine_size_type(
                smspp_parameters,
                dimensions,             
                conversion_dict,
                attr_name,
                key,
                func
            )

            converted_dict[key] = {
                "value": func,
                "type": variable_type,
                "size": variable_size
            }

    return converted_dict


def resolve_param_value(
    param,
    smspp_parameters,
    attr_name,
    key,
    components_df,
    components_t,
    n,
    components_type,
    component
):
    """
    Resolves the correct parameter value to be passed to the lambda function.

    Parameters
    ----------
    param : str
        Parameter name required by the lambda
    smspp_parameters : dict
        Parameters read from excel
    attr_name : str
        UnitBlock name
    key : str
        The *variable* name in the unitblock_parameters (e.g. MaxPower)
    ...
    """

    block_class = attr_name.split("_")[0]
    size = smspp_parameters[block_class]['Size'][key]

    if size not in [1, '[L]', '[Li]', '[NA]', '[NP]', '[NR]', '[NB]', '[Li] | [NB]']:
        weight = param in [
            'capital_cost', 'marginal_cost', 'marginal_cost_quadratic',
            'start_up_cost', 'stand_by_cost'
        ]
        arg = get_param_as_dense(n, components_type, param, weight)[[component]]
    elif param in components_df.index or param in components_df.columns:
        arg = components_df.get(param)
    elif param in components_t.keys():
        df = components_t[param]
        arg = df[components_df.index].values
    else:
        arg = None  # fallback
    return arg



def get_block_name(attr_name, index, components_df):
    """
    Computes a consistent block name.
    """
    if isinstance(components_df, pd.Series) and hasattr(components_df, "name"):
        return components_df.name
    elif index is None:
        return f"{attr_name.split('_')[0]}"
    else:
        return f"{attr_name.split('_')[0]}_{index}"
    
    
def determine_size_type(
    smspp_parameters,
    dimensions,
    conversion_dict,
    attr_name,
    key,
    args=None
):
    """
    Determines the size and type of a variable for NetCDF export.

    Parameters
    ----------
    smspp_parameters : dict
        Excel-parsed parameter sheets
    dimensions : dict
        Dictionary of dimension values across blocks
    conversion_dict : dict
        Maps PyPSA dimension names to SMS++ dimensions
    attr_name : str
        The block attribute name (e.g. ThermalUnitBlock_parameters)
    key : str
        The variable name to look up
    args : any
        The variable value (optional, default None)

    Returns
    -------
    variable_type : str
    variable_size : tuple
    """
    block_class = attr_name.split("_")[0]
    row = smspp_parameters[block_class].loc[key]
    variable_type = row['Type']

    # Compose unified dimension dict
    dim_map = {
        key: value
        for subdict in dimensions.values()
        for key, value in subdict.items()
    }
    dim_map[1] = 1
    dim_map['NumberLines'] = dim_map.get('Lines', 0)
    if 'NumAssets_partial' in dim_map:
        dim_map['NumAssets'] = dim_map['NumAssets_partial']

    # es:
    # [T][1] → "T,1"
    # [NA]|[T][NA] → "NA", "T,NA"

    size_arr = re.sub(r'\[|\]', '', str(row['Size']).replace("][", ","))
    size_arr = size_arr.replace(" ", "").split("|")
    
    variable_size = None

    if args is not None:
        if isinstance(args, (float, int, np.integer)):
            variable_size = ()
        else:
            shape = args.shape if isinstance(args, np.ndarray) else (len(args),)
    
            for size_expr in size_arr:
                if size_expr == '1' and shape == (1,):
                    variable_size = ()
                    break
                size_components = size_expr.split(",")
                try:
                    expected_shape = tuple(
                        dim_map[conversion_dict[s]]
                        for s in size_components
                    )
                except KeyError:
                    continue
    
                if shape == expected_shape:
                    if len(size_components) == 1 or "1" in size_components:
                        variable_size = (conversion_dict[size_components[0]],)
                    else:
                        variable_size = tuple(
                            conversion_dict[dim]
                            for dim in size_components
                        )
                    break
    
    # se ancora None → errore
    if variable_size is None:
        logger.warning(
            f"[determine_size_type] Mismatch on variable '{key}' "
            f"in block '{block_class}': expected one of {size_arr}, got shape {shape}"
        )
        raise ValueError(
            f"Size mismatch for variable '{key}' in '{attr_name}': "
            f"could not match shape {shape} with expected {size_arr}"
        )


    return variable_type, variable_size


def merge_lines_and_links(networkblock: dict) -> None:
    """
    Merge the variables of 'Lines' and 'Links' into a single block 'Lines'.
    This is required because SMS++ expects a unified DCNetworkBlock for 
    all transmission elements, treating links as lines with efficiencies < 1.

    Parameters
    ----------
    networkblock : dict
        The Transformation.networkblock dictionary.

    Notes
    -----
    If both Lines and Links exist, their variables are concatenated.
    """
    for key, value in networkblock["Lines"]["variables"].items():
        try:
            if not isinstance(value["value"], (int, float, np.integer)):
                networkblock["Lines"]["variables"][key]["value"] = np.concatenate([
                    networkblock["Lines"]["variables"][key]["value"],
                    networkblock["Links"]["variables"][key]["value"]
                ])
        except ValueError as e:
            logger.warning(f"Could not merge variable {key} due to shape mismatch: {e}")
    # after merging, drop the separate Links block
    networkblock.pop("Links", None)


def rename_links_to_lines(networkblock: dict) -> None:
    """
    Rename 'Links' block as 'Lines' if there are no actual Lines present.
    This is required because SMS++ expects a block named 'Lines'.

    Parameters
    ----------
    networkblock : dict
        The Transformation.networkblock dictionary.

    Notes
    -----
    Also adjusts the variable sizes from 'Links' to 'Lines'.
    """
    networkblock["Lines"] = networkblock.pop("Links")
    for key, var in networkblock["Lines"]["variables"].items():
        var["size"] = tuple("NumberLines" if x == "Links" else x for x in var["size"])


