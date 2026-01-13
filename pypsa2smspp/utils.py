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

    if field in n.components[component].static.columns:
        field_val = n.get_switchable_as_dense(component, field, sns)
    else:
        field_val = n.dynamic(component)[field]

    if weights:
        field_val = field_val.mul(weighting, axis=0)
    return field_val

         
def remove_zero_p_nom_opt_components(n, nominal_attrs):
    # Lista dei componenti che hanno l'attributo p_nom_opt
    components_with_p_nom_opt = ["Generator", "Link", "Store", "StorageUnit", "Line", "Transformer"]
    
    for components in n.components[["Line", "Generator", "Link", "Store", "StorageUnit"]]:
        if components.empty:
            continue
        components_df = components.static
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
    if extendable_attr not in components_df.columns:
        return components_df

    df = components_df[components_df[extendable_attr]]

    # Special handling for exploded Links
    if component_type == "Link" and df.index.str.contains("__").any():
        df = filter_primary_extendable_links(df)

    return df



def filter_primary_extendable_links(links_df: pd.DataFrame) -> pd.DataFrame:
    """
    From an exploded Link DataFrame, keep only one extendable link per
    original physical link (i.e. before '__').

    Priority:
      1) is_primary_branch == True (if column exists)
      2) first occurrence (stable)

    Parameters
    ----------
    links_df : pd.DataFrame
        DataFrame of PyPSA links (possibly exploded into branches).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only one extendable link per physical link.
    """
    if links_df.empty:
        return links_df

    df = links_df.copy()

    # Extract physical link name (before '__')
    physical_name = df.index.to_series().str.split("__", n=1).str[0]
    df["_physical_name"] = physical_name.values

    selected = []

    for _, group in df.groupby("_physical_name", sort=False):
        if "is_primary_branch" in group.columns:
            primary = group[group["is_primary_branch"]]
            if not primary.empty:
                selected.append(primary.iloc[0])
                continue

        # Fallback: keep first row
        selected.append(group.iloc[0])

    out = pd.DataFrame(selected)
    return out.drop(columns="_physical_name")



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


def first_scalar(x):
    """Return the first scalar value from a pandas/NumPy 1-length container, else cast to float.
    This keeps code robust when inputs come as 1-length Series/Index/ndarray.
    """
    try:
        # pandas Series/Index
        if hasattr(x, "iloc"):
            return float(x.iloc[0])
        # numpy array / list / tuple
        if hasattr(x, "__len__") and not hasattr(x, "shape") or (hasattr(x, "shape") and x.shape != ()):
            return float(list(x)[0])
        # 0-d numpy or plain scalar
        return float(getattr(x, "item", lambda: x)())
    except Exception:
        return float(x)


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


def networkblock_dimensions(n, expansion_ucblock):
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
    
    if expansion_ucblock:
        # count extendable AC lines (s_nom_extendable == True)
        if hasattr(n, "lines") and "s_nom_extendable" in n.lines:
            num_design_lines = int(
                n.lines["s_nom_extendable"]
                .fillna(False)
                .astype(bool)
                .sum()
            )
        else:
            num_design_lines = 0
    
        # count extendable links (p_nom_extendable == True)
        if hasattr(n, "links") and "p_nom_extendable" in n.links:
            num_design_links = int(
                n.links["p_nom_extendable"]
                .fillna(False)
                .astype(bool)
                .sum()
            )
        else:
            num_design_links = 0

        
        return {
            "Lines": lines_count,
            "Links": links_count,
            "combined": combined_count,
            "NumberLines": combined_count,
            "NumberDesignLines_lines": num_design_lines,
            "NumberDesignLines_links": num_design_links
        }

    # # --- detect extra outputs from multi-links to build branches ---
    # extra_outputs = 0
    # if links_count > 0:
    #     link_df = n.links
    #     # iterate bus2, bus3, ... only while column exists
    #     k = 2
    #     while f"bus{k}" in link_df.columns:
    #         s = link_df[f"bus{k}"]
    #         # count non-empty entries: notna and not just whitespace
    #         valid = s.notna() & (s.astype(str).str.strip() != "")
    #         extra_outputs += int(valid.sum())
    #         k += 1

    # # For branches: each physical line contributes 1 branch.
    # # Each physical link contributes 1 branch for bus1 (the first output),
    # # plus one branch for every additional non-empty bus{k>=2}.
    # number_lines = combined_count
    # number_branches = lines_count + links_count + extra_outputs

    return {
        "Lines": lines_count,
        "Links": links_count,
        "combined": combined_count,
        "NumberLines": combined_count,
    }



def investmentblock_dimensions(n, expansion_ucblock, nominal_attrs):
    """
    Computes the dimensions of the InvestmentBlock from the PyPSA network.
    If expansion is in UCBlocks, calculates for lines only
    
    """
    investment_components = ['lines', 'links'] if expansion_ucblock else ['generators', 'storage_units', 'stores', 'lines', 'links']
    num_assets = 0
    for comp in investment_components:
        df = getattr(n, comp)
        comp_type = comp[:-1].capitalize() if comp != "storage_units" else "StorageUnit"
        attr = nominal_attrs.get(comp_type)
        if attr and f"{attr}_extendable" in df.columns:
            num_assets += df[f"{attr}_extendable"].sum()

    return {"NumberDesignLines": int(num_assets), "NumberSubNetwork": int(len(n.snapshots))} if expansion_ucblock else {"NumAssets": int(num_assets)}


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

def correct_dimensions(dimensions, stores_df, links_merged_df, n, expansion_ucblock):
    """
    Correct SMS++ dimensions based on particular cases/flags
    1. if we merge links, reduce the number of lines associated
    2. if we expand lines with DesignNetworkBlock, define NumberNetworks
    3. if we are in sector coupled, reduce the number of branches associated (if merge_links)
    """
    
    number_merged_links = dimensions['NetworkBlock']['Links'] - len(links_merged_df)
    number_ext_merg_links = dimensions['NetworkBlock']['merged_links_ext']
    
    # Reduce the number of lines depending on the merged links
    dimensions['NetworkBlock']['Links'] -= number_merged_links
    dimensions['NetworkBlock']['combined'] -= number_merged_links
    dimensions['UCBlock']['NumberLines'] -= number_merged_links
    
    if expansion_ucblock:
       dimensions['InvestmentBlock']['NumberDesignLines'] -= number_ext_merg_links 
       dimensions['NetworkBlock']['NumberDesignLines_links'] -= number_ext_merg_links 
       if dimensions['InvestmentBlock']['NumberDesignLines'] > 0:
           dimensions['UCBlock']['NumberNetworks'] = 1
    else:
       dimensions['InvestmentBlock']['NumAssets'] -= number_ext_merg_links
    
    if "NumberBranches" in dimensions['NetworkBlock']:
        dimensions['NetworkBlock']['NumberBranches'] -= number_merged_links
        dimensions['UCBlock']['NumberBranches'] = dimensions['NetworkBlock']['NumberBranches']
        # dimensions['InvestmentBlock']['NumberDesignLines'] = dimensions['NetworkBlock']['NumberBranches_ext']
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
        elif carrier in ["slack", "load_shedding", "load shedding"]:
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

    IMPORTANT
    ---------
    Merging is only performed for specific technologies that PyPSA-Eur
    constrains to have tied charger/discharger (or forward/backward) capacities:
    - TES (heat) stores
    - Battery stores
    - Hydrogen "inverse" (reversed/forward) links

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

    merge_dim : int
        Number of extendable links "lost" by merging:
        extendable_links_initial - extendable_links_final

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

    def _count_extendable(df):
        """Count number of links with p_nom_extendable == True."""
        if df.empty or "p_nom_extendable" not in df.columns:
            return 0
        return int(df["p_nom_extendable"].fillna(False).astype(bool).sum())

    stores_df = n.stores.copy()
    links_merged_df = n.links.copy()

    # Add the two new columns with safe defaults
    for col in ["efficiency_store", "efficiency_dispatch"]:
        if col not in stores_df.columns:
            stores_df[col] = 1.0

    # Count extendable links before any merging
    extendable_initial = _count_extendable(links_merged_df)

    # If merging is disabled or network is trivial, exit early
    if not merge_links or links_merged_df.empty or stores_df.empty:
        merge_dim = 0
        return stores_df, links_merged_df, merge_dim

    # ------------------------------------------------------------------
    # Detect technologies for which merge is allowed (TES, batteries, H2)
    # ------------------------------------------------------------------

    # TES: detect heat buses similar to PyPSA-Eur extra functionalities
    tes_bus_mask = n.buses.index.to_series().str.contains(
        r"urban central heat|urban decentral heat|rural heat",
        case=False,
        na=False,
    )
    tes_buses = set(n.buses.index[tes_bus_mask])

    # Batteries: detect charger/discharger extendable links as in PyPSA-Eur
    link_index_series = n.links.index.to_series()
    charger_bool = link_index_series.str.contains(
        "battery charger", case=False, na=False
    )
    discharger_bool = link_index_series.str.contains(
        "battery discharger", case=False, na=False
    )

    battery_chargers_ext = set(
        n.links.loc[
            charger_bool & n.links["p_nom_extendable"].fillna(False)
        ].index
    )
    battery_dischargers_ext = set(
        n.links.loc[
            discharger_bool & n.links["p_nom_extendable"].fillna(False)
        ].index
    )

    # Hydrogen reversed: detect forward/backward pairs as in extra functionalities
    h2_backwards = set()
    h2_forwards = set()
    if "reversed" in n.links.columns:
        # carriers of reversed links
        carriers_rev = (
            n.links.loc[n.links["reversed"].fillna(False), "carrier"]
            .dropna()
            .unique()
        )
        if len(carriers_rev) > 0:
            mask_back = (
                n.links["carrier"].isin(carriers_rev)
                & n.links["p_nom_extendable"].fillna(False)
                & n.links["reversed"].fillna(False)
            )
            h2_backwards = set(n.links.index[mask_back])
            # forward link names obtained by removing "-reversed"
            h2_forwards = set(idx.replace("-reversed", "") for idx in h2_backwards)

    # We will collect rows to drop and rows to append
    rows_to_drop = []
    rows_to_append = []

    # Tolerance for p_nom equality check
    PNOM_TOL = 1e-6

    # Loop over stores and merge only if they belong to TES/battery/H2 logic
    for store_name, srow in stores_df.iterrows():
        bus_store = srow["bus"]

        # Heuristic: the links are the ones connected to the store bus
        mask_ch = (links_merged_df["bus0"] == bus_store) | (
            links_merged_df["bus1"] == bus_store
        )
        cand = links_merged_df[mask_ch]
        if cand.empty:
            # No links connected to this store -> nothing to merge
            continue

        # We assume at most one charge and one discharge per store
        charge_rows = cand[cand["bus1"] == bus_store]
        discharge_rows = cand[cand["bus0"] == bus_store]

        if charge_rows.empty or discharge_rows.empty:
            # We don't have a proper pair -> skip merging this store
            continue

        charge_row = charge_rows.iloc[0]
        discharge_row = discharge_rows.iloc[0]

        idx_ch = charge_row.name
        idx_dis = discharge_row.name

        bus_elec = (
            charge_row["bus0"]
            if charge_row["bus0"] == discharge_row["bus1"]
            else None
        )
        if bus_elec is None:
            # Could not determine the paired electrical bus; skip merge
            continue

        # --------------------------------------------------------------
        # Check if this store/link pair belongs to allowed technologies
        # --------------------------------------------------------------

        # TES: store bus is one of the heat buses
        is_tes = bus_store in tes_buses

        # Battery: names follow "battery charger/discharger" pattern
        is_battery_pair = (
            (idx_ch in battery_chargers_ext and idx_dis in battery_dischargers_ext)
            or (idx_dis in battery_chargers_ext and idx_ch in battery_dischargers_ext)
        )

        # Hydrogen "inverse": forward/backward pair identified by reversed flag
        is_h2_inv_pair = (
            (idx_ch in h2_backwards and idx_dis in h2_forwards)
            or (idx_dis in h2_backwards and idx_ch in h2_forwards)
        )

        # If none of the above holds, do NOT merge this store
        if not (is_tes or is_battery_pair or is_h2_inv_pair):
            continue

        # --------------------------------------------------------------
        # From here on, do the actual merge as before
        # --------------------------------------------------------------

        # Extract params with defaults
        # Charge (elec -> store)
        eta_ch = charge_row.efficiency
        p_nom_ch = charge_row.p_nom
        capex_ch = charge_row.capital_cost
        ext_ch = bool(charge_row.p_nom_extendable)
        name_ch = charge_row.name

        # Discharge (store -> elec)
        eta_dis = discharge_row.efficiency
        p_nom_dis = discharge_row.p_nom
        capex_dis = discharge_row.capital_cost
        ext_dis = bool(discharge_row.p_nom_extendable)
        name_dis = discharge_row.name

        # Extendability must match (as per your assumption)
        if ext_ch != ext_dis:
            logger.warning(
                f"[merge] Warning: extendability mismatch for store '{store_name}' "
                f"(charge={ext_ch}, discharge={ext_dis}). Using logical AND."
            )
        pnom_extendable = bool(ext_ch and ext_dis)

        # Choose p_nom for the merged link
        if abs(p_nom_ch - p_nom_dis) > PNOM_TOL:
            logger.warning(
                f"[merge] Warning: p_nom mismatch for store '{store_name}' "
                f"(ch={p_nom_ch}, dis={p_nom_dis}). Using min()."
            )
        p_nom_merged = float(min(p_nom_ch, p_nom_dis))

        # Capital cost is the SUM (two converters of same size)
        capex_merged = float(capex_ch + capex_dis)

        # Update store efficiencies
        stores_df.at[store_name, "efficiency_store"] = float(eta_ch)
        stores_df.at[store_name, "efficiency_dispatch"] = float(eta_dis)

        # Prepare merged link row:
        # We clone one of the originals to inherit optional columns, then override.
        new_row = charge_row if charge_row is not None else discharge_row
        new_row = new_row.copy()  # avoid SettingWithCopy issues

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
        new_row["p_min_pu"] = -eta_dis  # account for discharge limit perspective

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
        links_merged_df = links_merged_df.drop(
            index=[r for r in rows_to_drop if r in links_merged_df.index]
        )
    if rows_to_append:
        links_merged_df = pd.concat(
            [links_merged_df, pd.DataFrame(rows_to_append)], axis=0
        )

    # Count extendable links after merging
    extendable_final = _count_extendable(links_merged_df)
    merge_dim = extendable_initial - extendable_final

    return stores_df, links_merged_df, merge_dim



def explode_multilinks_into_branches(
    links_merged_df: pd.DataFrame,
    hyper_id,
    logger=print,
    return_efficiencies: bool = True,
):
    """
    Split multi-output links into separate branches, keeping track of efficiencies.
    If `return_efficiencies=True`, returns (exploded_df, efficiencies_dict),
    where efficiencies_dict maps each physical link -> [eff1, eff2, ...].
    Missing efficiencyN values are padded with 0.0 to ensure uniform length.
    """
    if links_merged_df.empty:
        return (links_merged_df.copy(), {}) if return_efficiencies else links_merged_df.copy()

    df = links_merged_df.copy()
    efficiencies_dict = {}

    # Identify extra bus/eff columns dynamically
    bus_extra_cols = []
    k = 2
    while f"bus{k}" in df.columns:
        bus_extra_cols.append(f"bus{k}")
        k += 1

    # Determine total number of efficiency columns present in the DF
    eff_cols = [c for c in df.columns if c.startswith("efficiency")]
    max_eff_count = 1 + sum([f"efficiency{i}" in df.columns for i in range(2, k)])  # e.g. efficiency, efficiency2, efficiency3...

    def _non_empty(val) -> bool:
        return pd.notna(val) and str(val).strip() != ""

    new_rows = []

    for link_name, row in df.iterrows():
        eff_list = []

        # Primary efficiency always exists (fill NaN with 0)
        try:
            eff_list.append(float(row.get("efficiency", 0.0)))
        except Exception:
            eff_list.append(0.0)

        # Extra efficiencies (efficiency2, efficiency3, ...)
        for idx, bcol in enumerate(bus_extra_cols, start=2):
            ecol = f"efficiency{idx}"
            if ecol in df.columns:
                val = row.get(ecol, np.nan)
                if _non_empty(val):
                    eff_list.append(float(val))
                else:
                    eff_list.append(0.0)
            else:
                eff_list.append(0.0)

        # Pad with zeros if needed (so all lists have equal length)
        if len(eff_list) < max_eff_count:
            eff_list += [0.0] * (max_eff_count - len(eff_list))

        efficiencies_dict[link_name] = eff_list

        # ---- build exploded rows ----
        extra_outputs = [(bcol, f"efficiency{idx}")
                         for idx, bcol in enumerate(bus_extra_cols, start=2)
                         if _non_empty(row.get(bcol, np.nan))]

        is_multilink = len(extra_outputs) > 0

        if not is_multilink:
            out_row = row.copy()
            out_row["hyper"] = hyper_id
            out_row["is_primary_branch"] = True
            new_rows.append(out_row)
            hyper_id += 1
            continue

        # true multilink
        primary_bus = row["bus1"]
        primary_eff = row["efficiency"]

        pr = row.copy()
        pr["bus1"] = primary_bus
        pr["efficiency"] = float(primary_eff)
        pr.name = f"{link_name}__to_{primary_bus}"
        pr["hyper"] = hyper_id
        pr["is_primary_branch"] = True
        new_rows.append(pr)

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

    # Drop redundant bus/eff columns
    cols_to_drop = [c for c in exploded.columns
                    if (c.startswith("bus") and c not in ("bus0", "bus1"))
                    or (c.startswith("efficiency") and c != "efficiency")]
    exploded = exploded.drop(columns=cols_to_drop, errors="ignore")

    # ---- QUI il conteggio dei branches espandibili (poche righe) ----
    n_phys = len(df)
    number_branches = len(exploded)

    if "p_nom_extendable" in exploded.columns:
        number_branches_expandable = int(
            exploded["p_nom_extendable"].fillna(False).astype(bool).sum()
        )
    else:
        number_branches_expandable = 0

    extra = number_branches - n_phys
    if callable(logger):
        logger(
            f"[multilink] Exploded {n_phys} physical links into "
            f"{number_branches} branches (+{extra}). "
            f"Expandable branches: {number_branches_expandable}."
        )

    if return_efficiencies:
        return exploded, efficiencies_dict, number_branches, number_branches_expandable
    return exploded, number_branches, number_branches_expandable




# Translate into generic once the ucblock\investmentblock general use is defined  
def add_sectorcoupled_parameters(
    Lines_parameters,
    Links_parameters,
    inverse_dict=None,     
    max_eff_len: int = 1,
):
    """
    Add a HyperArcID entry to Lines_parameters and Links_parameters and
    (optionally) patch DCNetworkBlock_links_inverse by adding p2..pn.

    For p2..pn we use the rule:
        if efficiency == 1 -> zeros_like(flowvalue)
        else               -> -flowvalue * efficiency
    p1 is kept as-is if already present in inverse_dict (fallback provided otherwise).
    """

    # --- existing behavior (unchanged) -----------------------------------------
    hyper_def = lambda hyper: hyper.values  # default HyperArcID

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
            np.zeros_like(p_nom[is_primary_branch].values),
        "NetworkCost": lambda marginal_cost, is_primary_branch:
            (marginal_cost[is_primary_branch].values)
    })

    # --- NEW: patch inverse_dict (in place, no return) -------------------------
    if inverse_dict is None:
        return  # nothing to patch

    # Special rule for p2..pn
    def _p_rule(flowvalue, efficiency):
        """Return zeros if efficiency==1, else -flowvalue*efficiency. Handles arrays/scalars."""
        fv = np.asarray(flowvalue)
        ef = np.asarray(efficiency)

        # Try to broadcast ef to fv shape (covers fv:(T,E) vs ef:(T,) cases)
        if ef.shape != fv.shape:
            try:
                ef = np.broadcast_to(ef, fv.shape)
            except ValueError:
                if ef.ndim == 1 and fv.ndim > 1 and ef.shape[0] == fv.shape[0]:
                    ef = ef.reshape((fv.shape[0],) + (1,) * (fv.ndim - 1))
                else:
                    ef = np.broadcast_to(ef, fv.shape)  # will raise if impossible

        mask_one = np.isclose(ef, 1.0)
        return np.where(mask_one, 0.0, -fv * ef)

    # Add/override p2..pn
    max_eff_len = int(max(1, max_eff_len))
    for k in range(2, max_eff_len + 1):
        inverse_dict[f"p{k}"] = _p_rule


    
# Sempre nella classe Transformation
def apply_expansion_overrides(IntermittentUnitBlock_parameters=None, BatteryUnitBlock_store_parameters=None, IntermittentUnitBlock_inverse=None, BatteryUnitBlock_inverse=None, InvestmentBlock=None):
    """
    Inject missing keys for UC expansion to be solved inside UCBlock instead of a separate InvestmentBlock.
    Keys are only added if missing, so it remains idempotent.
    """

    # --- IntermittentUnitBlock ---
    d = IntermittentUnitBlock_parameters

    # "InvestmentCost"
    if "InvestmentCost" not in d:
        # Pass-through of capital_cost (assumed already scalar or 1-length)
        d["InvestmentCost"] = lambda capital_cost, p_nom_extendable: capital_cost if bool(first_scalar(p_nom_extendable)) else 0.0

    # "MaxCapacityDesign"
    if "MaxCapacityDesign" not in d:
        # Replace +inf with a large sentinel (1e7), then pick scalar based on extendable flag
        def _max_cap_design(p_nom, p_nom_extendable, p_nom_max):
            p_nom_max_safe = p_nom_max.replace(np.inf, 1e7)
            return (first_scalar(p_nom_max_safe)
                    if bool(first_scalar(p_nom_extendable))
                    else first_scalar(p_nom))
        d["MaxCapacityDesign"] = _max_cap_design
        
    # "MinCapacityDesign
    if "MinCapacityDesign" not in d:
        def _min_cap_design(p_nom, p_nom_extendable, p_nom_min):
            p_nom_min_safe = p_nom_min.replace(np.inf, 1e-6)
            return (first_scalar(p_nom_min_safe)
                    if bool(first_scalar(p_nom_extendable))
                    else first_scalar(p_nom))
        d["MinCapacityDesign"] = _min_cap_design

    # --- BatteryUnitBlock_store ---
    b = BatteryUnitBlock_store_parameters

    # "BatteryInvestmentCost"
    if "BatteryInvestmentCost" not in b:
        b["BatteryInvestmentCost"] = lambda capital_cost, e_nom_extendable: capital_cost if bool(first_scalar(e_nom_extendable)) else 0.0

    # "ConverterInvestmentCost"
    if "ConverterInvestmentCost" not in b:
        b["ConverterInvestmentCost"] = lambda e_nom_extendable: 1e-6 if bool(first_scalar(e_nom_extendable)) else 0.0

    # "BatteryMaxCapacityDesign"
    if "BatteryMaxCapacityDesign" not in b:
        def _battery_max_cap_design(e_nom, e_nom_extendable, e_nom_max):
            e_nom_max_safe = e_nom_max.replace(np.inf, 1e7)
            return (first_scalar(e_nom_max_safe)
                    if bool(first_scalar(e_nom_extendable))
                    else first_scalar(e_nom))
        b["BatteryMaxCapacityDesign"] = _battery_max_cap_design
        
    # "BatteryMinCapacityDesign"
    if "BatteryMinCapacityDesign" not in b:
        def _battery_min_cap_design(e_nom, e_nom_extendable, e_nom_min):
            e_nom_min_safe = e_nom_min.replace(np.inf, 1e-6)
            return (first_scalar(e_nom_min_safe)
                    if bool(first_scalar(e_nom_extendable))
                    else first_scalar(e_nom))
        b["BatteryMinCapacityDesign"] = _battery_min_cap_design

    # "ConverterMaxCapacityDesign"
    if "ConverterMaxCapacityDesign" not in b:
        def _conv_max_cap_design(e_nom, e_nom_extendable, e_nom_max):
            e_nom_max_safe = e_nom_max.replace(np.inf, 1e7)
            # Your rule of thumb: 10x battery energy cap when extendable, else e_nom
            return (10.0 * first_scalar(e_nom_max_safe)
                    if bool(first_scalar(e_nom_extendable))
                    else first_scalar(e_nom))
        b["ConverterMaxCapacityDesign"] = _conv_max_cap_design
        
    
    # "ConverterMinCapacityDesign"
    if "ConverterMinCapacityDesign" not in b:
        def _conv_min_cap_design(e_nom, e_nom_extendable, e_nom_min):
            e_nom_min_safe = e_nom_min.replace(np.inf, 1e7)
            # Your rule of thumb: 10x battery energy cap when extendable, else e_nom
            return (10.0 * first_scalar(e_nom_min_safe)
                    if bool(first_scalar(e_nom_extendable))
                    else first_scalar(e_nom))
        b["ConverterMinCapacityDesign"] = _conv_min_cap_design

    
    # --- IntermittentUnitBlock_inverse ---
    IntermittentUnitBlock_inverse["p_nom"] = (
        lambda intermittentdesign: intermittentdesign
    )

    # --- BatteryUnitBlock_inverse ---
    BatteryUnitBlock_inverse["e_nom"] = (
        lambda batterydesign: batterydesign
    )
    
    
    # --- InvestmentBlockParameters ---
    i = InvestmentBlock
    
    # DesignLines
    i['InvestmentCost'] = i.pop('Cost')
    i['MinCapacityDesign'] = i.pop('LowerBound')
    i['MaxCapacityDesign'] = i.pop('UpperBound')
    i.pop('InstalledQuantity')    


def build_dc_index(n, links_merged_df_before_split, links_df_after_split):
    """
    Build a unified DC index registry capturing both physical and branch views.
    Returns a dict with:
      - physical: {'names': [...], 'types': [...]}           # NumberLines order
      - branch:   {'names': [...], 'types': [...]}            # NumberBranches order (links-only here)
      - map_df:   DataFrame with per-branch mapping:
          columns = ['kind','name','hyper','is_primary_branch','phys_name','phys_kind']
        where:
          - 'name' is branch name (for non-multilink, equals physical name)
          - 'phys_name' is the physical object name
          - 'kind' is 'line' or 'link' (branch-level)
          - 'phys_kind' is 'line' or 'link' (physical)
    """
    # --- physical view (NumberLines): lines + links (pre-split) ---
    phys_line_names = list(n.lines.index)
    phys_line_types = ['line'] * len(phys_line_names)

    phys_link_names = list(links_merged_df_before_split.index)
    phys_link_types = ['link'] * len(phys_link_names)

    phys_names = phys_line_names + phys_link_names
    phys_types = phys_line_types + phys_link_types

    # --- branch view (NumberBranches): after split (only links contribute >1) ---
    # Lines are not split, so their "branch view" is trivial and not needed for links_df_after_split
    # We keep only link branches here and rely on hyper offset based on len(lines).
    branch_names = list(links_df_after_split.index)
    branch_types = ['link'] * len(branch_names)

    # --- mapping per-branch -> physical ---
    # hyper of lines: 0..len(lines)-1
    # hyper of links: start from len(lines)
    # links_df_after_split must contain ['hyper','is_primary_branch']
    if not {'hyper','is_primary_branch'}.issubset(links_df_after_split.columns):
        raise ValueError("links_df_after_split must have 'hyper' and 'is_primary_branch' columns.")

    # Build DataFrame for link branches
    map_link = pd.DataFrame({
        'kind': ['link'] * len(branch_names),
        'name': branch_names,
        'hyper': links_df_after_split['hyper'].astype(int).values,
        'is_primary_branch': links_df_after_split['is_primary_branch'].astype(bool).values,
    }, index=branch_names)

    # Resolve phys_name from hyper
    # lines occupy the first block of hypers
    n_lines = len(phys_line_names)
    def _phys_from_hyper(h):
        if h < n_lines:
            return phys_line_names[h], 'line'
        else:
            return phys_link_names[h - n_lines], 'link'

    phys_resolved = map_link['hyper'].map(lambda h: _phys_from_hyper(int(h)))
    map_link['phys_name'] = [p[0] for p in phys_resolved]
    map_link['phys_kind'] = [p[1] for p in phys_resolved]

    # Return registry
    return {
        'physical': {'names': phys_names, 'types': phys_types},
        'branch':   {'names': branch_names, 'types': branch_types},
        'map_df':   map_link,
    }


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

    Notes
    -----
    - Indices (unitblock_index, lines_index) are always advanced for each row in components_df.
    - Investment metadata is registered only for extendable components.
    - For Links, extendable components are additionally collapsed to one per physical asset
      when exploded branches are detected (handled by filter_extendable_components).
    """
    # Get only extendable components (and for Links also collapse exploded branches)
    extendable_df = filter_extendable_components(components_df, components_name, nominal_attrs)

    # Fast membership test on index
    extendable_idx = set(extendable_df.index)

    # Loop over ALL components, advancing indices always
    for idx in components_df.index:
        if idx in extendable_idx:
            investment_meta["Blocks"].append(f"DCNetworkBlock_{unitblock_index}")
            investment_meta["index_extendable"].append(lines_index)
            investment_meta["design_lines"].append(lines_index)

        lines_index += 1
        unitblock_index += 1

    # asset_type: one entry per investment row (unchanged logic)
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


