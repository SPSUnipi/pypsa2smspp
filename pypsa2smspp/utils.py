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
from typing import Optional, Sequence, Union


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


def _normalize_selector(x: Union[bool, str, Sequence[str]]) -> Union[bool, list[str]]:
    """Normalize selector types: bool | str | list[str] -> bool | list[str]."""
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    out = [str(v).strip() for v in x]
    return [v for v in out if v]




#%%
#################################################################################################
############################### Dimensions for SMS++ ############################################
#################################################################################################

def ucblock_variables(n):
    """
    Build UCBlock variables related to names and topology indexing.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network.

    Returns
    -------
    dict
        Dictionary with UCBlock variables.
    """
    node_names = np.array(n.buses.index.astype(str), dtype=object)
    line_names = np.array(
        list(n.lines.index.astype(str)) + list(n.links.index.astype(str)),
        dtype=object,
    )

    variables = {
        "node_name": {
            "name": "NodeName",
            "type": "str",
            "size": ("NumberNodes",),
            "value": node_names,
        },
        "line_name": {
            "name": "LineName",
            "type": "str",
            "size": ("NumberLines",),
            "value": line_names,
        },
    }

    return variables

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
            "NumberDesignLines_links": num_design_links,
            "NumberBranches": 0,
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
        "NumberBranches": 0,
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
    
    # Prime righe facoltative perché se ho sector coupled lo gestisco già alla fine...
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
    
    if dimensions['NetworkBlock']['NumberBranches'] > 0:
        # dimensions['NetworkBlock']['NumberBranches'] -= number_merged_links # sbagliato perché viene calcolato dopo e quindi tiene già conto dei 100
        # dimensions['InvestmentBlock']['NumberDesignLines'] = dimensions['NetworkBlock']['NumberBranches_ext']
        # dimensions['NetworkBlock']['NumberLines'] = dimensions['NetworkBlock']['combined']
    
        # Correct number of links and branches based on real branches
        dimensions['NetworkBlock']['NumberBranches'] += dimensions['NetworkBlock']['Lines']
        dimensions['UCBlock']['NumberBranches'] = dimensions['NetworkBlock']['NumberBranches']
        dimensions['NetworkBlock']['Links'] = dimensions['NetworkBlock']['NumberBranches'] - dimensions['NetworkBlock']['Lines']
        


#%%
###############################################################################################
############################### Direct transformation #########################################
###############################################################################################

def _normalize_carrier_list(x: Union[str, Sequence[str]]) -> set[str]:
    """Normalize carrier list to lowercase set."""
    if isinstance(x, str):
        x = [x]
    return {str(v).strip().lower() for v in x if str(v).strip()}


def get_attr_name(
    component_type: str,
    carrier: str | None = None,
    *,
    enable_thermal_units: bool = False,
    intermittent_carriers: Optional[Union[str, Sequence[str]]] = None,
    default_intermittent: Sequence[str] = (),
) -> str:
    """
    Maps a PyPSA component type and its carrier to the corresponding
    UnitBlock attribute name.

    Logic for Generators:
      - slack/load shedding -> SlackUnitBlock_parameters
      - if enable_thermal_units=True -> IntermittentUnitBlock_parameters (no thermals)
      - else:
          intermittent set = intermittent_carriers if provided else default_intermittent
          carrier in intermittent set -> IntermittentUnitBlock_parameters
          otherwise -> ThermalUnitBlock_parameters
    """
    c = carrier.lower() if carrier else None

    # Generators
    if component_type == "Generator":
        if c in {"slack", "load_shedding", "load shedding", "load"}:
            return "SlackUnitBlock_parameters"

        if not enable_thermal_units:
            return "IntermittentUnitBlock_parameters"

        # Thermals are allowed: decide intermittent vs thermal by carrier list
        intermittent_set = (
            _normalize_carrier_list(intermittent_carriers)
            if intermittent_carriers is not None
            else {str(v).strip().lower() for v in default_intermittent if str(v).strip()}
        )

        if c is not None and c in intermittent_set:
            return "IntermittentUnitBlock_parameters"

        return "ThermalUnitBlock_parameters"

    # StorageUnit
    if component_type == "StorageUnit":
        if c in {"hydro", "phs"}:
            return "HydroUnitBlock_parameters"
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


def build_store_and_merged_links(n, merge_links=False, logger=None, merge_selector=None):
    """
    Build enriched stores_df (adds efficiency_store/efficiency_dispatch)
    and links_merged_df (replaces per-store charge/discharge link pair with
    a single merged link with eta=1 and summed capital_cost).

    Parameters
    ----------
    n : pypsa.Network
    merge_links : bool | str | list[str]
        - False -> no merge
        - True  -> merge only safe presets: tes, battery, h2
        - str/list[str] -> subset of presets and/or custom tags.
          If custom tags are provided (not in {tes,battery,h2}), merge_selector is REQUIRED.
    logger : logging.Logger | callable | None
        If None, uses standard logging. If callable (e.g., print), it will be called with the message.
        If it has .warning, that method is used.
    merge_selector : callable | None
        Power-user hook:
            merge_selector(n, store_name, srow, charge_row, discharge_row) -> bool
        If provided, it can authorize merging additional (custom) pairs beyond presets.

    Returns
    -------
    stores_df : pd.DataFrame
    links_merged_df : pd.DataFrame
    merge_dim : int
        extendable_links_initial - extendable_links_final
    """
    import logging
    import numpy as np
    import pandas as pd

    def _warn(msg: str):
        """Emit warnings robustly for logger=None / logging.Logger / print-like callables."""
        if logger is None:
            logging.getLogger(__name__).warning(msg)
        elif hasattr(logger, "warning"):
            logger.warning(msg)
        else:
            logger(msg)

    def _count_extendable(df):
        """Count number of links with p_nom_extendable == True."""
        if df.empty or "p_nom_extendable" not in df.columns:
            return 0
        return int(df["p_nom_extendable"].fillna(False).astype(bool).sum())

    def _normalize_merge_spec(x):
        """
        Return (preset_part, custom_part).
        Presets are: tes, battery, h2.
        """
        presets = {"tes", "battery", "h2"}

        if isinstance(x, bool):
            return (set(presets) if x else set()), set()

        if isinstance(x, str):
            x = [x]

        requested = {str(s).strip().lower() for s in x if str(s).strip()}
        aliases = {"hydrogen": "h2", "h₂": "h2"}
        requested = {aliases.get(s, s) for s in requested}

        preset_part = requested & presets
        custom_part = requested - presets
        return preset_part, custom_part

    def _to_float(v, default: float = 0.0) -> float:
        """Safe float conversion with NaN handling."""
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
                return float(default)
            return float(v)
        except Exception:
            return float(default)

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
        return stores_df, links_merged_df, 0

    preset_part, custom_part = _normalize_merge_spec(merge_links)

    if custom_part and merge_selector is None:
        raise ValueError(
            "merge_links contains custom entries "
            f"{sorted(custom_part)} but merge_selector is None. "
            "Provide merge_selector(n, store_name, srow, charge_row, discharge_row) -> bool "
            "or restrict merge_links to presets: tes, battery, h2."
        )

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
    charger_bool = link_index_series.str.contains("battery charger", case=False, na=False)
    discharger_bool = link_index_series.str.contains("battery discharger", case=False, na=False)

    battery_chargers_ext = set(
        n.links.loc[charger_bool & n.links["p_nom_extendable"].fillna(False)].index
    )
    battery_dischargers_ext = set(
        n.links.loc[discharger_bool & n.links["p_nom_extendable"].fillna(False)].index
    )

    # Hydrogen reversed: detect forward/backward pairs as in extra functionalities
    h2_backwards = set()
    h2_forwards = set()
    if "reversed" in n.links.columns:
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

    # Loop over stores and merge only if they belong to allowed logic
    for store_name, srow in stores_df.iterrows():
        bus_store = srow.get("bus", None)
        if bus_store is None:
            continue

        # Candidates: links connected to store bus
        cand = links_merged_df[
            (links_merged_df["bus0"] == bus_store) | (links_merged_df["bus1"] == bus_store)
        ]
        if cand.empty:
            continue

        # Charge: elec -> store (bus1 == store_bus); Discharge: store -> elec (bus0 == store_bus)
        charge_rows = cand[cand["bus1"] == bus_store]
        discharge_rows = cand[cand["bus0"] == bus_store]
        if charge_rows.empty or discharge_rows.empty:
            continue

        # Pairing strategy: pick a charge row and match a discharge row that returns to the same elec bus
        charge_row = None
        discharge_row = None
        bus_elec = None

        for _, ch in charge_rows.iterrows():
            be = ch.get("bus0", None)
            if be is None:
                continue
            dis_cand = discharge_rows[discharge_rows["bus1"] == be]
            if not dis_cand.empty:
                charge_row = ch
                discharge_row = dis_cand.iloc[0]
                bus_elec = be
                break

        if charge_row is None or discharge_row is None or bus_elec is None:
            continue

        idx_ch = charge_row.name
        idx_dis = discharge_row.name

        # --------------------------------------------------------------
        # Preset eligibility checks
        # --------------------------------------------------------------

        # TES: store bus is one of the heat buses
        is_tes = bus_store in tes_buses

        # Battery: names follow "battery charger/discharger" pattern AND extendable
        is_battery_pair = (
            (idx_ch in battery_chargers_ext and idx_dis in battery_dischargers_ext)
            or (idx_dis in battery_chargers_ext and idx_ch in battery_dischargers_ext)
        )

        # Hydrogen "inverse": forward/backward pair identified by reversed flag
        is_h2_inv_pair = (
            (idx_ch in h2_backwards and idx_dis in h2_forwards)
            or (idx_dis in h2_backwards and idx_ch in h2_forwards)
        )

        allowed_preset = (
            ("tes" in preset_part and is_tes)
            or ("battery" in preset_part and is_battery_pair)
            or ("h2" in preset_part and is_h2_inv_pair)
        )

        allowed_custom = False
        if merge_selector is not None:
            try:
                allowed_custom = bool(merge_selector(n, store_name, srow, charge_row, discharge_row))
            except Exception as e:
                _warn(f"[merge] merge_selector failed for store '{store_name}': {e}. Skipping.")
                allowed_custom = False

        if not (allowed_preset or allowed_custom):
            continue

        # --------------------------------------------------------------
        # Do the actual merge
        # --------------------------------------------------------------

        eta_ch = _to_float(charge_row.get("efficiency", 1.0), 1.0)
        eta_dis = _to_float(discharge_row.get("efficiency", 1.0), 1.0)

        p_nom_ch = _to_float(charge_row.get("p_nom", 0.0), 0.0)
        p_nom_dis = _to_float(discharge_row.get("p_nom", 0.0), 0.0)

        capex_ch = _to_float(charge_row.get("capital_cost", 0.0), 0.0)
        capex_dis = _to_float(discharge_row.get("capital_cost", 0.0), 0.0)

        ext_ch = bool(charge_row.get("p_nom_extendable", False))
        ext_dis = bool(discharge_row.get("p_nom_extendable", False))

        if ext_ch != ext_dis:
            _warn(
                f"[merge] extendability mismatch for store '{store_name}' "
                f"(charge={ext_ch}, discharge={ext_dis}). Using logical AND."
            )
        pnom_extendable = bool(ext_ch and ext_dis)

        if abs(p_nom_ch - p_nom_dis) > PNOM_TOL:
            _warn(
                f"[merge] p_nom mismatch for store '{store_name}' "
                f"(ch={p_nom_ch}, dis={p_nom_dis}). Using min()."
            )
        p_nom_merged = float(min(p_nom_ch, p_nom_dis))

        capex_merged = float(capex_ch + capex_dis)

        # Update store efficiencies
        stores_df.at[store_name, "efficiency_store"] = float(eta_ch)
        stores_df.at[store_name, "efficiency_dispatch"] = float(eta_dis)

        # Clone one of the originals to inherit optional columns, then override
        new_row = charge_row.copy()
        merged_name = f"{idx_ch or 'NA'}__{idx_dis or 'NA'}"

        new_row.name = merged_name
        new_row["bus0"] = bus_elec
        new_row["bus1"] = bus_store
        new_row["efficiency"] = 1.0
        new_row["marginal_cost"] = 0.0
        new_row["capital_cost"] = capex_merged
        new_row["p_nom"] = p_nom_merged
        new_row["p_nom_extendable"] = pnom_extendable
        new_row["p_min_pu"] = -float(eta_dis)  # discharge limit in merged convention

        # If there are p_nom_min/max columns, keep them consistent (safe defaults)
        for col in ["p_nom_min", "p_nom_max"]:
            if col in new_row.index and pd.isna(new_row[col]):
                new_row[col] = 0.0 if col.endswith("_min") else np.inf

        # Mark rows to drop (original charge/discharge)
        rows_to_drop.extend([idx_ch, idx_dis])
        rows_to_append.append(new_row)

    # Apply drops/appends
    if rows_to_drop:
        links_merged_df = links_merged_df.drop(
            index=[r for r in rows_to_drop if r in links_merged_df.index]
        )
    if rows_to_append:
        links_merged_df = pd.concat([links_merged_df, pd.DataFrame(rows_to_append)], axis=0)

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

    If return_efficiencies=True, returns:
        exploded_df, efficiencies_dict, number_branches, number_branches_expandable

    Otherwise returns:
        exploded_df, number_branches, number_branches_expandable

    efficiencies_dict maps each physical link to a list:
        [efficiency, efficiency2, efficiency3, ...]

    Important rule:
    An efficiency is considered valid only if the corresponding bus exists.
    If the bus is empty/missing, the efficiency is forced to 0.0 even if a
    non-zero value is present in the dataframe.
    """
    if links_merged_df.empty:
        if return_efficiencies:
            return links_merged_df.copy(), {}, 0, 0
        return links_merged_df.copy(), 0, 0

    df = links_merged_df.copy()
    efficiencies_dict = {}

    # Identify extra bus columns dynamically: bus2, bus3, ...
    bus_extra_cols = []
    k = 2
    while f"bus{k}" in df.columns:
        bus_extra_cols.append(f"bus{k}")
        k += 1

    # Total number of efficiency slots expected: efficiency, efficiency2, ...
    max_eff_count = 1 + len(bus_extra_cols)

    def _non_empty(val) -> bool:
        """Return True if value is not NaN and not an empty string."""
        return pd.notna(val) and str(val).strip() != ""

    def _get_valid_efficiency(row, bus_col: str, eff_col: str) -> float:
        """
        Return efficiency only if the corresponding bus exists.
        Otherwise force it to 0.0.
        """
        if not _non_empty(row.get(bus_col, np.nan)):
            return 0.0

        val = row.get(eff_col, np.nan)
        if not _non_empty(val):
            return 0.0

        try:
            return float(val)
        except Exception:
            return 0.0

    new_rows = []

    for link_name, row in df.iterrows():
        # ------------------------------------------------------------------
        # Build efficiencies_dict consistently with actual bus existence
        # ------------------------------------------------------------------
        eff_list = []

        # Primary branch: bus1 <-> efficiency
        eff_list.append(_get_valid_efficiency(row, "bus1", "efficiency"))

        # Extra branches: bus2 <-> efficiency2, bus3 <-> efficiency3, ...
        for idx, bcol in enumerate(bus_extra_cols, start=2):
            ecol = f"efficiency{idx}"
            eff_list.append(_get_valid_efficiency(row, bcol, ecol))

        # Pad just in case, though normally already correct
        if len(eff_list) < max_eff_count:
            eff_list += [0.0] * (max_eff_count - len(eff_list))

        efficiencies_dict[link_name] = eff_list

        # ------------------------------------------------------------------
        # Build exploded rows
        # Only create extra outputs for buses that actually exist
        # ------------------------------------------------------------------
        extra_outputs = []
        for idx, bcol in enumerate(bus_extra_cols, start=2):
            if _non_empty(row.get(bcol, np.nan)):
                ecol = f"efficiency{idx}"
                eff_val = _get_valid_efficiency(row, bcol, ecol)
                extra_outputs.append((bcol, ecol, eff_val))

        is_multilink = len(extra_outputs) > 0

        # If there are no extra outputs, keep the row as a single branch
        if not is_multilink:
            out_row = row.copy()
            out_row["hyper"] = hyper_id
            out_row["is_primary_branch"] = True

            # Force primary efficiency to 0.0 if bus1 is missing
            out_row["efficiency"] = _get_valid_efficiency(row, "bus1", "efficiency")

            new_rows.append(out_row)
            hyper_id += 1
            continue

        # True multilink: create explicit primary branch
        primary_bus = row.get("bus1", np.nan)
        primary_eff = _get_valid_efficiency(row, "bus1", "efficiency")

        pr = row.copy()
        pr["bus1"] = primary_bus
        pr["efficiency"] = primary_eff
        pr.name = f"{link_name}__to_{primary_bus}" if _non_empty(primary_bus) else f"{link_name}__to_bus1"
        pr["hyper"] = hyper_id
        pr["is_primary_branch"] = True
        new_rows.append(pr)

        # Create one child per valid extra bus
        for bcol, ecol, eff_val in extra_outputs:
            child = row.copy()
            child_bus = row[bcol]

            child["bus1"] = child_bus
            child["efficiency"] = eff_val
            child.name = f"{link_name}__to_{child_bus}"
            child["hyper"] = hyper_id
            child["is_primary_branch"] = False

            new_rows.append(child)

        hyper_id += 1

    exploded = pd.DataFrame(new_rows)

    # Drop redundant bus/efficiency columns after explosion
    cols_to_drop = [
        c for c in exploded.columns
        if (c.startswith("bus") and c not in ("bus0", "bus1"))
        or (c.startswith("efficiency") and c != "efficiency")
    ]
    exploded = exploded.drop(columns=cols_to_drop, errors="ignore")

    # Count branches
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
            p_nom_min_safe = p_nom_min
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
        b["ConverterInvestmentCost"] = lambda e_nom_extendable: 1e-12 if bool(first_scalar(e_nom_extendable)) else 0.0

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
            e_nom_min_safe = e_nom_min
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
        
        if components_name == 'Line' or components_df.loc[idx, 'is_primary_branch']:
            lines_index +=1
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

    if size not in [1, '[L]', '[Li]', '[NA]', '[NP]', '[NR]', '[NB]', '[Li] | [NB]', '[L] | [NB]']:
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
    
    if block_class in ['Lines', 'Links'] and dimensions['NetworkBlock']['NumberBranches'] > dimensions['NetworkBlock']['combined']:
        if key in ['StartLine', 'EndLine', 'Efficiency', 'HyperArcID']:
            variable_size = ('NumberBranches',) 
            return variable_type, variable_size
        else:
            variable_size = ('NumberLines',) if block_class == 'Lines' else ('Links',)
            return variable_type, variable_size

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


