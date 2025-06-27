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
    Computes the dimensions of the NetworkBlock from the PyPSA network.
    """
    network_components = {
        "Lines": ['lines'],
        "Links": ['links'],
        "combined": ['lines', 'links']
    }
    dimensions = {
        **{
            name: sum(len(getattr(n, comp)) for comp in comps)
            for name, comps in network_components.items()
        }
    }
    return dimensions


def investmentblock_dimensions(n, nominal_attrs):
    """
    Computes the dimensions of the InvestmentBlock from the PyPSA network.
    """
    investment_components = ['generators', 'stores', 'lines', 'links']
    num_assets = 0
    for comp in investment_components:
        df = getattr(n, comp)
        comp_type = comp[:-1].capitalize() if comp != "stores" else "Store"
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
            return "ThermalUnitBlock_parameters"

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
    unitblock_index += 1
    lines_index += 1

    investment_meta["asset_type"].extend([1] * len(df_investment))
    

    return unitblock_index, lines_index


