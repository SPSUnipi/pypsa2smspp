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
    Maps a bus label Series to its integer index in n.buses and stores it in a new column.

    Parameters
    ----------
    n : pypsa.Network
        The network.
    components_df : pd.DataFrame
        DataFrame of the component to update.
    bus_series : pd.Series
        Series of bus names (e.g., generators.bus).
    column_name : str
        Name of the new column to create with numeric indices.
    dtype : str, optional
        Data type of the index column (default: "uint32").

    Returns
    -------
    None
    """
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
