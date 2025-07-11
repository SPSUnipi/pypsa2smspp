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

    if size not in [1, '[L]', '[Li]', '[NA]', '[NP]', '[NR]']:
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


