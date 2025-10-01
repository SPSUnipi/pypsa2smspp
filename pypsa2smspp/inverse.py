# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 15:55:15 2025

@author: aless
"""

"""
inverse.py

This module handles the inverse transformation from an SMS++ solution 
back into a PyPSA-compatible xarray.Dataset. It reconstructs time-series 
and scalar variables from SMS unit blocks and prepares them for use with 
assign_solution in PyPSA.
"""

import numpy as np
import xarray as xr


def normalize_key(key: str) -> str:
    """
    Normalizes a key string by converting to lowercase and replacing spaces with underscores.
    """
    return key.lower().replace(" ", "_")


def component_definition(n, unit_block: dict) -> str:
    """
    Maps a unit block type to its corresponding PyPSA component name.
    """
    block = unit_block['block']
    match block:
        case "IntermittentUnitBlock":
            return "Generator"
        case "ThermalUnitBlock":
            return "Generator"
        case "HydroUnitBlock":
            return "StorageUnit"
        case "BatteryUnitBlock":
            return "StorageUnit" if unit_block['name'] in n.storage_units.index else "Store"
        case "DCNetworkBlock_lines":
            return "Line"
        case "DCNetworkBlock_links":
            return "Link"
        case "SlackUnitBlock":
            return "Generator"
        case _:
            raise ValueError(f"Unknown unit block type: {block}")


def evaluate_function(func, normalized_keys, unit_block, df):
    """
    Evaluates a callable inverse function by extracting arguments from unit block or network.
    """
    param_names = func.__code__.co_varnames[:func.__code__.co_argcount]
    args = []
    for param in param_names:
        param = normalize_key(param)
        if param in normalized_keys:
            args.append(unit_block[normalized_keys[param]])
        else:
            args.append(df.loc[unit_block['name']][param])
    return func(*args)


def dataarray_components(n, value, component, unit_block, key):
    """
    Generates dims/coords for a DataArray from inverse values.
    Handles scalars (0-D), 1-D (snapshot or ext), and (T,1) 2-D.
    """
    # Unmask masked arrays; use NaN for masked scalars
    if isinstance(value, np.ma.MaskedArray):
        value = value.filled(np.nan)

    # Coerce to ndarray
    value = np.asarray(value)

    # Case A: scalar (0-D) -> treat as single 'ext' value
    if value.ndim == 0:
        value = value.reshape(1)  # shape -> (1,)
        dims = [f"{component}-ext"]
        coords = {f"{component}-ext": [unit_block["name"]]}
    
    # Case B: 1-D
    elif value.ndim == 1:
        if len(value) == len(n.snapshots):
            # time series for a single unit
            value = value[:, np.newaxis]  # (T,) -> (T,1)
            dims = ["snapshot", component]
            coords = {"snapshot": n.snapshots, component: [unit_block["name"]]}
        else:
            # single value per unit (or param vector not time-based)
            dims = [f"{component}-ext"]
            coords = {f"{component}-ext": [unit_block["name"]]}
    
    # Case C: 2-D (accept (T,1) or (1,T))
    elif value.ndim == 2:
        T = len(n.snapshots)
        if value.shape == (T, 1):
            dims = ["snapshot", component]
            coords = {"snapshot": n.snapshots, component: [unit_block["name"]]}
        elif value.shape == (1, T):
            value = value.T
            dims = ["snapshot", component]
            coords = {"snapshot": n.snapshots, component: [unit_block["name"]]}
        else:
            raise ValueError(f"Unsupported shape for variable {key}: {value.shape}")
    
    else:
        raise ValueError(f"Unsupported ndim for variable {key}: {value.ndim}")

    var_name = f"{component}-{key}"
    return value, dims, coords, var_name



def block_to_dataarrays(n, unit_name, unit_block, component, config) -> dict:
    """
    Transforms a unit block into a dictionary of DataArrays.
    """
    attr_name = f"{unit_block['block']}_inverse"
    converted_dict = {}
    normalized_keys = {normalize_key(k): k for k in unit_block.keys()}

    if hasattr(config, attr_name):
        unitblock_parameters = getattr(config, attr_name)
    else:
        print(f"Block {unit_block['block']} not yet implemented")
        return {}

    df = getattr(n, config.component_mapping[component])

    for key, func in unitblock_parameters.items():
        if callable(func):
            value = evaluate_function(func, normalized_keys, unit_block, df)
            if isinstance(value, np.ndarray) and value.ndim == 2 and all(dim > 1 for dim in value.shape):
                value = value.sum(axis=0)
            value, dims, coords, var_name = dataarray_components(n, value, component, unit_block, key)
            converted_dict[var_name] = xr.DataArray(value, dims=dims, coords=coords, name=var_name)

    return converted_dict

