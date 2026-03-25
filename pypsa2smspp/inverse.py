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
    Build xarray dims/coords compliant with PyPSA 1.0 assign_solution():
    - static per-unit: dims -> ('name',)
    - time series:     dims -> ('snapshot','name')
    """
    # Unmask masked arrays; use NaN for masked scalars
    if isinstance(value, np.ma.MaskedArray):
        value = value.filled(np.nan)

    value = np.asarray(value)
    unit_name = unit_block["name"]

    if value.ndim == 0:
        # scalar -> one value for this unit: ('name',)
        value = value.reshape(1)
        dims = ["name"]
        coords = {"name": [unit_name]}

    elif value.ndim == 1:
        if len(value) == len(n.snapshots):
            # (T,) -> ('snapshot','name')
            value = value[:, np.newaxis]  # (T,1)
            dims = ["snapshot", "name"]
            coords = {"snapshot": n.snapshots, "name": [unit_name]}
        else:
            # vector but not time-based -> treat as per-unit static ('name',)
            # (se è un vettore con più elementi non-temporali, somma o valida prima)
            value = np.array([value.sum()]) if value.size > 1 else value.reshape(1)
            dims = ["name"]
            coords = {"name": [unit_name]}

    elif value.ndim == 2:
        T = len(n.snapshots)
        if value.shape == (T, 1):
            dims = ["snapshot", "name"]
            coords = {"snapshot": n.snapshots, "name": [unit_name]}
        elif value.shape == (1, T):
            value = value.T
            dims = ["snapshot", "name"]
            coords = {"snapshot": n.snapshots, "name": [unit_name]}
        else:
            raise ValueError(f"Unsupported shape for variable {key}: {value.shape}")
    else:
        raise ValueError(f"Unsupported ndim for variable {key}: {value.ndim}")

    var_name = f"{component}-{key}"  # e.g., 'Generator-p', 'Store-e_nom', 'Link-p'
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



def ensure_scenario_dim(da: xr.DataArray, scenario_name: str) -> xr.DataArray:
    """
    Add a scalar scenario dimension to a deterministic-looking DataArray.
    """
    if "scenario" in da.dims:
        return da

    return da.expand_dims(scenario=[scenario_name])


def block_to_dataarrays_stochastic(
    n,
    name,
    unit_block,
    component,
    config,
    problem_structure,
    block_to_dataarrays_func,
):
    """
    Convert one physical unit block with scenario-wise results into PyPSA-like
    xarray DataArrays.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network.
    name : str
        Name of the physical unit block in self.unitblocks.
    unit_block : dict
        Block dictionary with scenario-wise solution under unit_block["scenarios"].
    component : str
        Component type returned by component_definition(...).
    config : object
        Transformation config.
    problem_structure : dict
        Structure metadata containing scenario names.
    block_to_dataarrays_func : callable
        Existing deterministic converter, typically block_to_dataarrays(...).

    Returns
    -------
    dict[str, xr.DataArray]
        Mapping variable name -> DataArray.
    """
    scenario_names = problem_structure.get("scenario_names", [])
    if not scenario_names:
        return {}

    per_variable = {}

    for scenario_name in scenario_names:
        scenario_values = unit_block.get("scenarios", {}).get(scenario_name, {})
        if not scenario_values:
            continue

        # Build a temporary deterministic-style block:
        # same physical metadata, but inject scenario solution at top level
        temp_block = {
            k: v for k, v in unit_block.items() if k != "scenarios"
        }
        temp_block.update(scenario_values)

        scenario_dataarrays = block_to_dataarrays_func(
            n,
            name,
            temp_block,
            component,
            config,
        )

        for var_name, da in scenario_dataarrays.items():
            da = ensure_scenario_dim(da, scenario_name)
            per_variable.setdefault(var_name, []).append(da)

    merged = {}
    for var_name, da_list in per_variable.items():
        if da_list:
            merged[var_name] = xr.concat(da_list, dim="scenario")

    return merged


def broadcast_static_variables_over_scenarios(ds: xr.Dataset, scenario_names) -> xr.Dataset:
    """
    Broadcast static-like variables over scenarios if they do not already
    contain the scenario dimension.

    This is useful for nominal/design variables in stochastic runs.
    """
    if not scenario_names:
        return ds

    updated = {}

    for var_name, da in ds.data_vars.items():
        if "scenario" in da.dims:
            updated[var_name] = da
            continue

        # Heuristic: variables without snapshot dimension are treated as static
        if "snapshot" not in da.dims:
            updated[var_name] = da.expand_dims(scenario=scenario_names)
        else:
            updated[var_name] = da

    return xr.Dataset(updated)