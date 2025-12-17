# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 15:57:30 2025

@author: aless
"""

"""
io_parser.py

This module handles the parsing of SMS++ output files (both .txt and NetCDF formats)
and prepares data structures that can be used to populate the Transformation class 
or to re-assign results into a PyPSA network.

It includes:
- parsing unit blocks from .txt file
- parsing solution objects from SMSNetwork
- conversion of parsed data into xarray or PyPSA structures
"""

import numpy as np
import re
import xarray as xr


def parse_txt_to_unitblocks(file_path: str, unitblocks: dict) -> None:
    """
    Parses an SMS++ textual solution file and populates the unitblocks dictionary.

    Parameters
    ----------
    file_path : str
        Path to the text file.
    unitblocks : dict
        Dictionary of unitblocks to populate with parsed data.
    """
    current_block = None
    current_block_key = None

    with open(file_path, "r") as file:
        for line in file:
            match_time = re.search(r"Elapsed time:\s*([\deE\+\.-]+)\s*s", line)
            if match_time:
                continue  # Skip timing info

            block_match = re.search(r"(ThermalUnitBlock|BatteryUnitBlock|IntermittentUnitBlock|HydroUnitBlock)\s*(\d+)", line)
            if block_match:
                block_type, number = block_match.groups()
                number = int(number)
                current_block = block_type
                current_block_key = f"{block_type}_{number}"
                unitblocks[current_block_key]["block"] = block_type
                unitblocks[current_block_key]["enumerate"] = number
                continue

            match = re.match(r"([\w\s]+?)(?:\s*\[(\d+)\])?\s+=\s+\[([^\]]*)\]", line)
            if match and current_block_key:
                key_base, sub_index, values = match.groups()
                key_base = key_base.strip()
                values_array = np.array([float(x) for x in values.split()])

                if sub_index is not None:
                    sub_index = int(sub_index)
                    if key_base in unitblocks[current_block_key] and not isinstance(unitblocks[current_block_key][key_base], dict):
                        unitblocks[current_block_key][key_base] = {0: unitblocks[current_block_key][key_base]}
                    if key_base not in unitblocks[current_block_key]:
                        unitblocks[current_block_key][key_base] = {}
                    unitblocks[current_block_key][key_base][sub_index] = values_array
                else:
                    unitblocks[current_block_key][key_base] = values_array


def assign_design_variables_to_unitblocks(unitblocks, block_names_investment, design_vars):
    """
    Assigns design variable values to the corresponding unitblocks based on investment block mapping.

    Parameters
    ----------
    unitblocks : dict
        Dictionary of unitblocks.
    block_names_investment : list of str
        List of unitblock names that received investments.
    design_vars : np.ndarray
        Array of design variable values.

    Raises
    ------
    ValueError or KeyError
        If a mismatch in shapes or missing keys occurs.
    """
    if len(design_vars) != len(block_names_investment):
        raise ValueError("Mismatch between design variables and investment blocks")

    for name, value in zip(block_names_investment, design_vars):
        if name not in unitblocks:
            raise KeyError(f"DesignVariable refers to unknown unitblock '{name}'")
        unitblocks[name]["DesignVariable"] = value


def split_merged_dcnetworkblocks(unitblocks, delimiter="__", reuse_index_for_first=True, logger=print):
    """
    Split merged DCNetworkBlock_* entries in a unitblocks dict into two blocks.

    Parameters
    ----------
    unitblocks : dict
        Dictionary of unitblocks (as built by parse_solution_to_unitblocks).
    delimiter : str, default="__"
        String that separates the two original link names inside the merged name.
    reuse_index_for_first : bool, default=True
        If True, the first split block keeps the original index
        (e.g., DCNetworkBlock_7), the second gets a new index.
        If False, the original entry is removed and two new blocks are appended.
    logger : callable, default=print
        Logging function.

    Returns
    -------
    unitblocks : dict
        Modified dictionary with merged blocks split into two entries.
    """
    keys = list(unitblocks.keys())
    candidates = []
    for k in keys:
        if not k.startswith("DCNetworkBlock_"):
            continue
        blk = unitblocks[k]
        name = blk.get("name", "")
        if isinstance(name, str) and delimiter in name:
            candidates.append(k)

    if not candidates:
        logger("[split] No merged DCNetworkBlock entries found; nothing to do.")
        return unitblocks

    # Next fresh index for UnitBlock/DCNetworkBlock
    def _next_index():
        max_idx = -1
        for kk in unitblocks.keys():
            if "_" in kk and kk.split("_")[-1].isdigit():
                max_idx = max(max_idx, int(kk.split("_")[-1]))
        return max_idx + 1

    for k in candidates:
        blk = unitblocks[k]
        merged_name = blk.get("name", "")
        parts = merged_name.split(delimiter)
        if len(parts) != 2:
            logger(f"[split] Skipping '{k}' because name does not split cleanly: {merged_name}")
            continue
        name_ch, name_dis = parts[0].strip(), parts[1].strip()

        flow = blk.get("FlowValue", None)
        if flow is None:
            logger(f"[split] Block '{k}' has no FlowValue; skipping.")
            continue

        flow_charge = np.maximum(flow, 0.0)
        flow_dis    = np.maximum(-flow, 0.0)

        base_charge = dict(blk)
        base_dis    = dict(blk)

        base_charge["name"] = name_ch
        base_charge["FlowValue"] = flow_charge

        base_dis["name"] = name_dis
        base_dis["FlowValue"] = flow_dis

        if reuse_index_for_first:
            idx_first = int(k.split("_")[-1])
            key_first = f"DCNetworkBlock_{idx_first}"
            enum_first = f"UnitBlock_{idx_first}"

            idx_second = _next_index()
            key_second = f"DCNetworkBlock_{idx_second}"
            enum_second = f"UnitBlock_{idx_second}"

            base_charge["enumerate"] = enum_first
            unitblocks[key_first] = base_charge

            base_dis["enumerate"] = enum_second
            unitblocks[key_second] = base_dis
        else:
            del unitblocks[k]
            idx_first = _next_index()
            idx_second = idx_first + 1
            key_first = f"DCNetworkBlock_{idx_first}"
            key_second = f"DCNetworkBlock_{idx_second}"
            base_charge["enumerate"] = f"UnitBlock_{idx_first}"
            base_dis["enumerate"] = f"UnitBlock_{idx_second}"
            unitblocks[key_first] = base_charge
            unitblocks[key_second] = base_dis

        logger(f"[split] '{k}' -> '{name_ch}' + '{name_dis}'")

    return unitblocks


class FakeVariable:
    """
    A dummy wrapper used to emulate PyPSA-style model.variable.solution attributes.
    """
    def __init__(self, solution):
        self.solution = solution


def prepare_solution(n, ds: xr.Dataset, objective_smspp: float) -> None:
    """
    Prepares a fake PyPSA model that wraps the xarray Dataset as a PyPSA-compatible solution.

    Parameters
    ----------
    n : pypsa.Network
        The original PyPSA network.
    ds : xarray.Dataset
        The solution dataset to attach to the network.

    Returns
    -------
    None (modifies n in place)
    """
    n._model = type("FakeModel", (), {})()
    n._model.variables = {name: FakeVariable(solution=dataarray) for name, dataarray in ds.items()}

    n._model.parameters = type("FakeParameters", (), {})()
    n._model.parameters.snapshots = xr.DataArray(n.snapshots, dims=["snapshot"])

    n._model.constraints = type("FakeConstraints", (), {})()
    n._model.constraints.snapshots = xr.DataArray(n.snapshots, dims=["snapshot"])

    n._model.objective = type("FakeObjective", (), {})()
    n._model.objective.value = objective_smspp
