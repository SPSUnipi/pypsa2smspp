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


class FakeVariable:
    """
    A dummy wrapper used to emulate PyPSA-style model.variable.solution attributes.
    """
    def __init__(self, solution):
        self.solution = solution


def prepare_solution(n, ds: xr.Dataset) -> None:
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
    n.model = type("FakeModel", (), {})()
    n.model.variables = {name: FakeVariable(solution=dataarray) for name, dataarray in ds.items()}

    n.model.parameters = type("FakeParameters", (), {})()
    n.model.parameters.snapshots = xr.DataArray(n.snapshots, dims=["snapshot"])

    n.model.constraints = type("FakeConstraints", (), {})()
    n.model.constraints.snapshots = xr.DataArray(n.snapshots, dims=["snapshot"])

    n.model.objective = type("FakeObjective", (), {})()
    n.model.objective.value = 10000  # arbitrary
