# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 08:12:37 2024

@author: aless
"""

import pandas as pd
import pypsa
import numpy as np
from datetime import datetime 
import re
import xarray as xr
import os
from pypsa2smspp.transformation_config import TransformationConfig
from pysmspp import SMSNetwork, SMSFileType, Variable, Block, SMSConfig
from pypsa2smspp import logger
from copy import deepcopy
import pysmspp
from pathlib import Path
from typing import Union, Mapping, Any

from .constants import conversion_dict, nominal_attrs, renewable_carriers
from .utils import (
    get_param_as_dense,
    is_extendable,
    filter_extendable_components,
    get_bus_idx,
    get_nominal_aliases,
    remove_zero_p_nom_opt_components,
    ucblock_dimensions,
    networkblock_dimensions,
    investmentblock_dimensions,
    hydroblock_dimensions,
    get_attr_name,
    process_dcnetworkblock,
    resolve_param_value,
    get_block_name,
    parse_unitblock_parameters,
    determine_size_type,
    merge_lines_and_links,
    rename_links_to_lines,
    build_store_and_merged_links,
    correct_dimensions,
    explode_multilinks_into_branches,
    add_sectorcoupled_parameters,
    apply_expansion_overrides,
    build_dc_index,
)

from .pip_utils import (
    select_block_mode,
    _build_smspp_paths,
    build_optimize_call_from_cfg,
    load_yaml_config,
    StepTimer,
    step,
    AttrDict
)

from .inverse import (
    component_definition,
    block_to_dataarrays,
    normalize_key,
    evaluate_function,
    dataarray_components,
)
from .io_parser import (
    parse_txt_to_unitblocks,
    assign_design_variables_to_unitblocks,
    prepare_solution,
    split_merged_dcnetworkblocks
)

NP_DOUBLE = np.float64
NP_UINT = np.uint32

DIR = os.path.dirname(os.path.abspath(__file__))
FP_PARAMS = os.path.join(DIR, "data", "smspp_parameters.xlsx")


class Transformation:
    """
    Transformation class for converting the components of a PyPSA energy network into unit blocks.
    In particular, these are ready to be implemented in SMS++

    The class takes as input a PyPSA network.
    It reads the specified network components and converts them into a dictionary of unit blocks (`unitblocks`).
    
    Attributes:
    ----------
    unitblocks : dict
        Dictionary that holds the parameters for each unit block, organized by network components.
    
    IntermittentUnitBlock_parameters : dict
        Parameters for an IntermittentUnitBlock, like solar and wind turbines.
        The values set to a float number are absent in Pypsa, while lambda functions are used to get data from
        Pypa DataFrames
    
    ThermalUnitBlock_parameters : dict
        Parameters for a ThermalUnitBlock
    """

    def __init__(self, config: Union[str, Path, Mapping[str, Any], AttrDict]):
        """
        Initializes the Transformation class.

        Parameters:
        ----------
        
        n : PyPSA Network
            PyPSA energy network object containing components such as generators and storage units.

        Methods:
        ----------
        init : Start the workflow of the class
        
        """
        
        # Class with conversion dicts for PyPSA-SMS++
        config_conv = TransformationConfig()
        # Work on a private copy so we can safely mutate dicts
        self.config = deepcopy(config_conv)
        
        # Config file
        if isinstance(config, (str, Path)):
            self.cfg = load_yaml_config(config, as_attrdict=True)
        elif isinstance(config, Mapping):
            # Accept dict/AttrDict directly
            self.cfg = config if isinstance(config, AttrDict) else AttrDict(config)
        else:
            raise TypeError(
                "config must be a path (str/Path) or a mapping (dict/AttrDict); "
                f"got {type(config).__name__}"
            )

        
        # Attribute for unit blocks
        self.unitblocks = dict()
        self.networkblock = dict()
        self.investmentblock = {'Blocks': list()}
        
        
        self.dimensions = dict()

        # SMS
        self.sms_network = None
        self.result = None
 
        
##################################################################################################
####################### Pipeline #################################################################
##################################################################################################
    
    def run(self, n, verbose: bool = True):
        # Keep timings accessible after the run
        self.timer = StepTimer()
    
        with step(self.timer, "direct", verbose=verbose):
            self.direct(n)
    
        with step(self.timer, "convert_to_blocks", verbose=verbose):
            self.sms_network = self.convert_to_blocks()
    
        with step(self.timer, "optimize", verbose=verbose, extra={"mode": "auto"}):
            self.optimize()
    
        with step(self.timer, "parse_solution_to_unitblocks", verbose=verbose):
            self.parse_solution_to_unitblocks(self.result.solution, n)
    
        with step(self.timer, "inverse_transformation", verbose=verbose):
            self.inverse_transformation(self.result.objective_value, n)
    
        if verbose:
            self.timer.print_summary()
    
        return n
    
    
    def direct(self, n):
        """
        Direct transformation PyPSA -> internal unitblocks.
        """
    
        # Explicit network patching
        max_hours = self.cfg["transformation"].get("max_hours_stores", None)
        if max_hours is not None and "stores" in dir(n):
            n.stores["max_hours"] = max_hours
    
        # --- your existing logic ---
        self.read_excel_components() # 1
        self.add_dimensions(n) # 2
        self.iterate_components(n) # 3
        self.add_demand(n) # 4
        self.lines_links() # 5


    
    ### 1 ###
    def read_excel_components(self, fp=FP_PARAMS):
        """
        Reads Excel file for size and type of SMS++ parameters. Each sheet includes a class of components

        Returns:
        ----------
        all_sheets : dict
            Dictionary where keys are sheet names and values are DataFrames containing 
            data for each UnitBlock type (or lines).
        """
        self.smspp_parameters = pd.read_excel(fp, sheet_name=None, index_col=0)
    
 
    ### 2 ###          
    def add_dimensions(self, n):
        """
        Sets the .dimensions attribute with UCBlock, NetworkBlock, InvestmentBlock, HydroBlock dimensions.
        """
        self.dimensions['UCBlock'] = ucblock_dimensions(n)
        self.dimensions['NetworkBlock'] = networkblock_dimensions(n, self.cfg.transformation.expansion_ucblock)
        self.dimensions['InvestmentBlock'] = investmentblock_dimensions(n, self.cfg.transformation.expansion_ucblock, nominal_attrs)
        self.dimensions['HydroUnitBlock'] = hydroblock_dimensions()
        
        
        
    ### 3 ###
    def iterate_components(self, n):
        """
        Iterates over the network components and adds them as unit blocks.
        """
        
        # ------------- Preprocessing ----------------
        # Probably useful to group this part as 'preprocessing' as it is independent from the rest
        generator_node = []
        investment_meta = {"Blocks": [], "index_extendable": [], "asset_type": [], 'design_lines': []}
        unitblock_index = 0
        lines_index = 0
        self._dc_names = []
        self._dc_types = []
    
        
        stores_df, links_merged_df, self.dimensions['NetworkBlock']['merged_links_ext'] = build_store_and_merged_links(
            n, merge_links=self.cfg.transformation.merge_links, logger=logger)
        
        links_before = links_merged_df.copy()
        
        if "bus2" in n.links.columns and bool((n.links.bus2.notna() & (n.links.bus2.astype(str).str.strip() != "")).any()):
            # hyper alle linee
            n.lines["hyper"] = np.arange(0, len(n.lines), dtype=int)
            links_after, self.networkblock['efficiencies'], self.dimensions['NetworkBlock']['NumberBranches'], self.dimensions['NetworkBlock']['NumberBranches_ext'] = explode_multilinks_into_branches(links_merged_df, len(n.lines), logger=logger)
            self.networkblock["max_eff_len"] = max((len(v) for v in self.networkblock['efficiencies'].values()), default=1)
            add_sectorcoupled_parameters(self.config.Lines_parameters, self.config.Links_parameters, self.config.DCNetworkBlock_links_inverse, self.networkblock['max_eff_len'])
        else:
            links_after = links_merged_df.copy()
            # assicura colonne per coerenza (no split): un solo branch per link
            if "hyper" not in links_after.columns:
                links_after["hyper"] = np.arange(len(n.lines), len(n.lines) + len(links_after), dtype=int)
            if "is_primary_branch" not in links_after.columns:
                links_after["is_primary_branch"] = True
        
        correct_dimensions(self.dimensions, stores_df, links_merged_df, n, self.cfg.transformation.expansion_ucblock)
        
        self._dc_index = build_dc_index(n, links_before, links_after)
        
        # TODO remove when necessary
        self._dc_names  = list(self._dc_index['physical']['names'])
        self._dc_types  = list(self._dc_index['physical']['types'])


        if self.cfg.transformation.get("expansion_ucblock", False):
            apply_expansion_overrides(self.config.IntermittentUnitBlock_parameters, self.config.BatteryUnitBlock_store_parameters, self.config.IntermittentUnitBlock_inverse, self.config.BatteryUnitBlock_inverse, self.config.InvestmentBlock_parameters)
        
        # ------------- Main loop over components ----------------
        
        # Iterate in the same order as before
        for components in n.components[["Generator", "Store", "StorageUnit", "Line", "Link"]]:

            if components.empty:
                continue
    
            # --- CHANGED: pick the right dataframe per component ---
            # TODO build a proper definition to define the DataFrame
            if components.list_name == "stores":
                components_df = stores_df
                components_t = components.dynamic
            elif components.list_name == "links":
                components_df = links_after
                components_t = components.dynamic
            else:
                components_df = components.static
                components_t = components.dynamic
    
            components_type = components.list_name
    
            use_investmentblock = (
                not self.cfg.transformation.get("expansion_ucblock", False)
                or components_type in ["lines", "links"]
            )

            if use_investmentblock:
                df_investment = self.add_InvestmentBlock(n, components_df, components.name)
    
            # Lines and Links path unchanged
            if components_type in ["lines", "links"]:
                self._dc_names.extend(list(components_df.index))
                self._dc_types.extend(
                    ["line" if components_type == "lines" else "link"] * len(components_df)
                )
                get_bus_idx(
                    n,
                    components_df,
                    [components_df.bus0, components_df.bus1],
                    ["start_line_idx", "end_line_idx"]
                )
    
                attr_name = get_attr_name(components.name, None, renewable_carriers)
                self.add_UnitBlock(attr_name, components_df, components_t, components.name, n)
    
                unitblock_index, lines_index = process_dcnetworkblock(
                    components_df,
                    components.name,
                    investment_meta,
                    unitblock_index,
                    lines_index,
                    df_investment,
                    nominal_attrs,
                )
                continue
    
            # StorageUnits path unchanged (special hydro/PHS handling)
            elif components_type == "storage_units":
                get_bus_idx(n, components_df, components_df.bus, "bus_idx")
                for bus, carrier in zip(components_df["bus_idx"].values, components_df["carrier"]):
                    if carrier in ["hydro", "PHS"]:
                        generator_node.extend([bus] * 2)
                    else:
                        generator_node.append(bus)
    
            # Generators / Stores
            else:
                get_bus_idx(n, components_df, components_df.bus, "bus_idx")
                generator_node.extend(components_df["bus_idx"].values)
    
            # iterate each component one by one (unchanged)
            for component in components_df.index:
                carrier = components_df.loc[component].carrier if "carrier" in components_df.columns else None
                attr_name = get_attr_name(components.name, carrier, renewable_carriers)
    
                self.add_UnitBlock(
                    attr_name,
                    components_df.loc[[component]],
                    components_t,
                    components.name,
                    n,
                    component,
                    unitblock_index
                )
    
                if is_extendable(components_df.loc[[component]], components.name, nominal_attrs):
                    investment_meta["index_extendable"].append(unitblock_index)
                    investment_meta["Blocks"].append(f"{attr_name.split('_')[0]}_{unitblock_index}")
                    investment_meta["asset_type"].append(0)
    
                unitblock_index += 1
    
        # finalize (unchanged)
        self.networkblock['Design'] = self.investmentblock.copy()
        self.networkblock['Design']['DesignLines'] = {
            "value": np.array(investment_meta["design_lines"]),
            "type": "uint",
            "size": ("NumberDesignLines")
        }
        
        self.generator_node = {
            "name": "GeneratorNode",
            "type": "int",
            "size": ("NumberElectricalGenerators",),
            "value": generator_node
        }
        self.investmentblock["Blocks"] = investment_meta["Blocks"]
        self.investmentblock["Assets"] = {
            "value": np.array(investment_meta["index_extendable"]),
            "type": "uint",
            "size": "NumAssets"
        }
        self.investmentblock["AssetType"] = {
            "value": np.array(investment_meta["asset_type"]),
            "type": "int",
            "size": "NumAssets"
        }

    ### 4 ###  
    def add_InvestmentBlock(self, n, components_df, components_type):
        """
        Parse and add the InvestmentBlock to self.investmentblock.
        
        This method filters extendable components, renames columns for
        compatibility, and updates the InvestmentBlock variable values.
        """
        # filter extendable elements
        components_df = filter_extendable_components(components_df, components_type, nominal_attrs)
    
        # rename for compatibility with InvestmentBlock expected names
        aliases = get_nominal_aliases(components_type, nominal_attrs)
        df_alias = components_df.rename(columns=aliases)
    
        # store temporary dimension info
        if "Fake_dimension" not in self.dimensions:
            self.dimensions["Fake_dimension"] = {}
        self.dimensions["Fake_dimension"]["NumAssets_partial"] = len(df_alias)
    
        attr_name = "InvestmentBlock_parameters"
        unitblock_parameters = getattr(self.config, attr_name)
    
        for key, func in unitblock_parameters.items():
            param_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            args = [df_alias.get(param) for param in param_names]
            value = func(*args)
    
            variable_type, variable_size = determine_size_type(
                self.smspp_parameters,
                self.dimensions,
                conversion_dict,
                attr_name,
                key,
                value
            )
            
            if variable_size in [('NumberDesignLines_lines',), ('NumberDesignLines_links',)]:
                variable_size = ('NumberDesignLines',)
            
            self.investmentblock.setdefault(
                key,
                {"value": np.array([]), "type": variable_type, "size": variable_size}
            )
    
            if self.investmentblock[key]["value"].size == 0:
                self.investmentblock[key]["value"] = value
            else:
                self.investmentblock[key]["value"] = np.concatenate(
                    [self.investmentblock[key]["value"], value]
                )
    
        return df_alias
    
    ### 5 ###
    def add_UnitBlock(self, attr_name, components_df, components_t, components_type, n, component=None, index=None):
        """
        Adds a unit block to the `unitblocks` dictionary for a given component.

        Parameters:
        ----------
        attr_name : str
            Attribute name containing the unit block parameters (Intermittent or Thermal).
        
        components_df : DataFrame
            DataFrame containing information for a single component.
            For example, n.generators.loc['wind']

        components_t : DataFrame
            Temporal DataFrame (e.g., snapshot) for the component.
            For example, n.generators_t

        Sets:
        --------
        self.unitblocks[components_df.name] : dict
            Dictionary of transformed parameters for the component.
        """
        
        if hasattr(self.config, attr_name):
            unitblock_parameters = getattr(self.config, attr_name)
        else:
            print("Block not yet implemented") # TODO: Replace with logger
            
        converted_dict = parse_unitblock_parameters(
            attr_name,
            unitblock_parameters,
            self.smspp_parameters,
            self.dimensions,
            conversion_dict,
            components_df,
            components_t,
            n,
            components_type,
            component
        )
        
        name = get_block_name(attr_name, index, components_df)
        
        if attr_name in ['Lines_parameters', 'Links_parameters']:
            self.networkblock[name] = {"block": 'Lines', "variables": converted_dict}
        else:
            nom = nominal_attrs[components_type]
            ext = components_df[f"{nom}_extendable"].iloc[0]
            design_key = (
                "DesignVariable" if not self.cfg.transformation.expansion_ucblock else
                ("IntermittentDesign" if "IntermittentUnitBlock" in name else
                "BatteryDesign" if "BatteryUnitBlock" in name else
                "DesignVariable")   # fallback
            )
            self.unitblocks[name] = {"name": components_df.index[0],"enumerate": f"UnitBlock_{index}" ,"block": attr_name.split("_")[0], design_key: components_df[nom].values, "Extendable":ext, "variables": converted_dict}
        
        if attr_name == 'HydroUnitBlock_parameters':
            dimensions = self.dimensions['HydroUnitBlock']
            self.dimensions['UCBlock']["NumberElectricalGenerators"] += 1*dimensions["NumberReservoirs"] 
            
            self.unitblocks[name]['dimensions'] = dimensions
        
    ### 6 ###
    def add_demand(self, n):
       demand = n.loads_t.p_set.rename(columns=n.loads.bus)
       # To be sure index of demand matches with buses (probably useless since SMS++ does not care)
       demand = demand.T.reindex(n.buses.index).fillna(0.)
       self.demand = {'name': 'ActivePowerDemand', 'type': 'float', 'size': ("NumberNodes", "TimeHorizon"), 'value': demand}        
    
    ### 7 ###
    def lines_links(self):
        """
        Merge or rename network blocks to ensure a single 'Lines' block for SMS++.
    
        Explanation:
        ------------
        SMS++ currently only supports DCNetworkBlock for electrical lines.
        Links are interpreted as lines with efficiencies < 1 and merged
        into the Lines block. If no true Lines exist, Links are renamed to Lines.
        """
        if (
            self.dimensions["NetworkBlock"]["Lines"] > 0
            and self.dimensions["NetworkBlock"]["Links"] > 0
        ):
            merge_lines_and_links(self.networkblock)
        elif (
            self.dimensions["NetworkBlock"]["Lines"] == 0
            and self.dimensions["NetworkBlock"]["Links"] > 0
        ):
            rename_links_to_lines(self.networkblock)


            
###########################################################################################################################
############ PARSE OUPUT SMS++ FILE ###################################################################
###########################################################################################################################
    
    
    
    def parse_solution_to_unitblocks(self, solution, n):
        """
        Parse a loaded SMS++ solution structure and populate self.unitblocks with unit-level data.
    
        This function extracts the contents of UnitBlock_i from solution.blocks['Solution_0'],
        and stores them into the corresponding entries of self.unitblocks. If transmission lines
        are present, it also parses the NetworkBlock series and generates synthetic UnitBlocks
        for each line or link.
    
        Parameters
        ----------
        solution : SMSNetwork
            An in-memory SMS++ solution object (already parsed from file).
        n : pypsa.Network
            The PyPSA network object used to retrieve line and link names.
    
        Returns
        -------
        solution_data : dict
            A dictionary of blocks parsed from the SMSNetwork object (mainly for inspection).
        """
        num_units = self.dimensions['UCBlock']['NumberUnits']
        solution_data = {}
    
        solution_0 = solution.blocks['Solution_0']
        has_investment = "DesignVariables" in solution_0.variables
    
        if has_investment:
            inner_solution = solution_0.blocks["InnerSolution"]
            solution_data["UCBlock"] = inner_solution
        else:
            inner_solution = solution_0
            solution_data["UCBlock"] = solution_0
    
        if self.dimensions['UCBlock']['NumberLines'] > 0:
            self.parse_networkblock_lines(inner_solution)
            self.generate_line_unitblocks(n)
    
        if not hasattr(self, "unitblocks"):
            raise ValueError("self.unitblocks must be initialized before parsing the solution.")
    
        for i in range(num_units):
            block_key = f"UnitBlock_{i}"
            block = inner_solution.blocks[block_key]
            solution_data[block_key] = block
    
            matching_key = next(
                (key for key in self.unitblocks if key.endswith(f"_{i}")),
                None
            )
    
            if matching_key is None:
                raise KeyError(f"No matching key found in self.unitblocks for UnitBlock_{i}")
    
            for var_name, var_obj in block.variables.items():
                self.unitblocks[matching_key][var_name] = var_obj.data

    
        # Assign design variables if investment
        if has_investment:
            design_vars = solution_0.variables["DesignVariables"].data
            block_names = self.investmentblock.get("Blocks", [])
            assign_design_variables_to_unitblocks(self.unitblocks, block_names, design_vars)
       
        split_merged_dcnetworkblocks(self.unitblocks)
        return solution_data
    
    
    
    def parse_networkblock_lines(self, solution_0):
        """
        Parse line-level time series from an SMS++ solution.
    
        If Solution_0 contains a single 'NetworkBlock' already aggregated across time,
        read variables directly. Otherwise, fall back to stacking 'NetworkBlock_i'
        (one per snapshot). Result is stored in self.networkblock['Lines'][var] with
        shape (time, element).
        """
    
        vars_of_interest = ("FlowValue", "NodeInjection")
    
        blocks = solution_0.blocks
    
        # --- Case 1: new format, single aggregated block -------------------------
        if "NetworkBlock" in blocks:
            block = blocks["NetworkBlock"]
            
            if "DesignNetworkBlock_0" in block.blocks:
                block = block.blocks["DesignNetworkBlock_0"]
                vars_of_interest = vars_of_interest + ("DesignValue",)
    
            for var in vars_of_interest:
                if var not in block.variables:
                    raise KeyError(f"{var} not found in NetworkBlock")
    
                arr = block.variables[var].data  # expected shape: (time, element)
    
                # Sanity: make sure we end up with 2D (time, element)
                if arr.ndim == 1:
                    # If ndim==1, assume it is (element,) repeated over a single time
                    arr = arr[np.newaxis, :]
    
                if arr.ndim != 2:
                    raise ValueError(
                        f"Unexpected shape for {var} in NetworkBlock: {arr.shape} (expected 2D)"
                    )
    
                self.networkblock["Lines"][var] = arr
    
            return  # done
    
        # --- Case 2: legacy format, multiple NetworkBlock_i ----------------------
        # Collect and sort by numeric suffix to be safe w.r.t. missing/extra blocks
        nb_keys = [
            k for k in blocks.keys()
            if k.startswith("NetworkBlock_") and k[len("NetworkBlock_"):].isdigit()
        ]
        if not nb_keys:
            raise KeyError("No 'NetworkBlock' or 'NetworkBlock_i' blocks found in Solution_0")
    
        nb_keys.sort(key=lambda k: int(k.split("_")[-1]))
    
        # Stack per-time blocks into (time, element)
        variable_first_lengths = {v: None for v in vars_of_interest}
        stacked = {v: [] for v in vars_of_interest}
    
        for k in nb_keys:
            block = blocks[k]
            for var in vars_of_interest:
                if var not in block.variables:
                    raise KeyError(f"{var} not found in {k}")
                arr = block.variables[var].data
    
                # Each per-time block is expected to be 1D (element,) or 2D (1, element)
                if arr.ndim == 2 and arr.shape[0] == 1:
                    arr = arr[0]
                if arr.ndim != 1:
                    raise ValueError(f"Unexpected shape for {var} in {k}: {arr.shape} (expected 1D)")
    
                # Track element dimension consistency
                if variable_first_lengths[var] is None:
                    variable_first_lengths[var] = arr.shape[0]
                elif variable_first_lengths[var] != arr.shape[0]:
                    raise ValueError(
                        f"Inconsistent element size for {var}: "
                        f"expected {variable_first_lengths[var]}, got {arr.shape[0]} in {k}"
                    )
    
                stacked[var].append(arr)
    
        for var, lst in stacked.items():
            # Shape -> (time, element)
            self.networkblock["Lines"][var] = np.stack(lst, axis=0)



    
    def generate_line_unitblocks(self, n):
        """
        Generate synthetic UnitBlocks for lines and links based on combined FlowValue data.
    
        This function splits the FlowValue and DualCost arrays into individual unitblocks.
        Each block is labeled as 'DCNetworkBlock_lines' or 'DCNetworkBlock_links' based on type.
    
        Parameters
        ----------
        n : pypsa.Network
            PyPSA network object containing line and link names.
    
        Raises
        ------
        ValueError
            If array dimensions are inconsistent.
        """
        flow_matrix = self.networkblock['Lines']['FlowValue']
        if 'DesignValue' in self.networkblock['Lines'].keys():
           design_matrix = self.networkblock['Lines']['DesignValue'] 
        else:
           design_matrix = 0
    
        names, types = self.prepare_dc_unitblock_info(n)
        
        links_effs = self.networkblock.get("efficiencies", {})
        max_eff_len = self.networkblock.get("max_eff_len", 1)
    
        if len(names) != flow_matrix.shape[1]:
            raise ValueError("Mismatch between total network components and columns in FlowValue")
    
        current_index = len(self.unitblocks)
        n_elements = flow_matrix.shape[1]
        designlines = self.networkblock['Design']['DesignLines']['value']
        i_ext = 0
    
        for i in range(n_elements):
            block_index = current_index + i
            unitblock_name = f"DCNetworkBlock_{block_index}"
            block_type = types[i]
            block_label = "DCNetworkBlock_links" if block_type == "link" else "DCNetworkBlock_lines"
            
            if i in designlines:
                designvariable = design_matrix[:, i_ext] if isinstance(design_matrix, np.ndarray) else self.networkblock['Lines']['variables']['MaxPowerFlow']['value'][i]
                i_ext += 1
            else:
                designvariable = self.networkblock['Lines']['variables']['MaxPowerFlow']['value'][i]

            entry = {
                "enumerate": f"UnitBlock_{block_index}",
                "block": block_label,
                "name": names[i],
                "FlowValue": flow_matrix[:, i],
                # "DualCost": dual_matrix[:, i],
                "DesignVariable": designvariable,
            }
            
            if block_type == "link":
                # Add value of efficiency
                eff_list = links_effs.get(names[i], None)
                if eff_list is None:
                    # If not present, create [1.0, 0.0, ..., 0.0] max_eff_len long (fallback)
                    eff_list = [1.0] + [0.0] * max(0, max_eff_len - 1)
                entry["Efficiencies"] = eff_list
            
            self.unitblocks[unitblock_name] = entry
        

    def prepare_dc_unitblock_info(self, n):
        """
        Return the (names, types) for DCNetworkBlock unitblocks.
        Prefer the 'physical' view from self._dc_index (NumberLines),
        which matches FlowValue columns in NetworkBlock.
        """
        if hasattr(self, "_dc_index") and self._dc_index and 'physical' in self._dc_index:
            names = list(self._dc_index['physical']['names'])
            types = list(self._dc_index['physical']['types'])
            return names, types
    
        # Fallback legacy (se proprio manca il registry)
        num_lines = self.dimensions['NetworkBlock']['Lines']
        num_links = self.dimensions['NetworkBlock']['Links']
    
        line_names = list(n.lines.index)
        link_names = list(n.links.index)
    
        if len(line_names) != num_lines:
            raise ValueError(
                f"Mismatch between dimensions and n.lines "
                f"(expected {num_lines}, got {len(line_names)})"
            )
        if len(link_names) != num_links:
            raise ValueError(
                f"Mismatch between dimensions and n.links "
                f"(expected {num_links}, got {len(link_names)})"
            )
    
        names = line_names + link_names
        types = (['line'] * num_lines) + (['link'] * num_links)
        return names, types




###########################################################################################################################
############ INVERSE TRANSFORMATION INTO XARRAY DATASET ###################################################################
###########################################################################################
   
    
    def inverse_transformation(self, objective_smspp, n):
        '''
        Performs the inverse transformation from the SMS++ blocks to xarray object.
        The xarray wll be converted in a solution type Linopy file to get n.optimize()
    
        This method initializes the inverse process and sets inverse-conversion dicts
    
        Parameters
        ----------
        ojective_smspp: float
            The objective function of the smspp problem
        n : pypsa.Network
            A PyPSA network instance from which the data will be extracted.
        '''
        all_dataarrays = self.iterate_blocks(n)
        self.ds = xr.Dataset(all_dataarrays)
        
        prepare_solution(n, self.ds, objective_smspp)
        
        n.optimize.assign_solution()
        # n.optimize.assign_duals(n) # Still doesn't work
        
        n._multi_invest = 0
        n.calculate_dependent_values()
        n.optimize.post_processing()
        n._objective_constant = 0
        
        
    
    def iterate_blocks(self, n):
        '''
        Iterates over all unit blocks in the model and constructs their corresponding xarray.Dataset objects.
        
        For each unit block, this method determines the component type, generates DataArrays using
        `block_to_dataarrays`, and appends them to a list of datasets. At the end, all datasets are
        merged into a single xarray.Dataset.
        
        Parameters
        ----------
        n : pypsa.Network
            The PyPSA network from which values are extracted.
        
        Returns
        -------
        xr.Dataset
            A dataset containing all DataArrays from the unit blocks.
        '''
        datasets = []
    
        for name, unit_block in self.unitblocks.items():
            component = component_definition(n, unit_block)
            dataarrays = block_to_dataarrays(n, name, unit_block, component, self.config)
            if dataarrays:  # No emptry dicts
                ds = xr.Dataset(dataarrays)
                datasets.append(ds)
    
        # Merge in a single dataset
        # keep current behavior explicitly and avoid FutureWarnings
        return xr.merge(datasets, join="outer", compat="no_conflicts")





#########################################################################################
######################## Conversion with PySMSpp ########################################
#########################################################################################
    
    ## Create SMSNetwork
    def convert_to_blocks(self):
        """
        Builds the SMSNetwork block hierarchy depending on whether
        the problem is an investment (NumAssets > 0) or only unit commitment.
    
        Sets:
        -------
        self.sms_network : SMSNetwork
            The built SMSNetwork structure.
    
        Returns
        -------
        SMSNetwork
            The network with all blocks added.
        """
    
        # -----------------
        # Initialize empty SMSNetwork
        # -----------------
        sn = SMSNetwork(file_type=SMSFileType.eBlockFile)
        master = sn
        index_id = 0
    
        # -----------------
        # Check if investment problem
        # -----------------
        if (not self.cfg.transformation.expansion_ucblock) and (self.dimensions['InvestmentBlock']['NumAssets'] > 0):
             name_id = 'InvestmentBlock'
             sn = self.convert_to_investmentblock(master, index_id, name_id)
    
             # InnerBlock for UC is inside InvestmentBlock
             master = sn.blocks[name_id]
             name_id = 'InnerBlock'
             index_id += 1
        else:
            name_id = 'Block_0'
            self.cfg.run.mode = 'ucblock'
        
        # name_id = 'Block_0'
    
        # -----------------
        # Add UCBlock (always present)
        # -----------------
        self.convert_to_ucblock(master, index_id, name_id)
    
        # Save final
        self.sms_network = sn
        return sn
    
    def convert_to_investmentblock(self, master, index_id, name_id):
        """
        Adds an InvestmentBlock to the SMSNetwork, including the
        investment-related variables.
    
        Parameters
        ----------
        master : SMSNetwork
            The root SMSNetwork object
        index_id : int
            ID for block naming
        name_id : str
            Name for the InvestmentBlock
            
        Returns
        -------
        SMSNetwork
            The updated SMSNetwork with the InvestmentBlock added.
        """
    
        # -----------------
        # InvestmentBlock dimensions
        # -----------------
        kwargs = self.dimensions['InvestmentBlock']
    
        # -----------------
        # Add variables from investmentblock dictionary
        # -----------------
        for name, variable in self.investmentblock.items():
            if name != 'Blocks':
                kwargs[name] = Variable(
                    name,
                    variable['type'],
                    variable['size'],
                    variable['value']
                )
    
        # -----------------
        # Register block
        # -----------------
        master.add(
            "InvestmentBlock",
            name_id,
            id=f"{index_id}",
            **kwargs
        )
        return master
  
    def convert_to_ucblock(self, master, index_id, name_id):
        """
        Converts the unit blocks into a UCBlock (or InnerBlock) format.
    
        Parameters
        ----------
        master : SMSNetwork
            The SMSNetwork object to which to attach the UCBlock.
        index_id : int
            The block id.
        name_id : str
            The block name ("UCBlock" or "InnerBlock").
    
        Returns
        -------
        SMSNetwork
            The SMSNetwork with the UCBlock added.
        """
    
        # UCBlock dimensions (NumberUnits, NumberNodes, etc.)
        ucblock_dims = self.dimensions['UCBlock']
    
        # -----------------
        # Demand (load)
        # -----------------
        demand_var = {
            self.demand['name']: Variable(
                self.demand['name'],
                self.demand['type'],
                self.demand['size'],
                self.demand['value']
            )
        }
    
        # -----------------
        # GeneratorNode
        # -----------------
        gen_node_var = {
            self.generator_node['name']: Variable(
                self.generator_node['name'],
                self.generator_node['type'],
                self.generator_node['size'],
                self.generator_node['value']
            )
        }
    
        # -----------------
        # Network lines (Lines block only, merged with Links if needed)
        # -----------------
        line_vars = {}
        if ucblock_dims.get("NumberLines", 0) > 0:
            for var_name, var in self.networkblock['Lines']['variables'].items():
                line_vars[var_name] = Variable(
                    var_name,
                    var['type'],
                    var['size'],
                    var['value']
                )
    
        # -----------------
        # Assemble all kwargs
        # -----------------
        block_kwargs = {
            **ucblock_dims,
            **demand_var,
            **gen_node_var,
            **line_vars
        }
    
        # -----------------
        # Add UCBlock itself
        # -----------------
        master.add(
            "UCBlock",
            name_id,
            id=f"{index_id}",
            **block_kwargs
        )
    
        # -----------------
        # Add all UnitBlocks inside UCBlock
        # -----------------
        for ub_name, unit_block in self.unitblocks.items():
            ub_kwargs = {}
            for var_name, var in unit_block['variables'].items():
                ub_kwargs[var_name] = Variable(
                    var_name,
                    var['type'],
                    var['size'],
                    var['value']
                )
    
            # Add also any special dimensions
            if 'dimensions' in unit_block:
                for dim_name, dim_value in unit_block['dimensions'].items():
                    ub_kwargs[dim_name] = dim_value
    
            # create Block
            unit_block_obj = Block().from_kwargs(
                block_type=unit_block['block'],
                **ub_kwargs
            )
    
            # attach to UCBlock
            master.blocks[name_id].add_block(unit_block['enumerate'], block=unit_block_obj)
            
        # -----------------
        # Optionally add DesignNetworkBlock (only in expansion_ucblock mode)
        # -----------------
        self.convert_to_designnetworkblock(master, name_id)
    
        # -----------------
        # Done
        # -----------------
        return master
    
    
    def convert_to_designnetworkblock(self, master, ucblock_name):
        """
        Optionally adds a DesignNetworkBlock inside the UCBlock, used when
        expansion_ucblock is active and design lines are present.
    
        Parameters
        ----------
        master : SMSNetwork
            The SMSNetwork object containing the UCBlock.
        ucblock_name : str
            The name_id of the UCBlock inside master.blocks.
        """
    
        # Condition: only in expansion-ucblock mode AND if we actually have design lines
        if not self.cfg.transformation.get("expansion_ucblock", False):
            return
    
        num_design_lines = (
            self.dimensions
            .get("InvestmentBlock", {})
            .get("NumberDesignLines", 0)
        )
        if num_design_lines <= 0:
            return
    
        # Safety: if we do not have design information, just skip
        design_block_def = self.networkblock.get("Design")
        if design_block_def is None:
            return
    
        # Build kwargs for the DesignNetworkBlock
        design_kwargs = {}
    
        # Add variables from self.networkblock['Design']
        for var_name, var in design_block_def.items():
            if var_name != 'Blocks':
                design_kwargs[var_name] = Variable(
                    var_name,
                    var["type"],
                    var["size"],
                    var["value"]
                )
    
        # Add dimensions (from investmentblock)
        for dim_name, dim_value in self.dimensions['InvestmentBlock'].items():
            design_kwargs[dim_name] = dim_value
    
    
        # Create the DesignNetworkBlock
        design_block_obj = Block().from_kwargs(
            block_type="DesignNetworkBlock",
            **design_kwargs
        )
    
        # Attach it inside the UCBlock; use a stable id/label for the block
        master.blocks[ucblock_name].add_block(
            "NetworkBlock_0",
            block=design_block_obj
        )


    
    def optimize(self):
        if self.sms_network is None:
            raise ValueError("SMSNetwork not initialized.")
    
        # Decide mode based on cfg + dimensions
        mode = select_block_mode(self.cfg, self.dimensions)  # "ucblock" or "investmentblock"
    
        mode_cfg = getattr(self.cfg.smspp, mode)  # cfg.smspp.ucblock / investmentblock
    
        # Build configfile from template
        configfile = pysmspp.SMSConfig(template=mode_cfg.template)
    
        # Build output paths
        temporary_smspp_file, output_file, solution_file = _build_smspp_paths(self.cfg, mode_cfg.output_prefix)
    
        # Overwrite policy (optional)
        if getattr(self.cfg.io, "overwrite", True) and os.path.exists(solution_file):
            os.remove(solution_file)
    
        # Build args/kwargs automatically from cfg.smspp.<mode>
        args, kwargs = build_optimize_call_from_cfg(
            self.cfg,
            mode=mode,
            configfile=configfile,
            temporary_smspp_file=temporary_smspp_file,
            output_file=output_file,
            solution_file=solution_file,
        )
    
        # Call the flexible pass-through optimize() you already have,
        # but careful: this is inside optimize() itself, so call sms_network directly here.
        self.result = self.sms_network.optimize(configfile, *args, **kwargs)
        return self.result

    


#############################################################################################
############################## Backup #######################################################
#############################################################################################

    def add_slackunitblock(self):
        index = len(self.unitblocks) 
        
        for bus in range(len(self.demand['value'])):
            self.unitblocks[f"SlackUnitBlock_{index}"] = dict()
            
            slack = self.unitblocks[f"SlackUnitBlock_{index}"]
            
            slack['block'] = 'SlackUnitBlock'
            slack['enumerate'] = f"UnitBlock_{index}"
            slack['name'] = f"slack_variable_bus{bus}"
            slack['variables'] = dict()
            
            slack['variables']['MaxPower'] = dict()
            slack['variables']['ActivePowerCost'] = dict()
            
            slack['variables']['MaxPower']['value'] = self.demand['value'].sum().max() + 10
            slack['variables']['MaxPower']['type'] = 'float'
            slack['variables']['MaxPower']['size'] = ()
            
            slack['variables']['ActivePowerCost']['value'] = 1e5 # €/MWh)
            slack['variables']['ActivePowerCost']['type'] = 'float'
            slack['variables']['ActivePowerCost']['size'] = ()
            
            self.dimensions['UCBlock']['NumberUnits'] += 1
            self.dimensions['UCBlock']['NumberElectricalGenerators'] += 1
            
            self.generator_node['value'].append(bus)
            index += 1
