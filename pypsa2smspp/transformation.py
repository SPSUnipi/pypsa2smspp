# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 08:12:37 2024

@author: aless
"""

import pandas as pd
import pypsa
import numpy as np
from datetime import datetime 
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
import re
import numpy as np
import xarray as xr
import os
from pypsa2smspp.transformation_config import TransformationConfig
from pysmspp import SMSNetwork, SMSFileType, Variable, Block, SMSConfig
from pypsa.optimization.optimize import (assign_solution, assign_duals, post_processing)

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
    process_dcnetworkblock
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

    def __init__(self, n, config=TransformationConfig()):
        """
        Initializes the Transformation class.

        Parameters:
        ----------
        
        n : PyPSA Network
            PyPSA energy network object containing components such as generators and storage units.
        max_hours_stores: int
            Max hours parameter for stores, default is 10h. Stores do not have this parameter, but it is required to model them as BatteryUnitBlocks
        
        Methods:
        ----------
        init : Start the workflow of the class
        
        """
        
        # Attribute for unit blocks
        self.unitblocks = dict()
        self.networkblock = dict()
        self.investmentblock = {'Blocks': list()}
        
        self.dimensions = dict()
        
        self.config = config
        
        
        n.stores["max_hours"] = config.max_hours_stores
        
        # Initialize with the parser and network
        remove_zero_p_nom_opt_components(n, nominal_attrs)
        self.read_excel_components()
        self.add_dimensions(n)
        self.iterate_components(n)
        self.add_demand(n)
        self.lines_links()

        # SMS
        self.sms_network = None
        self.result = None


        
    def add_demand(self, n):
        demand = n.loads_t.p_set.rename(columns=n.loads.bus)
        demand = demand.T.reindex(n.buses.index).fillna(0.)
        self.demand = {'name': 'ActivePowerDemand', 'type': 'float', 'size': ("NumberNodes", "TimeHorizon"), 'value': demand}
                
    def add_dimensions(self, n):
        """
        Sets the .dimensions attribute with UCBlock, NetworkBlock, InvestmentBlock, HydroBlock dimensions.
        """
        self.dimensions['UCBlock'] = ucblock_dimensions(n)
        self.dimensions['NetworkBlock'] = networkblock_dimensions(n)
        self.dimensions['InvestmentBlock'] = investmentblock_dimensions(n, nominal_attrs)
        self.dimensions['HydroUnitBlock'] = hydroblock_dimensions()
    
        
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
        converted_dict = {}
        if hasattr(self.config, attr_name):
            unitblock_parameters = getattr(self.config, attr_name)
        else:
            print("Block not yet implemented") # TODO: Replace with logger
            
        
        for key, func in unitblock_parameters.items():
            if callable(func):
                # Extract parameter names from the function
                param_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                args = []
                
                for param in param_names:
                    if self.smspp_parameters[attr_name.split("_")[0]]['Size'][key] not in [1, '[L]', '[Li]', '[NA]', '[NP]', '[NR]']:
                        weight = True if param in ['capital_cost', 'marginal_cost', 'marginal_cost_quadratic', 'start_up_cost', 'stand_by_cost'] else False
                        arg = get_param_as_dense(n, components_type, param, weight)[[component]]
                    elif param in components_df.index or param in components_df.columns:
                        arg = components_df.get(param)
                    elif param in components_t.keys():
                        df = components_t[param]
                        arg = df[components_df.index].values
                    args.append(arg)
                
                # Apply function to the parameters
                value = func(*args)
                value = value[components_df.index].values if isinstance(value, pd.DataFrame) else value
                value = value.tolist() if isinstance(value, pd.Series) else value
                variable_type, variable_size = self.add_size_type(attr_name, key, value)
                converted_dict[key] = {"value": value, "type": variable_type, "size": variable_size}
            else:
                value = func
                variable_type, variable_size = self.add_size_type(attr_name, key)
                converted_dict[key] = {"value": value, "type": variable_type, "size": variable_size}
        
        name = components_df.name if isinstance(components_df, pd.Series) else attr_name.split("_")[0]
        
        if attr_name in ['Lines_parameters', 'Links_parameters']:
            self.networkblock[name] = {"block": 'Lines', "variables": converted_dict}
        else:
            self.unitblocks[f"{attr_name.split('_')[0]}_{index}"] = {"name": components_df.index[0],"enumerate": f"UnitBlock_{index}" ,"block": attr_name.split("_")[0], "DesignVariable": components_df['p_nom'].values, "variables": converted_dict}
        
        if attr_name == 'HydroUnitBlock_parameters':
            dimensions = self.dimensions['HydroUnitBlock']
            self.dimensions['UCBlock']["NumberElectricalGenerators"] += 1*dimensions["NumberReservoirs"] 
            
            self.unitblocks[f"{attr_name.split('_')[0]}_{index}"]['dimensions'] = dimensions
        

    
    def iterate_components(self, n):
        """
        Iterates over the network components and adds them as unit blocks.
    
        Parameters
        ----------
        n : PyPSA Network
            PyPSA network object containing components to iterate over.
        """
    
        generator_node = []
        investment_meta = {"Blocks": [], "index_extendable": [], "asset_type": []}
        unitblock_index = 0
        lines_index = 0
    
        for components in n.iterate_components(["Generator", "Store", "StorageUnit", "Line", "Link"]):
    
            components_df = components.df
            components_t = components.dynamic
            components_type = components.list_name
    
            if components_type not in ['storage_units']:
                df_investment = self.add_InvestmentBlock(n, components_df, components.name)
    
            # Lines and Links in unico blocco
            if components_type in ["lines", "links"]:
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
    
            # StorageUnits
            elif components_type == 'storage_units':
                get_bus_idx(n, components_df, components_df.bus, "bus_idx")
    
                for bus, carrier in zip(components_df['bus_idx'].values, components_df['carrier']):
                    if carrier in ['hydro', 'PHS']:
                        generator_node.extend([bus] * 2)
                    else:
                        generator_node.append(bus)
    
            # Generators / Stores
            else:
                get_bus_idx(n, components_df, components_df.bus, "bus_idx")
                generator_node.extend(components_df['bus_idx'].values)
    
            # iterate each component one by one
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
    
        # finalize
        self.generator_node = {
            'name': 'GeneratorNode',
            'type': 'float',
            'size': ("NumberElectricalGenerators",),
            'value': generator_node
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


        
    def add_InvestmentBlock(self, n, components_df, components_type):
        components_df = filter_extendable_components(components_df, components_type, nominal_attrs)
    
        # Rinomina temporanea per compatibilità col dizionario statico
        aliases = get_nominal_aliases(components_type, nominal_attrs)
        df_alias = components_df.rename(columns=aliases)
    
        if 'Fake_dimension' not in self.dimensions:
            self.dimensions['Fake_dimension'] = {}
        self.dimensions['Fake_dimension']['NumAssets_partial'] = len(df_alias)
    
        converted_dict = {}
        attr_name = 'InvestmentBlock_parameters'
        unitblock_parameters = getattr(self.config, attr_name)
    
        for key, func in unitblock_parameters.items():
            param_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            args = [df_alias.get(param) for param in param_names]
            value = func(*args)
            variable_type, variable_size = self.add_size_type(attr_name, key, value)
    
            if hasattr(self, 'investmentblock') and key in self.investmentblock:
                previous_value = self.investmentblock[key]["value"]
                new_value = previous_value + list(value) if isinstance(previous_value, list) else np.concatenate([previous_value, value])
                self.investmentblock[key]["value"] = new_value
            else:
                converted_dict[key] = {"value": value, "type": variable_type, "size": variable_size}
    
        if not hasattr(self, 'investmentblock'):
            self.investmentblock = converted_dict
        else:
            self.investmentblock.update(converted_dict)
        
        return df_alias

        

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
        
    
    def add_size_type(self, attr_name, key, args=None):
        """
        Adds the size and dtype of a variable (for the NetCDF file) based on the Excel file information.
        """
        # Ottieni i parametri del tipo di blocco e la riga corrispondente
        row = self.smspp_parameters[attr_name.split("_")[0]].loc[key]
        variable_type = row['Type']
        
        dimensions = {
            key: value
            for subdict in self.dimensions.values()
            for key, value in subdict.items()
        }
        
        # Useful only for this case. If variable, a solution must be found
        dimensions[1] = 1
        dimensions['NumberLines'] = dimensions['Lines']
        if 'NumAssets_partial' in dimensions:
            dimensions['NumAssets'] = dimensions['NumAssets_partial']


    
        # Determina la dimensione della variabile
        if args is None:
            variable_size = ()
        else:
            # Se args è un numero scalare, la dimensione è 1
            if isinstance(args, (float, int, np.integer)):
                variable_size = ()
            else:
                # Ottieni la forma se args è un array numpy
                if isinstance(args, np.ndarray):
                    shape = args.shape
                else:
                    shape = (len(args),)  # Se è una lista, trattala come un vettore
    
                # Estrai le dimensioni attese dal file Excel
                size_arr = re.sub(r'\[|\]', '', str(row['Size']).replace("][", ","))
                size_arr = size_arr.replace(" ", "").split("|")
    
                for size in size_arr:
                    if size == '1' and shape == (1,):
                        variable_size = ()
                        break
                    else:
                        # Scomponi espressioni tipo "T,L"
                        size_components = size.split(",")
                        expected_shape = tuple(dimensions[conversion_dict[s]] for s in size_components if s in conversion_dict)
    
                        if shape == expected_shape:
                            if "1" in size_components or len(size_components) == 1:
                                variable_size = (conversion_dict[size_components[0]],)  # Vettore
                            else:
                                variable_size = (conversion_dict[size_components[0]], conversion_dict[size_components[1]])  # Matrice
                            break
        return variable_type, variable_size
        
    def lines_links(self):
        if self.dimensions['NetworkBlock']['Lines'] > 0 and self.dimensions['NetworkBlock']['Links'] > 0:
            for key, value in self.networkblock['Lines']['variables'].items():
                # Required to avoid problems for line susceptance
                if not isinstance(self.networkblock['Lines']['variables'][key]['value'], (int, float, np.integer)):
                    self.networkblock['Lines']['variables'][key]['value'] = np.concatenate([
                        self.networkblock["Lines"]['variables'][key]['value'], 
                        self.networkblock["Links"]['variables'][key]['value']
                    ])
            self.networkblock.pop("Links", None)
    
        elif self.dimensions['NetworkBlock']['Lines'] == 0 and self.dimensions['NetworkBlock']['Links'] > 0:
            # Se ci sono solo i Links, rinominali in Lines
            self.networkblock["Lines"] = self.networkblock.pop("Links")
            for key, value in self.networkblock['Lines']['variables'].items():
                value['size'] = tuple('NumberLines' if x == 'Links' else x for x in value['size'])
            
    
            
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
        return solution_data
    
    
    
    def parse_networkblock_lines(self, solution_0):
        """
        Parse NetworkBlock_i blocks from an in-memory SMS++ solution and store line-level time series.
    
        This function reads the variables (DualCost, FlowValue, NodeInjection) from each
        NetworkBlock_i inside solution_0 and stacks them into 2D arrays of shape 
        (time, element). The result is stored in self.networkblock['Lines'].
    
        Parameters
        ----------
        solution_0 : Block
            The 'Solution_0' block from the loaded SMSNetwork object.
        """
    
        num_blocks = self.dimensions['UCBlock']['TimeHorizon']
        variable_shapes = {"DualCost": None, "FlowValue": None, "NodeInjection": None}
        stacked = {var: [] for var in variable_shapes}
    
        for i in range(num_blocks):
            block_key = f"NetworkBlock_{i}"
            block = solution_0.blocks.get(block_key)
    
            if block is None:
                raise KeyError(f"{block_key} not found in Solution_0")
    
            for var in stacked:
                if var not in block.variables:
                    raise KeyError(f"{var} not found in {block_key}")
                arr = block.variables[var].data
    
                if variable_shapes[var] is None:
                    variable_shapes[var] = arr.shape[0]
                elif variable_shapes[var] != arr.shape[0]:
                    raise ValueError(f"Inconsistent shape for {var} in block {i}: expected {variable_shapes[var]}, got {arr.shape[0]}")
    
                stacked[var].append(arr)
    
        if "Lines" not in self.networkblock:
            self.networkblock["Lines"] = {}
    
        for var, values in stacked.items():
            self.networkblock["Lines"][var] = np.stack(values, axis=0)


    
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
        dual_matrix = self.networkblock['Lines']['DualCost']
    
        if flow_matrix.shape != dual_matrix.shape:
            raise ValueError("Shape mismatch between FlowValue and DualCost")
    
        names, types = self.prepare_dc_unitblock_info(n)
    
        if len(names) != flow_matrix.shape[1]:
            raise ValueError("Mismatch between total network components and columns in FlowValue")
    
        current_index = len(self.unitblocks)
        n_elements = flow_matrix.shape[1]
    
        for i in range(n_elements):
            block_index = current_index + i
            unitblock_name = f"DCNetworkBlock_{block_index}"
            block_type = types[i]
            block_label = "DCNetworkBlock_links" if block_type == "link" else "DCNetworkBlock_lines"
    
            self.unitblocks[unitblock_name] = {
                "enumerate": f"UnitBlock_{block_index}",
                "block": block_label,
                "name": names[i],
                "FlowValue": flow_matrix[:, i],
                "DualCost": dual_matrix[:, i],
                "DesignVariable": self.networkblock['Lines']['variables']['MaxPowerFlow']['value'][i]
            }
        
        
    def prepare_dc_unitblock_info(self, n):
        """
        Prepare names and types for DCNetworkBlock unitblocks.
    
        This function extracts line and link names from the PyPSA network
        and returns them in the same order as stored in the combined NetCDF
        solution. It also returns a list of 'line' or 'link' types accordingly.
    
        Parameters
        ----------
        n : pypsa.Network
            PyPSA network object.
    
        Returns
        -------
        names : list of str
            Ordered names for each DC network component.
        types : list of str
            List of 'line' or 'link' corresponding to each element.
        """
        num_lines = self.dimensions['NetworkBlock']['Lines']
        num_links = self.dimensions['NetworkBlock']['Links']
    
        line_names = list(n.lines.index)
        link_names = list(n.links.index)
    
        if len(line_names) != num_lines:
            raise ValueError("Mismatch between dimensions and n.lines")
    
        if len(link_names) != num_links:
            raise ValueError("Mismatch between dimensions and n.links")
    
        names = line_names + link_names
        types = ['line'] * num_lines + ['link'] * num_links
    
        return names, types



###########################################################################################################################
############ INVERSE TRANSFORMATION INTO XARRAY DATASET ###################################################################
###########################################################################################
   
    
    def inverse_transformation(self, n):
        '''
        Performs the inverse transformation from the SMS++ blocks to xarray object.
        The xarray wll be converted in a solution type Linopy file to get n.optimize()
    
        This method initializes the inverse process and sets inverse-conversion dicts
    
        Parameters
        ----------
        n : pypsa.Network
            A PyPSA network instance from which the data will be extracted.
        '''
        all_dataarrays = self.iterate_blocks(n)
        self.ds = xr.Dataset(all_dataarrays)
        
        prepare_solution(n, self.ds)
        
        assign_solution(n)
        # assign_duals(n) # Still doesn't work
        
        n._multi_invest = False
        post_processing(n) # Forse non serve nemmeno
        
        
    
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
        return xr.merge(datasets)




#########################################################################################
######################## Conversion with PySMSpp ########################################
#########################################################################################
    
    ## Create SMSNetwork
    def convert_to_blocks(self):
        # pySMSpp
        sn = SMSNetwork(file_type=SMSFileType.eBlockFile) # Empty Block
        master = sn
        
        index_id = 0
        
        if self.dimensions['InvestmentBlock']['NumAssets'] > 0:
            name_id = 'InvestmentBlock'
            
            # self.add_slackunitblock()
            
            sn = self.convert_to_investmentblock(master, index_id, name_id)
            
            master = sn.blocks[name_id]
            name_id = 'InnerBlock'
            index_id += 1
            
        else:
            name_id = 'Block_0'
        
        self.convert_to_ucblock(master, index_id, name_id)
        
        self.sms_network = sn
        
        return sn
        
    
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
        
        
        
    
    def convert_to_investmentblock(self, master, index_id, name_id):
        """
        Converts the unit blocks into a InvestmentBlock format.
        
        Returns:
        ----------
        investmentblock : SMSNetwork
            SMSNetwork object containing the network in SMS++ InvestmentBlock format.
        """
        
        # Dimensions of the problem
        kwargs = self.dimensions['InvestmentBlock']
        
        for name, variable in self.investmentblock.items():
            if name != 'Blocks':
                kwargs[name] = Variable(
                    name,
                    variable['type'],
                    variable['size'],
                    variable['value'])
        
        master.add(
            "InvestmentBlock",  # block type
            f"{name_id}",  # block name
            id=f"{index_id}",  # block id
            **kwargs
        )
        
        return master
        
  
    def convert_to_ucblock(self, master, index_id, name_id):
        """
        Converts the unit blocks into a UCBlock format.
        
        Returns:
        ----------
        ucblock : SMSNetwork
            SMSNetwork object containing the network in SMS++ UCBlock format.
        """
        
        # Dimensions of the problem
        kwargs = self.dimensions['UCBlock']

        # Load
        demand_name = self.demand['name']
        demand_type = self.demand['type']
        demand_size = self.demand['size']
        demand_value = self.demand['value']

        demand = {demand_name: Variable(  # active power demand
                demand_name,
                demand_type,
                demand_size,
                demand_value )}

        kwargs = {**kwargs, **demand}

        # Generator node
        generator_node = {self.generator_node['name']: Variable(
            self.generator_node['name'],
            self.generator_node['type'],
            self.generator_node['size'],
            self.generator_node['value'])}

        kwargs = {**kwargs, **generator_node}

        # Lines
        if kwargs['NumberLines'] > 0:
            line_variables = {}
            for name, variable in self.networkblock['Lines']['variables'].items():
                line_variables[name] = Variable(
                    name,
                    variable['type'],
                    variable['size'],
                    variable['value'])

            kwargs = {**kwargs, **line_variables}

        # Add UC block
        master.add(
            "UCBlock",  # block type
            f"{name_id}",  # block name
            id=f"{index_id}",  # block id
            **kwargs
        )

        # Add unit blocks
        for name, unit_block in self.unitblocks.items():
            kwargs = {}
            for variable_name, variable in unit_block['variables'].items():
                kwargs[variable_name] = Variable(
                    variable_name,
                    variable['type'],
                    variable['size'],
                    variable['value'])
                
            if 'dimensions' in unit_block.keys():
                for dimension_name, dimension in unit_block['dimensions'].items():
                    kwargs[dimension_name] = dimension

            unit_block_toadd = Block().from_kwargs(
                block_type=unit_block['block'],
                **kwargs
            )

            # Why should I have name UnitBlock_0?
            master.blocks[f"{name_id}"].add_block(unit_block['enumerate'], block=unit_block_toadd)

        return master
    
    
    
    def optimize(self, configfile, *args, **kwargs):
        """
        Optimizes the UCBlock using the SMS++ solver.

        Parameters
        ----------
        configfile : str
            Path to the configuration file for the SMS++ solver.
        *args, **kwargs : additional arguments
        
        Returns
        --------
        result : dict
            The optimization result, including status and objective value.
        """
        if self.sms_network is None:
            raise ValueError("SMSNetwork not initialized.")
    
        self.result = self.sms_network.optimize(configfile, *args, **kwargs)
        
        return self.result
