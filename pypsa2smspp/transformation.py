# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 08:12:37 2024

@author: aless
"""

import pandas as pd
import pypsa
import numpy as np
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
import re
import numpy as np
import xarray as xr
import os
from pypsa2smspp.transformation_config import TransformationConfig
from pysmspp import SMSNetwork, SMSFileType, Variable, Block, SMSConfig
from pypsa.optimization.optimize import (assign_solution, assign_duals, post_processing)

NP_DOUBLE = np.float64
NP_UINT = np.uint32

DIR = os.path.dirname(os.path.abspath(__file__))
FP_PARAMS = os.path.join(DIR, "data", "smspp_parameters.xlsx")

class FakeVariable:
    def __init__(self, solution):
        self.solution = solution

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
        self.investmentblock = dict()
        
        self.dimensions = dict()
        
        self.config = config
        
        self.conversion_dict = {
            "T": "TimeHorizon",
            "NU": "NumberUnits",
            "NE": "NumberElectricalGenerators",
            "N": "NumberNodes",
            "L": "NumberLines",
            "Li": "NumberLinks",
            "NA": "NumberArcs",
            "NR": "NumberReservoirs",
            "NP": "TotalNumberPieces",
            "Nass": "NumberAssets",
            "1": 1
            }
        
        self.nominal_attrs = {
            "Generator": "p_nom",
            "Line": "s_nom",
            "Transformer": "s_nom",
            "Link": "p_nom",
            "Store": "e_nom",
            "StorageUnit": "p_nom",
        }
        
        n.stores["max_hours"] = config.max_hours_stores
        
        # Initialize with the parser and network
        self.remove_zero_p_nom_opt_components(n)
        self.read_excel_components()
        self.add_dimensions(n)
        self.iterate_components(n)
        self.add_demand(n)
        self.lines_links()

        # SMS
        self.sms_network = None
        self.result = None

    def get_paramer_as_dense(self, n, component, field, weights=True):
        """
        Get the parameters of a component as a dense DataFrame
    
        Parameters
        ----------
        n : pypsa.Network
            The PyPSA network
        component : str
            The component to get the parameters from
        field : str
            The field to get the parameters from
    
        Returns
        -------
        pd.DataFrame
            The parameters of the component as a dense DataFrame
        """
        sns = n.snapshots
        
        # Related to different investment periods
        if not n.investment_period_weightings.empty:  # TODO: check with different version
            periods = sns.unique("period")
            period_weighting = n.investment_period_weightings.objective[periods]
        weighting = n.snapshot_weightings.objective
        if not n.investment_period_weightings.empty:
            weighting = weighting.mul(period_weighting, level=0).loc[sns]
        else:
            weighting = weighting.loc[sns]
         
        # If static, it will be expanded
        if field in n.static(component).columns:
            field_val = get_as_dense(n, component, field, sns)
        else:
            field_val = n.dynamic(component)[field]
        
        # If economic, it will be weighted
        if weights:
            field_val = field_val.mul(weighting, axis=0)
        return field_val 
    
    @staticmethod
    def is_extendable(component, component_type, nominal_attrs):
        attr = nominal_attrs.get(component_type)
        extendable_attr = f"{attr}_extendable"
        
        return component[extendable_attr].values
             
        
    
    def add_demand(self, n):
        demand = n.loads_t.p_set.rename(columns=n.loads.bus)
        demand = demand.T.reindex(n.buses.index).fillna(0.)
        self.demand = {'name': 'ActivePowerDemand', 'type': 'float', 'size': ("NumberNodes", "TimeHorizon"), 'value': demand}
        
    def add_dimensions(self, n):
        # UCBlock
        
        def ucblock_dimensions(n):
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
        
        # NetworkBlock
        def networkblock_dimensions(n):
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
    
        # InvestmentBlock
        def investmentblock_dimensions(n):
            investment_components = ['generators', 'stores', 'lines', 'links']
            num_assets = 0
            for comp in investment_components:
                df = getattr(n, comp)
                comp_type = comp[:-1].capitalize() if comp != "stores" else "Store"  # es. "generators" -> "Generator"
                attr = self.nominal_attrs.get(comp_type)
                if attr and f"{attr}_extendable" in df.columns:
                    num_assets += df[f"{attr}_extendable"].sum()
        
            dimensions = {
                "NumberAssets": int(num_assets)
            }
            return dimensions
        
        # HydroUnitBlock
        def hydro_dimensions():
            dimensions = dict()
            dimensions["NumberReservoirs"] = 1
            dimensions["NumberArcs"] = 2 * dimensions["NumberReservoirs"]
            dimensions["TotalNumberPieces"] = 2
            
            return dimensions
        
        
        self.dimensions['UCBlock'] = ucblock_dimensions(n)
        self.dimensions['NetworkBlock'] = networkblock_dimensions(n)
        self.dimensions['InvestmentBlock'] = investmentblock_dimensions(n)
        self.dimensions['HydroUnitBlock'] = hydro_dimensions()
        
        
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
                        arg = self.get_paramer_as_dense(n, components_type, param, weight)[[component]]
                    elif param in components_df.index or param in components_df.columns:
                        arg = components_df.get(param)
                    elif param in components_t.keys():
                        df = components_t[param]
                        arg = df[components_df.index].values
                    args.append(arg)
                
                # Apply function to the parameters
                value = func(*args)
                value = value[components_df.index].values if isinstance(value, pd.DataFrame) else value
                value = value.item() if isinstance(value, pd.Series) else value
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
            self.unitblocks[f"{attr_name.split('_')[0]}_{index}"] = {"name": components_df.index[0],"enumerate": f"UnitBlock_{index}" ,"block": attr_name.split("_")[0], "variables": converted_dict}   
        
        if attr_name == 'HydroUnitBlock_parameters':
            dimensions = self.dimensions['HydroUnitBlock']
            self.dimensions['UCBlock']["NumberElectricalGenerators"] += 1*dimensions["NumberReservoirs"] 
            
            self.unitblocks[f"{attr_name.split('_')[0]}_{index}"]['dimensions'] = dimensions
            

            
    
    def remove_zero_p_nom_opt_components(self, n):
        # Lista dei componenti che hanno l'attributo p_nom_opt
        components_with_p_nom_opt = ["Generator", "Link", "Store", "StorageUnit", "Line", "Transformer"]
        
        for components in n.iterate_components(["Line", "Generator", "Link", "Store", "StorageUnit"]):
            components_df = components.df
            components_df = components_df[components_df[f"{self.nominal_attrs[components.name]}_opt"] > 0]
            setattr(n, components.list_name, components_df)

    
    def iterate_components(self, n):
        """
        Iterates over the network components and adds them as unit blocks.

        Parameters:
        ----------
        n : PyPSA Network
            PyPSA network object containing components to iterate over.
            
        Methods: add_UnitBlock
            Method to convert the DataFrame and get a UnitBlock
        
        Adds:
        ---------
        The components to the `unitblocks` dictionary, with distinct attributes for intermittent and thermal units.
        
        """
        renewable_carriers = ['solar', 'solar-hsat', 'onwind', 'offwind-ac', 'offwind-dc', 'offwind-float', 'PV', 'wind', 'ror']
        
        generator_node = []
        index_extendable = []
        asset_type = []
        index = 0
        for components in n.iterate_components(["Line", "Generator", "Link", "Store", "StorageUnit"]):
            # Static attributes of the class of components
            components_df = components.df
            # Dynamic attributes of the class of components
            components_t = components.dynamic
            # Class of components
            components_type = components.list_name
            # Get the index for each component (useful especially for lines)
            if components_type == 'lines':
                self.get_bus_idx(n, components_df, components_df.bus0, "start_line_idx")
                self.get_bus_idx(n, components_df, components_df.bus1, "end_line_idx")
                attr_name = "Lines_parameters"
                self.add_UnitBlock(attr_name, components_df, components_t, components.name, n)
                continue
            elif components_type == 'links':
                self.get_bus_idx(n, components_df, components_df.bus0, "start_line_idx")
                self.get_bus_idx(n, components_df, components_df.bus1, "end_line_idx")
                attr_name = "Links_parameters"
                self.add_UnitBlock(attr_name, components_df, components_t, components.name, n)
                continue
            elif components_type == 'storage_units':
                self.get_bus_idx(n, components_df, components_df.bus, "bus_idx")
                for bus, carrier in zip(components_df['bus_idx'].values, components_df['carrier']):
                    if carrier in ['hydro', 'PHS']:
                        generator_node.extend([bus] * 2)  # Repeat two times
                    else:
                        generator_node.append(bus)  # Normal case
            else:
                self.get_bus_idx(n, components_df, components_df.bus, "bus_idx")
                generator_node.extend(components_df['bus_idx'].values)


            if components_type not in ['lines', 'links', 'storage_units']:
                self.add_InvestmentBlock(n, components_df, components.name)
                
                
            # Understand which type of block we expect

            for component in components_df.index:
                if any(carrier in components_df.loc[component].carrier for carrier in renewable_carriers):
                    attr_name = "IntermittentUnitBlock_parameters"
                elif components_df.loc[component].carrier in ['hydro', 'PHS']:
                    attr_name = 'HydroUnitBlock_parameters'
                elif "storage_units" in components_type:
                    attr_name = "BatteryUnitBlock_parameters"
                elif "store" in components_type:
                    attr_name = "BatteryUnitBlock_store_parameters"
                else:
                    attr_name = "ThermalUnitBlock_parameters"
                
                self.add_UnitBlock(attr_name, components_df.loc[[component]], components_t, components.name, n, component, index)
                
                if Transformation.is_extendable(components_df.loc[[component]], components.name, self.nominal_attrs):
                    index_extendable.append(index)
                    
                    if components_type not in ['lines', 'links']:
                        asset_type.append(0)
                    else:
                        asset_type.append(1)
                
                index += 1    
                
        self.generator_node = {'name': 'GeneratorNode', 'type': 'float', 'size': ("NumberElectricalGenerators",), 'value': generator_node}
        self.investmentblock['Assets'] = {'value': np.array(index_extendable), 'type': 'int', 'size': 'NumberAssets'}
        self.investmentblock['AssetType'] = {'value': np.array(asset_type), 'type': 'int', 'size': 'NumberAssets'}

        
        
    def add_InvestmentBlock(self, n, components_df, components_type):
        components_df = Transformation.filter_extendable_components(components_df, components_type, self.nominal_attrs)
        
        if 'Fake_dimension' not in self.dimensions:
            self.dimensions['Fake_dimension'] = {}
        self.dimensions['Fake_dimension']['NumberAssets_partial'] = len(components_df)
        
        converted_dict = {}
        attr_name = 'InvestmentBlock_parameters'
        unitblock_parameters = getattr(self.config, attr_name)
    
        for key, func in unitblock_parameters.items():
            param_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            args = []
    
            if callable(func):
                for param in param_names:
                    arg = components_df.get(param)
                    args.append(arg)
    
                value = func(*args)
                variable_type, variable_size = self.add_size_type(attr_name, key, value)
    
                if hasattr(self, 'investmentblock') and key in self.investmentblock:
                    # Concateno i value (array, lista, ecc.)
                    previous_value = self.investmentblock[key]["value"]
                    if isinstance(previous_value, list):
                        new_value = previous_value + list(value)
                    else:
                        new_value = np.concatenate([previous_value, value])
                    self.investmentblock[key]["value"] = new_value
                    # non tocco "type" e "size"
                else:
                    converted_dict[key] = {"value": value, "type": variable_type, "size": variable_size}
    
        if not hasattr(self, 'investmentblock'):
            self.investmentblock = converted_dict
        else:
            # aggiungo solo le nuove chiavi calcolate
            for key in converted_dict:
                self.investmentblock[key] = converted_dict[key]

        
    @staticmethod
    def filter_extendable_components(components_df, components_type, nominal_attrs):
        """
        Filters the components DataFrame to keep only extendable units.
    
        Parameters
        ----------
        components_df : pd.DataFrame
            The static DataFrame of the component.
        components_type : str
            The capitalized component type (e.g., "Generator", "Store").
        nominal_attrs : dict
            Dictionary mapping component types to their nominal attribute.
    
        Returns
        -------
        pd.DataFrame
            Filtered DataFrame with only extendable components.
        """
        attr = nominal_attrs.get(components_type)
        if not attr:
            return components_df  # no filtering possible if type not recognized
    
        extendable_attr = f"{attr}_extendable"
    
        if extendable_attr in components_df.columns:
            return components_df[components_df[extendable_attr] == True]
        else:
            return components_df  # nothing to filter if column not found
        
    
    
    def get_bus_idx(self, n, components_df, bus_series, column_name, dtype="uint32"):
        """
        Returns the numeric index of the bus in the network n for each element of the bus_series.
        ----------
        n : PyPSA Network
        bus_series : series of buses. For example, n.lines.bus0 o n.generators.bus
        ----------
        Example: one single bus with two generators (wind and diesel 1)
                    n.generators.bus.map(n.buses.index.get_loc).astype("uint32")
                    Generator
                    wind        0
                    diesel 1    0
                    Name: bus, dtype: uint32
        """
        components_df[column_name] = bus_series.map(n.buses.index.get_loc).astype(dtype).values

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
        dimensions['NumberAssets'] = dimensions['NumberAssets_partial']

    
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
                        expected_shape = tuple(dimensions[self.conversion_dict[s]] for s in size_components if s in self.conversion_dict)
    
                        if shape == expected_shape:
                            if "1" in size_components or len(size_components) == 1:
                                variable_size = (self.conversion_dict[size_components[0]],)  # Vettore
                            else:
                                variable_size = (self.conversion_dict[size_components[0]], self.conversion_dict[size_components[1]])  # Matrice
                            break
        return variable_type, variable_size
        
    def lines_links(self):
        if "Lines" in self.networkblock and "Links" in self.networkblock:
            for key, value in self.networkblock['Lines']['variables'].items():
                # Required to avoid problems for line susceptance
                if not isinstance(self.networkblock['Lines']['variables'][key]['value'], (int, float, np.integer)):
                    self.networkblock['Lines']['variables'][key]['value'] = np.concatenate([
                        self.networkblock["Lines"]['variables'][key]['value'], 
                        self.networkblock["Links"]['variables'][key]['value']
                    ])
            self.networkblock.pop("Links", None)
    
        elif "Links" in self.networkblock and "Lines" not in self.networkblock:
            # Se ci sono solo i Links, rinominali in Lines
            self.networkblock["Lines"] = self.networkblock.pop("Links")
            for key, value in self.networkblock['Lines']['variables'].items():
                value['size'] = tuple('NumberLines' if x == 'NumberLinks' else x for x in value['size'])
            
    
            
###########################################################################################################################
############ PARSE OUPUT SMS++ FILE ###################################################################
###########################################################################################################################


    def parse_txt_to_unitblocks(self, file_path):
        current_block = None
        current_block_key = None
    
        with open(file_path, "r") as file:
            for line in file:
                match_time = re.search(r"Elapsed time:\s*([\deE\+\.-]+)\s*s", line)
                if match_time:
                    # puoi salvare elapsed_time separatamente se serve
                    continue
    
                # Match blocchi, es. BatteryUnitBlock 2
                block_match = re.search(r"(ThermalUnitBlock|BatteryUnitBlock|IntermittentUnitBlock|HydroUnitBlock)\s*(\d+)", line)
                if block_match:
                    block_type, number = block_match.groups()
                    number = int(number)
                    current_block = block_type
                    current_block_key = f"{block_type}_{number}"
    
                    self.unitblocks[current_block_key]["block"] = block_type
                    self.unitblocks[current_block_key]["enumerate"] = number
                    
                    continue
    
                # Match variabili: con o senza indice [0], [1], ...
                match = re.match(r"([\w\s]+?)(?:\s*\[(\d+)\])?\s+=\s+\[([^\]]*)\]", line)
                if match and current_block_key:
                    key_base, sub_index, values = match.groups()
                    key_base = key_base.strip()
                    values_array = np.array([float(x) for x in values.split()])
    
                    if sub_index is not None:
                        sub_index = int(sub_index)
                        # Se esiste già ed è un array, converti in dict
                        if key_base in self.unitblocks[current_block_key] and not isinstance(self.unitblocks[current_block_key][key_base], dict):
                            prev_value = self.unitblocks[current_block_key][key_base]
                            self.unitblocks[current_block_key][key_base] = {0: prev_value}
    
                        if key_base not in self.unitblocks[current_block_key]:
                            self.unitblocks[current_block_key][key_base] = {}
    
                        self.unitblocks[current_block_key][key_base][sub_index] = values_array
                    else:
                        # Caso semplice: array diretto
                        self.unitblocks[current_block_key][key_base] = values_array
    
    
    def parse_solution_to_unitblocks(self, file_path):
        num_units = self.dimensions['UCBlock']['NumberUnits']
        solution_data = {}
    
        # Load the top-level UCBlock (Solution_0 group)
        solution_data["UCBlock"] = xr.open_dataset(file_path, group="/Solution_0", engine="netcdf4")
    
        # Ensure self.unitblocks exists before assignment
        # TODO: remove this constraint if reverse transformation is needed without prior initialization
        if not hasattr(self, "unitblocks"):
            raise ValueError("self.unitblocks must be initialized before parsing the solution file.")
    
        # Iterate over all unit blocks
        for i in range(num_units):
            group_path = f"/Solution_0/UnitBlock_{i}"
            ds = xr.open_dataset(file_path, group=group_path, engine="netcdf4")
            solution_data[f"UnitBlock_{i}"] = ds
    
            # Match the corresponding key in self.unitblocks (e.g., endswith _0, _1, _2, ...)
            matching_key = next(
                (key for key in self.unitblocks if key.endswith(f"_{i}")),
                None
            )
    
            if matching_key is None:
                raise KeyError(f"No matching key found in self.unitblocks for UnitBlock_{i}")
    
            # Store all variables from the dataset into the corresponding unitblock
            for var_name in ds.data_vars:
                self.unitblocks[matching_key][var_name] = ds[var_name].values
                
            ds.close()
            
        solution_data["UCBlock"].close()
                
    
        return solution_data

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
        
        n = self.prepare_solution(n)
        
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
            component = Transformation.component_definition(n, unit_block)
            dataarrays = self.block_to_dataarrays(n, name, unit_block, component)
            if dataarrays:  # No emptry dicts
                ds = xr.Dataset(dataarrays)
                datasets.append(ds)
    
        # Merge in a single dataset
        return xr.merge(datasets)

          
    
    def block_to_dataarrays(self, n, unit_name, unit_block, component):
        '''
        Constructs a dictionary of DataArrays for a single unit block.
        
        It retrieves the inverse function mappings for the specific block type and evaluates
        each function based on the available parameters. The resulting values are formatted
        into xarray.DataArray objects using `dataarray_components`.
        
        Parameters
        ----------
        n : pypsa.Network
            The PyPSA network object.
        unit_name : str
            The name of the unit block.
        unit_block : dict
            The dictionary defining the unit block structure and parameters.
            Obtained in the first steps of the transformation class
        component : str
            The corresponding PyPSA component (e.g., 'Generator', 'StorageUnit').
        
        Returns
        -------
        dict
            A dictionary of xarray.DataArrays keyed by variable names.
        '''
        
        attr_name = f"{unit_block['block']}_inverse"
        converted_dict = {}
        normalized_keys = {Transformation.normalize_key(k): k for k in unit_block.keys()}
    
        if hasattr(self.config, attr_name):
            unitblock_parameters = getattr(self.config, attr_name)
        else:
            print(f"Block {unit_block['block']} not yet implemented")
            return {}
    
        df = getattr(n, self.config.component_mapping[component])
    
        for key, func in unitblock_parameters.items():
            if callable(func):
                value = self.evaluate_function(func, normalized_keys, unit_block, df)
                if isinstance(value, np.ndarray) and value.ndim == 2 and all(dim > 1 for dim in value.shape):
                    value = value.sum(axis=0)
                value, dims, coords, var_name = self.dataarray_components(n, value, component, unit_block, key)

                converted_dict[var_name] = xr.DataArray(value, dims=dims, coords=coords, name=var_name)
    
        return converted_dict

    
    def evaluate_function(self, func, normalized_keys, unit_block, df):
        '''
        Evaluates an inverse function by collecting its arguments from the unit block or network dataframe.
        
        Parameters
        ----------
        func : Callable
            The inverse function to evaluate.
        normalized_keys : dict
            A mapping of normalized parameter names to their original keys.
        unit_block : dict
            The dictionary defining the block parameters.
        df : pandas.DataFrame
            The dataframe from the PyPSA network corresponding to the block component.
        
        Returns
        -------
        value : Any
            The result of the inverse function evaluation.
        '''
        
        param_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        args = []

        for param in param_names:
            param = Transformation.normalize_key(param)
            if param in normalized_keys:
                arg = unit_block[normalized_keys[param]]
            else:
                arg = df.loc[unit_block['name']][param]
            args.append(arg)

        value = func(*args)
        return value
            
            
    def dataarray_components(self, n, value, component, unit_block, key):
        '''
        Determines the dimensions and coordinates of a DataArray based on the shape of the value.
        
        This function supports scalar values and 1D time series aligned with the network snapshots.
        It returns the value (reshaped if necessary), the corresponding dimension names,
        coordinate mappings, and a standardized variable name.
        
        Parameters
        ----------
        n : pypsa.Network
            The PyPSA network instance.
        value : array-like or scalar
            The evaluated parameter value.
        component : str
            The name of the PyPSA component (e.g., 'Generator').
        unit_block : dict
            The dictionary defining the block.
        key : str
            The name of the parameter being processed.
        
        Returns
        -------
        tuple
            A tuple (value, dims, coords, var_name) used to create an xarray.DataArray.
        '''
        if isinstance(value, np.ndarray):
            if value.ndim == 1 and len(value) == len(n.snapshots):
                dims = ["snapshot", component]
                coords = {
                    "snapshot": n.snapshots,
                    component: [unit_block["name"]]
                }
                value = value[:, np.newaxis]
            elif value.ndim == 1:
                dims = [f"{component}-ext"]
                coords = {f"{component}-ext": [unit_block["name"]]}
            else:
                raise ValueError(f"Unsupported shape for variable {key}: {value.shape}")
        else:
            value = np.array([value])
            dims = [f"{component}-ext"]
            coords = {f"{component}-ext": [unit_block["name"]]}

        var_name = f"{component}-{key}"
        return value, dims, coords, var_name
            

    @staticmethod 
    def component_definition(n, unit_block):
        '''
        Maps a unit block type to the corresponding PyPSA component.
        
        In some cases, such as the BatteryUnitBlock, this function dynamically chooses between
        StorageUnit and Store depending on the presence of the unit name in the network.
        
        Parameters
        ----------
        n : pypsa.Network
            The PyPSA network.
        unit_block : dict
            The dictionary defining the unit block.
        
        Returns
        -------
        str
            The name of the PyPSA component (e.g., 'Generator').
        '''
        
        block = unit_block['block']
        match block:
            case "IntermittentUnitBlock":
                component = "Generator"
            case "ThermalUnitBlock":
                component = "Generator"
            case "HydroUnitBlock":
                component = "StorageUnit"
            case "BatteryUnitBlock":
                if unit_block['name'] in n.storage_units.index:
                    component = "StorageUnit"
                else:
                    component = "Store"
        return component

    @staticmethod
    def normalize_key(key):
        '''
        Normalizes a parameter key by converting it to lowercase and replacing spaces with underscores.
        
        Parameters
        ----------
        key : str
            The parameter key to normalize.
        
        Returns
        -------
        str
            The normalized key.
        '''
        return key.lower().replace(" ", "_")
    
    
    def prepare_solution(self, n):
        """
        Prepares a fake PyPSA model on the network `n`, wrapping the solution dataset `self.ds`
        so that it can be used with `assign_solution` without modification.
    
        Parameters
        ----------
        n : pypsa.Network
            The PyPSA network object to update.
    
        Returns
        -------
        n : pypsa.Network
            The network updated with a fake model and fake solution.
        """
        # Create dictionary of fake variables
        m_variables = {}
        for var_name, dataarray in self.ds.items():
            m_variables[var_name] = FakeVariable(solution=dataarray)
    
        # Create the fake model
        n.model = type("FakeModel", (), {})()
        n.model.variables = m_variables
    
        # Create fake parameters (snapshots)
        n.model.parameters = type("FakeParameters", (), {})()
        n.model.parameters.snapshots = xr.DataArray(n.snapshots, dims=["snapshot"])
        
        # TODO associare duals in modo sensato non appena appaiono
        n.model.constraints = type("FakeConstraints", (), {})()
        n.model.constraints.snapshots = xr.DataArray(n.snapshots, dims=["snapshot"])
        
    
        # Create fake objective
        n.model.objective = type("FakeObjective", (), {})()
        n.model.objective.value = 10000  # arbitrary value
    
        return n


#########################################################################################
######################## Conversion with PySMSpp ########################################
#########################################################################################


    ## Create SMSNetwork
    def convert_to_ucblock(self):
        """
        Converts the unit blocks into a UCBlock format.
        
        Returns:
        ----------
        ucblock : SMSNetwork
            SMSNetwork object containing the network in SMS++ UCBlock format.
        """
        # pySMSpp
        sn = SMSNetwork(file_type=SMSFileType.eBlockFile) # Empty Block

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
        sn.add(
            "UCBlock",  # block type
            "Block_0",  # block name
            id="0",  # block id
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
            sn.blocks["Block_0"].add_block(unit_block['enumerate'], block=unit_block_toadd)
        
        self.sms_network = sn

        return sn
    
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
