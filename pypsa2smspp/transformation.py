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
            "1": 1
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
    
    
    def add_demand(self, n):
        demand = n.loads_t.p_set.rename(columns=n.loads.bus)
        demand = demand.T.reindex(n.buses.index).fillna(0.)
        self.demand = {'name': 'ActivePowerDemand', 'type': 'float', 'size': ("NumberNodes", "TimeHorizon"), 'value': demand}
        
    def add_dimensions(self, n):
        components = {
            "NumberUnits": ["generators", "storage_units", "stores"],
            "NumberElectricalGenerators": ["generators", "storage_units", "stores"],
            "NumberNodes": ["buses"],
            "NumberLines": ["lines", "links"],
        }
        
        # Calcola le dimensioni sommando le lunghezze
        self.dimensions = {
            "TimeHorizon": len(n.snapshots),
            **{
                name: sum(len(getattr(n, comp)) for comp in comps)
                for name, comps in components.items()
            },
            
            }
        
        
        components = {
            "Lines": ['lines'],
            "Links": ['links'],
            "combined": ['lines', 'links']
            }
        self.dimensions_lines = {
            **{
                name: sum(len(getattr(n, comp)) for comp in comps)
                for name, comps in components.items()
            }
            }
        
        
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
                    if self.smspp_parameters[attr_name.split("_")[0]]['Size'][key] not in [1, '[L]', '[Li]', '[NA]', '[NP]']:
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
            dimensions = self.hydro_dimensions()
            self.unitblocks[f"{attr_name.split('_')[0]}_{index}"]['dimensions'] = dimensions
            
    
    def hydro_dimensions(self):
        dimensions = {}
        dimensions["NumberReservoirs"] = 1
        dimensions["NumberArcs"] = 3 * dimensions["NumberReservoirs"]
        dimensions["TotalNumberPieces"] = 3
        self.dimensions["NumberElectricalGenerators"] += 2*dimensions["NumberReservoirs"] 
        
        return dimensions
            
    
    def remove_zero_p_nom_opt_components(self, n):
        # Lista dei componenti che hanno l'attributo p_nom_opt
        components_with_p_nom_opt = ["Generator", "Link", "Store", "StorageUnit", "Line", "Transformer"]
        nominal_attrs = {
            "Generator": "p_nom",
            "Line": "s_nom",
            "Transformer": "s_nom",
            "Link": "p_nom",
            "Store": "e_nom",
            "StorageUnit": "p_nom",
        }
        
        for components in n.iterate_components(["Line", "Generator", "Link", "Store", "StorageUnit"]):
            components_df = components.df
            components_df = components_df[components_df[f"{nominal_attrs[components.name]}_opt"] > 0]
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
        nominal_attrs = {
            "Generator": "p_nom",
            "Line": "s_nom",
            "Transformer": "s_nom",
            "Link": "p_nom",
            "Store": "e_nom",
            "StorageUnit": "p_nom",
        }
        
        generator_node = []
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
                    if carrier == 'hydro':
                        generator_node.extend([bus] * 3)  # Repeat three times
                    else:
                        generator_node.append(bus)  # Normal case
            else:
                self.get_bus_idx(n, components_df, components_df.bus, "bus_idx")
                generator_node.extend(components_df['bus_idx'].values)


                # Understand which type of block we expect

            for component in components_df.index:
                if any(carrier in components_df.loc[component].carrier for carrier in renewable_carriers):
                    attr_name = "IntermittentUnitBlock_parameters"
                elif components_df.loc[component].carrier == 'hydro':
                    attr_name = 'HydroUnitBlock_parameters'
                elif "storage_units" in components_type:
                    attr_name = "BatteryUnitBlock_parameters"
                elif "store" in components_type:
                    attr_name = "BatteryUnitBlock_store_parameters"
                else:
                    attr_name = "ThermalUnitBlock_parameters"
                
                self.add_UnitBlock(attr_name, components_df.loc[[component]], components_t, components.name, n, component, index)
                index += 1
        self.generator_node = {'name': 'GeneratorNode', 'type': 'float', 'size': ("NumberElectricalGenerators",), 'value': generator_node}
        
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
        
        dimensions = self.dimensions.copy()
        
        # Useful only for this case. If variable, a solution must be found
        dimensions[1] = 1
        dimensions["NumberReservoirs"] = 1
        dimensions["NumberArcs"] = 3 * dimensions["NumberReservoirs"]
        dimensions["TotalNumberPieces"] = 3
        dimensions["NumberLines"] = self.dimensions_lines['Lines']
        dimensions["NumberLinks"] = self.dimensions_lines['Links']
    
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
############ INVERSE TRANSFORMATION INTO XARRAY DATASET ###################################################################
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
