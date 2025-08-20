# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:07:31 2024

@author: aless
"""

import pandas as pd
import numpy as np
import logging
import pypsa

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -10s %(funcName) '
              '-10s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logging.getLogger("Article").setLevel(logging.WARNING)

#%% Definition of the network
class NetworkDefinition:
    """
    NetworkDefinition class for building a PyPSA network from input Excel files.

    This class reads data from multiple Excel files, including files for components, 
    demand profiles, costs, and renewable energy sources. The paths to these files are 
    provided by a parser object, reading from config/application.ini.
    It then constructs a PyPSA network based on this data.

    Attributes:
    ----------
    parser : object
        Object containing the paths for input data files.

    n : pypsa.Network
        PyPSA network object that holds the components, costs, demand, and renewables data.
    """

    def __init__(self, parser):
        """
        Initializes the NetworkDefinition class.

        Parameters:
        ----------
        parser : object
            Parser containing paths to input data files.
        """
        self.parser = parser
        self.init()
        
    def init(self):
        """
        Sets the workflow for the class

        This method defines the network's snapshots and reads all components from the 
        specified input files. It then adds costs for components, demand profiles, 
        and renewables to the network.
        """
        self.n = pypsa.Network()
        
        self.define_snapshots()
        
        all_sheets = self.read_excel_components()
        self.add_all_components(all_sheets)
        
        #self.add_costs_components()
        self.add_demand()
        self.add_renewables()
        
    def define_snapshots(self):
        """
        Defines the network snapshots based on the parser input.

        Sets:
        --------
        self.n.snapshots : range
            Range of snapshots based on the number of snapshots from the parser.
        
        self.n.snapshot_weightings.objective : float
            Weighting factor for snapshots, taken from the parser.
        """
        self.n.snapshots = range(0, self.parser.n_snapshots)
        self.n.snapshot_weightings.objective = self.parser.weight
        self.n.snapshot_weightings.generators = self.parser.weight
        # self.n.snapshot_weightings.stores = self.parser.weight
    
    def read_excel_components(self):
        """
        Reads components from an Excel file specified in the parser. Each sheet includes a class of components

        Returns:
        ----------
        all_sheets : dict
            Dictionary where keys are sheet names and values are DataFrames containing 
            data for each component type.
        """
        file_path = f"{self.parser.input_data_path}/{self.parser.input_name_components}"
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        return all_sheets
    
    def add_all_components(self, all_sheets):
        """
        Iterates over the sheets and adds each component type to the network.

        Parameters:
        ----------
        all_sheets : dict
            Dictionary containing component data for each type, organized by sheet name.
        """
        for sheet_name, data in all_sheets.items():
            self.add_component(self.n, sheet_name, data)
        
    def add_component(self, network, component_type, data):
        """
        Adds a single component to the network.

        Parameters:
        ----------
        network : pypsa.Network
            The PyPSA network to which the component will be added.

        component_type : str
            The type of component (e.g., Bus, Link, etc.) to add.
        
        data : DataFrame
            Data for the component, with each row representing a specific unit.
        """
        for _, row in data.iterrows():
            # Name of the component (expected to be the first column in the data)
            name = row[data.columns[0]]
            params = {col: row[col] for col in data.columns if col != 'name' and pd.notna(row[col])}
            network.add(component_type, name, **params)
            
    def add_costs_components(self):
        """
        Adds cost data to network components based on an Excel file.

        Reads costs from an Excel file, iterates over generator and storage components,
        and assigns capital and marginal costs to each component.

        Notes:
        ---------
        Requires specific column names in the Excel file, e.g., 'Capital cost [€/MW]' 
        and 'Marginal cost [€/MWh]'.
        """
        file_path = f"{self.parser.input_data_path}/{self.parser.input_name_costs}"
        costs = pd.read_excel(file_path, index_col=0)
        
        for components in self.n.iterate_components(["Generator", "StorageUnit", "Link", "Store"]): 
            components_df = components.df
            for component in components_df.index:
                components_df.loc[component, 'capital_cost'] = costs.at[component.split(" ")[0], 'Capital cost [€/MW]']
                components_df.loc[component, 'marginal_cost'] = costs.at[component.split(" ")[0], 'Marginal cost [€/MWh]']
        
    def add_demand(self):
        """
        Adds demand profiles to the network's loads based on daily demand data.

        Reads daily demand data from a CSV file, then scales it to match the 
        number of snapshots in the network. Demand profiles are set for each load 
        in the network using this yearly demand profile.

        Notes:
        --------
        Uses a random normal distribution to introduce variability in the demand profile.
        """
        file_path = f"{self.parser.input_data_path}/{self.parser.input_name_demand}"
        df_demand_day = pd.read_csv(file_path)
        df_demand_day["hour"] = range(0, 24)
        n_days = int(len(self.n.snapshots) / 24)

        for load in self.n.loads.index:
            df_demand_year = np.random.normal(
                np.tile(df_demand_day["demand"], n_days),
                np.tile(df_demand_day["standard_deviation"], n_days),
            ) * 100
            self.n.loads_t.p_set[load] = df_demand_year
                    
    def add_renewables(self):
        """
        Adds per-unit power profiles for renewable generators (solar and wind).

        Reads power profiles for photovoltaic (PV) and wind generators from CSV files.
        Assigns the profiles to generators in the network based on the generator name.

        Notes:
        ---------
        Assumes specific column names in the CSV files for solar and wind profiles.
        """
        file_path_PV = f"{self.parser.input_data_path}/{self.parser.input_name_pv}"
        df_pv = pd.read_csv(file_path_PV, skiprows=3, nrows=len(self.n.snapshots))

        file_path_wind = f"{self.parser.input_data_path}/{self.parser.input_name_wind}"
        df_wind = pd.read_csv(file_path_wind, skiprows=3, nrows=len(self.n.snapshots))
        
        for generator in self.n.generators.index:
            if 'solar' in generator.lower() or 'pv' in generator.lower():
                self.n.generators_t.p_max_pu[generator] = df_pv['electricity']
            elif 'wind' in generator.lower():
                self.n.generators_t.p_max_pu[generator] = df_wind['electricity']
                
        
        
                    
