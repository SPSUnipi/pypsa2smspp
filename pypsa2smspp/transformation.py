# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 08:12:37 2024

@author: aless
"""

import pandas as pd
import numpy as np
import xarray as xr
import os
from pypsa2smspp.transformation_config import TransformationConfig
from pysmspp import SMSNetwork, SMSFileType, Attribute, Dimension, Variable, Block, SMSConfig
from pypsa2smspp import logger
from copy import deepcopy
import pysmspp
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union, Literal, Callable
import warnings

from .constants import conversion_dict, nominal_attrs, renewable_carriers
from .utils import (
    is_extendable,
    filter_extendable_components,
    get_bus_idx,
    get_nominal_aliases,
    ucblock_dimensions,
    networkblock_dimensions,
    investmentblock_dimensions,
    hydroblock_dimensions,
    get_attr_name,
    process_dcnetworkblock,
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
    get_param_as_dense
)

from .pip_utils import (
    StepTimer,
    step,
)

from .inverse import (
    component_definition,
    block_to_dataarrays,
    normalize_key,
    evaluate_function,
    dataarray_components,
    block_to_dataarrays_stochastic,
    broadcast_static_variables_over_scenarios,
)
from .io_parser import (
    parse_txt_to_unitblocks,
    assign_design_variables_to_unitblocks,
    prepare_solution,
    split_merged_dcnetworkblocks
)

from .stochastic_utils import (
    get_base_scenario_network,
    describe_problem_structure,
    build_dss_demand,
    build_dss_marginal,
    build_dss_renewables,
    merge_tssb_dss_parts,
    calculate_design_variables,
    build_tssb_static_abstract_path,
    build_stochastic_mapping_demand,
    build_tssb_stochastic_block_data,
)

NP_DOUBLE = np.float64
NP_UINT = np.uint32

DIR = os.path.dirname(os.path.abspath(__file__))
FP_PARAMS = os.path.join(DIR, "data", "smspp_parameters.xlsx")



class Transformation:
    """
    Convert a PyPSA network into SMS++ blocks, run SMS++ via pySMSpp, and map results back.

    Typical usage:
        n = Transformation(...).run(n)

    The pipeline is roughly:
      - consistency checks / pre-processing
      - direct conversion (PyPSA -> internal representation)
      - block construction (SMSNetwork)
      - SMS++ optimization (pySMSpp)
      - solution parsing + inverse mapping to PyPSA
    """

    def __init__(
        self,
        *,
        # --- transformation options ---
        merge_links: Union[bool, str, Sequence[str]] = True,
        merge_selector: Optional[Callable[..., bool]] = None,
        capacity_expansion_ucblock: bool = True,
        enable_thermal_units: bool = False,
        intermittent_carriers: Optional[Union[str, Sequence[str]]] = None,

        # --- I/O ---
        workdir: Union[str, Path] = "output",
        name: str = "test_case",
        overwrite: bool = True,
        fp_temp: Union[str, Path] = "temp_{name}.nc",
        fp_log: Optional[Union[str, Path]] = "log_{name}.txt",
        fp_solution: Optional[Union[str, Path]] = "solution_{name}.nc",

        # --- SMS++ ---
        configfile: Optional[Union[str, Path, "pysmspp.SMSConfig"]] = "auto",
        pysmspp_options: Optional[Mapping[str, Any]] = None,
        
        # Stochastic
        stochastic_parameters: Optional[Mapping[str, Any]] = None,
        
    ):
        """
        Parameters
        ----------
        merge_links : bool | str | Sequence[str], default True
            Controls whether store-related charge/discharge link pairs are merged into a single
            merged link representation (useful to match PyPSA-Eur modelling conventions).

            Supported values:
              - False: disable merging entirely.
              - True: enable merging for built-in safe presets (TES, battery, H2 reversed).
              - str / list[str]: enable merging for a subset of presets and/or custom tags.
                If custom tags are provided (i.e., values not in {"tes","battery","h2"}),
                `merge_selector` MUST be provided.

        merge_selector : callable, optional
            Optional power-user hook to authorize additional merges beyond the built-in presets.

            Signature:
                merge_selector(n, store_name, srow, charge_row, discharge_row) -> bool

            Notes:
              - If `merge_links` contains custom entries, this selector is required.
              - Any exception raised inside the selector will cause the corresponding store
                merge to be skipped.

        capacity_expansion_ucblock : bool, default True
            Selects the internal block structure / modelling mode:
              - True  -> UCBlock-style representation.
              - False -> InvestmentBlock-style representation.

            Used to choose default SMS++ templates when `configfile="auto"`.

        enable_thermal_units : bool, default False
            Controls whether "thermal" generator units are allowed.

            Behaviour:
              - False: all non-slack generators are treated as intermittent (i.e., no thermals).
              - True : both intermittent and thermal generators are allowed.

            When enabled, the intermittent/thermal split is determined by `intermittent_carriers`
            (user override) or by the library default intermittent set (typically `renewable_carriers`).

        intermittent_carriers : str | Sequence[str], optional
            Defines which generator carriers are treated as intermittent when
            `enable_thermal_units=True`.

            Behaviour:
              - None: use the library default intermittent set (typically `renewable_carriers`).
              - str / list[str]: explicit override (case-insensitive).

            Notes:
              - Ignored when `enable_thermal_units=False`.

        workdir : str | Path, default "output"
            Output directory for SMS++ artifacts. The directory is created if it does not exist.

        name : str, default "test_case"
            Case name used to render file templates (e.g., "temp_{name}.nc").

        overwrite : bool, default True
            If True, existing artifacts at the resolved paths are removed before optimization.

        fp_temp : str | Path, default "temp_{name}.nc"
            Temporary network file path template passed to `SMSNetwork.optimize(fp_temp=...)`.
            Supports formatting with `{name}`. If relative, interpreted relative to `workdir`.

        fp_log : str | Path | None, default "log_{name}.txt"
            Log file path template passed to `SMSNetwork.optimize(fp_log=...)`.
            If None, logging to file is disabled. Supports `{name}`.

        fp_solution : str | Path | None, default "solution_{name}.nc"
            Solution file path template passed to `SMSNetwork.optimize(fp_solution=...)`.
            If None, solver-dependent behaviour. Supports `{name}`.

        configfile : str | Path | pysmspp.SMSConfig | None, default "auto"
            SMS++ configuration passed to `SMSNetwork.optimize(configfile=...)`.

            Supported values:
              - "auto" or None: use built-in default template based on `capacity_expansion_ucblock`.
              - str/Path: interpreted as a template path used to build `pysmspp.SMSConfig(template=...)`.
              - pysmspp.SMSConfig: used directly.

        pysmspp_options : Mapping[str, Any], optional
            Extra keyword arguments forwarded to `SMSNetwork.optimize(...)`.

            Typical keys include (all optional; pySMSpp provides defaults):
              - smspp_solver : str | SMSPPSolverTool
              - inner_block_name : str
              - logging : bool
              - tracking_period : float

            Any other keys are forwarded as solver-constructor kwargs (power-user usage).
        """
        config = TransformationConfig()
        self.config = deepcopy(config)

        self.merge_links = merge_links
        if merge_selector is not None:
            self.merge_selector = merge_selector

        self.capacity_expansion_ucblock = bool(capacity_expansion_ucblock)
        self.enable_thermal_units = bool(enable_thermal_units)
        self.intermittent_carriers = intermittent_carriers

        self.workdir = Path(workdir)
        self.name = str(name)
        self.overwrite = bool(overwrite)

        self.fp_temp = fp_temp
        self.fp_log = fp_log
        self.fp_solution = fp_solution

        self.configfile = configfile
        self.pysmspp_options = dict(pysmspp_options or {})

        # internal state
        self.unitblocks = {}
        self.networkblock = {}
        self.investmentblock = {"Blocks": []}
        self.dimensions = {}

        self.sms_network = None
        self.result = None
        self.problem_structure = {}
        self.tssb_data = None
        self.design_variables = []
        self.stochastic_parameters = dict(stochastic_parameters or {})
        self.unitblock_design_data = []

 
        
##################################################################################################
####################### Pipeline #################################################################
##################################################################################################
    
    def run(self, n, verbose: bool = True):
        
        self.create_model(n, verbose=verbose)
        self.optimize(verbose=verbose)
        self.retrieve_solution(n, verbose=verbose)

        if verbose:
            self.timer.print_summary()

        return n
    
    def create_model(self, n, verbose: bool = True):
        # Keep timings accessible after the run
        self.timer = StepTimer()
        n.calculate_dependent_values()
        n_direct = get_base_scenario_network(n)

        with step(self.timer, "consistency_check", verbose=verbose):
            self.consistency_check(n)

        with step(self.timer, "direct", verbose=verbose):
            n.stores['max_hours'] = self.config.max_hours_stores
            self.direct(n_direct)

        with step(self.timer, "prepare_tssb_interface", verbose=verbose):
            self.prepare_tssb_interface(n)

        with step(self.timer, "convert_to_blocks", verbose=verbose):
            self.sms_network = self.convert_to_blocks()
        
        return self.sms_network
    
    def optimize(self, verbose: bool = True):
        if self.sms_network is None:
            raise ValueError("Model must be created before optimization. Call create_model(n) first.")
        with step(self.timer, "optimize", verbose=verbose, extra={"mode": "auto"}):
            return self._optimize()
    
    def retrieve_solution(self, n, verbose: bool = True):
        if self.result is None:
            raise ValueError("Optimization must be run before retrieving solution. Call optimize() first.")
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
        self.dimensions['NetworkBlock'] = networkblock_dimensions(n, self.capacity_expansion_ucblock)
        self.dimensions['InvestmentBlock'] = investmentblock_dimensions(n, self.capacity_expansion_ucblock, nominal_attrs)
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
        self.unitblock_design_data = []
        unitblock_index = 0
        lines_index = 0
        self._dc_names = []
        self._dc_types = []
    
        
        stores_df, links_merged_df, self.dimensions["NetworkBlock"]["merged_links_ext"] = build_store_and_merged_links(
            n,
            merge_links=self.merge_links,
            logger=logger,
            merge_selector=getattr(self, "merge_selector", None),
        )
        
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
        
        correct_dimensions(self.dimensions, stores_df, links_merged_df, n, self.capacity_expansion_ucblock)
        
        self._dc_index = build_dc_index(n, links_before, links_after)
        
        # TODO remove when necessary
        self._dc_names  = list(self._dc_index['physical']['names'])
        self._dc_types  = list(self._dc_index['physical']['types'])


        if self.capacity_expansion_ucblock:
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
                not self.capacity_expansion_ucblock
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
    
                attr_name = get_attr_name(components.name)
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
                attr_name = get_attr_name(components.name, carrier, enable_thermal_units=self.enable_thermal_units, intermittent_carriers=self.intermittent_carriers, default_intermittent=renewable_carriers)
    
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
                    self._store_unitblock_design_variable(attr_name, unitblock_index)
    
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
                "DesignVariable" if not self.capacity_expansion_ucblock else
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
        """
        Build nodal active power demand for SMS++.
    
        This includes both dynamic and static loads.
        """
        demand_by_load = get_param_as_dense(n, "Load", "p_set", weights=False)
        demand = demand_by_load.rename(columns=n.loads.bus).groupby(level=0, axis=1).sum()
        demand = demand.T.reindex(n.buses.index).fillna(0.0)
    
        self.demand = {
            "name": "ActivePowerDemand",
            "type": "float",
            "size": ("NumberNodes", "TimeHorizon"),
            "value": demand,
        }
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
        Parse a loaded SMS++ solution structure and populate self.unitblocks.
    
        Deterministic case
        ------------------
        Parse Solution_0 directly and store unit-level variables in the legacy flat
        structure, e.g. self.unitblocks[block_name]["ActivePower"].
    
        Stochastic case
        ---------------
        Parse Solution_0 / ScenarioSolution_i blocks and store variables under:
            self.unitblocks[block_name]["scenarios"][scenario_name][var_name]
    
        Parameters
        ----------
        solution : SMSNetwork
            An in-memory SMS++ solution object.
        n : pypsa.Network
            The PyPSA network object used to retrieve line and link names.
    
        Returns
        -------
        solution_data : dict
            Dictionary with parsed block references, mainly for inspection/debugging.
        """
        if not hasattr(self, "unitblocks"):
            raise ValueError("self.unitblocks must be initialized before parsing the solution.")
    
        if "Solution_0" not in solution.blocks:
            raise KeyError("'Solution_0' not found in solution.blocks")
    
        if not hasattr(self, "networkblock") or self.networkblock is None:
            self.networkblock = {}
    
        solution_data = {}
    
        if self.problem_structure.get("is_stochastic", False):
            return self._parse_stochastic_solution_to_unitblocks(solution, n, solution_data)
    
        return self._parse_deterministic_solution_to_unitblocks(solution, n, solution_data)
    
    
    def _parse_deterministic_solution_to_unitblocks(self, solution, n, solution_data):
        """
        Parse the legacy deterministic solution structure.
        """
        num_units = self.dimensions["UCBlock"]["NumberUnits"]
    
        solution_0 = solution.blocks["Solution_0"]
        has_investment = "DesignVariables" in solution_0.variables
    
        if has_investment:
            inner_solution = solution_0.blocks["InnerSolution"]
            solution_data["UCBlock"] = inner_solution
        else:
            inner_solution = solution_0
            solution_data["UCBlock"] = solution_0
    
        if self.dimensions["UCBlock"]["NumberLines"] > 0:
            self.parse_networkblock_lines(inner_solution, scenario_name=None)
            self.generate_line_unitblocks(n, scenario_name=None)
    
        self._parse_unitblocks_from_solution_block(
            solution_block=inner_solution,
            num_units=num_units,
            scenario_name=None,
            solution_data=solution_data,
        )
    
        # Assign design variables if investment
        if has_investment:
            design_vars = solution_0.variables["DesignVariables"].data
            block_names = self.investmentblock.get("Blocks", [])
            assign_design_variables_to_unitblocks(self.unitblocks, block_names, design_vars)
    
        split_merged_dcnetworkblocks(self.unitblocks)
        return solution_data
    
    
    def _parse_stochastic_solution_to_unitblocks(self, solution, n, solution_data):
        """
        Parse the stochastic TSSB solution structure.
    
        Expected layout:
            Solution_0
                ScenarioSolution_0
                ScenarioSolution_1
                ...
    
        Variables are stored in:
            self.unitblocks[matching_key]["scenarios"][scenario_name][var_name]
        """
        num_units = self.dimensions["UCBlock"]["NumberUnits"]
        scenario_names = list(self.problem_structure["scenario_names"])
        expected_n_scenarios = self.problem_structure["number_scenarios"]
    
        if len(scenario_names) != expected_n_scenarios:
            raise ValueError(
                f"Mismatch between problem_structure['scenario_names'] ({len(scenario_names)}) "
                f"and problem_structure['number_scenarios'] ({expected_n_scenarios})."
            )
    
        solution_0 = solution.blocks["Solution_0"]
        solution_data["TSSB"] = solution_0
    
        # Keep a scenario-wise container for parsed network data
        self.networkblock.setdefault("Scenarios", {})
    
        for i, scenario_name in enumerate(scenario_names):
            scenario_block_key = f"ScenarioSolution_{i}"
    
            if scenario_block_key not in solution_0.blocks:
                raise KeyError(
                    f"{scenario_block_key} not found in Solution_0. "
                    f"Expected one scenario block per scenario in problem_structure."
                )
    
            scenario_block = solution_0.blocks[scenario_block_key]
            solution_data[scenario_block_key] = scenario_block
    
            if self.dimensions["UCBlock"]["NumberLines"] > 0:
                self.parse_networkblock_lines(scenario_block, scenario_name=scenario_name)
                self.generate_line_unitblocks(n, scenario_name=scenario_name)
    
            self._parse_unitblocks_from_solution_block(
                solution_block=scenario_block,
                num_units=num_units,
                scenario_name=scenario_name,
                solution_data=solution_data,
            )
    
        split_merged_dcnetworkblocks(self.unitblocks)
        return solution_data
    
    
    def _parse_unitblocks_from_solution_block(
        self,
        solution_block,
        num_units,
        scenario_name=None,
        solution_data=None,
    ):
        """
        Parse UnitBlock_i from a generic solution block.
    
        Parameters
        ----------
        solution_block : Block
            Solution block containing UnitBlock_i.
        num_units : int
            Number of physical unit blocks.
        scenario_name : str or None, optional
            If None, variables are stored in legacy flat format.
            Otherwise they are stored under ["scenarios"][scenario_name].
        solution_data : dict or None, optional
            Optional dictionary used for inspection/debugging.
        """
        for i in range(num_units):
            block_key = f"UnitBlock_{i}"
    
            if block_key not in solution_block.blocks:
                raise KeyError(f"{block_key} not found in solution block")
    
            block = solution_block.blocks[block_key]
    
            if solution_data is not None:
                if scenario_name is None:
                    solution_data[block_key] = block
                else:
                    solution_data.setdefault("scenarios", {})
                    solution_data["scenarios"].setdefault(scenario_name, {})
                    solution_data["scenarios"][scenario_name][block_key] = block
    
            matching_key = self._get_matching_unitblock_key(i)
    
            for var_name, var_obj in block.variables.items():
                self._store_unitblock_variable(
                    matching_key=matching_key,
                    var_name=var_name,
                    data=var_obj.data,
                    scenario_name=scenario_name,
                )
    
    
    def _get_matching_unitblock_key(self, unit_index):
        """
        Return the key in self.unitblocks corresponding to UnitBlock_{unit_index}.
        """
        matching_key = next(
            (key for key in self.unitblocks if key.endswith(f"_{unit_index}")),
            None,
        )
    
        if matching_key is None:
            raise KeyError(f"No matching key found in self.unitblocks for UnitBlock_{unit_index}")
    
        return matching_key
    
    
    def _store_unitblock_variable(self, matching_key, var_name, data, scenario_name=None):
        """
        Store a parsed variable into self.unitblocks.
    
        Deterministic:
            self.unitblocks[matching_key][var_name] = data
    
        Stochastic:
            self.unitblocks[matching_key]["scenarios"][scenario_name][var_name] = data
        """
        if scenario_name is None:
            self.unitblocks[matching_key][var_name] = data
            return
    
        self.unitblocks[matching_key].setdefault("scenarios", {})
        self.unitblocks[matching_key]["scenarios"].setdefault(scenario_name, {})
        self.unitblocks[matching_key]["scenarios"][scenario_name][var_name] = data
    
    
    def parse_networkblock_lines(self, solution_block, scenario_name=None):
        """
        Parse line-level time series from a generic SMS++ solution block.
    
        If the block contains a single aggregated 'NetworkBlock', variables are read
        directly. Otherwise, it falls back to stacking 'NetworkBlock_i'.
    
        Deterministic storage:
            self.networkblock["Lines"][var] -> shape (time, element)
    
        Stochastic storage:
            self.networkblock["Scenarios"][scenario_name]["Lines"][var] -> shape (time, element)
        """
        vars_of_interest = ("FlowValue", "NodeInjection")
    
        if scenario_name is None:
            self.networkblock.setdefault("Lines", {})
            target = self.networkblock["Lines"]
        else:
            self.networkblock.setdefault("Scenarios", {})
            self.networkblock["Scenarios"].setdefault(scenario_name, {})
            self.networkblock["Scenarios"][scenario_name].setdefault("Lines", {})
            target = self.networkblock["Scenarios"][scenario_name]["Lines"]
    
        blocks = solution_block.blocks
    
        # --- Case 1: aggregated NetworkBlock -------------------------------------
        if "NetworkBlock" in blocks:
            block = blocks["NetworkBlock"]
    
            if "DesignNetworkBlock_0" in block.blocks:
                block = block.blocks["DesignNetworkBlock_0"]
                vars_local = vars_of_interest + ("DesignValue",)
            else:
                vars_local = vars_of_interest
    
            for var in vars_local:
                if var not in block.variables:
                    raise KeyError(f"{var} not found in NetworkBlock")
    
                arr = block.variables[var].data
    
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
    
                if arr.ndim != 2:
                    raise ValueError(
                        f"Unexpected shape for {var} in NetworkBlock: {arr.shape} "
                        f"(expected 2D)"
                    )
    
                target[var] = arr
    
            return
    
        # --- Case 2: legacy per-time NetworkBlock_i ------------------------------
        nb_keys = [
            k for k in blocks.keys()
            if k.startswith("NetworkBlock_") and k[len("NetworkBlock_"):].isdigit()
        ]
    
        if not nb_keys:
            raise KeyError("No 'NetworkBlock' or 'NetworkBlock_i' blocks found in solution block")
    
        nb_keys.sort(key=lambda k: int(k.split("_")[-1]))
    
        variable_first_lengths = {v: None for v in vars_of_interest}
        stacked = {v: [] for v in vars_of_interest}
    
        for k in nb_keys:
            block = blocks[k]
    
            for var in vars_of_interest:
                if var not in block.variables:
                    raise KeyError(f"{var} not found in {k}")
    
                arr = block.variables[var].data
    
                if arr.ndim == 2 and arr.shape[0] == 1:
                    arr = arr[0]
    
                if arr.ndim != 1:
                    raise ValueError(
                        f"Unexpected shape for {var} in {k}: {arr.shape} (expected 1D)"
                    )
    
                if variable_first_lengths[var] is None:
                    variable_first_lengths[var] = arr.shape[0]
                elif variable_first_lengths[var] != arr.shape[0]:
                    raise ValueError(
                        f"Inconsistent element size for {var}: "
                        f"expected {variable_first_lengths[var]}, got {arr.shape[0]} in {k}"
                    )
    
                stacked[var].append(arr)
    
        for var, lst in stacked.items():
            target[var] = np.stack(lst, axis=0)
    
    
    def generate_line_unitblocks(self, n, scenario_name=None):
        """
        Generate or update synthetic DCNetworkBlock_* unitblocks for lines and links.
    
        Deterministic case
        ------------------
        Store directly:
            self.unitblocks[unitblock_name]["FlowValue"]
            self.unitblocks[unitblock_name]["DesignVariable"]
    
        Stochastic case
        ---------------
        Store under:
            self.unitblocks[unitblock_name]["scenarios"][scenario_name]["FlowValue"]
            self.unitblocks[unitblock_name]["scenarios"][scenario_name]["DesignVariable"]
        """
        if scenario_name is None:
            lines_data = self.networkblock["Lines"]
        else:
            if "Scenarios" not in self.networkblock or scenario_name not in self.networkblock["Scenarios"]:
                raise KeyError(f"Scenario '{scenario_name}' not found in self.networkblock['Scenarios']")
            lines_data = self.networkblock["Scenarios"][scenario_name]["Lines"]
    
        if "FlowValue" not in lines_data:
            raise KeyError("FlowValue not found in parsed networkblock lines")
    
        flow_matrix = lines_data["FlowValue"]
    
        if "DesignValue" in lines_data:
            design_matrix = lines_data["DesignValue"]
        else:
            design_matrix = None
    
        names, types = self.prepare_dc_unitblock_info(n)
    
        links_effs = self.networkblock.get("efficiencies", {})
        max_eff_len = self.networkblock.get("max_eff_len", 1)
    
        if len(names) != flow_matrix.shape[1]:
            raise ValueError("Mismatch between total network components and columns in FlowValue")
    
        n_elements = flow_matrix.shape[1]
    
        # Fixed base index: DC blocks start right after physical UC blocks
        base_index = self.dimensions["UCBlock"]["NumberUnits"]
    
        designlines = self.networkblock["Design"]["DesignLines"]["value"]
        designlines_set = set(np.atleast_1d(designlines).tolist())
    
        i_ext = 0
    
        for i in range(n_elements):
            block_index = base_index + i
            unitblock_name = f"DCNetworkBlock_{block_index}"
            block_type = types[i]
            block_label = "DCNetworkBlock_links" if block_type == "link" else "DCNetworkBlock_lines"
    
            if unitblock_name not in self.unitblocks:
                entry = {
                    "enumerate": f"UnitBlock_{block_index}",
                    "block": block_label,
                    "name": names[i],
                }
    
                if block_type == "link":
                    eff_list = links_effs.get(names[i], None)
                    if eff_list is None:
                        eff_list = [1.0] + [0.0] * max(0, max_eff_len - 1)
                    entry["Efficiencies"] = eff_list
    
                self.unitblocks[unitblock_name] = entry
    
            if i in designlines_set:
                if design_matrix is None:
                    design_value = self.networkblock["Lines"]["variables"]["MaxPowerFlow"]["value"][i]
                else:
                    if isinstance(design_matrix, np.ndarray):
                        if design_matrix.ndim == 1:
                            design_value = design_matrix[i_ext]
                        elif design_matrix.ndim == 2:
                            design_value = design_matrix[:, i_ext]
                        else:
                            raise ValueError(
                                f"Unexpected DesignValue shape: {design_matrix.shape}"
                            )
                    else:
                        design_value = design_matrix
                i_ext += 1
            else:
                design_value = self.networkblock["Lines"]["variables"]["MaxPowerFlow"]["value"][i]
    
            flow_value = flow_matrix[:, i]
    
            if scenario_name is None:
                self.unitblocks[unitblock_name]["FlowValue"] = flow_value
                self.unitblocks[unitblock_name]["DesignVariable"] = design_value
            else:
                self.unitblocks[unitblock_name].setdefault("scenarios", {})
                self.unitblocks[unitblock_name]["scenarios"].setdefault(scenario_name, {})
                self.unitblocks[unitblock_name]["scenarios"][scenario_name]["FlowValue"] = flow_value
                self.unitblocks[unitblock_name]["scenarios"][scenario_name]["DesignVariable"] = design_value
        
        
    def prepare_dc_unitblock_info(self, n):
        """
        Return the (names, types) for DCNetworkBlock unitblocks.
        Prefer the physical view from self._dc_index, which matches FlowValue columns.
        """
        if hasattr(self, "_dc_index") and self._dc_index and "physical" in self._dc_index:
            names = list(self._dc_index["physical"]["names"])
            types = list(self._dc_index["physical"]["types"])
            return names, types
    
        num_lines = self.dimensions["NetworkBlock"]["Lines"]
        num_links = self.dimensions["NetworkBlock"]["Links"]
    
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
        types = (["line"] * num_lines) + (["link"] * num_links)
        return names, types




###########################################################################################################################
############ INVERSE TRANSFORMATION INTO XARRAY DATASET ###################################################################
###########################################################################################
   
    
    def inverse_transformation(self, objective_smspp, n):
        """
        Performs the inverse transformation from the SMS++ blocks to xarray object.
        The xarray will be converted in a solution type Linopy file to get n.optimize().
    
        Parameters
        ----------
        objective_smspp : float
            The objective function value of the SMS++ problem.
        n : pypsa.Network
            A PyPSA network instance from which the data will be extracted.
        """
        if self.problem_structure.get("is_stochastic", False):
            self._inverse_transformation_stochastic(objective_smspp, n)
        else:
            self._inverse_transformation_deterministic(objective_smspp, n)
    
    
    def _inverse_transformation_deterministic(self, objective_smspp, n):
        """
        Existing deterministic inverse transformation.
        """
        all_dataarrays = self.iterate_blocks(n)
        self.ds = xr.Dataset(all_dataarrays)
    
        prepare_solution(
            n,
            self.ds,
            objective_smspp,
            is_stochastic=False,
        )
    
        n.optimize.assign_solution()
        # n.optimize.assign_duals(n)  # Still doesn't work
    
        n._multi_invest = 0
        n._objective_constant = 0
    
    
    def _inverse_transformation_stochastic(self, objective_smspp, n):
        """
        Stochastic inverse transformation.
    
        Builds an xarray.Dataset whose variables mimic a PyPSA stochastic model,
        i.e. operational variables with dimensions including 'scenario' and
        design variables optionally duplicated over scenarios.
        """
        all_dataarrays = self.iterate_blocks_stochastic(n)
        self.ds = xr.Dataset(all_dataarrays)
    
        prepare_solution(
            n,
            self.ds,
            objective_smspp,
            is_stochastic=True,
        )
    
        n.optimize.assign_solution()
    
        n._multi_invest = 0
        n._objective_constant = 0
    
        # Post-processing is mostly scenario-compatible in PyPSA.
        # If something breaks here, this is the first thing to temporarily disable
        # while debugging variable assignment.
        n.optimize.post_processing()
        
        
    
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

    def iterate_blocks_stochastic(self, n):
        """
        Stochastic path: build PyPSA-like DataArrays with an additional 'scenario'
        dimension whenever scenario-wise results are available.
        """
        datasets = []
    
        for name, unit_block in self.unitblocks.items():
            component = component_definition(n, unit_block)
    
            if "scenarios" in unit_block:
                dataarrays = block_to_dataarrays_stochastic(
                    n=n,
                    name=name,
                    unit_block=unit_block,
                    component=component,
                    config=self.config,
                    problem_structure=self.problem_structure,
                    block_to_dataarrays_func=block_to_dataarrays,
                )
            else:
                # Fallback for blocks that stayed deterministic-looking
                dataarrays = block_to_dataarrays(
                    n,
                    name,
                    unit_block,
                    component,
                    self.config,
                )
    
            if dataarrays:
                datasets.append(xr.Dataset(dataarrays))
    
        if not datasets:
            return {}
    
        ds = xr.merge(datasets, join="outer", compat="no_conflicts")
        ds = broadcast_static_variables_over_scenarios(
            ds,
            self.problem_structure.get("scenario_names", []),
        )
    
        return dict(ds.data_vars)



#########################################################################################
######################## Conversion with PySMSpp ########################################
#########################################################################################
    
    ## Create SMSNetwork
    def convert_to_blocks(self):
        """
        Build the SMSNetwork hierarchy depending on:
        - deterministic vs stochastic
        - UC-only vs InvestmentBlock + UCBlock
        """
        sn = SMSNetwork(file_type=SMSFileType.eBlockFile)
        master = sn
        index_id = 0
        inside_tssb = False
    
        # --------------------------------------------------
        # Optional outer stochastic layer
        # --------------------------------------------------
        if self.problem_structure.get("is_stochastic", False):
            stochastic_type = self.problem_structure.get("stochastic_type", None)
    
            if stochastic_type != "tssb":
                raise ValueError(
                    f"Unsupported stochastic_type in convert_to_blocks: {stochastic_type!r}"
                )
    
            self.convert_to_tssb(master, index_id=0, name_id="Block_0")
    
            # Move master to the inner block container of StochasticBlock
            master = sn.blocks["Block_0"].blocks["StochasticBlock"]
            index_id = 0
            inside_tssb = True
    
        # --------------------------------------------------
        # Deterministic investment / UC nesting
        # --------------------------------------------------
        if self.problem_structure.get("has_investment_block", False):
            name_id = "InvestmentBlock"
            self.convert_to_investmentblock(master, index_id, name_id)
    
            master = master.blocks[name_id]
            index_id += 1
            name_id = "InnerBlock"
        else:
            name_id = "Block" if inside_tssb else "Block_0"
    
        # --------------------------------------------------
        # UCBlock always present
        # --------------------------------------------------
        self.convert_to_ucblock(master, index_id, name_id)
    
        self.sms_network = sn
        return sn
    
    def convert_to_tssb(self, master, index_id, name_id):
        """
        Add a TwoStageStochasticBlock to the SMSNetwork hierarchy.
    
        Structure:
        TwoStageStochasticBlock
        ├── DiscreteScenarioSet
        ├── StaticAbstractPath
        └── StochasticBlock
        """
        dims = self.dimensions["tssb"]["dss"]
        number_scenarios = dims["NumberScenarios"]
    
        master.add(
            "TwoStageStochasticBlock",
            name_id,
            id=f"{index_id}",
            NumberScenarios=Dimension("NumberScenarios", number_scenarios),
        )
    
        tssb_block = master.blocks[name_id]
    
        self.convert_to_discrete_scenario_set(tssb_block, "DiscreteScenarioSet")
        self.convert_to_static_abstract_path(tssb_block, "StaticAbstractPath")
        self.convert_to_stochastic_block(tssb_block, "StochasticBlock")
    
        return master
    
    def convert_to_discrete_scenario_set(self, master, name_id="DiscreteScenarioSet"):
        """
        Add the DiscreteScenarioSet block to a TSSB block.
        """
        dss_data = self.tssb_data["discrete_scenario_set"]
        dims = self.dimensions["tssb"]["dss"]
    
        dss_block = Block(
            block_type="DiscreteScenarioSet",
            NumberScenarios=Dimension("NumberScenarios", dims["NumberScenarios"]),
            ScenarioSize=Dimension("ScenarioSize", dims["ScenarioSize"]),
            Scenarios=Variable(
                "Scenarios",
                "double",
                ("NumberScenarios", "ScenarioSize"),
                dss_data["scenarios"],
            ),
            PoolWeights=Variable(
                "PoolWeights",
                "double",
                ("NumberScenarios",),
                dss_data["pool_weights"],
            ),
        )
    
        master.add_block(name_id, block=dss_block)
        return master
    
    
    def convert_to_static_abstract_path(self, master, name_id="StaticAbstractPath"):
        """
        Add the StaticAbstractPath block to a TSSB block.
        """
        sap_data = self.tssb_data["static_abstract_path"]
        dims = self.dimensions["tssb"]["sap"]
    
        sap_block = Block(
            block_type="AbstractPath",
            PathDim=Dimension("PathDim", dims["PathDim"]),
            TotalLength=Dimension("TotalLength", dims["TotalLength"]),
            PathStart=Variable(
                "PathStart",
                "u4",
                ("PathDim",),
                sap_data["PathStart"],
            ),
            PathNodeTypes=Variable(
                "PathNodeTypes",
                "c",
                ("TotalLength",),
                sap_data["PathNodeTypes"],
            ),
            PathGroupIndices=Variable(
                "PathGroupIndices",
                "str",
                ("TotalLength",),
                sap_data["PathGroupIndices"],
            ),
            PathElementIndices=Variable(
                "PathElementIndices",
                "u4",
                ("TotalLength",),
                sap_data["PathElementIndices"],
            ),
            PathRangeIndices=Variable(
                "PathRangeIndices",
                "u4",
                ("TotalLength",),
                sap_data["PathRangeIndices"],
            ),
        )
    
        master.add_block(name_id, block=sap_block)
        return master
    
    
    def convert_to_stochastic_block(self, master, name_id="StochasticBlock"):
        """
        Add the StochasticBlock to a TSSB block.
        """
        sb_data = self.tssb_data["stochastic_block"]
        dims = self.dimensions["tssb"]["sb"]
    
        sb_block = Block(
            block_type="StochasticBlock",
            NumberDataMappings=Dimension(
                "NumberDataMappings",
                dims["NumberDataMappings"],
            ),
            SetSize_dim=Dimension("SetSize_dim", dims["SetSize_dim"]),
            SetElements_dim=Dimension("SetElements_dim", dims["SetElements_dim"]),
            FunctionName=Variable(
                "FunctionName",
                "str",
                ("NumberDataMappings",),
                sb_data["FunctionName"],
            ),
            Caller=Variable(
                "Caller",
                "c",
                ("NumberDataMappings",),
                sb_data["Caller"],
            ),
            DataType=Variable(
                "DataType",
                "c",
                ("NumberDataMappings",),
                sb_data["DataType"],
            ),
            SetSize=Variable(
                "SetSize",
                "u4",
                ("SetSize_dim",),
                sb_data["SetSize"],
            ),
            SetElements=Variable(
                "SetElements",
                "u4",
                ("SetElements_dim",),
                sb_data["SetElements"],
            ),
        )
    
        self.add_sb_abstract_path(sb_block, sb_data["AbstractPath"])
    
        master.add_block(name_id, block=sb_block)
        return master
    
    def add_sb_abstract_path(self, sb_block, ap_data, name_id="AbstractPath"):
        """
        Add the AbstractPath block inside a StochasticBlock.
        """
        ap_block = Block(
            block_type="AbstractPath",
            PathDim=Dimension("PathDim", ap_data["PathDim"]),
            TotalLength=Dimension("TotalLength", ap_data["TotalLength"]),
            PathStart=Variable(
                "PathStart",
                "u4",
                ("PathDim",),
                ap_data["PathStart"],
            ),
            PathNodeTypes=Variable(
                "PathNodeTypes",
                "c",
                ("TotalLength",),
                ap_data["PathNodeTypes"],
            ),
            PathGroupIndices=Variable(
                "PathGroupIndices",
                "u4",
                ("TotalLength",),
                ap_data["PathGroupIndices"],
            ),
            PathElementIndices=Variable(
                "PathElementIndices",
                "u4",
                ("TotalLength",),
                ap_data["PathElementIndices"],
            ),
            PathRangeIndices=Variable(
                "PathRangeIndices",
                "u4",
                ("TotalLength",),
                ap_data["PathRangeIndices"],
            ),
        )
    
        sb_block.add_block(name_id, block=ap_block)
        return sb_block
    
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
        # Optionally add DesignNetworkBlock (only in capacity_expansion_ucblock mode)
        # -----------------
        self.convert_to_designnetworkblock(master, name_id)
    
        # -----------------
        # Done
        # -----------------
        return master
    
    
    def convert_to_designnetworkblock(self, master, ucblock_name):
        """
        Optionally adds a DesignNetworkBlock inside the UCBlock, used when
        capacity_expansion_ucblock is active and design lines are present.
    
        Parameters
        ----------
        master : SMSNetwork
            The SMSNetwork object containing the UCBlock.
        ucblock_name : str
            The name_id of the UCBlock inside master.blocks.
        """
    
        # Condition: only in expansion-ucblock mode AND if we actually have design lines
        if not self.capacity_expansion_ucblock:
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


#############################################################################################
################################ Optimize ##########################################
#############################################################################################

    
    def _optimize(self):
        """
        Optimize the already-built SMSNetwork.
    
        Solver/config selection is based on the top-level problem structure:
        - deterministic UCBlock
        - deterministic InvestmentBlock
        - stochastic TwoStageStochasticBlock
        """
    
        if self.sms_network is None:
            raise ValueError("SMSNetwork not initialized.")
    
        # --------------------------------------------------
        # Decide top-level block type for solver selection
        # --------------------------------------------------
        if self.problem_structure.get("is_stochastic", False):
            stochastic_type = self.problem_structure.get("stochastic_type", None)
    
            if stochastic_type != "tssb":
                raise ValueError(
                    f"Unsupported stochastic_type in optimize: {stochastic_type!r}"
                )
    
            block_type = "TwoStageStochasticBlock"
            inner_block_name = "Block_0"
    
        elif self.problem_structure.get("has_investment_block", False):
            block_type = "InvestmentBlock"
            inner_block_name = "InvestmentBlock"
    
        else:
            block_type = "UCBlock"
            inner_block_name = "Block_0"
    
        # --------------------------------------------------
        # Resolve configfile/template
        # --------------------------------------------------
        default_template_map = {
            "UCBlock": "UCBlock/uc_solverconfig.txt",
            "InvestmentBlock": "InvestmentBlock/BSPar.txt",
            "TwoStageStochasticBlock": "TSSBlock/TSSBSCfg.txt",
        }
    
        cfg = getattr(self, "configfile", "auto")
    
        if cfg is None or cfg == "auto":
            if block_type not in default_template_map:
                raise ValueError(
                    f"No default config template is defined for block type {block_type!r}. "
                    f"Please provide self.configfile explicitly."
                )
            template = default_template_map[block_type]
            configfile = pysmspp.SMSConfig(template=str(template))
        else:
            if isinstance(cfg, pysmspp.SMSConfig):
                configfile = cfg
            else:
                configfile = pysmspp.SMSConfig(template=str(cfg))
    
        # --------------------------------------------------
        # Workdir and filepaths
        # --------------------------------------------------
        workdir = Path(self.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    
        fp_temp = str(workdir / str(self.fp_temp).format(name=self.name))
        fp_log = None if self.fp_log is None else str(workdir / str(self.fp_log).format(name=self.name))
        fp_solution = None if self.fp_solution is None else str(workdir / str(self.fp_solution).format(name=self.name))
    
        # --------------------------------------------------
        # Overwrite policy
        # --------------------------------------------------
        if self.overwrite:
            for p in (fp_temp, fp_log, fp_solution):
                if p is None:
                    continue
                pp = Path(p)
                if pp.exists():
                    pp.unlink()
    
        # --------------------------------------------------
        # Solver options
        # --------------------------------------------------
        solver_options = dict(self.pysmspp_options or {})
    
        self.result = self.sms_network.optimize(
            configfile=configfile,
            fp_temp=fp_temp,
            fp_log=fp_log,
            fp_solution=fp_solution,
            inner_block_name=inner_block_name,
            **solver_options,
        )
    
        return self.result


#############################################################################################
################################ Consistency check ##########################################
#############################################################################################

    def consistency_check(self, n):
        """
        Validate configuration + network compatibility before running the pipeline.
    
        Notes
        -----
        Keep this cheap and deterministic. Fail fast with clear error messages.
        """
    
        # ---- Basic type checks ----
        if not isinstance(self.merge_links, bool):
            raise TypeError("merge_links must be a boolean.")
        if not isinstance(self.capacity_expansion_ucblock, bool):
            raise TypeError("capacity_expansion_ucblock must be a boolean.")
    
        # ---- Describe high-level problem structure ----
        self.problem_structure = describe_problem_structure(
            n,
            capacity_expansion_ucblock=self.capacity_expansion_ucblock,
            stochastic_parameters=self.stochastic_parameters,
        )
    
        # ---- Minimal stochastic consistency checks ----
        if self.problem_structure["is_stochastic"]:
            if self.problem_structure["stochastic_type"] is None:
                raise ValueError(
                    "The network is stochastic but no stochastic_type was provided "
                    "in stochastic_parameters."
                )
    
            if self.problem_structure["stochastic_type"] != "tssb":
                raise ValueError(
                    f"Unsupported stochastic type: "
                    f"{self.problem_structure['stochastic_type']!r}"
                )
    
            if self.problem_structure["number_scenarios"] <= 0:
                raise ValueError(
                    "The network is marked as stochastic but no scenarios were found."
                )
    
            if not hasattr(n, "get_scenario"):
                raise ValueError(
                    "The network is marked as stochastic but does not expose "
                    "'get_scenario'."
                )
    
            if not (
                self.problem_structure["stochastic_demand"]
                or self.problem_structure["stochastic_price"]
                or self.problem_structure["stochastic_renewables"]
            ):
                raise ValueError(
                    "The network is stochastic but no stochastic parameter was declared. "
                    "Set stochastic_parameters={'stochastic_type': 'tssb', "
                    "'parameters': [...]}."
                )
    
        return True

#############################################################################################
############################## TSSB methods #################################################
#############################################################################################
    

    def prepare_tssb_interface(self, n):
        """
        Prepare internal data structures needed for a TwoStageStochasticBlock (TSSB).

        This method does not assemble SMS++ blocks yet. It only computes and stores:
        - TSSB-related dimensions
        - DiscreteScenarioSet payload
        - design-variable descriptors
        - a preliminary StaticAbstractPath
        - a minimal demand-only StochasticBlock mapping
        """

        if not self.problem_structure.get("is_stochastic", False):
            return None

        if self.problem_structure.get("stochastic_type") != "tssb":
            raise ValueError(
                f"prepare_tssb_interface only supports 'tssb', got "
                f"{self.problem_structure.get('stochastic_type')!r}."
            )
        
        #TODO build demand node by node instead of node0, node1, node0, node1
        dss_data = self.build_tssb_dss(n)
        
        design_variables = self._collect_design_variables()
        sap_data = build_tssb_static_abstract_path(design_variables)
        
        self.dimensions["tssb"]["sap"] = {
            "PathDim": sap_data["PathDim"],
            "TotalLength": sap_data["TotalLength"],
        }

        stochastic_block = self.build_tssb_stochastic_block()

        self.tssb_data = {
            "enabled": True,
            "discrete_scenario_set": dss_data,
            "static_abstract_path": sap_data,
            "stochastic_block": stochastic_block,
        }

        return self.tssb_data
    
    def build_tssb_dss(self, n):
        """
        Build the payload for the DiscreteScenarioSet of a TSSB problem.
    
        The stochastic sources are selected from self.problem_structure.
        For now only stochastic demand is implemented.
        """
        dss_parts = []
    
        if self.problem_structure.get("stochastic_demand", False):
            dss_parts.append(build_dss_demand(n))
    
        if self.problem_structure.get("stochastic_marginal", False):
            dss_parts.append(build_dss_marginal(n))
    
        if self.problem_structure.get("stochastic_renewables", False):
            dss_parts.append(build_dss_renewables(n))
    
        dss_data = merge_tssb_dss_parts(dss_parts)
    
        self.dimensions.setdefault("tssb", {})
        self.dimensions["tssb"]["dss"] = {
            "NumberScenarios": int(dss_data["number_scenarios"]),
            "ScenarioSize": int(dss_data["scenario_size"]),
        }
    
        return dss_data


    def _store_unitblock_design_variable(self, attr_name, unitblock_index):
        """
        Store design-variable metadata for StaticAbstractPath construction.

        Parameters
        ----------
        attr_name : str
            Unit block parameter family name.
        unitblock_index : int
            Unit block index in the SMS++ structure.
        """
        block_type = attr_name.split("_")[0]

        if block_type == "BatteryUnitBlock":
            self.unitblock_design_data.append(
                {
                    "block_index": unitblock_index,
                    "var_name": "x_battery",
                    "component_type": "unit",
                    "element_index": 0,
                    "range_index": 1,
                }
            )
            self.unitblock_design_data.append(
                {
                    "block_index": unitblock_index,
                    "var_name": "x_converter",
                    "component_type": "unit",
                    "element_index": 0,
                    "range_index": 1,
                }
            )

        elif block_type == "ThermalUnitBlock":
            self.unitblock_design_data.append(
                {
                    "block_index": unitblock_index,
                    "var_name": "x_thermal",
                    "component_type": "unit",
                    "element_index": 0,
                    "range_index": 1,
                }
            )

        elif block_type == "IntermittentUnitBlock":
            self.unitblock_design_data.append(
                {
                    "block_index": unitblock_index,
                    "var_name": "x_intermittent",
                    "component_type": "unit",
                    "element_index": 0,
                    "range_index": 1,
                }
            )

        elif block_type == "HydroUnitBlock":
            self.unitblock_design_data.append(
                {
                    "block_index": unitblock_index,
                    "var_name": "x_hydro",
                    "component_type": "unit",
                    "element_index": 0,
                    "range_index": 1,
                }
            )

        else:
            self.unitblock_design_data.append(
                {
                    "block_index": unitblock_index,
                    "var_name": "x_design",
                    "component_type": "unit",
                    "element_index": 0,
                    "range_index": 1,
                }
            )
            
            
    def _collect_design_variables(self):
        """
        Collect design-variable descriptors for the TSSB StaticAbstractPath.
    
        Design variables are gathered during iterate_components for unit blocks
        and reconstructed from investment_meta['design_lines'] for the network.
        """
        investment_meta = {
            "design_lines": list(
                self.networkblock.get("Design", {})
                .get("DesignLines", {})
                .get("value", [])
            )
        }
        
        network_block_index = len(self.unitblocks)
    
        self.design_variables = calculate_design_variables(
            investment_meta=investment_meta,
            unitblock_design_data=self.unitblock_design_data,
            network_block_index=network_block_index,
        )
        return self.design_variables

    
    def build_tssb_stochastic_block(self):
        """
        Build the StochasticBlock payload for TSSB.

        Notes
        -----
        For now only stochastic demand is implemented.
        Future stochastic mappings (e.g. marginal costs, renewables)
        can be appended to the data_mappings list.
        """
        # TODO:
        # Future mappings for marginal costs and renewables may require one mapping
        # per modified unit / variable, depending on how the scenario vector is encoded.
        
        data_mappings = []

        if self.problem_structure.get("stochastic_demand", False):
            data_mappings.append(
                build_stochastic_mapping_demand(
                    scenario_size=self.dimensions['UCBlock']['TimeHorizon']
                )
            )

        if self.problem_structure.get("stochastic_marginal", False):
            # TODO: add mapping(s) for set_linear when implemented
            pass

        if self.problem_structure.get("stochastic_renewables", False):
            # TODO: add mapping(s) for set_max_power when implemented
            pass

        if not data_mappings:
            raise ValueError(
                "No stochastic data mappings were built for the TSSB StochasticBlock."
            )

        stochastic_block = build_tssb_stochastic_block_data(data_mappings)

        self.dimensions["tssb"]["sb"] = {
            "NumberDataMappings": stochastic_block["NumberDataMappings"],
            "SetSize_dim": int(stochastic_block["SetSize"].shape[0]),
            "SetElements_dim": int(stochastic_block["SetElements"].shape[0]),
            "AbstractPath": {
                "PathDim": int(stochastic_block["AbstractPath"]["PathDim"]),
                "TotalLength": int(stochastic_block["AbstractPath"]["TotalLength"]),
            },
        }

        return stochastic_block
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


    def __repr__(self):
        return (
            f"Transformation(pypsa->sms++, "
            f"stochastic={self.problem_structure.get('is_stochastic', False)}, "
            f"stochastic_type={self.problem_structure.get('stochastic_type', None)}, "
            f"has_investment_block={self.problem_structure.get('has_investment_block', None)})"
        )
    
    __str__ = __repr__