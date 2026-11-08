# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 08:12:37 2024

@author: aless
"""

import pandas as pd
import numpy as np
import xarray as xr
import os

from pypsa2smspp.multistage_utils import slice_network_for_stage
from pypsa2smspp.transformation_config import TransformationConfig
from pysmspp import SMSNetwork, SMSFileType, Attribute, Dimension, Variable, Block, SMSConfig
from pypsa2smspp import logger
import pysmspp
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union, Literal, Callable, List
import warnings

from pypsa2smspp.constants import (
    conversion_dict,
    nominal_attrs,
    renewable_carriers,
    nuclear_rules_default,
    read_nuclear_rules,
    STOCHASTIC_PARAMETER_REGISTRY,
)

from pypsa2smspp.utils import (
    is_extendable,
    filter_extendable_components,
    get_bus_idx,
    get_nominal_aliases,
    ucblock_dimensions,
    networkblock_dimensions,
    investmentblock_dimensions,
    hydroblock_dimensions,
    add_unit_block,
    hydro_system_unit_solutions,
    hydro_system_mapping_paths,
    get_attr_name,
    process_dcnetworkblock,
    get_block_name,
    parse_unitblock_parameters,
    nuclear_rule_variables,
    conversion_links,
    conversion_unit_variables,
    bus_carriers,
    carrier_unit_links,
    conversion_carrier_links,
    link_ports,
    forbid_unreachable_switches,
    free_initial_ramp,
    fix_commitment_on,
    determine_size_type,
    merge_lines_and_links,
    rename_links_to_lines,
    build_store_and_merged_links,
    correct_dimensions,
    explode_multilinks_into_branches,
    add_sectorcoupled_parameters,
    apply_expansion_overrides,
    apply_investment_module_overrides,
    build_dc_index,
    get_param_as_dense,
    pollutant_budget_data,
    ucblock_variables,
    fold_fom_cost_into_capital_cost,
    preprocess_zero_capital_cost_extendable_generators,
    preprocess_zero_capital_cost_extendable_lines_links,
    get_bus_demand_matrix,
    preprocess_dynamic_link_parameters_to_static_means,
    apply_time_dependent_link_data_to_lines
)

from pypsa2smspp.pip_utils import (
    StepTimer,
    step,
)

from pypsa2smspp.inverse import (
    component_definition,
    block_to_dataarrays,
    normalize_key,
    evaluate_function,
    dataarray_components,
    block_to_dataarrays_stochastic,
    broadcast_static_variables_over_scenarios,
    merge_by_variable,
)
from pypsa2smspp.io_parser import (
    parse_txt_to_unitblocks,
    assign_design_variables_to_unitblocks,
    prepare_solution,
    split_merged_dcnetworkblocks
)

from pypsa2smspp.stochastic_utils import (
    get_base_scenario_network,
    describe_problem_structure,
    build_dss_demand,
    build_dss_unitblock_timeseries_parameter,
    merge_tssb_dss_parts,
    calculate_design_variables,
    build_tssb_static_abstract_path,
    build_stochastic_mapping_demand,
    build_stochastic_mapping_single_unit,
    build_tssb_stochastic_block_data,
    get_stochastic_parameter_asset_names,
    collect_unitblock_indices_by_names_and_type,
)

from pypsa2smspp import multistage_utils as mu

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
        merge_links: Union[bool, str, Sequence[str]] = False,
        merge_selector: Optional[Callable[..., bool]] = None,
        capacity_expansion_ucblock: bool = True,
        enable_thermal_units: bool = False,
        intermittent_carriers: Optional[Union[str, Sequence[str]]] = None,
        nuclear_units: Optional[Mapping[str, Union[bool, Mapping[str, Any], str, Path]]] = None,
        hydro_system: bool = False,
        enable_conversion_units: bool = False,
        multi_carrier: bool = False,

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
        merge_links : bool | str | Sequence[str], default False
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

        nuclear_units : Mapping[str, bool | Mapping[str, Any] | str | Path], optional
            Carriers (case-insensitive) whose thermal generators become
            NuclearUnitBlocks, i.e., ThermalUnitBlocks subject to the
            operating rules of a load-following nuclear unit. Each carrier
            maps to True, which applies the rules of
            `data/nuclear_rules.yaml`, to the path of a YAML file written as
            that one, or to a mapping; a file or a mapping may give some of
            the rules only, the others keeping their default (a rule set to
            None is not emitted). Durations are in hours, and need a uniform
            snapshot weighting.

            Notes:
              - Ignored when `enable_thermal_units=False`.

        hydro_system : bool, default False
            If True, the hydro StorageUnits (carrier hydro or PHS), instead
            of being HydroUnitBlocks of the UCBlock, are the HydroUnitBlock_j
            of a single HydroSystemUnitBlock, which also holds the future
            cost of the water left in all the reservoirs at the end of the
            horizon (a PolyhedralFunctionBlock with one active Variable per
            reservoir and no cut). This is the form of the UCBlock of each
            stage of an SDDPBlock, whose cuts are rows of that function.

            Notes:
              - The function written has no row and the data of
                `TransformationConfig.HydroSystemUnitBlock_parameters`, by
                default the bound 0: it adds nothing to the cost.
              - A MILPSolver needs the "linearized" PolyhedralFunctionBlock,
                which the InnerBCfg.txt of smspp_ucblock_solver asks for: to
                solve the UCBlock directly, the `configfile` has to be in a
                directory holding it, as the installed ucblock_solver/config.
              - The hydro StorageUnits have to come after the other
                StorageUnits, so that the generators of the
                HydroSystemUnitBlock keep their place, and they cannot be
                extendable.
              - The DataMappings of a stochastic parameter of a hydro unit
                (e.g., `hydro_inflow`) reach its HydroUnitBlock inside the
                HydroSystemUnitBlock.

        enable_conversion_units : bool, default False
            Controls whether the committable links become ConversionUnitBlocks
            (and, with `multi_carrier`, the links between carriers that a
            unit with one generator cannot express, see below).

            Behaviour:
              - False: every link is a (hyper)arc of the network, whose
                commitment is not represented.
              - True : a committable link is a ConversionUnitBlock, a unit
                with one commitment and one generator per port: the input
                on bus0, whose power is - p0, and an output on each bus1,
                bus2, ... with a nonzero efficiency, whose power is the
                efficiency times p0, as equality rows of its operating
                region. The other links are (hyper)arcs as before.

            Notes:
              - An efficiency that changes over time, a delay, the modules
                of a modular link and the design of an extendable one are
                translated (see conversion_unit_variables() for what is
                refused).

        multi_carrier : bool, default False
            Controls whether the carriers of the buses are carriers of the
            UCBlock.

            Behaviour:
              - False: every bus is a node of the one network of the
                UCBlock, whatever its carrier, and a link between buses of
                different carriers is a (hyper)arc of it.
              - True : the electric buses (AC, DC or no carrier) are the
                nodes of the electricity, the carrier 0, and the buses of
                each other carrier those of a carrier of its own (see
                bus_carriers()), whose network has the lines and links
                between them; a link whose ports are on more than one
                carrier is an IntermittentUnitBlock whose power is p0, with
                an output on the carrier of each port (efficiency -1 on bus0
                and efficiency, efficiency2, ... on bus1, bus2, ...), or a
                ConversionUnitBlock if `enable_conversion_units` is True and
                it is committable, has two ports on the same carrier (a
                pump, a compressor) or delays its output, whose generators
                have each its output on the carrier of its bus and its own
                node there (a link that is not committable being a unit on
                throughout the horizon); any other unit has its output on the
                carrier of its bus. The demand of the loads is that of the
                nodes of their carriers.

            Notes:
              - Without `enable_conversion_units`, a link with two ports on
                the same carrier, or that delays its output, is refused, as
                is the merge of the links (`merge_links`).
              - The lines and links of a carrier other than the electricity
                can be neither extendable nor hyperarcs.

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

        stochastic_parameters : Mapping[str, Any], optional
        Dictionary configuring stochastic behaviour.

            For TSSB:
                {
                    "stochastic_type": "tssb",
                    "parameters": ["demand", "renewable_maxpower"]
                }

            For SDDP:
                {
                    "stochastic_type": "sddp",
                    "parameters": ["demand", "renewable_maxpower"],
                    "snapshots_per_stage": 4,
                    "top_path": {
                        "K": 1,
                    },
                }

            SDDP-specific fields:

            - "snapshots_per_stage" (int >= 1, required): number of PyPSA
              snapshots that belong to each SDDP stage. The stages are
              derived by partitioning n.snapshots into contiguous blocks of
              this size. The number of snapshots must be divisible by this
              value.

            - "top_path" (dict, required): parameters for the top-level
              AbstractPath of the SDDPBlock, which points from each
              BendersBlock to the corresponding PolyhedralFunction inside
              the external Block_i.nc4 file. Keys:
                * "K" (int >= 0): NumPolyhedralFunctionsPerSubBlock.
                * "second_group_index" (int >= 0): position of the
                  HydroSystemUnitBlock among the children of the UCBlock
                  inside Block_i.nc4.
                * "third_group_index" (int >= 0): position of the
                  PolyhedralFunctionBlock inside the HydroSystemUnitBlock.

            - "benders_path" (dict, required): parameters for the
              AbstractPath inside each BendersBFunction, which points to
              the state-balance constraints of the reservoirs. Keys:
                * "n_bacini" (int >= 0): number of reservoirs whose state
                  is carried across stages. Use 0 if no state is modelled.

            These values depend on the structure of the external
            Block_i.nc4 files, which are generated by prepare_sddp_interface
            (via generate_stage_blocks).
            N.B. The network must have at least one StorageUnit with hydro carrier (or PHS).
            SDDP wants a state and a PolyhedralFunctionBlock in order to work; without hydro,
            prepare_sddp_interface will raise ValueError.
        """
        # NB: assign a fresh config directly (do NOT deepcopy).
        self.config = TransformationConfig()

        self.merge_links = merge_links
        if merge_selector is not None:
            self.merge_selector = merge_selector

        self.capacity_expansion_ucblock = bool(capacity_expansion_ucblock)
        self.enable_thermal_units = bool(enable_thermal_units)
        self.enable_conversion_units = bool(enable_conversion_units)
        self.multi_carrier = bool(multi_carrier)
        if self.multi_carrier and merge_links:
            raise ValueError(
                "multi_carrier: the links are not merged, each of them "
                "being a unit or a line of its carrier.")
        self.intermittent_carriers = intermittent_carriers
        self.nuclear_units = {}
        for carrier, rules in (nuclear_units or {}).items():
            if rules is False:
                continue
            merged = dict(nuclear_rules_default)
            if isinstance(rules, (str, Path)):
                rules = read_nuclear_rules(rules)
            if isinstance(rules, Mapping):
                unknown = set(rules) - set(nuclear_rules_default)
                if unknown:
                    raise ValueError(
                        f"Unknown nuclear rules for carrier {carrier!r}: "
                        f"{sorted(unknown)}"
                    )
                merged.update(rules)
            self.nuclear_units[str(carrier).strip().lower()] = merged

        self.hydro_system = bool(hydro_system)

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
        self.ucblock_variables = {}

        self.sms_network = None
        self.result = None
        self.problem_structure = {}
        self.tssb_data = None
        self.design_variables = []
        self.stochastic_parameters = dict(stochastic_parameters or {})
        self.unitblock_design_data = []
        self._conversion_links = pd.Index([])
        self._conversion_data = {}
        self._carrier_links = pd.Index([])

        self.sddp_data = None
        self.sddp_stage_blocks = []
        self.sddp_stage_names = []
        self.sddp_state_info = {
            "state_size": 0,
            "initial_state": None,
            "admissible_state": None,
        }
        self.sddp_parameter_specs = {}
        self.sddp_parameter_asset_order = {}
        self.sddp_dss_offsets = {}

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
        n.stores['max_hours'] = self.config.max_hours_stores
        n.storage_units['snapshots_weighting'] = n.snapshot_weightings['stores'].iloc[0]
        n.stores['snapshots_weighting'] = n.snapshot_weightings['stores'].iloc[0]

        n_direct = get_base_scenario_network(n)

        with step(self.timer, "consistency_check", verbose=verbose):
            self.consistency_check(n)

        with step(self.timer, "direct", verbose=verbose):
            self.direct(n_direct)

        with step(self.timer, "prepare_tssb_interface", verbose=verbose):
            self.prepare_tssb_interface(n)

        with step(self.timer, "prepare_sddp_interface", verbose=verbose):
            self.prepare_sddp_interface(n)

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

        # PyPSA non ha un modello dati nativo per SDDP: skip del mapping,
        # l'objective resta in self.result.objective_value.
        if (
                self.problem_structure.get("is_stochastic", False)
                and self.problem_structure.get("stochastic_type") == "sddp"
        ):
            if verbose:
                print(
                    f"[SDDP] Objective: {self.result.objective_value:.6e}. "
                    "Mapping in PyPSA non supportato."
                )
            return n

        with step(self.timer, "parse_solution_to_unitblocks", verbose=verbose):
            self.parse_solution_to_unitblocks(self.result.solution, n)

        with step(self.timer, "inverse_transformation", verbose=verbose):
            self.inverse_transformation(self.result.objective_value, n)


        return n

    def direct(self, n):
        """
        Direct transformation PyPSA -> internal unitblocks.
        """

        # --- your existing logic ---
        self.read_excel_components() # 1
        self.add_dimensions(n) # 2
        # the dense series of an attribute are built once per conversion and
        # shared by all the components of its type [see resolve_param_value()]
        self._dense_cache = {}
        try:
            self.iterate_components(n) # 3
        finally:
            self._dense_cache = None
        self.add_pollutant_budget(n) # 3b
        self.add_demand(n) # 4
        self.lines_links(n) # 5



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
        # the thermal part of a NuclearUnitBlock has the thermal sizes and types
        self.smspp_parameters["NuclearUnitBlock"] = self.smspp_parameters["ThermalUnitBlock"]

    def add_dimensions(self, n):
        """
        Sets the .dimensions attribute with UCBlock, NetworkBlock, InvestmentBlock, HydroBlock dimensions.
        """

        self.dimensions['UCBlock'] = ucblock_dimensions(n)
        self.dimensions['NetworkBlock'] = networkblock_dimensions(n, self.capacity_expansion_ucblock)
        self.dimensions['InvestmentBlock'] = investmentblock_dimensions(
            n , self.capacity_expansion_ucblock , nominal_attrs )
        self.dimensions['HydroUnitBlock'] = hydroblock_dimensions()



    ### 3 ###
    def _initialize_component_iteration_state(self):
        """
        Initialize local state used during component iteration.

        Returns
        -------
        dict
            Dictionary containing mutable state used by iterate_components.
        """
        self.unitblock_design_data = []
        self._dc_names = []
        self._dc_types = []

        return {
            "generator_node": [],
            "generator_owner": [],
            "storage_owner": [],
            "investment_meta": {
                "Blocks": [],
                "index_extendable": [],
                "asset_type": [],
                "design_lines": [],
            },
            "unitblock_index": 0,
            "lines_index": 0,
        }


    def _preprocess_network_for_iteration(self, n):
        """
        Apply preprocessing steps to the input network before iterating over
        components and building SMS++ blocks.

        Parameters
        ----------
        n : pypsa.Network
            Input PyPSA network.

        Returns
        -------
        dict
            Dictionary containing the preprocessed network and auxiliary
            dataframes needed during iteration.
        """
        # the per-unit reactances of the lines, which their susceptances
        # in Kirchhoff's voltage law come from, as PyPSA computes them
        n.calculate_dependent_values()
        # PyPSA counts fom_cost with capital_cost on the capacity of an
        # extendable asset, hence so does everything below
        n = fold_fom_cost_into_capital_cost(n)

        n, fixed_investment_generators = preprocess_zero_capital_cost_extendable_generators(
            n,
            fixed_capacity=1e9,
            update_bounds=True,
            logger=logger,
            return_fixed_count=True,
        )

        # the links that the merge absorbs keep their extendability (in
        # PyPSA-Eur the battery discharger has no capital cost, and fixing it
        # would make the pair no longer mergeable)
        _, links_merge_probe, _ = build_store_and_merged_links(
            n,
            merge_links=self.merge_links,
            logger=lambda msg: None,
            merge_selector=getattr(self, "merge_selector", None),
        )
        links_absorbed = n.links.index.difference(links_merge_probe.index)
        # a committable link that is a ConversionUnitBlock keeps its design,
        # as it is that of the unit, and so does a link between carriers
        # that is one [see conversion_carrier_links()]
        if self.enable_conversion_units:
            links_absorbed = links_absorbed.union(
                conversion_links(links_merge_probe))
            if self.multi_carrier:
                links_absorbed = links_absorbed.union(
                    conversion_carrier_links(n, links_merge_probe,
                                             bus_carriers(n)[1]))

        n, fixed_investment_lines_links = preprocess_zero_capital_cost_extendable_lines_links(
            n,
            fixed_capacity=1e9,
            update_bounds=True,
            logger=logger,
            return_fixed_count=True,
            exclude=links_absorbed,
        )


        # n = preprocess_dynamic_link_parameters_to_static_means(
        #     n,
        #     fields=("efficiency", "p_min_pu", "p_max_pu"),
        #     logger=logger,
        #     drop_dynamic=True
        # )

        stores_df, links_merged_df, self.dimensions["NetworkBlock"]["merged_links_ext"] = build_store_and_merged_links(
            n,
            merge_links=self.merge_links,
            logger=logger,
            merge_selector=getattr(self, "merge_selector", None),
        )
        # the committable links are ConversionUnitBlocks, not lines of the
        # network [see _add_conversion_units()]
        self._conversion_links = (
            conversion_links(links_merged_df) if self.enable_conversion_units
            else links_merged_df.index[:0]
        )
        links_merged_df = links_merged_df.drop(self._conversion_links)
        # with the carriers, the links between two of them are units too
        # [see _add_carrier_link_units()]
        if self.multi_carrier:
            self._carriers, self._bus_node = bus_carriers(n)
            self._line_layout_cache = None
            # those that a unit with one generator cannot express are
            # ConversionUnitBlocks, one generator per port
            if self.enable_conversion_units:
                more = conversion_carrier_links(n, links_merged_df,
                                                self._bus_node)
                self._conversion_links = self._conversion_links.union(more)
                links_merged_df = links_merged_df.drop(more)
            self._carrier_links = carrier_unit_links(n, links_merged_df,
                                                     self._bus_node)
            links_merged_df = links_merged_df.drop(self._carrier_links)
        unit_links = self._conversion_links.union(self._carrier_links)
        # in the UCBlock form an extendable one has the design of its unit,
        # not that of a line
        if self.capacity_expansion_ucblock:
            designs = int(n.links.loc[unit_links,
                                      "p_nom_extendable"].astype(bool).sum())
            self.dimensions["InvestmentBlock"]["NumberDesignLines"] -= designs
            self.dimensions["NetworkBlock"]["NumberDesignLines_links"] -= designs
        network_links = n.links.drop(unit_links)

        links_before = links_merged_df.copy()

        self.ucblock_variables = ucblock_variables(n, links_before)

        has_multilinks = (
            "bus2" in network_links.columns
            and bool((network_links.bus2.notna() & (network_links.bus2.astype(str).str.strip() != "")).any())
        )

        if has_multilinks:
            n.lines["hyper"] = np.arange(0, len(n.lines), dtype=int)

            (
                links_after,
                self.networkblock["efficiencies"],
                self.dimensions["NetworkBlock"]["NumberBranches"],
                self.dimensions["NetworkBlock"]["NumberBranches_ext"],
            ) = explode_multilinks_into_branches(
                links_merged_df,
                len(n.lines),
                logger=logger,
                n=n
            )

            self.networkblock["max_eff_len"] = max(
                (len(v) for v in self.networkblock["efficiencies"].values()),
                default=1,
            )

            add_sectorcoupled_parameters(
                self.config.Lines_parameters,
                self.config.Links_parameters,
                self.config.DCNetworkBlock_links_inverse,
                self.networkblock["max_eff_len"],
            )
        else:
            links_after = links_merged_df.copy()

            if "hyper" not in links_after.columns:
                links_after["hyper"] = np.arange(
                    len(n.lines),
                    len(n.lines) + len(links_after),
                    dtype=int,
                )

            if "is_primary_branch" not in links_after.columns:
                links_after["is_primary_branch"] = True

        correct_dimensions(
            self.dimensions,
            stores_df,
            links_merged_df,
            n,
            self.capacity_expansion_ucblock,
            fixed_investment_generators=fixed_investment_generators,
            fixed_investment_lines_links=fixed_investment_lines_links,
        )
        # each committable link is one unit with a generator per port, and
        # correct_dimensions() has taken it out of the lines as it does with
        # a merged link
        self._conversion_data = {
            name: conversion_unit_variables(
                n, name, scaled=not self.capacity_expansion_ucblock)
            for name in self._conversion_links
        }
        self.dimensions["UCBlock"]["NumberUnits"] += len(self._conversion_data)
        self.dimensions["UCBlock"]["NumberElectricalGenerators"] += sum(
            len(buses) for _, _, buses in self._conversion_data.values()
        )
        # and each link between carriers one unit with one generator
        self.dimensions["UCBlock"]["NumberUnits"] += len(self._carrier_links)
        self.dimensions["UCBlock"]["NumberElectricalGenerators"] += len(
            self._carrier_links)
        self._dc_index = build_dc_index(n, links_before, links_after)

        # TODO remove when no longer needed
        self._dc_names = list(self._dc_index["physical"]["names"])
        self._dc_types = list(self._dc_index["physical"]["types"])
        self._investment_modules = False
        if self.capacity_expansion_ucblock:
            apply_expansion_overrides(
                self.config.IntermittentUnitBlock_parameters,
                self.config.BatteryUnitBlock_store_parameters,
                self.config.IntermittentUnitBlock_inverse,
                self.config.BatteryUnitBlock_inverse,
                self.config.InvestmentBlock_parameters,
                self.config.BatteryUnitBlock_parameters,
                self.config.BatteryUnitBlock_StorageUnit_inverse,
            )
        elif self._has_investment_modules(n):
            # the InvestmentBlock writes the modular extendable assets as
            # modules too, with an integer design
            apply_investment_module_overrides(
                self.config.IntermittentUnitBlock_parameters,
                self.config.IntermittentUnitBlock_inverse,
                self.config.InvestmentBlock_parameters,
            )
            self._investment_modules = True
        return {
            "n": n,
            "stores_df": stores_df,
            "links_after": links_after,
        }


    def iterate_components(self, n):
        """
        Iterates over the network components and adds them as unit blocks.
        """


        state = self._initialize_component_iteration_state()
        prep = self._preprocess_network_for_iteration(n)


        n = prep["n"]
        # the network the UnitBlocks are built from, which the data of the
        # scenarios is evaluated on [see build_tssb_dss()]
        self.n_preprocessed = n

        stores_df = prep["stores_df"]
        links_after = prep["links_after"]

        generator_node = state["generator_node"]
        generator_owner = state["generator_owner"]
        storage_owner = state["storage_owner"]
        investment_meta = state["investment_meta"]
        unitblock_index = state["unitblock_index"]
        lines_index = state["lines_index"]

        for components in n.components[["Generator", "Store", "StorageUnit", "Line", "Link"]]:
            # the ConversionUnitBlocks are the last units, before the lines
            if components.list_name == "lines":
                unitblock_index = self._add_conversion_units(
                    n, generator_node, generator_owner, investment_meta,
                    unitblock_index)
                unitblock_index = self._add_carrier_link_units(
                    n, generator_node, generator_owner, investment_meta,
                    unitblock_index)
            if components.empty:
                continue

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
            # all the links may be units, and then the network has none
            if components_type == "links" and components_df.empty \
                    and len(self._conversion_links.union(
                        self._carrier_links)):
                continue
            use_investmentblock = (
                not self.capacity_expansion_ucblock
                or components_type in ["lines", "links"]
            )

            if use_investmentblock:
                df_investment = self.add_InvestmentBlock(n, components_df, components.name)

            if components_type in ["lines", "links"]:
                self._dc_names.extend(list(components_df.index))
                self._dc_types.extend(
                    ["line" if components_type == "lines" else "link"] * len(components_df)
                )

                get_bus_idx(
                    n,
                    components_df,
                    [components_df.bus0, components_df.bus1],
                    ["start_line_idx", "end_line_idx"],
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

            elif components_type == "storage_units":
                get_bus_idx(n, components_df, components_df.bus, "bus_idx")
                # a battery pays its marginal cost on the dispatch only, which
                # a price on the net injection gives only if it never draws
                battery = ~components_df.carrier.isin(["hydro", "PHS"])
                drawing = (n.get_switchable_as_dense("StorageUnit", "p_min_pu") < 0).any().reindex(components_df.index, fill_value=False)
                priced = (n.get_switchable_as_dense("StorageUnit", "marginal_cost") != 0).any().reindex(components_df.index, fill_value=False)
                for name in components_df.index[battery & drawing & priced]:
                    logger.warning(
                        f"StorageUnit {name} can draw and has a marginal cost: "
                        f"it is written as a price on the net injection, "
                        f"which the unit also earns when it draws."
                    )
                for name, bus, carrier in zip(components_df.index,
                                              components_df["bus_idx"].values,
                                              components_df["carrier"]):
                    # the turbine, the pump and the spillway of a hydro unit
                    k = 3 if carrier in ["hydro", "PHS"] else 1
                    generator_node.extend([bus] * k)
                    generator_owner.extend([(components_type, name)] * k)
                    # one storage: the charge of a battery, the reservoir of
                    # a hydro unit
                    storage_owner.append((components_type, name))
    
            else:
                get_bus_idx(n, components_df, components_df.bus, "bus_idx")
                generator_node.extend(components_df["bus_idx"].values)
                generator_owner.extend(
                    (components_type, name) for name in components_df.index
                )
                if components_type == "stores":
                    storage_owner.extend(
                        (components_type, name) for name in components_df.index
                    )

            for component in components_df.index:
                carrier = (
                    components_df.loc[component].carrier
                    if "carrier" in components_df.columns
                    else None
                )

                attr_name = get_attr_name(
                    components.name,
                    carrier,
                    enable_thermal_units=self.enable_thermal_units,
                    intermittent_carriers=self.intermittent_carriers,
                    default_intermittent=renewable_carriers,
                    nuclear_carriers=list(self.nuclear_units),
                )

                # UCBlock cannot expand a thermal unit: its commitment u_t is a
                # binary variable and a continuous design would multiply it in
                # the maximum power constraint, making the model bilinear. Such
                # a unit would be normalized to a 1 MW module with no design
                # variable, silently corrupting the model.
                if (
                    self.capacity_expansion_ucblock
                    and attr_name in ("ThermalUnitBlock_parameters",
                                      "NuclearUnitBlock_parameters")
                    and is_extendable(components_df.loc[[component]],
                                      components.name, nominal_attrs)
                ):
                    raise ValueError(
                        f"{component} is an extendable thermal unit, which "
                        "UCBlock cannot expand. Set "
                        f"{nominal_attrs[components.name]}_extendable=False on "
                        "it or use the InvestmentBlock path "
                        "(capacity_expansion_ucblock=False) to expand thermal "
                        "units."
                    )

                self.add_UnitBlock(
                    attr_name,
                    components_df.loc[[component]],
                    components_t,
                    components.name,
                    n,
                    component,
                    unitblock_index,
                )

                if is_extendable(components_df.loc[[component]], components.name, nominal_attrs):
                    investment_meta["index_extendable"].append(unitblock_index)
                    investment_meta["Blocks"].append(f"{attr_name.split('_')[0]}_{unitblock_index}")
                    investment_meta["asset_type"].append(0)
                    self._store_unitblock_design_variable(attr_name, unitblock_index)

                unitblock_index += 1

        self.networkblock["Design"] = self.investmentblock.copy()
        self.networkblock["Design"]["DesignLines"] = {
            "value": np.array(investment_meta["design_lines"]),
            "type": "uint",
            "size": ("NumberDesignLines"),
        }

        self.ucblock_variables["generator_node"] = {
            "name": "GeneratorNode",
            "type": "int",
            "size": ("NumberElectricalGenerators",),
            "value": generator_node,
        }

        self._generator_owner = generator_owner
        self._storage_owner = storage_owner

        self.investmentblock["Blocks"] = investment_meta["Blocks"]
        self.investmentblock["Assets"] = {
            "value": np.array(investment_meta["index_extendable"]),
            "type": "uint",
            "size": "NumAssets",
        }
        self.investmentblock["AssetType"] = {
            "value": np.array(investment_meta["asset_type"]),
            "type": "int",
            "size": "NumAssets",
        }

    def _add_conversion_units(self, n, generator_node, generator_owner,
                              investment_meta, unitblock_index):
        """
        Adds a ConversionUnitBlock for each committable link (and, in the
        carrier form, for each link between carriers that a unit with one
        generator cannot express, see conversion_carrier_links()), with the
        variables of conversion_unit_variables(), and its generators to the
        nodes and owners of the generators of the UCBlock; an extendable
        one (in the InvestmentBlock form) is an asset of the InvestmentBlock,
        whose design scales the unit.

        Returns
        -------
        int
            The index of the next UnitBlock.
        """
        for name, (variables, dimensions, buses) in self._conversion_data.items():
            generator_node.extend(
                int(n.buses.index.get_loc(bus)) for bus in buses)
            generator_owner.extend(("links", name) for _ in buses)

            block_name = f"ConversionUnitBlock_{unitblock_index}"
            extendable = bool(n.links.at[name, "p_nom_extendable"])
            self.unitblocks[block_name] = {
                "name": name,
                "enumerate": f"UnitBlock_{unitblock_index}",
                "block": "ConversionUnitBlock",
                "DesignVariable": n.links.loc[[name], "p_nom"].values,
                "Extendable": extendable,
                # the MW of a unit of its design, if it has one
                "ModuleSize": float(n.links.at[name, "p_nom_mod"] or 1.0)
                if "p_nom_mod" in n.links.columns else 1.0,
                "variables": variables,
                "dimensions": dimensions,
            }
            # the design of the InvestmentBlock scales the unit [see
            # conversion_unit_variables()], that of the UCBlock form is in it
            self.unitblocks[block_name]["Scaled"] = (
                extendable and not self.capacity_expansion_ucblock)
            if self.unitblocks[block_name]["Scaled"]:
                self.add_InvestmentBlock(n, n.links.loc[[name]], "Link")
                investment_meta["index_extendable"].append(unitblock_index)
                investment_meta["Blocks"].append(block_name)
                investment_meta["asset_type"].append(0)
            unitblock_index += 1

        return unitblock_index

    def _add_carrier_link_units(self, n, generator_node, generator_owner,
                                investment_meta, unitblock_index):
        """
        Adds an IntermittentUnitBlock for each link between carriers [see
        carrier_unit_links()], whose power is p0 within the bounds of the
        link and with its marginal cost, and its generator to the nodes and
        owners of the generators of the UCBlock, on bus0; its outputs on the
        carriers of its ports are written by convert_to_ucblock(). An
        extendable one is an asset as an extendable generator is.

        Returns
        -------
        int
            The index of the next UnitBlock.
        """
        if self._carrier_links.empty:
            return unitblock_index

        links = n.links.loc[self._carrier_links]
        # a link has no bound on the energy of the horizon
        parameters = {
            key: value
            for key, value in self.config.IntermittentUnitBlock_parameters.items()
            if key not in ("MinGeneration", "MaxGeneration")
        }
        attr_name = "IntermittentUnitBlock_parameters"
        if not self.capacity_expansion_ucblock:
            self.add_InvestmentBlock(n, links, "Link")
        for name in links.index:
            generator_node.append(int(n.buses.index.get_loc(
                links.at[name, "bus0"])))
            generator_owner.append(("links", name))
            self.add_UnitBlock(attr_name, links.loc[[name]], n.links_t,
                               "Link", n, name, unitblock_index,
                               parameters=parameters)
            block_name = get_block_name(attr_name, unitblock_index,
                                        links.loc[[name]])
            self.unitblocks[block_name]["component"] = "Link"
            # its outputs, one per port
            self.unitblocks[block_name]["ports"] = link_ports(n, name)
            if is_extendable(links.loc[[name]], "Link", nominal_attrs):
                investment_meta["index_extendable"].append(unitblock_index)
                investment_meta["Blocks"].append(
                    f"IntermittentUnitBlock_{unitblock_index}")
                investment_meta["asset_type"].append(0)
                self._store_unitblock_design_variable(attr_name,
                                                      unitblock_index)
            unitblock_index += 1

        return unitblock_index

    ### 3b ###
    def add_pollutant_budget(self, n):
        """
        Add the pollutant budget constraints of the UCBlock, one per global
        constraint of the network on the dispatch (a primary energy or an
        operational limit), as translated by pollutant_budget_data().
        """
        self.pollutant_budget = pollutant_budget_data(
            n, self._generator_owner, self._storage_owner
        )
        if self.pollutant_budget is None:
            return

        number_generators = self.dimensions["UCBlock"]["NumberElectricalGenerators"]
        if self.pollutant_budget["rho"].shape[2] != number_generators:
            raise ValueError(
                "add_pollutant_budget: the conversion factors cover "
                f"{self.pollutant_budget['rho'].shape[2]} electrical generators, "
                f"the UCBlock has {number_generators}."
            )

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

        # the size of a module where the asset is a modular IntermittentUnitBlock,
        # 0 elsewhere [see apply_investment_module_overrides()]
        if getattr(self, "_investment_modules", False):
            df_alias["p_nom_mod"] = self._intermittent_module_size(
                components_df, components_type)

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
    
    def _intermittent_module_size(self, components_df, components_type):
        """The size of the module of each modular IntermittentUnitBlock or
        ConversionUnitBlock.

        For each row of components_df, the p_nom_mod of an extendable
        generator that becomes an IntermittentUnitBlock, or of an extendable
        committable link that becomes a ConversionUnitBlock, 0 for any other
        asset, whose design stays a capacity.
        """
        size = np.zeros(len(components_df))
        if "p_nom_mod" not in components_df.columns:
            return size

        # a committable link that is a ConversionUnitBlock of modules
        if components_type == "Link":
            for k, name in enumerate(components_df.index):
                if (name in self._conversion_links
                        and bool(components_df.at[name, "p_nom_extendable"])):
                    size[k] = max(float(components_df.at[name, "p_nom_mod"]),
                                  0.0)
            return size

        if components_type != "Generator":
            return size

        for k, (_, row) in enumerate(components_df.iterrows()):
            attr_name = get_attr_name(
                components_type,
                row.get("carrier"),
                enable_thermal_units=self.enable_thermal_units,
                intermittent_carriers=self.intermittent_carriers,
                default_intermittent=renewable_carriers,
                nuclear_carriers=list(self.nuclear_units),
            )
            if (attr_name == "IntermittentUnitBlock_parameters"
                    and bool(row.get("p_nom_extendable", False))):
                size[k] = max(float(row["p_nom_mod"]), 0.0)
        return size

    def _has_investment_modules(self, n):
        """True if n has a modular extendable IntermittentUnitBlock or
        ConversionUnitBlock."""
        return bool(np.any(
            self._intermittent_module_size(n.generators, "Generator") > 0)
            or np.any(self._intermittent_module_size(n.links, "Link") > 0))

    ### 5 ###
    def add_UnitBlock(self, attr_name, components_df, components_t, components_type, n, component=None, index=None, parameters=None):
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
        if parameters is not None:
            unitblock_parameters = parameters
        elif hasattr(self.config, attr_name):
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
            component,
            dense_cache=getattr(self, "_dense_cache", None)
        )

        dimensions = None
        if attr_name == "BatteryUnitBlock_parameters":
            # the Cost of a BatteryUnitBlock is a price on the net injection,
            # paid for the energy drawn and earned for the energy delivered,
            # while PyPSA charges the marginal cost of a StorageUnit on its
            # dispatch only: the two agree only for a unit that cannot store
            cost = np.asarray(converted_dict.get("Cost", {}).get("value", 0.0),
                              dtype=float)
            min_power = np.asarray(converted_dict["MinPower"]["value"],
                                   dtype=float)
            if np.any(cost != 0) and np.any(min_power < 0):
                raise ValueError(
                    f"{components_df.index[0]} has a marginal cost and can "
                    "store, while PyPSA charges that cost on the dispatch "
                    "only, which the price on the net injection of a "
                    "BatteryUnitBlock cannot express."
                )
        if attr_name in ("ThermalUnitBlock_parameters",
                         "NuclearUnitBlock_parameters"):
            # a ThermalUnitBlock has neither a bound on the energy of the
            # whole horizon (which no Dynamic Programming Solver of it could
            # enforce either) nor modules of its capacity, so a unit that has
            # them is refused instead of being written without them
            name = components_df.index[0]
            for field in ("e_sum_min", "e_sum_max"):
                if field in components_df and \
                        np.isfinite(components_df[field].iloc[0]):
                    raise ValueError(
                        f"{name} has {field}, a bound on the energy it "
                        "generates over the whole horizon, which a "
                        "ThermalUnitBlock cannot express."
                    )
            if "p_nom_mod" in components_df and \
                    float(components_df["p_nom_mod"].iloc[0]) != 0.0:
                raise ValueError(
                    f"{name} is a modular unit (p_nom_mod), whose commitment "
                    "PyPSA counts in modules, which a ThermalUnitBlock cannot "
                    "express."
                )

            # a unit that pays nothing to shut down has no such variable,
            # as it has none in the block
            if "ShutDownCost" in converted_dict and \
                    not np.any(converted_dict["ShutDownCost"]["value"]):
                del converted_dict["ShutDownCost"]

            forbid_unreachable_switches(converted_dict, len(n.snapshots))
            # PyPSA ramps from the output before the horizon only when the
            # network gives it, while a ThermalUnitBlock always does
            if "p_init" not in components_df or \
                    components_df["p_init"].isna().all():
                free_initial_ramp(converted_dict, len(n.snapshots))
            # a generator that PyPSA does not commit is on at every snapshot
            if not bool(components_df["committable"].iloc[0]):
                fix_commitment_on(converted_dict, len(n.snapshots))
        if attr_name == "NuclearUnitBlock_parameters":
            carrier = str(components_df["carrier"].iloc[0]).strip().lower()
            rule_variables, dimensions = nuclear_rule_variables(
                self.nuclear_units[carrier],
                converted_dict,
                self._uniform_snapshot_hours(n),
            )
            converted_dict.update(rule_variables)

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
            if dimensions:
                self.unitblocks[name]["dimensions"] = dimensions
        
        if attr_name == 'HydroUnitBlock_parameters':
            dimensions = self.dimensions['HydroUnitBlock']
            # the loop above counts one generator for the storage unit, while
            # the block has one per arc: a turbine, a pump and a spillway
            self.dimensions['UCBlock']["NumberElectricalGenerators"] += (
                dimensions["NumberArcs"] - 1)
            
            self.unitblocks[name]['dimensions'] = dimensions
        
    @staticmethod
    def _uniform_snapshot_hours(n):
        """
        Returns the duration of a snapshot in hours, which the durations of the
        nuclear rules are divided by; the weighting must be uniform.
        """
        weights = n.snapshot_weightings["generators"].to_numpy(dtype=float)
        if weights.size == 0 or not np.allclose(weights, weights[0]):
            raise ValueError(
                "The nuclear rules need a uniform snapshot weighting, "
                "their durations being given in hours."
            )
        return float(weights[0])

    def add_demand(self, n):
        """
        Build nodal active power demand for SMS++.

        This includes both dynamic and static loads.
        """
        demand = get_bus_demand_matrix(n)

        self.demand = {
            "name": "ActivePowerDemand",
            "type": "float",
            "size": ("NumberNodes", "TimeHorizon"),
            "value": demand,
        }

    ### 7 ###
    def lines_links(self, n):
        """
        Merge or rename network blocks to ensure a single 'Lines' block for SMS++.
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

        apply_time_dependent_link_data_to_lines(
            n=n,
            networkblock=self.networkblock,
            links=n.links.index.drop(self._conversion_links),
        )




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

            if self.problem_structure.get("is_stochastic", False):
                if self.problem_structure.get("stochastic_type") == "sddp":
                    return self._parse_sddp_solution_to_unitblocks(solution, n, solution_data)
                if self.problem_structure.get("stochastic_type") == "mssb":
                    return self._parse_multistage_solution_to_unitblocks(
                        solution, n, solution_data
                    )
                return self._parse_stochastic_solution_to_unitblocks(solution, n, solution_data)

        return self._parse_deterministic_solution_to_unitblocks(solution, n, solution_data)


    def _parse_deterministic_solution_to_unitblocks(self, solution, n, solution_data):
        """
        Parse the legacy deterministic solution structure.
        """
        num_units = self.dimensions["UCBlock"]["NumberUnits"]

        solution_0 = solution.blocks["Solution_0"]
        has_investment = "DesignVariables" in solution_0.variables

        if not self.capacity_expansion_ucblock:
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

        # the design of a ConversionUnitBlock of the UCBlock form, in modules
        # or in MW
        for unit_block in self.unitblocks.values():
            if "ConversionDesign" in unit_block:
                unit_block["DesignVariable"] = (
                    float(np.asarray(unit_block["ConversionDesign"]))
                    * unit_block["ModuleSize"])
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
        or, with the investment outside, one level deeper:
            Solution_0                  (the InvestmentBlock, DesignVariables)
                InnerSolution
                    ScenarioSolution_0
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
        solution_0, design_vars = self._split_investment_outside(
            solution.blocks["Solution_0"])
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

        self._assign_investment_outside_design(design_vars)
        split_merged_dcnetworkblocks(self.unitblocks)
        return solution_data
    
    
    def _split_investment_outside(self, solution_0):
        """
        The solution of the stochastic Block, and the design of the
        InvestmentBlock above it if the investment is outside.

        With "investment_outside" Solution_0 is the solution of the
        InvestmentBlock, whose DesignVariables are the investment, and the one
        of the stochastic Block is its InnerSolution; otherwise Solution_0 is
        that of the stochastic Block, and the design (None) is read in the
        scenarios.
        """
        if not self.problem_structure.get("investment_outside", False):
            return solution_0, None

        if "InnerSolution" not in solution_0.blocks:
            raise KeyError(
                "InnerSolution not found in Solution_0: with "
                "'investment_outside' Solution_0 is the solution of the "
                "InvestmentBlock and that of the stochastic Block is inside it."
            )
        return (solution_0.blocks["InnerSolution"],
                solution_0.variables["DesignVariables"].data)

    def _assign_investment_outside_design(self, design_vars):
        """
        Give every scenario of an asset the design of the InvestmentBlock.

        The investment outside is one for all the scenarios, and a scenario
        is read back as the asset with its own values on top [see
        block_to_dataarrays_stochastic()]: the design goes both on the asset
        and in each of its scenarios, where it replaces whatever the
        scenario carries (e.g., the nominal capacity of a line).
        """
        if design_vars is None:
            return

        block_names = self.investmentblock.get("Blocks", [])
        assign_design_variables_to_unitblocks(self.unitblocks, block_names,
                                              design_vars)
        for name, value in zip(block_names, design_vars):
            for scenario in self.unitblocks[name].get("scenarios", {}).values():
                scenario["DesignVariable"] = value

    def _parse_multistage_solution_to_unitblocks(self, solution, n, solution_data):
        """
        Parse the solution of a MultiStageStochasticBlock.

        Expected layout, one level deeper than the two-stage one, since each
        outer scenario is itself a TwoStageStochasticBlock:
            Solution_0
                ScenarioSolution_0              (outer scenario 0)
                    ScenarioSolution_0          (its inner scenarios)
                    ScenarioSolution_1
                    ...
                ScenarioSolution_1
                ...

        The leaves are stored under the very scenario names of the flat
        network, so that everything downstream reads them as usual.
        """
        num_units = self.dimensions["UCBlock"]["NumberUnits"]
        groups = self.problem_structure["scenario_tree"]["groups"]

        solution_0, design_vars = self._split_investment_outside(
            solution.blocks["Solution_0"])
        solution_data["MSSB"] = solution_0

        self.networkblock.setdefault("Scenarios", {})

        for outer, group in enumerate(groups):
            outer_key = f"ScenarioSolution_{outer}"

            if outer_key not in solution_0.blocks:
                raise KeyError(
                    f"{outer_key} not found in Solution_0. Expected one block "
                    "per outer scenario of the scenario tree."
                )

            outer_block = solution_0.blocks[outer_key]

            for inner, (scenario_name, _) in enumerate(group["scenarios"]):
                inner_key = f"ScenarioSolution_{inner}"

                if inner_key not in outer_block.blocks:
                    raise KeyError(
                        f"{inner_key} not found in {outer_key}. Expected one "
                        f"block per inner scenario of {group['name']!r}."
                    )

                scenario_block = outer_block.blocks[inner_key]
                solution_data[f"{outer_key}/{inner_key}"] = scenario_block

                if self.dimensions["UCBlock"]["NumberLines"] > 0:
                    self.parse_networkblock_lines(
                        scenario_block, scenario_name=scenario_name
                    )
                    self.generate_line_unitblocks(n, scenario_name=scenario_name)

                self._parse_unitblocks_from_solution_block(
                    solution_block=scenario_block,
                    num_units=num_units,
                    scenario_name=scenario_name,
                    solution_data=solution_data,
                )

        self._assign_investment_outside_design(design_vars)
        split_merged_dcnetworkblocks(self.unitblocks)
        return solution_data

    def _parse_sddp_solution_to_unitblocks(self, solution, n, solution_data):
        """
        SDDP: parsing della soluzione non implementato.

        Per ora questo metodo ritorna solution_data vuoto: l'objective
        resta in self.result.objective_value, i dettagli nei file del solver.
        """
        import warnings
        warnings.warn(
            "Il parsing della soluzione SDDP non è ancora implementato",
            stacklevel=2,
        )
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
        if self.hydro_system:
            solution_block = hydro_system_unit_solutions(
                solution_block, self.unitblocks, num_units)
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
        if self.multi_carrier:
            self._read_carrier_network_lines(solution_block, target)
        else:
            self._read_network_lines(solution_block.blocks, target)

    def _read_carrier_network_lines(self, solution_block, target):
        """
        Reads the flows of the lines of all the carriers, those of the
        electricity from the solution of the UCBlock and those of a carrier
        c from its group "Carrier_c", into the arrays over all the lines of
        the network, in the order of LineName [see _line_layout()]; a carrier
        whose flows are not in the solution has them at 0. The injections
        are those of the electricity.
        """
        line_carrier, index = self._line_layout()
        T = int(self.dimensions["UCBlock"]["TimeHorizon"])
        flows = np.zeros((T, line_carrier.size))
        design = np.asarray(self.networkblock.get("Design", {}).get(
            "DesignLines", {}).get("value", []), dtype=int)
        design_values = {}
        for c in range(len(self._carriers)):
            selected = np.flatnonzero(line_carrier == c)
            if selected.size == 0:
                continue
            blocks = solution_block.blocks if c == 0 else getattr(
                solution_block.blocks.get(f"Carrier_{c}"), "blocks", {})
            local = {}
            if c == 0 or blocks:
                self._read_network_lines(blocks, local)
            if c == 0:
                target.update(local)
            if "FlowValue" in local:
                flows[:, selected] = np.atleast_2d(
                    local["FlowValue"])[:, index[selected]]
            if "DesignValue" in local:
                # the design lines of the carrier, in the order of the
                # network as those of all of them
                values = np.atleast_1d(local["DesignValue"])
                for k, j in enumerate(np.sort(design[line_carrier[design]
                                                     == c])):
                    design_values[j] = values[..., k]
        target["FlowValue"] = flows
        if design_values:
            target["DesignValue"] = np.stack(
                [design_values[j] for j in sorted(design_values)], axis=-1)

    def _read_network_lines(self, blocks, target):
        """
        Reads FlowValue and NodeInjection (and DesignValue) of the
        NetworkBlocks among `blocks` into `target`
        [see parse_networkblock_lines()].
        """
        vars_of_interest = ("FlowValue", "NodeInjection")

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
        # iterate_blocks() merges the blocks into a Dataset already, and
        # xarray refuses to build one out of a Dataset
        self.ds = self.iterate_blocks(n)
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
            dataarrays = block_to_dataarrays(n, name, unit_block, component, self.config, scenario_name=None)
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
                datasets.append(dataarrays)

        if not datasets:
            return {}

        # the same merge as one xr.merge() of a Dataset per block, done
        # variable by variable [see merge_by_variable()]
        ds = merge_by_variable(datasets)
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

        # --------------------------------------------------
        # SDDP branch
        # --------------------------------------------------
        # SDDP ha una struttura top-level completamente diversa da TSSB:
        #   SDDPBlock
        #   ├── AbstractPath
        #   ├── StochasticBlock_0 -> BendersBlock -> BendersBFunction -> Block(filename)
        #   ├── StochasticBlock_1 -> ...
        #   └── ...
        #
        # NON annidiamo un UCBlock al top-level: ogni stadio ha il proprio
        # UCBlock salvato in un file .nc4 separato, referenziato via filename.
        #
        # Per questo, dopo aver costruito l'SDDPBlock, usciamo immediatamente
        # dalla funzione: NON dobbiamo eseguire la logica deterministica
        # (InvestmentBlock / UCBlock) che segue.

            if stochastic_type == "sddp":
                self.convert_to_sddp_block(master, index_id=0, name_id="Block_0")
                self.sms_network = sn
                return sn

            if stochastic_type not in ("tssb", "mssb"):
                raise ValueError(
                    f"Unsupported stochastic_type in convert_to_blocks: {stochastic_type!r}"
                )

            if self.problem_structure.get("investment_outside", False):
                self.convert_to_investmentblock(master, 0, "Block_0")
                master = sn.blocks["Block_0"]
                index_id = 0

                if stochastic_type == "mssb":
                    self.convert_to_mssb(master, name_id="InnerBlock")
                else:
                    self.convert_to_tssb(master, index_id=0,
                                         name_id="InnerBlock")
                    self.add_deterministic_model(
                        master.blocks["InnerBlock"].blocks["StochasticBlock"],
                        0, inside_stochastic=True, with_investment=False)

                self.sms_network = sn
                return sn

            if stochastic_type == "mssb":
                self.convert_to_mssb(master, name_id="Block_0")
                self.sms_network = sn
                return sn

            self.convert_to_tssb(master, index_id=0, name_id="Block_0")
            master = sn.blocks["Block_0"].blocks["StochasticBlock"]
            index_id = 0
            inside_tssb = True
            self.convert_to_tssb(master, index_id=0, name_id="Block_0")

            # Move master to the inner block container of StochasticBlock
            master = sn.blocks["Block_0"].blocks["StochasticBlock"]
            index_id = 0
            inside_tssb = True

        # --------------------------------------------------
        # Deterministic investment / UC nesting
        # --------------------------------------------------
        self.add_deterministic_model(master, index_id, inside_stochastic=inside_tssb)
    
        self.sms_network = sn
        return sn
    
    def add_deterministic_model(self, master, index_id=0,
                                inside_stochastic=False, with_investment=True):
        """
        Add the deterministic model, i.e. the optional InvestmentBlock and the
        UCBlock inside it, to master. This is what every scenario sees, and
        it is therefore built once per inner Block of a multi-stage problem.

        With with_investment false the InvestmentBlock is left out: this is
        what the scenarios of the "Benders form" look like, the investment
        having been taken out of them and stated once, above.
        """
        if with_investment and self.problem_structure.get("has_investment_block",
                                                          False):
            name_id = "InvestmentBlock"
            self.convert_to_investmentblock(master, index_id, name_id)

            master = master.blocks[name_id]
            index_id += 1
            name_id = "InnerBlock"
        else:
            name_id = "Block" if inside_stochastic else "Block_0"
    
        self.convert_to_ucblock(master, index_id, name_id)
        return master

    def convert_to_tssb(self, master, index_id, name_id):
        """
        Add a TwoStageStochasticBlock to the SMSNetwork hierarchy.

        Structure:
        TwoStageStochasticBlock
        ├── DiscreteScenarioSet
        ├── StaticAbstractPath      (only if it has first-stage variables)
        └── StochasticBlock

        With the investment stated above, in an InvestmentBlock wrapping this
        Block, the scenarios hold no here-and-now variable at all: there is
        then nothing for the StaticAbstractPath to address and nothing for
        the non-anticipativity Constraint to tie, so the path is left out.
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
        if not self.problem_structure.get("investment_outside", False):
            self.convert_to_static_abstract_path(tssb_block,
                                                 "StaticAbstractPath")
        self.convert_to_stochastic_block(tssb_block, "StochasticBlock")

        return master
    
    def convert_to_mssb(self, master, name_id="Block_0"):
        """
        Add a MultiStageStochasticBlock to the SMSNetwork hierarchy.

        Structure:
        MultiStageStochasticBlock
        ├── ScenarioGenerator          (the whole scenario tree)
        ├── StaticAbstractPath         (the first-stage design variables)
        ├── Block_0                    (TwoStageStochasticBlock, outer scenario 0)
        │   ├── StaticAbstractPath
        │   └── StochasticBlock        (data mappings + the model template)
        └── Block_1 ...

        Unlike the two-stage case the inner blocks carry no DiscreteScenarioSet
        of their own: each of them reads its own scenarios from the shared tree,
        which is what ties the inner realizations to the outer branch they hang
        from.
        """
        groups = self.problem_structure["scenario_tree"]["groups"]

        # pySMSpp knows nothing of this block type, so it is built as a plain
        # Block carrying the type: everything below it is standard again
        mssb_block = Block(
            block_type="MultiStageStochasticBlock",
            NumberSubBlocks=Dimension("NumberSubBlocks", len(groups)),
        )
        master.add_block(name_id, block=mssb_block)
        mssb_block = master.blocks[name_id]

        self.convert_to_scenario_tree(mssb_block, "ScenarioGenerator")
        if not self.problem_structure.get("investment_outside", False):
            self.convert_to_static_abstract_path(
                mssb_block , "StaticAbstractPath" ,
                root_only=bool( self.tssb_data.get(
                                    "static_abstract_path_root" ) ) )

        for index, group in enumerate(groups):
            inner_id = f"Block_{index}"

            mssb_block.add(
                "TwoStageStochasticBlock",
                inner_id,
                id=f"{index}",
                NumberScenarios=Dimension(
                    "NumberScenarios", len(group["scenarios"])
                ),
            )

            inner_block = mssb_block.blocks[inner_id]

            if not self.problem_structure.get("investment_outside", False):
                self.convert_to_static_abstract_path(inner_block,
                                                     "StaticAbstractPath")
            self.convert_to_stochastic_block(inner_block, "StochasticBlock")
            # with the investment stated above, in an InvestmentBlock
            # wrapping this one, the scenarios hold the deterministic model
            # alone [see add_deterministic_model()]
            self.add_deterministic_model(
                inner_block.blocks["StochasticBlock"], 0,
                inside_stochastic=True,
                with_investment=not self.problem_structure.get(
                    "investment_outside", False),
            )

        return master


    def convert_to_scenario_tree(self, master, name_id="ScenarioGenerator"):
        """
        Add the scenario tree, a MultiStageDiscreteScenarioSet, to a MSSB block.

        The tree has one node per realization: the root, one node per outer
        scenario carrying its probability, and one leaf per inner scenario
        carrying its probability conditional on the outer one it hangs from,
        together with the very data the flat two-stage form would have put in
        the DiscreteScenarioSet. Only the leaves carry data, which is what
        StageScenarioSize says.
        """
        dss_data = self.tssb_data["discrete_scenario_set"]
        dims = self.dimensions["tssb"]["dss"]
        groups = self.problem_structure["scenario_tree"]["groups"]

        scenarios = np.asarray(dss_data["scenarios"], dtype=float)
        scenario_size = int(dims["ScenarioSize"])
        row_of = {
            name: index
            for index, name in enumerate(self.problem_structure["scenario_names"])
        }

        stages, parents, probabilities, data = [], [], [], []

        stages.append(0)                            # the root
        parents.append(-1)
        probabilities.append(1.0)
        data.append(np.zeros(scenario_size))

        for group in groups:
            outer = len(stages)
            stages.append(1)
            parents.append(0)
            probabilities.append(group["probability"])
            data.append(np.zeros(scenario_size))

            for name, probability in group["scenarios"]:
                stages.append(2)
                parents.append(outer)
                probabilities.append(probability)
                data.append(scenarios[row_of[name]])

        number_nodes = len(stages)
        # the root has no parent: any index past the last node says so
        parents = [number_nodes if p < 0 else p for p in parents]

        tree_block = Block(
            block_type="MultiStageDiscreteScenarioSet",
            NumberStages=Dimension("NumberStages", 3),
            NumberNodes=Dimension("NumberNodes", number_nodes),
            ScenarioDataSize=Dimension("ScenarioDataSize", scenario_size),
            StageScenarioSize=Variable(
                "StageScenarioSize",
                "u4",
                ("NumberStages",),
                np.array([0, 0, scenario_size], dtype="u4"),
            ),
            NodeStage=Variable(
                "NodeStage", "u4", ("NumberNodes",),
                np.array(stages, dtype="u4"),
            ),
            NodeParent=Variable(
                "NodeParent", "u4", ("NumberNodes",),
                np.array(parents, dtype="u4"),
            ),
            NodeProbability=Variable(
                "NodeProbability", "double", ("NumberNodes",),
                np.array(probabilities, dtype=float),
            ),
            NodeData=Variable(
                "NodeData", "double", ("NumberNodes", "ScenarioDataSize"),
                np.vstack(data),
            ),
        )

        master.add_block(name_id, block=tree_block)
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

    def convert_to_static_abstract_path(self, master,
                                        name_id="StaticAbstractPath",
                                        root_only=False):
        """
        Add the StaticAbstractPath block to a TSSB block.

        With root_only the path names the design decisions taken at the
        root alone, which is what the outer level of a tree with a second
        decision stage ties; everywhere else the path names them all.
        """
        which = "static_abstract_path_root" if root_only \
                else "static_abstract_path"
        sap_data = self.tssb_data[ which ]
        dims = self.dimensions["tssb"][ "sap_root" if root_only else "sap" ]

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
        )

        # Rendiamo queste variabili opzionali per includere anche il caso SDDP
        if "PathElementIndices" in ap_data:
            ap_block.add_variable(
                "PathElementIndices",
                "u4",
                ("TotalLength",),
                ap_data["PathElementIndices"],
            )

        if "PathRangeIndices" in ap_data:
            ap_block.add_variable(
                "PathRangeIndices",
                "u4",
                ("TotalLength",),
                ap_data["PathRangeIndices"],
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
                value = variable['value']
                # with the carriers, the lines of an asset are numbered one
                # carrier after the other
                if self.multi_carrier and name == "Assets":
                    value = np.array(value).copy()
                    lines = np.asarray(
                        self.investmentblock["AssetType"]["value"]) == 1
                    value[lines] = self._carrier_line_indices(value[lines])
                kwargs[name] = Variable(
                    name,
                    variable['type'],
                    variable['size'],
                    value
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

    def _hydro_first_order(self, owners):
        """
        Permutazione che porta gli elementi hydro in testa, mantenendo
        l'ordine relativo degli altri.
        """
        hydro_names = {
            str(ub["name"]) for ub in self.unitblocks.values()
            if ub["block"] == "HydroUnitBlock"
        }
        is_hydro = [
            comp == "storage_units" and str(name) in hydro_names
            for comp, name in owners
        ]
        return np.array(
            [i for i, h in enumerate(is_hydro) if h]
            + [i for i, h in enumerate(is_hydro) if not h],
            dtype=int,
        )

    def convert_to_ucblock(self, master, index_id, name_id, wrap_hydro_in_system: bool = False):
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
        wrap_hydro_in_system : bool, default False
            Se True e esistono HydroUnitBlock in self.unitblocks, questi vengono raggruppati in un unico HydroSystemUnitBlock (primo figlio dell'UCBlock, nome UnitBlock_0), che contiene un HydroUnitBlock_i per ogni hydro e un PolyhedralFunctionBlock vuoto con PolyFunction_NumVar = n_reservoirs_total. Questo è il formato richiesto da SDDP, perché il PolyhedralFunctionBlock rappresenta il costo futuro dell'acqua e il BendersBFunction lo raggiunge via AbstractPath.
            Se False, gli hydro sono UnitBlock_i normali dentro l'UCBlock (comportamento della pipeline deterministica e TSSB).
            Se True ma non ci sono HydroUnitBlock, il flag non ha effetto.

        Returns
        -------
        SMSNetwork
            The SMSNetwork with the UCBlock added.
        """

        # UCBlock dimensions (NumberUnits, NumberNodes, etc.)
        ucblock_dims = self.dimensions["UCBlock"]

        # -----------------
        # Demand (load)
        # -----------------
        demand_var = {
            self.demand["name"]: Variable(
                self.demand["name"],
                self.demand["type"],
                self.demand["size"],
                self.demand["value"],
            )
        }

        # -----------------
        # UCBlock variables
        # -----------------
        ucblock_vars = {}
        for _, var in self.ucblock_variables.items():
            ucblock_vars[var["name"]] = Variable(
                var["name"],
                var["type"],
                var["size"],
                var["value"],
            )

        # Permuta GeneratorNode per allinearsi all'ordine idro-first
        # di wrap_hydro_in_system=True. Senza wrapper, gen_order resta None
        # e il comportamento è invariato.
        gen_order = None
        if wrap_hydro_in_system and any(
                ub["block"] == "HydroUnitBlock" for ub in self.unitblocks.values()
        ):
            gen_order = self._hydro_first_order(self._generator_owner)
            gn = np.asarray(self.ucblock_variables["generator_node"]["value"])
            assert len(gn) == len(gen_order), (
                "GeneratorNode e generator_owner incoerenti: "
                f"{len(gn)} vs {len(gen_order)}"
            )
            gv = self.ucblock_variables["generator_node"]
            ucblock_vars[gv["name"]] = Variable(
                gv["name"], gv["type"], gv["size"], gn[gen_order],
            )

        # -----------------
        # Network lines (Lines block only, merged with Links if needed)
        # -----------------
        line_vars = {}
        if ucblock_dims.get("NumberLines", 0) > 0:
            for var_name, var in self.networkblock["Lines"]["variables"].items():
                line_vars[var_name] = Variable(
                    var_name,
                    var["type"],
                    var["size"],
                    var["value"],
                )

        # --------------------
        # INTEGRAZIONE WRAPPER
        # --------------------

        # Calcoliamo n_hydro e n_reservoirs_total
        hydro_unitblocks = [(name, ub) for name, ub in self.unitblocks.items() if ub["block"] == "HydroUnitBlock"]
        # n_hydro server per calcolare NumberUnits
        n_hydro = len(hydro_unitblocks)
        # n_reservoirs_total serve per calcolare PolyFunction_NumVar
        n_reservoirs_total = sum(int(ub["dimensions"]["NumberReservoirs"]) for _, ub in hydro_unitblocks)

        if wrap_hydro_in_system and n_hydro >= 1:
            ucblock_dims = dict(ucblock_dims)
            ucblock_dims["NumberUnits"] = ucblock_dims["NumberUnits"] - (n_hydro - 1)

        # -----------------
        # Assemble all kwargs
        # -----------------
        block_kwargs = {
            **ucblock_dims,
            **demand_var,
            **ucblock_vars,
            **line_vars,
        }

        # the nodes, the lines and the demand of the carriers other than the
        # electricity are in their groups
        carrier_blocks = {}
        if self.multi_carrier:
            block_kwargs, carrier_blocks = self._split_carriers(block_kwargs)

        # -----------------
        # Add UCBlock itself
        # -----------------
        master.add(
            "UCBlock",
            name_id,
            id=f"{index_id}",
            **block_kwargs,
        )

        if self.multi_carrier:
            ucblock = master.blocks[name_id]
            ucblock.add_dimension("NumberCarriers", len(self._carriers))
            ucblock.add_variable("CarrierName", "str", ("NumberCarriers",),
                                 np.array(self._carriers, dtype=object))
            # the nodes of the generators carrier by carrier, -1 where a
            # generator has none: those of the electricity in the UCBlock,
            # those of each other carrier in its group, omitted for a
            # carrier of one node, where every generator is
            nodes = self._generator_carrier_nodes()
            ucblock.add_variable(
                "GeneratorNode", "int", ("NumberElectricalGenerators",),
                nodes[:, 0])
            for c, carrier_block in carrier_blocks.items():
                if int(carrier_block.dimensions["NumberNodes"].value) > 1:
                    carrier_block.add_variable(
                        "GeneratorNode", "int",
                        ("NumberElectricalGenerators",), nodes[:, c])
                ucblock.add_block(f"Carrier_{c}", block=carrier_block)

        # -----------------
        # Pollutant budget, added to the Block directly
        # -----------------
        pb = getattr(self, "pollutant_budget", None)
        if pb:
            ucblock = master.blocks[name_id]
            number_pollutants = len(pb["names"])
            ucblock.add_dimension("NumberPollutants", number_pollutants)
            ucblock.add_dimension("TotalNumberPollutantZones", number_pollutants)
            ucblock.add_variable(
                "PollutantBudget", "float", ("TotalNumberPollutantZones",),
                pb["budget"],
            )
            if np.isfinite(pb["min_budget"]).any():
                ucblock.add_variable(
                    "PollutantMinBudget", "float", ("TotalNumberPollutantZones",),
                    pb["min_budget"],
                )

            # --- permutazione idro-first (nuovo) ---
            rho = pb["rho"]
            storage_rho = pb["storage_rho"]
            if gen_order is not None:
                rho = rho[:, :, gen_order]
                storage_rho = storage_rho[
                    :, :, self._hydro_first_order(self._storage_owner)
                ]
            # --- fine permutazione ---

            ucblock.add_variable(
                "PollutantRho", "float",
                ("TimeHorizon", "NumberPollutants", "NumberElectricalGenerators"),
                rho,  # <-- era pb["rho"]
            )
            if np.any(storage_rho):  # <-- era pb["storage_rho"]
                ucblock.add_dimension("NumberStorages", storage_rho.shape[2])
                ucblock.add_variable(
                    "PollutantStorageRho", "float",
                    ("TimeHorizon", "NumberPollutants", "NumberStorages"),
                    storage_rho,  # <-- era pb["storage_rho"]
                )

        # --------------------
        # INTEGRAZIONE WRAPPER
        # --------------------

        # Costruzione vera e propria del wrapper
        if wrap_hydro_in_system and n_hydro >= 1:

            # Contenitore: HydroSystemUnitBlock con NumberHydroUnits = n_hydro
            hydro_sys = Block().from_kwargs(
                block_type="HydroSystemUnitBlock",
                NumberHydroUnits=Dimension("NumberHydroUnits", n_hydro),
            )

            # Figli: un HydroUnitBlock_i per ogni hydro.
            # I nomi sono HydroUnitBlock_0, HydroUnitBlock_1, ...
            # nell'ordine in cui compaiono in hydro_unitblocks.
            for local_idx, (_, ub) in enumerate(hydro_unitblocks):
                ub_kwargs = {}

                # Variabili
                for var_name, var in ub["variables"].items():
                    ub_kwargs[var_name] = Variable(
                        var_name,
                        var["type"],
                        var["size"],
                        var["value"],
                    )

                # Dimensioni (per gli HydroUnitBlock sono sempre scalari)
                if "dimensions" in ub:
                    for dim_name, dim_value in ub["dimensions"].items():
                        ub_kwargs[dim_name] = Dimension(dim_name, dim_value)

                hydro_ub = Block().from_kwargs(
                    block_type="HydroUnitBlock",
                    name=ub["name"],
                    **ub_kwargs,
                )

                hydro_sys.add_block(
                    f"HydroUnitBlock_{local_idx}",
                    block=hydro_ub,
                )

            # PolyhedralFunctionBlock vuoto: solo NumVar = n_reservoirs_total.
            # Gli altri campi (A, b, Vert, sign, lb, NumRow) sono opzionali.
            # Il solver SDDP ci aggiunge i cut a runtime.
            pfb = Block().from_kwargs(
                block_type="PolyhedralFunctionBlock",
                PolyFunction_NumVar=Dimension(
                    "PolyFunction_NumVar", n_reservoirs_total,
                ),
            )
            hydro_sys.add_block("PolyhedralFunctionBlock", block=pfb)

            # Aggiungiamo il wrapper all'UCBlock come UnitBlock_0
            # (primo figlio, così hydro_system_index = 0 come nel riferimento).
            master.blocks[name_id].add_block("UnitBlock_0", block=hydro_sys)

        next_slot = 1 if (wrap_hydro_in_system and n_hydro >= 1) else 0

        # -----------------
        # Add all UnitBlocks inside UCBlock
        # -----------------
        for ub_name, unit_block in self.unitblocks.items():
            # Non vogliamo produrre duplicati
            if wrap_hydro_in_system and unit_block["block"] == "HydroUnitBlock":
                continue

            ub_kwargs = {}
            for var_name, var in unit_block["variables"].items():
                ub_kwargs[var_name] = Variable(
                    var_name,
                    var["type"],
                    var["size"],
                    var["value"],
                )

            # Add also any special dimensions
            if "dimensions" in unit_block:
                for dim_name, dim_value in unit_block["dimensions"].items():
                    # without a table pySMSpp would take an int for an Attribute
                    ub_kwargs[dim_name] = (
                        Dimension(dim_name, dim_value)
                        if unit_block["block"] in ("NuclearUnitBlock",
                                                   "ConversionUnitBlock")
                        else dim_value
                    )
    
            # pySMSpp has no ShutDownCost in its table of the
            # ThermalUnitBlock: it is added to the Block afterwards, as the
            # pollutant budget of the UCBlock is
            shut_down_cost = ub_kwargs.pop("ShutDownCost", None)

            # Create Block (pySMSpp has no table for NuclearUnitBlock and
            # ConversionUnitBlock and infers the kinds from the values, which
            # is what we want)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Block type (NuclearUnitBlock|ConversionUnitBlock) not found"
                )
                unit_block_obj = Block().from_kwargs(
                    block_type=unit_block["block"],
                    name=unit_block["name"],
                    **ub_kwargs,
                )
    
            if shut_down_cost is not None:
                unit_block_obj.add_variable(
                    "ShutDownCost", shut_down_cost.var_type,
                    shut_down_cost.dimensions, shut_down_cost.data,
                )


            if wrap_hydro_in_system:
                # Il path SDDP: idro wrappato in UnitBlock_0, gli altri
                # UnitBlock_i con nome sequenziale
                master.blocks[name_id].add_block(
                    f"UnitBlock_{next_slot}",
                    block=unit_block_obj,
                )
            else:
                # Il path upstream (TSSB/deterministico):
                # add_unit_block gestisce anche hydro_system=True
                if self.multi_carrier:
                    self._add_unit_outputs(unit_block_obj, ub_name, unit_block)
                add_unit_block(
                    master.blocks[name_id], unit_block, unit_block_obj,
                    self.hydro_system,
                    self.config.HydroSystemUnitBlock_parameters,
                )
            next_slot += 1

        # -----------------
        # Optionally add DesignNetworkBlock (only in capacity_expansion_ucblock mode)
        # -----------------
        self.convert_to_designnetworkblock(master, name_id)

        # -----------------
        # Done
        # -----------------
        return master

    def _demand_bus_order(self, n):
        """
        The buses in the order of the nodes of the demand: None for that of
        n.buses, or with the carriers that of UCBlock::set_energy_demand, the
        nodes of the electricity first and then those of each other carrier.
        """
        if not self.multi_carrier:
            return None
        # the buses of a stochastic network are indexed by scenario too
        buses = dict.fromkeys(n.buses.index.get_level_values(-1))
        return sorted(buses, key=lambda bus: self._bus_node[bus])

    def _line_layout(self):
        """
        The carrier of each line of the network (in the order of LineName)
        and its index among the lines of its carrier; the links between two
        carriers being units, each line joins two nodes of one carrier. A
        hyperarc must be one of the electricity.

        Returns
        -------
        tuple of numpy.ndarray
            The carriers and the indices.
        """
        if self._line_layout_cache is not None:
            return self._line_layout_cache
        number_lines = int(self.dimensions["UCBlock"].get("NumberLines", 0))
        if number_lines == 0:
            self._line_layout_cache = (np.zeros(0, dtype=int),
                                       np.zeros(0, dtype=int))
            return self._line_layout_cache
        variables = self.networkblock["Lines"]["variables"]
        start = np.ravel(np.asarray(variables["StartLine"]["value"],
                                    dtype=int))
        end = np.ravel(np.asarray(variables["EndLine"]["value"], dtype=int))
        carrier = np.array([c for c, _ in self._bus_node.values])
        if np.any(carrier[start] != carrier[end]):
            raise ValueError(
                "multi_carrier: a line joins two carriers, while it should "
                "be a unit.")
        if start.size != number_lines:
            # hyperarcs, whose branches all go to the electricity
            if np.any(carrier[start] != 0):
                raise ValueError(
                    "multi_carrier: a hyperarc of a carrier other than the "
                    "electricity is not translated.")
            line_carrier = np.zeros(number_lines, dtype=int)
        else:
            line_carrier = carrier[start]
        index = np.zeros(number_lines, dtype=int)
        for c in np.unique(line_carrier):
            index[line_carrier == c] = np.arange(
                int((line_carrier == c).sum()))
        self._line_layout_cache = (line_carrier, index)
        return self._line_layout_cache

    def _carrier_line_indices(self, lines):
        """
        The indices of the given lines of the network in the UCBlock, where
        the lines of the carriers are numbered one carrier after the other,
        those of the electricity first (as the assets of an InvestmentFunction
        are).
        """
        line_carrier, index = self._line_layout()
        lines = np.asarray(lines, dtype=int)
        first = np.concatenate(([0], np.cumsum(np.bincount(
            line_carrier, minlength=len(self._carriers)))))
        return (first[line_carrier[lines]] + index[lines]).astype(
            lines.dtype if lines.size else int)

    def _generator_ports(self):
        """
        The buses of each generator of the UCBlock: that of its node, or
        the ports of the link if it is the generator of a link between
        carriers.
        """
        if getattr(self, "_generator_ports_cache", None) is not None:
            return self._generator_ports_cache
        buses = self._bus_node.index
        nodes = np.asarray(self.ucblock_variables["generator_node"]["value"],
                           dtype=int)
        link_ports = {u["name"]: [bus for bus, _ in u["ports"]]
                      for u in self.unitblocks.values() if "ports" in u}
        ports = []
        for g, (kind, name) in enumerate(self._generator_owner):
            if kind == "links" and name in link_ports:
                ports.append(link_ports[name])
            else:
                ports.append([buses[nodes[g]]])
        self._generator_ports_cache = ports
        return ports

    def _generator_carrier_nodes(self):
        """
        The node of each generator on each carrier, -1 where it has none,
        [generators, carriers]: column c is GeneratorNode of the carrier c.
        """
        ports = self._generator_ports()
        table = -np.ones((len(ports), len(self._carriers)), dtype=int)
        for g, buses in enumerate(ports):
            for bus in buses:
                c, node = self._bus_node[bus]
                table[g, c] = node
        return table

    def _add_unit_outputs(self, block, name, unit_block):
        """
        Adds to the Block of the unit `name` its outputs [see
        UnitBlock::get_output_carrier()]: one on the carrier of each port of
        a link between carriers, with its efficiency, and otherwise one with
        efficiency 1 on the carrier of the node of its generators, given per
        generator if they are not all on the same carrier (the ports of a
        ConversionUnitBlock); a unit whose generators are all on the
        electricity has the default output and nothing is written.
        """
        T = int(self.dimensions["UCBlock"]["TimeHorizon"])
        if "ports" in unit_block:
            ports = unit_block["ports"]
            block.add_dimension("NumberOutputs", len(ports))
            block.add_variable(
                "OutputCarrier", "str", ("NumberOutputs",),
                np.array([self._carriers[self._bus_node[bus][0]]
                          for bus, _ in ports], dtype=object))
            block.add_variable(
                "OutputEfficiency", "float", ("TimeHorizon", "NumberOutputs"),
                np.column_stack([np.broadcast_to(eff, (T,))
                                 for _, eff in ports]))
            return

        if not hasattr(self, "_unit_generators"):
            # the generators of each unit, which are numbered unit after unit
            owners = self._generator_owner
            self._unit_generators = {}
            g = 0
            for key in self.unitblocks:
                if key.startswith("DCNetworkBlock"):
                    continue
                first = g
                while g < len(owners) and owners[g] == owners[first]:
                    g += 1
                self._unit_generators[key] = list(range(first, g))
        ports = self._generator_ports()
        carriers = [self._bus_node[ports[g][0]][0]
                    for g in self._unit_generators[name]]
        if not any(carriers):
            return
        block.add_dimension("NumberOutputs", 1)
        if len(set(carriers)) == 1:
            block.add_variable(
                "OutputCarrier", "str", ("NumberOutputs",),
                np.array([self._carriers[carriers[0]]], dtype=object))
            block.add_variable("OutputEfficiency", "float",
                               ("TimeHorizon", "NumberOutputs"),
                               np.ones((T, 1)))
            return
        if "NumberGenerators" not in block.dimensions:
            block.add_dimension("NumberGenerators", len(carriers))
        block.add_variable(
            "OutputCarrier", "str", ("NumberGenerators", "NumberOutputs"),
            np.array([[self._carriers[c]] for c in carriers], dtype=object))
        block.add_variable(
            "OutputEfficiency", "float",
            ("TimeHorizon", "NumberGenerators", "NumberOutputs"),
            np.ones((T, len(carriers), 1)))

    def _split_carriers(self, kwargs):
        """
        Splits the nodes, the demand and the lines of the UCBlock among the
        carriers: those of the electricity stay in `kwargs`, with the lines
        and the nodes numbered among them, those of each other carrier c go
        to the Block of its group "Carrier_c", the demand being its
        "EnergyDemand".

        Returns
        -------
        tuple
            The keyword arguments of the UCBlock and the Block of each
            carrier other than the electricity, by carrier.
        """
        kwargs = dict(kwargs)
        kwargs.pop("GeneratorNode", None)
        carrier = np.array([c for c, _ in self._bus_node.values])

        # one network for the whole horizon only if the electricity has
        # extendable lines [see _carrier_design_networks()]
        if "NumberNetworks" in kwargs and self.capacity_expansion_ucblock:
            line_carrier, _ = self._line_layout()
            design = np.asarray(self.networkblock["Design"]["DesignLines"]
                                ["value"], dtype=int)
            if not np.any(line_carrier[design] == 0):
                del kwargs["NumberNetworks"]
        node = np.array([k for _, k in self._bus_node.values])
        blocks = {c: Block() for c in range(1, len(self._carriers))}

        def dims_of(var):
            d = var.dimensions
            return (d,) if isinstance(d, str) else tuple(d)

        # the nodes and their demand
        demand = kwargs["ActivePowerDemand"]
        values = np.asarray(demand.data, dtype=float)
        kwargs["NumberNodes"] = int((carrier == 0).sum())
        kwargs["ActivePowerDemand"] = Variable(
            demand.name, demand.var_type, dims_of(demand),
            values[carrier == 0])
        for c, block in blocks.items():
            block.add_dimension("NumberNodes", int((carrier == c).sum()))
            block.add_variable("EnergyDemand", demand.var_type,
                               dims_of(demand), values[carrier == c])
        if "NodeName" in kwargs:
            names = np.asarray(kwargs["NodeName"].data, dtype=object)
            kwargs["NodeName"] = Variable(
                "NodeName", kwargs["NodeName"].var_type, ("NumberNodes",),
                names[carrier == 0])

        # the lines; a network of more than one node has their number even
        # if it has none, as that of the electricity
        line_carrier, _ = self._line_layout()
        for c, block in blocks.items():
            if (carrier == c).sum() > 1 and not np.any(line_carrier == c):
                block.add_dimension("NumberLines", 0)
        if line_carrier.size == 0:
            return kwargs, blocks
        originals = {key: var for key, var in kwargs.items()
                     if isinstance(var, Variable)
                     and "NumberLines" in dims_of(var)}
        for c in range(len(self._carriers)):
            selected = np.flatnonzero(line_carrier == c)
            if c == 0:
                kwargs["NumberLines"] = int(selected.size)
            elif selected.size:
                blocks[c].add_dimension("NumberLines", int(selected.size))
            else:
                continue
            extra_dims = set()
            for key, var in originals.items():
                dims = dims_of(var)
                data = np.take(np.asarray(var.data), selected,
                               axis=dims.index("NumberLines"))
                # the nodes of the lines, numbered in their carrier
                if key in ("StartLine", "EndLine"):
                    data = node[data.astype(int)].astype(data.dtype)
                if c == 0:
                    if selected.size:
                        kwargs[key] = Variable(key, var.var_type, dims, data)
                    else:
                        del kwargs[key]
                else:
                    blocks[c].add_variable(key, var.var_type, dims, data)
                    extra_dims.update(d for d in dims if d not in (
                        "NumberLines", "NumberNodes", "TimeHorizon"))
            # the dimensions of the network data are read in its group
            for d in sorted(extra_dims):
                blocks[c].add_dimension(d, int(kwargs[d]))
        # the branches of hyperarcs, all of the electricity [see
        # _line_layout()]
        for key in ("StartLine", "EndLine"):
            var = kwargs.get(key)
            if isinstance(var, Variable) and key not in originals:
                data = np.asarray(var.data)
                kwargs[key] = Variable(key, var.var_type, dims_of(var),
                                       node[data.astype(int)].astype(
                                           data.dtype))
        return kwargs, blocks

    def _carrier_design_networks(self, ucblock, design_kwargs):
        """
        Adds, for each carrier with extendable lines, the DesignNetworkBlock
        of its network, the only one covering the whole horizon: that of the
        electricity in the UCBlock, those of the other carriers in their
        groups "Carrier_c" [see UCBlock::deserialize()]. The data of the
        design lines are those of `design_kwargs`, split among the carriers,
        and the lines are numbered within their carrier.
        """
        line_carrier, index = self._line_layout()
        lines = np.asarray(design_kwargs["DesignLines"].data, dtype=int)
        for c in np.unique(line_carrier[lines]):
            selected = np.flatnonzero(line_carrier[lines] == c)
            kwargs = {}
            for key, var in design_kwargs.items():
                if not isinstance(var, Variable):
                    kwargs[key] = var
                    continue
                dims = ((var.dimensions,) if isinstance(var.dimensions, str)
                        else tuple(var.dimensions))
                data = np.asarray(var.data)
                if "NumberDesignLines" in dims:
                    data = np.take(data, selected,
                                   axis=dims.index("NumberDesignLines"))
                if key == "DesignLines":
                    data = index[lines[selected]].astype(data.dtype)
                kwargs[key] = Variable(key, var.var_type, dims, data)
            kwargs["NumberDesignLines"] = int(selected.size)
            design_block = Block().from_kwargs(
                block_type="DesignNetworkBlock", **kwargs)
            if c == 0:
                ucblock.add_block("NetworkBlock_0", block=design_block)
            else:
                carrier = ucblock.blocks[f"Carrier_{c}"]
                carrier.add_dimension("NumberNetworks", 1)
                carrier.add_block("NetworkBlock_0", block=design_block)

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

        # with the carriers, the design lines of each carrier are in the
        # DesignNetworkBlock of its network
        if self.multi_carrier:
            self._carrier_design_networks(master.blocks[ucblock_name],
                                          design_kwargs)
            return

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
            if stochastic_type == "sddp":
                block_type = "SDDPBlock"
                inner_block_name = "Block_0"
            else:
                if stochastic_type not in ("tssb", "mssb"):
                    raise ValueError(
                        f"Unsupported stochastic_type in optimize: {stochastic_type!r}"
                    )
                if self.problem_structure.get("investment_outside", False):
                    block_type = "InvestmentBlock"
                    inner_block_name = "Block_0"
                else:
                    block_type = ("MultiStageStochasticBlock"
                                  if stochastic_type == "mssb"
                                  else "TwoStageStochasticBlock")
                    inner_block_name = "Block_0"
        elif self.problem_structure.get("has_investment_block", False):
            block_type = "InvestmentBlock"
            inner_block_name = "InvestmentBlock"

        else:
            block_type = "UCBlock"
            inner_block_name = "Block_0"

        # --------------------------------------------------
        # Workdir
        # --------------------------------------------------

        workdir = Path(self.workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------
        # Resolve configfile/template
        # --------------------------------------------------

        default_template_map = {
            "UCBlock": "UCBlock/uc_solverconfig.txt",
            "InvestmentBlock": "InvestmentBlock/BSPar.txt",
            "TwoStageStochasticBlock": "TSSBlock/TSSBSCfg.txt",
            "MultiStageStochasticBlock": "TSSBlock/TSSBSCfg.txt",
        }

        cfg = self.configfile

        if block_type == "SDDPBlock":
            # SDDP richiede 10 file di configurazione SMSPP nella workdir.
            # I primi due sono i punti di ingresso (-S e -B), gli altri sono
            # referenziati al loro interno.
            required_sddp_configs = [
                "SDDPSCfg.txt",  # -S: SolverConfig
                "SDDPBCfg.txt",  # -B: BlockConfig (meta-map)
                "PFBCfg.txt",  # PolyhedralFunctionBlock
                "SUBCfg.txt",  # SlackUnitBlock
                "BUBCfg.txt",  # BatteryUnitBlock
                "TUBCfg-OVC.txt",  # ThermalUnitBlock
                "DCNBCfg.txt",  # DCNetworkBlock
                "ACBCfg.txt",  # ACNetworkBlock
                "BCfg.txt",  # default per gli altri Block
                "UCBSCfg.txt",  # strInnerBSC dentro SDDPSCfg.txt
            ]

            missing = [
                name for name in required_sddp_configs
                if not (workdir / name).exists()
            ]

            if missing:
                raise FileNotFoundError(
                    f"Mancano {len(missing)} dei {len(required_sddp_configs)} "
                    f"file di configurazione SMSPP nella workdir {workdir}:\n"
                    + "\n".join(f"  - {name}" for name in missing)
                    + "\n\nQuesti file si trovano nel repository SMSPP, "
                      "tipicamente sotto tools/sddp_solver/config/ o una "
                      "cartella simile. Vanno copiati nella workdir prima "
                      "di lanciare l'ottimizzazione."
                )
            configfile = pysmspp.SMSConfig(
                fp=str((workdir / "SDDPSCfg.txt").resolve())
            )
        elif cfg is None or cfg == "auto":
            if block_type not in default_template_map:
                raise ValueError(
                    f"No default config template is defined for block type {block_type!r}. "
                    f"Please provide self.configfile explicitly."
                )
            template = default_template_map[block_type]
            # modules make the design integer, which the BundleSolver of the
            # InvestmentBlock has to be told to keep integer
            if (block_type == "InvestmentBlock"
                    and getattr(self, "_investment_modules", False)):
                template = "InvestmentBlock/BSPar-int.txt"
            configfile = pysmspp.SMSConfig(template=str(template))
        else:
            if isinstance(cfg, pysmspp.SMSConfig):
                configfile = cfg
            elif isinstance(cfg, str):
                if os.path.isfile(cfg):
                    configfile = pysmspp.SMSConfig(fp=str(os.path.abspath(cfg)))
                else:
                    configfile = pysmspp.SMSConfig(template=str(cfg))
            elif isinstance(cfg, Path):
                configfile = pysmspp.SMSConfig(fp=str(cfg.resolve()))
            else:
                raise ValueError(
                    f"Invalid type for configfile: {type(cfg)}. "
                    f"Expected None, 'auto', str path, or SMSConfig instance."
                )

        # --------------------------------------------------
        # filepaths
        # --------------------------------------------------

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

        # pySMSpp non ha un template per SDDP, quindi usiamo i file di configurazione di SMSPP
        if block_type == "SDDPBlock":
            solver_options.setdefault("B", "SDDPBCfg.txt")
            solver_options.setdefault("smspp_solver", "SDDPSolver")
            solver_options.setdefault("solver_path", "sddp_solver")

        # pySMSpp has no tool of its own for the multi-stage block: the command
        # line of mssb_solver is the one of tssb_solver, so the same wrapper
        # serves, pointed at the other executable
        if block_type == "MultiStageStochasticBlock":
            solver_options.setdefault("smspp_solver", "TSSBSolver")
            solver_options.setdefault("solver_path", "mssb_solver")

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
        if not (isinstance(self.merge_links, (bool, str))
                or (isinstance(self.merge_links, Sequence)
                    and all(isinstance(m, str) for m in self.merge_links))):
            raise TypeError("merge_links must be a boolean, a string or a "
                            "sequence of strings.")
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

            if self.problem_structure["stochastic_type"] not in ("tssb", "mssb","sddp"):
                raise ValueError(
                    f"Unsupported stochastic type: "
                    f"{self.problem_structure['stochastic_type']!r}"
                )

            if ( self.problem_structure.get("investment_outside", False)
                 and not self.problem_structure.get("has_investment_block",
                                                    False) ):
                raise ValueError(
                    "'investment_outside' states the investment in an "
                    "InvestmentBlock wrapping the stochastic Block, hence it "
                    "needs the investment to go through an InvestmentBlock: "
                    "set capacity_expansion_ucblock=False."
                )

            if self.problem_structure["stochastic_type"] == "mssb":
                tree = self.problem_structure.get("scenario_tree", None)
                if tree is None:
                    raise ValueError(
                        "A multi-stage problem needs the scenario tree saying "
                        "how the scenarios are grouped. Set stochastic_parameters"
                        "={'stochastic_type': 'mssb', 'parameters': [...], "
                        "'tree': {...}}."
                    )

                in_tree = [name for group in tree["groups"]
                           for name, _ in group["scenarios"]]
                in_network = list(self.problem_structure["scenario_names"])
                if sorted(in_tree) != sorted(in_network):
                    raise ValueError(
                        "The leaves of the scenario tree are not the scenarios "
                        f"of the network: {sorted(in_tree)} against "
                        f"{sorted(in_network)}."
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

            stochastic_parameters = self.problem_structure.get(
                "stochastic_parameters", []
            )

            if not stochastic_parameters:
                raise ValueError(
                    "The network is stochastic but no stochastic parameter was declared. "
                    "Set stochastic_parameters={'stochastic_type': 'tssb', "
                    "'parameters': [...]}."
                )

            for parameter in stochastic_parameters:
                spec = STOCHASTIC_PARAMETER_REGISTRY[parameter]

                if spec.get("requires_enable_thermal_units", False):
                    if not self.enable_thermal_units:
                        raise ValueError(
                            f"Stochastic parameter {parameter!r} requires "
                            "enable_thermal_units=True."
                        )
            if self.problem_structure["stochastic_type"] == "sddp":
                # Normalizza e valida i parametri SDDP:
                # - snapshots_per_stage
                # - top_path
                # - benders_path
                # normalize_sddp_parameters solleva ValueError con messaggi chiari
                # se uno dei campi manca o è malformato.
                sddp_parameters = mu.normalize_sddp_parameters(self.stochastic_parameters)

                # Verifica che snapshots_per_stage divida il numero totale di snapshot
                # della rete, altrimenti la partizione in stadi non è esatta.
                total_snapshots = len(n.snapshots)
                sps = sddp_parameters["snapshots_per_stage"]
                if total_snapshots % sps != 0:
                    raise ValueError(
                        f"'snapshots_per_stage' = {sps} non divide il numero totale "
                        f"di snapshot della rete ({total_snapshots}). Scegli un valore "
                        f"che divida esattamente {total_snapshots}."
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

        stochastic_type = self.problem_structure.get("stochastic_type")
        if stochastic_type == "sddp":
            return None
        if stochastic_type not in ("tssb", "mssb"):
            raise ValueError(
                f"prepare_tssb_interface only supports 'tssb' and 'mssb', got "
                f"{stochastic_type!r}."
            )

        #TODO build demand node by node instead of node0, node1, node0, node1
        dss_data = self.build_tssb_dss(n)

        design_variables = self._collect_design_variables()
        sap_data = build_tssb_static_abstract_path(design_variables)

        self.dimensions["tssb"]["sap"] = {
            "PathDim": sap_data["PathDim"],
            "TotalLength": sap_data["TotalLength"],
        }

        # with a second decision stage the two levels of the tree tie
        # different things: the outer one only what is decided at the root,
        # the inner one everything, so that what is decided once the branch
        # is known is common to its leaves and free across branches
        root_variables = self._root_design_variables(design_variables)
        if root_variables is not None:
            sap_root = build_tssb_static_abstract_path(root_variables)
            self.dimensions["tssb"]["sap_root"] = {
                "PathDim": sap_root["PathDim"],
                "TotalLength": sap_root["TotalLength"],
            }
        else:
            sap_root = None

        stochastic_block = self.build_tssb_stochastic_block(n)

        self.tssb_data = {
            "enabled": True,
            "discrete_scenario_set": dss_data,
            "static_abstract_path": sap_data,
            "static_abstract_path_root": sap_root,
            "stochastic_block": stochastic_block,
        }

        return self.tssb_data

    def build_tssb_dss(self, n):
        """
        Build the payload for the DiscreteScenarioSet of a TSSB problem.

        The selected stochastic parameters are read from the registry. Demand is
        handled by a dedicated UCBlock-level builder, while all UnitBlock-level
        time series parameters share the same generic DSS builder.
        """
        dss_parts = []

        self.tssb_parameter_specs = {}
        self.tssb_parameter_asset_order = {}

        for parameter in self.problem_structure.get("stochastic_parameters", []):
            spec = STOCHASTIC_PARAMETER_REGISTRY[parameter]
            mapping_kind = spec["mapping_kind"]

            self.tssb_parameter_specs[parameter] = dict(spec)

            if mapping_kind == "ucblock_timeseries":
                if parameter != "demand":
                    raise ValueError(
                        f"Unsupported UCBlock stochastic parameter {parameter!r}."
                    )

                dss_parts.append(build_dss_demand(
                    n, bus_order=self._demand_bus_order(n)))
                continue

            if mapping_kind == "unitblock_timeseries":
                asset_names = get_stochastic_parameter_asset_names(
                    n=n,
                    parameter=parameter,
                    spec=spec,
                    intermittent_carriers=self.intermittent_carriers,
                    default_intermittent_carriers=renewable_carriers,
                    enable_thermal_units=self.enable_thermal_units,
                )

                self.tssb_parameter_asset_order[parameter] = list(asset_names)

                dss_parts.append(
                    build_dss_unitblock_timeseries_parameter(
                        n=n,
                        parameter=parameter,
                        pypsa_component=spec["pypsa_component"],
                        field=spec["field"],
                        asset_names=asset_names,
                        function_name=spec["function_name"],
                        unitblock_type=spec["unitblock_type"],
                        target=spec["target"],
                        transformation_config=self.config,
                        smspp_parameter=spec.get("smspp_parameter", None),
                        weights=bool(spec.get("weights", False)),
                        static_reference=getattr(self, "n_preprocessed", None),
                    )
                )
                continue

            raise ValueError(
                f"Unsupported stochastic mapping_kind={mapping_kind!r} "
                f"for parameter {parameter!r}."
            )

        dss_data = merge_tssb_dss_parts(dss_parts)

        self.dimensions.setdefault("tssb", {})
        self.dimensions["tssb"]["dss"] = {
            "NumberScenarios": int(dss_data["number_scenarios"]),
            "ScenarioSize": int(dss_data["scenario_size"]),
        }

        self.tssb_dss_offsets = {
            part["parameter"]: {
                "start": int(part["offset_start"]),
                "end": int(part["offset_end"]),
                "size": int(part["scenario_size"]),
            }
            for part in dss_data["parts"]
        }

        return dss_data


    def _store_unitblock_design_variable(self, attr_name, unitblock_index):
        """
        Store design-variable metadata for StaticAbstractPath construction.

        A path is stored only for a design Variable that the unit block has,
        which is the case when its investment cost is not zero (UCBlock
        creates x_battery and x_converter on BatteryInvestmentCost and
        ConverterInvestmentCost, x_thermal and x_intermittent on
        InvestmentCost), while a HydroUnitBlock has none: a path to a
        Variable that is not there stops the solver when it reads the
        stochastic Block.

        Parameters
        ----------
        attr_name : str
            Unit block parameter family name.
        unitblock_index : int
            Unit block index in the SMS++ structure.
        """
        block_type = attr_name.split("_")[0]
        unit = next((u for u in self.unitblocks.values()
                     if u.get("enumerate") == f"UnitBlock_{unitblock_index}"),
                    None)
        variables = unit["variables"] if unit is not None else {}

        def has_cost(key):
            entry = variables.get(key)
            if entry is None:
                return False
            value = entry["value"] if isinstance(entry, dict) else entry
            return bool(np.any(np.asarray(value, dtype=float) != 0))

        def store(var_name):
            self.unitblock_design_data.append(
                {
                    "block_index": unitblock_index,
                    "var_name": var_name,
                    "component_type": "unit",
                    "element_index": 0,
                    "range_index": 1,
                }
            )

        if block_type == "BatteryUnitBlock":
            if has_cost("BatteryInvestmentCost"):
                store("x_battery")
            if has_cost("ConverterInvestmentCost"):
                store("x_converter")
        elif block_type in ("ThermalUnitBlock", "NuclearUnitBlock"):
            if has_cost("InvestmentCost"):
                store("x_thermal")
        elif block_type == "IntermittentUnitBlock":
            if has_cost("InvestmentCost"):
                store("x_intermittent")
        elif block_type == "HydroUnitBlock":
            pass
        elif has_cost("InvestmentCost") or "InvestmentCost" not in variables:
            store("x_design")

    def _root_design_variables(self, design_variables):
        """
        The subset of the design descriptors that is decided at the root.

        Returns None when every decision is taken there, which is the ordinary
        two-stage case and the one where the two levels of the tree tie the
        very same things. The assignment comes by component name, and the
        descriptors carry the index of their Block, so the names are resolved
        through the unit blocks the converter has just built.
        """
        stages = self.problem_structure.get("design_stages", None)
        if not stages:
            return None

        wanted = set( stages[ "root" ] )
        indices = set()
        for name, unitblock in self.unitblocks.items():
            if ( name in wanted ) or ( unitblock.get( "name" ) in wanted ):
                position = str( unitblock.get( "enumerate" , "" ) ).split( "_" )
                if len( position ) == 2 and position[ 1 ].isdigit():
                    indices.add( int( position[ 1 ] ) )

        missing = len( wanted ) - len( indices )
        if missing > 0:
            raise ValueError(
                f"{missing} of the components said to be decided at the root "
                "have no unit Block: design_stages names components of the "
                "network, and only the expandable ones have a design Variable."
            )

        root = [ dv for dv in design_variables
                 if int( dv[ "block_index" ] ) in indices ]

        if not root:
            raise ValueError(
                "no design Variable is left at the root: with everything "
                "decided after the branch is known the branches do not share "
                "anything and the problem is not a tree."
            )

        return root

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

        if self.multi_carrier and investment_meta["design_lines"]:
            self.design_variables = calculate_design_variables(
                investment_meta={"design_lines": []},
                unitblock_design_data=self.unitblock_design_data,
                network_block_index=network_block_index,
            ) + self._carrier_design_variables(
                investment_meta["design_lines"], network_block_index)
            return self.design_variables

        self.design_variables = calculate_design_variables(
            investment_meta=investment_meta,
            unitblock_design_data=self.unitblock_design_data,
            network_block_index=network_block_index,
        )
        return self.design_variables

    def _carrier_design_variables(self, design_lines, first_network):
        """
        The design variables of the extendable lines with the carriers, in
        the order of the lines of the network: that of a line of carrier c
        is in the DesignNetworkBlock of the carrier [see
        _carrier_design_networks()], at the position of the line among the
        design lines of the carrier. The NetworkBlock of the UCBlock follow
        its units, those of the electricity first and then those of each
        other carrier: one with a design, as many as the instants if it has
        more than one node, none otherwise.
        """
        line_carrier, _ = self._line_layout()
        design_lines = np.asarray(design_lines, dtype=int)
        bus_carrier = np.array([c for c, _ in self._bus_node.values])
        T = int(self.dimensions["UCBlock"]["TimeHorizon"])
        block_index = []
        index = first_network
        for c in range(len(self._carriers)):
            block_index.append(index)
            if np.any(line_carrier[design_lines] == c):
                index += 1
            elif (bus_carrier == c).sum() > 1:
                index += T
        positions = {}
        variables = []
        for line in design_lines:
            c = int(line_carrier[line])
            position = positions.get(c, 0)
            positions[c] = position + 1
            variables.append({
                "block_index": int(block_index[c]),
                "var_name": "x_network",
                "component_type": "network",
                "element_index": position,
                "range_index": position + 1,
            })
        return variables

    def build_tssb_stochastic_block(self, n):
        """
        Build the StochasticBlock payload for TSSB.

        Demand maps directly to UCBlock::set_active_power_demand.
        UnitBlock-level stochastic parameters are mapped one asset at a time
        using the asset order already used in the DSS.
        """
        data_mappings = []
        time_horizon = int(self.dimensions["UCBlock"]["TimeHorizon"])

        for parameter in self.problem_structure.get("stochastic_parameters", []):
            spec = STOCHASTIC_PARAMETER_REGISTRY[parameter]
            mapping_kind = spec["mapping_kind"]

            if parameter not in self.tssb_dss_offsets:
                raise KeyError(
                    f"No DSS offset found for stochastic parameter {parameter!r}."
                )

            offset = self.tssb_dss_offsets[parameter]

            if mapping_kind == "ucblock_timeseries":
                if parameter != "demand":
                    raise ValueError(
                        f"Unsupported UCBlock stochastic parameter {parameter!r}."
                    )

                data_mappings.append(
                    build_stochastic_mapping_demand(
                        set_from_start=offset["start"],
                        set_from_end=offset["end"],
                        scenario_size=offset["size"],
                        # with the carriers, the demand of all their nodes
                        function_name=(
                            "UCBlock::set_energy_demand"
                            if self.multi_carrier else
                            spec["function_name"]),
                    )
                )
                continue

            if mapping_kind == "unitblock_timeseries":
                asset_names = self.tssb_parameter_asset_order.get(parameter, None)

                if asset_names is None:
                    raise KeyError(
                        f"No asset order was stored for stochastic parameter "
                        f"{parameter!r}."
                    )

                unitblock_indices = collect_unitblock_indices_by_names_and_type(
                    unitblocks=self.unitblocks,
                    names=asset_names,
                    block_type=spec["unitblock_type"],
                )

                number_units = len(unitblock_indices)
                expected_size = number_units * time_horizon

                if offset["size"] != expected_size:
                    raise ValueError(
                        f"Mismatch between stochastic {parameter!r} DSS size and "
                        f"UnitBlock mappings. DSS size is {offset['size']}, while "
                        f"{number_units} units and TimeHorizon={time_horizon} imply "
                        f"{expected_size}."
                    )

                for local_idx, unitblock_index in enumerate(unitblock_indices):
                    set_from_start = offset["start"] + local_idx * time_horizon
                    set_from_end = set_from_start + time_horizon

                    data_mappings.append(
                        build_stochastic_mapping_single_unit(
                            target=spec["target"],
                            function_name=spec["function_name"],
                            set_from_start=set_from_start,
                            set_from_end=set_from_end,
                            time_horizon=time_horizon,
                            unitblock_index=unitblock_index,
                        )
                    )

                self.tssb_parameter_unitblock_indices = getattr(
                    self, "tssb_parameter_unitblock_indices", {}
                )
                self.tssb_parameter_unitblock_indices[parameter] = unitblock_indices

                continue

            raise ValueError(
                f"Unsupported stochastic mapping_kind={mapping_kind!r} "
                f"for parameter {parameter!r}."
            )

        if not data_mappings:
            raise ValueError(
                "No stochastic data mappings were built for the TSSB StochasticBlock."
            )

        if self.hydro_system:
            data_mappings = hydro_system_mapping_paths(data_mappings,
                                                       self.unitblocks)
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
    ############################## SDDP methods #################################################
    #############################################################################################

    def _generate_stage_ucblock(
            self,
            n_stage,
            stage_idx: int,
            workdir: Path,
    ):
        """
        Prende una rete PyPSA già slicizzata (deterministica, con solo gli snapshot
        di uno stage) e produce un file Block_{stage_idx}.nc4, che contiene un UCBlock
        costruito da quella rete.

        Un'accortezza che potrebbe rallentare la pipeline è il fatto che usa una Transformation
        fresca per stage. Questo perché self.unitblocks, self.demand, ecc. non si contaminano
        tra uno stage e l'altro.

        Parameters
        ----------
        n_stage: pypsa.Network
            Rete PyPSA deterministica, già slicizzata
        stage_idx: int
            Indice dello stage usato per il nome del file
        workdir: Path
            Directory dove scrivere il file Block_{stage_idx}.nc4

        Returns
        -------
        Path al file generato: workdir / f"Block_{stage_idx}.nc4"
        """
        # Controlliamo che la directory di lavoro esista
        workdir.mkdir(parents=True, exist_ok=True)

        # Istanziamo una nuova Transformation con gli stessi parametri di configurazione della principale
        sub = Transformation(
            merge_links=self.merge_links,
            enable_thermal_units=self.enable_thermal_units,
            intermittent_carriers=self.intermittent_carriers,
            nuclear_units=self.nuclear_units,
            capacity_expansion_ucblock=self.capacity_expansion_ucblock,
            workdir=workdir,
            name=f"stage_{stage_idx}",
            overwrite=True,
        )

        # Prepariamo la rete
        n_stage.calculate_dependent_values()
        if len(n_stage.stores) != 0:
            n_stage.stores['max_hours'] = sub.config.max_hours_stores
            n_stage.stores['snapshots_weighting'] = n_stage.snapshot_weightings['stores'].iloc[0]

        if len(n_stage.storage_units) != 0:
            n_stage.storage_units['snapshots_weighting'] = n_stage.snapshot_weightings['stores'].iloc[0]

        # HydroUnitBlock::set_initial_volume lancia std::invalid_argument se
        # InitialVolumetric < 0, quindi il Block_i.nc4 non può contenere la
        # sentinella ciclica -1. Forziamo sulla rete slicizzata:
        #   - cyclic_state_of_charge = False  (così la lambda non emette -1)
        #   - state_of_charge_initial = V0    (>= 0)
        # con V0 = fraction * p_nom * max_hours = fraction * MaxVolumetric.
        hydro_carriers = {"hydro", "phs"}

        if not n_stage.storage_units.empty:
            # carriers delle storage unit, normalizzati (strip + lower) per
            # confrontarli con hydro_carriers senza sorprese da spazi/maiuscole
            carriers = (
                n_stage.storage_units["carrier"]
                .astype(str)
                .str.strip()
                .str.lower()
            )
            hydro_mask = carriers.isin(hydro_carriers)

            if hydro_mask.any():
                # Frazione letta da self (default 0.5 se non passata dall'utente)
                fraction = self.config.sddp_initial_state_fraction

                # Itera sulle storage unit idro dello stage (rete deterministica,
                # quindi Index semplice, non MultiIndex)
                for name, row in n_stage.storage_units[hydro_mask].iterrows():
                    # p_nom può essere NaN se la rete è malformata: in quel caso
                    # usiamo 0 (V0 = 0 è ammissibile perché MinVolumetric = 0)
                    p_nom = row["p_nom"]
                    p_nom = 0.0 if pd.isna(p_nom) else float(p_nom)

                    # max_hours è l'altro fattore di MaxVolumetric. Se NaN o <= 0
                    # la capacità dell'invaso è 0, quindi anche V0 = 0.
                    max_hours = row["max_hours"]
                    if pd.isna(max_hours) or float(max_hours) <= 0:
                        v0 = 0.0
                    else:
                        v0 = fraction * p_nom * float(max_hours)

                    # Disattiva la sentinella ciclica: la lambda in
                    # transformation_config.py restituisce -1 SOLO se
                    # cyclic_state_of_charge è True. Mettendola a False, la
                    # lambda restituisce state_of_charge_initial.values, cioè v0.
                    n_stage.storage_units.loc[name, "cyclic_state_of_charge"] = False

                    # Imposta il volume iniziale numerico. La lambda farà:
                    # InitialVolumetric = state_of_charge_initial.values
                    # che è esattamente v0.
                    n_stage.storage_units.loc[name, "state_of_charge_initial"] = v0

        sub.direct(n_stage)

        # Forma di MaxVolumetric, MinVolumetric, InitialVolumetric

        _hydro_params_to_patch = (
            "MaxVolumetric",
            "MinVolumetric",
            "InitialVolumetric",
        )

        for _ub_name, _ub in sub.unitblocks.items():
            if _ub.get("block") != "HydroUnitBlock":
                continue

            _n_res = int(
                _ub.get("dimensions", {}).get("NumberReservoirs", 1)
            )
            if _n_res < 1:
                raise ValueError(
                    f"{_ub_name}: NumberReservoirs = {_n_res} non valido."
                )

            for _param in _hydro_params_to_patch:
                if _param not in _ub["variables"]:
                    continue

                _var = _ub["variables"][_param]
                _arr = np.asarray(_var["value"], dtype=float).reshape(-1)

                if _arr.size == 0:
                    raise ValueError(
                        f"{_ub_name}/{_param}: valore vuoto, impossibile "
                        "determinare il volume rappresentativo."
                    )

                if _arr.size == _n_res:
                    _new_value = _arr
                else:
                    # Valore costante nel tempo: prendiamo il primo e
                    # broadcast su n_res posizioni.
                    if _arr.size > 1 and not np.allclose(_arr, _arr[0]):
                        warnings.warn(
                            f"{_ub_name}/{_param}: valore non costante nel "
                            f"tempo (size={_arr.size}), verrà sostituito dal "
                            f"primo elemento {_arr[0]}."
                        )
                    _new_value = np.full(_n_res, _arr[0], dtype=float)

                _ub["variables"][_param] = {
                    "value": _new_value,
                    "type": "float",
                    "size": ("NumberReservoirs",),
                }

        sn = SMSNetwork(file_type=SMSFileType.eBlockFile)

        # Costruiamo lo UCBlock dentro sn
        # Internamente ogni blocco è nominato Block_0
        # Il nome esterno è quello che cambia
        sub.convert_to_ucblock(sn, index_id=stage_idx, name_id="Block_0", wrap_hydro_in_system=True)

        # Salviamo su disco
        path = workdir / f"Block_{stage_idx}.nc4"

        sn.to_netcdf(path, force=True)

        return path


    def generate_stage_blocks(
            self,
            n,
            stage_snapshot_map: Mapping[str, pd.Index],
            workdir: Path|None,
    ) -> List[Path]:
        """
        Itera su tutti gli stage e per ciascuno:
            1. Slide dellar ete agli snapshot dello stage;
            2. Generazione del Block_{stage_idx}.nc4;
        Restituisce la lista dei path generati, nello stesso ordine degli stage.

        Parameters
        ----------
        n: pypsa.Network
        stage_snapshot_map: Mapping[str, pd.Index]
            {stage_name: pd.Index(snapshots)} -> Tipicamente l'output di get_sddp_stage_snapshots
        workdir: Path|None
            Directory dove scrivere i file. Se None, usa self.workdir

        Returns
        -------
        List[Path]
            Lista che contiene i path dei file generati, nell'ordine degli stage
        """
        # Verifichiamo la directory
        if workdir is None:
            workdir = Path(self.workdir)
        else:
            workdir = Path(workdir)

        workdir.mkdir(parents=True, exist_ok=True)

        # Inizializziamo la lista che sarà restituita in output
        # Contiene i path dei file generati
        paths: List[Path] = []

        # Iteriamo sugli stage
        for stage_idx, (stage_name, stage_snapshots) in enumerate(stage_snapshot_map.items()):
            print(f"[SDDP] Generando Block_{stage_idx}.nc4 per {stage_name} ({len(stage_snapshots)} snapshot)")

            # Slice della rete
            # slice_network_for_stage gestisce anche il caso in cui stage_snapshots è un list
            # invece di un pd.Index, in quanto effettua il casting internamente
            n_stage = slice_network_for_stage(n, stage_snapshots)

            # Generiamo il blocco
            path = self._generate_stage_ucblock(n_stage, stage_idx, workdir)

            paths.append(path)

        return paths


    def prepare_sddp_interface(self, n):
        """
        Questa funzione popola self.sddp_data con le informazioni necessarie
        per costruire un SDDPBlock

        Passi:
        1. Normalizza i parametri stocastici SDDP (validando snapshots_per_stage).
        2. Deriva i nomi degli stadi (da make_sddp_periods, via get_sddp_stage_names).
        3. Partiziona gli snapshot della rete in stadi (uno stadio = una fetta di
            n.snapshots) tramite get_sddp_stage_snapshots.
        4. Calcola il TimeHorizon (numero di snapshot) di ogni stadio.
        5. Per ogni stadio estrae i dati stocastici di tutti i parametri richiesti
            e li concatena in un unico vettore di scenario.
        6. Costruisce la matrice scenari globale e le dimensioni dell'SDDPBlock.

        Salva il risultato in self.sddp_data, insieme agli offset dei parametri
        dentro il vettore di scenario di ogni stadio (serviranno ai data mapping).
        """
        if not self.problem_structure.get("is_stochastic", False):
            return None

        if self.problem_structure.get("stochastic_type") != "sddp":
            return None

        # --- SDDP richiede una rete con storage idro ---
        if not mu.network_has_hydro_storage(n):
            raise ValueError(
                "La pipeline SDDP richiede che la rete abbia almeno una "
                "StorageUnit con carrier idroelettrico ('hydro' o 'PHS'). "
                "Senza stato (i volumi dei bacini), SDDP degenera in "
                "ottimizzazione deterministica multi-stage, che non è "
                "supportata da SDDPBlock in SMSPP. "
                "Se vuoi risolvere i singoli stage indipendentemente, usa "
                "la pipeline deterministica (direct + convert_to_ucblock) "
                "su ogni stage separatamente."
            )

        # Normalizziamo subito i parametri
        sddp_parameters = mu.normalize_sddp_parameters(self.stochastic_parameters)

        # Override la config solo se l'utente ha passato un valore esplicito.
        # Altrimenti usa il default da TransformationConfig.
        if "initial_state_fraction" in self.stochastic_parameters:
            self.config.sddp_initial_state_fraction = float(
                sddp_parameters["initial_state_fraction"]
            )

        # Otteniamo i nomi degli stadi
        stage_names = mu.get_sddp_stage_names(n, sddp_parameters)

        # Partizioniamo degli snapshot in stadi.
        stage_snapshots_map = mu.get_sddp_stage_snapshots(
            n,
            stage_names=stage_names,
            stochastic_parameters=sddp_parameters,
        )

        # TimeHorizon di ogni stadio = numero di snapshot di quello stadio.
        # Attenzione: non va confuso con il TimeHorizon dell'SDDPBlock, che
        # invece è len(stage_names) = numero di stadi.
        stage_time_horizons = mu.get_sddp_stage_time_horizons(stage_snapshots_map)

        # Azzeriamo gli accumulatori degli offset.
        #    - self.sddp_dss_offsets[stage_name][parameter] -> {start, end, size}
        #      Posizione del parametro dentro il vettore di scenario dello stadio.
        #    - self.sddp_parameter_asset_order[parameter] -> lista di nomi asset
        #      Ordine usato durante l'appiattimento, serve per costruire i data
        #      mapping in Fase 3 (SDDP).
        #    Vanno azzerati a ogni chiamata per evitare che chiamate successive
        #    accumulino dati stantii.
        self.sddp_dss_offsets = {}
        self.sddp_parameter_asset_order = {}

        # Lista dei parametri stocastici da processare
        stochastic_parameters = sddp_parameters["parameters"]

        stage_data_list = []

        for stage_name in stage_names:
            # Solo gli snapshot di questo stadio
            stage_snapshots = stage_snapshots_map[stage_name]

            # Estraiamo i dati di tutti i parametri stocastici per questo stadio
            stage_data = mu.build_sddp_stage_data(
                n=n,
                stage_name=stage_name,
                stage_snapshots=stage_snapshots,
                stochastic_parameters=stochastic_parameters,
                intermittent_carriers=self.intermittent_carriers,
                default_intermittent_carriers=renewable_carriers,
                enable_thermal_units=self.enable_thermal_units,
                transformation_config=self.config,
            )

            # Uniamo le parti in un unico vettore di scenario per questo stadio
            stage_merged = mu.merge_sddp_stage_data(
                stage_data["parts"],
                stage_name=stage_data["stage"],
            )

            stage_data_list.append(stage_merged)

            # Raccogliamo gli offset di ciascun parametro dentro il vettore di
            # scenario di questo stadio. Serviranno per costruire
            # i data mapping dello StochasticBlock dello stadio.
            stage_offsets = {}
            for part in stage_merged["parts"]:
                parameter = part["parameter"]
                stage_offsets[parameter] = {
                    "start": int(part["offset_start"]),
                    "end": int(part["offset_end"]),
                    "size": int(part["offset_end"] - part["offset_start"]),
                }
                # Ordine degli asset: uguale su tutti gli stadi per costruzione
                if "asset_order" in part and part["asset_order"] is not None:
                    self.sddp_parameter_asset_order[parameter] = list(part["asset_order"])

            self.sddp_dss_offsets[stage_name] = stage_offsets

        # Creazione e parsing dei Block_i.nc4
        workdir = Path(self.workdir)
        stage_block_paths = self.generate_stage_blocks(n, stage_snapshots_map, workdir=workdir)

        stage_structures = mu.parse_all_stage_blocks(stage_block_paths)
        mu.validate_stage_structures(stage_structures)

        # Costruiamo lo state_info dallo stage 0
        state_info = mu.build_sddp_state_info(
            stage_structures[0],
            time_horizon=len(stage_names)
        )

        # Costruiamo la matrice globale degli scenari
        scenarios_info = mu.build_sddp_scenarios(stage_data_list)
        # Calcoliamo le dimensioni per il costruttore di SDDPBlock
        dimensions = mu.build_sddp_dimensions(stage_data_list, scenarios_info, state_info=state_info)

        self.sddp_data = {
            "stage_names": stage_names,
            "stage_snapshots": stage_snapshots_map,
            "stage_time_horizons": stage_time_horizons,
            "parameters": stochastic_parameters,
            "stage_data_list": stage_data_list,
            "scenarios_info": scenarios_info,
            "dimensions": dimensions,
            "stage_structures": stage_structures,
            "state_info": state_info,
        }

        return self.sddp_data

    def add_sddp_top_abstract_path(self, sddp_block, ap_data, name_id="AbstractPath"):
        """
        Crea una blocco AbstractPath con 3 variabili e lo attacca al
        blocco padre SDDPBlock
        """
        ap_block = Block(
            block_type="AbstractPath",
            PathDim = Dimension(
                "PathDim",
                ap_data["PathDim"],
            ),
            TotalLength = Dimension(
                "TotalLength",
                ap_data["TotalLength"],
            ),
            PathStart = Variable(
                "PathStart",
                "u4",
                ("PathDim",),
                ap_data["PathStart"],
            ),
            PathNodeTypes = Variable(
                "PathNodeTypes",
                "c",
                ("TotalLength",),
                ap_data["PathNodeTypes"],
            ),
            PathGroupIndices = Variable(
                "PathGroupIndices",
                "u4",
                ("TotalLength",),
                ap_data["PathGroupIndices"],
            ),
        )

        sddp_block.add_block(name_id, block = ap_block)

        return sddp_block

    def add_benders_abstract_path(self, benders_func_block, ap_data, name_id="AbstractPath"):
        """
        Crea un blocco AbstractPath con 4 variabili (include PathElementIndices)
        e lo attacca alla BendersBFunction.
        """
        ap_block = Block(
            block_type = "AbstractPath",
            PathDim = Dimension(
                "PathDim",
                ap_data["PathDim"],
            ),
            TotalLength = Dimension(
                "TotalLength",
                ap_data["TotalLength"],
            ),
            PathStart = Variable(
                "PathStart",
                "u4",
                ("PathDim",),
                ap_data["PathStart"],
            ),
            PathNodeTypes = Variable(
                "PathNodeTypes",
                "c",
                ("TotalLength",),
                ap_data["PathNodeTypes"],
            ),
            PathGroupIndices = Variable(
                "PathGroupIndices",
                "u4",
                ("TotalLength",),
                ap_data["PathGroupIndices"],
            ),
            PathElementIndices = Variable(
                "PathElementIndices",
                "u4",
                ("TotalLength",),
                ap_data["PathElementIndices"],
            )
        )

        benders_func_block.add_block(name_id, block = ap_block)

        return benders_func_block


    def convert_to_sddp_block(self, master, index_id, name_id):
        """
        Aggiunge un SDDPBlock alla rete SMS++ (versione minimale).

        SDDPBlock
        ├── AbstractPath (top-level, "OBBB" con second/third_group_index dal parser)
        ├── ScenarioGenerator (DiscreteScenarioSet con la matrice scenari completa)
        ├── StochasticBlock_0
        │   ├── AbstractPath (path per ogni data mapping)
        │   ├── NumberDataMappings, SetSize, SetElements (popolati)
        │   ├── FunctionName, Caller, DataType (popolati)
        │   └── Block (BendersBlock)
        │         └── BendersBFunction
        │               ├── NumVar, NumRow, NumNonzero (da bf_data)
        │               ├── Column, A, NumNonzeroAtRow (A = identità sparse)
        │               ├── b (zeri, l'inflow lo scrive il DataMapping)
        │               ├── ConstraintSide ('R' per ogni riga)
        │               ├── AbstractPath ("BBC" verso il bilancio idrico)
        │               └── Block (segnaposto con id e filename)
        ├── StochasticBlock_1
        │     └── ...
        └── ...

        IMPORTANTE: non costruiamo davvero l'UCBlock dentro il BendersBlock.
        Inseriamo solo un Block vuoto con attributi "id" e "filename" che
        puntano al file .nc4 dove vive l'UCBlock di quello stadio. Sarà il
        solver (SDDPSolver) a caricarlo.
        """
        # -----------------------------------------------------------------
        # 0. Recupera i dati preparati da prepare_sddp_interface
        # -----------------------------------------------------------------
        sddp_data = self.sddp_data
        dimensions = sddp_data["dimensions"]
        scenarios_info = sddp_data["scenarios_info"]
        stage_names = sddp_data["stage_names"]

        # Numero di stadi (= TimeHorizon dell'SDDPBlock)
        time_horizon = dimensions["TimeHorizon"]

        # Verifichiamo che i nomi degli stadi e il TimeHorizon coincidano
        # Se non coincidono, significa che prepare_sddp_interface e build_sddp_dimensions
        # si sono disallineati
        if len(stage_names) != time_horizon:
            raise ValueError(
                f"Numero di stadi incoerente: {len(stage_names)} nomi di stadio "
                f"({list(stage_names)}) contro TimeHorizon={time_horizon}"
            )

        # -----------------------------------------------------------------
        # 1. Costruzione delle dimensioni dell'SDDPBlock
        # -----------------------------------------------------------------
        # Ogni "Dimension" del blocco SDDPBlock. I valori vengono da
        # build_sddp_dimensions(); usiamo int() per garantire che siano scalari.
        sddp_dim_kwargs = {
            "NumPolyhedralFunctionsPerSubBlock": Dimension(
                "NumPolyhedralFunctionsPerSubBlock",
                int(dimensions.get("NumPolyhedralFunctionsPerSubBlock", 1)),
            ),
            "TimeHorizon": Dimension("TimeHorizon", int(time_horizon)),
            "NumSubBlocksPerStage": Dimension(
                "NumSubBlocksPerStage",
                int(dimensions["NumSubBlocksPerStage"]),
            ),
            "SubScenarioSize": Dimension(
                "SubScenarioSize",
                dimensions["SubScenarioSize"],
            ),
            "NumberRandomDataGroups": Dimension(
                "NumberRandomDataGroups",
                int(dimensions["NumberRandomDataGroups"]),
            ),
            "AdmissibleStateSize": Dimension(
                "AdmissibleStateSize",
                int(dimensions["AdmissibleStateSize"]),
            ),
            "InitialStateSize": Dimension(
                "InitialStateSize",
                int(dimensions["InitialStateSize"]),
            ),
        }

        # -----------------------------------------------------------------
        # 2. Preparazione degli array che diventano Variabili dell'SDDPBlock
        # -----------------------------------------------------------------

        # Dimensione dei "random data groups": lista di interi per ogni stadio.
        size_random_data_groups_arr = np.asarray(
            dimensions["SizeRandomDataGroups"], dtype=np.uint32
        )

        # StateSize può essere uno scalare o un array; lo trattiamo come scalare
        # (nel notebook era np.array(3, dtype=uint32), quindi scalare).
        state_size_val = dimensions["StateSize"]
        if isinstance(state_size_val, np.ndarray):
            # Se è già un array, prendiamo il primo valore (caso scalare).
            state_size_arr = np.array(int(state_size_val.reshape(-1)[0]), dtype=np.uint32)
        else:
            state_size_arr = np.array(int(state_size_val), dtype=np.uint32)

        # InitialState e AdmissibleState: array 1D piatti
        initial_state_arr = np.asarray(
            dimensions["InitialState"], dtype=float
        ).reshape(-1)
        admissible_state_arr = np.asarray(
            dimensions["AdmissibleState"], dtype=float
        ).reshape(-1)

        # -----------------------------------------------------------------
        # 3. Variabili dell'SDDPBlock (Scenarios, SizeRandomDataGroups, ecc.)
        # -----------------------------------------------------------------
        sddp_var_kwargs = {
            # SizeRandomDataGroups: dimensione dei gruppi stocastici per stadio
            "SizeRandomDataGroups": Variable(
                "SizeRandomDataGroups",
                "u4",
                ("NumberRandomDataGroups",),
                size_random_data_groups_arr,
            ),
            # StateSize: scalare (dimensione dello stato per stadio)
            "StateSize": Variable(
                "StateSize",
                "u4",
                (),
                state_size_arr,
            ),
            # AdmissibleState: concatenazione degli stati ammissibili finali
            "AdmissibleState": Variable(
                "AdmissibleState",
                "double",
                ("AdmissibleStateSize",),
                admissible_state_arr,
            ),
            # InitialState: stato iniziale
            "InitialState": Variable(
                "InitialState",
                "float",
                ("InitialStateSize",),
                initial_state_arr,
            ),
        }

        # -----------------------------------------------------------------
        # 4. Crea l'SDDPBlock e lo aggiunge al master (SMSNetwork o blocco padre)
        # -----------------------------------------------------------------
        # master.add("SDDPBlock", name_id, id=..., **dims, **vars) costruisce
        # e registra il blocco con tutte le dimensioni/variabili passate.
        master.add(
            "SDDPBlock",
            name_id,
            id=f"{index_id}",
            **sddp_dim_kwargs,
            **sddp_var_kwargs,
        )
        # Recupera il riferimento al blocco appena creato
        sddp_block = master.blocks[name_id]

        # -----------------------------------------------------------------
        # 5. AbstractPath dell'SDDPBlock
        # -----------------------------------------------------------------
        # I parametri vengono da stochastic_parameters["top_path"], già validati
        # da normalize_sddp_parameters. Dipendono dalla struttura del file
        # Block_i.nc4, che è esterno e non generato da questa libreria.
        # La struttura dello stage 0 è rappresentativa di tutti gli stage,
        # perché validate_stage_structures garantisce che hydro_system_index
        # e polyhedral_index siano uniformi.
        stage_structures = self.sddp_data["stage_structures"]
        structure0 = stage_structures[0]

        if structure0["hydro_system_index"] != 0:
            raise NotImplementedError(
                f"Il path top-level attuale assume che l'HydroSystemUnitBlock "
                f"sia figlio 0 del UCBlock, ma il parser ha trovato "
                f"hydro_system_index={structure0['hydro_system_index']}. "
                "Il codice usa il valore dal parser, quindi è parametrico, ma "
                "questo caso non è mai stato testato. Va verificata la struttura "
                "dei Block_i.nc4 e, se necessario, aggiunto un test dedicato."
            )

        # K resta un parametro utente: il parser non lo può dedurre dai
        # Block_i.nc4, perché nel file di riferimento è dichiarato a
        # livello SDDPBlock, non di stage.
        K = int(self.stochastic_parameters["top_path"]["K"])

        top_ap_data = mu.build_sddp_top_abstract_path(
            n_polyhedral_per_sub_block=K,
            second_group_index=structure0["hydro_system_index"],
            third_group_index=structure0["polyhedral_index"],
        )
        self.add_sddp_top_abstract_path(sddp_block, top_ap_data)

        # -----------------------------------------------------------------
        # 6. ScenarioGenerator
        # Ha come tipo "DiscreteScenarioSet"
        # -----------------------------------------------------------------

        # Matrice Scenarios: forma (NumberScenarios, ScenarioSize).
        # Se per qualche motivo non ha 2 dimensioni, la ridefiniamo con la
        # forma attesa usando i valori delle dimensioni.
        scenarios_arr = np.asarray(scenarios_info["scenarios"], dtype=float)
        if scenarios_arr.ndim != 2:
            scenarios_arr = scenarios_arr.reshape(
                int(dimensions["NumberScenarios"]),
                int(dimensions["ScenarioSize"]),
            )

        scenario_generator = Block(
            block_type = "DiscreteScenarioSet",
            NumberScenarios = Dimension(
                "NumberScenarios", int(dimensions["NumberScenarios"]),
            ),
            ScenarioSize = Dimension(
                "ScenarioSize", int(dimensions["ScenarioSize"]),
            ),
            # Scenarios: la matrice completa (tutti gli stadi concatenati)
            Scenarios = Variable(
                "Scenarios",
                "double",
                ("NumberScenarios", "ScenarioSize"),
                scenarios_arr
            )
        )
        sddp_block.add_block("ScenarioGenerator", block=scenario_generator)

        # -----------------------------------------------------------------
        # 7. Per ogni stadio, crea StochasticBlock + BendersBlock + BendersBFunction
        # -----------------------------------------------------------------
        for stage_idx in range(time_horizon):
            # -------------------------------------------------------------
            # 6a. Blocco interno (segnaposto con id e filename)
            # -------------------------------------------------------------
            # NON costruiamo l'UCBlock qui: creiamo solo un Block vuoto con
            # attributi "id" e "filename". Il filename è il percorso del file
            # .nc4 che contiene l'UCBlock dello stadio (che verrà creato in una
            # fase successiva, o è già esistente nel workdir).
            inner_block_filename = f"Block_{stage_idx}.nc4"
            inner_block = Block().from_kwargs(
                id=str(stage_idx),
                filename=inner_block_filename,
            )

            # -------------------------------------------------------------
            # 7b. BendersBFunction
            # -------------------------------------------------------------
            # Nel notebook ha dimensioni NumNonzero, NumVar, NumRow (per ora 0)
            # e contiene un AbstractPath interno + il blocco segnaposto.
            benders_b_func = Block().from_kwargs()

            structure_t = stage_structures[stage_idx]
            bf_data = mu.build_sddp_benders_function_data(structure_t)

            # Aggiungiamo le dimensioni
            benders_b_func.add_dimension(
                "NumNonzero", int(bf_data["NumNonzero"])
            )
            benders_b_func.add_dimension(
                "NumVar", int(bf_data["NumVar"])
            )
            benders_b_func.add_dimension(
                "NumRow", int(bf_data["NumRow"])
            )

            # Aggiungiamo le variabili
            benders_b_func.add_variable(
                "Column",
                "u4",
                ("NumNonzero",),
                bf_data["Column"],
            )

            benders_b_func.add_variable(
                "A",
                "double",
                ("NumNonzero",),
                bf_data["A"],
            )

            benders_b_func.add_variable(
                "NumNonzeroAtRow",
                "u4",
                ("NumRow",),
                bf_data["NumNonzeroAtRow"],
            )

            benders_b_func.add_variable(
                "b",
                "double",
                ("NumRow",),
                bf_data["b"],
            )

            benders_b_func.add_variable(
                "ConstraintSide",
                "S1",
                ("NumRow",),
                bf_data["ConstraintSide"],
            )

            # AbstractPath interno della BendersBFunction

            benders_ap_data = mu.build_sddp_benders_abstract_path(
                hydro_unit_indices=[
                    h["index"] for h in structure_t["hydro_unit_blocks"]
                ],
            )
            self.add_benders_abstract_path(benders_b_func, benders_ap_data)

            # Aggiunge il blocco interno (segnaposto con filename)
            benders_b_func.add_block("Block", block=inner_block)

            # -------------------------------------------------------------
            # 7c. BendersBlock (contiene BendersBFunction)
            # -------------------------------------------------------------
            benders_block = Block().from_kwargs(block_type="BendersBlock")
            # nella BendersBFunction, la NumVar è il numero di colonne di A = numero di variabili di stato = n_reservoirs_total
            benders_block.add_dimension("NumVar", int(bf_data["NumVar"]))
            benders_block.add_block("BendersBFunction", block=benders_b_func)

            # -------------------------------------------------------------
            # 7d. StochasticBlock (AbstractPath + BendersBlock)
            # -------------------------------------------------------------
            stage_name = stage_names[stage_idx]

            payload = mu.build_sddp_stochastic_block(
                stochastic_parameters=self.problem_structure.get(
                    "stochastic_parameters", []
                ),
                dss_offsets=self.sddp_dss_offsets[stage_name],
                structure=structure_t,
                time_horizon=int(
                    self.sddp_data["stage_time_horizons"][stage_name]
                ),
            )

            stochastic_block = Block().from_kwargs(block_type="StochasticBlock")
            stochastic_block.add_dimension(
                "NumberDataMappings",
                int(payload["NumberDataMappings"]),
            )
            stochastic_block.add_dimension(
                "SetSizeSize",
                int(payload["SetSize"].shape[0]),
            )
            stochastic_block.add_dimension(
                "SetElementsSize",
                int(payload["SetElements"].shape[0]),
            )

            stochastic_block.add_variable(
                "DataType", "S1", ("NumberDataMappings",),
                np.asarray(payload["DataType"], dtype="S1"),
            )
            stochastic_block.add_variable(
                "FunctionName", "str", ("NumberDataMappings",),
                np.asarray(payload["FunctionName"], dtype="U200"),
            )
            stochastic_block.add_variable(
                "Caller", "S1", ("NumberDataMappings",),
                np.asarray(payload["Caller"], dtype="S1"),
            )
            stochastic_block.add_variable(
                "SetSize", "u4", ("SetSizeSize",),
                np.asarray(payload["SetSize"], dtype=np.uint32),
            )
            stochastic_block.add_variable(
                "SetElements", "u4", ("SetElementsSize",),
                np.asarray(payload["SetElements"], dtype=np.uint32),
            )

            # AbstractPath dello StochasticBlock
            sb_ap = payload["AbstractPath"]
            sb_abstract_path = Block().from_kwargs(block_type="AbstractPath")
            sb_abstract_path.add_dimension("PathDim", int(sb_ap["PathDim"]))
            sb_abstract_path.add_dimension(
                "TotalLength", int(sb_ap["TotalLength"])
            )
            sb_abstract_path.add_variable(
                "PathStart", "u4", ("PathDim",),
                np.asarray(sb_ap["PathStart"], dtype=np.uint32),
            )
            sb_abstract_path.add_variable(
                "PathNodeTypes", "c", ("TotalLength",),
                np.asarray(sb_ap["PathNodeTypes"], dtype="S1"),
            )
            sb_abstract_path.add_variable(
                "PathGroupIndices", "u4", ("TotalLength",),
                np.asarray(sb_ap["PathGroupIndices"], dtype=np.uint32),
            )
            stochastic_block.add_block("AbstractPath", block=sb_abstract_path)

            # Aggiunge il BendersBlock allo StochasticBlock
            stochastic_block.add_block("Block", block=benders_block)

            # Aggiunge lo StochasticBlock all'SDDPBlock con nome progressivo
            sddp_block.add_block(
                f"StochasticBlock_{stage_idx}", block=stochastic_block
            )

        # -----------------------------------------------------------------
        # 8. Salva l'SDDPBlock nella workdir
        # -----------------------------------------------------------------
        # Il file .nc4 deve stare nella stessa directory dei Block_i.nc4,
        # perché i filename nel SDDPBlock sono relativi ("Block_0.nc4[0]").
        # La pipeline lo salva qui automaticamente, così è disponibile per
        # ispezione (ncdump), debug, confronto col riferimento, o per
        # l'invocazione diretta di sddp_solver.
        sddp_block_path = Path(self.workdir) / f"SDDPBlock_{self.name}.nc4"
        sddp_block_path.parent.mkdir(parents=True, exist_ok=True)
        master.to_netcdf(sddp_block_path, force=True)

        # Restituisce il master (SMSNetwork), come fanno gli altri convert_*
        return master

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

            self.ucblock_variables['generator_node']['value'].append(bus)
            index += 1


    def __repr__(self):
        return (
            f"Transformation(pypsa->sms++, "
            f"stochastic={self.problem_structure.get('is_stochastic', False)}, "
            f"stochastic_type={self.problem_structure.get('stochastic_type', None)}, "
            f"has_investment_block={self.problem_structure.get('has_investment_block', None)})"
        )

    __str__ = __repr__
