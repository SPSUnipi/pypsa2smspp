from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Utilities for stochastic PyPSA networks.

This module keeps stochastic-network inspection and lightweight data extraction
outside of the main Transformation class, so the pipeline can remain readable.
"""



from typing import Any, Dict, Mapping, Optional, Sequence, List, Tuple
import warnings
import numpy as np
import pandas as pd

from .utils import (get_bus_demand_matrix,
                    get_param_as_dense,
                    )

def is_stochastic_network(n) -> bool:
    """Return True if the network exposes stochastic scenarios."""
    return bool(getattr(n, "has_scenarios", False))


def get_scenario_names(n) -> List[Any]:
    """Return scenario names in a deterministic list form."""
    if not is_stochastic_network(n):
        return []

    scenarios = getattr(n, "scenarios", [])
    try:
        return list(scenarios)
    except TypeError:
        return []


def _extract_probability_series_from_obj(obj, scenario_names: Sequence[Any]) -> pd.Series | None:
    """Try to extract scenario probabilities from a generic object."""
    if obj is None:
        return None

    # Series indexed by scenarios
    if isinstance(obj, pd.Series):
        s = obj.copy()
        s = s.reindex(scenario_names)
        if s.notna().all():
            return s.astype(float)

    # DataFrame indexed by scenarios
    if isinstance(obj, pd.DataFrame):
        candidate_columns = [
            "probability",
            "Probability",
            "probabilities",
            "Probabilities",
            "weight",
            "Weight",
            "weights",
            "Weights",
            "objective",
            "Objective",
        ]
        for col in candidate_columns:
            if col in obj.columns:
                s = obj[col].reindex(scenario_names)
                if s.notna().all():
                    return s.astype(float)

        # Single-column DataFrame fallback
        if obj.shape[1] == 1:
            s = obj.iloc[:, 0].reindex(scenario_names)
            if s.notna().all():
                return s.astype(float)

    # Dict-like
    if isinstance(obj, dict):
        try:
            s = pd.Series({k: obj[k] for k in scenario_names}, dtype=float)
            if s.notna().all():
                return s
        except Exception:
            pass

    return None


def get_scenario_probabilities(n) -> np.ndarray:
    """Return a probability vector aligned with get_scenario_names(n).

    If no explicit probabilities are found, use uniform weights.
    The vector is always normalized to sum to 1.
    """
    scenario_names = get_scenario_names(n)
    if not scenario_names:
        return np.array([], dtype=float)

    candidate_attrs = [
        "scenario_weightings",
        "scenario_probabilities",
        "scenario_probability",
        "probabilities",
        "weights",
    ]

    prob_series = None
    for attr in candidate_attrs:
        obj = getattr(n, attr, None)
        prob_series = _extract_probability_series_from_obj(obj, scenario_names)
        if prob_series is not None:
            break

    if prob_series is None:
        prob = np.full(len(scenario_names), 1.0 / len(scenario_names), dtype=float)
        return prob

    prob = prob_series.to_numpy(dtype=float)
    total = float(np.sum(prob))
    if total <= 0.0:
        prob = np.full(len(scenario_names), 1.0 / len(scenario_names), dtype=float)
    else:
        prob = prob / total

    return prob


def has_extendable_assets(n) -> bool:
    """Return True if any supported component has extendable capacity."""
    checks = [
        ("generators", "p_nom_extendable"),
        ("storage_units", "p_nom_extendable"),
        ("stores", "e_nom_extendable"),
        ("links", "p_nom_extendable"),
        ("lines", "s_nom_extendable"),
    ]

    for attr, col in checks:
        df = getattr(n, attr, None)
        if df is None or getattr(df, "empty", True):
            continue
        if col in df.columns and bool(df[col].fillna(False).astype(bool).any()):
            return True

    return False


def _normalize_stochastic_parameters(
    stochastic_parameters: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Normalize user-provided stochastic metadata.

    Expected example
    ----------------
    {
        "stochastic_type": "tssb",
        "parameters": ["demand", "price"]
    }
    """
    sp = dict(stochastic_parameters or {})

    stochastic_type = sp.get("stochastic_type", None)
    parameters = sp.get("parameters", [])

    if parameters is None:
        parameters = []
    elif isinstance(parameters, str):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    parameters = [str(p).strip().lower() for p in parameters if str(p).strip()]

    valid_parameters = {"demand", "price", "renewables"}
    invalid = sorted(set(parameters) - valid_parameters)
    if invalid:
        raise ValueError(
            f"Unsupported stochastic parameters: {invalid}. "
            f"Supported values are {sorted(valid_parameters)}."
        )

    return {
        "stochastic_type": stochastic_type,
        "parameters": parameters,
    }


def describe_problem_structure(
    n,
    *,
    capacity_expansion_ucblock: bool,
    stochastic_parameters: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Describe the high-level optimization structure of the PyPSA network."""
    is_stochastic = is_stochastic_network(n)
    scenario_names = get_scenario_names(n)

    sp = _normalize_stochastic_parameters(stochastic_parameters)

    stochastic_type = sp["stochastic_type"] if is_stochastic else None
    stochastic_parameter_set = set(sp["parameters"]) if is_stochastic else set()

    return {
        "is_stochastic": is_stochastic,
        "stochastic_type": stochastic_type,
        "number_scenarios": len(scenario_names),
        "scenario_names": scenario_names,
        "has_investment_block": not bool(capacity_expansion_ucblock),
        "stochastic_demand": "demand" in stochastic_parameter_set,
        "stochastic_marginal": "marginal" in stochastic_parameter_set,
        "stochastic_renewables": "renewables" in stochastic_parameter_set,
    }



def flatten_bus_demand_time_major(demand_by_bus: pd.DataFrame) -> np.ndarray:
    """Flatten bus demand with time-major / node-minor ordering.

    Ordering:
        (t0, n0), (t0, n1), ..., (t1, n0), (t1, n1), ...

    Parameters
    ----------
    demand_by_bus : pd.DataFrame
        Index = buses, columns = snapshots.

    Returns
    -------
    np.ndarray
        1D flattened vector.
    """
    return demand_by_bus.T.to_numpy(dtype=float).reshape(-1)

def flatten_bus_demand_node_major(demand_by_bus: pd.DataFrame) -> np.ndarray:
    """
    Flatten bus demand with node-major / time-minor ordering.

    Ordering:
        (n0, t0), (n0, t1), ..., (n1, t0), (n1, t1), ...

    Parameters
    ----------
    demand_by_bus : pd.DataFrame
        Index = buses, columns = snapshots.

    Returns
    -------
    np.ndarray
        1D flattened vector.
    """
    return demand_by_bus.to_numpy(dtype=float).reshape(-1)


def get_base_scenario_network(n):
    """Return a deterministic network for direct conversion.

    If the input network is stochastic, return the first scenario.
    Otherwise, return the original network unchanged.
    """
    has_scenarios = getattr(n, "has_scenarios", False)

    if has_scenarios:
        scenarios = list(n.scenarios)
        if not scenarios:
            raise ValueError("The network is marked as stochastic but has no scenarios.")
        return n.get_scenario(scenarios[0])

    return n

def _unique_component_names(component_index):
    """
    Return unique physical component names from a possibly scenario-expanded index.
    """
    if isinstance(component_index, pd.MultiIndex):
        if "name" in component_index.names:
            name_level = component_index.names.index("name")
        else:
            name_level = -1

        names = component_index.get_level_values(name_level)
        return pd.Index(names).drop_duplicates()

    return pd.Index(component_index).drop_duplicates()


def _drop_scenario_level_from_columns(df):
    """
    Collapse scenario-expanded columns to physical component names.

    This is useful after get_param_as_dense() on scenario networks where columns
    may still be a MultiIndex such as (scenario, name).
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    if "name" in df.columns.names:
        name_level = df.columns.names.index("name")
    else:
        name_level = -1

    out = df.copy()
    out.columns = out.columns.get_level_values(name_level)
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    return out


def flatten_generator_timeseries_generator_major(p_max_pu):
    """
    Flatten a snapshot x generator DataFrame as generator-major, time-minor.

    Input shape:
        index   = snapshots
        columns = generators

    Output:
        gen_0 all snapshots, then gen_1 all snapshots, etc.
    """
    return p_max_pu.T.to_numpy(dtype=float).reshape(-1)

# Build Discrete Scenario Set

def build_dss_demand(
    n,
) -> Dict[str, Any]:
    
    """Build DSS data for stochastic demand.
    Returns
    -------
    scenarios : np.ndarray
        Shape = (NumberScenarios, ScenarioSize)
    pool_weights : np.ndarray
        Shape = (NumberScenarios,)
    bus_order : list
        Bus ordering used in the scenario flattening.
    snapshot_order : list
        Snapshot ordering used in the scenario flattening.
    flattening : str
        Human-readable description of the flattening convention.
    """
    
    scenario_names = get_scenario_names(n)
    if not scenario_names:
        raise ValueError("DSS demand extraction requested on a deterministic network.")

    scenarios = []
    bus_order = None
    snapshot_order = None

    for scenario_name in scenario_names:
        n_s = n.get_scenario(scenario_name)
        demand_by_bus = get_bus_demand_matrix(n_s)

        if bus_order is None:
            bus_order = list(demand_by_bus.index)
            snapshot_order = list(demand_by_bus.columns)
        else:
            demand_by_bus = demand_by_bus.reindex(
                index=bus_order,
                columns=snapshot_order,
                fill_value=0.0,
            )

        scenarios.append(flatten_bus_demand_node_major(demand_by_bus))

    scenario_matrix = np.vstack(scenarios).astype(float)
    pool_weights = get_scenario_probabilities(n).astype(float)
    flattening = "node_major_time_minor"

    return {
        "parameter": "demand",
        "scenarios": scenario_matrix,
        "pool_weights": pool_weights,
        "node_order": bus_order,
        "snapshot_order": snapshot_order,
        "flattening": flattening,
        "scenario_size": int(scenario_matrix.shape[1]),
        "number_scenarios": int(scenario_matrix.shape[0]),
    }


def build_dss_marginal(n) -> Dict[str, Any] | None:
    """Placeholder builder for stochastic marginal costs."""
    warnings.warn(
        "build_dss_marginal was called, but stochastic marginal costs are not implemented yet.",
        UserWarning,
    )
    return None


def build_dss_renewables(
    n,
    intermittent_carriers,
) -> Dict[str, Any]:
    """
    Build DSS data for stochastic renewable maximum power profiles.

    Source:
        Generator.p_max_pu

    Flattening:
        generator-major, time-minor

    For each scenario:
        gen_0[t0], ..., gen_0[tT],
        gen_1[t0], ..., gen_1[tT],
        ...
    """
    scenario_names = get_scenario_names(n)
    if not scenario_names:
        raise ValueError("DSS renewable extraction requested on a deterministic network.")

    if intermittent_carriers is None:
        raise ValueError("intermittent_carriers must be provided for stochastic renewables.")

    intermittent_carriers = set(intermittent_carriers)

    physical_generator_names = _unique_component_names(n.generators.index)

    generators_static = n.generators.copy()
    if isinstance(generators_static.index, pd.MultiIndex):
        if "name" in generators_static.index.names:
            name_level = generators_static.index.names.index("name")
        else:
            name_level = -1

        generators_static = generators_static.copy()
        generators_static.index = generators_static.index.get_level_values(name_level)
        generators_static = generators_static.loc[
            ~generators_static.index.duplicated(keep="first")
        ]

    generators_static = generators_static.reindex(physical_generator_names)

    renewable_generators = generators_static.index[
        generators_static["carrier"].isin(intermittent_carriers)
    ].tolist()

    if not renewable_generators:
        raise ValueError(
            "stochastic_renewables=True, but no generators with carriers in "
            f"intermittent_carriers={sorted(intermittent_carriers)} were found."
        )

    scenarios = []
    generator_order = None
    snapshot_order = None

    for scenario_name in scenario_names:
        n_s = n.get_scenario(scenario_name)

        p_max_pu = get_param_as_dense(
            n_s,
            component="Generator",
            field="p_max_pu",
            weights=False,
        )

        p_max_pu = _drop_scenario_level_from_columns(p_max_pu)
        p_max_pu = p_max_pu.reindex(columns=renewable_generators)

        if generator_order is None:
            generator_order = list(p_max_pu.columns)
            snapshot_order = list(p_max_pu.index)
        else:
            p_max_pu = p_max_pu.reindex(
                index=snapshot_order,
                columns=generator_order,
            )

        if p_max_pu.isna().any().any():
            missing = p_max_pu.columns[p_max_pu.isna().any(axis=0)].tolist()
            raise ValueError(
                "Missing p_max_pu values after reindexing renewable generators. "
                f"Missing/invalid generators: {missing}"
            )

        scenarios.append(flatten_generator_timeseries_generator_major(p_max_pu))

    scenario_matrix = np.vstack(scenarios).astype(float)
    pool_weights = get_scenario_probabilities(n).astype(float)

    return {
        "parameter": "renewables",
        "variable": "MaxPower",
        "source": "Generator.p_max_pu",
        "scenarios": scenario_matrix,
        "pool_weights": pool_weights,
        "generator_order": generator_order,
        "snapshot_order": snapshot_order,
        "flattening": "generator_major_time_minor",
        "scenario_size": int(scenario_matrix.shape[1]),
        "number_scenarios": int(scenario_matrix.shape[0]),
    }


def merge_tssb_dss_parts(dss_parts):
    """
    Merge DSS parts by concatenating their scenario vectors horizontally.

    Each part must have:
    - scenarios: shape = (NumberScenarios, part_scenario_size)
    - pool_weights: shape = (NumberScenarios,)
    - scenario_size
    - number_scenarios

    The returned parts include offset_start and offset_end, used by the
    StochasticBlock data mappings.
    """
    dss_parts = [part for part in dss_parts if part is not None]

    if not dss_parts:
        raise ValueError("No DSS parts were provided.")

    number_scenarios = int(dss_parts[0]["number_scenarios"])
    pool_weights = np.asarray(dss_parts[0]["pool_weights"], dtype=float)

    checked_parts = []
    scenario_arrays = []

    offset = 0

    for part in dss_parts:
        part_number_scenarios = int(part["number_scenarios"])
        if part_number_scenarios != number_scenarios:
            raise ValueError(
                "All DSS parts must have the same NumberScenarios. "
                f"Expected {number_scenarios}, got {part_number_scenarios} "
                f"for parameter {part.get('parameter')!r}."
            )

        part_pool_weights = np.asarray(part["pool_weights"], dtype=float)
        if not np.allclose(part_pool_weights, pool_weights):
            raise ValueError(
                "All DSS parts must have the same PoolWeights. "
                f"Mismatch found for parameter {part.get('parameter')!r}."
            )

        scenarios = np.asarray(part["scenarios"], dtype=float)
        scenario_size = int(scenarios.shape[1])

        part = dict(part)
        part["scenario_size"] = scenario_size
        part["offset_start"] = int(offset)
        part["offset_end"] = int(offset + scenario_size)

        checked_parts.append(part)
        scenario_arrays.append(scenarios)

        offset += scenario_size

    scenario_matrix = np.hstack(scenario_arrays)

    return {
        "scenarios": scenario_matrix,
        "pool_weights": pool_weights,
        "parts": checked_parts,
        "scenario_size": int(scenario_matrix.shape[1]),
        "number_scenarios": int(number_scenarios),
    }


# Build Abstract Path

def calculate_design_variables(
    investment_meta,
    unitblock_design_data,
    network_block_index,
):
    """
    Build design-variable descriptors for the TSSB StaticAbstractPath.
    """
    design_variables = []

    for item in unitblock_design_data:
        design_variables.append(item)

    design_lines = list(investment_meta.get("design_lines", []))
    if design_lines:
        for line_idx in design_lines:
            design_variables.append(
                {
                    "block_index": int(network_block_index),
                    "var_name": "x_network",
                    "component_type": "network",
                    "element_index": int(line_idx),
                    "range_index": int(line_idx) + 1,
                }
            )

    return design_variables


def build_tssb_static_abstract_path(design_variables):
    """
    Build a preliminary StaticAbstractPath representation from design variables.
    """
    path_dim = len(design_variables)
    total_length = 2 * path_dim

    path_group_indices = []
    path_node_types = []
    path_element_indices = []
    path_range_indices = []
    path_start = []

    current = 0
    for dv in design_variables:
        path_start.append(current)

        # Block node
        path_group_indices.append(str(dv["block_index"]))
        path_node_types.append("B")
        path_element_indices.append(0)
        path_range_indices.append(0)

        # Variable node
        path_group_indices.append(dv["var_name"])
        path_node_types.append("V")
        path_element_indices.append(int(dv["element_index"]))
        path_range_indices.append(int(dv["range_index"]))

        current += 2

    path_node_types = np.array(path_node_types, dtype="object")
    path_group_indices = np.array(path_group_indices, dtype="object")

    path_element_indices = np.ma.masked_array(
        np.array(path_element_indices, dtype=np.uint32),
        mask=(path_node_types == "B"),
    )
    path_range_indices = np.ma.masked_array(
        np.array(path_range_indices, dtype=np.uint32),
        mask=(path_node_types == "B"),
    )

    return {
        "PathDim": path_dim,
        "TotalLength": total_length,
        "PathGroupIndices": path_group_indices,
        "PathNodeTypes": path_node_types,
        "PathElementIndices": path_element_indices,
        "PathRangeIndices": path_range_indices,
        "PathStart": np.array(path_start, dtype=np.uint32),
    }


# Build stochastic block

def build_stochastic_mapping_demand(
    set_from_start,
    set_from_end,
    scenario_size,
):
    """
    Build the stochastic data mapping for demand.

    The scenario slice [set_from_start, set_from_end) is passed to
    UCBlock::set_active_power_demand through a Range/Range mapping.
    """
    return {
        "target": "demand",
        "function_name": "UCBlock::set_active_power_demand",
        "caller": "B",
        "data_type": "D",
        "set_size": [0, 0],
        "set_elements": [
            int(set_from_start),
            int(set_from_end),
            0,
            int(scenario_size),
        ],
        "abstract_paths": [
            {
                "node_types": "",
                "group_indices": [],
                "element_indices": [],
                "range_indices": [],
            }
        ],
    }

def build_stochastic_mapping_renewables_maxpower(
    set_from_start,
    set_from_end,
    scenario_size,
    unitblock_indices,
):
    """
    Build the stochastic data mapping for renewable MaxPower.

    A single data mapping is used for all intermittent unit blocks. The target
    blocks are specified by concatenating one AbstractPath per UnitBlock_i:

        B("i")

    where i is the nested UnitBlock index inside the UCBlock.
    """
    abstract_paths = []

    for unitblock_index in unitblock_indices:
        abstract_paths.append(
            {
                "node_types": "B",
                "group_indices": [str(int(unitblock_index))],
                "element_indices": [None],
                "range_indices": [None],
            }
        )

    return {
        "target": "renewables",
        "function_name": "IntermittentUnitBlock::set_maximum_power",
        "caller": "B",
        "data_type": "D",
        "set_size": [0, 0],
        "set_elements": [
            int(set_from_start),
            int(set_from_end),
            0,
            int(scenario_size),
        ],
        "abstract_paths": abstract_paths,
    }


def build_tssb_stochastic_block_data(data_mappings):
    """
    Build the StochasticBlock payload from a list of data mappings.
    """
    if not data_mappings:
        raise ValueError("At least one stochastic data mapping is required.")

    function_names = []
    callers = []
    data_types = []
    set_size = []
    set_elements = []

    for mapping in data_mappings:
        function_names.append(mapping["function_name"])
        callers.append(mapping["caller"])
        data_types.append(mapping["data_type"])
        set_size.extend(mapping["set_size"])
        set_elements.extend(mapping["set_elements"])

    abstract_path = build_tssb_stochastic_block_abstract_path(data_mappings)

    stochastic_block = {
        "NumberDataMappings": len(data_mappings),
        "FunctionName": np.array(function_names, dtype="object"),
        "Caller": np.array(callers, dtype="object"),
        "DataType": np.array(data_types, dtype="object"),
        "SetSize": np.array(set_size, dtype=np.uint32),
        "SetElements": np.array(set_elements, dtype=np.uint32),
        "AbstractPath": abstract_path,
    }

    return stochastic_block


def build_tssb_stochastic_block_abstract_path(data_mappings):
    """
    Build the AbstractPath for the StochasticBlock mappings.

    The AbstractPath is built by concatenating all paths declared by each
    mapping. Demand normally contributes one empty path. Renewable MaxPower
    contributes one B("UnitBlock_i") path per intermittent UnitBlock.
    """
    path_start = []
    node_types = []
    group_indices = []
    element_indices = []
    range_indices = []

    cursor = 0

    for mapping in data_mappings:
        mapping_paths = mapping.get("abstract_paths", None)

        if mapping_paths is None:
            mapping_paths = [
                {
                    "node_types": "",
                    "group_indices": [],
                    "element_indices": [],
                    "range_indices": [],
                }
            ]

        for path in mapping_paths:
            path_node_types = path.get("node_types", "")
            path_group_indices = path.get("group_indices", [])
            path_element_indices = path.get("element_indices", [])
            path_range_indices = path.get("range_indices", [])

            path_length = len(path_node_types)

            if len(path_group_indices) != path_length:
                raise ValueError(
                    "Invalid AbstractPath: group_indices length does not match "
                    f"node_types length for mapping {mapping['target']!r}."
                )

            if len(path_element_indices) != path_length:
                raise ValueError(
                    "Invalid AbstractPath: element_indices length does not match "
                    f"node_types length for mapping {mapping['target']!r}."
                )

            if len(path_range_indices) != path_length:
                raise ValueError(
                    "Invalid AbstractPath: range_indices length does not match "
                    f"node_types length for mapping {mapping['target']!r}."
                )

            path_start.append(cursor)

            for k in range(path_length):
                node_types.append(path_node_types[k])
                group_indices.append(path_group_indices[k])
                element_indices.append(path_element_indices[k])
                range_indices.append(path_range_indices[k])

            cursor += path_length

    path_dim = len(path_start)
    total_length = len(node_types)

    return {
        "PathDim": int(path_dim),
        "TotalLength": int(total_length),
        "PathStart": np.array(path_start, dtype=np.uint32),
        "PathNodeTypes": np.ma.masked_array(
            np.array(node_types, dtype="S1"),
            mask=np.zeros(total_length, dtype=bool),
        ),
        "PathGroupIndices": np.ma.masked_array(
            np.array(group_indices, dtype=object),
            mask=np.zeros(total_length, dtype=bool),
        ),
        "PathElementIndices": np.ma.masked_array(
            np.array(
                [
                    0 if value is None else int(value)
                    for value in element_indices
                ],
                dtype=np.uint32,
            ),
            mask=np.array(
                [value is None for value in element_indices],
                dtype=bool,
            ),
        ),
        "PathRangeIndices": np.ma.masked_array(
            np.array(
                [
                    0 if value is None else int(value)
                    for value in range_indices
                ],
                dtype=np.uint32,
            ),
            mask=np.array(
                [value is None for value in range_indices],
                dtype=bool,
            ),
        ),
    }