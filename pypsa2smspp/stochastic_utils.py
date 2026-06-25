from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Utilities for stochastic PyPSA networks.

This module keeps stochastic-network inspection and lightweight data extraction
outside of the main Transformation class, so the pipeline can remain readable.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import inspect

from pypsa2smspp.constants import STOCHASTIC_PARAMETER_REGISTRY
from pypsa2smspp.utils import (
    get_bus_demand_matrix,
    get_param_as_dense,
)


# =============================================================================
# Stochastic network inspection
# =============================================================================

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


def get_base_scenario_network(n):
    """
    Return a deterministic network for direct conversion.

    If the input network is stochastic, return the first scenario.
    Otherwise, return the original network unchanged.
    """
    has_scenarios = getattr(n, "has_scenarios", False)

    if has_scenarios:
        scenarios = list(n.scenarios)
        if not scenarios:
            raise ValueError(
                "The network is marked as stochastic but has no scenarios."
            )
        return n.get_scenario(scenarios[0])

    return n


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


# =============================================================================
# Scenario probabilities
# =============================================================================

def _extract_probability_series_from_obj(
    obj,
    scenario_names: Sequence[Any],
) -> pd.Series | None:
    """Try to extract scenario probabilities from a generic object."""
    if obj is None:
        return None

    # Series indexed by scenarios.
    if isinstance(obj, pd.Series):
        s = obj.copy()
        s = s.reindex(scenario_names)

        if s.notna().all():
            return s.astype(float)

    # DataFrame indexed by scenarios.
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

        # Single-column DataFrame fallback.
        if obj.shape[1] == 1:
            s = obj.iloc[:, 0].reindex(scenario_names)

            if s.notna().all():
                return s.astype(float)

    # Dict-like fallback.
    if isinstance(obj, dict):
        try:
            s = pd.Series({k: obj[k] for k in scenario_names}, dtype=float)

            if s.notna().all():
                return s
        except Exception:
            pass

    return None


def get_scenario_probabilities(n) -> np.ndarray:
    """
    Return a probability vector aligned with get_scenario_names(n).

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
        return np.full(
            len(scenario_names),
            1.0 / len(scenario_names),
            dtype=float,
        )

    prob = prob_series.to_numpy(dtype=float)
    total = float(np.sum(prob))

    if total <= 0.0:
        prob = np.full(len(scenario_names), 1.0 / len(scenario_names), dtype=float)
    else:
        prob = prob / total

    return prob


# =============================================================================
# Stochastic parameter normalization and problem structure
# =============================================================================

def _normalize_stochastic_parameters(
    stochastic_parameters: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Normalize user-provided stochastic metadata.

    Expected example
    ----------------
    {
        "stochastic_type": "tssb",
        "parameters": ["demand", "renewable_maxpower"]
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

    valid_parameters = set(STOCHASTIC_PARAMETER_REGISTRY)
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
    stochastic_parameters_list = list(sp["parameters"]) if is_stochastic else []
    stochastic_parameter_set = set(stochastic_parameters_list)

    return {
        "is_stochastic": is_stochastic,
        "stochastic_type": stochastic_type,
        "number_scenarios": len(scenario_names),
        "scenario_names": scenario_names,
        "has_investment_block": not bool(capacity_expansion_ucblock),
        "stochastic_parameters": stochastic_parameters_list,
        "stochastic_parameter_set": stochastic_parameter_set,
    }


# =============================================================================
# Physical asset helpers
# =============================================================================

def _unique_component_names(component_index) -> pd.Index:
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


def _as_physical_component_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a static component frame indexed by physical component names.

    Stochastic PyPSA objects may carry a scenario-expanded MultiIndex. This
    helper drops the scenario level and keeps one row per physical asset.
    """
    out = df.copy()

    if isinstance(out.index, pd.MultiIndex):
        if "name" in out.index.names:
            name_level = out.index.names.index("name")
        else:
            name_level = -1

        out.index = out.index.get_level_values(name_level)
        out = out.loc[~out.index.duplicated(keep="first")]

    return out


def _drop_scenario_level_from_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def get_physical_generators_by_carriers(n, carriers) -> List[Any]:
    """
    Return physical generator names whose carrier belongs to `carriers`.

    The returned order follows the physical generator order in n.generators.
    """
    if carriers is None:
        raise ValueError("carriers must be provided.")

    carriers = set(carriers)
    physical_generator_names = _unique_component_names(n.generators.index)

    generators_static = _as_physical_component_frame(n.generators)
    generators_static = generators_static.reindex(physical_generator_names)

    generator_names = generators_static.index[
        generators_static["carrier"].isin(carriers)
    ].tolist()

    if not generator_names:
        raise ValueError(
            "No PyPSA generators with carriers in "
            f"carriers={sorted(carriers)} were found."
        )

    return generator_names


def get_physical_generators_excluding_carriers(n, carriers) -> List[Any]:
    """
    Return physical generator names whose carrier does not belong to `carriers`.

    This is used for ThermalUnitBlock parameters when thermal units are enabled.
    """
    if carriers is None:
        raise ValueError("carriers must be provided.")

    carriers = set(carriers)
    physical_generator_names = _unique_component_names(n.generators.index)

    generators_static = _as_physical_component_frame(n.generators)
    generators_static = generators_static.reindex(physical_generator_names)

    generator_names = generators_static.index[
        ~generators_static["carrier"].isin(carriers)
    ].tolist()

    if not generator_names:
        raise ValueError(
            "No PyPSA thermal generators were found after excluding "
            f"intermittent carriers={sorted(carriers)}."
        )

    return generator_names


def get_physical_storage_units(n) -> List[Any]:
    """
    Return physical StorageUnit names.

    In the current transformation, StorageUnit assets are mapped to
    HydroUnitBlock.
    """
    if getattr(n, "storage_units", None) is None or n.storage_units.empty:
        raise ValueError("No PyPSA StorageUnit assets were found.")

    return _unique_component_names(n.storage_units.index).tolist()


# =============================================================================
# Flattening helpers
# =============================================================================

def flatten_bus_demand_time_major(demand_by_bus: pd.DataFrame) -> np.ndarray:
    """
    Flatten bus demand with time-major / node-minor ordering.

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


def flatten_generator_timeseries_generator_major(values: pd.DataFrame) -> np.ndarray:
    """
    Flatten a snapshot x generator DataFrame as generator-major, time-minor.

    Input shape:
        index   = snapshots
        columns = generators

    Output:
        gen_0 all snapshots, then gen_1 all snapshots, etc.
    """
    return values.T.to_numpy(dtype=float).reshape(-1)


def flatten_asset_timeseries_asset_major(values: pd.DataFrame) -> np.ndarray:
    """
    Flatten a snapshot x asset DataFrame as asset-major, time-minor.

    Input shape:
        index   = snapshots
        columns = assets

    Output:
        asset_0 all snapshots, then asset_1 all snapshots, etc.
    """
    return values.T.to_numpy(dtype=float).reshape(-1)


# =============================================================================
# Generic stochastic asset selection
# =============================================================================

def get_effective_intermittent_carriers(
    intermittent_carriers,
    default_carriers,
) -> List[Any]:
    """
    Return the carrier set used to identify intermittent generators.

    If intermittent_carriers is None, fall back to default_carriers.
    """
    if intermittent_carriers is None:
        carriers = default_carriers
    elif isinstance(intermittent_carriers, str):
        carriers = [intermittent_carriers]
    else:
        carriers = list(intermittent_carriers)

    return carriers


def get_stochastic_parameter_asset_names(
    n,
    *,
    parameter: str,
    spec: Mapping[str, Any],
    intermittent_carriers,
    default_intermittent_carriers,
    enable_thermal_units: bool,
) -> List[Any]:
    """
    Return the physical PyPSA asset names affected by a stochastic parameter.

    Parameters
    ----------
    n : pypsa.Network
        Possibly stochastic PyPSA network.
    parameter : str
        Stochastic parameter name, e.g. "renewable_maxpower".
    spec : mapping
        Registry entry for the stochastic parameter.
    intermittent_carriers : sequence | str | None
        User-provided intermittent carriers.
    default_intermittent_carriers : sequence
        Fallback carriers used when intermittent_carriers is None.
    enable_thermal_units : bool
        Whether thermal generators are represented as ThermalUnitBlock.
    """
    asset_filter = spec.get("asset_filter", None)

    if asset_filter == "intermittent_generators":
        intermittent = get_effective_intermittent_carriers(
            intermittent_carriers=intermittent_carriers,
            default_carriers=default_intermittent_carriers,
        )

        return get_physical_generators_by_carriers(
            n=n,
            carriers=intermittent,
        )

    if asset_filter == "thermal_generators":
        if not enable_thermal_units:
            raise ValueError(
                f"Stochastic parameter {parameter!r} requires "
                "enable_thermal_units=True."
            )

        intermittent = get_effective_intermittent_carriers(
            intermittent_carriers=intermittent_carriers,
            default_carriers=default_intermittent_carriers,
        )

        return get_physical_generators_excluding_carriers(
            n=n,
            carriers=intermittent,
        )

    if asset_filter == "storage_units":
        return get_physical_storage_units(n)

    raise ValueError(
        f"Unsupported asset_filter={asset_filter!r} for stochastic "
        f"parameter {parameter!r}."
    )

# =============================================================================
# SMS++ parameter evaluation
# =============================================================================

def _get_unitblock_parameter_map(transformation_config, unitblock_type: str):
    """
    Return the TransformationConfig parameter dictionary for a UnitBlock type.
    """
    attr_name = f"{unitblock_type}_parameters"

    if not hasattr(transformation_config, attr_name):
        raise AttributeError(
            f"TransformationConfig does not define {attr_name!r}."
        )

    return getattr(transformation_config, attr_name)


def _get_parameter_transform(
    transformation_config,
    *,
    unitblock_type: str,
    smspp_parameter: str,
):
    """
    Return the transformation rule for one SMS++ UnitBlock parameter.
    """
    parameter_map = _get_unitblock_parameter_map(
        transformation_config=transformation_config,
        unitblock_type=unitblock_type,
    )

    if smspp_parameter not in parameter_map:
        raise KeyError(
            f"Parameter {smspp_parameter!r} was not found in "
            f"{unitblock_type}_parameters."
        )

    return parameter_map[smspp_parameter]


def _get_transform_argument_names(transform) -> List[str]:
    """
    Return argument names required by a callable transformation rule.

    Constant rules have no arguments.
    """
    if not callable(transform):
        return []

    return list(inspect.signature(transform).parameters)


def _to_snapshot_asset_frame(
    values,
    *,
    snapshot_order: Sequence[Any],
    asset_order: Sequence[Any],
    parameter: str,
) -> pd.DataFrame:
    """
    Convert a transformation output to a snapshot x asset DataFrame.

    The direct transformation sometimes returns DataFrames, sometimes Series,
    scalars, or numpy arrays. This helper normalizes the result for DSS
    flattening.
    """
    snapshot_order = list(snapshot_order)
    asset_order = list(asset_order)

    if isinstance(values, pd.DataFrame):
        out = values.copy()
        out = _drop_scenario_level_from_columns(out)
        return out.reindex(index=snapshot_order, columns=asset_order)

    if isinstance(values, pd.Series):
        if values.index.equals(pd.Index(asset_order)):
            data = np.tile(values.reindex(asset_order).to_numpy(dtype=float), (len(snapshot_order), 1))
            return pd.DataFrame(data, index=snapshot_order, columns=asset_order)

        if values.index.equals(pd.Index(snapshot_order)) and len(asset_order) == 1:
            return pd.DataFrame(
                values.reindex(snapshot_order).to_numpy(dtype=float).reshape(-1, 1),
                index=snapshot_order,
                columns=asset_order,
            )

        raise ValueError(
            f"Cannot convert Series output for stochastic parameter "
            f"{parameter!r} to a snapshot x asset frame."
        )

    arr = np.asarray(values)

    if arr.ndim == 0:
        return pd.DataFrame(
            float(arr),
            index=snapshot_order,
            columns=asset_order,
        )

    if arr.ndim == 1:
        if arr.shape[0] == len(snapshot_order) and len(asset_order) == 1:
            return pd.DataFrame(
                arr.reshape(-1, 1),
                index=snapshot_order,
                columns=asset_order,
            )

        if arr.shape[0] == len(asset_order):
            data = np.tile(arr.reshape(1, -1), (len(snapshot_order), 1))
            return pd.DataFrame(data, index=snapshot_order, columns=asset_order)

        raise ValueError(
            f"Cannot convert 1D output with shape {arr.shape} for stochastic "
            f"parameter {parameter!r}. Expected length {len(snapshot_order)} "
            f"or {len(asset_order)}."
        )

    if arr.ndim == 2:
        if arr.shape == (len(snapshot_order), len(asset_order)):
            return pd.DataFrame(arr, index=snapshot_order, columns=asset_order)

        if arr.shape == (len(asset_order), len(snapshot_order)):
            return pd.DataFrame(arr.T, index=snapshot_order, columns=asset_order)

        raise ValueError(
            f"Cannot convert 2D output with shape {arr.shape} for stochastic "
            f"parameter {parameter!r}. Expected "
            f"{(len(snapshot_order), len(asset_order))} or "
            f"{(len(asset_order), len(snapshot_order))}."
        )

    raise ValueError(
        f"Cannot convert output with ndim={arr.ndim} for stochastic "
        f"parameter {parameter!r}."
    )


def evaluate_unitblock_parameter_timeseries(
    n_s,
    *,
    parameter: str,
    pypsa_component: str,
    field: str,
    asset_names: Sequence[Any],
    transformation_config,
    unitblock_type: str,
    smspp_parameter: str | None,
    weights: bool = False,
) -> pd.DataFrame:
    """
    Evaluate the actual SMS++ UnitBlock parameter for one scenario.

    If smspp_parameter is provided, the corresponding TransformationConfig rule
    is used. Otherwise, the raw PyPSA field is returned.

    The returned object is always a snapshot x asset DataFrame.
    """
    raw_reference = get_param_as_dense(
        n_s,
        component=pypsa_component,
        field=field,
        weights=weights,
    )

    raw_reference = _drop_scenario_level_from_columns(raw_reference)
    raw_reference = raw_reference.reindex(columns=list(asset_names))

    snapshot_order = list(raw_reference.index)
    asset_order = list(raw_reference.columns)

    if smspp_parameter is None:
        return raw_reference

    transform = _get_parameter_transform(
        transformation_config=transformation_config,
        unitblock_type=unitblock_type,
        smspp_parameter=smspp_parameter,
    )

    argument_names = _get_transform_argument_names(transform)

    if not argument_names:
        evaluated = transform
    else:
        kwargs = {}

        for arg_name in argument_names:
            arg_values = get_param_as_dense(
                n_s,
                component=pypsa_component,
                field=arg_name,
                weights=weights,
            )

            arg_values = _drop_scenario_level_from_columns(arg_values)
            arg_values = arg_values.reindex(
                index=snapshot_order,
                columns=asset_order,
            )

            if arg_values.isna().any().any():
                missing = arg_values.columns[arg_values.isna().any(axis=0)].tolist()
                raise ValueError(
                    f"Missing dependency {pypsa_component}.{arg_name} while "
                    f"evaluating stochastic parameter {parameter!r} as "
                    f"{unitblock_type}.{smspp_parameter}. "
                    f"Missing/invalid assets: {missing}"
                )

            kwargs[arg_name] = arg_values

        evaluated = transform(**kwargs)

    evaluated = _to_snapshot_asset_frame(
        evaluated,
        snapshot_order=snapshot_order,
        asset_order=asset_order,
        parameter=parameter,
    )

    if evaluated.isna().any().any():
        missing = evaluated.columns[evaluated.isna().any(axis=0)].tolist()
        raise ValueError(
            f"NaN values found after evaluating stochastic parameter "
            f"{parameter!r} as {unitblock_type}.{smspp_parameter}. "
            f"Missing/invalid assets: {missing}"
        )

    return evaluated


# =============================================================================
# Discrete Scenario Set
# =============================================================================

def build_dss_demand(n) -> Dict[str, Any]:
    """
    Build DSS data for stochastic demand.

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

    return {
        "parameter": "demand",
        "scenarios": scenario_matrix,
        "pool_weights": pool_weights,
        "node_order": bus_order,
        "snapshot_order": snapshot_order,
        "flattening": "node_major_time_minor",
        "scenario_size": int(scenario_matrix.shape[1]),
        "number_scenarios": int(scenario_matrix.shape[0]),
    }


def build_dss_unitblock_timeseries_parameter(
    n,
    *,
    parameter: str,
    pypsa_component: str,
    field: str,
    asset_names: Sequence[Any],
    function_name: str,
    unitblock_type: str,
    target: str,
    transformation_config=None,
    smspp_parameter: str | None = None,
    weights: bool = False,
) -> Dict[str, Any]:
    """
    Build DSS data for a dense UnitBlock time series parameter.

    The parameter is extracted scenario by scenario and flattened with
    asset-major, time-minor ordering:

        asset_0[t0], ..., asset_0[tT],
        asset_1[t0], ..., asset_1[tT],
        ...

    If smspp_parameter is provided, the corresponding TransformationConfig
    rule is applied. This ensures that the DSS contains the actual SMS++
    parameter passed to the setter, not necessarily the raw PyPSA field.
    """
    scenario_names = get_scenario_names(n)

    if not scenario_names:
        raise ValueError(
            f"DSS extraction for {parameter!r} requested on a deterministic network."
        )

    asset_names = list(asset_names)

    if not asset_names:
        raise ValueError(
            f"No assets were provided for stochastic parameter {parameter!r}."
        )

    if smspp_parameter is not None and transformation_config is None:
        raise ValueError(
            f"transformation_config must be provided when smspp_parameter is set "
            f"for stochastic parameter {parameter!r}."
        )

    scenarios = []
    asset_order = None
    snapshot_order = None

    for scenario_name in scenario_names:
        n_s = n.get_scenario(scenario_name)

        values = evaluate_unitblock_parameter_timeseries(
            n_s,
            parameter=parameter,
            pypsa_component=pypsa_component,
            field=field,
            asset_names=asset_names,
            transformation_config=transformation_config,
            unitblock_type=unitblock_type,
            smspp_parameter=smspp_parameter,
            weights=weights,
        )

        if asset_order is None:
            asset_order = list(values.columns)
            snapshot_order = list(values.index)
        else:
            values = values.reindex(
                index=snapshot_order,
                columns=asset_order,
            )

        if values.isna().any().any():
            missing = values.columns[values.isna().any(axis=0)].tolist()
            raise ValueError(
                f"Missing values for stochastic parameter {parameter!r} "
                f"after evaluating {pypsa_component}.{field}. "
                f"Missing/invalid assets: {missing}"
            )

        scenarios.append(flatten_asset_timeseries_asset_major(values))

    scenario_matrix = np.vstack(scenarios).astype(float)
    pool_weights = get_scenario_probabilities(n).astype(float)

    return {
        "parameter": parameter,
        "target": target,
        "function_name": function_name,
        "unitblock_type": unitblock_type,
        "smspp_parameter": smspp_parameter,
        "pypsa_component": pypsa_component,
        "field": field,
        "source": f"{pypsa_component}.{field}",
        "scenarios": scenario_matrix,
        "pool_weights": pool_weights,
        "asset_order": asset_order,
        "snapshot_order": snapshot_order,
        "flattening": "asset_major_time_minor",
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


# =============================================================================
# Static Abstract Path
# =============================================================================

def calculate_design_variables(
    investment_meta,
    unitblock_design_data,
    network_block_index,
):
    """Build design-variable descriptors for the TSSB StaticAbstractPath."""
    design_variables = []

    for item in unitblock_design_data:
        design_variables.append(item)

    design_lines = list(investment_meta.get("design_lines", []))

    if design_lines:
        # x_network is indexed by design-line position (0..NumberDesignLines-1),
        # not by the network line id: v_design[ p ] holds the design variable of
        # the p-th designable line (whose line id is design_lines[ p ]).
        for position, line_idx in enumerate(design_lines):
            design_variables.append(
                {
                    "block_index": int(network_block_index),
                    "var_name": "x_network",
                    "component_type": "network",
                    "element_index": int(position),
                    "range_index": int(position) + 1,
                }
            )

    return design_variables


def build_tssb_static_abstract_path(design_variables):
    """Build a preliminary StaticAbstractPath representation from design variables."""
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

        # Block node.
        path_group_indices.append(str(dv["block_index"]))
        path_node_types.append("B")
        path_element_indices.append(0)
        path_range_indices.append(0)

        # Variable node.
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


# =============================================================================
# StochasticBlock mappings
# =============================================================================

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


def build_stochastic_mapping_single_unit(
    *,
    target: str,
    function_name: str,
    set_from_start,
    set_from_end,
    time_horizon,
    unitblock_index,
):
    """
    Build one stochastic data mapping for one UnitBlock time series.

    The mapping applies one contiguous slice of the DSS scenario vector to one
    setter call on one nested UnitBlock_i.

    Abstract path:
        B("unitblock_index")
    """
    return {
        "target": target,
        "function_name": function_name,
        "caller": "B",
        "data_type": "D",
        "set_size": [0, 0],
        "set_elements": [
            int(set_from_start),
            int(set_from_end),
            0,
            int(time_horizon),
        ],
        "abstract_paths": [
            {
                "node_types": "B",
                "group_indices": [str(int(unitblock_index))],
                "element_indices": [None],
                "range_indices": [None],
            }
        ],
    }


def build_tssb_stochastic_block_data(data_mappings):
    """Build the StochasticBlock payload from a list of data mappings."""
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

    return {
        "NumberDataMappings": len(data_mappings),
        "FunctionName": np.array(function_names, dtype="object"),
        "Caller": np.array(callers, dtype="object"),
        "DataType": np.array(data_types, dtype="object"),
        "SetSize": np.array(set_size, dtype=np.uint32),
        "SetElements": np.array(set_elements, dtype=np.uint32),
        "AbstractPath": abstract_path,
    }


def build_tssb_stochastic_block_abstract_path(data_mappings):
    """
    Build the AbstractPath for the StochasticBlock mappings.

    The AbstractPath is built by concatenating all paths declared by each
    mapping. Demand normally contributes one empty path. UnitBlock-level
    stochastic parameters contribute one B("UnitBlock_i") path per affected
    UnitBlock.
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


# =============================================================================
# UnitBlock lookup helpers
# =============================================================================

def collect_unitblock_indices_by_names_and_type(
    unitblocks: Mapping[str, Mapping[str, Any]],
    *,
    names: Sequence[Any],
    block_type: str,
) -> List[int]:
    """
    Collect UnitBlock indices by matching PyPSA asset names and SMS++ block type.

    The matching relies on unitblocks entries such as:
        {
            "name": "<pypsa asset name>",
            "enumerate": "UnitBlock_i",
            "block": "<block_type>",
        }

    Parameters
    ----------
    unitblocks : mapping
        Transformation unitblocks dictionary.
    names : sequence
        PyPSA physical asset names in DSS flattening order.
    block_type : str
        SMS++ block type, e.g. "IntermittentUnitBlock",
        "ThermalUnitBlock", or "HydroUnitBlock".
    """
    matches = {}

    for _, unitblock in unitblocks.items():
        name = unitblock.get("name", None)
        block = unitblock.get("block", None)
        enumerate_name = unitblock.get("enumerate", None)

        if name is None or block is None or enumerate_name is None:
            continue

        key = (name, block)
        matches.setdefault(key, []).append(enumerate_name)

    unitblock_indices = []

    for name in names:
        key = (name, block_type)
        candidates = matches.get(key, [])

        if not candidates:
            available = sorted(
                f"{asset_name!r}/{asset_block!r}"
                for asset_name, asset_block in matches
            )
            raise ValueError(
                f"Could not find a UnitBlock for asset {name!r} with "
                f"block type {block_type!r}. Available name/block pairs: "
                f"{available}"
            )

        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous UnitBlock match for asset {name!r} with "
                f"block type {block_type!r}: {candidates}"
            )

        enumerate_name = str(candidates[0])

        try:
            unitblock_index = int(enumerate_name.split("_")[-1])
        except Exception as exc:
            raise ValueError(
                f"Invalid UnitBlock enumerate value {enumerate_name!r} "
                f"for asset {name!r}."
            ) from exc

        unitblock_indices.append(unitblock_index)

    return unitblock_indices