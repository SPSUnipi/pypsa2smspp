from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Utilities for stochastic PyPSA networks.

This module keeps stochastic-network inspection and lightweight data extraction
outside of the main Transformation class, so the pipeline can remain readable.
"""



from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


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
    """Return a deterministic network for direct conversion.

    If the input network is stochastic, return the first scenario.
    Otherwise, return the original network unchanged.
    """
    scenario_names = get_scenario_names(n)

    if scenario_names:
        return n.get_scenario(scenario_names[0])

    return n


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


def describe_problem_structure(n) -> Dict[str, Any]:
    """Describe the high-level optimization structure of the PyPSA network."""
    is_stochastic = is_stochastic_network(n)
    scenario_names = get_scenario_names(n)

    return {
        "is_stochastic": is_stochastic,
        "stochastic_type": "tssb" if is_stochastic else None,
        "number_scenarios": len(scenario_names),
        "scenario_names": scenario_names,
        "has_investment": has_extendable_assets(n),
        "has_uc": True,
        "has_scenario_dependent_demand": is_stochastic,
    }


def get_bus_demand_matrix(n) -> pd.DataFrame:
    """Aggregate load time series by bus.

    Returns
    -------
    pd.DataFrame
        Index = buses, columns = snapshots.
    """
    if getattr(n, "loads", None) is None or n.loads.empty:
        return pd.DataFrame(index=n.buses.index, columns=n.snapshots, dtype=float).fillna(0.0)

    demand = n.loads_t.p_set.rename(columns=n.loads.bus)
    demand = demand.T.groupby(level=0).sum()
    demand = demand.reindex(index=n.buses.index, fill_value=0.0)
    demand = demand.reindex(columns=n.snapshots, fill_value=0.0)
    demand = demand.fillna(0.0)

    return demand


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


def build_tssb_demand_scenarios(
    n,
) -> Tuple[np.ndarray, np.ndarray, List[Any], List[Any], str]:
    """Build scenario matrix and probabilities for demand-only TSSB.

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
        raise ValueError("TSSB scenario extraction requested on a deterministic network.")

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
            demand_by_bus = demand_by_bus.reindex(index=bus_order, columns=snapshot_order, fill_value=0.0)

        scenarios.append(flatten_bus_demand_time_major(demand_by_bus))

    scenario_matrix = np.vstack(scenarios).astype(float)
    pool_weights = get_scenario_probabilities(n).astype(float)
    flattening = "time_major_node_minor"

    return scenario_matrix, pool_weights, bus_order, snapshot_order, flattening

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