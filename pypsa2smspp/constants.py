# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 15:50:01 2025

@author: aless
"""

"""
constants.py

This module contains structural constants used throughout the PyPSA2SMSpp 
transformation process. These values define generic mappings and categories
used repeatedly in the Transformation class and related utilities.

They are not meant to be modified by the user and do not depend on any 
specific instance or configuration of the network.
"""

# Dictionary mapping internal shorthand dimensions to full SMS++ dimension names
conversion_dict = {
    "T": "TimeHorizon",
    "NU": "NumberUnits",
    "NE": "NumberElectricalGenerators",
    "N": "NumberNodes",
    "L": "NumberLines",
    "Li": "Links",
    "NA": "NumberArcs",
    "NR": "NumberReservoirs",
    "NP": "TotalNumberPieces",
    "Nass": "NumAssets",
    "NB": "NumberBranches",
    "NDL": "NumberDesignLines",
    "NDLL": "NumberDesignLines_lines",
    "NDLLi": "NumberDesignLines_links",
    "1": 1
}

# Mapping from PyPSA component types to their nominal attribute
nominal_attrs = {
    "Generator": "p_nom",
    "Line": "s_nom",
    "Transformer": "s_nom",
    "Link": "p_nom",
    "Store": "e_nom",
    "StorageUnit": "p_nom",
}

# List of renewable carriers used to identify IntermittentUnitBlocks
renewable_carriers = [
    "solar",
    "solar-hsat",
    "onwind",
    "offwind-ac",
    "offwind-dc",
    "offwind-float",
    "PV",
    "wind",
    "ror"
]


# Registry of stochastic parameters supported by the TSSB interface.
#
# Each entry describes:
# - how the stochastic data are extracted from PyPSA;
# - how they are flattened in the DiscreteScenarioSet;
# - which SMS++ setter receives the stochastic slice.

STOCHASTIC_PARAMETER_REGISTRY = {
    "demand": {
        "mapping_kind": "ucblock_timeseries",
        "function_name": "UCBlock::set_active_power_demand",
        "target": "demand",
    },

    "renewable_maxpower": {
        "mapping_kind": "unitblock_timeseries",
        "pypsa_component": "Generator",
        "field": "p_max_pu",
        "asset_filter": "intermittent_generators",
        "unitblock_type": "IntermittentUnitBlock",
        "smspp_parameter": "MaxPower",
        "function_name": "IntermittentUnitBlock::set_maximum_power",
        "target": "renewable_maxpower",
        "weights": False,
    },

    "renewable_marginal_cost": {
        "mapping_kind": "unitblock_timeseries",
        "pypsa_component": "Generator",
        "field": "marginal_cost",
        "asset_filter": "intermittent_generators",
        "unitblock_type": "IntermittentUnitBlock",
        "smspp_parameter": "ActivePowerCost",
        "function_name": "IntermittentUnitBlock::set_active_power_cost",
        "target": "renewable_marginal_cost",
        "weights": False,
    },

    "thermal_stand_by_cost": {
        "mapping_kind": "unitblock_timeseries",
        "pypsa_component": "Generator",
        "field": "stand_by_cost",
        "asset_filter": "thermal_generators",
        "unitblock_type": "ThermalUnitBlock",
        "smspp_parameter": "ConstTerm",
        "function_name": "ThermalUnitBlock::set_const_term",
        "target": "thermal_stand_by_cost",
        "weights": False,
        "requires_enable_thermal_units": True,
    },

    "thermal_marginal_cost": {
        "mapping_kind": "unitblock_timeseries",
        "pypsa_component": "Generator",
        "field": "marginal_cost",
        "asset_filter": "thermal_generators",
        "unitblock_type": "ThermalUnitBlock",
        "smspp_parameter": "LinearTerm",
        "function_name": "ThermalUnitBlock::set_linear_term",
        "target": "thermal_marginal_cost",
        "weights": False,
        "requires_enable_thermal_units": True,
    },

    "thermal_marginal_cost_quadratic": {
        "mapping_kind": "unitblock_timeseries",
        "pypsa_component": "Generator",
        "field": "marginal_cost_quadratic",
        "asset_filter": "thermal_generators",
        "unitblock_type": "ThermalUnitBlock",
        "smspp_parameter": "QuadTerm",
        "function_name": "ThermalUnitBlock::set_quad_term",
        "target": "thermal_marginal_cost_quadratic",
        "weights": False,
        "requires_enable_thermal_units": True,
    },

    "hydro_inflow": {
        "mapping_kind": "unitblock_timeseries",
        "pypsa_component": "StorageUnit",
        "field": "inflow",
        "asset_filter": "storage_units",
        "unitblock_type": "HydroUnitBlock",
        "smspp_parameter": "Inflows",
        "function_name": "HydroUnitBlock::set_inflow",
        "target": "hydro_inflow",
        "weights": False,
    },
}