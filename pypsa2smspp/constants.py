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