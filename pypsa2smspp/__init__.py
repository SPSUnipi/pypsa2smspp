
"""
pypsa2smspp package init

Exposes high-level transformation and network correction utilities.
"""

import logging

logger = logging.getLogger("pypsa2smspp")
logger.setLevel(logging.INFO)

if not logger.handlers:  # 👈 evita di aggiungere più handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

# opzionale ma consigliato in una libreria:
logger.propagate = False



# Transformation logic (PyPSA ↔ SMS++)
from pypsa2smspp.transformation import Transformation
from pypsa2smspp.transformation_config import TransformationConfig

# PyPSA network correction tools
from pypsa2smspp.network_correction import (
    clean_marginal_cost,
    clean_global_constraints,
    clean_e_sum,
    clean_efficiency_link,
    clean_ciclicity_storage,
    clean_marginal_cost_intermittent,
    clean_storage_units,
    clean_stores,
    parse_txt_file,
    clean_p_min_pu,
    one_bus_network,
)

__all__ = [
    "Transformation",
    "TransformationConfig",
    # network correction tools
    "clean_marginal_cost",
    "clean_global_constraints",
    "clean_e_sum",
    "clean_efficiency_link",
    "clean_ciclicity_storage",
    "clean_marginal_cost_intermittent",
    "clean_storage_units",
    "clean_stores",
    "parse_txt_file",
    "clean_p_min_pu",
    "one_bus_network",
]
