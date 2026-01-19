# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:14:38 2024

@author: aless
"""

# --- Force working directory to this file's folder and build robust paths ---
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../pypsa2smspp/test
os.chdir(HERE)                                  # force CWD regardless of VSCode

print(">>> FORCED CWD:", Path.cwd())

# Ensure PYTHONPATH for imports from repo root (e.g., scripts/)
REPO_ROOT = HERE.parent                          # .../pypsa2smspp
SCRIPTS = (REPO_ROOT / "scripts").resolve()
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Safe output directory (create if missing, check writability)
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)
if not os.access(OUT, os.W_OK):
    raise PermissionError(f"Output dir not writable: {OUT} (check owner/perms)")

# Build your output file path here and pass it as string to downstream code
# OUTPUT_NC = OUT / "network_uc_hydro_0011.nc"
# If a stale, read-only file is blocking writes, you can optionally remove it:
# if OUTPUT_NC.exists():
#     OUTPUT_NC.unlink()

# print(">>> Output target:", OUTPUT_NC)
# ---------------------------------------------------------------------------


from configs.test_config import TestConfig
from network_definition import NetworkDefinition
from pypsa2smspp.transformation import Transformation
from datetime import datetime
import pysmspp

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
    compare_networks,
    add_slack_unit
    )

def get_datafile(fname):
    return os.path.join(os.path.dirname(__file__), "test_data", fname)

name = 'test_statistics'

#%% Network definition with PyPSA
config = TestConfig()
if "sector" in config.input_name_components:
    config.load_sign = -1

nd = NetworkDefinition(config)

nd.n = clean_ciclicity_storage(nd.n)

# if "sector" not in config.input_name_components:
#     nd.n = add_slack_unit(nd.n)
nd.n = add_slack_unit(nd.n)

network = nd.n.copy()
network.optimize(solver_name='gurobi')

network.export_to_netcdf(f"output/pypsa_{name}.nc")

network.model.to_file(fn = f"output/pypsa_{name}.lp")

#%% Transformation class

transformation = Transformation("..\\pypsa2smspp\\data\\config_default.yaml")
nd.n = transformation.run(nd.n)

objective_pypsa = network.objective + network.objective_constant
objective_smspp = nd.n.objective

error = (objective_pypsa - objective_smspp) / objective_pypsa * 100
    
print(f"Error PyPSA-SMS++ of {error}%")

statistics_pypsa = network.statistics()
statistics_smspp = nd.n.statistics()

