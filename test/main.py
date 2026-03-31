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
import pypsa

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
    add_slack_unit,
    reduce_snapshots_and_scale_costs,
    )

SOLVER_OPTIONS = {
    "Threads": 32,
    "Method": 2,       # barrier
    "Crossover": 0,
    "BarConvTol": 1e-5,
    "Seed": 123,
    "AggFill": 0,
    "PreDual": 0,
}


def get_datafile(fname):
    return os.path.join(os.path.dirname(__file__), "test_data", fname)

name = 'sector_coupled_pypsaeur'

#%% Network definition with PyPSA
config = TestConfig()

if name == None:
    name = config.input_name_components.split("/")[-1].split(".")[0]

# if "sector" in config.input_name_components:
#     config.load_sign = -1

nd = NetworkDefinition(config)

# nd.n = clean_ciclicity_storage(nd.n)

# if "sector" not in config.input_name_components:
#     nd.n = add_slack_unit(nd.n)
# nd.n = add_slack_unit(nd.n)
# nd.n = clean_storage_units(nd.n)

network = nd.n.copy()
n = pypsa.Network(r"C:\Users\aless\sms\transformation_pypsa_smspp\test\networks\network_pypsa_network_giga_small.nc")

df_diff = compare_networks(
    n,
    network,
    rtol=1e-9,
    atol=1e-12,
    compare_dtypes=True,
)

network.optimize(solver_name='gurobi', solver_options=SOLVER_OPTIONS)

# network.export_to_netcdf(f"output/pypsa_{name}.nc")

network.model.to_file(fn = f"output/develop/pypsa_{name}.lp")

#%% Transformation class

transformation = Transformation(name=name,
                                workdir="output/develop/sector_coupled",
                                enable_thermal_units=False,
                                capacity_expansion_ucblock=True,
                                configfile="UCBlock/uc_solverconfig_grb.txt",
                                merge_links=False)
nd.n = transformation.run(nd.n)

# nd.n = nd.n.smspp(verbose=True)

objective_pypsa = network.objective + network.objective_constant
objective_smspp = nd.n.objective

error = (objective_smspp - objective_pypsa) / objective_pypsa * 100
    
print(f"Error PyPSA-SMS++ of {error}%")

stats_pypsa = network.statistics()
stats_smspp = nd.n.statistics()

