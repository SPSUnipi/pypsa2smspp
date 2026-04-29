# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 09:38:56 2025

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

SOLVER_OPTIONS = {
    "Threads": 32,
    "Method": 2,       # barrier
    "Crossover": 0,
    # "BarConvTol": 1e-6,
    "Seed": 123,
    "AggFill": 0,
    "PreDual": 0,
}

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
    add_slack_unit
    )

def get_datafile(fname):
    return os.path.join(os.path.dirname(__file__), "test_data", fname)

name = 'stochastic_base_load_equi_demand_maxpower'
folder = 'develop/tssb'

#%% Network definition with PyPSA
config = TestConfig(fp="application_stochastic.ini")

nd = NetworkDefinition(config)

# if "sector" not in config.input_name_components:
#     nd.n = add_slack_unit(nd.n)
nd.n = add_slack_unit(nd.n)

SCENARIOS = ["low", "med", "high"]
PROB = {"low": 0.333333333333, "med": 0.333333333333, "high": 0.333333333333}  # Scenario probabilities

load = nd.n.loads_t.p_set
pmaxpu = nd.n.generators_t.p_max_pu

nd.n.set_scenarios(PROB)

for st_p in config.stochastic_parameters:
    if st_p == "demand":
        LOAD_VALUE = {"low": load, "med": load * 2, "high": load * 4}
        for scenario in SCENARIOS:
            nd.n.loads_t.p_set[scenario] = LOAD_VALUE[scenario]
    elif st_p == "renewables":
        PMAXPU_VALUE = {"low": pmaxpu / 2, "med": pmaxpu, "high": pmaxpu * 2/3}
        for scenario in SCENARIOS:
            nd.n.generators_t.p_max_pu[scenario] = PMAXPU_VALUE[scenario]


n_pypsa = nd.n.copy()

n_pypsa.optimize(solver_name='gurobi') # , solver_options=SOLVER_OPTIONS)
obj_pypsa = n_pypsa.objective + n_pypsa.objective_constant

# n_pypsa.export_to_netcdf("output/develop/tssb/pypsa_stoch_load.nc")

statistics_pypsa = n_pypsa.statistics()

transformation = Transformation(name=name,
                                configfile="TSSBlock/TSSBSCfg_grb.txt",
                                enable_thermal_units=False,
                                workdir=f"output/{folder}",
                                stochastic_parameters={
                                    "stochastic_type": "tssb",
                                    "parameters": config.stochastic_parameters,
                                }
                                )
nd.n = transformation.run(nd.n)
statistics_smspp = nd.n.statistics()

obj_smspp = nd.n.objective
error = (obj_smspp - obj_pypsa) / obj_pypsa * 100
print(f"Error PyPSA-SMS++ of {error}%")

n_pypsa.export_to_netcdf(f"output/{folder}/pypsa_{name}.nc")
nd.n.export_to_netcdf(f"output/{folder}/smspp_{name}.nc")

n_pypsa.model.to_file(fn = f"output/{folder}/pypsa_{name}.lp")


