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

name = 'stochastic_base_marginal'

#%% Network definition with PyPSA
config = TestConfig()
if "sector" in config.input_name_components:
    config.load_sign = -1

nd = NetworkDefinition(config)

nd.n = clean_ciclicity_storage(nd.n)

# if "sector" not in config.input_name_components:
#     nd.n = add_slack_unit(nd.n)
nd.n = add_slack_unit(nd.n)

SCENARIOS = ["low", "med", "high"]
DIESEL_PRICES = {"low": 30, "med": 70, "high": 100}  # EUR/MWh_th
load = nd.n.loads_t.p_set
LOAD_VALUE = {"low": load, "med": load * 2, "high": load * 4}
PROB = {"low": 0.4, "med": 0.3, "high": 0.3}  # Scenario probabilities

network_stoch_load = nd.n.copy()
network_stoch_price = nd.n.copy()

# Become stochastic
network_stoch_load.set_scenarios(PROB)
network_stoch_price.set_scenarios(PROB)

for scenario in SCENARIOS:
    network_stoch_price.generators.loc[(scenario, 'IT0 0 diesel'), "marginal_cost"] = DIESEL_PRICES[scenario]
    network_stoch_load.loads_t.p_set[(scenario, "IT0 0")] = LOAD_VALUE[scenario]

nd.n.optimize(solver_name='gurobi')
network_stoch_load.optimize(solver_name='gurobi')
network_stoch_price.optimize(solver_name='gurobi')

nd.n.export_to_netcdf("output/pypsa_deterministic.nc")
network_stoch_load.export_to_netcdf("output/pypsa_stoch_load.nc")
network_stoch_price.export_to_netcdf("output/pypsa_stoch_price.nc")


# network.model.to_file(fn = f"output/pypsa_{name}.lp")

statistics = nd.n.statistics()
statistics_stoch_load = network_stoch_load.statistics()
statistics_stoch_price = network_stoch_price.statistics()

#%% Transformation class
# then = datetime.now()
# transformation = Transformation(network, merge_links=True, expansion_ucblock=True)
# print(f"La classe di trasformazione ci mette {datetime.now() - then} secondi")


# tran = transformation.convert_to_blocks()


# if transformation.expansion_ucblock or transformation.dimensions['InvestmentBlock']['NumAssets'] == 0:
#     ### UCBlock configuration ###
#     configfile = pysmspp.SMSConfig(template="UCBlock/uc_solverconfig_grb")  # load a default config file [highs solver]
#     temporary_smspp_file = f"output/network_{name}.nc"  # path to temporary SMS++ file
#     output_file = f"output/log_{name}.txt"  # path to the output file (optional)
#     solution_file = f"output/solution_{name}.nc"
    
#     # Check if the file exists
#     if os.path.exists(solution_file):
#         os.remove(solution_file)
    
#     result = tran.optimize(configfile, temporary_smspp_file, output_file, solution_file, log_executable_call=True)
    
#     statistics = network.statistics()
#     operational_cost = statistics['Operational Expenditure'].sum()
#     # error = (operational_cost - result.objective_value) / operational_cost * 100

#     objective_pypsa = network.objective + network.objective_constant
#     objective_smspp = result.objective_value
#     error = (objective_pypsa - objective_smspp) / objective_pypsa * 100
    
#     print(f"Error PyPSA-SMS++ of {error}%")
    
#     # Esegui la funzione sul file di testo
#     data_dict = parse_txt_file(output_file)

#     print(f"Il solver ci ha messo {data_dict['elapsed_time']}s")
#     print(f"Il tempo totale (trasformazione+pysmspp+ottimizzazione smspp) è {datetime.now() - then}")

    
#     solution = transformation.parse_solution_to_unitblocks(result.solution, nd.n)
#     # transformation.parse_txt_to_unitblocks(output_file)
#     transformation.inverse_transformation(nd.n)

#     differences = compare_networks(network, nd.n)
#     statistics_smspp = nd.n.statistics()
    

# else:
#     ### InvestmentBlock configuration ###
#     configfile = pysmspp.SMSConfig(template="InvestmentBlock/BSPar.txt")
#     temporary_smspp_file = f"output/inv_network_{name}.nc"  # path to temporary SMS++ file
#     output_file = f"output/inv_log_{name}.txt"  # path to the output file (optional)
#     solution_file = f"output/inv_solution_{name}.nc"
    
#     # Check if the file exists
#     if os.path.exists(solution_file):
#         os.remove(solution_file)
    
#     result = tran.optimize(configfile, temporary_smspp_file, output_file, solution_file, inner_block_name='InvestmentBlock', log_executable_call=True)
    
    
#     objective_pypsa = network.objective + network.objective_constant
#     objective_smspp = result.objective_value
#     error = (objective_pypsa - objective_smspp) / objective_pypsa * 100
    
#     print(f"Error PyPSA-SMS++ of {error}%")
#     print(f"Il tempo totale (trasformazione+pysmspp+ottimizzazione smspp) è {datetime.now() - then}")

#     solution = transformation.parse_solution_to_unitblocks(result.solution, nd.n)
#     transformation.inverse_transformation(nd.n)
    
#     statistics = network.statistics()
#     statistics_smspp = nd.n.statistics()

