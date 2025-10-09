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

#%% Network definition with PyPSA
config = TestConfig()
nd = NetworkDefinition(config)

nd.n = add_slack_unit(nd.n)
nd.n = clean_ciclicity_storage(nd.n)


network = nd.n.copy()
network.optimize(solver_name='gurobi')

# network.export_to_netcdf("test_pypsa.nc")

network.model.to_file(fn = "pypsa.lp")
#%% Transformation class
then = datetime.now()
transformation = Transformation(network, merge_links=True, expansion_ucblock=True)
print(f"La classe di trasformazione ci mette {datetime.now() - then} secondi")

tran = transformation.convert_to_blocks()

if transformation.dimensions['InvestmentBlock']['NumAssets'] == 0 or transformation.expansion_ucblock:
    ### UCBlock configuration ###
    configfile = pysmspp.SMSConfig(template="UCBlock/uc_solverconfig_grb")  # load a default config file [highs solver]
    temporary_smspp_file = "output/network_uc_hydro_0011.nc"  # path to temporary SMS++ file
    output_file = "output/temp_log_file.txt"  # path to the output file (optional)
    solution_file = "output/solution_uc_hydro_0011.nc"
    
    # Check if the file exists
    if os.path.exists(solution_file):
        os.remove(solution_file)
    
    result = tran.optimize(configfile, temporary_smspp_file, output_file, solution_file, log_executable_call=True)
    
    statistics = network.statistics()
    operational_cost = statistics['Operational Expenditure'].sum()
    # error = (operational_cost - result.objective_value) / operational_cost * 100

    objective_pypsa = network.objective + network.objective_constant
    objective_smspp = result.objective_value
    error = (objective_pypsa - objective_smspp) / objective_pypsa * 100
    
    print(f"Error PyPSA-SMS++ of {error}%")
    
    # Esegui la funzione sul file di testo
    data_dict = parse_txt_file(output_file)

    print(f"Il solver ci ha messo {data_dict['elapsed_time']}s")
    print(f"Il tempo totale (trasformazione+pysmspp+ottimizzazione smspp) è {datetime.now() - then}")

    
    solution = transformation.parse_solution_to_unitblocks(result.solution, nd.n)
    # transformation.parse_txt_to_unitblocks(output_file)
    transformation.inverse_transformation(nd.n)

    differences = compare_networks(network, nd.n)
    statistics_smspp = nd.n.statistics()
    

else:
    ### InvestmentBlock configuration ###
    configfile = pysmspp.SMSConfig(template="InvestmentBlock/BSPar.txt")
    temporary_smspp_file = "output/temp_network_investment_daily.nc"
    output_file = "output/temp_log_file_investment.txt"  # path to the output file (optional)
    solution_file = "output/temp_solution_file_investment.nc"
    
    # Check if the file exists
    if os.path.exists(solution_file):
        os.remove(solution_file)
    
    result = tran.optimize(configfile, temporary_smspp_file, output_file, solution_file, inner_block_name='InvestmentBlock', log_executable_call=True)
    
    
    objective_pypsa = network.objective + network.objective_constant
    objective_smspp = result.objective_value
    error = (objective_pypsa - objective_smspp) / objective_pypsa * 100
    
    print(f"Error PyPSA-SMS++ of {error}%")
    print(f"Il tempo totale (trasformazione+pysmspp+ottimizzazione smspp) è {datetime.now() - then}")

    solution = transformation.parse_solution_to_unitblocks(result.solution, nd.n)
    transformation.inverse_transformation(nd.n)
    
    statistics = network.statistics()
    statistics_smspp = nd.n.statistics()

