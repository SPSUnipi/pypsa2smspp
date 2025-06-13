# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:14:38 2024

@author: aless
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Aggiunge il percorso relativo per la cartella `config`
sys.path.append(os.path.abspath("../scripts"))
# Aggiunge il percorso relativo per la cartella `scripts`
sys.path.append(os.path.abspath("."))

DIR = os.path.dirname(os.path.abspath(__file__))


from configs.test_config import TestConfig
from network_definition import NetworkDefinition
from pypsa2smspp.transformation import Transformation
from datetime import datetime
import pysmspp
# from pysmspp import SMSNetwork, SMSFileType, Variable, Block, SMSConfig

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
    compare_networks
    )

#%% Network definition with PyPSA
config = TestConfig()
nd = NetworkDefinition(config)

# nd.n.storage_units_t.inflow["hydro"] = 0
# nd.n.storage_units_t.inflow.at[0,'hydro'] = 1

network = nd.n.copy()
network.optimize(solver_name='gurobi')

# network.export_to_netcdf("network_pypsa.nc")

# network.model.to_file(fn = "f.lp")
#%% Transformation class
then = datetime.now()
transformation = Transformation(network)
print(f"La classe di trasformazione ci mette {datetime.now() - then} secondi")

tran = transformation.convert_to_blocks()

configfile = pysmspp.SMSConfig(template="uc_solverconfig")  # load a default config file [highs solver]
temporary_smspp_file = "output/temp_network.nc"  # path to temporary SMS++ file
output_file = "output/temp_log_file.txt"  # path to the output file (optional)
solution_file = "output/temp_solution_file.nc"

# Check if the file exists
if os.path.exists(solution_file):
    os.remove(solution_file)

result = tran.optimize(configfile, temporary_smspp_file, output_file, solution_file)

statistics = network.statistics()
operational_cost = statistics['Operational Expenditure'].sum()
error = (operational_cost - result.objective_value) / operational_cost * 100
print(f"Error PyPSA-SMS++ of {error}%")




###############################################################################################
##################### Inverse transformation (UCBlock only for now) ###########################
###############################################################################################


# Esegui la funzione sul file di testo
# data_dict = parse_txt_file(output_file)

# print(f"Il solver ci ha messo {data_dict['elapsed_time']}s")
# print(f"Il tempo totale (trasformazione+pysmspp+ottimizzazione smspp) è {datetime.now() - then}")


# solution = transformation.parse_solution_to_unitblocks(solution_file)
# # transformation.parse_txt_to_unitblocks(output_file)
# transformation.inverse_transformation(nd.n)

# differences = compare_networks(network, nd.n)
# statistics_smspp = nd.n.statistics()

