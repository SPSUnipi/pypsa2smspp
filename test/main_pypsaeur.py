# -*- coding: utf-8 -*-
"""
Created on Wed May 28 16:25:30 2025

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
import pypsa
import pandas as pd
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
    reduced_snapshot,
    parse_txt_file,
    compare_networks,
    add_slack_unit
    )

times = dict()
#%% Network definition with PyPSA
n_smspp = pypsa.Network("networks/base_s_5_elec_1h.nc")
investment_bool = False

n_smspp = clean_global_constraints(n_smspp)
n_smspp = clean_e_sum(n_smspp)
n_smspp = clean_ciclicity_storage(n_smspp)


if investment_bool:
    n_smspp = add_slack_unit(n_smspp)
else:
    n_smspp.generators.p_nom_extendable = False
    n_smspp.lines.s_nom_extendable = False

network = n_smspp.copy()
then = datetime.now()
network.optimize(solver_name='gurobi')
times['PyPSA'] = (datetime.now() - then).total_seconds()
print(f"Il tempo per ottimizzare con PyPSA è di {datetime.now() - then} secondi")

# network.export_to_netcdf("network_errore.nc")

# network.model.to_file(fn = "pypsa.lp")

#%% Transformation class
then = datetime.now()
transformation = Transformation(network)
times['Direct transformation'] = (datetime.now() - then).total_seconds()
print(f"Il tempo per la trasformazione diretta è di {datetime.now() - then} secondi")

then = datetime.now()
tran = transformation.convert_to_blocks()
times['PySMSpp conversion'] = (datetime.now() - then).total_seconds()
print(f"Il tempo per la conversione con pysmspp è di {datetime.now() - then} secondi")


if transformation.dimensions['InvestmentBlock']['NumAssets'] == 0:
    ### UCBlock configuration ###
    configfile = pysmspp.SMSConfig(template="UCBlock/uc_solverconfig")  # load a default config file [highs solver]
    temporary_smspp_file = "output/network_pypsaeur_0110.nc"  # path to temporary SMS++ file
    output_file = "output/temp_log_file.txt"  # path to the output file (optional)
    solution_file = "output/solution_pypsaeur_0110.nc"
    
    # Check if the file exists
    if os.path.exists(solution_file):
        os.remove(solution_file)
    
    then = datetime.now()
    result = tran.optimize(configfile, temporary_smspp_file, output_file, solution_file, log_executable_call=True)
    times['SMS++ (solver+writing)'] = (datetime.now() - then).total_seconds()
    print(f"Il tempo di tran.optimize è di {datetime.now() - then} secondi")
    
    statistics = network.statistics()
    operational_cost = statistics['Operational Expenditure'].sum()
    error = (operational_cost - result.objective_value) / operational_cost * 100
    
    print(f"Error PyPSA-SMS++ of {error}%")
    
    # Esegui la funzione sul file di testo
    data_dict = parse_txt_file(output_file)

    print(f"Il solver ci ha messo {data_dict['elapsed_time']}s (preso da file di testo)")
    times['SMS++ solver'] = data_dict['elapsed_time']
    times['SMS++ writing'] = times['SMS++ (solver+writing)'] - times['SMS++ solver']

    then = datetime.now()
    solution = transformation.parse_solution_to_unitblocks(result.solution, n_smspp)
    # print(f"Il tempo di conversione inversa in unitblocks è di {datetime.now() - then} secondi")
    
    # then = datetime.now()
    # transformation.parse_txt_to_unitblocks(output_file)
    transformation.inverse_transformation(n_smspp)
    times['Inverse transformation'] = (datetime.now() - then).total_seconds()
    print(f"Il tempo per la trasformazione inversa è di {datetime.now() - then} secondi")

    # differences = compare_networks(network, n_smspp)
    statistics_smspp = n_smspp.statistics()

else:
    ### InvestmentBlock configuration ###
    configfile = pysmspp.SMSConfig(template="InvestmentBlock/BSPar.txt")
    temporary_smspp_file = "output/temp_network_investment.nc"
    output_file = "output/temp_log_file_investment.txt"  # path to the output file (optional)
    solution_file = "output/temp_solution_file_investment.nc"
    
    # Check if the file exists
    if os.path.exists(solution_file):
        os.remove(solution_file)
    
    then = datetime.now()
    result = tran.optimize(configfile, temporary_smspp_file, output_file, solution_file, inner_block_name='InvestmentBlock', log_executable_call=True)
    times['SMS++ (solver+writing)'] = (datetime.now() - then).total_seconds()
    print(f"Il tempo di tran.optimize è di {datetime.now() - then} secondi")
    
    objective_pypsa = network.objective + network.objective_constant
    objective_smspp = result.objective_value
    error = (objective_pypsa - objective_smspp) / objective_pypsa * 100
    
    print(f"Error PyPSA-SMS++ of {error}%")

    then = datetime.now()
    solution = transformation.parse_solution_to_unitblocks(result.solution, n_smspp)
    transformation.inverse_transformation(n_smspp)
    times['Inverse transformation'] = (datetime.now() - then).total_seconds()
    print(f"Il tempo per la trasformazione inversa è di {datetime.now() - then} secondi")
    
    statistics = network.statistics()
    statistics_smspp = n_smspp.statistics()
    
    
times = pd.DataFrame.from_dict(times, orient="index", columns=["Seconds"])