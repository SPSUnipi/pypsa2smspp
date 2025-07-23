# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:14:38 2024

@author: aless
"""

import sys
import os
import pytest

# Aggiunge il percorso relativo per la cartella `config`
sys.path.append(os.path.abspath("../scripts"))
# Aggiunge il percorso relativo per la cartella `scripts`
sys.path.append(os.path.abspath("."))

from configs.test_config import TestConfig
from network_definition import NetworkDefinition
from pypsa2smspp.transformation import Transformation
from datetime import datetime
import pysmspp
import pypsa
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
DIR = os.path.dirname(os.path.abspath(__file__))


def process_network(network_name="test_pypsa"):
    #%% Network definition with PyPSA
    fp = os.path.join(DIR, "networks", f"{network_name}.nc")
    n_smspp = pypsa.Network(fp)

    n_smspp = clean_global_constraints(n_smspp)
    n_smspp = clean_e_sum(n_smspp)
    n_smspp = clean_ciclicity_storage(n_smspp)
    
    n_smspp.generators.p_nom_extendable = False
    n_smspp.lines.s_nom_extendable = False
    n_smspp.links.p_nom_extendable = False
    n_smspp.storage_units.p_nom_extendable = False


    network = n_smspp.copy()
    network.optimize(solver_name='gurobi')


    # network.export_to_netcdf("network_errore.nc")

    # network.model.to_file(fn = "pypsa.lp")
    #%% Transformation class
    then = datetime.now()
    transformation = Transformation(network)
    print(f"Il tempo per la trasformazione diretta è di {datetime.now() - then} secondi")
    then = datetime.now()

    tran = transformation.convert_to_blocks()
    print(f"Il tempo per la conversione con pysmspp è di {datetime.now() - then} secondi")


    if transformation.dimensions['InvestmentBlock']['NumAssets'] == 0:
        ### UCBlock configuration ###
        configfile = pysmspp.SMSConfig(template="uc_solverconfig")  # load a default config file [highs solver]
        temporary_smspp_file = "output/network_pypsaeur_0110.nc"  # path to temporary SMS++ file
        output_file = "output/temp_log_file.txt"  # path to the output file (optional)
        solution_file = "output/solution_pypsaeur_0110.nc"
        
        # Check if the file exists
        if os.path.exists(solution_file):
            os.remove(solution_file)
        
        then = datetime.now()
        result = tran.optimize(configfile, temporary_smspp_file, output_file, solution_file)
        print(f"Il tempo di tran.optimize è di {datetime.now() - then} secondi")
        
        statistics = network.statistics()
        operational_cost = statistics['Operational Expenditure'].sum()
        error = (operational_cost - result.objective_value) / operational_cost * 100
        
        print(f"Error PyPSA-SMS++ of {error}%")
        
        # Esegui la funzione sul file di testo
        data_dict = parse_txt_file(output_file)

        print(f"Il solver ci ha messo {data_dict['elapsed_time']}s (preso da file di testo)")
        

        then = datetime.now()
        solution = transformation.parse_solution_to_unitblocks(result.solution, n_smspp)
        print(f"Il tempo di conversione inversa in unitblocks è di {datetime.now() - then} secondi")
        
        then = datetime.now()
        # transformation.parse_txt_to_unitblocks(output_file)
        transformation.inverse_transformation(n_smspp)
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
        
        result = tran.optimize(configfile, temporary_smspp_file, output_file, solution_file, inner_block_name='InvestmentBlock')
        
        
        objective_pypsa = network.objective + network.objective_constant
        objective_smspp = result.objective_value
        error = (objective_pypsa - objective_smspp) / objective_pypsa
        
        print(f"Error PyPSA-SMS++ of {error}%")

    assert "success" in result.status.lower()
    assert "error" not in result.log.lower()
    assert pytest.approx(error, abs=5e-5) == 0.


def test_network():
    process_network()
    
    