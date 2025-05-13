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

from pypsa2smspp.transformation import Transformation
from datetime import datetime
from pysmspp import SMSNetwork, SMSFileType, Variable, Block, SMSConfig
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
    clean_p_min_pu,
    one_bus_network,
    parse_txt_file
)

DIR = os.path.dirname(os.path.abspath(__file__))


def process_network(network_name="microgrid_microgrid_ALL_4N"):
    #%% Network definition with PyPSA
    fp = os.path.join(DIR, "networks", f"{network_name}.nc")
    network = pypsa.Network(fp)

    network = clean_marginal_cost(network)
    network = clean_global_constraints(network)
    network = clean_e_sum(network)
    network = clean_ciclicity_storage(network)

    # network = one_bus_network(network)
    # network = clean_efficiency_link(network)


    # network = clean_p_min_pu(network)
    # network = clean_marginal_cost_intermittent(network)
    # network = clean_storage_units(network)
    # network = clean_stores(network)


    network.optimize(solver_name='highs')
    # network.export_to_netcdf()


    #%% Transformation class
    then = datetime.now()
    transformation = Transformation(network)
    print(f"La classe di trasformazione ci mette {datetime.now() - then} secondi")

    transformation.convert_to_ucblock()
    configfile = SMSConfig(template="uc_solverconfig")

    temporary_smspp_file = os.path.join(DIR, "output", f"{network_name}.nc")  # path to temporary SMS++ file
    output_file = os.path.join(DIR, "output", f"{network_name}.txt")  # path to the output file (optional)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result = transformation.optimize(configfile, temporary_smspp_file, output_file)

    # Esegui la funzione sul file di testo
    data_dict = parse_txt_file(output_file)

    print(f"Il solver ci ha messo {data_dict['elapsed_time']}s")
    print(f"Il tempo totale (trasformazione+pysmspp+ottimizzazione smspp) è {datetime.now() - then}")

    statistics = network.statistics()
    operational_cost = statistics['Operational Expenditure'].sum()
    error = (operational_cost - result.objective_value) / operational_cost * 100
    print(f"Error PyPSA-SMS++ of {error}%")

    unitblocks = transformation.unitblocks
    transformation.parse_txt_to_unitblocks(output_file)
    transformation.inverse_transformation(network)

    assert "success" in result.status.lower()
    assert "error" not in result.log.lower()
    assert pytest.approx(error, abs=1e-5) == 0.


def test_network():
    process_network()
    

