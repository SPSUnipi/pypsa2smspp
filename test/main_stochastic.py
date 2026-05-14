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
import numpy as np
import pandas as pd

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

name = 'stochastic_marginal_intermittent'
folder = 'develop/tssb'

#%% Network definition with PyPSA
config = TestConfig(fp="application_stochastic.ini")

nd = NetworkDefinition(config)

# if "sector" not in config.input_name_components:
#     nd.n = add_slack_unit(nd.n)
nd.n = add_slack_unit(nd.n)

# SCENARIOS = ["low", "med", "high"]
# PROB = {"low": 0.333333333333, "med": 0.333333333333, "high": 0.333333333333}  # Scenario probabilities

# load = nd.n.loads_t.p_set
# pmaxpu = nd.n.generators_t.p_max_pu
# marginal = nd.n.generators.marginal_cost

# nd.n.set_scenarios(PROB)

# for st_p in config.stochastic_parameters:
#     if st_p == "demand":
#         LOAD_VALUE = {"low": load, "med": load * 2, "high": load * 4}
#         for scenario in SCENARIOS:
#             nd.n.loads_t.p_set[scenario] = LOAD_VALUE[scenario]
#     elif st_p == "renewable_maxpower":
#         PMAXPU_VALUE = {"low": pmaxpu / 2, "med": pmaxpu, "high": pmaxpu * 2/3}
#         for scenario in SCENARIOS:
#             nd.n.generators_t.p_max_pu[scenario] = PMAXPU_VALUE[scenario]
#     elif st_p == "renewable_marginal_cost":
#         MARGINAL_VALUE = {"low": marginal, "med": marginal * 2, "high": marginal * 4}
#         for scenario in SCENARIOS:
#             nd.n.generators.marginal_cost[scenario] = MARGINAL_VALUE[scenario]


# ---------------------------------------------------------------------
# Stochastic scenario configuration
# ---------------------------------------------------------------------

N_SCENARIOS = 10
SEED = 123

SCENARIOS = [f"scenario_{i + 1}" for i in range(N_SCENARIOS)]
PROB = {scenario: 1.0 / N_SCENARIOS for scenario in SCENARIOS}

# Standard deviations of the Gaussian multipliers.
# Example: sigma = 0.20 means values are typically varied by about ±20%.
SIGMA = {
    "demand": 0.20,
    "renewable_maxpower": 0.15,
    "renewable_marginal_cost": 0.30,
}

# Choose how uncertainty is sampled:
# - "global": one multiplier per scenario and parameter
# - "elementwise": one multiplier per value, i.e. snapshot/asset-specific noise
DRAW_MODE = "global"

rng = np.random.default_rng(SEED)


def gaussian_multiplier(base, sigma, *, rng, mode="global", lower=0.0):
    """
    Generate Gaussian multipliers centered around 1.

    Parameters
    ----------
    base : pd.DataFrame | pd.Series
        Base object used only to infer the output shape.
    sigma : float
        Standard deviation of the Gaussian multiplier.
    rng : np.random.Generator
        Random number generator.
    mode : {"global", "elementwise"}
        If "global", one scalar multiplier is used for the whole object.
        If "elementwise", one multiplier is sampled for each entry.
    lower : float
        Minimum allowed multiplier.

    Returns
    -------
    float | pd.DataFrame | pd.Series
        Gaussian multiplier.
    """
    if mode == "global":
        multiplier = rng.normal(loc=1.0, scale=sigma)
        return max(multiplier, lower)

    if mode == "elementwise":
        values = rng.normal(loc=1.0, scale=sigma, size=base.shape)
        values = np.maximum(values, lower)

        if isinstance(base, pd.DataFrame):
            return pd.DataFrame(values, index=base.index, columns=base.columns)

        if isinstance(base, pd.Series):
            return pd.Series(values, index=base.index)

        return values

    raise ValueError(f"Unknown draw mode: {mode}")


# ---------------------------------------------------------------------
# Store original deterministic values before activating scenarios
# ---------------------------------------------------------------------

base_load = nd.n.loads_t.p_set.copy()
base_pmaxpu = nd.n.generators_t.p_max_pu.copy()
base_marginal = nd.n.generators.marginal_cost.copy()

# Optional: restrict renewable_maxpower uncertainty only to renewable generators.
# Adjust this list if your carrier names differ.
renewable_carriers = {
    "solar",
    "solar-hsat",
    "onwind",
    "offwind-ac",
    "offwind-dc",
    "offwind-float",
    "ror",
}

renewable_generators = nd.n.generators.index[
    nd.n.generators.carrier.isin(renewable_carriers)
]

# ---------------------------------------------------------------------
# Activate stochastic scenarios
# ---------------------------------------------------------------------

nd.n.set_scenarios(PROB)


# ---------------------------------------------------------------------
# Assign scenario-dependent values
# ---------------------------------------------------------------------

for scenario in SCENARIOS:

    if "demand" in config.stochastic_parameters:
        mult = gaussian_multiplier(
            base_load,
            SIGMA["demand"],
            rng=rng,
            mode=DRAW_MODE,
            lower=0.0,
        )

        nd.n.loads_t.p_set[scenario] = (base_load * mult).clip(lower=0.0)

    if "renewable_maxpower" in config.stochastic_parameters:
        mult = gaussian_multiplier(
            base_pmaxpu[renewable_generators],
            SIGMA["renewable_maxpower"],
            rng=rng,
            mode=DRAW_MODE,
            lower=0.0,
        )

        scenario_pmaxpu = base_pmaxpu.copy()

        scenario_pmaxpu.loc[:, renewable_generators] = (
            base_pmaxpu.loc[:, renewable_generators] * mult
        ).clip(lower=0.0, upper=1.0)

        nd.n.generators_t.p_max_pu[scenario] = scenario_pmaxpu

    if "renewable_marginal_cost" in config.stochastic_parameters:
        mult = gaussian_multiplier(
            base_marginal[renewable_generators],
            SIGMA["renewable_marginal_cost"],
            rng=rng,
            mode=DRAW_MODE,
            lower=0.0,
        )

        scenario_marginal = base_marginal.copy()

        scenario_marginal.loc[renewable_generators] = (
            base_marginal.loc[renewable_generators] * mult
        ).clip(lower=0.0)

        nd.n.generators.marginal_cost[scenario] = scenario_marginal


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


