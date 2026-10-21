import os, sys, traceback
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import time
import pandas as pd
import numpy as np
import netCDF4
import pypsa
import pysmspp
import pytest

REL_TOL = 1e-3   # relative tolerance for objective comparison. TODO: tighten tolerance
ABS_TOL = 1e-4   # absolute tolerance for objective comparison

# --- Force working directory to this file's folder and build robust paths ---
HERE = Path(__file__).resolve().parent            # .../pypsa2smspp/test

# Safe output dirs
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)
OUT_TEST = OUT / "test"
OUT_TEST.mkdir(parents=True, exist_ok=True)

# --- Domain imports (after PYTHONPATH is set) ---
from configs.test_config import TestConfig
from network_definition import NetworkDefinition
from pypsa2smspp.transformation import Transformation
from pypsa2smspp.stochastic_utils import has_extendable_assets

from pypsa2smspp.network_correction import (
    clean_global_constraints,
    clean_e_sum,
    clean_ciclicity_storage,
    clean_stores,
    parse_txt_file,
    add_slack_unit,
    compare_networks,  # optional: not used in timings but kept for debugging
)

# ---------- Utilities ----------

def safe_remove(p: Path):
    """Remove path if exists."""
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass

def create_test_config(xlsx_path: Path, fp: str | Path = "application_test.ini") -> TestConfig:
    """Create a TestConfig object pointing to the given input file."""
    parser = TestConfig(fp=str(fp))
    parser.input_data_path = str(xlsx_path.parent)
    parser.input_name_components = xlsx_path.name

    if "sector" in xlsx_path.name:
        parser.load_sign = -1

    return parser


def get_tssb_test_cases(inputs_dir: Path = HERE / "configs" / "data" / "test"):
    """
    Get all Excel test cases whose full path contains 'tssb', but those with
    a committable link where the solver does not know the
    ConversionUnitBlock [see CONVERSION_UNIT_CASES].
    """
    files = sorted(
        p for p in inputs_dir.rglob("*.xlsx")
        if "tssb" in str(p).lower()
    )
    if not solver_reads_conversion_unit():
        files = [p for p in files
                 if not p.name.startswith(CONVERSION_UNIT_CASES)]

    names = [
        f"{i}: {p.relative_to(HERE)}"
        for i, p in enumerate(files)
    ]

    return {"xlsx_paths": files, "ids": names}


# the test networks that need what no released SMS++ has: pollutant budgets
# (co2_, UCBlock b5e68de9), the operating rules of a nuclear unit (nuc_,
# UCBlock 1cf3cba5), thermal ramps that change over time (uc_, UCBlock
# c4f67f56) and an integer design of more than one module (mod_, UCBlock
# 87edf11f, and in an InvestmentBlock InvestmentBlock e43b146, BundleSolver
# e1c624f, SMS++ 8f41958); the solver that reads a pollutant budget is taken
# to have all of them
UNRELEASED_CASES = ("co2_", "nuc_", "uc_", "mod_")

# the test networks with an extendable AC line, which has a susceptance and
# hence Kirchhoff's voltage law: the DCNetworkBlock of SMS++ 0.6.3 does not
# solve it (the network comes out infeasible), the one of the develop of
# UCBlock does; they are left out where UNRELEASED_CASES are
UNRELEASED_NETWORKS = ("3n_3c_1gext_1h_1bext_2l",)

# the test networks with a committable link, which is a ConversionUnitBlock
# (chp_, UCBlock bab404c7), whose DCNetworkBlock also takes a hyperarc
# together with lines of nonzero susceptance; they are left out where the
# solver does not know the ConversionUnitBlock
CONVERSION_UNIT_CASES = ("chp_",)

# the test networks with a committable link that has an efficiency changing
# over time, a delay, modules or a design (chpx_, UCBlock 74c8c414); they
# are left out where the solver does not know these data of the
# ConversionUnitBlock
CONVERSION_EXT_CASES = ("chpx_",)

# the test networks with a committable link whose continuous design has ramps
# or is on before the horizon, or whose modules built are on before it
# (chpy_, UCBlock f092aa81); they are left out where the solver does not
# take such a design
CONVERSION_DESIGN_CASES = ("chpy_",)

# the test networks of several nodes with no line, joined by the units only:
# the DCNetworkBlock solves them since UCBlock 4892479b; they are left out
# where the solver does not
NO_LINE_NETWORKS = ("chp_3n_3g_1cl_0l",)


def solver_reads_pollutant_budget():
    """
    True if the smspp_ucblock_solver on PATH loads a UCBlock with a pollutant
    budget (under the name it had before the prefix, ucblock_solver, if it is
    not found).

    The pollutant budget constraints are in SMS++ since UCBlock b5e68de9: an
    older smspp_ucblock_solver does not accept the file, and then the test is
    skipped rather than failed. The check loads, without solving it, a one-unit UCBlock
    with a CO2 budget.
    """
    probe = OUT_TEST / "pollutant_budget_probe.nc4"
    with netCDF4.Dataset(probe, "w") as nc:
        nc.setncattr("SMS++_file_type", 1)
        uc = nc.createGroup("Block_0")
        uc.setncattr("type", "UCBlock")
        for dim, size in (("TimeHorizon", 1), ("NumberUnits", 1),
                          ("NumberElectricalGenerators", 1), ("NumberNodes", 1),
                          ("NumberPollutants", 1), ("TotalNumberPollutantZones", 1)):
            uc.createDimension(dim, size)
        uc.createVariable("ActivePowerDemand", "f8", ("NumberNodes", "TimeHorizon"))[:] = 1.0
        uc.createVariable("PollutantBudget", "f8", ("TotalNumberPollutantZones",))[:] = 1.0
        uc.createVariable("PollutantRho", "f8", ("TimeHorizon", "NumberPollutants",
                                                 "NumberElectricalGenerators"))[:] = 1.0
        unit = uc.createGroup("UnitBlock_0")
        unit.setncattr("type", "SlackUnitBlock")
        unit.createVariable("MaxPower", "f8")[...] = 10.0
        unit.createVariable("ActivePowerCost", "f8")[...] = 1.0

    return solver_loads(probe)


def solver_reads_conversion_unit():
    """
    True if the smspp_ucblock_solver on PATH loads a UCBlock with a
    ConversionUnitBlock, which is in SMS++ since UCBlock bab404c7; the check
    loads, without solving it, a one-unit UCBlock with a ConversionUnitBlock
    of one generator.
    """
    probe = OUT_TEST / "conversion_unit_probe.nc4"
    with netCDF4.Dataset(probe, "w") as nc:
        nc.setncattr("SMS++_file_type", 1)
        uc = nc.createGroup("Block_0")
        uc.setncattr("type", "UCBlock")
        for dim, size in (("TimeHorizon", 1), ("NumberUnits", 1),
                          ("NumberElectricalGenerators", 1), ("NumberNodes", 1)):
            uc.createDimension(dim, size)
        uc.createVariable("ActivePowerDemand", "f8", ("NumberNodes", "TimeHorizon"))[:] = 1.0
        unit = uc.createGroup("UnitBlock_0")
        unit.setncattr("type", "ConversionUnitBlock")
        unit.createVariable("MaxPower", "f8")[...] = 10.0

    return solver_loads(probe)


def solver_reads_carriers():
    """
    True if the smspp_ucblock_solver on PATH loads a UCBlock with more than
    one energy carrier, which is in SMS++ since UCBlock 0.8 (the format of
    Transformation(multi_carrier=True)); the check loads, without solving
    it, a UCBlock with the electricity and the heat, one node each, and a
    unit with its output on the heat.
    """
    probe = OUT_TEST / "carriers_probe.nc4"
    with netCDF4.Dataset(probe, "w") as nc:
        nc.setncattr("SMS++_file_type", 1)
        uc = nc.createGroup("Block_0")
        uc.setncattr("type", "UCBlock")
        for dim, size in (("TimeHorizon", 1), ("NumberUnits", 1),
                          ("NumberElectricalGenerators", 1), ("NumberNodes", 1),
                          ("NumberCarriers", 2)):
            uc.createDimension(dim, size)
        uc.createVariable("ActivePowerDemand", "f8", ("NumberNodes", "TimeHorizon"))[:] = 0.0
        uc.createVariable("CarrierName", str, ("NumberCarriers",))[:] = \
            np.array(["electricity", "heat"], dtype=object)
        uc.createVariable("GeneratorNode", "i4",
                          ("NumberElectricalGenerators", "NumberCarriers"))[:] = [[-1, 0]]
        heat = uc.createGroup("Carrier_1")
        heat.createDimension("NumberNodes", 1)
        heat.createVariable("EnergyDemand", "f8", ("NumberNodes", "TimeHorizon"))[:] = 1.0
        unit = uc.createGroup("UnitBlock_0")
        unit.setncattr("type", "IntermittentUnitBlock")
        unit.createVariable("MaxPower", "f8")[...] = 10.0
        unit.createDimension("NumberOutputs", 1)
        unit.createVariable("OutputCarrier", str, ("NumberOutputs",))[:] = \
            np.array(["heat"], dtype=object)
        unit.createVariable("OutputEfficiency", "f8",
                            ("TimeHorizon", "NumberOutputs"))[:] = 1.0

    return solver_loads(probe)


# the two forms of the buses of a network: the nodes of one network, or those
# of a carrier of the UCBlock for each carrier of the buses [see
# Transformation(multi_carrier=...)]
BUS_FORMS = [False, True]
BUS_FORM_IDS = ["network", "carriers"]


def skip_carrier_form(n, multi_carrier):
    """
    Skips the carrier form of a network whose buses are all electric, which
    is the network form, and that of any network if the solver on PATH does
    not read the carriers [see solver_reads_carriers()].
    """
    if not multi_carrier:
        return
    from pypsa2smspp.utils import ELECTRIC_BUS_CARRIERS
    if set(n.buses.carrier.fillna("").astype(str)) <= set(ELECTRIC_BUS_CARRIERS):
        pytest.skip("all the buses are electric: no carrier form")
    if not solver_reads_carriers():
        pytest.skip("the SMS++ solver does not read the carriers")


def solver_reads_conversion_extensions():
    """
    True if the smspp_ucblock_solver on PATH loads a ConversionUnitBlock
    whose OperatingMatrix is indexed over the instants, which it does since
    UCBlock 74c8c414 together with the delays, the modules and the design;
    the check loads, without solving it, a one-unit UCBlock with such a unit
    of two generators.
    """
    probe = OUT_TEST / "conversion_extensions_probe.nc4"
    with netCDF4.Dataset(probe, "w") as nc:
        nc.setncattr("SMS++_file_type", 1)
        uc = nc.createGroup("Block_0")
        uc.setncattr("type", "UCBlock")
        for dim, size in (("TimeHorizon", 2), ("NumberUnits", 1),
                          ("NumberElectricalGenerators", 2), ("NumberNodes", 1)):
            uc.createDimension(dim, size)
        uc.createVariable("ActivePowerDemand", "f8", ("NumberNodes", "TimeHorizon"))[:] = 1.0
        unit = uc.createGroup("UnitBlock_0")
        unit.setncattr("type", "ConversionUnitBlock")
        unit.createDimension("NumberGenerators", 2)
        unit.createDimension("NumberOperatingRows", 1)
        unit.createVariable("MinPower", "f8", ("TimeHorizon", "NumberGenerators"))[:] = [[-10.0, 0.0]] * 2
        unit.createVariable("MaxPower", "f8", ("TimeHorizon", "NumberGenerators"))[:] = [[0.0, 30.0]] * 2
        unit.createVariable("OperatingMatrix", "f8", ("TimeHorizon", "NumberOperatingRows",
                                                      "NumberGenerators"))[:] = [[[2.0, 1.0]], [[3.0, 1.0]]]
        for side in ("OperatingLHS", "OperatingRHS"):
            unit.createVariable(side, "f8", ("TimeHorizon", "NumberOperatingRows"))[:] = 0.0

    return solver_loads(probe)


def solver_reads_conversion_design():
    """
    True if the smspp_ucblock_solver on PATH loads a ConversionUnitBlock with
    a continuous design and ramps, which it does since UCBlock f092aa81
    together with a design on before the horizon; the check loads, without
    solving it, a one-unit UCBlock with such a unit of one generator.
    """
    probe = OUT_TEST / "conversion_design_probe.nc4"
    with netCDF4.Dataset(probe, "w") as nc:
        nc.setncattr("SMS++_file_type", 1)
        uc = nc.createGroup("Block_0")
        uc.setncattr("type", "UCBlock")
        for dim, size in (("TimeHorizon", 2), ("NumberUnits", 1),
                          ("NumberElectricalGenerators", 1), ("NumberNodes", 1)):
            uc.createDimension(dim, size)
        uc.createVariable("ActivePowerDemand", "f8", ("NumberNodes", "TimeHorizon"))[:] = 1.0
        unit = uc.createGroup("UnitBlock_0")
        unit.setncattr("type", "ConversionUnitBlock")
        unit.createVariable("MaxPower", "f8")[...] = 1.0
        unit.createVariable("DeltaRampUp", "f8")[...] = 0.5
        unit.createVariable("InvestmentCost", "f8")[...] = 1.0
        unit.createVariable("MaxCapacityDesign", "f8")[...] = 10.0
        unit.createVariable("InitUpDownTime", "i4")[...] = -1

    return solver_loads(probe)


def solver_solves_no_line_network():
    """
    True if the smspp_ucblock_solver on PATH solves a UCBlock of two nodes
    with no line, each with its own slack unit, which it does since UCBlock
    4892479b; the check solves it, the error being raised where the network
    constraints are generated.
    """
    probe = OUT_TEST / "no_line_probe.nc4"
    with netCDF4.Dataset(probe, "w") as nc:
        nc.setncattr("SMS++_file_type", 1)
        uc = nc.createGroup("Block_0")
        uc.setncattr("type", "UCBlock")
        for dim, size in (("TimeHorizon", 1), ("NumberUnits", 2),
                          ("NumberElectricalGenerators", 2), ("NumberNodes", 2)):
            uc.createDimension(dim, size)
        uc.createDimension("NumberLines", 0)
        uc.createVariable("ActivePowerDemand", "f8", ("NumberNodes", "TimeHorizon"))[:] = 1.0
        uc.createVariable("GeneratorNode", "u4", ("NumberElectricalGenerators",))[:] = [0, 1]
        for i in range(2):
            unit = uc.createGroup(f"UnitBlock_{i}")
            unit.setncattr("type", "SlackUnitBlock")
            unit.createVariable("MaxPower", "f8")[...] = 10.0
            unit.createVariable("ActivePowerCost", "f8")[...] = 1.0

    return solver_loads(probe, dry_run=False)


def solver_loads(probe, dry_run=True):
    """
    True if the smspp_ucblock_solver on PATH (under the name it had before
    the prefix, ucblock_solver, if it is not found) loads the UCBlock in the
    file probe without an error, and also solves it if not dry_run.
    """
    solver = shutil.which("smspp_ucblock_solver") or shutil.which("ucblock_solver")
    if solver is None:
        return False

    config = Path(pysmspp.__file__).parent / "data" / "configs" / "UCBlock" / "uc_solverconfig.txt"
    try:
        run = subprocess.run([solver] + (["-D"] if dry_run else [])
                             + ["-S", str(config), str(probe)],
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False

    return run.returncode == 0 and "valid Block" not in run.stdout + run.stderr


def get_test_cases(inputs_dir = HERE / "configs" / "data" / "test"):
    """
    Get all test case Excel files and their names for parametrization.

    A network that needs what no released SMS++ has is left out where the
    solver at hand does not read a pollutant budget [see UNRELEASED_CASES and
    UNRELEASED_NETWORKS], so is a network with a committable link where it
    does not know the ConversionUnitBlock [see CONVERSION_UNIT_CASES], and
    so is a network of several nodes with no line where it does not solve it
    [see NO_LINE_NETWORKS], or a committable link with the data of
    CONVERSION_EXT_CASES or CONVERSION_DESIGN_CASES where it does not read
    them.
    """
    files = list(sorted(inputs_dir.glob("*.xlsx")))
    if not solver_reads_pollutant_budget():
        files = [f for f in files if not f.name.startswith(UNRELEASED_CASES)
                 and f.stem not in UNRELEASED_NETWORKS]
    if not solver_reads_conversion_unit():
        files = [f for f in files
                 if not f.name.startswith(CONVERSION_UNIT_CASES)]
    if not solver_solves_no_line_network():
        files = [f for f in files if f.stem not in NO_LINE_NETWORKS]
    if not solver_reads_conversion_extensions():
        files = [f for f in files
                 if not f.name.startswith(CONVERSION_EXT_CASES)]
    if not solver_reads_conversion_design():
        files = [f for f in files
                 if not f.name.startswith(CONVERSION_DESIGN_CASES)]
    names = [f"{i}: {f.name}" for (i,f) in enumerate(files)]
    return {"xlsx_paths": files, "ids": names}

test_cases = get_test_cases()
tssb_test_cases = get_tssb_test_cases()

def scenario_reference_objective(n: pypsa.Network) -> float | None:
    """
    The optimum of PyPSA for a stochastic network with a committable
    component, or None if it has none or if it has here-and-now variables.

    PyPSA does not build the model of a stochastic network with a committable
    component; with no extendable asset nothing is here-and-now, the
    scenarios are independent, and the optimum is the sum of the optima of
    the scenarios, each solved by PyPSA, weighted by their probabilities.
    """
    committable = any(
        "committable" in df.columns and bool(df["committable"].any())
        for df in (n.generators, n.links)
    )
    if not committable or has_extendable_assets(n):
        return None

    objective = 0.0
    for scenario, probability in n.scenario_weightings["weight"].items():
        deterministic = n.get_scenario(scenario)
        deterministic.optimize(solver_name="highs")
        objective += float(probability) * float(
            deterministic.objective
            + getattr(deterministic, "objective_constant", 0.0))
    return objective


def modular_conversion_reference(n: pypsa.Network,
                                 solver_name: str = "highs") -> float | None:
    """
    The reference of the InvestmentBlock form for a network with an
    extendable modular committable link, or None if it has none.

    The InvestmentFunction solves the continuous relaxation of the UCBlock,
    in which a unit of k modules committed one by one is the unit of one
    module scaled by k, i.e., the link with p_nom = k p_nom_mod, no modules
    and the costs of being on, of a start-up and of a shut-down k times
    those of a module; PyPSA does not relax the commitment of a modular
    link, hence the reference is the smallest of the optima of PyPSA with
    the commitment relaxed over the number k of modules built, plus the
    capital cost of the k modules, which PyPSA does not count for a fixed
    capacity.
    """
    links = n.links
    if "p_nom_mod" not in links.columns:
        return None
    modular = links.index[links.committable.astype(bool)
                          & links.p_nom_extendable.astype(bool)
                          & (links.p_nom_mod > 0)]
    if modular.empty:
        return None
    if len(modular) > 1:
        raise ValueError("modular_conversion_reference: one modular "
                         "extendable committable link at most")

    name = modular[0]
    size = float(links.at[name, "p_nom_mod"])
    low = int(np.ceil(float(links.at[name, "p_nom_min"]) / size - 1e-9))
    high = int(np.floor(float(links.at[name, "p_nom_max"]) / size + 1e-9))

    best = None
    for k in range(low, high + 1):
        m = n.copy()
        m.links.loc[name, ["p_nom_mod", "p_nom_extendable", "p_nom"]] = \
            [0.0, False, k * size]
        for cost in ("start_up_cost", "shut_down_cost", "stand_by_cost"):
            m.links.loc[name, cost] = k * float(links.at[name, cost])
        m.optimize(solver_name=solver_name, linearized_unit_commitment=True)
        if m.objective is None:
            continue
        value = float(m.objective + getattr(m, "objective_constant", 0.0)) \
            + k * size * float(links.at[name, "capital_cost"])
        best = value if best is None else min(best, value)
    return best


def get_network(fp: Path | str) -> str:
    """Helper to load a network from a .nc file."""
    return str(Path(HERE) / "networks" / fp)
