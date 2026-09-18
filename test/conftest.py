import os, sys, traceback
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import time
import pandas as pd
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
    Get all Excel test cases whose full path contains 'tssb'.
    """
    files = sorted(
        p for p in inputs_dir.rglob("*.xlsx")
        if "tssb" in str(p).lower()
    )

    names = [
        f"{i}: {p.relative_to(HERE)}"
        for i, p in enumerate(files)
    ]

    return {"xlsx_paths": files, "ids": names}


tssb_test_cases = get_tssb_test_cases()


# the test networks whose global constraints become pollutant budgets, which
# an SMS++ older than UCBlock b5e68de9 does not read
POLLUTANT_CASES = "co2_"


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
    solver = shutil.which("smspp_ucblock_solver") or shutil.which("ucblock_solver")
    if solver is None:
        return False

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

    config = Path(pysmspp.__file__).parent / "data" / "configs" / "UCBlock" / "uc_solverconfig.txt"
    try:
        run = subprocess.run([solver, "-D", "-S", str(config), str(probe)],
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False

    return run.returncode == 0 and "valid Block" not in run.stdout + run.stderr


def get_test_cases(inputs_dir = HERE / "configs" / "data" / "test"):
    """
    Get all test case Excel files and their names for parametrization.

    A network with a global constraint is left out where the solver at hand
    does not read the pollutant budget it becomes.
    """
    files = list(sorted(inputs_dir.glob("*.xlsx")))
    if not solver_reads_pollutant_budget():
        files = [f for f in files if not f.name.startswith(POLLUTANT_CASES)]
    names = [f"{i}: {f.name}" for (i,f) in enumerate(files)]
    return {"xlsx_paths": files, "ids": names}

test_cases = get_test_cases()

def get_network(fp: Path | str) -> str:
    """Helper to load a network from a .nc file."""
    return str(Path(HERE) / "networks" / fp)
