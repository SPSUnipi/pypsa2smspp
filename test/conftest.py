import os, sys, traceback
from pathlib import Path
from datetime import datetime
import time
import pandas as pd
import pypsa
import pysmspp
import pytest

REL_TOL = 1e-1   # relative tolerance for objective comparison. TODO: tighten tolerance
ABS_TOL = 1e-2   # absolute tolerance for objective comparison

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

def create_test_config(xlsx_path: Path) -> TestConfig:
    """Create a TestConfig object pointing to the given Excel file."""
    parser = TestConfig()
    parser.input_data_path = str(xlsx_path.parent)       # folder of the excel
    parser.input_name_components = xlsx_path.name        # excel filename
    if "sector" in xlsx_path.name:
        parser.load_sign = -1
    return parser


def get_test_cases(inputs_dir = HERE / "configs" / "data" / "test"):
    """
    Get all test case Excel files and their names for parametrization.
    """
    files = list(sorted(inputs_dir.glob("*.xlsx")))
    names = [f"{i}: {f.name}" for (i,f) in enumerate(files)]
    return {"xlsx_paths": files, "ids": names}

test_cases = get_test_cases()

