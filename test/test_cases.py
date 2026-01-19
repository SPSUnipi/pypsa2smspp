# -*- coding: utf-8 -*-
"""
Unified pytest runner for dispatch (UCBlock) and investment (InvestmentBlock).

New style:
- No pysmspp usage here
- Single-call pipeline via Transformation(cfg).run(network)
- Timings are handled inside Transformation (transformation.timer)

Author: aless (refactored)
"""

from pathlib import Path
import pytest
import pypsa

REL_TOL = 1e-5
ABS_TOL = 1e-3

HERE = Path(__file__).resolve().parent

OUT = HERE / "output" / "test"
OUT.mkdir(parents=True, exist_ok=True)

# --- Domain imports ---
from configs.test_config import TestConfig
from network_definition import NetworkDefinition
from pypsa2smspp.transformation import Transformation

from pypsa2smspp.network_correction import (
    clean_ciclicity_storage,
    add_slack_unit,
)

# NEW: load YAML as AttrDict so we can override io.name per case
from pypsa2smspp.pip_utils import load_yaml_config

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def safe_remove(p: Path) -> None:
    """Remove a file if it exists."""
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def create_test_config(xlsx_path: Path) -> TestConfig:
    """Create a TestConfig object pointing to the given Excel file."""
    parser = TestConfig()
    parser.input_data_path = str(xlsx_path.parent)
    parser.input_name_components = xlsx_path.name
    if "sector" in xlsx_path.name:
        parser.load_sign = -1
    return parser


def build_network_from_excel(xlsx_path: Path):
    """Build and clean a PyPSA network from a single Excel input."""
    parser = create_test_config(xlsx_path)
    nd = NetworkDefinition(parser)

    n = nd.n
    n = clean_ciclicity_storage(n)
    if "sector" not in xlsx_path.name:
        n = add_slack_unit(n)

    return n, parser


def pypsa_reference_objective(network: "pypsa.Network") -> float:
    """Robust reference objective extraction."""
    try:
        return float(network.objective + getattr(network, "objective_constant", 0.0))
    except Exception:
        return float(network.objective)


def get_test_cases(inputs_dir: Path = HERE / "configs" / "data" / "test"):
    """Get all test case Excel files and their names for parametrization."""
    files = list(sorted(inputs_dir.glob("*.xlsx")))
    names = [f"{i}: {f.name}" for (i, f) in enumerate(files)]
    return files, names


def make_case_cfg(config_yaml: Path, case_name: str):
    """
    Load YAML config and override io.name so each pytest case writes unique artifacts.
    This prevents cross-test collisions in output/test/.
    """
    cfg = load_yaml_config(config_yaml)
    # ensure nested keys exist
    if not hasattr(cfg, "io"):
        cfg.io = {}
    cfg.io.name = f"pytest_{case_name}"
    # keep workdir from YAML (or default)
    return cfg


# ---------------------------------------------------------------------
# Core runners
# ---------------------------------------------------------------------

def run_case(xlsx_path: Path, config_yaml: Path, solver_name: str = "highs", verbose: bool = False) -> None:
    case_name = xlsx_path.stem
    prefix = case_name

    # Optional artifacts (PyPSA-side)
    network_nc = OUT / f"network_{prefix}.nc"
    pypsa_lp = OUT / f"pypsa_{prefix}.lp"
    safe_remove(network_nc)
    safe_remove(pypsa_lp)

    # Build + clean
    n, _parser = build_network_from_excel(xlsx_path)

    # Reference solve
    network = n.copy()
    network.optimize(solver_name=solver_name)

    # Export LP for debugging (best effort)
    network.model.to_file(fn=str(pypsa_lp))

    obj_pypsa = pypsa_reference_objective(network)

    # SMS++ pipeline (one call) with per-case cfg override
    cfg = make_case_cfg(config_yaml, case_name)
    transformation = Transformation(cfg)  # IMPORTANT: Transformation must accept dict/AttrDict configs
    n = transformation.run(network, verbose=verbose)

    obj_smspp = float(transformation.result.objective_value)

    # If this still fails sometimes, consider relaxing rel tol to 1e-4 for investment tests.
    assert obj_smspp == pytest.approx(obj_pypsa, rel=REL_TOL, abs=ABS_TOL)

    # Optional export
    try:
        network.export_to_netcdf(str(network_nc))
    except Exception:
        pass


# ---------------------------------------------------------------------
# Pytest parametrization
# ---------------------------------------------------------------------

test_cases, test_names = get_test_cases()

CFG_UCBLOCK = Path(__file__).resolve().parents[1] / "test" / "configs" / "config_test_ucblock.yaml"
CFG_INVEST = Path(__file__).resolve().parents[1] / "test" / "configs" / "config_test_investment.yaml"


@pytest.mark.parametrize("test_case_xlsx", test_cases, ids=test_names)
def test_dispatch(test_case_xlsx):
    if not CFG_UCBLOCK.exists():
        pytest.skip(f"Missing UCBlock test config: {CFG_UCBLOCK}")
    run_case(test_case_xlsx, CFG_UCBLOCK, solver_name="highs", verbose=False)


@pytest.mark.parametrize("test_case_xlsx", test_cases, ids=test_names)
def test_investment(test_case_xlsx):
    if not CFG_INVEST.exists():
        pytest.skip(f"Missing InvestmentBlock test config: {CFG_INVEST}")
    name_l = test_case_xlsx.name.lower()
    if "inv" not in name_l:
        pytest.skip("Skipping case for investment block")
    run_case(test_case_xlsx, CFG_INVEST, solver_name="highs", verbose=False)


if __name__ == "__main__":
    if not CFG_INVEST.exists():
        raise FileNotFoundError(CFG_INVEST)
    if not CFG_UCBLOCK.exists():
        raise FileNotFoundError(CFG_UCBLOCK)
    run_case(test_cases[5], CFG_INVEST, solver_name="highs", verbose=True)
