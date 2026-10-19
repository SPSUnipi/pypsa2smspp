# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import csv
import re
import shutil
from typing import Iterable, Sequence, Tuple

import numpy as np
import pytest
import pypsa
import pysmspp

from conftest import (
    create_test_config,
    safe_remove,
    tssb_test_cases,
    test_cases,
    REL_TOL,
    ABS_TOL,
    OUT_TEST,
)

import math
from network_definition import NetworkDefinition

from pypsa2smspp.transformation import Transformation
from pypsa2smspp.network_correction import (
    clean_ciclicity_storage,
    add_slack_unit,
)


SCENARIOS = ["low", "med", "high"]

PROB = {
    "low": 0.333333333333,
    "med": 0.333333333333,
    "high": 0.333333333333,
}

DEFAULT_STOCHASTIC_PARAMETERS = (
    "demand",
    "renewable_maxpower",
    # "renewable_marginal_cost",
)

TSSB_APPLICATION_INI = "application_tssb_test.ini"
TSSB_CONFIGFILE = "TSSBlock/TSSBSCfg.txt"

OBJECTIVES_CSV = OUT_TEST / "tssb_objectives_report.csv"
_OBJECTIVES_HEADER_WRITTEN = False


def stochastic_parameter_cases(
    parameters: Sequence[str],
) -> list[Tuple[str, ...]]:
    """
    Build stochastic-parameter test cases.

    The generated cases are:
    - each parameter alone;
    - all parameters together.

    Examples
    --------
    ["A", "B", "C"] -> [("A",), ("B",), ("C",), ("A", "B", "C")]
    """
    params = tuple(parameters)

    if not params:
        return []

    cases = [(p,) for p in params]

    if len(params) > 1:
        cases.append(params)

    return cases


STOCHASTIC_PARAMETER_CASES = stochastic_parameter_cases(
    DEFAULT_STOCHASTIC_PARAMETERS
)

STOCHASTIC_PARAMETER_IDS = [
    "+".join(case)
    for case in STOCHASTIC_PARAMETER_CASES
]


def _parameter_tag(stochastic_parameters: Sequence[str]) -> str:
    """
    Create a filesystem-friendly tag for stochastic parameters.
    """
    return "__".join(stochastic_parameters)


def _ensure_objectives_csv() -> None:
    """
    Create the CSV report with header if it does not exist yet.
    """
    global _OBJECTIVES_HEADER_WRITTEN

    if OBJECTIVES_CSV.exists() and OBJECTIVES_CSV.stat().st_size > 0:
        _OBJECTIVES_HEADER_WRITTEN = True
        return

    OBJECTIVES_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OBJECTIVES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_name",
                "network_path",
                "stochastic_parameters",
                "objective_smspp",
            ]
        )

    _OBJECTIVES_HEADER_WRITTEN = True


def _append_objective_row(
    case_name: str,
    network_path: Path,
    stochastic_parameters: Sequence[str],
    obj_smspp: float,
) -> None:
    """
    Append one TSSB objective row to the objectives CSV report.
    """
    _ensure_objectives_csv()

    with OBJECTIVES_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                case_name,
                str(network_path),
                "+".join(stochastic_parameters),
                obj_smspp,
            ]
        )


def apply_tssb_stochastic_data(
    n: pypsa.Network,
    stochastic_parameters: Iterable[str],
) -> pypsa.Network:
    """
    Convert a deterministic PyPSA network into a stochastic network for TSSB tests.

    The stochastic data are intentionally aligned with the standalone example
    script used for TSSB validation.
    """
    stochastic_parameters = set(stochastic_parameters)

    load = n.loads_t.p_set.copy()
    pmaxpu = n.generators_t.p_max_pu.copy()
    marginal = n.generators.marginal_cost.copy()
    inflow = n.storage_units_t.inflow.copy()

    n.set_scenarios(PROB)

    for st_p in stochastic_parameters:
        if st_p == "demand":
            load_value = {
                "low": load,
                "med": load * 2,
                "high": load * 4,
            }

            for scenario in SCENARIOS:
                n.loads_t.p_set[scenario] = load_value[scenario]

        elif st_p == "renewable_maxpower":
            pmaxpu_value = {
                "low": pmaxpu / 2,
                "med": pmaxpu,
                "high": pmaxpu * 2 / 3,
            }

            for scenario in SCENARIOS:
                n.generators_t.p_max_pu[scenario] = pmaxpu_value[scenario]

        elif st_p == "renewable_marginal_cost":
            marginal_value = {
                "low": marginal,
                "med": marginal * 2,
                "high": marginal * 4,
            }

            for scenario in SCENARIOS:
                n.generators.marginal_cost[scenario] = marginal_value[scenario]

        elif st_p == "hydro_inflow":
            inflow_value = {
                "low": inflow * 0,
                "med": inflow,
                "high": inflow * 4,
            }

            for scenario in SCENARIOS:
                n.storage_units_t.inflow[scenario] = inflow_value[scenario]

        else:
            raise ValueError(f"Unknown stochastic parameter: {st_p}")

    return n


def run_tssb_block(
    xlsx_path: Path,
    stochastic_parameters: Sequence[str],
) -> None:
    """
    TSSB regression test:
    - build a PyPSA network from Excel;
    - apply stochastic data;
    - run the SMS++ TSSB pipeline;
    - save the SMS++ objective to CSV.
    """
    parameter_tag = _parameter_tag(stochastic_parameters)
    case_name = f"{xlsx_path.stem}__{parameter_tag}"

    workdir = OUT_TEST / "tssb" / case_name
    workdir.mkdir(parents=True, exist_ok=True)

    network_nc = workdir / f"network_{case_name}.nc"

    safe_remove(network_nc)

    parser = create_test_config(
        xlsx_path,
        fp=TSSB_APPLICATION_INI,
    )
    parser.stochastic_parameters = list(stochastic_parameters)

    nd = NetworkDefinition(parser)
    n = nd.n

    n = clean_ciclicity_storage(n)

    if "sector" not in xlsx_path.name.lower():
        n = add_slack_unit(n)

    n = apply_tssb_stochastic_data(
        n,
        stochastic_parameters=parser.stochastic_parameters,
    )

    transformation = Transformation(
        name=case_name,
        configfile=TSSB_CONFIGFILE,
        enable_thermal_units=False,
        workdir=workdir,
        stochastic_parameters={
            "stochastic_type": "tssb",
            "parameters": parser.stochastic_parameters,
        },
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
        fp_log="smspp_{name}_log.txt",
        fp_solution="smspp_{name}_solution.nc",
        pysmspp_options={},
    )

    n = transformation.run(n, verbose=False)

    obj_smspp = float(transformation.result.objective_value)

    _append_objective_row(
        case_name=case_name,
        network_path=xlsx_path,
        stochastic_parameters=parser.stochastic_parameters,
        obj_smspp=obj_smspp,
    )

    assert transformation.result is not None
    assert math.isfinite(obj_smspp)

    try:
        n.export_to_netcdf(str(network_nc))
    except Exception:
        pass


@pytest.mark.parametrize(
    "xlsx_path",
    tssb_test_cases["xlsx_paths"],
    ids=tssb_test_cases["ids"],
)
@pytest.mark.parametrize(
    "stochastic_parameters",
    STOCHASTIC_PARAMETER_CASES,
    ids=STOCHASTIC_PARAMETER_IDS,
)
def test_tssb(xlsx_path, stochastic_parameters):
    if not tssb_test_cases["xlsx_paths"]:
        pytest.skip("No TSSB Excel test cases found")

    run_tssb_block(
        xlsx_path=xlsx_path,
        stochastic_parameters=stochastic_parameters,
    )


# ---------------------------------------------------------------------------
# design_cost_outside, which is gone
# ---------------------------------------------------------------------------

def test_design_cost_outside_is_refused():
    """
    A unit has its design Variable only when its investment cost is not zero,
    hence stating the cost outside the units took the Variable away with it:
    the option is gone, and asking for it says so instead of doing nothing.
    """
    n = pypsa.Network(
        str(Path(__file__).resolve().parent / "networks"
            / "pypsa_stoch_load.nc"))

    transformation = Transformation(
        name="tssb_design_cost_outside",
        configfile="TSSBlock/TSSBSCfg.txt",
        enable_thermal_units=False,
        capacity_expansion_ucblock=True,
        workdir=str(OUT_TEST / "tssb" / "design_cost_outside"),
        stochastic_parameters={
            "stochastic_type": "tssb",
            "parameters": ["demand"],
            "design_cost_outside": True,
        },
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
    )

    with pytest.raises(ValueError, match="design_cost_outside"):
        transformation.create_model(n, verbose=False)


if __name__ == "__main__":
    safe_remove(OBJECTIVES_CSV)

    if not tssb_test_cases["xlsx_paths"]:
        raise RuntimeError("No TSSB Excel test cases found")
    
    run_tssb_block(
        xlsx_path=tssb_test_cases["xlsx_paths"][0],
        stochastic_parameters=STOCHASTIC_PARAMETER_CASES[0],
    )

    print(f"Objective report written to: {OBJECTIVES_CSV}")


# ---------------------------------------------------------------------------
# hydro_system: the hydro units in a HydroSystemUnitBlock, stochastic inflows
# ---------------------------------------------------------------------------

# the networks with a hydro StorageUnit
hydro_cases = [(path, name)
               for path, name in zip(test_cases["xlsx_paths"], test_cases["ids"])
               if re.search(r"_\d+h(_|$)", path.stem)]


def tssb_solver_config():
    """
    The BlockSolverConfig of the installed smspp_tssb_solver, None if there is
    none. Its directory holds the InnerBCfg.txt that asks for the
    "linearized" PolyhedralFunctionBlock that a MILPSolver needs.
    """
    solver = shutil.which("smspp_tssb_solver")
    if solver is None:
        return None
    config = (Path(solver).resolve().parent.parent / "share" / "SMS++_tools"
              / "tssb_solver" / "config")
    inner = config / "InnerBCfg.txt"
    if not (inner.exists() and (config / "TSSBSCfg.txt").exists()
            and "PolyhedralFunctionBlock" in inner.read_text()):
        return None
    return pysmspp.SMSConfig(fp=config / "TSSBSCfg.txt")


def run_tssb_hydro(xlsx_path: Path, hydro_system: bool):
    """
    The two-stage problem of a network whose demand and hydro inflows are
    stochastic, solved by PyPSA and by SMS++; returns the two objectives and
    the Transformation.
    """
    stochastic_parameters = ["demand", "hydro_inflow"]
    case_name = (f"{xlsx_path.stem}__hydro_inflow"
                 + ("_hsub" if hydro_system else ""))
    workdir = OUT_TEST / "tssb" / case_name
    workdir.mkdir(parents=True, exist_ok=True)

    # the profiles that the xlsx does not give are drawn at random: the
    # networks are drawn alike
    np.random.seed(0)
    nd = NetworkDefinition(create_test_config(xlsx_path,
                                              fp=TSSB_APPLICATION_INI))
    n = add_slack_unit(clean_ciclicity_storage(nd.n))
    # PyPSA 1.2 does not build the ramp limits of a stochastic network (it
    # broadcasts the extendable flags over the names only); the start-up and
    # shut-down limits of 1 in the xlsx are no limit for a unit that is not
    # committable, and SMS++ does not read them
    for component in (n.generators, n.links):
        component[["ramp_limit_start_up", "ramp_limit_shut_down"]] = np.nan
    # the reservoirs start empty, otherwise what they hold covers the
    # horizon and the inflows would not change the optimum
    n.storage_units["state_of_charge_initial"] = 0.0
    n = apply_tssb_stochastic_data(n, stochastic_parameters)

    reference = n.copy()
    reference.optimize(solver_name="highs")
    obj_pypsa = float(reference.objective
                      + getattr(reference, "objective_constant", 0.0))

    transformation = Transformation(
        name=case_name,
        configfile=tssb_solver_config() if hydro_system else TSSB_CONFIGFILE,
        enable_thermal_units=False,
        hydro_system=hydro_system,
        workdir=workdir,
        stochastic_parameters={
            "stochastic_type": "tssb",
            "parameters": stochastic_parameters,
        },
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
        fp_log="smspp_{name}_log.txt",
        fp_solution="smspp_{name}_solution.nc",
        # the MILPSolver needs the "linearized" PolyhedralFunctionBlock of the
        # HydroSystemUnitBlock, which InnerBCfg.txt asks for
        pysmspp_options={"B": "InnerBCfg.txt"} if hydro_system else {},
    )
    transformation.run(n, verbose=False)

    return obj_pypsa, float(transformation.result.objective_value), \
        transformation


@pytest.mark.parametrize("xlsx_path",
                         [path for path, _ in hydro_cases],
                         ids=[name for _, name in hydro_cases])
def test_tssb_hydro_system(xlsx_path):
    """
    The stochastic inflows of the hydro units reach their HydroUnitBlocks
    inside the HydroSystemUnitBlock: the optimum is the one of PyPSA and of
    the HydroUnitBlocks in the UCBlock.
    """
    if tssb_solver_config() is None:
        pytest.skip("no installed smspp_tssb_solver linearizing a "
                    "PolyhedralFunctionBlock")

    obj_pypsa, obj_plain, _ = run_tssb_hydro(xlsx_path, hydro_system=False)
    obj_pypsa_hsub, obj_hsub, transformation = run_tssb_hydro(
        xlsx_path, hydro_system=True)

    assert obj_pypsa_hsub == pytest.approx(obj_pypsa, rel=1e-9)
    assert obj_plain == pytest.approx(obj_pypsa, rel=REL_TOL, abs=ABS_TOL)
    assert obj_hsub == pytest.approx(obj_plain, rel=REL_TOL, abs=ABS_TOL)

    # the mapping of set_inflow of the j-th hydro unit goes down to the
    # HydroUnitBlock_j inside the HydroSystemUnitBlock, the last UnitBlock
    stochastic = transformation.tssb_data["stochastic_block"]
    path = stochastic["AbstractPath"]
    starts = [start for start, name in zip(path["PathStart"],
                                           stochastic["FunctionName"])
              if name == "HydroUnitBlock::set_inflow"]
    assert starts
    system = transformation.dimensions["UCBlock"]["NumberUnits"] - len(starts)
    for j, start in enumerate(starts):
        assert list(path["PathNodeTypes"][start:start + 2]) == [b"B", b"B"]
        assert [str(g) for g in path["PathGroupIndices"][start:start + 2]] \
            == [str(system), str(j)]
