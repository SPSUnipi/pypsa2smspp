# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import csv
from typing import Iterable, Sequence, Tuple

import pytest
import pypsa

from conftest import (
    create_test_config,
    safe_remove,
    tssb_test_cases,
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
TSSB_CONFIGFILE = "TSSBlock/TSSBSCfg_grb.txt"

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


if __name__ == "__main__":
    safe_remove(OBJECTIVES_CSV)

    if not tssb_test_cases["xlsx_paths"]:
        raise RuntimeError("No TSSB Excel test cases found")
    
    run_tssb_block(
        xlsx_path=tssb_test_cases["xlsx_paths"][0],
        stochastic_parameters=STOCHASTIC_PARAMETER_CASES[0],
    )

    print(f"Objective report written to: {OBJECTIVES_CSV}")