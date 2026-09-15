# -*- coding: utf-8 -*-
from pathlib import Path
import pytest

from conftest import (
    create_test_config,
    test_cases,
    REL_TOL,
    ABS_TOL,
    OUT_TEST,
)

from network_definition import NetworkDefinition
from pypsa2smspp.transformation import Transformation

from pypsa2smspp.network_correction import (
    clean_ciclicity_storage,
    add_slack_unit,
)

# CO2 intensity of the fossil carriers of the test networks [t/MWh_th]
CO2_EMISSIONS = {"CCGT": 0.35, "gas": 0.2, "diesel": 0.27}

# fraction of the unconstrained emissions allowed by the CO2 limit
CO2_FRACTIONS = [2.0, 0.5]


def emissions(n):
    """Emissions of the dispatch of an optimized network, as PyPSA counts them."""
    weights = n.snapshot_weightings.generators
    efficiency = n.get_switchable_as_dense("Generator", "efficiency")
    total = 0.0
    for gen, carrier in n.generators.carrier.items():
        rate = n.carriers.at[carrier, "co2_emissions"]
        if rate:
            total += rate * (n.generators_t.p[gen] / efficiency[gen] * weights).sum()
    return total


def run_pollutant_budget(xlsx_path: Path, fraction: float) -> None:
    """
    Pollutant budget regression test:
    - build network from Excel and give the fossil carriers a CO2 intensity
    - bound the emissions by a fraction of the unconstrained ones
    - solve reference with PyPSA
    - run full SMS++ pipeline, where the limit is a pollutant budget
    - compare objectives
    """
    parser = create_test_config(xlsx_path)
    n = NetworkDefinition(parser).n
    n = clean_ciclicity_storage(n)
    n = add_slack_unit(n)

    for carrier, rate in CO2_EMISSIONS.items():
        if carrier in n.carriers.index:
            n.carriers.at[carrier, "co2_emissions"] = rate

    solver_name = getattr(parser, "solver_name", "highs")

    free = n.copy()
    free.optimize(solver_name=solver_name)
    free_emissions = emissions(free)
    if free_emissions <= 0:
        pytest.skip("the unconstrained dispatch emits no CO2")

    n.add(
        "GlobalConstraint",
        "CO2Limit",
        type="primary_energy",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=fraction * free_emissions,
    )

    network = n.copy()
    network.optimize(solver_name=solver_name)
    obj_pypsa = float(network.objective + getattr(network, "objective_constant", 0.0))

    case_name = f"{xlsx_path.stem}_co2_{fraction}"
    transformation = Transformation(
        capacity_expansion_ucblock=True,
        workdir=OUT_TEST,
        name=case_name,
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
        fp_log="smspp_{name}_log.txt",
        fp_solution="smspp_{name}_solution.nc",
        configfile="auto",
        pysmspp_options={},
    )

    transformation.run(network, verbose=False)
    obj_smspp = float(transformation.result.objective_value)

    assert obj_smspp == pytest.approx(obj_pypsa, rel=REL_TOL, abs=ABS_TOL)


fossil_cases = [
    (p, i) for p, i in zip(test_cases["xlsx_paths"], test_cases["ids"])
    if p.stem == "3n_3c_1gext_1h_1bext_2l"
]


@pytest.mark.parametrize("fraction", CO2_FRACTIONS)
@pytest.mark.parametrize(
    "test_case_xlsx", [p for p, _ in fossil_cases], ids=[i for _, i in fossil_cases]
)
def test_pollutant_budget(test_case_xlsx, fraction):
    run_pollutant_budget(test_case_xlsx, fraction)


if __name__ == "__main__":
    run_pollutant_budget(fossil_cases[0][0], CO2_FRACTIONS[-1])
