# -*- coding: utf-8 -*-
from pathlib import Path
import pytest

from conftest import (
    create_test_config,
    safe_remove,
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

def test_split(xlsx_path: Path = test_cases["xlsx_paths"][0]) -> None:
    case_name = xlsx_path.stem

    # Artifacts (optional)
    network_nc = OUT_TEST / f"network_{case_name}.nc"
    pypsa_lp = OUT_TEST / f"pypsa_{case_name}.lp"

    for p in (network_nc, pypsa_lp):
        safe_remove(p)

    # ---- Build network from Excel ----
    parser = create_test_config(xlsx_path)
    nd = NetworkDefinition(parser)

    n = nd.n
    n = clean_ciclicity_storage(n)
    if "sector" not in xlsx_path.name:
        n = add_slack_unit(n)

    # Work on a copy for reference solve
    network = n.copy()

    # run call
    transformation = Transformation(
        capacity_expansion_ucblock=True,  # UCBlock
        workdir=OUT_TEST,
        name=case_name,
        overwrite=True,
        fp_temp="smspp_{name}_temp_split.nc",
        fp_log="smspp_{name}_log_split.txt",
        fp_solution="smspp_{name}_solution_split.nc",
        configfile="auto",
        pysmspp_options={},  # keep pySMSpp defaults
    )

    n = transformation.run(network, verbose=False)

    obj_smspp = float(transformation.result.objective_value)

    try:
        obj_pypsa = float(network.objective + getattr(network, "objective_constant", 0.0))
    except Exception:
        obj_pypsa = float(network.objective)

    assert obj_smspp == pytest.approx(obj_pypsa, rel=REL_TOL, abs=ABS_TOL)

def test_split_merged_link_per_scenario() -> None:
    """
    A merged charger/discharger link is read back as the two original links
    also when its flows are stored per scenario, as in a TSSB solution.
    """
    import numpy as np
    from pypsa2smspp.io_parser import split_merged_dcnetworkblocks

    flow = {"s1": np.array([2.0, -1.0]), "s2": np.array([-3.0, 0.5])}
    unitblocks = {
        "DCNetworkBlock_0": {
            "name": "IT0 0 battery charger__IT0 0 battery discharger",
            "enumerate": "UnitBlock_0",
            "scenarios": {s: {"FlowValue": f} for s, f in flow.items()},
        }
    }
    split_merged_dcnetworkblocks(unitblocks, logger=lambda msg: None)

    names = {b["name"]: b for b in unitblocks.values()}
    charger = names["IT0 0 battery charger"]
    discharger = names["IT0 0 battery discharger"]
    for s, f in flow.items():
        assert np.array_equal(charger["scenarios"][s]["FlowValue"], np.maximum(f, 0.0))
        assert np.array_equal(discharger["scenarios"][s]["FlowValue"], np.maximum(-f, 0.0))


def test_merge_battery_with_free_discharger() -> None:
    """
    The battery of PyPSA-Eur, whose discharger has no capital cost, is merged
    by merge_links: the discharger is kept extendable rather than fixed at a
    large capacity, which would make the pair no longer mergeable.
    """
    import pypsa
    from pypsa2smspp.utils import (
        build_store_and_merged_links,
        preprocess_zero_capital_cost_extendable_lines_links,
    )

    n = pypsa.Network()
    n.add("Bus", ["IT0 0", "IT0 0 battery"])
    n.add("Store", "IT0 0 battery", bus="IT0 0 battery", e_nom_extendable=True)
    n.add("Link", "IT0 0 battery charger", bus0="IT0 0", bus1="IT0 0 battery",
          efficiency=0.98, capital_cost=100.0, p_nom_extendable=True)
    n.add("Link", "IT0 0 battery discharger", bus0="IT0 0 battery", bus1="IT0 0",
          efficiency=0.98, capital_cost=0.0, p_nom_extendable=True)

    _, probe, _ = build_store_and_merged_links(n, merge_links=["battery"],
                                               logger=lambda msg: None)
    absorbed = n.links.index.difference(probe.index)
    assert set(absorbed) == {"IT0 0 battery charger", "IT0 0 battery discharger"}

    n, fixed = preprocess_zero_capital_cost_extendable_lines_links(
        n, logger=lambda msg: None, return_fixed_count=True, exclude=absorbed)
    assert fixed == 0

    _, links, _ = build_store_and_merged_links(n, merge_links=["battery"],
                                               logger=lambda msg: None)
    assert list(links.index) == ["IT0 0 battery charger__IT0 0 battery discharger"]
    assert links["capital_cost"].iloc[0] == 100.0
