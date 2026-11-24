# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import pypsa
import pytest

from pypsa2smspp.inverse import (
    block_to_dataarrays,
    block_to_dataarrays_stochastic,
    dataarray_components,
)
from pypsa2smspp.transformation_config import TransformationConfig

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


def _pump_network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range('2020-01-01', periods=3, freq='h'))
    n.add('Bus', 'in')
    n.add('Bus', 'out')
    n.add('Link', 'pump', bus0='in', bus1='out', efficiency=1.,
          p_nom=10., efficiency2=1.)
    n.links_t.efficiency['pump'] = [0.25, 0.5, 0.75]
    return n


def _pump_block():
    return dict(name='pump', block='DCNetworkBlock_links',
                FlowValue=np.array([4., 6., 8.]), DesignVariable=np.array([42.]))


def test_inverse_uses_dynamic_efficiency_and_static_capacity():
    n = _pump_network()
    result = block_to_dataarrays(n, 'pump', _pump_block(), 'Link', TransformationConfig())
    np.testing.assert_allclose(result['Link-p1'].values[:, 0], [-1., -3., -6.])
    assert result['Link-p_nom'].dims == ('name',)
    assert result['Link-p_nom'].item() == 42.


def test_stochastic_inverse_uses_each_scenario_efficiency():
    n = _pump_network()
    n.set_scenarios({'low': 0.5, 'high': 0.5})
    n.links_t.efficiency.loc[:, ('high', 'pump')] = [0.5, 0.6, 0.7]
    b = _pump_block()
    b['scenarios'] = {s: dict(FlowValue=b['FlowValue'], DesignVariable=42.) for s in n.scenarios}
    result = block_to_dataarrays_stochastic(n, 'pump', b, 'Link', TransformationConfig(),
        {'scenario_names': list(n.scenarios)}, block_to_dataarrays)
    np.testing.assert_allclose(result['Link-p1'].sel(scenario='low').values[:, 0], [-1., -3., -6.])
    np.testing.assert_allclose(result['Link-p1'].sel(scenario='high').values[:, 0], [-2., -3.6, -5.6])
    assert 'snapshot' not in result['Link-p_nom'].dims


def test_repeated_nominal_capacity_stays_static():
    n = _pump_network()
    _, dims, _, _ = dataarray_components(n, np.full(3, 10.), 'Link', _pump_block(), 'p_nom')
    assert dims == ['name']
    with pytest.raises(ValueError, match='varies in time'):
        dataarray_components(n, [10., 11., 12.], 'Link', _pump_block(), 'p_nom')


def test_fixed_link_capacity_is_not_its_dynamic_availability():
    n = _pump_network()
    n.links_t.p_max_pu['pump'] = [0.2, 0.8, 1.]
    t = Transformation()
    t.create_model(n, verbose=False)
    t.networkblock['Lines']['FlowValue'] = np.zeros((3, 1))
    t.generate_line_unitblocks(n)
    b = next(b for b in t.unitblocks.values() if b.get('name') == 'pump')
    assert b['DesignVariable'] == 10.


def _biomass_network(hours=4.):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2020-01-01", periods=2, freq="h"))
    n.snapshot_weightings["generators"] = hours
    n.snapshot_weightings["objective"] = 2.
    n.snapshot_weightings["stores"] = 3.
    n.add("Bus", "bus")
    n.add("Generator", "biomass", bus="bus", carrier="solid biomass",
          p_nom=10., e_sum_min=8., e_sum_max=20.)
    n.add("Load", "demand", bus="bus", p_set=5.)
    return n


@pytest.mark.parametrize("weights", [[1., 2.], [0., 0.], [-1., -1.], [np.nan, np.nan]])
def test_unsupported_weights_are_not_silently_mistranslated(weights):
    n = _biomass_network()
    n.snapshot_weightings["generators"] = weights
    with pytest.raises(ValueError, match="snapshot_weightings.generators"):
        Transformation().create_model(n, verbose=False)


def test_nonuniform_weights_without_energy_limits():
    n = _biomass_network()
    n.snapshot_weightings["generators"] = [1., 2.]
    n.generators["e_sum_min"] = -np.inf
    n.generators["e_sum_max"] = np.inf
    Transformation().create_model(n, verbose=False)
