"""Generator horizon limits must retain PyPSA's energy weighting."""
import numpy as np
import pandas as pd
import pypsa
import pytest

from pypsa2smspp import Transformation


def network(hours=4.):
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


@pytest.mark.parametrize("ucblock", [True, False])
@pytest.mark.parametrize("hours", [1., 4., 8760./576.])
def test_energy_limits_in_converted_block(ucblock, hours):
    n = network(hours)
    t = Transformation(capacity_expansion_ucblock=ucblock)
    t.create_model(n, verbose=False)
    b = next(b for b in t.unitblocks.values() if b.get("name") == "biomass")
    lower = np.asarray(b["variables"]["MinGeneration"]["value"]).item()
    upper = np.asarray(b["variables"]["MaxGeneration"]["value"]).item()
    assert lower == pytest.approx(8./hours)
    assert upper == pytest.approx(20./hours)
    # A candidate at the physical annual cap is accepted, an excess rejected.
    at_limit = np.full(2, 10./hours)
    assert at_limit.sum() == pytest.approx(upper)
    assert (at_limit * 1.1).sum() > upper


def test_stochastic_energy_limits_are_scaled_once():
    n = network()
    n.set_scenarios({"low": .5, "high": .5})
    t = Transformation(stochastic_parameters={"stochastic_type": "tssb", "parameters": ["demand"]})
    t.create_model(n, verbose=False)
    b = next(b for b in t.unitblocks.values() if b.get("name") == "biomass")
    assert np.asarray(b["variables"]["MaxGeneration"]["value"]).item() == 5.


@pytest.mark.parametrize("weights", [[1., 2.], [0., 0.], [-1., -1.], [np.nan, np.nan]])
def test_unsupported_weights_are_not_silently_mistranslated(weights):
    n = network()
    n.snapshot_weightings["generators"] = weights
    with pytest.raises(ValueError, match="snapshot_weightings.generators"):
        Transformation().create_model(n, verbose=False)


def test_nonuniform_weights_without_energy_limits():
    n = network()
    n.snapshot_weightings["generators"] = [1., 2.]
    n.generators["e_sum_min"] = -np.inf
    n.generators["e_sum_max"] = np.inf
    Transformation().create_model(n, verbose=False)
