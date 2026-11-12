import numpy as np
import pandas as pd
import pypsa
import pytest

from pypsa2smspp.inverse import block_to_dataarrays, block_to_dataarrays_stochastic, dataarray_components
from pypsa2smspp import Transformation
from pypsa2smspp.transformation_config import TransformationConfig


def network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range('2020-01-01', periods=3, freq='h'))
    n.add('Bus', 'in')
    n.add('Bus', 'out')
    n.add('Link', 'pump', bus0='in', bus1='out', efficiency=1.,
          p_nom=10., efficiency2=1.)
    n.links_t.efficiency['pump'] = [0.25, 0.5, 0.75]
    return n


def block():
    return dict(name='pump', block='DCNetworkBlock_links',
                FlowValue=np.array([4., 6., 8.]), DesignVariable=np.array([42.]))


def test_inverse_uses_dynamic_efficiency_and_static_capacity():
    n = network()
    result = block_to_dataarrays(n, 'pump', block(), 'Link', TransformationConfig())
    np.testing.assert_allclose(result['Link-p1'].values[:, 0], [-1., -3., -6.])
    assert result['Link-p_nom'].dims == ('name',)
    assert result['Link-p_nom'].item() == 42.


def test_stochastic_inverse_uses_each_scenario_efficiency():
    n = network()
    n.set_scenarios({'low': 0.5, 'high': 0.5})
    n.links_t.efficiency.loc[:, ('high', 'pump')] = [0.5, 0.6, 0.7]
    b = block()
    b['scenarios'] = {s: dict(FlowValue=b['FlowValue'], DesignVariable=42.) for s in n.scenarios}
    result = block_to_dataarrays_stochastic(n, 'pump', b, 'Link', TransformationConfig(),
        {'scenario_names': list(n.scenarios)}, block_to_dataarrays)
    np.testing.assert_allclose(result['Link-p1'].sel(scenario='low').values[:, 0], [-1., -3., -6.])
    np.testing.assert_allclose(result['Link-p1'].sel(scenario='high').values[:, 0], [-2., -3.6, -5.6])
    assert 'snapshot' not in result['Link-p_nom'].dims


def test_repeated_nominal_capacity_stays_static():
    n = network()
    _, dims, _, _ = dataarray_components(n, np.full(3, 10.), 'Link', block(), 'p_nom')
    assert dims == ['name']
    with pytest.raises(ValueError, match='varies in time'):
        dataarray_components(n, [10., 11., 12.], 'Link', block(), 'p_nom')


def test_fixed_link_capacity_is_not_its_dynamic_availability():
    n = network()
    n.links_t.p_max_pu['pump'] = [0.2, 0.8, 1.]
    t = Transformation()
    t.create_model(n, verbose=False)
    t.networkblock['Lines']['FlowValue'] = np.zeros((3, 1))
    t.generate_line_unitblocks(n)
    b = next(b for b in t.unitblocks.values() if b.get('name') == 'pump')
    assert b['DesignVariable'] == 10.
