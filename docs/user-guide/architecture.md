# Architecture

## Architecture at a glance

The public entry point is `pypsa2smspp.Transformation`. A typical call is:

```python
import pypsa2smspp

tran = pypsa2smspp.Transformation()
n_smspp = tran.run(n)
```

Internally, `run()` is split into three stages:

```text
PyPSA Network
    |
    v
create_model()
    - validate the network and transformation options
    - read parameter metadata
    - build internal unit, network, investment, and stochastic data
    - assemble a pySMSpp SMSNetwork
    |
    v
optimize()
    - choose the SMS++ configuration template
    - write temporary SMS++ artifacts
    - call SMSNetwork.optimize()
    |
    v
retrieve_solution()
    - parse the SMS++ solution
    - rebuild PyPSA-compatible xarray variables
    - call PyPSA solution assignment
```

The main internal data structures are dictionaries that mirror the eventual SMS++ model:

- `unitblocks`: generator, storage, store, hydro, slack, and line/link unit blocks.
- `networkblock`: network-flow data, line/link mappings, and optional design-network data.
- `investmentblock`: investment variables and bounds when the model uses an outer `InvestmentBlock`.
- `dimensions`: SMS++ dimensions such as time horizon, number of units, nodes, lines, scenarios, and design variables.
- `tssb_data`: stochastic data for `TwoStageStochasticBlock` and `MultiStageStochasticBlock` models, including scenario sets and data mappings.

In future releases, this may change to leverage on the modularity of pysmspp.

## SMS++ block hierarchy

The generated SMS++ structure depends on the model type:

- Deterministic operational or capacity-expansion-in-UC mode:
  `SMSNetwork -> UCBlock`
- Deterministic investment mode:
  `SMSNetwork -> InvestmentBlock -> UCBlock`
- Stochastic mode:
  `SMSNetwork -> TwoStageStochasticBlock`, with `DiscreteScenarioSet`, `StaticAbstractPath`, and `StochasticBlock`; the stochastic block then contains the deterministic inner structure.
- Multi-stage stochastic mode:
  `SMSNetwork -> MultiStageStochasticBlock`, with a `ScenarioGenerator` holding the whole scenario tree, a `StaticAbstractPath` for the first-stage design, and one `TwoStageStochasticBlock` per outer-stage scenario. The inner blocks carry no `DiscreteScenarioSet` of their own: each of them reads its own scenarios from the shared tree, which is what ties the inner realizations to the outer branch they hang from.

### The multi-stage scenario tree

A PyPSA network carries a flat list of scenarios, which is all a two-stage
problem needs. A multi-stage problem needs to know more, namely which of those
scenarios descend from the same outer-stage realization and with which
conditional probability, and that is what turns the flat list into a tree. The
tree is therefore given from the outside, together with the stochastic
parameters:

```python
tran = pypsa2smspp.Transformation(
    stochastic_parameters={
        "stochastic_type": "mssb",
        "parameters": ["demand", "renewable_maxpower"],
        "tree": {
            "groups": {
                "dry":    {"probability": 0.3,
                           "scenarios": {"dry_low": 0.2, "dry_high": 0.8}},
                "wet":    {"probability": 0.7,
                           "scenarios": {"wet_low": 0.5, "wet_high": 0.5}},
            }
        },
    },
)
```

The leaves of the tree are the scenarios of the network, each of them exactly
once, the probabilities of the outer scenarios sum to one, and so do the
conditional probabilities of the inner ones of each outer scenario. The tree
has two levels, i.e. the problem has three stages: the design, the outer
realizations and the inner ones.

Note that the tree is what expresses the dependence on the history: two leaves
of different branches may well hold different values *and* different
probabilities, which is precisely what a `TwoStageStochasticBlock` over the
same leaves cannot say.

### Where the investment decision is stated

An investment decision taken before the uncertainty is revealed can be stated
in either of two ways, which describe the same problem. By default it is
replicated in every scenario, with the copies tied by the non-anticipativity
`Constraint` of the extensive form. With `investment_outside` it is instead
stated once, in an `InvestmentBlock` wrapping the whole stochastic `Block`:

```python
stochastic_parameters={
    "stochastic_type": "tssb",
    "parameters": ["demand"],
    "investment_outside": True,
}
```

which gives `SMSNetwork -> InvestmentBlock -> InnerBlock`, the latter being
the `TwoStageStochasticBlock` (or the `MultiStageStochasticBlock`) with the
scenarios below it. This is the arrangement a Benders decomposition of the
problem wants to see, the first-stage variables being the ones of the master.
It needs the investment to go through an `InvestmentBlock`, i.e.
`capacity_expansion_ucblock=False`.

Inside the `UCBlock`, pypsa2smspp adds:

- unit blocks for PyPSA components, such as `IntermittentUnitBlock`, `ThermalUnitBlock`, `BatteryUnitBlock`, `HydroUnitBlock`, and `SlackUnitBlock`;
- a network representation for lines and links, using SMS++ line variables and, when needed, a `DesignNetworkBlock`;
- active-power demand and other UC-level variables derived from the PyPSA network.

## Repository layout

The repository is organized around the transformation pipeline:

- `pypsa2smspp/transformation.py` contains the `Transformation` class, the pipeline orchestration, SMS++ block assembly, optimization call, and solution retrieval.
- `pypsa2smspp/transformation_config.py` defines the mapping from PyPSA attributes to SMS++ parameters, together with the inverse mappings used when rebuilding PyPSA variables from SMS++ results.
- `pypsa2smspp/utils.py` and `pypsa2smspp/constants.py` provide dimension handling, component filtering, line/link processing, nominal-attribute mappings, and parameter resolution helpers.
- `pypsa2smspp/stochastic_utils.py` builds stochastic metadata for `TwoStageStochasticBlock` and `MultiStageStochasticBlock` models, including scenario probabilities, demand and renewable scenario matrices, stochastic data mappings, and the scenario tree of a multi-stage problem.
- `pypsa2smspp/inverse.py` converts solved SMS++ unit blocks back into PyPSA-compatible `xarray.DataArray` objects.
- `pypsa2smspp/io_parser.py` parses SMS++ solution objects or textual outputs and prepares fake PyPSA model objects used by PyPSA's solution-assignment routines.
- `pypsa2smspp/network_correction.py` contains optional utilities for cleaning, simplifying, reducing, and comparing PyPSA networks before or after conversion.
- `pypsa2smspp/data/` contains default transformation and SMS++ parameter metadata used by the converter.
- `docs/examples/` and `test/` provide executable examples and regression tests for deterministic, investment, unit-commitment, PyPSA-Eur, and stochastic workflows.

This separation is intentional: `Transformation` coordinates the process, while configuration, preprocessing, stochastic modelling, SMS++ I/O, and inverse mapping remain in focused modules.
