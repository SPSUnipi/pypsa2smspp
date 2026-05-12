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
- `tssb_data`: stochastic data for `TwoStageStochasticBlock` models, including scenario sets and data mappings.

In future releases, this may change to leverage on the modularity of pysmspp.

## SMS++ block hierarchy

The generated SMS++ structure depends on the model type:

- Deterministic operational or capacity-expansion-in-UC mode:
  `SMSNetwork -> UCBlock`
- Deterministic investment mode:
  `SMSNetwork -> InvestmentBlock -> UCBlock`
- Stochastic mode:
  `SMSNetwork -> TwoStageStochasticBlock`, with `DiscreteScenarioSet`, `StaticAbstractPath`, and `StochasticBlock`; the stochastic block then contains the deterministic inner structure.

Inside the `UCBlock`, pypsa2smspp adds:

- unit blocks for PyPSA components, such as `IntermittentUnitBlock`, `ThermalUnitBlock`, `BatteryUnitBlock`, `HydroUnitBlock`, and `SlackUnitBlock`;
- a network representation for lines and links, using SMS++ line variables and, when needed, a `DesignNetworkBlock`;
- active-power demand and other UC-level variables derived from the PyPSA network.

## Repository layout

The repository is organized around the transformation pipeline:

- `pypsa2smspp/transformation.py` contains the `Transformation` class, the pipeline orchestration, SMS++ block assembly, optimization call, and solution retrieval.
- `pypsa2smspp/transformation_config.py` defines the mapping from PyPSA attributes to SMS++ parameters, together with the inverse mappings used when rebuilding PyPSA variables from SMS++ results.
- `pypsa2smspp/utils.py` and `pypsa2smspp/constants.py` provide dimension handling, component filtering, line/link processing, nominal-attribute mappings, and parameter resolution helpers.
- `pypsa2smspp/stochastic_utils.py` builds stochastic metadata for `TwoStageStochasticBlock` models, including scenario probabilities, demand and renewable scenario matrices, and stochastic data mappings.
- `pypsa2smspp/inverse.py` converts solved SMS++ unit blocks back into PyPSA-compatible `xarray.DataArray` objects.
- `pypsa2smspp/io_parser.py` parses SMS++ solution objects or textual outputs and prepares fake PyPSA model objects used by PyPSA's solution-assignment routines.
- `pypsa2smspp/network_correction.py` contains optional utilities for cleaning, simplifying, reducing, and comparing PyPSA networks before or after conversion.
- `pypsa2smspp/data/` contains default transformation and SMS++ parameter metadata used by the converter.
- `docs/examples/` and `test/` provide executable examples and regression tests for deterministic, investment, unit-commitment, PyPSA-Eur, and stochastic workflows.

This separation is intentional: `Transformation` coordinates the process, while configuration, preprocessing, stochastic modelling, SMS++ I/O, and inverse mapping remain in focused modules.
