# Introduction

## What is pypsa2smspp?

pypsa2smspp is a Python package that connects [PyPSA](https://github.com/PyPSA/pypsa) energy-system models with [SMS++](https://smspp.gitlab.io/smspp-project/), using the [pySMSpp](https://github.com/SPSUnipi/pySMSpp) Python interface.

PyPSA remains the user-facing modelling environment: networks, components, time series, and optimized results are represented as PyPSA objects. SMS++ provides the block-structured mathematical model and the solvers that can exploit that structure. pypsa2smspp sits between them: it converts a PyPSA network into an SMS++ block hierarchy, runs the optimization through pySMSpp, and maps the solution back onto the original PyPSA network.

The package is under active development. The current documentation is intended both for users who want to run conversions and for developers who want to understand and extend the transformation pipeline. There is not yet a stable release on PyPI or Conda, so the package is currently installed by cloning the repository and installing it locally.

## Why pypsa2smspp?

PyPSA is a flexible framework for defining and optimizing energy-system models in Python. SMS++ is designed for advanced optimization of block-structured mathematical models, including decomposition methods and specialized solvers.

pypsa2smspp combines these strengths. Users can continue to build networks in PyPSA while experimenting with SMS++ formulations such as `UCBlock`, `InvestmentBlock`, `DesignNetworkBlock`, and `TwoStageStochasticBlock`. This is especially useful for capacity expansion, unit-commitment-style formulations, network design, and stochastic problems where the mathematical structure matters.

## What is SMS++?

SMS++ is a C++ framework for modelling and solving complex block-structured optimization problems. A model is represented as a hierarchy of blocks; each block can contain variables, constraints, objectives, data, and nested sub-blocks.

This block structure is important because it lets solvers exploit the mathematical organization of the problem. For example, different units, networks, investment variables, and scenario-dependent data can be represented as separate but connected blocks. SMS++ can then use decomposition-oriented algorithms or specialized solvers that are aware of this structure.

For more information, see the [SMS++ website](https://smspp.gitlab.io/smspp-project/).
