# Introduction

## What is pypsa2smspp?

pypsa2smspp is a Python package to enable advanced mathematical decomposition in the energy modeling framework [PyPSA](https://github.com/PyPSA/pypsa). This package interfaces [PyPSA](https://github.com/PyPSA/pypsa) using [Structured Modeling System for mathematical models (SMS++)](https://smspp.gitlab.io/smspp-project/) and its python interface [pySMSpp](https://github.com/SPSUnipi/pySMSpp).

The package is under active development and the current documentation aims to facilitate the development of the package. Currently, there is not yet a stable release of the package on PyPI or Conda, so the installation can be performed by cloning the repository and installing the package locally.

## What is SMS++?

SMS++ is a software tool for advanced optimization of mathematical models by adopting advanced decomposition tools.
It is a collection of C++ classes for modeling complex, block-structured mathematical models and solving them via sophisticated, structure-exploiting algorithms such as decomposition methods and specialized Interior-Point approaches.

SMS++ preserves the block-structure of the model and allows the user to define the model in terms of blocks, which can be solved by different solvers. Each block may describe a specific physical system to model (e.g. a generator of a power system) or a specific mathematical structure (e.g. to allow Lagrangian relaxation). Each block may be characterized by a set of variables, constraints, and objectives, and may be solved by highly specialized solvers and decomposition techniques. As each block is solved by a specialized solver, the overall solution process is highly efficient and can exploit the structure of the model.

SMS++ supports a hierarchical model structure, where blocks may contain other blocks. This allows the user to define complex models in a modular way, where each block can be solved by a specialized solver. The nested structure combined with the specialized solvers allows the user to exploit the structure of the model and aims to break down computational resources to solve large-scale models.

For more information about SMS++, please refer to the [SMS++ website](https://smspp.gitlab.io/smspp-project/).

## Why pypsa2smspp?

As mentioned above SMS++ is a powerful tool for solving complex mathematical models. On the other hand, PyPSA is a powerful tool for modeling and optimizing energy systems. By interfacing PyPSA with SMS++, we can leverage the power of SMS++ to solve complex energy system models defined in PyPSA. This allows us to solve larger and more complex models than what is possible with the default solvers available in PyPSA, and to exploit the structure of the model to achieve faster solution times.

By leveraging on [pySMSpp](https://pysmspp.readthedocs.io/en/latest/), that is the Python interface to SMS++, pypsa2smspp allows to convert PyPSA models into SMS++ objects in Python, and to write, read and optimize SMS++ models. This allows us to leverage the power of SMS++ while still being able to define the model in Python, which is a widely used programming language in the energy modeling community.
