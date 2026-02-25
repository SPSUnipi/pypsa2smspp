# pypsa2smspp

[![Tests](https://github.com/SPSUnipi/pypsa2smspp/actions/workflows/test.yml/badge.svg)](https://github.com/SPSUnipi/pypsa2smspp/actions/workflows/test.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/SPSUnipi/pypsa2smspp/main.svg)](https://results.pre-commit.ci/latest/github/SPSUnipi/pypsa2smspp/main)
 
This package aims at providing a python interface between [PyPSA](https://github.com/PyPSA/pypsa) and [SMS++](https://gitlab.com/smspp/smspp-project) models using a simple python interface.
The package aims to support:
- Convert a PyPSA model to SMS++
- Execute the optimization of the so-created SMS++ model
- Parse the solution from the SMS++ model to PyPSA


## How to develop

1. First, clone the repository using git:

    ```bash
        git clone https://github.com/SPSUnipi/pypsa2smspp
    ```

2. Create a virtual environment using venv or conda.
    For exaample, using venv:

    ```bash
        python -m venv .venv
        source .venv/bin/activate
    ```
   
    Alternatively, using conda:

    ```bash
        conda create -n pypsa2smspp python=3.10
        conda activate pypsa2smspp
    ```

3. Install the required packages and pre-commit hooks:

    ```bash
        pip install -e .[dev]
        pre-commit install
    ```

    Note that the `-e` command line option installs the package in editable mode, so that changes to the source code are immediately available in the environment being used. The `[dev]` option installs the packages required for development. The `pre-commit install` command installs the pre-commit hooks, which are used to check the code before committing to ensure code quality standards.

4. Develop and test the code. For testing, please run:

    ```bash
        pytest
    ```
