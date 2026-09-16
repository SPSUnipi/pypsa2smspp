# -*- coding: utf-8 -*-
"""
A binding emission limit for the networks of the resilient instances.

The generators of the resilient instances remove the global constraints of the
PyPSA-Eur network they start from, and then add none. add_emission_limit()
puts back a primary energy limit on a carrier attribute (by default
co2_emissions), set to a fraction of the value that the unconstrained dispatch
gives to that same limit, so that the limit is binding whenever the fraction is
below 1.

The value of the unconstrained dispatch is read off the model PyPSA builds for
the limit itself, so that it is whatever PyPSA counts in it: the emissions of
the generators and the change of the level of the stores and storage units
whose carrier has the attribute (e.g., the store of the CO2 in the atmosphere
of a sector-coupled network), with the constants of their initial level. The
network is solved once, on a copy, with the limit set far from any dispatch.
"""

import numpy as np

LIMIT_NAME = "emission_limit"


def add_emission_limit(n, fraction, attribute="co2_emissions",
                       solver_name="highs", solver_options=None):
    """
    Add to n a primary energy limit on attribute at fraction of the value of
    the unconstrained dispatch, and return that value.

    Parameters
    ----------
    n : pypsa.Network
        A deterministic network (the limit is added before any scenario is
        set, and PyPSA then states it for each scenario).
    fraction : float
        The fraction of the value of the unconstrained dispatch the limit is
        set to.
    attribute : str
        The carrier attribute the limit is on.
    solver_name, solver_options
        The solver of the unconstrained dispatch.
    """
    if attribute not in n.carriers.columns or not n.carriers[attribute].any():
        raise ValueError(f"add_emission_limit: no carrier has {attribute}")

    # the limit is first stated with a constant no dispatch can reach, so that
    # it binds nothing: the model has the terms on the variables on the left,
    # and the constant minus the constants of the initial levels on the right
    loose = 1e12
    free = n.copy()
    free.add("GlobalConstraint", LIMIT_NAME, type="primary_energy",
             carrier_attribute=attribute, sense="<=", constant=loose)
    free.optimize(solver_name=solver_name, solver_options=solver_options)

    con = free.model.constraints[f"GlobalConstraint-{LIMIT_NAME}"]
    value = float(con.lhs.solution.sum()) + loose - float(con.rhs.item())
    if value <= 0:
        raise ValueError(f"add_emission_limit: the unconstrained dispatch has "
                         f"{attribute} {value}, no fraction of it is a limit")

    n.add("GlobalConstraint", LIMIT_NAME, type="primary_energy",
          carrier_attribute=attribute, sense="<=", constant=fraction * value)
    return value
