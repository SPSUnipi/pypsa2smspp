# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 18:10:52 2026

@author: aless
"""

def get_base_scenario_network(n):
    """Return a deterministic network for direct conversion.

    If the input network is stochastic, return the first scenario.
    Otherwise, return the original network unchanged.
    """
    has_scenarios = getattr(n, "has_scenarios", False)

    if has_scenarios:
        scenarios = list(n.scenarios)
        if not scenarios:
            raise ValueError("The network is marked as stochastic but has no scenarios.")
        return n.get_scenario(scenarios[0])

    return n