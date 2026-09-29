"""Reference value of a `--no-design` instance of gen_thermal_tssb.py.

With no expandable capacity the two-stage problem has no here-and-now
variable, so the scenarios decouple and the optimum of the deterministic
equivalent is the weighted sum of the per-scenario unit-commitment optima;
PyPSA solves those one at a time, which is the only way round its inability to
optimize a network that is at once stochastic and committable.

    python crosscheck_thermal_tssb.py tuc_u8_t24_s3_b1_nd [gurobi]
"""

import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "output" / "thermal_tssb"

import pypsa

name = sys.argv[1] if len(sys.argv) > 1 else "tuc_u8_t24_s3_b1_nd"
solver = sys.argv[2] if len(sys.argv) > 2 else "gurobi"

stochastic = pypsa.Network(str(DATA / f"{name}_flat.nc"))
weightings = stochastic.scenario_weightings["weight"]

# the sum of the per-scenario optima is the optimum of the two-stage problem
# only as long as nothing ties the scenarios together
expandable = sum(int(frame[column].any())
                 for frame, column in ((stochastic.generators, "p_nom_extendable"),
                                       (stochastic.storage_units, "p_nom_extendable"),
                                       (stochastic.stores, "e_nom_extendable"),
                                       (stochastic.links, "p_nom_extendable"),
                                       (stochastic.lines, "s_nom_extendable"))
                 if not frame.empty)
if expandable:
    raise SystemExit(f"{name} has expandable capacity, hence here-and-now "
                     "variables: its scenarios do not decouple and this "
                     "reference does not apply; generate it with --no-design")

total = 0.0
for scenario in stochastic.scenarios:
    weight = float(weightings.loc[scenario])
    n = stochastic.get_scenario(scenario)
    n.optimize(solver_name=solver, linearized_unit_commitment=False)
    value = float(n.objective + n.objective_constant)
    total += weight * value
    print(f"[{scenario}] weight {weight:.6f}  objective {value:.6f}")

print(f"[reference] {total:.6f}")
