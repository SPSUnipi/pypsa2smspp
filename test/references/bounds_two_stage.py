"""The two values the optimum of the two-decision tree has to sit between:
deciding everything at the root, which is what a two-stage problem can say,
and deciding everything knowing the climate at the root price, which is the
wait-and-see relaxation. The distance between them is what the second decision
stage is worth."""
import json, sys, warnings
from pathlib import Path

warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "output" / "mssb_tree"
DATA.mkdir(parents=True, exist_ok=True)
import pypsa

name = sys.argv[1] if len(sys.argv) > 1 else "pypsa_2stage_c3_d3_b1_t1"
tree = json.load(open(DATA / f"{name}_tree.json"))

root = pypsa.Network(str(DATA / f"{name}_flat.nc"))
root.optimize(solver_name="gurobi")
upper = float(root.objective + root.objective_constant)
print(f"tutto alla radice        = {upper:.6f}")

lower = 0.0
for branch, data in tree["groups"].items():
    net = pypsa.Network(str(DATA / f"{name}_{branch}.nc"))
    net.optimize(solver_name="gurobi")
    value = float(net.objective + net.objective_constant)
    lower += data["probability"] * value
    print(f"  ramo {branch} (p = {data['probability']:.4f}) = {value:.6f}")
print(f"tutto dopo il clima      = {lower:.6f}")
print(f"valore della seconda decisione, al piu' = {upper - lower:.6f}")
