# Cross-check: the flat equivalent of the two-level tree, solved by PyPSA and
# by SMS++ through the existing TwoStageStochasticBlock path. Since the only
# here-and-now variables are the design ones, this same number is what the
# MultiStageStochasticBlock built on the tree has to reproduce.
import os, sys, warnings
from pathlib import Path

warnings.simplefilter("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "output" / "mssb_tree"
DATA.mkdir(parents=True, exist_ok=True)          # .../test/output/mssb_tree
TEST = HERE.parents[1]                          # .../test
os.chdir(TEST)
sys.path.insert(0, str(TEST))

import pypsa
from pypsa2smspp.transformation import Transformation

NET = DATA / "pypsa_ec_2level_flat.nc"
WORK = DATA / "smspp"
WORK.mkdir(parents=True, exist_ok=True)

n = pypsa.Network(str(NET))

n_ref = n.copy()
n_ref.optimize(solver_name="gurobi")
obj_pypsa = float(n_ref.objective + n_ref.objective_constant)
print(f"[PyPSA]  objective = {obj_pypsa:.6f}")

t = Transformation(
    name="mssb_flat",
    configfile="TSSBlock/TSSBSCfg.txt",
    enable_thermal_units=False,
    workdir=str(WORK),
    stochastic_parameters={"stochastic_type": "tssb",
                           "parameters": ["demand", "renewable_maxpower"]},
    overwrite=True,
    fp_temp="smspp_{name}_temp.nc",
    fp_log="smspp_{name}_log.txt",
    fp_solution="smspp_{name}_solution.nc",
    pysmspp_options={"B": "TSSBCfg.txt"},
)
n = t.run(n, verbose=False)
obj_smspp = float(t.result.objective_value)
print(f"[SMS++]  objective = {obj_smspp:.6f}")
print(f"[ERROR]  = {(obj_smspp - obj_pypsa) / obj_pypsa * 100.0:.6f} %")
