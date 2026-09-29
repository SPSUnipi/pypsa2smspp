# The two-level tree instance emitted as a MultiStageStochasticBlock, and
# solved by SMS++. It has to reproduce the optimum of its flat equivalent.
import json, os, sys, warnings
from pathlib import Path

warnings.simplefilter("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "output" / "mssb_tree"
DATA.mkdir(parents=True, exist_ok=True)
TEST = HERE.parents[1]
os.chdir(TEST)
sys.path.insert(0, str(TEST))

import pypsa
from pypsa2smspp.transformation import Transformation

NET = DATA / "pypsa_ec_2level_flat.nc"
TREE = json.load(open(DATA / "pypsa_ec_2level_tree.json"))
WORK = DATA / "smspp_mssb"
WORK.mkdir(parents=True, exist_ok=True)

n = pypsa.Network(str(NET))

t = Transformation(
    name="mssb_tree",
    configfile="TSSBlock/TSSBSCfg.txt",
    enable_thermal_units=False,
    workdir=str(WORK),
    stochastic_parameters={"stochastic_type": "mssb",
                           "parameters": ["demand", "renewable_maxpower"],
                           "tree": {"groups": TREE["groups"]}},
    overwrite=True,
    fp_temp="smspp_{name}_temp.nc",
    fp_log="smspp_{name}_log.txt",
    fp_solution="smspp_{name}_solution.nc",
    pysmspp_options={"B": "TSSBCfg.txt"},
)
n = t.run(n, verbose=False)
obj = float(t.result.objective_value)
ref = 2112415.084977
print(f"[SMS++ MSSB] objective = {obj:.6f}")
print(f"[reference ] objective = {ref:.6f}")
print(f"[ERROR     ] = {(obj - ref) / ref * 100.0:.6f} %")
