"""Emit the tree with the second decision stage as a MultiStageStochasticBlock:
the outer level ties the design decided at the root, the inner ones tie those
plus the ones decided once the climate is known."""
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

name = sys.argv[1] if len(sys.argv) > 1 else "pypsa_2stage_c3_d3_b1_t1"
tree = json.load(open(DATA / f"{name}_tree.json"))
WORK = DATA / "smspp_2stage"
WORK.mkdir(parents=True, exist_ok=True)

n = pypsa.Network(str(DATA / f"{name}_full.nc"))

t = Transformation(
    name=name,
    configfile="TSSBlock/TSSBSCfg.txt",
    enable_thermal_units=False,
    workdir=str(WORK),
    stochastic_parameters={"stochastic_type": "mssb",
                           "parameters": ["demand", "renewable_maxpower"],
                           "tree": {"groups": tree["groups"]},
                           "design_stages": tree["design_stages"]},
    overwrite=True,
    fp_temp="smspp_{name}_temp.nc",
)
t.create_model(n, verbose=False)
t.sms_network.to_netcdf(str(WORK / f"smspp_{name}_temp.nc"), force=True)
print("[emesso ]", WORK / f"smspp_{name}_temp.nc")
print("[radice ]", tree["design_stages"]["root"])
print("[dopo   ]", tree["design_stages"]["branch"])
