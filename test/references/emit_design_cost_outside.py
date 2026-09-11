"""The form a generic Benders solver asks for: the design Variable stay in the
units, one copy per scenario, and their cost is stated once in the
InvestmentBlock above them, so that the value of a scenario is monotone in the
design. Writes the instance and, for reference, solves the network in PyPSA."""
import os, sys, warnings
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

NET = TEST / "networks" / "pypsa_stoch_load.nc"
WORK = DATA / "smspp_dco"
WORK.mkdir(parents=True, exist_ok=True)

n = pypsa.Network(str(NET))

ref = n.copy()
ref.optimize(solver_name="gurobi")
print(f"[PyPSA] objective = {float(ref.objective + ref.objective_constant):.6f}")

t = Transformation(
    name="dco",
    configfile="TSSBlock/TSSBSCfg.txt",
    enable_thermal_units=False,
    capacity_expansion_ucblock=True,
    workdir=str(WORK),
    stochastic_parameters={"stochastic_type": "tssb",
                           "parameters": ["demand"],
                           "investment_outside": True,
                           "design_cost_outside": True},
    overwrite=True,
    fp_temp="smspp_{name}_temp.nc",
)
t.create_model(n, verbose=False)
t.sms_network.to_netcdf(str(WORK / "smspp_dco_temp.nc"), force=True)
print("[emesso ]", WORK / "smspp_dco_temp.nc")
