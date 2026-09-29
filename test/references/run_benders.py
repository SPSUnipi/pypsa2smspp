"""Solve one generated instance both ways: the flat network in PyPSA, and the
scenario tree as a MultiStageStochasticBlock in SMS++, and compare."""
import argparse, json, os, sys, time, warnings
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

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--solver", default="gurobi")
parser.add_argument("--parameters", default="demand,renewable_maxpower",
                    help="the stochastic parameters the tree perturbs")
parser.add_argument("--convert-only", action="store_true",
                    help="only write the SMS++ instance, solve neither way")
args = parser.parse_args()

net = DATA / f"{args.name}_flat.nc"
tree = json.load(open(DATA / f"{args.name}_tree.json"))
work = DATA / "smspp" / args.name
work.mkdir(parents=True, exist_ok=True)

n = pypsa.Network(str(net))
print(f"instance {args.name}: {len(n.scenarios)} leaves, "
      f"{len(n.snapshots)} snapshots, {len(n.buses)} buses, "
      f"{len(n.generators.groupby(level=-1).first())} generators")

if args.convert_only:
    t = Transformation(
        name=args.name,
        configfile="TSSBlock/TSSBSCfg.txt",
        enable_thermal_units=False,
    capacity_expansion_ucblock=False,
        workdir=str(work),
        stochastic_parameters={"stochastic_type": "mssb",
                               "parameters": args.parameters.split( "," ),
                               "tree": {"groups": tree["groups"]},
                               "investment_outside": True},
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
    )
    start = time.time()
    t.create_model(n, verbose=False)
    t.sms_network.to_netcdf(str(work / f"smspp_{args.name}_temp.nc"),
                            force=True)
    print(f"[convert   ] written in {time.time() - start:.1f} s")
    sys.exit(0)

start = time.time()
reference = n.copy()
reference.optimize(solver_name=args.solver)
flat = float(reference.objective + reference.objective_constant)
print(f"[PyPSA flat] objective = {flat:.6f}  ({time.time() - start:.1f} s)")

start = time.time()
t = Transformation(
    name=args.name,
    configfile="TSSBlock/TSSBSCfg.txt",
    enable_thermal_units=False,
    capacity_expansion_ucblock=False,
    workdir=str(work),
    stochastic_parameters={"stochastic_type": "mssb",
                           "parameters": args.parameters.split( "," ),
                           "tree": {"groups": tree["groups"]},
                               "investment_outside": True},
    overwrite=True,
    fp_temp="smspp_{name}_temp.nc",
    fp_log="smspp_{name}_log.txt",
    fp_solution="smspp_{name}_solution.nc",
    pysmspp_options={"B": "TSSBCfg.txt"},
)
t.run(n, verbose=False)
mssb = float(t.result.objective_value)
print(f"[SMS++ MSSB] objective = {mssb:.6f}  ({time.time() - start:.1f} s)")
print(f"[ERROR     ] = {(mssb - flat) / flat * 100.0:.6f} %")
