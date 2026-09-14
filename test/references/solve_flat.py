import warnings, pypsa
warnings.filterwarnings("ignore")
n = pypsa.Network("pypsa_ec_2level_flat.nc")
print("scenarios:", len(n.scenarios), "snapshots:", len(n.snapshots))
solver = "gurobi"
try:
    n.optimize(solver_name=solver)
except Exception as e:
    print("gurobi failed:", e); solver = "highs"; n.optimize(solver_name=solver)
print("SOLVER:", solver)
print("OBJECTIVE: %.10e" % n.objective)
print("design (p_nom_opt):")
print(n.generators.p_nom_opt.groupby(level=-1).first() if n.has_scenarios else n.generators.p_nom_opt)
