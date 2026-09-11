import warnings, pypsa
warnings.filterwarnings("ignore")
n = pypsa.Network("pypsa_ec_2level_flat.nc")
n.optimize(solver_name="gurobi")
print("objective          = %.6f" % n.objective)
print("objective_constant = %.6f" % n.objective_constant)
print("TOTAL (reference)  = %.6f" % (n.objective + n.objective_constant))
