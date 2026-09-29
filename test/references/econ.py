import warnings, pypsa
warnings.filterwarnings("ignore")
n = pypsa.Network("pypsa_ec_2level_flat.nc")
g = n.generators.groupby(level=-1).first()
print(g[["carrier","p_nom","p_nom_extendable","p_nom_max","marginal_cost","capital_cost"]])
print("peak load:", float(n.loads_t.p_set.max().max()), " mean:", float(n.loads_t.p_set.mean().mean()))
print("solar profile sum over horizon:", float(n.generators_t.p_max_pu.iloc[:,0].sum()))
