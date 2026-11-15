"""Prepara la rete 'solo operativa' (capacita' fissate al valore ottimo
dell'espansione sullo scenario 1, costi capitali nulli, idro/PHS non ciclici
con volume iniziale = meta' del massimo, batteria IT0 a capacita' zero)."""
import logging, pypsa, numpy as np
logging.disable(logging.WARNING)

n = pypsa.Network("rete_inflow.nc")
d = n.get_scenario("scenario_001")
d.optimize(solver_name="highs")
print("espansione, obiettivo", d.objective + d.objective_constant)

def fix(n, d):
    for comp, nom, ext in (("generators", "p_nom", "p_nom_extendable"),
                           ("links", "p_nom", "p_nom_extendable"),
                           ("stores", "e_nom", "e_nom_extendable"),
                           ("storage_units", "p_nom", "p_nom_extendable"),
                           ("lines", "s_nom", "s_nom_extendable")):
        df = getattr(d, comp)
        if len(df) == 0:
            continue
        opt = df[nom + "_opt"]
        tgt = getattr(n, comp)
        if tgt.index.nlevels == 2:
            for s in n.scenarios:
                tgt.loc[(s, df.index), nom] = opt.values
        else:
            tgt[nom] = opt
        tgt[ext] = False
        if "capital_cost" in tgt:
            tgt["capital_cost"] = 0.0

fix(n, d); fix(d, d)
for net in (n, d):
    su = net.storage_units
    h = su.carrier.isin(["hydro", "PHS"])
    su.loc[h, "cyclic_state_of_charge"] = False
    su.loc[h, "state_of_charge_initial"] = (0.5 * su.p_nom * su.max_hours)[h]
    st = net.stores
    idx = [i for i in st.index if (i[-1] if isinstance(i, tuple) else i) == "IT0 0 battery"]
    st.loc[idx, "e_nom"] = 0.0
n.export_to_netcdf("rete_operativa.nc")
d.export_to_netcdf("rete_operativa_det.nc")
d.optimize(solver_name="highs")
print("operativa, scenario 1:", d.objective + d.objective_constant)
print(d.storage_units[["p_nom", "max_hours", "state_of_charge_initial"]])
print(d.storage_units_t.inflow.sum())
