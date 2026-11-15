import logging, numpy as np, pandas as pd, pypsa
logging.disable(logging.WARNING)
T, NS = 100, 4            # snapshot, stadi
L = T // NS               # snapshot per stadio
HYDRO = ["IT0 0 hydro", "IT1 0 hydro"]
PHS = ["IT0 0 PHS", "IT1 0 PHS"]

def load_det(max_hours=None):
    d = pypsa.Network("rete_operativa_det.nc")
    set_hours(d, max_hours)
    return d

def set_hours(d, max_hours):
    """riduce i serbatoi idro (non PHS) a max_hours ore di potenza nominale"""
    if max_hours is None:
        return
    su = d.storage_units
    su.loc[HYDRO, "max_hours"] = max_hours
    h = su.carrier.isin(["hydro", "PHS"])
    su.loc[h, "state_of_charge_initial"] = (0.5 * su.p_nom * su.max_hours)[h]

def stage_of(t):
    return np.arange(T)[t] // L

def inflow_traj(base, f):
    """base: DataFrame (T x hydro); f: fattore per stadio (len NS)"""
    return base * np.repeat(np.asarray(f, float), L)[:, None]

def opex_by_snapshot(n):
    o = n.statistics.opex(aggregate_time=False)
    return o.sum(axis=0) if isinstance(o, pd.DataFrame) else o

def stage_cycle_constraints(n, m, first_stage):
    """store ciclici per stadio (stessa ipotesi di SMS++): e[fine stadio j] = e[fine stadio j-1]"""
    snaps = n.snapshots
    cyc = n.stores.index[n.stores.e_cyclic]
    if len(cyc) == 0:
        return
    e = m.variables["Store-e"]
    for j in range(first_stage + 1, NS):
        a = (j - first_stage) * L - 1      # fine stadio precedente (indice nella finestra)
        b = a + L
        for s in cyc:
            m.add_constraints(e.loc[snaps[b], s] - e.loc[snaps[a], s] == 0,
                              name=f"stagecyc_{j}_{s}".replace(" ", "_"))

def solve(d, inflow, t0=0, soc0=None, tend=T):
    """risolve le snapshot [t0*L, tend) con afflussi `inflow` (DataFrame T x hydro);
    soc0: volume iniziale (Series su storage_units). Ritorna (costo totale, n)"""
    n = d.copy()
    sl = n.snapshots[t0 * L:tend]
    n.set_snapshots(sl)
    n.storage_units_t.inflow.loc[:, HYDRO] = inflow.iloc[t0 * L:tend].values
    if soc0 is not None:
        n.storage_units["state_of_charge_initial"] = soc0
    first = t0
    n.optimize(solver_name="highs",
               extra_functionality=lambda n_, s_: stage_cycle_constraints(n_, n_.model, first),
               )
    return n
