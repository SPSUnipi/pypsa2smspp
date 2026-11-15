"""Rete stocastica con afflussi molto incerti + file SMS++ (Block_i, SDDPBlock)."""
import sys, shutil, logging, pypsa, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import *
from pypsa2smspp.transformation import Transformation

MAXH = 12.0
# fattore sugli afflussi idro per scenario (stadi 1-3); stadio 0 uguale per tutti
FS = {"scenario_001": 1.0, "scenario_002": 0.2, "scenario_003": 1.8}
F = {s: [1.0] + [f] * (NS - 1) for s, f in FS.items()}

if __name__ == "__main__":
    n = pypsa.Network("rete_operativa.nc")
    su = n.storage_units
    for s in n.scenarios:
        su.loc[(s, HYDRO), "max_hours"] = MAXH
    h = su.carrier.isin(["hydro", "PHS"])
    su.loc[h, "state_of_charge_initial"] = (0.5 * su.p_nom * su.max_hours)[h]
    for s in n.scenarios:
        for u in HYDRO:
            n.storage_units_t.inflow[(s, u)] = n.storage_units_t.inflow[(s, u)] * np.repeat(F[s], L)
    n.export_to_netcdf("rete_inflow.nc")
    wd = Path("smspp"); wd.mkdir(exist_ok=True)
    sp = {"stochastic_type": "sddp", "parameters": ["hydro_inflow"],
          "snapshots_per_stage": L, "top_path": {"K": 1}, "initial_state_fraction": 0.5}
    t = Transformation(stochastic_parameters=sp, workdir=str(wd), name="inflow",
                       capacity_expansion_ucblock=False, enable_thermal_units=False)
    t.create_model(n, verbose=False)
    print(sorted(p.name for p in wd.glob("*")))
