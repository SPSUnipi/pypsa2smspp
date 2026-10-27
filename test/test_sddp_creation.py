"""
Test 1: verifica la creazione dell'SDDPBlock e dei sotto-blocchi.

Non lancia il solver. Carica la rete, costruisce SDDPBlock + Block_i.nc4,
stampa un riepilogo della struttura e la salva su file.

Uso:
    python test_sddp_creation.py
    python test_sddp_creation.py --trim-hydro   # riduce a 1 solo bacino
"""

import sys
import argparse
import traceback
from pathlib import Path

import numpy as np
import pypsa
import pypsa2smspp
import pysmspp
import netCDF4

from pypsa2smspp.transformation import Transformation
from pypsa2smspp.multistage_utils import parse_block_structure

# ================================================================
# Configurazione
# ================================================================
NETWORK_PATH = Path("network_stochastic_network_small_fewsectors__snap100__scen3.nc")
WORKDIR      = Path("test_sddp_workdir")
SNAPSHOTS_PER_STAGE = 25   # 100/25 = 4 stadi. Metti 4 se vuoi testare 25 stadi.

# ================================================================
# Helper opzionale: riduce la rete a un solo hydro
# ================================================================
import pandas as pd  # serve per MultiIndex

def trim_to_single_hydro(n, verbose=True):
    """
    Tiene UNA storage unit hydro per scenario, con lo stesso nome in ogni scenario.
    Preferisce il carrier 'hydro' (ha inflow) al carrier 'PHS' (non ce l'ha).
    """
    if n.storage_units.empty:
        return n

    carriers = n.storage_units["carrier"].astype(str).str.strip().str.lower()
    hydro_mask = carriers.isin(["hydro", "phs"])

    if not hydro_mask.any():
        if verbose:
            print("[trim] Nessuna storage unit hydro/PHS: nessuna modifica.")
        return n

    is_multi = isinstance(n.storage_units.index, pd.MultiIndex)

    # ------------------------------------------------------------
    # 1. Decidi cosa tenere e cosa rimuovere
    # ------------------------------------------------------------
    to_remove = []

    if is_multi:
        hydro_df = n.storage_units[hydro_mask]
        for scenario, group in hydro_df.groupby(level=0):
            names = group.index.tolist()
            if len(names) <= 1:
                continue
            group_carriers = group["carrier"].astype(str).str.strip().str.lower()
            hydro_only = group.index[group_carriers == "hydro"].tolist()
            keep = hydro_only[0] if hydro_only else names[0]
            to_remove.extend([name for name in names if name != keep])
    else:
        names = n.storage_units.index[hydro_mask].tolist()
        if len(names) > 1:
            # Preferisci carrier 'hydro'
            sub_carriers = n.storage_units.loc[names, "carrier"] \
                .astype(str).str.strip().str.lower()
            hydro_only = [nm for nm in names if sub_carriers.loc[nm] == "hydro"]
            keep = hydro_only[0] if hydro_only else names[0]
            to_remove = [nm for nm in names if nm != keep]

    if not to_remove:
        if verbose:
            print("[trim] Nulla da rimuovere.")
        return n

    if verbose:
        print(f"[trim] Rimuovo {len(to_remove)} storage unit:")
        for name in to_remove:
            print(f"       - {name}")

    # ------------------------------------------------------------
    # 2. Rimuovi dai DataFrame
    # ------------------------------------------------------------
    # static
    n.storage_units = n.storage_units.drop(index=to_remove)

    # dynamic: ogni attributo in n.storage_units_t è un DataFrame con
    # colonne = MultiIndex (scenario, name). Filtra le colonne da rimuovere.
    for attr in list(n.storage_units_t.keys()):
        df = n.storage_units_t[attr]
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        drop_cols = [c for c in df.columns if c in to_remove]
        if drop_cols:
            n.storage_units_t[attr] = df.drop(columns=drop_cols)

    # ------------------------------------------------------------
    # 3. Verifica finale
    # ------------------------------------------------------------
    if verbose:
        print("\n[trim] Stato dopo il trim:")
        kept_per_scenario = {}
        for scenario in n.storage_units.index.get_level_values(0).unique() \
                if is_multi else [None]:
            sub = (n.storage_units.loc[scenario]
                   if is_multi else n.storage_units)
            sub_carriers = sub["carrier"].astype(str).str.strip().str.lower()
            h_names = sub.index[sub_carriers.isin(["hydro", "phs"])].tolist()
            label = scenario if is_multi else "det"
            print(f"       {label}: {h_names}")
            kept_per_scenario[label] = tuple(h_names)

        if is_multi and len(set(kept_per_scenario.values())) != 1:
            print(f"\n[trim] ATTENZIONE: scenari con hydro diverse!")
            print(f"       {kept_per_scenario}")

    return n

# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trim-hydro", action="store_true",
                        help="Riduce la rete a una sola StorageUnit hydro/PHS")
    args = parser.parse_args()

    # ---------- rete ----------
    print("=" * 70)
    print("Caricamento rete")
    print("=" * 70)
    if not NETWORK_PATH.exists():
        print(f"ERRORE: file non trovato: {NETWORK_PATH.resolve()}")
        print("Metti lo script nella stessa cartella del .nc, oppure modifica NETWORK_PATH.")
        sys.exit(1)

    n = pypsa.Network(str(NETWORK_PATH))

    print(f"Snapshots:      {len(n.snapshots)}")
    print(f"has_scenarios:  {getattr(n, 'has_scenarios', False)}")
    print(f"scenarios:      {list(getattr(n, 'scenarios', []))}")
    print(f"Buses:          {len(n.buses)}")
    print(f"Generators:     {len(n.generators)}")
    print(f"StorageUnits:   {len(n.storage_units)}")
    print(f"Stores:         {len(n.stores)}")
    print(f"Links:          {len(n.links)}")
    print(f"Loads:          {len(n.loads)}")

    if not n.storage_units.empty:
        carriers = n.storage_units["carrier"].astype(str).str.strip().str.lower()
        n_hydro = int(carriers.isin(["hydro", "phs"]).sum())
        print(f"Hydro/PHS:      {n_hydro} storage unit")
        if n_hydro != 1 and not args.trim_hydro:
            print("\nATTENZIONE: la pipeline SDDP supporta al momento 1 solo bacino.")
            print("Con più bacini fallirà in build_sddp_stochastic_block.")
            print("Rilancia con --trim-hydro per ridurre a un bacino, oppure")
            print("estendi build_sddp_benders_mapping per supportare N serbatoi.")

    if args.trim_hydro:
        n = trim_to_single_hydro(n)

    # ---------- parametri SDDP ----------
    sp = {
        "stochastic_type": "sddp",
        "parameters": ["demand", "hydro_inflow"],   # hydro_inflow è OBBLIGATORIO se c'è hydro
        "snapshots_per_stage": SNAPSHOTS_PER_STAGE,
        "top_path": {"K": 1},
    }

    # ---------- esecuzione ----------
    WORKDIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Creazione modello SDDP")
    print("=" * 70)

    t = Transformation(
        stochastic_parameters=sp,
        workdir=str(WORKDIR),
        name="test_sddp",
        capacity_expansion_ucblock=True,
        enable_thermal_units=False,
    )

    try:
        t.create_model(n, verbose=True)
    except Exception:
        print("\nERRORE durante create_model:")
        traceback.print_exc()
        sys.exit(1)

    # ---------- riepilogo ----------
    struct = parse_block_structure("test_sddp_workdir/Block_0.nc4")
    print("polyhedral_index:", struct["polyhedral_index"])
    for h in struct["hydro_unit_blocks"]:
        print(f"  index={h['index']}, name={h['name']!r}")

    with netCDF4.Dataset("test_sddp_workdir/Block_0.nc4", "r") as ds:
        hub = ds.groups["Block_0"].groups["UnitBlock_0"].groups["HydroUnitBlock_0"]
        for name in ["MaxVolumetric", "MinVolumetric", "InitialVolumetric", "Inflows"]:
            v = hub.variables[name]
            print(f"{name}: dims={v.dimensions}, shape={v.shape}, value={v[:]}")

    sddp = pysmspp.SMSNetwork("test_sddp_workdir/SDDPBlock_test_sddp.nc4")
    sb0 = sddp.blocks["Block_0"].blocks["StochasticBlock_0"]
    ap = sb0.blocks["Block"].blocks["BendersBFunction"].blocks["AbstractPath"]
    print("PathGroupIndices:", ap.variables["PathGroupIndices"].data)

    se = sb0.variables["SetElements"].data
    print("Ultimi 8 SetElements:", se[-8:])

    print("\n" + "=" * 70)
    print("Riepilogo")
    print("=" * 70)

    if t.sms_network is not None:
        print("SMSNetwork creata correttamente.")
        print(f"Blocchi top-level: {list(t.sms_network.blocks.keys())}")

    if t.sddp_data is not None:
        d = t.sddp_data
        print(f"\nStadi ({len(d['stage_names'])}): {d['stage_names']}")
        print(f"Parametri stocastici: {d['parameters']}")
        print(f"TimeHorizon per stage: {d['stage_time_horizons']}")
        print(f"\nScenari: {d['scenarios_info']['number_scenarios']}, "
              f"scenario_size = {d['scenarios_info']['scenario_size']}")
        print(f"State info: {d['state_info']}")

        # Dimensioni compatte dell'SDDPBlock
        print("\nDimensioni SDDPBlock:")
        for k, v in d["dimensions"].items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: array shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {v}")

        # Verifica coerenza fondamentale
        time_horizon = d["dimensions"]["TimeHorizon"]
        if len(d["stage_names"]) != time_horizon:
            print(f"\n  ERRORE: |stage_names|={len(d['stage_names'])} "
                  f"!= TimeHorizon={time_horizon}")
        else:
            print(f"\n  OK: numero stadi ({time_horizon}) coerente.")

    # ---------- file prodotti ----------
    print("\n" + "=" * 70)
    print(f"File prodotti in {WORKDIR.resolve()}:")
    print("=" * 70)
    for p in sorted(WORKDIR.glob("*.nc4")):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:30s} {size_kb:10.1f} KB")

    print("\nNOTA: il file SDDPBlock_*.nc4 fa riferimento ai Block_i.nc4 con path")
    print("relativi. Se lo sposti, i riferimenti si rompono. Tienili insieme.")


if __name__ == "__main__":
    main()