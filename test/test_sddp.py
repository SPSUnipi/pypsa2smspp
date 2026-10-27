"""
Test completo pipeline SDDP: create_model + optimize + retrieve_solution.

Requisiti:
  1. I 10 file di configurazione SMSPP nella workdir
     (possono essere copiati da SDDP_CONFIG_SOURCE se esiste).
  2. L'eseguibile 'sddp_solver' raggiungibile: in PATH, oppure
     indicato in SDDP_SOLVER_PATH.
"""

from __future__ import annotations

import shutil
import sys
import time
import traceback
from pathlib import Path

import netCDF4
import pypsa

from pypsa2smspp.transformation import Transformation


# ----------------------------------------------------------------------
# Configurazione
# ----------------------------------------------------------------------
NETWORK_PATH = Path("network_stochastic_network_small_fewsectors__snap100__scen3.nc")
WORKDIR = Path("test_sddp_workdir")

SNAPSHOTS_PER_STAGE = 25
INITIAL_STATE_FRACTION = 0.5

SDDP_CONFIG_SOURCE = Path("sddp_configs")
SDDP_SOLVER_PATH = r"C:\Users\bianc\Anaconda\envs\pypsa2smspp\Library\bin\sddp_solver.exe"

SDDP_CONFIG_FILES = [
    "SDDPSCfg.txt", "SDDPBCfg.txt", "PFBCfg.txt", "SUBCfg.txt",
    "BUBCfg.txt", "TUBCfg-OVC.txt", "DCNBCfg.txt", "ACBCfg.txt",
    "BCfg.txt", "UCBSCfg.txt",
]


# ----------------------------------------------------------------------
# Pre-check
# ----------------------------------------------------------------------
def check_config_files() -> bool:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    missing = [f for f in SDDP_CONFIG_FILES if not (WORKDIR / f).exists()]
    if not missing:
        print(f"[ok] Config SMSPP: {len(SDDP_CONFIG_FILES)}/{len(SDDP_CONFIG_FILES)} presenti")
        return True

    print(f"[--] Config SMSPP mancanti: {len(missing)}/{len(SDDP_CONFIG_FILES)}")
    if SDDP_CONFIG_SOURCE.exists():
        for f in SDDP_CONFIG_FILES:
            src = SDDP_CONFIG_SOURCE / f
            if src.exists() and not (WORKDIR / f).exists():
                shutil.copy(src, WORKDIR / f)

    missing = [f for f in SDDP_CONFIG_FILES if not (WORKDIR / f).exists()]
    if missing:
        print(f"[ERRORE] Mancano ancora: {missing}")
        print(f"         Copiali in {WORKDIR.resolve()} e rilancia.")
        return False
    print(f"[ok] Config SMSPP copiati da {SDDP_CONFIG_SOURCE}")
    return True


def check_solver() -> bool:
    solver = SDDP_SOLVER_PATH or "sddp_solver"
    p = Path(solver)
    if p.is_file():
        print(f"[ok] Solver: {p.resolve()}")
        return True
    resolved = shutil.which(solver)
    if resolved:
        print(f"[ok] Solver: {resolved}")
        return True
    print(f"[ERRORE] 'sddp_solver' non trovato in PATH.")
    print("         Aggiungilo a PATH oppure imposta SDDP_SOLVER_PATH.")
    return False


# ----------------------------------------------------------------------
# Ispezione idro nei Block_i.nc4
# ----------------------------------------------------------------------
def inspect_hydro_blocks(block_path: Path) -> None:
    print("\nIdro nei Block_i.nc4:")
    with netCDF4.Dataset(block_path, "r") as ds:
        root = ds.groups[next(iter(ds.groups))]
        for g in root.groups.values():
            if g.getncattr("type") != "HydroSystemUnitBlock":
                continue
            for c in g.groups.values():
                if c.getncattr("type") != "HydroUnitBlock":
                    continue
                name = c.getncattr("name")
                inflow = float(c.variables["Inflows"][:].sum())
                v0 = float(c.variables["InitialVolumetric"][:].ravel()[0])
                vmax = float(c.variables["MaxVolumetric"][:].ravel()[0])
                print(
                    f"  {name:20s} "
                    f"inflow={inflow:10.2f}  V0={v0:12.2f}  Vmax={vmax:12.2f}"
                )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("Test completo pipeline SDDP")
    print("=" * 70)

    if not check_config_files():
        return 1
    if not check_solver():
        return 1

    if not NETWORK_PATH.exists():
        print(f"[ERRORE] Rete non trovata: {NETWORK_PATH.resolve()}")
        return 1

    n = pypsa.Network(str(NETWORK_PATH))
    print(
        f"\nRete: {len(n.snapshots)} snapshots, "
        f"{len(n.storage_units)} storage, "
        f"{len(n.generators)} generatori, "
        f"{len(n.buses)} bus"
    )

    sp = {
        "stochastic_type": "sddp",
        "parameters": ["demand", "hydro_inflow"],
        "snapshots_per_stage": SNAPSHOTS_PER_STAGE,
        "top_path": {"K": 1},
        "initial_state_fraction": INITIAL_STATE_FRACTION,
    }

    solver_kwargs = {}
    if SDDP_SOLVER_PATH is not None:
        solver_kwargs["solver_path"] = SDDP_SOLVER_PATH

    t = Transformation(
        stochastic_parameters=sp,
        workdir=str(WORKDIR),
        name="test_sddp",
        capacity_expansion_ucblock=True,
        enable_thermal_units=False,
        pysmspp_options=solver_kwargs or None,
    )

    t0 = time.time()
    try:
        print("\n" + "=" * 70)
        print("Pipeline")
        print("=" * 70)
        n = t.run(n, verbose=True)
    except Exception:
        print("\n[ERRORE] Pipeline fallita:")
        traceback.print_exc()
        return 1
    elapsed = time.time() - t0

    try:
        inspect_hydro_blocks(WORKDIR / "Block_0.nc4")
    except Exception as exc:
        print(f"  (ispezione saltata: {exc})")

    print("\n" + "=" * 70)
    print("Risultato")
    print("=" * 70)
    print(f"Tempo totale: {elapsed:.1f}s")

    if t.result is not None:
        print(f"Objective: {t.result.objective_value:.6e}")
        if t.fp_solution:
            spath = Path(t.fp_solution)
            if spath.exists():
                print(f"Solution:  {spath.resolve()}")
    else:
        print("[ERRORE] t.result è None")

    if hasattr(n.generators_t, "p") and len(n.generators_t.p) > 0:
        print(f"n.generators_t['p'] shape: {n.generators_t['p'].shape}")
    if hasattr(n.storage_units_t, "p") and len(n.storage_units_t.p) > 0:
        print(f"n.storage_units_t['p'] shape: {n.storage_units_t['p'].shape}")

    print(f"\nFile in {WORKDIR.resolve()}:")
    for f in sorted(WORKDIR.glob("*")):
        if f.is_file():
            print(f"  {f.name:35s} {f.stat().st_size / 1024:10.1f} KB")

    print("\n[OK] Test completato.")
    return 0


if __name__ == "__main__":
    sys.exit(main())