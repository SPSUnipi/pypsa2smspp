# test_step4_prepare_sddp_interface.py
"""
Test di prepare_sddp_interface in isolamento.

Verifica che:
  - il parser dei Block_i.nc4 venga chiamato
  - stage_structures sia nel dict di output
  - state_info sia nel dict di output
  - le dimensioni dell'SDDPBlock riflettano state_info
"""
import pypsa
from pypsa2smspp.transformation import Transformation


def main():
    # ------------------------------------------------------------------
    # 1. Carica la rete
    # ------------------------------------------------------------------
    # La rete ha 24 snapshot. Con snapshots_per_stage=8 otteniamo 3
    # stage, che è il numero di Block_i.nc4 che abbiamo.
    n = pypsa.Network("pypsa_stoch_load.nc")

    # ------------------------------------------------------------------
    # 2. Parametri stocastici SDDP
    # ------------------------------------------------------------------
    sp = {
        "stochastic_type": "sddp",
        "parameters": ["demand"],
        "snapshots_per_stage": 8,
        "top_path": {"K": 1, "second_group_index": 0, "third_group_index": 0},
        "benders_path": {"n_bacini": 0},
    }

    # ------------------------------------------------------------------
    # 3. Costruisci la Transformation e chiama prepare_sddp_interface
    # ------------------------------------------------------------------
    # workdir="." → i Block_i.nc4 devono essere nella cartella corrente
    t = Transformation(stochastic_parameters=sp, workdir=".")
    t.consistency_check(n)
    t.prepare_sddp_interface(n)

    # ------------------------------------------------------------------
    # 4. Verifiche
    # ------------------------------------------------------------------
    sddp_data = t.sddp_data

    print("Chiavi di sddp_data:", sorted(sddp_data.keys()))

    # Chiavi attese
    assert "stage_structures" in sddp_data, \
        "stage_structures mancante in sddp_data"
    assert "state_info" in sddp_data, \
        "state_info mancante in sddp_data"

    structures = sddp_data["stage_structures"]
    state_info = sddp_data["state_info"]
    dimensions = sddp_data["dimensions"]

    print(f"\nStrutture parsate: {len(structures)}")
    assert len(structures) == 3, \
        f"Attese 3 strutture, trovate {len(structures)}"

    print(f"\nstate_info:")
    print(f"  state_size       = {state_info['state_size']}")
    print(f"  initial_state    = {state_info['initial_state']}")
    print(f"  admissible_state = {state_info['admissible_state']}")

    assert state_info["state_size"] == 1, \
        f"Atteso state_size=1, ottenuto {state_info['state_size']}"
    assert state_info["initial_state"].shape == (1,)
    assert state_info["admissible_state"].shape == (3,)

    print(f"\ndimensions (SDDPBlock):")
    print(f"  StateSize           = {dimensions['StateSize']}")
    print(f"  InitialStateSize    = {dimensions['InitialStateSize']}")
    print(f"  AdmissibleStateSize = {dimensions['AdmissibleStateSize']}")

    assert dimensions["StateSize"] == 1
    assert dimensions["InitialStateSize"] == 1
    assert dimensions["AdmissibleStateSize"] == 3

    print("\nStep 4: OK")


if __name__ == "__main__":
    main()