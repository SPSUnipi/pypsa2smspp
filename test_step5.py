# test_step5_pipeline_completa.py
"""
Test della pipeline completa (create_model) che include convert_to_sddp_block.

ATTENZIONE: la rete pypsa_stoch_load.nc e i Block_i.nc4 che abbiamo non
sono coerenti. Il test verifica solo che il codice Python arrivi in fondo
e produca un file .nc4, non che il file sia semanticamente corretto.
"""
import pypsa
from pypsa2smspp.transformation import Transformation


def main():
    n = pypsa.Network("pypsa_stoch_load.nc")

    sp = {
        "stochastic_type": "sddp",
        "parameters": ["demand"],
        "snapshots_per_stage": 8,
        "top_path": {"K": 1, "second_group_index": 0, "third_group_index": 0},
        "benders_path": {"n_bacini": 0},
    }

    t = Transformation(stochastic_parameters=sp, workdir=".")

    # create_model fa: consistency_check + direct + prepare_sddp_interface
    # + prepare_tssb_interface + prepare_sddp_interface + convert_to_blocks
    # convert_to_blocks, per SDDP, chiama convert_to_sddp_block
    t.create_model(n)

    # Salva il file prodotto
    output_path = "test_sddp_step5.nc4"
    t.sms_network.to_netcdf(output_path)
    print(f"\nFile salvato: {output_path}")

    # Verifica che il file esista
    from pathlib import Path
    assert Path(output_path).exists(), \
        f"Il file {output_path} non è stato creato"

    print("Step 5: OK (pipeline completa, file prodotto)")


if __name__ == "__main__":
    main()