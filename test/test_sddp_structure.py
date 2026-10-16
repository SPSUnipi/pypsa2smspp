import pypsa
from pypsa2smspp.transformation import Transformation

path = "pypsa_stoch_load.nc"   # se è nella stessa cartella del test
n = pypsa.Network(path)

print("Snapshots:", len(n.snapshots))
print("has_scenarios:", getattr(n, "has_scenarios", False))
print("scenarios:", list(getattr(n, "scenarios", [])))

sp = {
    "stochastic_type": "sddp",
    "parameters": ["demand"],
    "snapshots_per_stage": 4,
    "top_path": {
        "K": 1,
        "second_group_index": 0,
        "third_group_index": 3,
    },
    "benders_path": {
        "n_bacini": 3,   # nessuna variabile di stato per il primo test
        # (13/10/2026) cambiato a 3 per verificare che il Path sia costruito correttamente
    },
}

t = Transformation(stochastic_parameters=sp)
t.create_model(n)

t.sms_network.to_netcdf("test_sddp_4.nc4")