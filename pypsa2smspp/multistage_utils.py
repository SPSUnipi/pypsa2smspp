# -*- coding: utf-8 -*-
"""
Utilities for multistage stochastic PyPSA networks.
"""

from __future__ import annotations

import netCDF4
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, Tuple
from pathlib import Path
import pypsa

import pypsa2smspp.stochastic_utils as su
from pypsa2smspp.constants import STOCHASTIC_PARAMETER_REGISTRY

# =================================================
# Normalizzazione parametri stocastici
# Analogo di stochastic_utils.normalize_stochastic_parameters
# =================================================

def normalize_sddp_parameters(
        stochastic_parameters: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Normalizza e valida i parametri stocastici specifici per SDDP.

    Formato atteso in input (esempio):
        {
            "stochastic_type": "sddp",
            "parameters": ["demand", "renewable_maxpower"],
            "snapshots_per_stage": 31,
            "top_path": {
                "K": 1,
            },
            "initial_state_fraction": 0.5,
        }

    L'utente fornisce solo il numero di snapshot per stadio
    (`snapshots_per_stage`). La lista dei periodi (uno per stadio, ciascuno
    con i suoi snapshot) viene derivata internamente da `make_sddp_periods`
    a partire da `n.snapshots`.

    Returns
    -------
    dict
        Dizionario pulito con chiavi:
        - "stochastic_type": "sddp"
        - "parameters": lista di parametri stocastici validi
        - "snapshots_per_stage": intero >= 1
    """

    sp = dict(stochastic_parameters or {})

    # Controllo tipo stocastico
    stochastic_type = sp.get("stochastic_type", None)
    if stochastic_type != "sddp":
        raise ValueError(
            f"stochastic_type {stochastic_type} non valido."
            f"normalize_sddp_parameters richiede stochastic_type='sddp',"
            f"ricevuto {stochastic_type!r}."
        )

    # Normalizzazione parametri stocastici
    parameters = sp.get("parameters", [])
    if parameters is None:
        parameters = []
    elif isinstance(parameters, str):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    parameters = [str(p).strip().lower() for p in parameters if str(p).strip()]

    valid_parameters = set(STOCHASTIC_PARAMETER_REGISTRY)
    invalid = sorted(set(parameters) - valid_parameters)

    if invalid:
        raise ValueError(
            f"Parametri stocastici non supportati per SDDP: {invalid}. "
            f"Valori validi (STOCHASTIC_PARAMETER_REGISTRY): "
            f"{sorted(valid_parameters)}."
        )

    # Validazione snapshots_per_stage
    snapshots_per_stage = sp.get("snapshots_per_stage", None)

    if snapshots_per_stage is None:
        raise ValueError(
            "SDDP richiede il campo 'snapshots_per_stage' in "
            "stochastic_parameters: un intero >= 1 che indica quanti "
            "snapshot della rete appartengono a ciascun stadio."
        )

    # bool è sottoclasse di int, quindi escludiamolo esplicitamente.
    if isinstance(snapshots_per_stage, bool) or not isinstance(snapshots_per_stage, int):
        raise ValueError(
            f"'snapshots_per_stage' deve essere un intero, ricevuto "
            f"{type(snapshots_per_stage).__name__} ({snapshots_per_stage!r})."
        )

    if snapshots_per_stage < 1:
        raise ValueError(
            f"'snapshots_per_stage' deve essere >= 1, ricevuto "
            f"{snapshots_per_stage}."
        )


    # Validazione top_path (obbligatorio per SDDP)
    top_path = sp.get("top_path", None)

    if top_path is None:
        raise ValueError(
            "SDDP richiede 'top_path' in stochastic_parameters: "
            "un dizionario con le chiavi 'K', 'second_group_index', "
            "'third_group_index'."
        )

    if not isinstance(top_path, dict):
        raise ValueError(
            f"'top_path' deve essere un dizionario, ricevuto "
            f"{type(top_path).__name__}."
        )

    if "K" not in top_path:
        raise ValueError(
            "'top_path' manca della chiave obbligatoria 'K'."
        )

    K = top_path["K"]
    if isinstance(K, bool) or not isinstance(K, int):
        raise ValueError(
            f"'top_path['K']' deve essere un intero, ricevuto "
            f"{type(K).__name__} ({K!r})."
        )
    if K < 0:
        raise ValueError(
            f"'top_path['K']' deve essere >= 0, ricevuto {K}."
        )

    # Solo K=1 è supportato (un solo PolyhedralFunctionBlock per
    # HydroSystemUnitBlock, creato da convert_to_ucblock)
    if K != 1:
        raise ValueError(
            f"'top_path['K']' = {K} non è supportato: "
            "attualmente l'implementazione crea un solo "
            "PolyhedralFunctionBlock per HydroSystemUnitBlock, quindi "
            "solo K=1 è coerente con la struttura prodotta."
        )

    # Validazione initial_state_fraction
    # Frazione di MaxVolumetric usata come volume iniziale nei Block_i.nc4.
    # Default 0.5 (= metà della capacità dell'invaso). Deve stare in [0, 1].
    initial_state_fraction = sp.get("initial_state_fraction", 0.5)

    # bool è sottoclasse di int, quindi lo escludiamo esplicitamente
    if isinstance(initial_state_fraction, bool) or not isinstance(
            initial_state_fraction, (int, float)):
        raise ValueError(
            f"'initial_state_fraction' deve essere un numero, ricevuto "
            f"{type(initial_state_fraction).__name__} ({initial_state_fraction!r})."
        )

    if not (0.0 <= float(initial_state_fraction) <= 1.0):
        raise ValueError(
            f"'initial_state_fraction' deve essere in [0, 1], ricevuto "
            f"{initial_state_fraction}."
        )

    return {
        "stochastic_type": "sddp",
        "parameters": parameters,
        "snapshots_per_stage": snapshots_per_stage,
        "top_path": {"K": int(top_path["K"])},
        "initial_state_fraction": float(initial_state_fraction),
    }



# =================================================
# Partizione in stadi
# =================================================

def get_sddp_stage_names(n, stochastic_parameters=None) -> List[Any]:
    """
    Restituisce la lista ordinata dei nomi degli stadi SDDP.

    I nomi vengono derivati da `make_sddp_periods`, che partiziona
    n.snapshots in base a `snapshots_per_stage` (fornito dall'utente in
    `stochastic_parameters`).

    Parameters
    ----------
    n : pypsa.Network
        Rete PyPSA (possibilmente stocastica).
    stochastic_parameters : dict, optional
        Parametri stocastici già forniti dall'utente. Deve contenere snapshots_per_stage

    Returns
    -------
    list
        Nomi degli stadi (es. ["2020", "2021", ...]).
    """
    # Normalizza e valida i parametri SDDP, ottenendo anche i periodi
    sp = normalize_sddp_parameters(stochastic_parameters)

    # Deriviamo i periods a partire da n.snapshots
    periods =make_sddp_periods(n, sp["snapshots_per_stage"])

    # Caso 1: periodi forniti esplicitamente dall'utente
    if periods:
        return [p["name"] for p in periods]

    # Estraiamo solo i nomi
    return [p["name"] for p in periods]

# ================================================
# Estrazione dei dati per ogni singolo stadio
# ================================================

def build_sddp_demand_stage(n, stage_snapshots: pd.Index) -> Dict[str, Any]:
    """
    Estrae i dati di domanda per un singolo stadio SDDP.

    Questa funzione è l'equivalente di `stochastic_utils.build_dss_demand`
    ma limitata agli snapshot di un solo stadio.

    Parametri
    ----------
    n : pypsa.Network
        Rete stocastica PyPSA.
    stage_snapshots : pd.Index o lista di timestamp
        Snapshot appartenenti a questo stadio.

    Ritorna
    -------
    dict
        Dizionario con:
        - "scenarios": matrice (NumberScenarios, SubScenarioSize_demand)
        - "pool_weights": probabilità scenario (vettore)
        - "node_order": ordine dei bus
        - "snapshot_order": ordine degli snapshot (dello stadio)
        - "scenario_size": dimensione del vettore di scenario per questo stadio
        - "number_scenarios": numero di scenari
        - altri metadati di flattening
    """
    scenario_names = su.get_scenario_names(n)
    if not scenario_names:
        raise ValueError(
            "build_sddp_demand_stage richiede una rete stocastica "
            "(n.has_scenarios)."
        )

    # Converti stage_snapshots in un pd.Index per operazioni di reindexing
    stage_snapshots = pd.Index(stage_snapshots)

    scenarios = []
    bus_order = None

    for scenario_name in scenario_names:
        # Prendi la rete per questo scenario
        n_s = n.get_scenario(scenario_name)

        # Estrai domanda per bus: DataFrame con righe=bus, colonne=snapshots
        demand_by_bus = su.get_bus_demand_matrix(n_s)

        # Verifica che tutti gli snapshot dello stadio siano presenti
        # Nota: se i tipi non coincidono, prova a convertire in DatetimeIndex
        if not all(s in demand_by_bus.columns for s in stage_snapshots):
            stage_snapshots_dt = pd.to_datetime(stage_snapshots)
            missing = [s for s in stage_snapshots_dt if s not in demand_by_bus.columns]
            if missing:
                raise ValueError(
                    f"Snapshot dello stadio non trovati nella domanda dello "
                    f"scenario {scenario_name!r}: {missing[:5]}"
                    f"{'...' if len(missing) > 5 else ''}"
                )
            # Se tutti trovati, aggiorna stage_snapshots alla versione Datetime
            stage_snapshots = stage_snapshots_dt

        # Filtra le colonne della domanda mantenendo solo gli snapshot dello stadio
        demand_by_bus = demand_by_bus.reindex(columns=stage_snapshots)

        # Salva l'ordine dei bus la prima volta e riordina le altre per coerenza
        if bus_order is None:
            bus_order = list(demand_by_bus.index)
        else:
            demand_by_bus = demand_by_bus.reindex(index=bus_order, fill_value=np.nan)

        # Appiattisci con ordine node_major_time_minor (come TSSB)
        scenarios.append(su.flatten_bus_demand_node_major(demand_by_bus))

    # Impila tutte le righe scenario in una matrice
    scenario_matrix = np.vstack(scenarios).astype(float)

    # Probabilità scenario (usate per dopo; di solito uniformi)
    pool_weights = su.get_scenario_probabilities(n).astype(float)

    return {
        "parameter": "demand",
        "scenarios": scenario_matrix,
        "pool_weights": pool_weights,
        "node_order": bus_order,
        "snapshot_order": list(stage_snapshots),
        "flattening": "node_major_time_minor",
        "scenario_size": int(scenario_matrix.shape[1]),
        "number_scenarios": int(scenario_matrix.shape[0]),
    }

def build_sddp_unitblock_timeseries_parameter_stage(
        n,
        stage_snapshots: pd.Index,
        parameter: str,
        pypsa_component: str,
        field: str,
        asset_names: Sequence[Any],
        function_name: str,
        unitblock_type: str,
        target: str,
        transformation_config,
        smspp_parameter: str | None,
        weights: bool,
)-> Dict[str, Any]:
    """
    Estrae dati di un parametro stocastico di tipo UnitBlock per un
    singolo stadio.

    Analogo a build_dss_unitblock_timeseries_parameter(...)
    """

    # Otteniamo i nomi degli scenari
    scenario_names = su.get_scenario_names(n)

    if not scenario_names:
        raise ValueError(
            f"build_sddp_unitblock_timeseries_parameter_stage richiede una rete stocastica (n.hash_scenarios)"
        )

    # Verifichiamo che asset_names non sia vuoto
    if not asset_names:
        raise ValueError(
            f"asset_names non può essere vuoto"
        )

    # Convertiamo stage_snapshots in pd.Index
    stage_snapshots = pd.Index(stage_snapshots)
    if not isinstance(stage_snapshots, pd.DatetimeIndex):
        stage_snapshots_dt = pd.to_datetime(stage_snapshots)
    else:
        stage_snapshots_dt = stage_snapshots

    # Inizializziamo i contenitori
    scenarios = []
    asset_order = None
    snapshot_order = None

    for scenario_name in scenario_names:
        # Otteniamo la rete dello scenario
        n_s = n.get_scenario(scenario_name)
        values = su.evaluate_unitblock_parameter_timeseries(
            n_s,
            parameter=parameter,
            pypsa_component=pypsa_component,
            field=field,
            asset_names=asset_names,
            transformation_config=transformation_config,
            unitblock_type=unitblock_type,
            smspp_parameter=smspp_parameter,
            weights=weights,
        )
        # Questa funzione restituisce un DataFrame con indice -> snapshot e colonne -> asset

        # Verifica che tutti gli snapshot dello stadio siano presenti
        missing = [s for s in stage_snapshots_dt if s not in values.index]
        if missing:
            raise ValueError(
                f"Snapshot dello stadio non trovati per {parameter!r} "
                f"(scenario {scenario_name!r}): {missing[:5]}"
            )

        # Filtra per gli snapshot dello stadio
        values = values.reindex(index=stage_snapshots_dt)

        if asset_order is None:
            asset_order = list(values.columns)
            snapshot_order = list(values.index)
        else:
            values = values.reindex(
                index=snapshot_order,
                columns=asset_order,
            )

        # Controllo valori mancanti
        if values.isna().any().any():
            missing_assets = values.columns[values.isna().any(axis=0)].tolist()
            raise ValueError(
                f"Valori mancanti per {parameter!r} in questo stadio. "
                f"Asset coinvolti: {missing_assets}"
            )

        scenarios.append(su.flatten_asset_timeseries_asset_major(values))

    # Costruiamo matrice finale
    scenario_matrix = np.vstack(scenarios).astype(float)
    # Otteniamo pool_weights
    pool_weights = su.get_scenario_probabilities(n).astype(float)

    return {
        "parameter": parameter,
        "target": target,
        "function_name": function_name,
        "unitblock_type": unitblock_type,
        "smspp_parameter": smspp_parameter,
        "pypsa_component": pypsa_component,
        "field": field,
        "source": f"{pypsa_component}.{field}",
        "scenarios": scenario_matrix,
        "pool_weights": pool_weights,
        "asset_order": asset_order,
        "snapshot_order": snapshot_order,
        "flattening": "asset_major_time_minor",
        "scenario_size": int(scenario_matrix.shape[1]),
        "number_scenarios": int(scenario_matrix.shape[0]),
    }

def build_sddp_stage_data(
        n,
        stage_name,
        stage_snapshots: pd.Index,
        stochastic_parameters: Sequence[str],
        intermittent_carriers,
        default_intermittent_carriers,
        enable_thermal_units: bool,
        transformation_config,
) -> Dict[str, Any]:
    """
    Per un singolo stadio, estrae i dati di tutti i parametri stocastici
    richiesti e li raggruppa in una lista parts
    """
    parts = []
    for parameter in stochastic_parameters:
        spec = STOCHASTIC_PARAMETER_REGISTRY[parameter]
        mapping_kind = spec["mapping_kind"]
        if mapping_kind == "ucblock_timeseries":
            if parameter != "demand":
                raise NotImplementedError("Per il momento supportato solo demand")
            parts.append(build_sddp_demand_stage(n, stage_snapshots))

        elif mapping_kind == "unitblock_timeseries":
            asset_names = su.get_stochastic_parameter_asset_names(
                n=n,
                parameter=parameter,
                spec=spec,
                intermittent_carriers=intermittent_carriers,
                default_intermittent_carriers=default_intermittent_carriers,
                enable_thermal_units=enable_thermal_units,
            )
            parts.append(
                build_sddp_unitblock_timeseries_parameter_stage(
                    n,
                    stage_snapshots,
                    parameter=parameter,
                    pypsa_component=spec["pypsa_component"],
                    field=spec["field"],
                    asset_names=asset_names,
                    function_name=spec["function_name"],
                    unitblock_type=spec["unitblock_type"],
                    target=spec["target"],
                    transformation_config=transformation_config,
                    smspp_parameter=spec.get("smspp_parameter", None),
                    weights=bool(spec.get("weights", False)),
                )
            )
        else:
            raise ValueError("mapping_kind deve essere supportato")

    return {
        "stage": stage_name,
        "parts": parts
    }


def merge_sddp_stage_data(stage_parts, stage_name=None) -> Dict[str, Any]:
    """
    Data una lista di parti stocastiche di un singolo stadio, produce il vettore
    di scenario completo per quello stadio e le informazioni necessarie per
    il costruttore di SDDPBlock.

    Parametri
    ----------
    stage_parts : list of dict
        Lista di dizionari, ognuno dei quali è il risultato di
        build_sddp_demand_stage o build_sddp_unitblock_timeseries_parameter_stage.
    stage_name : str, optional
        Nome dello stadio; se fornito viene incluso nell'output.

    Ritorna
    -------
    dict
        Dizionario con le chiavi:
        - "stage": nome stadio (solo se stage_name non è None)
        - "scenarios": matrice (NumberScenarios, SubScenarioSize)
        - "sub_scenario_size": somma delle dimensioni delle parti
        - "size_random_data_groups": lista delle dimensioni di ciascun gruppo
        - "pool_weights": vettore delle probabilità scenario
        - "parts": copia di stage_parts con offset_start/offset_end aggiunti
    """
    if not stage_parts:
        raise ValueError("stage_parts non può essere vuoto")

    # Il numero di scenari deve essere lo stesso per tutte le parti
    number_scenarios = stage_parts[0]["number_scenarios"]
    if number_scenarios == 0:
        raise ValueError("number_scenarios non può essere zero")

    # Lista per raccogliere gli array di scenario di ogni parte
    scenario_arrays = []
    # Lista delle dimensioni dei gruppi (una per parte)
    size_random_data_groups = []
    # Riferimento ai pool_weights (vettore probabilità)
    pool_weights = None

    # Lista per le parti con offset calcolati
    checked_parts = []
    offset = 0

    for part in stage_parts:
        # Verifica coerenza del numero di scenari
        if int(part["number_scenarios"]) != number_scenarios:
            raise ValueError(
                f"Tutte le parti devono avere lo stesso number_scenarios. "
                f"Atteso {number_scenarios}, ricevuto {part['number_scenarios']} "
                f"per la parte {part.get('parameter')!r}."
            )

        # Verifica coerenza delle probabilità scenario
        part_pool_weights = np.asarray(part["pool_weights"], dtype=float)
        if pool_weights is None:
            pool_weights = part_pool_weights
        elif not np.allclose(part_pool_weights, pool_weights):
            raise ValueError(
                "Tutte le parti devono avere gli stessi pool_weights. "
                f"Mismatch trovato per la parte {part.get('parameter')!r}."
            )

        # Estrai la matrice degli scenari come array float
        scenarios = np.asarray(part["scenarios"], dtype=float)
        part_size = scenarios.shape[1]

        # Aggiungi alla lista degli array
        scenario_arrays.append(scenarios)
        # Registra la dimensione del gruppo
        size_random_data_groups.append(part_size)

        # Crea una copia della parte con offset start/end
        part_copy = dict(part)
        part_copy["offset_start"] = offset
        part_copy["offset_end"] = offset + part_size
        checked_parts.append(part_copy)

        # Aggiorna l'offset per la prossima parte
        offset += part_size

        # Concatenazione orizzontale di tutte le parti
    stage_scenarios = np.hstack(scenario_arrays)  # shape (NumberScenarios, SubScenarioSize)
    sub_scenario_size = int(stage_scenarios.shape[1])

    # Costruzione del dizionario di output
    result = {
        "scenarios": stage_scenarios,
        "sub_scenario_size": sub_scenario_size,
        "size_random_data_groups": size_random_data_groups,
        "pool_weights": pool_weights,
        "parts": checked_parts,
    }

    # Includi il nome dello stadio solo se fornito
    if stage_name is not None:
        result["stage"] = stage_name

    return result

def build_sddp_scenarios(stage_data_list: list[dict]) -> Dict[str, Any]:
    """
    Concatena i vettori di ogni stadio
    in una matrice unica (NumberScenarios, ScenarioSize)
    """
    if not stage_data_list:
        raise ValueError("stage_data_list non è conforme")

    ref_stage = stage_data_list[0]
    # Restituiremo poi una lista con un valore per ogni stadio
    # SMS++ accetta che ogni stadio abbia valori diversi
    sub_scenario_sizes = [stage["sub_scenario_size"] for stage in stage_data_list]
    pool_weights_ref = np.asarray(ref_stage["pool_weights"], dtype=float)
    number_scenarios_ref = ref_stage["scenarios"].shape[0]

    for stage in stage_data_list:
        stage_pool_weights = np.asarray(stage["pool_weights"], dtype=float)
        if not np.allclose(stage_pool_weights, pool_weights_ref):
            raise ValueError(
                "Tutti gli stadi devono avere gli stessi pool_weights."
            )

        if stage["scenarios"].shape[0] != number_scenarios_ref:
            raise ValueError(
                "Tutti gli stadi devono avere lo stesso number_scenarios."
            )

    stage_scenario_list = [stage["scenarios"] for stage in stage_data_list]
    scenarios = np.hstack(stage_scenario_list)
    scenario_size = scenarios.shape[1]

    return {
        "scenarios": scenarios,
        "number_scenarios": number_scenarios_ref,
        "scenario_size": scenario_size,
        "sub_scenario_size": sub_scenario_sizes,
        "pool_weights": pool_weights_ref,
        # opzionale:
        "stage_names": [stage.get("stage") for stage in stage_data_list]
    }


def build_sddp_dimensions(
        stage_data_list,
        scenarios_info,
        state_info = None,
        num_sub_blocks_per_stage=1,
) -> Dict[str, Any]:
    """
    Calcola le dimensioni necessarie per il costruttore di SDDPBlock
    """
    time_horizon = len(stage_data_list)

    # Estraiamo informazioni da scenarios_info
    sub_scenario_size = list(scenarios_info["sub_scenario_size"])
    number_scenarios = scenarios_info["number_scenarios"]
    scenario_size = scenarios_info["scenario_size"]

    # Determiniamo se tutti gli stadi hanno la stessa dimensione
    uniform = len(set(sub_scenario_size)) == 1

    if uniform:
        # Caso uniforme: SubScenarioSize è scalare, i random data groups
        # vengono letti normalmente.
        sub_scenario_size_out = sub_scenario_size[0]
        size_random_data_groups = stage_data_list[0]["size_random_data_groups"]
        num_random_data_groups = len(size_random_data_groups)
    else:
        # Caso non uniforme: SubScenarioSize è un array di lunghezza TimeHorizon,
        # e SMS++ ignora SizeRandomDataGroups (si usa un unico gruppo).
        sub_scenario_size_out = np.asarray(sub_scenario_size, dtype=np.uint32)
        size_random_data_groups = [scenario_size]
        num_random_data_groups = 1

    # Gestione variabili di stato
    if state_info is None:
        state_size = 0
        initial_state = np.array([], dtype = float)
        admissible_state = np.array([], dtype = float)
        admissible_state_size = 0
        initial_state_size = 0
    else:
        state_size = state_info["state_size"]
        initial_state = np.asarray(state_info["initial_state"], dtype=float)
        admissible_state = np.asarray(state_info["admissible_state"], dtype=float)
        initial_state_size = initial_state.size  # o len(initial_state) se 1D
        admissible_state_size = admissible_state.size

    return {
    "TimeHorizon": time_horizon,
    "NumSubBlocksPerStage": num_sub_blocks_per_stage,
    "NumberScenarios": number_scenarios,
    "ScenarioSize": scenario_size,
    "SubScenarioSize": sub_scenario_size_out,
    "NumberRandomDataGroups": num_random_data_groups,
    "SizeRandomDataGroups": size_random_data_groups,
    "AdmissibleStateSize": admissible_state_size,
    "InitialStateSize": initial_state_size,
    "StateSize": state_size,
    "InitialState": initial_state,
    "AdmissibleState": admissible_state,
}

def make_sddp_periods(
        n,
        snapshots_per_stage: int,
        names: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Data una rete PyPSA n e un intero snapshots_per_stage, produrremo una lista di
    periods - uno per stadio - ciascuno con un name e la lista degli snapshosts che
    gli appartengono.
    """
    all_snapshots = pd.Index(n.snapshots)
    total = len(all_snapshots)

    if total == 0:
        raise ValueError(
            "La rete non ha snapshots."
        )

    if isinstance(snapshots_per_stage, bool) or not isinstance(snapshots_per_stage, int):
        raise ValueError(
            "Il parametro snapshots_per_stage deve essere un intero."
        )
    if snapshots_per_stage < 1:
        raise ValueError(
            "Il parametro snapshots_per_stage deve essere un intero POSITIVO."
        )
    if snapshots_per_stage > total:
        raise ValueError(
            f"snapshots_per_stage ({snapshots_per_stage}) non può superare il numero totale di snapshots ({total}) della rete."
        )
    if total % snapshots_per_stage != 0:
        raise ValueError(
            f"Il numero totale di snapshot ({total}) non è divisibile per il parametro "
            f"snapshots_per_stage inserito dall'utente ({snapshots_per_stage}). "
        )

    num_stages = total // snapshots_per_stage
    if names is not None and len(names) != num_stages:
        raise ValueError(
            f"names ({names}) non è della lunghezza corretta. Deve essere uguale a "
            f"num_stages ({num_stages})."
        )
    if names is None:
        names = [f"stage_{i}" for i in range(num_stages)]

    periods = []

    for i in range(num_stages):
        start = i * snapshots_per_stage
        end = start + snapshots_per_stage
        periods.append({"name": str(names[i]), "snapshots": list(all_snapshots[start:end])})

    return periods

def _coerce_snapshot_labels(
        labels,
        all_snapshots,
        stage_name) -> pd.Index:
    """
    Obiettivo: prendere una lista di etichette di snapshot e restituirle come
    pd.Index con lo stesso tipo di n.snapshots. Serve a evitare che confronti
    tra stringhe e DatatimeIndex falliscano silenziosamente.
    """
    # Normalizziamo l'input in un pd.Index per poter usare .isin()
    candidate = pd.Index(list(labels))

    # Caso in cui le etichette sono già del tipo giusto
    if candidate.isin(all_snapshots).all():
        return candidate

    # Caso in cui tentiamo la conversione a datatime
    # pd.to_datetime può sollevare un'eccezione su input non convertibili
    # Usiamo, quindi, try/except
    if not isinstance(all_snapshots, pd.MultiIndex):
        try:
            converted = pd.Index(pd.to_datetime(candidate))
        except (TypeError, ValueError):
            converted = None

        # Se la conversione è riuscita e tutti gli elementi sono presenti
        # in all_snapshots, possiamo usare la versione convertita.
        if converted is not None and converted.isin(all_snapshots).all():
            return converted

    # Nessuna conversione funziona -> errore con dettagli utili.
    missing = [s for s in candidate if s not in all_snapshots]
    raise ValueError(
        f"Lo stadio {stage_name!r} contiene {len(missing)} snapshot non "
        f"presenti in n.snapshots. Primi esempi: {missing[:5]}"
    )

def validate_sddp_stage_partition(
        all_snapshots: pd.Index,
        stage_snapshots: Mapping[Any, pd.Index],
        strict_coverage: bool = True,
) -> Dict[str, Any]:
    """
    Verifica che `stage_snapshots` sia una partizione sensata di
    `all_snapshots`.

    La partizione è "sensata" se:
      1. non è vuota;
      2. ogni stadio contiene almeno uno snapshot;
      3. nessuno snapshot compare in più di uno stadio;
      4. (opzionale) l'unione degli snapshot copre interamente `all_snapshots`.

    NON impone che tutti gli stadi abbiano la stessa lunghezza: SMS++ accetta
    stadi di dimensione diversa, quindi non è compito di questa funzione
    rifiutarli.

    Parameters
    ----------
    all_snapshots : pd.Index
        Tutti gli snapshot della rete (`pd.Index(n.snapshots)`).
    stage_snapshots : Mapping[Any, pd.Index]
        Dizionario {nome_stadio: pd.Index(snapshot dello stadio)}.
    strict_coverage : bool, default True
        Se True, verifica che ogni snapshot di `all_snapshots` appartenga
        ad almeno uno stadio. Se False, permette partizioni parziali.

    Returns
    -------
    dict
        Riepilogo con:
        - "lengths": {nome_stadio: numero di snapshot}
        - "covered": numero totale di snapshot coperti (contando i duplicati)
        - "total":   numero totale di snapshot in `all_snapshots`

    Raises
    ------
    ValueError
        Se una delle condizioni 1–4 è violata.
    """

    # 'stage_snapshots' non deve essere vuoto
    # Un dizionario vuoto significa che non c'è nessuno stadio da costruire.
    # Questo è quasi sempre un errore a monte (periods malformati o
    # snapshots_per_stage troppo grande), quindi meglio fallire subito.
    if not stage_snapshots:
        raise ValueError(
            "validate_sddp_stage_partition: 'stage_snapshots' è vuoto, "
            "non ci sono stadi da validare."
        )

    # `seen` conterrà tutti gli snapshot di tutti gli stadi, nell'ordine in
    # cui li incontriamo. Serve per il controllo di sovrapposizione.
    seen: List[Any] = []

    # `lengths` conterrà il numero di snapshot per ogni stadio, per il
    # riepilogo di ritorno.
    lengths: Dict[Any, int] = {}

    # Ogni stadio deve avere almeno uno snapshot
    for name, snaps in stage_snapshots.items():

        snaps = pd.Index(snaps)

        if len(snaps) == 0:
            raise ValueError(
                f"validate_sddp_stage_partition: lo stadio {name!r} "
                f"non contiene nessuno snapshot."
            )

        lengths[name] = int(len(snaps))

        # Accumuliamo gli snapshot per i controlli successivi. Nota: usiamo
        # `list(snaps)` perché `snaps` è un pd.Index e vogliamo una lista
        # di valori "nudi" per poter usare set() e Series.
        seen.extend(list(snaps))

    # Nessuna sovrapposizione
    # Se `len(seen) != len(set(seen))`, ci sono valori duplicati, cioè
    # almeno uno snapshot compare in più di uno stadio.
    if len(seen) != len(set(seen)):
        # Per dare un messaggio utile, identifichiamo i duplicati.
        # `pd.Series(seen).duplicated()` restituisce un array booleano:
        # True in posizione i se seen[i] è già comparso prima.
        seen_series = pd.Series(seen)
        # `pd.unique` sui valori duplicati ci dà l'elenco dei valori che
        # compaiono più di una volta (una volta ciascuno).
        duplicated = list(pd.unique(seen_series[seen_series.duplicated()]))

        raise ValueError(
            f"validate_sddp_stage_partition: {len(duplicated)} snapshot "
            f"sono assegnati a più di uno stadio. "
            f"Primi esempi: {duplicated[:5]}"
        )

    # Copertura completa di all_snapshots
    # Solo se l'utente ha chiesto strict_coverage=True.
    if strict_coverage:
        # Trasformiamo `seen` in un set per lookup O(1).
        seen_set = set(seen)

        # Cerchiamo gli snapshot di all_snapshots che non compaiono in seen_set.
        missing = [s for s in all_snapshots if s not in seen_set]

        if missing:
            raise ValueError(
                f"validate_sddp_stage_partition: {len(missing)} snapshot "
                f"di n.snapshots non appartengono a nessuno stadio. "
                f"Primi esempi: {missing[:5]}"
            )

    return {
        "lengths": lengths,
        "covered": int(len(seen)),
        "total": int(len(all_snapshots)),
    }

def get_sddp_stage_snapshots(
        n,
        stage_names = None,
        stochastic_parameters = None,
):
    """
    Restituisce un {stage_name: pd.Index(snapshots)} a partire da stochastic_parameters.
    Internamente chiama make_sddp_periods e valida il risultato

    Parameters
    ----------
    n: rete pypsa.
    stage_names: opzionale. Se None, vengono derivati dai periods generati. Se fornito,
    filtra solo quei nomi.
    stochastic_parameters: dict con snapshots_per_stage.

    Returns
    -------
    Dict {stage_name: pd.Index(snapshots)}.
    """

    # Normalizzazione parametri SDDP
    # Otteniamo e validiamo snapshosts_per_stage
    sp = normalize_sddp_parameters(stochastic_parameters)

    # Generiamo i periods: lista di {name, snapshots}
    periods = make_sddp_periods(n, sp["snapshots_per_stage"])

    # Indice globale degli snapshot
    all_snapshots = pd.Index(n.snapshots)

    # Indicizziamo i periods per nome, per efficientare la ricerca
    by_name = {p["name"]: p for p in periods}

    # Prendiamo tutti gli stadi generati, casomai l'utente non li avesse specificati
    if stage_names is None:
        stage_names = [p["name"] for p in periods]

    # Inizializziamo il dizionario di output
    stage_snapshots: Dict[Any, pd.Index] = {}

    # Per ogni stadio richiesto, recuperiamo gli snapshots dal periodo corrispondente
    # Inoltre, li convertiamo nel tipo giusto
    for name in stage_names:
        if name not in by_name:
            raise ValueError(
                f"Lo stadio {name!r} non compare tra i periods generati. "
                f"Nomi disponibili: {sorted(by_name)}."
            )
        stage_snapshots[name] = _coerce_snapshot_labels(by_name[name]["snapshots"], all_snapshots, name)

    # Controlliamo che la mappa sia una partizione sensata
    validate_sddp_stage_partition(all_snapshots, stage_snapshots)

    return stage_snapshots

def get_sddp_stage_time_horizons(
        stage_snapshots: Mapping[Any, pd.Index],
) -> Dict[Any, int]:
    """
    Restituisce il numero di snapshot per ogni stadio.

    Questo valore è il "TimeHorizon" del singolo stadio, cioè l'orizzonte
    operativo del suo UCBlock. Si distingue dal TimeHorizon dell'SDDPBlock,
    che invece è il numero di stadi.

    Parameters
    ----------
    stage_snapshots : Mapping[Any, pd.Index]
        Dizionario {nome_stadio: pd.Index(snapshot dello stadio)},
        tipicamente il risultato di `get_sddp_stage_snapshots`.

    Returns
    -------
    dict
        {nome_stadio: numero di snapshot}.
    """
    # Per ogni coppia (nome, snapshot), convertiamo gli snapshot in pd.Index
    # (difensivo: potrebbero arrivare come lista) e prendiamo la lunghezza.
    # int() garantisce che il valore sia un intero Python e non un numpy
    # int, che è più comodo per la serializzazione successiva.
    return {
        name: int(len(pd.Index(snaps)))
        for name, snaps in stage_snapshots.items()
    }

# ---------------------------------------------
# --------- Parser per Block_i.nc4 ------------
# ---------------------------------------------

def parse_block_structure(block_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Legge un singolo Block_i.nc4 e restituisce un dizionario completo con
    tutta la struttura

    ----------
    Parameters
    block_path: Union[str, Path]
        Percorso del file .nc4 che contiene un UCBlock esterno.

    ---------
    Returns
    {
        # Generale
        "block_id": "0",
        "time_horizon": 4,
        "number_units": 40,
        "number_electrical_generators": 41,
        "number_nodes": 3,
        "number_primary_zones": 3,
        "number_lines": 3,

        # Per il top-level AbstractPath
        "hydro_system_index": 0,          # = second_group_index
        "polyhedral_index": 1,            # = third_group_index

        # Per il Benders AbstractPath
        "hydro_unit_blocks": [            # lista di dict, un elemento per HydroUnitBlock
            {"index": 0, "name": "SS1_Z1_0", "n_reservoirs": 1, "n_arcs": 2, "initial_volumetric": [1000.0], "min_volumetric": [...], "max_volumetric": [...],},
        ],
        "n_hydro_units": 1,
        "n_reservoirs_total": 1,

        # Per il data mapping
        "unit_blocks_by_type": {
            "ThermalUnitBlock":      [(1, "TU1_Z1_0"), ..., (25, "TU10_Z3_0")],
            "IntermittentUnitBlock": [(26, "RES1_Z1_0"), ..., (34, "RES3_Z3_0")],
            "BatteryUnitBlock":      [(35, "STS_Z1_0"), (36, "STS_Z2_0")],
            "SlackUnitBlock":        [(37, "SlackUnit_Z1"), (38, "SlackUnit_Z2"), (39, "SlackUnit_Z3")],
        },

        # Per il parsing soluzione
        "unit_block_names": {             # mappa nome unitblock -> indice
            "SS1_Z1_0": 0,
            "TU1_Z1_0": 1,
            ...,
        },
    }
    """

    # ================================================================
    # Helper interni
    # ================================================================

    def _dim_size(group, dim_name: str, default=None):
        """Legge una dimensione, con default opzionale."""
        if dim_name not in group.dimensions:
            if default is not None:
                return default
            raise KeyError(
                f"Dimensione {dim_name!r} non trovata in {group.path!r}."
            )
        return int(group.dimensions[dim_name].size)

    def _display_name(group, fallback: str) -> str:
        """Nome leggibile di un gruppo: :name se c'è, altrimenti fallback."""
        return group.getncattr("name") if "name" in group.ncattrs() else fallback

    def _read_variable(group, var_name: str, default=None):
        """
        Legge una variabile numerica da un gruppo netCDF4 e la restituisce
        come lista 1D di float.
        """
        if var_name not in group.variables:
            return default

        # group.variables[var_name][:] restituisce un ndarray (o un
        # masked array) con la stessa forma della variabile nel file.
        # Es. InitialVolumetric ha shape (NumberReservoirs,) quindi
        # otteniamo un array 1D. Per variabili scalari (shape ())
        # otteniamo un array 0-D, che .ravel() porta a shape (1,)
        arr = np.asarray(group.variables[var_name][:], dtype=float)

        # .ravel() appiattisce qualsiasi forma a 1D; .tolist() converte
        # in lista Python di float. Così il chiamante riceve sempre una
        # lista, indipendentemente dalla forma originale.
        return arr.ravel().tolist()

    # ================================================================
    # Apertura file
    # ================================================================
    with netCDF4.Dataset(block_path, "r") as ds:

        if not ds.groups:
            raise ValueError(f"Il file {block_path!r} non contiene gruppi.")

        # Il primo gruppo radice. Il nome cambia ("Block_0"), ma ce n'è
        # sempre uno solo in questo formato.
        root_name = next(iter(ds.groups))
        root = ds.groups[root_name]

        # Controllo tipo
        root_type = root.getncattr("type")
        if root_type != "UCBlock":
            raise ValueError(
                f"Il blocco radice {root_name!r} ha tipo {root_type!r}, "
                f"atteso 'UCBlock'."
            )

        # Dimensioni generali del UCBlock
        block_id = root.getncattr("id") if "id" in root.ncattrs() else None

        time_horizon = _dim_size(root, "TimeHorizon")
        number_units = _dim_size(root, "NumberUnits")
        number_electrical_generators = _dim_size(root, "NumberElectricalGenerators")
        number_nodes = _dim_size(root, "NumberNodes")
        number_primary_zones = (
            _dim_size(root, "NumberPrimaryZones")
            if "NumberPrimaryZones" in root.dimensions
            else None
        )
        number_lines = _dim_size(root, "NumberLines", default=0)

        # ================================================================
        # Contenitori
        # ================================================================
        unit_blocks_by_type: Dict[str, List[Tuple[int, str]]] = {}
        unit_block_names: Dict[str, int] = {}

        hydro_system_index: Optional[int] = None
        hydro_unit_blocks: List[Dict[str, Any]] = []
        polyhedral_index: Optional[int] = None
        n_hydro_units = 0
        n_reservoirs_total = 0

        # ================================================================
        # Iteriamo i figli del root (UnitBlock_i) usando la posizione come
        # indice, non il suffisso numerico del nome. Per UnitBlock_i le
        # due coincidono, ma questa scelta è più robusta.
        # ================================================================
        for pos, (child_name, child) in enumerate(root.groups.items()):

            child_type = child.getncattr("type")

            # Il nome leggibile può essere :name oppure il nome del gruppo.
            child_display_name = _display_name(child, child_name)

            # Registra sempre la mappa nome -> posizione per Fase 7.
            unit_block_names[child_display_name] = pos

            # --------------------------------------------------------
            # Caso speciale: HydroSystemUnitBlock (non va in
            # unit_blocks_by_type, è un contenitore).
            # --------------------------------------------------------
            if child_type == "HydroSystemUnitBlock":
                hydro_system_index = pos

                # Cross-check con la dimensione dichiarata.
                declared_n_hydro = _dim_size(
                    child, "NumberHydroUnits", default=None
                )

                # Itera i figli dell'HydroSystemUnitBlock. Anche qui
                # uso la posizione, non il suffisso del nome: il
                # PolyhedralFunctionBlock non ha suffisso.
                for sub_pos, (sub_name, sub) in enumerate(child.groups.items()):
                    sub_type = sub.getncattr("type")

                    # --- HydroUnitBlock -------------------------------
                    if sub_type == "HydroUnitBlock":
                        hydro_name = _display_name(sub, sub_name)
                        n_reservoirs = _dim_size(
                            sub, "NumberReservoirs", default=1
                        )
                        n_arcs = _dim_size(sub, "NumberArcs", default=0)

                        # Leggiamo i volumi caratteristici del bacino

                        initial_volumetric = _read_variable(
                            sub, "InitialVolumetric" # Volume iniziale, usato per costruire InitialState e AdmissibleState
                        )

                        # MinVolumetric e MaxVolumetric definiscono il range ammissibile per validare poi AdmissibleState

                        min_volumetric = _read_variable(
                            sub, "MinVolumetric"
                        )
                        max_volumetric = _read_variable(
                            sub, "MaxVolumetric"
                        )

                        hydro_unit_blocks.append({
                            # `index` = posizione nell'HydroSystemUnitBlock:
                            # è quello che serve al Benders AbstractPath.
                            "index": sub_pos,
                            "name": hydro_name,
                            "n_reservoirs": n_reservoirs,
                            "n_arcs": n_arcs,
                            "initial_volumetric": initial_volumetric,
                            "min_volumetric": min_volumetric,
                            "max_volumetric": max_volumetric,
                        })
                        n_hydro_units += 1
                        n_reservoirs_total += n_reservoirs

                        # Mappa nome -> posizione (utile se in Fase 7
                        # cercherai un HydroUnitBlock per nome).
                        unit_block_names[hydro_name] = sub_pos

                    # --- PolyhedralFunctionBlock -----------------------
                    elif sub_type == "PolyhedralFunctionBlock":
                        # L'indice è la POSIZIONE, non un suffisso (che
                        # non esiste). Sarà il third_group_index.
                        polyhedral_index = sub_pos

                # Cross-check: se il file dichiara NumberHydroUnits,
                # deve corrispondere al numero di HydroUnitBlock trovati.
                if declared_n_hydro is not None and \
                        declared_n_hydro != n_hydro_units:
                    raise ValueError(
                        f"In {child_name!r} la dimensione NumberHydroUnits "
                        f"dichiara {declared_n_hydro} unità idro, ma ne ho "
                        f"trovate {n_hydro_units}."
                    )

                # L'HydroSystemUnitBlock è stato processato: non va in
                # unit_blocks_by_type. Passa al prossimo figlio del root.
                continue

            # --------------------------------------------------------
            # Altri UnitBlock. Interessano solo indice (posizione) e
            # nome per la Fase 3.
            # --------------------------------------------------------
            if child_type in (
                "ThermalUnitBlock",
                "IntermittentUnitBlock",
                "BatteryUnitBlock",
                "SlackUnitBlock",
            ):
                unit_blocks_by_type.setdefault(child_type, []).append(
                    (pos, child_display_name)
                )

        # ================================================================
        # Controlli finali
        # ================================================================
        if hydro_system_index is None:
            raise ValueError("HydroSystemUnitBlock non trovato nel UCBlock.")
        if polyhedral_index is None:
            raise ValueError(
                "PolyhedralFunctionBlock non trovato nell'HydroSystemUnitBlock."
            )

        # ================================================================
        # Output
        # ================================================================
        return {
            # Generale
            "block_id": block_id,
            "time_horizon": time_horizon,
            "number_units": number_units,
            "number_electrical_generators": number_electrical_generators,
            "number_nodes": number_nodes,
            "number_primary_zones": number_primary_zones,
            "number_lines": number_lines,

            # Per il top-level AbstractPath
            "hydro_system_index": hydro_system_index,  # = second_group_index
            "polyhedral_index": polyhedral_index,      # = third_group_index

            # Per il Benders AbstractPath
            "hydro_unit_blocks": hydro_unit_blocks,
            "n_hydro_units": n_hydro_units,
            "n_reservoirs_total": n_reservoirs_total,

            # Per il data mapping
            "unit_blocks_by_type": unit_blocks_by_type,

            # Per il parsing soluzione
            "unit_block_names": unit_block_names,
        }


def parse_all_stage_blocks(stage_block_paths: Sequence[Union[str, Path]]) -> List[Dict[str, Any]]:
    """
    Chiama parse_block_structure su ognuno dei Block_i e restituisce
    la lista dei dizionari, nello stesso ordine dei path in input.
    Affinché funzioni, l'utente deve indicare i file esterni tramite path espliciti.
    """
    # Verifichiamo che la lista non sia vuota
    if not stage_block_paths:
        raise ValueError(
            "La lista contenente i path non può essere vuota"
        )

    # Accumulatore che conterrà un dizionario per ogni riga
    acc: List[Dict[str, Any]] = []

    for path in stage_block_paths:
        # parse_block_structure apre il file, lo legge e lo chiude
        try:
            structure = parse_block_structure(path)
        except (FileNotFoundError, OSError) as exc:
            raise FileNotFoundError(
                f"Impossibile leggere il Block allo stage corrispondente "
                f"al path {path!r}: {exc}"
            ) from exc
        acc.append(structure)

    return acc

def validate_stage_structures(
    stage_structures: Sequence[Dict[str, Any]],
) -> None:
    """
    Verifica che le strutture dei vari stage siano coerenti tra loro.

    Controlli effettuati:
    - la lista non è vuota;
    - i campi dimensionali (time_horizon, number_nodes, ...) sono
      uguali su tutti gli stage;
    - hydro_system_index e polyhedral_index sono uguali su tutti gli
      stage (servono all'AbstractPath top-level);
    - n_hydro_units e n_reservoirs_total sono uguali (servono a
      StateSize);
    - ogni tipo di UnitBlock compare su tutti gli stage con lo stesso
      numero di elementi, e gli stessi nomi;
    - gli HydroUnitBlock hanno la stessa forma (stessi reservoir,
      stessi archi) su tutti gli stage;
    - i block_id sono univoci (non è un errore se differiscono, anzi:
      devono differire).

    Solleva ValueError al primo problema trovato.
    """
    # ---------------------------------------------------------------
    # Controllo di base: lista non vuota.
    # ---------------------------------------------------------------
    if not stage_structures:
        raise ValueError("La lista in input non può essere vuota.")

    # ---------------------------------------------------------------
    # Helper interno: verifica che un campo scalare sia uguale su
    # tutti gli stage. Restituisce il valore di riferimento.
    # ---------------------------------------------------------------
    def _check_uniform(key: str) -> Any:
        reference = stage_structures[0][key]
        for i, structure in enumerate(stage_structures[1:], start=1):
            if structure[key] != reference:
                raise ValueError(
                    f"Il campo {key!r} è incoerente tra gli stage: "
                    f"lo stage 0 ha {reference!r}, lo stage {i} ha "
                    f"{structure[key]!r}."
                )
        return reference

    # ---------------------------------------------------------------
    # Campi scalari che devono essere uguali su tutti gli stage.
    # ---------------------------------------------------------------
    _check_uniform("time_horizon")
    _check_uniform("number_nodes")
    _check_uniform("number_primary_zones")
    _check_uniform("number_lines")

    _check_uniform("hydro_system_index")
    _check_uniform("polyhedral_index")

    _check_uniform("n_hydro_units")
    _check_uniform("n_reservoirs_total")

    # ---------------------------------------------------------------
    # block_id: devono essere TUTTI diversi (ogni stage è un file
    # separato, quindi ha il suo id).
    # ---------------------------------------------------------------
    ids = [s["block_id"] for s in stage_structures]
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"I block_id degli stage non sono univoci: {ids}. "
            "Ogni Block_i.nc4 dovrebbe avere un id distinto."
        )

    # ---------------------------------------------------------------
    # unit_blocks_by_type: stessi tipi, stesso numero di elementi
    # per tipo, stessi nomi.
    # ---------------------------------------------------------------
    reference_types = stage_structures[0]["unit_blocks_by_type"]

    for i, structure in enumerate(stage_structures[1:], start=1):
        current_types = structure["unit_blocks_by_type"]

        # Stesso insieme di tipi.
        if set(current_types) != set(reference_types):
            raise ValueError(
                f"I tipi di UnitBlock differiscono tra lo stage 0 e lo "
                f"stage {i}: stage 0 ha {sorted(reference_types)}, "
                f"stage {i} ha {sorted(current_types)}."
            )

        # Stesso numero di elementi e stessi nomi per ogni tipo.
        for tipo in reference_types:
            ref_items = reference_types[tipo]
            cur_items = current_types[tipo]

            if len(ref_items) != len(cur_items):
                raise ValueError(
                    f"Il tipo {tipo!r} ha un numero diverso di elementi "
                    f"tra lo stage 0 ({len(ref_items)}) e lo stage {i} "
                    f"({len(cur_items)})."
                )

            # Confronto dei nomi come insiemi: l'ordine può variare
            # tra stage, purché i nomi coincidano.
            ref_names = {name for _, name in ref_items}
            cur_names = {name for _, name in cur_items}
            if ref_names != cur_names:
                only_ref = ref_names - cur_names
                only_cur = cur_names - ref_names
                raise ValueError(
                    f"I nomi degli UnitBlock di tipo {tipo!r} differiscono "
                    f"tra lo stage 0 e lo stage {i}. "
                    f"Solo nello stage 0: {sorted(only_ref)}. "
                    f"Solo nello stage {i}: {sorted(only_cur)}."
                )

    # ---------------------------------------------------------------
    # hydro_unit_blocks: stessa "forma" (nome, reservoir, archi),
    # ignorando l'indice che potrebbe variare.
    # ---------------------------------------------------------------
    def _hydros_without_index(structure):
        return [
            {k: v for k, v in h.items() if k not in ("index", "initial_volumetric")}
            for h in structure["hydro_unit_blocks"]
        ]

    reference_hydros = _hydros_without_index(stage_structures[0])

    for i, structure in enumerate(stage_structures[1:], start=1):
        current_hydros = _hydros_without_index(structure)
        if current_hydros != reference_hydros:
            raise ValueError(
                f"Gli HydroUnitBlock differiscono tra lo stage 0 e lo "
                f"stage {i}: stage 0 ha {reference_hydros}, "
                f"stage {i} ha {current_hydros}."
            )

# -------------------------------------------------------
# -------------------- STATE INFO -----------------------
# -------------------------------------------------------

def build_sddp_state_info(
        structure: Mapping[str, Any],
        time_horizon: int,
) -> Dict[str, Any]:
    """
    Costruisce le informazioni di stato per l'SDDPBlock.

    Lo stato dello stage è il vettore dei volumi iniziali dei bacini.
    Nel caso multi-bacino, StateSize è la somma di n_reservoirs su tutti
    gli HydroUnitBlock e i valori sono concatenati nello stesso ordine.

    Parameters
    ----------
    structure : Mapping[str, Any]
        Struttura dello stage dal parser. Deve contenere
        "hydro_unit_blocks", ciascuno con "n_reservoirs" e
        "initial_volumetric".
    time_horizon : int
        Numero di stadi dell'SDDPBlock. Serve per costruire
        AdmissibleState come ripetizione di InitialState.

    Returns
    -------
    dict
        - "state_size": int, dimensione dello stato
        - "initial_state": np.ndarray 1D, valori iniziali
        - "admissible_state": np.ndarray 1D, valori ammissibili
          (replicati TimeHorizon volte)
    """
    # Verifichiamo gli input
    if time_horizon < 1:
        raise ValueError(
            f"{time_horizon} deve essere maggiore o uguale a 1"
        )

    if "hydro_unit_blocks" not in structure:
        raise KeyError(
            f"{structure} non contiene 'hydro_unit_blocks'"
        )

    hydro_units = structure["hydro_unit_blocks"]

    # StateSize = somma di n_reservoirs su tutti gli HydroUnitBlock
    state_size = sum(int(h["n_reservoirs"]) for h in hydro_units)

    # InitialState = concatenazione di InitialVolumetric
    initial_state: List[float] = []
    for h in hydro_units:
        vals = h.get("initial_volumetric")
        if vals is None:
            raise ValueError(
                f"HydroUnitBlock {h.get('name')!r} non ha "
                "InitialVolumetric: il parser non lo ha letto dal "
                "Block_i.nc4, oppure la variabile manca nel file."
            )
        if len(vals) != int(h["n_reservoirs"]):
            raise ValueError(
                f"HydroUnitBlock {h.get('name')!r}: InitialVolumetric "
                f"ha {len(vals)} valori, attesi {h['n_reservoirs']}."
            )
        initial_state.extend(float(v) for v in vals)


    # AdmissibleState = InitialState x TimeHorizon
    admissible_state = initial_state * time_horizon

    return {
        "state_size": int(state_size),
        "initial_state": np.asarray(initial_state, dtype=float),
        "admissible_state": np.asarray(admissible_state, dtype=float),
    }

# ------------------------------------------------------
# ------------- SLICING RETE PER STAGE ----------------
# ------------------------------------------------------

def slice_network_for_stage(
        n,
        stage_snapshots: pd.Index,
) -> "pypsa.Network":
    """
    Restituisce una rete PyPSA deterministica contenente solo gli
    snapshot dello stage specificato.

    La rete risultante è una copia della rete base (pimo scenario se
    stocastica), con n.snapshots filtrato a stage_snapshots e con tutte
    le serie temporali filtrate di conseguenza.
    Gli asset restano gli stessi.

    Parameters
    ----------
    n: pypsa.Network
    stage_snapshots: pd.Index
        Snapshot dello stage.

    Returns
    -------
    pypsa.Network
    """
    # Se stocastica, prendiamo il primo scenario, altrimenti è già n
    n_base = su.get_base_scenario_network(n)

    # Copiamo per non modificare l'originale
    n_stage = n_base.copy()

    # Filtriamo gli snapshot
    n_stage.set_snapshots(pd.Index(stage_snapshots))

    return n_stage

# -------------------------------------------------------
# ---------------- CHECK RETE PER SDDP -----------------
# -------------------------------------------------------

def network_has_hydro_storage(n) -> bool:
    """
    Verifica se la rete ha almeno una StorageUnit con carrier idroelettrico.

    I carrier riconosciuti come idroelettrici sono gli stessi che
    `iterate_components` (in transformation.py) mappa a HydroUnitBlock,
    cioè "hydro" e "PHS".

    Parameters
    ----------
    n : pypsa.Network
        Rete PyPSA (stocastica o deterministica).

    Returns
    -------
    bool
        True se almeno una StorageUnit ha carrier idroelettrico.
    """
    n_base = su.get_base_scenario_network(n)

    if n_base.storage_units.empty:
        return False

    hydro_carriers = {"hydro", "PHS"}
    carriers = (
        n_base.storage_units["carrier"]
        .astype(str)
        .str.strip()
    )
    return bool(carriers.isin(hydro_carriers).any())

# -------------------------------------------------------
# ------------------- BENDERS DATA ----------------------
# -------------------------------------------------------

def build_sddp_benders_function_data(
        structure: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Costruisce le informazioni per il blocco BendersBFunction.

    BendersBFunction implementa la mappattura affine:
    M(x) = A · x + b
    dove:
    - x è lo stato in ingresso (vettore di StateSize componenti, i volumi iniziali dei bacini)
    - A è una matrice NumRow × NumVar
    - b è un vettore di NumRow componenti

    Parameters
    ----------
    structure: Mapping[str, Any]
        Dizionario restituito da parse_block_structure per lo stage corrente

    Returns
    -------
    dict:
        dizionario composto da nove chiavi:
            {
                "NumVar": int, -> Numero di colonne di A (= dimensione dello stato),
                "NumRow": int, -> Numero di righe di A (= numero di vincoli modificati),
                "NumNonzero": int, -> Numero di elementi non-zero di A
                "Column": np.ndarray, -> Indici di colonna dei non-zero
                "A": np.ndarray, -> Valori dei non-zero
                "NumNonzeroAtRow": np.ndarray, -> Quanti non-zero ci sono in ciascuna riga
                "b": np.ndarray, -> Termini costanti
                "ConstraintSide": np.ndarray, -> Lato modificato per ogni riga
            }
    """
    # Verifichiamo i valori dell'input
    if "hydro_unit_blocks" not in structure:
        raise KeyError(
            f"{structure} non contiene 'hydro_unit_blocks'"
        )

    hydro_units = structure["hydro_unit_blocks"]

    if not hydro_units:
        raise ValueError(
            "hydro_unit_blocks è vuoto: nessuno stato da modellare"
        )

    n_reservoirs_total = sum(int(h["n_reservoirs"]) for h in hydro_units)

    # Costruiamo Column: gli indici di colonna 0..N-1. range(N) li genera
    # già in ordine, quindi basta convertire in array uint32.
    column = np.array(
        list(range(n_reservoirs_total)),
        dtype=np.uint32,
    )

    # Costruiamo la matrice A
    a_values = np.array(
        [1.0] * n_reservoirs_total,
        dtype=float,
    )

    # NumNonzeroAtRow: un solo non-zero per riga.
    num_nonzero_at_row = np.array(
        [1] * n_reservoirs_total,
        dtype=np.uint32,
    )

    # Costruiamo b
    # b = 0 staticamente: l'inflow all'istante 0 viene scritto
    # in b dal DataMapping BendersBFunction::modify_constants a
    # runtime, non da noi. Quindi qui mettiamo zero come
    # placeholder, e il solver lo sovrascriverà applicando il
    # DataMapping.
    b_values = np.array(
        [0.0] * n_reservoirs_total,
        dtype=float,
    )

    # Costruiamo ConstraintSide
    # il volume iniziale del bacino entra
    # nel lato destro del proprio vincolo di bilancio idrico
    # (v_0 + flussi = V_init + inflow_0).
    constraint_sides = np.array(
        ["B"] * n_reservoirs_total,
        dtype="S1",
    )

    # Calcoliamo le tre dimensioni
    num_var = int(n_reservoirs_total)
    num_row = int(n_reservoirs_total)
    num_non_zero = int(n_reservoirs_total)

    return {
        "NumVar": num_var,
        "NumRow": num_row,
        "NumNonzero": num_non_zero,
        "Column": column,
        "A": a_values,
        "NumNonzeroAtRow": num_nonzero_at_row,
        "b": b_values,
        "ConstraintSide": constraint_sides,
    }


# -------------------------------------------------------
# ------------------ ABSTRACT PATHS ---------------------
# -------------------------------------------------------

def build_sddp_top_abstract_path(
    n_polyhedral_per_sub_block: int,
    second_group_index: int,
    third_group_index: int
) -> Dict[str, Any]:
    """
    AbstractPath top-level dell'SDDPBlock.

    Ogni path va da un BendersBlock (reference) a una PolyhedralFunction
    (target). La struttura di ogni path è "OBBB" con:

        PathNodeTypes   = "O", "B", "B", "B"
        PathGroupIndices = [MASK, 0, second_group_index, third_group_index]

    I due indici non sono costanti: vanno estratti dalla struttura
    dell'UCBlock esterno (il file Block_i.nc4) tramite il parser.

    - `second_group_index` è la posizione dell'HydroSystemUnitBlock tra
      i figli del UCBlock (tipicamente 0).
    - `third_group_index` è la posizione del PolyhedralFunctionBlock
      dentro l'HydroSystemUnitBlock, che coincide con il numero di
      HydroUnitBlock dell'HydroSystemUnitBlock (tipicamente 1 se c'è
      un solo bacino).

    Parameters
    ----------
    n_polyhedral_per_sub_block : int
        Numero di PolyhedralFunction per sotto-blocco (K nel
        gergo del notebook). Se 0, restituisce un AbstractPath vuoto.
    second_group_index : int
        Posizione dell'HydroSystemUnitBlock tra i figli del UCBlock.
    third_group_index : int
        Posizione del PolyhedralFunctionBlock dentro l'HydroSystemUnitBlock.

    Returns
    -------
    dict
        PathDim, TotalLength, PathStart, PathNodeTypes, PathGroupIndices.
    """
    _UINT32_MASK = np.uint32(4294967295)

    if n_polyhedral_per_sub_block < 0:
        raise ValueError("n_polyhedral_per_sub_block devono essere >= 0.")

    num_paths = n_polyhedral_per_sub_block
    if num_paths == 0:
        return {
            "PathDim": 0,
            "TotalLength": 0,
            "PathStart": np.array([], dtype=np.uint32),
            "PathNodeTypes": np.array([], dtype="S1"),
            "PathGroupIndices": np.array([], dtype=np.uint32),
        }

    # Ogni path ha 4 nodi: O, B, B, B
    nodes_per_path = 4

    # PathStart: [0, 4, 8, 12, ...]
    path_start = np.arange(
        0, nodes_per_path * num_paths, nodes_per_path, dtype=np.uint32
    )

    # PathNodeTypes: "OBBB" ripetuto num_paths volte
    path_node_types = np.tile(
        np.array(["O", "B", "B", "B"], dtype="S1"),
        num_paths,
    )

    # PathGroupIndices: [MASK, 0, 0, 3] ripetuto num_paths volte
    path_group_indices = np.tile(
        np.array([_UINT32_MASK, 0, second_group_index, third_group_index], dtype=np.uint32),
        num_paths,
    )

    return {
        "PathDim": int(num_paths),
        "TotalLength": int(nodes_per_path * num_paths),
        "PathStart": path_start,
        "PathNodeTypes": path_node_types,
        "PathGroupIndices": path_group_indices,
    }

def build_sddp_benders_abstract_path(
    hydro_unit_indices: Sequence[int],
) -> Dict[str, Any]:
    """
    AbstractPath della BendersBFunction: un percorso per ogni bacino.

    Ogni percorso ha 3 nodi:

        PathNodeTypes       = "B", "B", "C"
        PathGroupIndices    = [0, k, 0]
        PathElementIndices  = [MASK, MASK, 0]

    dove `k` è la posizione dell'HydroUnitBlock dentro il suo
    HydroSystemUnitBlock (NON l'indice globale dello UnitBlock nel UCBlock).

    Parameters
    ----------
    hydro_unit_indices : Sequence[int]
        Lista esplicita degli indici degli HydroUnitBlock. Ogni valore è
        la posizione dell'HydroUnitBlock dentro il suo
        HydroSystemUnitBlock. Una lista vuota produce un AbstractPath
        vuoto (nessun bacino).

    Returns
    -------
    dict
        PathDim, TotalLength, PathStart, PathNodeTypes, PathGroupIndices,
        PathElementIndices.
    """
    _UINT32_MASK = np.uint32(4294967295)

    # Normalizziamo in lista di int (accetta tuple, array numpy, ecc.).
    indices = [int(i) for i in hydro_unit_indices]

    # Validazione: nessun indice negativo.
    for idx in indices:
        if idx < 0:
            raise ValueError(
                f"Tutti gli indici in 'hydro_unit_indices' devono essere >= 0, "
                f"ricevuto {idx}."
            )

    # Caso degenere: nessun bacino.
    if not indices:
        return {
            "PathDim": 0,
            "TotalLength": 0,
            "PathStart": np.array([], dtype=np.uint32),
            "PathNodeTypes": np.array([], dtype="S1"),
            "PathGroupIndices": np.array([], dtype=np.uint32),
            "PathElementIndices": np.array([], dtype=np.uint32),
        }

    n_paths = len(indices)

    # PathStart: [0, 3, 6, 9, ...]
    path_start = np.arange(0, 3 * n_paths, 3, dtype=np.uint32)

    # PathNodeTypes: "BBC" ripetuto n_paths volte.
    path_node_types = np.tile(
        np.array(["B", "B", "C"], dtype="S1"),
        n_paths,
    )

    # PathGroupIndices: [0, idx, 0] per ogni idx nella lista.
    groups = []
    for idx in indices:
        groups.extend([0, idx, 0])
    path_group_indices = np.array(groups, dtype=np.uint32)

    # PathElementIndices: [MASK, MASK, 0] ripetuto n_paths volte.
    path_element_indices = np.tile(
        np.array([_UINT32_MASK, _UINT32_MASK, 0], dtype=np.uint32),
        n_paths,
    )

    return {
        "PathDim": int(n_paths),
        "TotalLength": int(3 * n_paths),
        "PathStart": path_start,
        "PathNodeTypes": path_node_types,
        "PathGroupIndices": path_group_indices,
        "PathElementIndices": path_element_indices,
    }

def build_sddp_stochastic_block_abstract_path(
    data_mappings: list,
) -> Dict[str, Any]:
    """
    AbstractPath dello StochasticBlock: un percorso per ogni data mapping.

    Struttura per tipo di mapping (dal file di riferimento):
    - UCBlock::set_active_power_demand          -> "OB",    groups [MASK, 0]
    - HydroUnitBlock::set_inflow                -> "OBBB",  groups [MASK, 0, 0, k]
    - IntermittentUnitBlock::set_maximum_power  -> "OBB",   groups [MASK, 0, k]
    - BendersBFunction::modify_constants        -> "O",     groups [MASK]

    Parameters
    ----------
    data_mappings : list
        Lista di data mapping, ognuno un dict con almeno "function_name".
        Per i mapping unitblock, "abstract_paths"[0]["group_indices"][0]
        contiene l'indice dell'UnitBlock come stringa.

    Returns
    -------
    dict
        PathDim, TotalLength, PathStart, PathNodeTypes, PathGroupIndices.
    """
    _UINT32_MASK = np.uint32(4294967295)

    if not data_mappings:
        raise ValueError("data_mappings non può essere vuoto.")

    path_start = []
    node_types = []
    group_indices = []

    cursor = 0

    for mapping in data_mappings:
        fn = mapping["function_name"]

        if fn == "UCBlock::set_active_power_demand":
            nodes = ["O", "B"]
            groups = [_UINT32_MASK, 0]

        elif fn == "HydroUnitBlock::set_inflow":
            # L'indice dell'HydroUnitBlock viene letto dal mapping, come per gli intermittenti
            k = int(mapping["abstract_paths"][0]["group_indices"][0])
            nodes = ["O", "B", "B", "B"]
            groups = [_UINT32_MASK, 0, 0, k]

        elif fn == "IntermittentUnitBlock::set_maximum_power":
            # L'indice dell'UnitBlock è in mapping["abstract_paths"][0]["group_indices"][0]
            k = int(mapping["abstract_paths"][0]["group_indices"][0])
            nodes = ["O", "B", "B"]
            groups = [_UINT32_MASK, 0, k]

        elif fn == "BendersBFunction::modify_constants":
            nodes = ["O"]
            groups = [_UINT32_MASK]

        else:
            raise ValueError(
                f"build_sddp_stochastic_block_abstract_path: "
                f"function_name non supportata: {fn!r}."
            )

        path_start.append(cursor)
        node_types.extend(nodes)
        group_indices.extend(groups)
        cursor += len(nodes)

    return {
        "PathDim": len(data_mappings),
        "TotalLength": cursor,
        "PathStart": np.array(path_start, dtype=np.uint32),
        "PathNodeTypes": np.array(node_types, dtype="S1"),
        "PathGroupIndices": np.array(group_indices, dtype=np.uint32),
    }


# -------------------------------------------------------
# ---------------- PARSER DATA MAPPING ------------------
# -------------------------------------------------------

def build_sddp_demand_mappings(
        *,
        number_nodes: int,
        time_horizon: int,
        dss_start: int,
) -> List[Dict[str, Any]]:
    """
    Produce una lista di NumberNodes dizionari, uno per nodo del UCBlock.
    Ogni dizionario è un data mapping completo.
    Usiamo Subset per il SetForm dei demand e Range per il SetTo

    Parameters
    ----------
    number_nodes: int
        NumberNodes del UCBlock -> Quanti nodi ci sono
    time_horizon: int
        TimeHorizon dello stage -> Quanti snapshot per stage
    dss_start: int
        Offset del parametro demand dentro il vettore di scenario dello stage

    Returns
    ---------
    list of dict
        Una lista di number_nodes data mapping, uno per nodo del UCBlock
    """
    # Verifichiamo che i parametri siano validi
    if number_nodes < 1:
        raise ValueError(
            f"{number_nodes} deve essere maggiore o uguale ad 1"
        )
    if time_horizon < 1:
        raise ValueError(
            f"{time_horizon} deve essere maggiore o uguale ad 1"
        )
    if dss_start < 0:
        raise ValueError(
            f"{dss_start} deve essere maggiore o uguale a 0"
        )

    mappings: List[Dict[str, Any]] = []

    for i in range(number_nodes):
        # Indici per questo nodo
        set_from_start = dss_start + i * time_horizon
        set_from_end = set_from_start + time_horizon
        set_to_start = i * time_horizon
        set_to_end = set_to_start + time_horizon

        # SetFrom è una Subset esplicita: tutti i valori da
        # set_from_start a (set_from_end - 1)
        set_from_elements = list(range(set_from_start, set_from_end))

        # SetTo è una Range: due valori [set_to_start, set_to_end)
        set_to_elements = [set_to_start, set_to_end]

        # set_elements è SetFrom seguito da SetTo
        set_elements = set_from_elements + set_to_elements

        set_size = [time_horizon, 0]

        mappings.append({
            "target": "demand",
            "function_name": "UCBlock::set_active_power_demand",
            "caller": "B",
            "data_type": "D",
            "set_size": set_size,
            "set_elements": set_elements,
            "abstract_paths": [
                {
                    "node_types": "",
                    "group_indices": [],
                    "element_indices": [],
                    "range_indices": [],
                }
            ],
        })

    return mappings

def build_sddp_hydro_mappings(
        *,
        hydro_units: List[Dict[str, Any]],
        time_horizon: int,
        dss_start: int,
) -> List[Dict[str, Any]]:
    """
    Costruisce i data mapping per gli inflow idroelettrici di uno stage SDDP.

    Restituisce una lista di len(hydro_units) dizionari, uno per ogni
    HydroUnitBlock. Ogni dizionario mappa una fetta del vettore di scenario
    dello stage nel campo Inflows del proprio HydroUnitBlock, secondo la
    struttura:

        mapping j (hydro j):
            SetFrom = Range [dss_start + j*N, dss_start + (j+1)*N)
            SetTo   = Range [0, N)
        dove N = n_reservoirs * time_horizon.

    Parameters
    ----------
    hydro_units : list of dict
        Lista di HydroUnitBlock, ciascuno con chiavi "index" (posizione
        dentro il suo HydroSystemUnitBlock), "name", "n_reservoirs",
        "n_arcs". Tipicamente e' structure["hydro_unit_blocks"].
    time_horizon : int
        Numero di snapshot per stage.
    dss_start : int
        Offset del parametro inflow dentro il vettore di scenario dello stage.

    Returns
    -------
    list of dict
        Una lista di len(hydro_units) data mapping.
    """
    # Verifichiamo che l'input sia valido
    if time_horizon < 1:
        raise ValueError(
            f"{time_horizon} deve essere maggiore o uguale ad 1"
        )
    if dss_start < 0:
        raise ValueError(
            f"{dss_start} deve essere maggiore o uguale a 0"
        )

    # Consideriamo il caso vuoto
    if not hydro_units:
        return []

    hydro_mappings: List[Dict[str, Any]] = []

    for j, hydro in enumerate(hydro_units):
        # Estraiamo i dati dell'hydro j
        hydro_index = hydro["index"]
        n_reservoirs = hydro["n_reservoirs"]

        values_per_hydro = n_reservoirs * time_horizon

        # Calcoliamo i quattro indici
        set_from_start = dss_start + j * values_per_hydro
        set_from_end = set_from_start + values_per_hydro
        set_to_start = 0
        set_to_end = values_per_hydro

        hydro_mappings.append({
            "target": "inflow",
            "function_name": "HydroUnitBlock::set_inflow",
            "caller": "B",
            "data_type": "D",
            "set_size": [0, 0],
            "set_elements": [
                set_from_start, set_from_end, # SetFrom: Range
                set_to_start, set_to_end, # SetTo: Range
            ],
            "abstract_paths": [
                {
                    "node_types": "B",
                    "group_indices": [str(hydro_index)],
                    "element_indices": [None],
                    "range_indices": [None],
                }
            ]
        })

    return hydro_mappings

def build_sddp_intermittent_mappings(
        *,
        intermittent_units: List[Tuple[int, str]],
        time_horizon: int,
        dss_start: int,
) -> List[Dict[str, Any]]:
    """
    Costruisce i data mapping per la potenza massima intermittente di uno
    stage SDDP.

    Restituisce una lista di len(intermittent_units) dizionari, uno per
    ogni IntermittentUnitBlock. Ogni dizionario mappa una fetta del vettore
    di scenario dello stage nel campo MaxPower del proprio
    IntermittentUnitBlock, secondo la struttura:

        mapping j (intermittente j):
            SetFrom = Range [dss_start + j*T, dss_start + (j+1)*T)
            SetTo   = Range [0, T)
        dove T = time_horizon.

    Parameters
    ----------
    intermittent_units : list of tuple
        Lista di coppie (unitblock_index, name) come fornite dal parser
        in unit_blocks_by_type["IntermittentUnitBlock"]. L'indice e'
        la posizione dello UnitBlock_i nel UCBlock (26, 27, ...).
    time_horizon : int
        Numero di snapshot per stage.
    dss_start : int
        Offset del parametro nel vettore di scenario dello stage.

    Returns
    -------
    list of dict
        Una lista di len(intermittent_units) data mapping.
    """
    # Verifichiamo i dati in input
    if time_horizon < 1:
        raise ValueError(
            f"{time_horizon} deve essere maggiore o uguale ad 1"
        )
    if dss_start < 0:
        raise ValueError(
            f"{dss_start} deve essere maggiore o uguale a 0"
        )

    if not intermittent_units:
        return []

    intermittent_mappings: List[Dict[str, Any]] = []

    for j, (unitblock_index, _name) in enumerate(intermittent_units):
        # Ogni intermittente ha un solo campo MaxPower di lunghezza TimeHorizon
        values_per_unit = time_horizon

        set_from_start = dss_start + j * values_per_unit
        set_from_end = set_from_start + values_per_unit
        set_to_start = 0
        set_to_end = values_per_unit

        intermittent_mappings.append({
            "target": "renewable_maxpower",
            "function_name": "IntermittentUnitBlock::set_maximum_power",
            "caller": "B",
            "data_type": "D",
            "set_size": [0, 0],
            "set_elements": [
                set_from_start, set_from_end,
                set_to_start, set_to_end,
            ],
            "abstract_paths": [
                {
                    "node_types": "B",
                    "group_indices": [str(unitblock_index)],
                    "element_indices": [None],
                    "range_indices": [None],
                }
            ]
        })

    return intermittent_mappings

def build_sddp_benders_mapping(
        *,
        dss_start: int,
        n_reservoirs: int,
        time_horizon: int,
) -> Dict[str, Any]:
    """
    Data mapping per la BendersBFunction di uno stage SDDP.

    Per N serbatoi, legge il primo valore (t=0) della serie di inflow di
    ciascun serbatoio dal vettore di scenario e lo scrive in b[0..N-1].

    Layout del vettore di scenario per il parametro hydro_inflow:
      [ r0_t0, r0_t1, ..., r0_t(T-1), r1_t0, r1_t1, ..., r1_t(T-1), ... ]
    quindi il primo valore del reservoir k è a dss_start + k*T (passo T,
    non 1).

    Parameters
    ----------
    dss_start : int
        Offset del primo valore di inflow nel vettore di scenario.
    n_reservoirs : int
        Numero di serbatoi da mappare (>= 1).
    time_horizon : int
        Numero di snapshot per stage (= lunghezza della serie di inflow
        di un singolo serbatoio).

    Returns
    -------
    dict
        Data mapping per BendersBFunction.
    """
    if dss_start < 0:
        raise ValueError(f"dss_start deve essere >= 0, ricevuto {dss_start}")
    if n_reservoirs < 1:
        raise ValueError(f"n_reservoirs deve essere >= 1, ricevuto {n_reservoirs}")
    if time_horizon < 1:
        raise ValueError(f"time_horizon deve essere >= 1, ricevuto {time_horizon}")

    # SetFrom: primo valore (t=0) della serie di inflow di ciascun serbatoio.
    # I serbatoi sono contigui nel DSS, ognuno con T valori → passo T.
    set_from_elements = [
        dss_start + k * time_horizon for k in range(n_reservoirs)
    ]
    # SetTo: b[0], b[1], ..., b[N-1]
    set_to_elements = list(range(n_reservoirs))

    return {
        "target": "benders",
        "function_name": "BendersBFunction::modify_constants",
        "caller": "F",
        "data_type": "D",
        "set_size": [n_reservoirs, n_reservoirs],
        "set_elements": set_from_elements + set_to_elements,
        "abstract_paths": [
            {
                "node_types": "O",
                "group_indices": [0xFFFFFFFF],
                "element_indices": [None],
                "range_indices": [None],
            }
        ],
    }

# -----------------------------------------------------------
# --------- Produttore payload dello StochasticBlock --------
# -----------------------------------------------------------

def build_sddp_stochastic_block_data(data_mappings):
    """
    Assembla una lsita di data mapping nel payload dello StochasticBlock
    di uno stage SDDP.
    """

    if not data_mappings:
        raise ValueError(
            f"{data_mappings} non può essere vuoto"
        )

    function_names = []
    callers = []
    data_types = []
    set_size = []
    set_elements = []

    for mapping in data_mappings:
        function_names.append(mapping["function_name"])
        callers.append(mapping["caller"])
        data_types.append(mapping["data_type"])
        set_size.extend(mapping["set_size"])
        set_elements.extend(mapping["set_elements"])

    abstract_path = build_sddp_stochastic_block_abstract_path(data_mappings)

    return {
        "NumberDataMappings": len(data_mappings),
        "FunctionName": np.array(function_names, dtype="object"),
        "Caller": np.array(callers, dtype="object"),
        "DataType": np.array(data_types, dtype="object"),
        "SetSize": np.array(set_size, dtype=np.uint32),
        "SetElements": np.array(set_elements, dtype=np.uint32),
        "AbstractPath": abstract_path,
    }

def build_sddp_stochastic_block(
        *,
        stochastic_parameters: Sequence[str],
        dss_offsets: Mapping[str, Mapping[str, int]],
        structure: Mapping[str, Any],
        time_horizon: int,
) -> Dict[str, Any]:
    """
    Costruisce il payload dello StochasticBlock per uno stage SDDP.

    Orchestra i quattro builder di mapping (demand, hydro, intermittent,
    benders) e produce un unico payload pronto per essere serializzato
    nel blocco StochasticBlock_i.

    Parameters
    ----------
    stochastic_parameters : Sequence[str]
        Lista dei parametri stocastici selezionati dall'utente (chiavi
        del registry STOCHASTIC_PARAMETER_REGISTRY).
    dss_offsets : Mapping[str, Mapping[str, int]]
        Offset di ciascun parametro nel vettore di scenario dello stage.
        Deve avere chiavi in {parametro: {"start": int, "end": int,
        "size": int}}. Corrisponde a self.sddp_dss_offsets[stage_name].
    structure : Mapping[str, Any]
        Struttura dello stage, dal parser. Deve contenere almeno
        "number_nodes", "hydro_unit_blocks", "unit_blocks_by_type".
    time_horizon : int
        Numero di snapshot per stage (TimeHorizon dello stage).

    Returns
    -------
    Dict[str, Any]
        Payload dello StochasticBlock.
    """
    # Verifichiamo la validità degli input
    if not stochastic_parameters:
        raise ValueError(
            f"{stochastic_parameters} non può essere vuoto"
        )
    if not dss_offsets:
        raise ValueError(
            f"{dss_offsets} non può essere vuoto"
        )
    for key in ("number_nodes", "hydro_unit_blocks", "unit_blocks_by_type"):
        if key not in structure:
            raise KeyError(
                f"structure non contiene la chiave obbligatoria {key!r}."
            )
    if time_horizon < 1:
        raise ValueError(
            f"{time_horizon} deve essere maggiore o uguale a 1"
        )

    # Inizializziamo la lista dei mapping
    data_mappings: List[Dict[str, Any]] = []

    for parameter in stochastic_parameters:
        # Recuperiamo lo spec dal registry
        spec = STOCHASTIC_PARAMETER_REGISTRY[parameter]
        mapping_kind = spec["mapping_kind"]

        # Recuperiamo l'offset DSS per questo parametro
        offset = dss_offsets.get(parameter)
        if offset is None:
            raise KeyError(
                f"Parametro {parameter!r} selezionato ma senza offset DSS "
                f"per questo stage. Offset disponibili: {sorted(dss_offsets)}."
            )

        # Dispatch in base a mapping_kind
        if mapping_kind == "ucblock_timeseries":
            data_mappings.extend(build_sddp_demand_mappings(
                number_nodes=structure["number_nodes"],
                time_horizon=time_horizon,
                dss_start=offset["start"],
            ))
            continue

        if mapping_kind == "unitblock_timeseries":
            unitblock_type = spec["unitblock_type"]

            if unitblock_type == "HydroUnitBlock":
                data_mappings.extend(build_sddp_hydro_mappings(
                    hydro_units=structure["hydro_unit_blocks"],
                    time_horizon=time_horizon,
                    dss_start=offset["start"],
                ))
                continue

            if unitblock_type == "IntermittentUnitBlock":
                intermittent_units = structure["unit_blocks_by_type"].get(
                    "IntermittentUnitBlock", []
                )
                data_mappings.extend(build_sddp_intermittent_mappings(
                    intermittent_units=list(intermittent_units),
                    time_horizon=time_horizon,
                    dss_start=offset["start"],
                ))
                continue

            raise NotImplementedError(
                f"unitblock_type {unitblock_type!r} non supportato dal "
                f"produttore SDDP per il parametro {parameter!r}."
            )

        raise ValueError(
            f"mapping_kind {mapping_kind!r} non supportato per il "
            f"parametro {parameter!r}."
        )

    # -----------------------------------------------------------------
    # Benders mapping: serve solo se c'è uno stato da modellare, cioè
    # se la struttura ha almeno un HydroUnitBlock.
    # -----------------------------------------------------------------
    hydro_units = structure.get("hydro_unit_blocks", [])
    n_reservoirs_total = sum(int(h["n_reservoirs"]) for h in hydro_units)

    if n_reservoirs_total > 0:
        if "hydro_inflow" not in dss_offsets:
            raise ValueError(
                f"La struttura dello stage ha {n_reservoirs_total} "
                "bacini (HydroUnitBlock nei Block_i.nc4), ma "
                "'hydro_inflow' non è tra i parametri stocastici "
                "selezionati in stochastic_parameters. Il benders "
                "mapping non può essere costruito: senza l'offset "
                "dell'inflow, il termine costante b della "
                "BendersBFunction non può essere popolato. Aggiungi "
                "'hydro_inflow' alla lista dei parametri, oppure "
                "verifica che la rete e i Block_i.nc4 siano coerenti."
            )
        data_mappings.append(
            build_sddp_benders_mapping(
                dss_start=int(dss_offsets["hydro_inflow"]["start"]),
                n_reservoirs=n_reservoirs_total,
                time_horizon=time_horizon,
            )
        )
    # Se n_reservoirs_total == 0, non c'è stato da propagare e il
    # benders mapping viene saltato.

    return build_sddp_stochastic_block_data(data_mappings)