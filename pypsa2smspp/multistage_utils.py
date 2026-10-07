# -*- coding: utf-8 -*-
"""
Utilities for multistage stochastic PyPSA networks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Mapping, Optional, Sequence

from openpyxl.styles.builtins import total

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
            "snapshots_per_stage": 31
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

    return {
        "stochastic_type": "sddp",
        "parameters": parameters,
        "snapshots_per_stage": snapshots_per_stage,
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
    scenario_names = su.get_scenario_names(n);

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

# -------------------------------------------------------
# ------------------ ABSTRACT PATHS ---------------------
# -------------------------------------------------------

def build_sddp_top_abstract_path(
    n_stages: int,
    n_polyhedral_per_sub_block: int = 1,
) -> Dict[str, Any]:
    """
    AbstractPath top-level dell'SDDPBlock.

    Secondo SDDPBlock::serialize():
        PathDim = TimeHorizon × NumPolyhedralFunctionsPerSubBlock
    Ogni path va da un BendersBlock (reference) a una PolyhedralFunction
    (target). La struttura di ogni path è "OBBB" con
    PathGroupIndices = [MASK, 0, 0, 3], derivata dal notebook di riferimento.

    Parameters
    ----------
    n_stages : int
        Numero di stadi (TimeHorizon dell'SDDPBlock).
    n_polyhedral_per_sub_block : int, default 1
        Numero di PolyhedralFunction per sotto-blocco.

    Returns
    -------
    dict
        PathDim, TotalLength, PathStart, PathNodeTypes, PathGroupIndices.

    TODO
    ----
    Il numero di path (n_stages * n_polyhedral_per_sub_block) è dedotto dal
    codice C++ SDDPBlock::serialize. La struttura del singolo path ("OBBB"
    con [MASK, 0, 0, 3]) è dedotta dal notebook scritto a mano: va
    verificata con un .nc4 prodotto da serialize() o con il professore.
    """
    _UINT32_MASK = np.uint32(4294967295)

    if n_stages < 0 or n_polyhedral_per_sub_block < 0:
        raise ValueError("n_stages e n_polyhedral_per_sub_block devono essere >= 0.")

    num_paths = n_stages * n_polyhedral_per_sub_block
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
        np.array([_UINT32_MASK, 0, 0, 3], dtype=np.uint32),
        num_paths,
    )

    return {
        "PathDim": int(num_paths),
        "TotalLength": int(nodes_per_path * num_paths),
        "PathStart": path_start,
        "PathNodeTypes": path_node_types,
        "PathGroupIndices": path_group_indices,
    }

def build_sddp_benders_abstract_path(n_bacini: int) -> Dict[str, Any]:
    """
    AbstractPath della BendersBFunction: un percorso per ogni bacino.

    Ogni percorso ha 3 nodi 'B', 'B', 'C':
    - Primo 'B': scende nel BendersBlock (group=0)
    - Secondo 'B': scende nel k-esimo blocco figlio (group=k)
    - 'C': punta al vincolo 0 di quel blocco (element=0)

    Parameters
    ----------
    n_bacini : int
        Numero di bacini. Se 0, restituisce un AbstractPath vuoto.

    Returns
    -------
    dict
        PathDim, TotalLength, PathStart, PathNodeTypes, PathGroupIndices,
        PathElementIndices.
    """
    _UINT32_MASK = np.uint32(4294967295)

    if n_bacini < 0:
        raise ValueError(f"n_bacini deve essere >= 0, ricevuto {n_bacini}.")

    # Caso degenere: nessun bacino -> nessun percorso.
    if n_bacini == 0:
        return {
            "PathDim": 0,
            "TotalLength": 0,
            "PathStart": np.array([], dtype=np.uint32),
            "PathNodeTypes": np.array([], dtype="S1"),
            "PathGroupIndices": np.array([], dtype=np.uint32),
            "PathElementIndices": np.array([], dtype=np.uint32),
        }

    # PathStart: [0, 3, 6, 9, ...]
    path_start = np.arange(0, 3 * n_bacini, 3, dtype=np.uint32)

    # PathNodeTypes: "BBC" ripetuto n_bacini volte
    path_node_types = np.tile(
        np.array(["B", "B", "C"], dtype="S1"),
        n_bacini,
    )

    # PathGroupIndices: [0, k, 0] per ogni k = 0..n-1
    groups = []
    for k in range(n_bacini):
        groups.extend([0, k, 0])
    path_group_indices = np.array(groups, dtype=np.uint32)

    # PathElementIndices: [MASK, MASK, 0] ripetuto n volte
    path_element_indices = np.tile(
        np.array([_UINT32_MASK, _UINT32_MASK, 0], dtype=np.uint32),
        n_bacini,
    )

    return {
        "PathDim": int(n_bacini),
        "TotalLength": int(3 * n_bacini),
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
    hydro_counter = 0   # contatore progressivo per i bacini

    for mapping in data_mappings:
        fn = mapping["function_name"]

        if fn == "UCBlock::set_active_power_demand":
            nodes = ["O", "B"]
            groups = [_UINT32_MASK, 0]

        elif fn == "HydroUnitBlock::set_inflow":
            nodes = ["O", "B", "B", "B"]
            groups = [_UINT32_MASK, 0, 0, hydro_counter]
            hydro_counter += 1

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