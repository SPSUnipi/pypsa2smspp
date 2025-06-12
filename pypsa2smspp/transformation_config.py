import numpy as np


class TransformationConfig:
    """
    Class for defining the configuration parameter of the PyPSA2SMSpp transformation.
    This class is used to set up the parameters for different types of units in the network.
    Attributes:
        IntermittentUnitBlock_parameters (dict): Parameters for intermittent units.
        ThermalUnitBlock_parameters (dict): Parameters for thermal units.
        BatteryUnitBlock_parameters (dict): Parameters for battery units.
        BatteryUnitBlock_store_parameters (dict): Parameters for battery storage units.
        Lines_parameters (dict): Parameters for lines in the network.
        Links_parameters (dict): Parameters for links in the network.
        HydroUnitBlock_parameters (dict): Parameters for hydro units.
        max_hours_stores_parameters (double): Maximum hours of storage capacity.
    """
    def __init__(self, *args, **kwargs):
        self.reset()

        if len(args) > 0:
            raise ValueError("No positional arguments are allowed. Use keyword arguments instead.")

        for key, value in kwargs.items():
            setattr(self, key, value)

    def reset(self):
        # Parameters for intermittent units
        self.IntermittentUnitBlock_parameters = {
            "Gamma": 0.0,
            "Kappa": 1.0,
            "MaxPower": lambda p_nom_opt, p_max_pu: p_nom_opt * p_max_pu,
            "MinPower": lambda p_nom_opt, p_min_pu: p_nom_opt * p_min_pu,
            "InertiaPower": 1.0,
            "ActivePowerCost": lambda marginal_cost: marginal_cost,
        }

        # Parameters for thermal units
        self.ThermalUnitBlock_parameters = {
            "InitUpDownTime": lambda up_time_before: up_time_before,
            "MinUpTime": lambda min_up_time: min_up_time,
            "MinDownTime": lambda min_down_time: min_down_time, 
            #"DeltaRampUp": lambda ramp_limit_up: ramp_limit_up if not np.isnan(ramp_limit_up) else 0,
            #"DeltaRampDown": lambda ramp_limit_down: ramp_limit_down if not np.isnan(ramp_limit_down) else 0,
            "MinPower": lambda p_nom_opt, p_min_pu: p_nom_opt * p_min_pu,
            "MaxPower": lambda p_nom_opt, p_max_pu: p_nom_opt * p_max_pu,
            "PrimaryRho": 0.0,
            "SecondaryRho": 0.0,
            "Availability": 1,
            "QuadTerm": lambda marginal_cost_quadratic: marginal_cost_quadratic,
            "LinearTerm": lambda marginal_cost: marginal_cost,
            "ConstantTerm": 0.0,
            "StartUpCost": lambda start_up_cost: start_up_cost,
            "InitialPower": lambda p: p[0][0],
            "FixedConsumption": 0.0,
            "InertiaCommitment": 1.0
        }

        self.BatteryUnitBlock_parameters = {
            "Kappa": 1.0,
            "MaxPower": lambda p_nom_opt, p_max_pu: p_nom_opt * p_max_pu,
            "MinPower": lambda p_nom_opt, p_min_pu: p_nom_opt * p_min_pu,
            # "DeltaRampUp": np.nan,
            # "DeltaRampDown": np.nan,
            "ExtractingBatteryRho": lambda efficiency_dispatch: 1 / efficiency_dispatch,
            "StoringBatteryRho": lambda efficiency_store: efficiency_store,
            "Demand": 0.0,
            "MinStorage": 0.0,
            "MaxStorage": lambda p_nom_opt, p_max_pu, max_hours: p_nom_opt * p_max_pu * max_hours,
            "MaxPrimaryPower": 0.0,
            "MaxSecondaryPower": 0.0,
            "InitialPower": lambda p: p[0][0],
            "InitialStorage": lambda state_of_charge, cyclic_state_of_charge: -1 if cyclic_state_of_charge.values else state_of_charge[0][0],
            "Cost": lambda marginal_cost: marginal_cost
            }

        self.BatteryUnitBlock_store_parameters = {
            "Kappa": 1.0,
            "MaxPower": lambda e_nom_opt, e_max_pu, max_hours: e_nom_opt * e_max_pu / max_hours,
            "MinPower": lambda e_nom_opt, e_max_pu, max_hours: - e_nom_opt * e_max_pu / max_hours,
            # "DeltaRampUp": np.nan,
            # "DeltaRampDown": np.nan,
            "ExtractingBatteryRho": lambda e_max_pu: np.ones_like(e_max_pu),
            "StoringBatteryRho": lambda e_max_pu: np.ones_like(e_max_pu),
            "Demand": 0.0,
            "MinStorage": 0.0,
            "MaxStorage": lambda e_nom_opt: e_nom_opt,
            "MaxPrimaryPower": 0.0,
            "MaxSecondaryPower": 0.0,
            "InitialPower": lambda e_initial, max_hours: e_initial / max_hours,
            "InitialStorage": lambda e_initial, e_cyclic: -1 if e_cyclic.values else e_initial,
            "Cost": lambda marginal_cost: marginal_cost
            }

        self.Lines_parameters = {
            "StartLine": lambda start_line_idx: start_line_idx.values,
            "EndLine": lambda end_line_idx: end_line_idx.values,
            "MinPowerFlow": lambda s_nom_opt: -s_nom_opt.values,
            "MaxPowerFlow": lambda s_nom_opt: s_nom_opt.values,
            "LineSusceptance": lambda s_nom_opt: np.zeros_like(s_nom_opt)
            }

        self.Links_parameters = {
            "StartLine": lambda start_line_idx: start_line_idx.values,
            "EndLine": lambda end_line_idx: end_line_idx.values,
            "MinPowerFlow": lambda p_nom_opt, p_min_pu: p_nom_opt.values * p_min_pu.values,
            "MaxPowerFlow": lambda p_nom_opt, p_max_pu: p_nom_opt.values * p_max_pu.values,
            "LineSusceptance": lambda s_nom_opt: np.zeros_like(s_nom_opt)
            }

        self.HydroUnitBlock_parameters = {
            # "StartArc": lambda p_nom: np.full(len(p_nom)*2, 0),
            # "EndArc": lambda p_nom: np.full(len(p_nom)*2, 1),
            "StartArc": lambda p_nom: np.array([0, 1]),
            "EndArc": lambda p_nom: np.array([1, 0]),
            "MaxVolumetric": lambda p_nom_opt, max_hours: (p_nom_opt * max_hours),
            "MinVolumetric": 0.0,
            "Inflows": lambda inflow: inflow.values.transpose(),
            "MaxFlow": lambda inflow, p_nom, efficiency_dispatch: (np.array([max(100 * inflow.values.max(), (p_nom / efficiency_dispatch).values.max()), 0.])).squeeze().transpose(),
            "MinFlow": lambda inflow, p_nom, efficiency_dispatch: (np.array([0., min(-100 * inflow.values.max(), -(p_nom / efficiency_dispatch).values.max())])).squeeze().transpose(),
            "MaxPower": lambda p_nom_opt, p_max_pu: (np.array([(p_nom_opt*p_max_pu), (0.*p_max_pu)])).squeeze().transpose(),
            "MinPower": lambda p_nom_opt, p_min_pu: (np.array([(0.*p_min_pu), (p_nom_opt*p_min_pu)])).squeeze().transpose(),
            # "PrimaryRho": lambda p_nom: np.full(len(p_nom)*3, 0.),
            # "SecondaryRho": lambda p_nom: np.full(len(p_nom)*3, 0.),
            "NumberPieces": lambda p_nom: np.full(len(p_nom)*2, 1),
            "ConstantTerm": lambda p_nom: np.full(len(p_nom)*2, 0),
            "LinearTerm": lambda efficiency_dispatch, efficiency_store: np.array([efficiency_dispatch.values.max(), 1 / efficiency_store.values.max() if efficiency_store.values.max() != 0 else 0]),
            # "DeltaRampUp": np.nan,
            # "DeltaRampDown": np.nan,
            "DownhillFlow": lambda p_nom: np.full(len(p_nom)*2, 0.),
            "UphillFlow": lambda p_nom: np.full(len(p_nom)*2, 0.),
            #"InertiaPower": 1.0,
            # "InitialFlowRate": lambda inflow: inflow.values[0],
            "InitialVolumetric": lambda state_of_charge_initial, cyclic_state_of_charge: -1 if cyclic_state_of_charge.values else state_of_charge_initial.values
        }

        self.IntermittentUnitBlock_inverse = {
            "p_nom": lambda p_nom: p_nom,
            "p": lambda activepower: activepower
            }
        
        self.ThermalUnitBlock_inverse = {
            "p_nom": lambda p_nom: p_nom,
            "p": lambda activepower: activepower
            }
        
        self.HydroUnitBlock_inverse = {
            "p_nom": lambda p_nom: p_nom,
            "p_dispatch": lambda activepower: activepower[0],
            "p_store": lambda activepower: -activepower[1],
            # "p": lambda activepower: activepower
            }
        
        self.BatteryUnitBlock_inverse = {
            "p_nom": lambda p_nom: p_nom,
            "p_dispatch": lambda activepower: np.maximum(activepower, 0),
            "p_store": lambda activepower: np.maximum(-activepower, 0),
            # "state_of_charge": lambda storage_level: storage_level
            }
        
        self.component_mapping = {
            "Generator": "generators",
            "StorageUnit": "storage_units",
            "Store": "stores",
            "Load": "loads",
            "Link": "links",
            "Bus": "buses"
        }

        self.max_hours_stores = 10
