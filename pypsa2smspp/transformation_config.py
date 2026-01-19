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
            # "Gamma": 0.0,
            # "Kappa": 1.0,
            "MaxPower": lambda p_nom, p_max_pu, p_nom_extendable: (p_nom * p_max_pu).where(~p_nom_extendable, p_max_pu),
            "MinPower": lambda p_nom, p_min_pu, p_nom_extendable: (p_nom * p_min_pu).where(~p_nom_extendable, p_min_pu),
            # "InertiaPower": 1.0,
            "ActivePowerCost": lambda marginal_cost: marginal_cost,
        }

        # Parameters for thermal units
        self.ThermalUnitBlock_parameters = {
            "InitUpDownTime": lambda up_time_before: up_time_before,
            "MinUpTime": lambda min_up_time: min_up_time,
            "MinDownTime": lambda min_down_time: min_down_time, 
            #"DeltaRampUp": lambda ramp_limit_up: ramp_limit_up if not np.isnan(ramp_limit_up) else 0,
            #"DeltaRampDown": lambda ramp_limit_down: ramp_limit_down if not np.isnan(ramp_limit_down) else 0,
            "MaxPower": lambda p_nom, p_max_pu, p_nom_extendable: (p_nom * p_max_pu).where(~p_nom_extendable, p_max_pu),
            "MinPower": lambda p_nom, p_min_pu, p_nom_extendable: (p_nom * p_min_pu).where(~p_nom_extendable, p_min_pu),
            "PrimaryRho": 0.0,
            "SecondaryRho": 0.0,
            "Availability": 1,
            "QuadTerm": lambda marginal_cost_quadratic: marginal_cost_quadratic,
            "LinearTerm": lambda marginal_cost: marginal_cost,
            "ConstTerm": 0.0,
            "StartUpCost": lambda start_up_cost: start_up_cost,
            "InitialPower": lambda p: p[0][0],
            "FixedConsumption": 0.0,
            "InertiaCommitment": 1.0
        }

        self.BatteryUnitBlock_parameters = {
            # "Kappa": 1.0,
            "MaxPower": lambda p_nom, p_max_pu, p_nom_extendable: (p_nom * p_max_pu).where(~p_nom_extendable, p_max_pu),
            "MinPower": lambda p_nom, p_min_pu, p_nom_extendable: (p_nom * p_min_pu).where(~p_nom_extendable, p_min_pu),
            # "DeltaRampUp": np.nan,
            # "DeltaRampDown": np.nan,
            "ExtractingBatteryRho": lambda efficiency_dispatch: 1 / efficiency_dispatch,
            "StoringBatteryRho": lambda efficiency_store: efficiency_store,
            "Demand": 0.0,
            "MinStorage": 0.0,
            "MaxStorage": lambda p_nom, p_max_pu, max_hours: p_nom * p_max_pu * max_hours,
            "MaxPrimaryPower": 0.0,
            "MaxSecondaryPower": 0.0,
            "InitialPower": lambda p: p[0][0],
            "InitialStorage": lambda state_of_charge, cyclic_state_of_charge: -1 if cyclic_state_of_charge.values else state_of_charge[0][0],
            "Cost": lambda marginal_cost: marginal_cost,
            # "BatteryInvestmentCost": lambda capital_cost: capital_cost,
            # "ConverterInvestmentCost": 0.0,
            # "BatteryMaxCapacityDesign": lambda p_nom, p_nom_extendable, p_nom_max: p_nom_max.replace(np.inf, 1e7).item() if p_nom_extendable.item() else p_nom.item(),
            # "ConverterMaxCapacityDesign": lambda p_nom, p_nom_extendable, p_nom_max: 10*p_nom_max.replace(np.inf, 1e7).item() if p_nom_extendable.item() else p_nom.item()
            }

        self.BatteryUnitBlock_store_parameters = {
            # "Kappa": 1.0,
            "MaxPower": lambda e_nom, e_max_pu, max_hours, e_nom_extendable: (e_nom * e_max_pu / max_hours).where(~e_nom_extendable, e_max_pu),
            "MinPower": lambda e_nom, e_max_pu, max_hours, e_nom_extendable: - (e_nom * e_max_pu / max_hours).where(~e_nom_extendable, e_max_pu),
            # "ConverterMaxPower": lambda e_nom, e_max_pu, max_hours, e_nom_extendable: (e_nom * e_max_pu / max_hours).where(~e_nom_extendable, e_max_pu),
            # "DeltaRampUp": np.nan,
            # "DeltaRampDown": np.nan,
            "ExtractingBatteryRho": lambda efficiency_dispatch: 1 / efficiency_dispatch.iloc[0],
            "StoringBatteryRho": lambda efficiency_store: efficiency_store.iloc[0],
            "Demand": 0.0,
            "MinStorage": 0.0,
            "MaxStorage": lambda e_nom, e_max_pu, e_nom_extendable: (e_nom * e_max_pu).where(~e_nom_extendable, e_max_pu),
            "MaxPrimaryPower": 0.0,
            "MaxSecondaryPower": 0.0,
            "InitialPower": lambda e_initial, max_hours: (e_initial / max_hours).iloc[0],
            "InitialStorage": lambda e_initial, e_cyclic: -1 if e_cyclic.values else e_initial,
            "Cost": lambda marginal_cost: marginal_cost,
            }

        self.Lines_parameters = {
            "StartLine": lambda start_line_idx: start_line_idx.values,
            "EndLine": lambda end_line_idx: end_line_idx.values,
            "MinPowerFlow": lambda s_nom, s_max_pu, s_nom_extendable: - (s_nom * s_max_pu).where(~s_nom_extendable, s_max_pu),
            "MaxPowerFlow": lambda s_nom, s_max_pu, s_nom_extendable: (s_nom * s_max_pu).where(~s_nom_extendable, s_max_pu),
            "LineSusceptance": lambda s_nom: np.zeros_like(s_nom),
            "Efficiency": lambda s_nom: np.ones_like(s_nom),
            "NetworkCost": lambda s_nom: np.zeros_like(s_nom),
            }

        self.Links_parameters = {
            "StartLine": lambda start_line_idx: start_line_idx.values,
            "EndLine": lambda end_line_idx: end_line_idx.values,
            "MaxPowerFlow": lambda p_nom, p_max_pu, p_nom_extendable: (p_nom * p_max_pu).where(~p_nom_extendable, p_max_pu),
            "MinPowerFlow": lambda p_nom, p_min_pu, p_nom_extendable: (p_nom * p_min_pu).where(~p_nom_extendable, p_min_pu),
            "LineSusceptance": lambda p_nom: np.zeros_like(p_nom),
            "Efficiency": lambda efficiency: efficiency,
            "NetworkCost": lambda marginal_cost: marginal_cost.values
            }

        self.HydroUnitBlock_parameters = {
            # "StartArc": lambda p_nom: np.full(len(p_nom)*2, 0),
            # "EndArc": lambda p_nom: np.full(len(p_nom)*2, 1),
            "StartArc": lambda p_nom: np.array([0, 0]),
            "EndArc": lambda p_nom: np.array([1, 1]),
            "MaxVolumetric": lambda p_nom, max_hours: (p_nom * max_hours),
            "MinVolumetric": 0.0,
            "Inflows": lambda inflow: inflow.values.transpose(),
            # "MaxFlow": lambda inflow, p_nom, efficiency_dispatch: (np.array([max(100 * inflow.values.max(), (p_nom / efficiency_dispatch).values.max()), 0.])).squeeze().transpose(),
            # "MinFlow": lambda inflow, p_nom, efficiency_dispatch: (np.array([0., min(-100 * inflow.values.max(), -(p_nom / efficiency_dispatch).values.max())])).squeeze().transpose(),
            "MaxFlow": lambda p_nom, p_max_pu, max_hours: (np.array([(p_nom*p_max_pu*max_hours), (0.*p_max_pu)])).squeeze().transpose(),
            "MinFlow": lambda p_nom, p_min_pu, max_hours: (np.array([(0.*p_min_pu), (p_nom*p_min_pu*max_hours)])).squeeze().transpose(),
            "MaxPower": lambda p_nom, p_max_pu: (np.array([(p_nom*p_max_pu), (0.*p_max_pu)])).squeeze().transpose(),
            "MinPower": lambda p_nom, p_min_pu: (np.array([(0.*p_min_pu), (p_nom*p_min_pu)])).squeeze().transpose(),
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
        
        self.InvestmentBlock_parameters = {
            "Cost": lambda capital_cost: capital_cost.values,
            "LowerBound": lambda p_nom_min: p_nom_min.replace(0, 1e-6).values,
            "UpperBound": lambda p_nom_max: p_nom_max.replace(np.inf, 1e7).values,
            # "InstalledQuantity": lambda p_nom: p_nom.replace(0, 1e-6).values,
            "InstalledQuantity": lambda p_nom: np.zeros_like(p_nom), # This is used now that we want to add objective constant as separated
            }
        
        self.SlackUnitBlock_parameters = {
            "ActivePowerCost": lambda marginal_cost: marginal_cost,
            "MaxPower": lambda p_nom: p_nom
            }

        self.IntermittentUnitBlock_inverse = {
            "p_nom": lambda designvariable: designvariable,
            "p": lambda activepower: activepower,
            }
        
        self.ThermalUnitBlock_inverse = {
            "p_nom": lambda designvariable: designvariable,
            "p": lambda activepower, designvariable, extendable: activepower * designvariable if extendable else activepower,
            }
        
        self.HydroUnitBlock_inverse = {
            "p_nom": lambda designvariable: designvariable,
            "p_dispatch": lambda activepower: activepower[0],
            "p_store": lambda activepower: -activepower[1],
            "state_of_charge": lambda volumetriclevel: volumetriclevel,
            }
        
        # TODO manage them as stores (or distinguish, but probably storage units wil always be treated as hydrounitblocks)
        self.BatteryUnitBlock_inverse = {
            "e_nom": lambda designvariable: designvariable,
            # "p_dispatch": lambda activepower: np.maximum(activepower, 0),
            # "p_store": lambda activepower: np.maximum(-activepower, 0),
            "p": lambda activepower: activepower,
            "e": lambda storagelevel: storagelevel,
            }
        
        self.DCNetworkBlock_lines_inverse = {
            "p0": lambda flowvalue: flowvalue,
            "p1": lambda flowvalue: -flowvalue,
            # "mu_lower": lambda dualcost: dualcost,
            # "mu_upper": lambda dualcost: dualcost,
            "s_nom": lambda designvariable: designvariable,
            }
        
        self.DCNetworkBlock_links_inverse = {
            "p0": lambda flowvalue: flowvalue,
            "p1": lambda flowvalue, efficiency: -flowvalue * efficiency,
            # "mu_lower": lambda dualcost: dualcost,
            # "mu_upper": lambda dualcost: dualcost,
            "p_nom": lambda designvariable: designvariable,
            }
        
        self.SlackUnitBlock_inverse = {
            "p": lambda activepower: activepower,
            "p_nom": lambda designvariable: designvariable
            }
        
        self.component_mapping = {
            "Generator": "generators",
            "StorageUnit": "storage_units",
            "Store": "stores",
            "Load": "loads",
            "Link": "links",
            "Line": "lines",
            "Bus": "buses"
        }

