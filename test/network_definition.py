# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:07:31 2024

@author: aless
"""

import pandas as pd
import numpy as np
import logging
import pypsa

LOG_FORMAT = (
    '%(levelname) -10s %(asctime)s %(name) -10s %(funcName) '
    '-10s %(lineno) -5d: %(message)s'
)
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logging.getLogger("Article").setLevel(logging.WARNING)


class NetworkDefinition:
    """
    NetworkDefinition class for building a PyPSA network from input Excel files.

    This class reads data from Excel files and constructs a PyPSA network.
    It supports:
    - static component sheets (Bus, Generator, Link, ...)
    - generic PyPSA-style time-series override sheets with naming:
      '<component>_t_<attribute>'
      examples:
          - loads_t_p_set
          - generators_t_p_max_pu
          - storage_units_t_inflow
          - links_t_efficiency

    Fallback logic is kept only for:
    - loads_t_p_set -> add_demand()
    - generators_t_p_max_pu -> add_renewables()
    - storage_units_t_inflow -> add_hydro_inflow()

    All other time-series sheets are optional:
    if present, they are loaded; if absent, nothing is done.
    """

    # Fallback-supported special sheets
    FALLBACK_TS_SHEETS = {
        "loads_t_p_set",
        "generators_t_p_max_pu",
        "storage_units_t_inflow",
    }

    def __init__(self, parser):
        """
        Initialize the NetworkDefinition class.

        Parameters
        ----------
        parser : object
            Parser containing paths to input data files.
        """
        self.parser = parser
        self.init()

    def init(self):
        """
        Define the workflow for building the network.
        """
        self.n = pypsa.Network()
        
        all_sheets = self.read_excel_components()

        self.define_snapshots(all_sheets)

        static_sheets, ts_sheets = self.split_static_and_timeseries_sheets(all_sheets)

        self.add_all_components(static_sheets)

        if self.parser.add_costs_components:
            self.add_costs_components()

        self.apply_timeseries_sheets(ts_sheets)

        self.apply_timeseries_fallbacks(ts_sheets)

    def define_snapshots(self, all_sheets=None):
        """
        Define network snapshots.
    
        Priority:
        1. If an Excel sheet named 'snapshots' exists, use it.
        2. Otherwise, fall back to parser-based snapshots.
    
        Supported Excel formats
        -----------------------
        Sheet name: 'snapshots'
    
        Accepted columns:
        - 'snapshot'                -> snapshot labels
        - 'objective' (optional)   -> snapshot objective weighting
        - 'generators' (optional)  -> snapshot generator weighting
        - 'stores' (optional)      -> snapshot store weighting
    
        If the sheet has only one column, it is interpreted as the snapshot labels.
        Missing weighting columns fall back to parser.weight.
        """
        if all_sheets is not None and "snapshots" in all_sheets:
            df = all_sheets["snapshots"].copy()
    
            # Drop fully empty rows/cols from Excel junk
            df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    
            if df.empty:
                raise ValueError("The Excel sheet 'snapshots' is empty after cleaning.")
    
            # Case 1: explicit 'snapshot' column
            if "snapshot" in df.columns:
                snapshots = df["snapshot"].tolist()
            # Case 2: single unnamed/other column -> use first column
            elif df.shape[1] == 1:
                snapshots = df.iloc[:, 0].tolist()
            else:
                raise ValueError(
                    "Sheet 'snapshots' must either contain a 'snapshot' column "
                    "or a single column with snapshot labels."
                )
    
            self.n.set_snapshots(pd.Index(snapshots))
    
            # Weightings
            default_weight = self.parser.weight
    
            if "objective" in df.columns:
                self.n.snapshot_weightings.loc[self.n.snapshots, "objective"] = df["objective"].to_numpy()
            else:
                self.n.snapshot_weightings.loc[self.n.snapshots, "objective"] = default_weight
    
            if "generators" in df.columns:
                self.n.snapshot_weightings.loc[self.n.snapshots, "generators"] = df["generators"].to_numpy()
            else:
                self.n.snapshot_weightings.loc[self.n.snapshots, "generators"] = default_weight
    
            if "stores" in df.columns:
                self.n.snapshot_weightings.loc[self.n.snapshots, "stores"] = df["stores"].to_numpy()
    
            LOGGER.info(
                "Defined %d snapshots from Excel sheet 'snapshots'.",
                len(self.n.snapshots),
            )
    
        else:
            self.n.snapshots = range(0, self.parser.n_snapshots)
            self.n.snapshot_weightings.objective = self.parser.weight
            self.n.snapshot_weightings.generators = self.parser.weight
            # self.n.snapshot_weightings.stores = self.parser.weight
    
            LOGGER.info(
                "Defined %d snapshots from parser fallback.",
                len(self.n.snapshots),
            )

    def read_excel_components(self):
        """
        Read all sheets from the Excel file containing the network definition.

        Returns
        -------
        dict
            Dictionary {sheet_name: DataFrame}.
        """
        file_path = f"{self.parser.input_data_path}/{self.parser.input_name_components}"
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        return all_sheets

    def split_static_and_timeseries_sheets(self, all_sheets):
        """
        Split Excel sheets into static component sheets and time-series sheets.

        A time-series sheet must follow the naming convention:
            '<component>_t_<attribute>'

        Parameters
        ----------
        all_sheets : dict
            Dictionary of all Excel sheets.

        Returns
        -------
        tuple[dict, dict]
            (static_sheets, ts_sheets)
        """
        static_sheets = {}
        ts_sheets = {}
    
        for sheet_name, df in all_sheets.items():
            if sheet_name == "snapshots":
                continue
    
            parsed = self.parse_timeseries_sheet_name(sheet_name)
            if parsed is None:
                static_sheets[sheet_name] = df
            else:
                ts_sheets[sheet_name] = df
    
        return static_sheets, ts_sheets

    def parse_timeseries_sheet_name(self, sheet_name):
        """
        Parse a time-series sheet name of the form '<component>_t_<attribute>'.

        Examples
        --------
        loads_t_p_set -> ('loads', 'p_set')
        generators_t_p_max_pu -> ('generators', 'p_max_pu')
        storage_units_t_inflow -> ('storage_units', 'inflow')

        Parameters
        ----------
        sheet_name : str

        Returns
        -------
        tuple[str, str] | None
            (component_name, attribute_name) if valid, otherwise None.
        """
        marker = "_t_"
        if marker not in sheet_name:
            return None

        component_name, attribute_name = sheet_name.split(marker, 1)

        if not component_name or not attribute_name:
            return None

        if not hasattr(self.n, component_name):
            return None

        ts_container_name = f"{component_name}_t"
        if not hasattr(self.n, ts_container_name):
            return None

        return component_name, attribute_name

    def add_all_components(self, static_sheets):
        """
        Add all static component sheets to the network.

        Parameters
        ----------
        static_sheets : dict
            Dictionary containing only static component sheets.
        """
        for sheet_name, data in static_sheets.items():
            self.add_component(self.n, sheet_name, data)

    def add_component(self, network, component_type, data):
        """
        Add a single static component sheet to the network.

        Parameters
        ----------
        network : pypsa.Network
            The PyPSA network.
        component_type : str
            Component type (e.g. Bus, Generator, Link, Store, ...).
        data : pandas.DataFrame
            Static component table.
        """
        for _, row in data.iterrows():
            name = row[data.columns[0]]
            params = {
                col: row[col]
                for col in data.columns
                if col != "name" and pd.notna(row[col])
            }
            network.add(component_type, name, **params)

    def add_costs_components(self):
        """
        Add cost data to network components based on a costs Excel file.
        """
        file_path = f"{self.parser.input_data_path}/{self.parser.input_name_costs}"
        costs = pd.read_excel(file_path, index_col=0)

        for components in self.n.components[["Generator", "StorageUnit", "Link", "Store"]]:
            if components.empty:
                continue
            components_df = components.static
            for component in components_df.index:
                components_df.loc[component, "capital_cost"] = costs.at[
                    component.split(" ")[0], "Capital cost [€/MW]"
                ]
                components_df.loc[component, "marginal_cost"] = costs.at[
                    component.split(" ")[0], "Marginal cost [€/MWh]"
                ]

    def _prepare_timeseries_sheet(self, df, expected_columns, object_name):
        """
        Prepare and validate a time-series override sheet read from Excel.

        Expected format:
        - rows = snapshots in the same order as self.n.snapshots
        - columns = PyPSA component names
        - no explicit snapshot column is required

        Parameters
        ----------
        df : pandas.DataFrame
            Raw dataframe from Excel.
        expected_columns : iterable
            Expected PyPSA object names.
        object_name : str
            Human-readable object name for logging/errors.

        Returns
        -------
        pandas.DataFrame
            Clean dataframe aligned to snapshots and filtered to valid columns.
        """
        df_clean = df.copy()

        # Drop fully empty rows/columns often introduced by Excel exports
        df_clean = df_clean.dropna(axis=0, how="all").dropna(axis=1, how="all")

        if df_clean.empty:
            raise ValueError(f"The Excel sheet for {object_name} is empty after cleaning.")

        # Normalize column names to strings
        df_clean.columns = df_clean.columns.map(str)
        expected_columns = pd.Index(expected_columns).map(str)

        n_rows = len(df_clean)
        n_snapshots = len(self.n.snapshots)
        if n_rows != n_snapshots:
            raise ValueError(
                f"Invalid number of rows in Excel sheet for {object_name}: "
                f"found {n_rows}, expected {n_snapshots} (one per snapshot)."
            )

        matching_columns = [col for col in df_clean.columns if col in expected_columns]
        missing_in_network = [col for col in df_clean.columns if col not in expected_columns]
        missing_in_sheet = [col for col in expected_columns if col not in df_clean.columns]

        if missing_in_network:
            LOGGER.warning(
                "The following columns in the Excel sheet for %s do not match any existing PyPSA object "
                "and will be ignored: %s",
                object_name,
                missing_in_network,
            )

        if missing_in_sheet:
            LOGGER.warning(
                "The following PyPSA %s are not present in the Excel sheet and will keep their default values: %s",
                object_name,
                missing_in_sheet,
            )

        if not matching_columns:
            raise ValueError(
                f"No valid columns found in Excel sheet for {object_name}. "
                f"Expected names matching the network {object_name}."
            )

        df_clean = df_clean[matching_columns].copy()
        df_clean.index = self.n.snapshots

        return df_clean

    def set_timeseries_from_excel(self, component_name, attribute_name, df):
        """
        Assign a generic PyPSA time-series attribute from an Excel sheet.

        Parameters
        ----------
        component_name : str
            PyPSA component table name in plural form, e.g. 'loads', 'generators',
            'storage_units', 'links', 'stores'.
        attribute_name : str
            Time-series attribute name, e.g. 'p_set', 'p_max_pu', 'inflow'.
        df : pandas.DataFrame
            Raw Excel dataframe.
        """
        static_df = getattr(self.n, component_name)
        ts_container_name = f"{component_name}_t"
        ts_container = getattr(self.n, ts_container_name)

        if static_df.empty:
            LOGGER.warning(
                "Skipping sheet '%s_t_%s': network has no static '%s' entries.",
                component_name,
                attribute_name,
                component_name,
            )
            return

        df_clean = self._prepare_timeseries_sheet(
            df=df,
            expected_columns=static_df.index,
            object_name=component_name,
        )

        target_df = getattr(ts_container, attribute_name, None)
        if target_df is None:
            LOGGER.info(
                "Creating time-series table '%s.%s' from Excel sheet.",
                ts_container_name,
                attribute_name,
            )
            setattr(ts_container, attribute_name, pd.DataFrame(index=self.n.snapshots))

        target_df = getattr(ts_container, attribute_name)

        for obj_name in df_clean.columns:
            target_df[obj_name] = df_clean[obj_name].to_numpy()

        LOGGER.info(
            "Assigned %s.%s from Excel for %d objects over %d snapshots.",
            ts_container_name,
            attribute_name,
            len(df_clean.columns),
            len(df_clean.index),
        )

    def apply_timeseries_sheets(self, ts_sheets):
        """
        Apply all generic time-series sheets found in the Excel file.

        Parameters
        ----------
        ts_sheets : dict
            Dictionary {sheet_name: DataFrame} for time-series sheets only.
        """
        for sheet_name, df in ts_sheets.items():
            parsed = self.parse_timeseries_sheet_name(sheet_name)
            if parsed is None:
                LOGGER.warning(
                    "Sheet '%s' looks like a time-series sheet but could not be parsed. Skipping.",
                    sheet_name,
                )
                continue

            component_name, attribute_name = parsed
            self.set_timeseries_from_excel(component_name, attribute_name, df)

    def apply_timeseries_fallbacks(self, ts_sheets):
        """
        Apply fallback generators only for a small set of legacy-supported sheets.

        Fallbacks are used only when the corresponding sheet is absent:
        - loads_t_p_set -> add_demand()
        - generators_t_p_max_pu -> add_renewables()
        - storage_units_t_inflow -> add_hydro_inflow()

        Parameters
        ----------
        ts_sheets : dict
            Dictionary of time-series sheets found in Excel.
        """
        if "loads_t_p_set" not in ts_sheets:
            LOGGER.info("Sheet 'loads_t_p_set' not found: using default add_demand().")
            self.add_demand()
        else:
            LOGGER.info("Found Excel sheet 'loads_t_p_set': fallback add_demand() not used.")

        if "generators_t_p_max_pu" not in ts_sheets:
            LOGGER.info("Sheet 'generators_t_p_max_pu' not found: using default add_renewables().")
            self.add_renewables()
        else:
            LOGGER.info("Found Excel sheet 'generators_t_p_max_pu': fallback add_renewables() not used.")

        if "storage_units_t_inflow" not in ts_sheets:
            LOGGER.info("Sheet 'storage_units_t_inflow' not found: using default add_hydro_inflow().")
            self.add_hydro_inflow()
        else:
            LOGGER.info("Found Excel sheet 'storage_units_t_inflow': fallback add_hydro_inflow() not used.")

    def add_demand(self):
        """
        Add demand profiles to the network loads based on daily demand data.
        """
        file_path = f"{self.parser.input_data_path}/{self.parser.input_name_demand}"
        df_demand_day = pd.read_csv(file_path)
        df_demand_day["hour"] = range(0, 24)
        n_days = int(len(self.n.snapshots) / 24)

        for load in self.n.loads.index:
            df_demand_year = self.parser.load_sign * np.random.normal(
                np.tile(df_demand_day["demand"], n_days),
                np.tile(df_demand_day["standard_deviation"], n_days),
            ) * 100
            self.n.loads_t.p_set[load] = df_demand_year

    def add_renewables(self):
        """
        Add per-unit power profiles for renewable generators (solar and wind).
        """
        file_path_pv = f"{self.parser.input_data_path}/{self.parser.input_name_pv}"
        df_pv = pd.read_csv(file_path_pv, skiprows=3, nrows=len(self.n.snapshots))

        file_path_wind = f"{self.parser.input_data_path}/{self.parser.input_name_wind}"
        df_wind = pd.read_csv(file_path_wind, skiprows=3, nrows=len(self.n.snapshots))

        for generator in self.n.generators.index:
            if "solar" in generator.lower() or "pv" in generator.lower():
                self.n.generators_t.p_max_pu[generator] = df_pv["electricity"]
            elif "wind" in generator.lower():
                self.n.generators_t.p_max_pu[generator] = df_wind["electricity"]

    def add_hydro_inflow(self):
        """
        Add inflow time series for hydro StorageUnits, if present.

        This is only a fallback when 'storage_units_t_inflow' is not provided in Excel.
        """
        if self.n.storage_units.empty:
            LOGGER.info("No StorageUnits found in the network; skipping hydro inflow initialization.")
            return

        hydro_mask = self.n.storage_units.index.to_series().str.contains("hydro", case=False, na=False)
        hydro_units = self.n.storage_units.index[hydro_mask]

        if hydro_units.empty:
            LOGGER.info("No hydro StorageUnits found; skipping hydro inflow initialization.")
            return

        n_snapshots = len(self.n.snapshots)

        for unit in hydro_units:
            if "p_nom" not in self.n.storage_units.columns:
                LOGGER.warning(
                    "StorageUnits have no 'p_nom' column; cannot compute inflow for '%s'. Skipping.",
                    unit,
                )
                continue

            capacity = self.n.storage_units.at[unit, "p_nom"]

            if pd.isna(capacity):
                LOGGER.warning(
                    "Capacity 'p_nom' for StorageUnit '%s' is NaN; cannot compute inflow. Skipping.",
                    unit,
                )
                continue

            base_inflow = 0.25 * capacity
            std_inflow = 0.10 * base_inflow
            inflow_profile = np.random.normal(loc=base_inflow, scale=std_inflow, size=n_snapshots)

            self.n.storage_units_t.inflow[unit] = inflow_profile

            LOGGER.info(
                "Hydro inflow initialized for StorageUnit '%s' with mean %.3f and std %.3f per snapshot.",
                unit,
                base_inflow,
                std_inflow,
            )