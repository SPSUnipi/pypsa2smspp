# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 16:19:44 2026

@author: aless
"""

import yaml
from copy import deepcopy
from pathlib import Path
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional



class AttrDict(dict):
    """
    Dictionary with attribute-style access.
    Nested dictionaries are automatically wrapped.
    """

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e

        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[key] = value
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



def load_yaml_config(path, defaults=None, as_attrdict=True):
    """
    Load a YAML configuration file and return a plain dictionary of dictionaries.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.
    defaults : dict, optional
        Default configuration to be recursively updated by the YAML content.

    Returns
    -------
    cfg : dict
        Nested dictionary with configuration values.
    """
    path = Path(path)

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    if defaults is not None:
        cfg = _recursive_update(deepcopy(defaults), cfg)
        
    if as_attrdict:
        cfg = AttrDict(cfg)

    return cfg


def _recursive_update(base, new):
    """
    Recursively update a dictionary.
    """
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _recursive_update(base[k], v)
        else:
            base[k] = v
    return base


def select_block_mode(cfg, dimensions):
    mode = cfg.run.get("mode", "auto")

    if mode != "auto":
        return mode

    if cfg["transformation"].get("expansion_ucblock", False):
        return "ucblock"

    if dimensions.get("InvestmentBlock", {}).get("NumAssets", 0) == 0:
        return "ucblock"

    return "investmentblock"


def _build_smspp_paths(cfg, output_prefix: str):
    """
    Build standard SMS++ artifact paths using output_prefix + cfg.io.name.
    """
    workdir = Path(cfg.io.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    name = cfg.io.name
    stem = f"{output_prefix}_{name}"

    temporary_smspp_file = str(workdir / f"{stem}_network.nc")
    output_file = str(workdir / f"{stem}_log.txt")
    solution_file = str(workdir / f"{stem}_solution.nc")

    return temporary_smspp_file, output_file, solution_file



def build_optimize_call_from_cfg(cfg, mode: str, configfile, temporary_smspp_file, output_file, solution_file):
    """
    Build args and kwargs for pysmspp.optimize from cfg.smspp.<mode>.

    Rules:
    - args are always: (temporary_smspp_file, output_file, solution_file)
    - kwargs are derived automatically from cfg.smspp.<mode>, excluding reserved keys
    - global cfg.smspp.log_executable_call is included (can be overridden by per-mode key if present)
    """
    mode_cfg = getattr(cfg.smspp, mode)

    # Required positional args for pysmspp.optimize
    args = (temporary_smspp_file, output_file, solution_file)

    # Convert mode config to a plain dict (AttrDict is a dict subclass)
    mode_dict = dict(mode_cfg)

    # Keys that are NOT forwarded as kwargs
    reserved = {"template", "output_prefix"}

    kwargs = {k: v for k, v in mode_dict.items() if k not in reserved}

    # Add global flag (unless overridden locally)
    if "log_executable_call" not in kwargs:
        kwargs["log_executable_call"] = bool(cfg.smspp.get("log_executable_call", True))

    return args, kwargs


@dataclass
class StepTimer:
    """
    Collect timings for pipeline steps.
    Stores rows as a list of dicts to make it easy to convert to DataFrame later.
    """
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def start(self, step: str, extra: Optional[Dict[str, Any]] = None):
        row = {"step": step, "t_start": time.time(), "elapsed_s": None, "status": "RUNNING"}
        if extra:
            row.update(extra)
        self.rows.append(row)

    def stop(self, status: str = "OK", extra: Optional[Dict[str, Any]] = None):
        row = self.rows[-1]
        row["t_end"] = time.time()
        row["elapsed_s"] = row["t_end"] - row["t_start"]
        row["status"] = status
        if extra:
            row.update(extra)

    def total(self) -> float:
        if not self.rows:
            return 0.0
        t0 = self.rows[0]["t_start"]
        t1 = self.rows[-1].get("t_end", time.time())
        return t1 - t0

    def print_summary(self):
        # Simple aligned print without external deps
        if not self.rows:
            print("No timing rows.")
            return

        step_w = max(len(r["step"]) for r in self.rows)
        print("\nTiming summary:")
        print(f"{'STEP'.ljust(step_w)}  STATUS     ELAPSED [s]")
        print("-" * (step_w + 24))

        for r in self.rows:
            elapsed = r["elapsed_s"]
            elapsed_str = f"{elapsed:.3f}" if elapsed is not None else "-"
            print(f"{r['step'].ljust(step_w)}  {str(r['status']).ljust(9)} {elapsed_str}")

        print("-" * (step_w + 24))
        print(f"{'TOTAL'.ljust(step_w)}  {'':9} {self.total():.3f}\n")


class step:
    """
    Context manager for timing + printing step progress.

    Usage:
        with step(timer, "direct", verbose=True):
            ...
    """
    def __init__(self, timer: StepTimer, name: str, verbose: bool = True, extra: Optional[Dict[str, Any]] = None):
        self.timer = timer
        self.name = name
        self.verbose = verbose
        self.extra = extra

    def __enter__(self):
        self.timer.start(self.name, extra=self.extra)
        if self.verbose:
            print(f"[START] {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            self.timer.stop(status="OK")
            if self.verbose:
                elapsed = self.timer.rows[-1]["elapsed_s"]
                print(f"[ OK  ] {self.name} ({elapsed:.3f}s)")
            return False  # do not swallow exceptions
        else:
            self.timer.stop(status="FAIL", extra={"error": str(exc)})
            if self.verbose:
                elapsed = self.timer.rows[-1]["elapsed_s"]
                print(f"[FAIL ] {self.name} ({elapsed:.3f}s) -> {exc}")
            return False

