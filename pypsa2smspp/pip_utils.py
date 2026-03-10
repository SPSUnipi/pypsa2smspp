# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 16:19:44 2026

@author: aless
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional



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

