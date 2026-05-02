"""Core dataclasses and benchmark reference data for effect-sizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


# Cohen (1988) / Lakens (2013) benchmark thresholds
BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "d": {"small": 0.2, "medium": 0.5, "large": 0.8, "label": "Cohen's d"},
    "g": {"small": 0.2, "medium": 0.5, "large": 0.8, "label": "Hedges' g"},
    "delta": {"small": 0.2, "medium": 0.5, "large": 0.8, "label": "Glass's Δ"},
    "h": {"small": 0.2, "medium": 0.5, "large": 0.8, "label": "Cohen's h"},
    "f": {"small": 0.1, "medium": 0.25, "large": 0.4, "label": "Cohen's f"},
    "w": {"small": 0.1, "medium": 0.3, "large": 0.5, "label": "Cohen's w"},
    "r": {"small": 0.1, "medium": 0.3, "large": 0.5, "label": "Pearson r"},
    "eta2": {"small": 0.01, "medium": 0.06, "large": 0.14, "label": "η²"},
    "omega2": {"small": 0.01, "medium": 0.06, "large": 0.14, "label": "ω²"},
}

# Cramér's V thresholds by df_min = min(rows, cols) - 1 — Lakens (2013)
CRAMERS_V_THRESHOLDS: Dict[int, Dict[str, float]] = {
    1: {"small": 0.10, "medium": 0.30, "large": 0.50},
    2: {"small": 0.07, "medium": 0.21, "large": 0.35},
    3: {"small": 0.06, "medium": 0.17, "large": 0.29},
    4: {"small": 0.05, "medium": 0.15, "large": 0.25},
}
CRAMERS_V_FALLBACK: Dict[str, float] = {"small": 0.04, "medium": 0.13, "large": 0.22}


@dataclass
class EffectSizeResult:
    """Container for a single effect size calculation."""

    method: str
    value: float
    magnitude: str  # negligible | small | medium | large
    interpretation: str
    benchmarks: Dict[str, Any]
    inputs: Dict[str, Any]
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    narrative: Optional[str] = None
    power_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dict (JSON-friendly)."""
        out: Dict[str, Any] = {
            "method": self.method,
            "value": self.value,
            "magnitude": self.magnitude,
            "interpretation": self.interpretation,
            "benchmarks": self.benchmarks,
            "inputs": self.inputs,
        }
        if self.ci_lower is not None:
            out["confidence_interval"] = {
                "lower": self.ci_lower,
                "upper": self.ci_upper,
            }
        if self.narrative is not None:
            out["narrative"] = self.narrative
        if self.power_note is not None:
            out["power_note"] = self.power_note
        return out
