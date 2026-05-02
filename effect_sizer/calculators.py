"""
Effect size calculators.

All functions are pure Python + scipy — no API keys required.
Each returns an EffectSizeResult with value, magnitude, interpretation,
optional CI, and benchmarks used.
"""

from __future__ import annotations

import math
from typing import Tuple

from scipy import stats  # type: ignore[import-untyped]

from .models import BENCHMARKS, CRAMERS_V_FALLBACK, CRAMERS_V_THRESHOLDS, EffectSizeResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def classify_magnitude(value: float, small: float, medium: float, large: float) -> str:
    """Map an absolute effect size value onto Cohen's verbal label."""
    abs_val = abs(value)
    if abs_val < small:
        return "negligible"
    if abs_val < medium:
        return "small"
    if abs_val < large:
        return "medium"
    return "large"


def _d_variance(d: float, n1: int, n2: int) -> float:
    """Hedges & Olkin (1985) variance estimate for Cohen's d / Hedges' g."""
    return (n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2 - 2))


def _d_ci(d: float, n1: int, n2: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Symmetric CI for d/g using Hedges & Olkin SE approximation."""
    alpha = 1.0 - confidence
    df = n1 + n2 - 2
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    se = math.sqrt(_d_variance(d, n1, n2))
    return d - t_crit * se, d + t_crit * se


def _r_ci(r: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Fisher z-transformation CI for Pearson r."""
    alpha = 1.0 - confidence
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    return math.tanh(z - z_crit * se), math.tanh(z + z_crit * se)


def _cramers_v_thresholds(df_min: int) -> dict:
    return CRAMERS_V_THRESHOLDS.get(df_min, CRAMERS_V_FALLBACK)


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------


def cohens_d(
    m1: float,
    sd1: float,
    n1: int,
    m2: float,
    sd2: float,
    n2: int,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Cohen's d for two independent groups (pooled SD denominator)."""
    if sd1 <= 0 or sd2 <= 0:
        raise ValueError("Standard deviations must be positive.")
    if n1 < 2 or n2 < 2:
        raise ValueError("Group sizes must be >= 2.")

    pooled_sd = math.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    d = (m1 - m2) / pooled_sd

    b = BENCHMARKS["d"]
    magnitude = classify_magnitude(d, b["small"], b["medium"], b["large"])
    ci_l, ci_u = _d_ci(d, n1, n2, confidence)

    direction = "higher" if d >= 0 else "lower"
    interpretation = (
        f"Group 1 scored {direction} than Group 2 by {abs(d):.2f} pooled SDs "
        f"({magnitude} effect by Cohen's benchmarks). "
        f"{confidence * 100:.0f}% CI [{ci_l:.2f}, {ci_u:.2f}]."
    )

    return EffectSizeResult(
        method="Cohen's d",
        value=round(d, 4),
        magnitude=magnitude,
        interpretation=interpretation,
        benchmarks=b,
        inputs={"m1": m1, "sd1": sd1, "n1": n1, "m2": m2, "sd2": sd2, "n2": n2},
        ci_lower=round(ci_l, 4),
        ci_upper=round(ci_u, 4),
    )


# ---------------------------------------------------------------------------
# Hedges' g
# ---------------------------------------------------------------------------


def hedges_g(
    m1: float,
    sd1: float,
    n1: int,
    m2: float,
    sd2: float,
    n2: int,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Hedges' g — bias-corrected Cohen's d (preferred for small samples)."""
    if sd1 <= 0 or sd2 <= 0:
        raise ValueError("Standard deviations must be positive.")
    if n1 < 2 or n2 < 2:
        raise ValueError("Group sizes must be >= 2.")

    pooled_sd = math.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    d = (m1 - m2) / pooled_sd
    df = n1 + n2 - 2
    j = 1.0 - 3.0 / (4.0 * df - 1.0)  # small-sample correction factor
    g = d * j

    b = BENCHMARKS["g"]
    magnitude = classify_magnitude(g, b["small"], b["medium"], b["large"])
    ci_l, ci_u = _d_ci(g, n1, n2, confidence)

    direction = "higher" if g >= 0 else "lower"
    interpretation = (
        f"Bias-corrected effect (J = {j:.4f}): Group 1 scored {direction} than Group 2 "
        f"by {abs(g):.2f} pooled SDs ({magnitude} effect). "
        f"Hedges' g corrects the positive bias in Cohen's d for small samples. "
        f"{confidence * 100:.0f}% CI [{ci_l:.2f}, {ci_u:.2f}]."
    )

    return EffectSizeResult(
        method="Hedges' g",
        value=round(g, 4),
        magnitude=magnitude,
        interpretation=interpretation,
        benchmarks=b,
        inputs={"m1": m1, "sd1": sd1, "n1": n1, "m2": m2, "sd2": sd2, "n2": n2},
        ci_lower=round(ci_l, 4),
        ci_upper=round(ci_u, 4),
    )


# ---------------------------------------------------------------------------
# Glass's Δ
# ---------------------------------------------------------------------------


def glass_delta(
    m1: float,
    sd1: float,
    n1: int,
    m2: float,
    sd2_control: float,
    n2: int,
    confidence: float = 0.95,
) -> EffectSizeResult:
    """Glass's Δ — uses the control group SD as denominator."""
    if sd2_control <= 0:
        raise ValueError("Control group SD must be positive.")
    if sd1 <= 0:
        raise ValueError("Treatment group SD must be positive.")
    if n1 < 2 or n2 < 2:
        raise ValueError("Group sizes must be >= 2.")

    delta = (m1 - m2) / sd2_control

    b = BENCHMARKS["delta"]
    magnitude = classify_magnitude(delta, b["small"], b["medium"], b["large"])
    ci_l, ci_u = _d_ci(delta, n1, n2, confidence)

    direction = "higher" if delta >= 0 else "lower"
    interpretation = (
        f"Treatment group scored {direction} than control by {abs(delta):.2f} control SDs "
        f"({magnitude} effect). Glass's Δ anchors to the control SD, preserving the baseline scale "
        f"when group variances differ. "
        f"{confidence * 100:.0f}% CI [{ci_l:.2f}, {ci_u:.2f}]."
    )

    return EffectSizeResult(
        method="Glass's Δ",
        value=round(delta, 4),
        magnitude=magnitude,
        interpretation=interpretation,
        benchmarks=b,
        inputs={"m1": m1, "sd1": sd1, "n1": n1, "m2": m2, "sd2_control": sd2_control, "n2": n2},
        ci_lower=round(ci_l, 4),
        ci_upper=round(ci_u, 4),
    )


# ---------------------------------------------------------------------------
# Cohen's h
# ---------------------------------------------------------------------------


def cohens_h(p1: float, p2: float) -> EffectSizeResult:
    """Cohen's h for the difference between two proportions."""
    if not (0.0 <= p1 <= 1.0) or not (0.0 <= p2 <= 1.0):
        raise ValueError("Proportions must be in [0, 1].")

    phi1 = 2.0 * math.asin(math.sqrt(p1))
    phi2 = 2.0 * math.asin(math.sqrt(p2))
    h = phi1 - phi2

    b = BENCHMARKS["h"]
    magnitude = classify_magnitude(h, b["small"], b["medium"], b["large"])

    direction = "higher" if h >= 0 else "lower"
    interpretation = (
        f"Proportion 1 ({p1 * 100:.1f}%) is {direction} than Proportion 2 ({p2 * 100:.1f}%) "
        f"with h = {h:.2f} ({magnitude} effect). "
        f"Cohen's h applies the arcsine transformation to stabilise variance across the 0–1 range."
    )

    return EffectSizeResult(
        method="Cohen's h",
        value=round(h, 4),
        magnitude=magnitude,
        interpretation=interpretation,
        benchmarks=b,
        inputs={"p1": p1, "p2": p2},
    )


# ---------------------------------------------------------------------------
# Cohen's f (ANOVA)
# ---------------------------------------------------------------------------


def cohens_f(ss_effect: float, ss_error: float) -> EffectSizeResult:
    """Cohen's f from ANOVA sum-of-squares components."""
    if ss_effect < 0 or ss_error < 0:
        raise ValueError("Sum of squares values must be non-negative.")
    if ss_error == 0:
        raise ValueError("SS_error must be > 0.")

    eta2 = ss_effect / (ss_effect + ss_error)
    f = math.sqrt(eta2 / (1.0 - eta2))

    b = BENCHMARKS["f"]
    magnitude = classify_magnitude(f, b["small"], b["medium"], b["large"])

    interpretation = (
        f"Cohen's f = {f:.2f} (η² = {eta2:.3f}), indicating a {magnitude} effect. "
        f"f represents the SD of group means relative to the within-group SD."
    )

    return EffectSizeResult(
        method="Cohen's f",
        value=round(f, 4),
        magnitude=magnitude,
        interpretation=interpretation,
        benchmarks=b,
        inputs={"ss_effect": ss_effect, "ss_error": ss_error, "eta2_derived": round(eta2, 4)},
    )


# ---------------------------------------------------------------------------
# Cohen's w (chi-square tests)
# ---------------------------------------------------------------------------


def cohens_w(chi2: float, n: int, df: int) -> EffectSizeResult:
    """Cohen's w for chi-square goodness-of-fit or contingency tests."""
    if chi2 < 0:
        raise ValueError("chi2 must be non-negative.")
    if n <= 0:
        raise ValueError("N must be positive.")

    w = math.sqrt(chi2 / n)

    b = BENCHMARKS["w"]
    magnitude = classify_magnitude(w, b["small"], b["medium"], b["large"])

    interpretation = (
        f"Cohen's w = {w:.2f} ({magnitude} effect) for χ²({df}) with N = {n}. "
        f"w measures the deviation between observed and expected cell frequencies."
    )

    return EffectSizeResult(
        method="Cohen's w",
        value=round(w, 4),
        magnitude=magnitude,
        interpretation=interpretation,
        benchmarks=b,
        inputs={"chi2": chi2, "n": n, "df": df},
    )


# ---------------------------------------------------------------------------
# Cramér's V
# ---------------------------------------------------------------------------


def cramers_v(chi2: float, n: int, rows: int, cols: int) -> EffectSizeResult:
    """Cramér's V for contingency tables with df-adjusted benchmarks."""
    if chi2 < 0:
        raise ValueError("chi2 must be non-negative.")
    if n <= 0:
        raise ValueError("N must be positive.")
    if rows < 2 or cols < 2:
        raise ValueError("rows and cols must both be >= 2.")

    k = min(rows, cols)
    df_min = k - 1
    v = math.sqrt(chi2 / (n * df_min))

    thresholds = _cramers_v_thresholds(df_min)
    magnitude = classify_magnitude(v, thresholds["small"], thresholds["medium"], thresholds["large"])

    b = {**thresholds, "label": "Cramér's V", "df_min": df_min}

    interpretation = (
        f"Cramér's V = {v:.2f} ({magnitude} effect for {rows}×{cols} table). "
        f"Thresholds adjusted for df_min = {df_min}: "
        f"small ≥ {thresholds['small']}, "
        f"medium ≥ {thresholds['medium']}, "
        f"large ≥ {thresholds['large']}."
    )

    return EffectSizeResult(
        method="Cramér's V",
        value=round(v, 4),
        magnitude=magnitude,
        interpretation=interpretation,
        benchmarks=b,
        inputs={"chi2": chi2, "n": n, "rows": rows, "cols": cols},
    )


# ---------------------------------------------------------------------------
# Eta-squared (η²)
# ---------------------------------------------------------------------------


def eta_squared(ss_effect: float, ss_total: float) -> EffectSizeResult:
    """Eta-squared (η²): proportion of total variance explained by the effect."""
    if ss_total <= 0:
        raise ValueError("SS_total must be positive.")
    if ss_effect < 0:
        raise ValueError("SS_effect must be non-negative.")
    if ss_effect > ss_total:
        raise ValueError("SS_effect cannot exceed SS_total.")

    eta2 = ss_effect / ss_total

    b = BENCHMARKS["eta2"]
    magnitude = classify_magnitude(eta2, b["small"], b["medium"], b["large"])

    interpretation = (
        f"η² = {eta2:.3f} — the effect accounts for {eta2 * 100:.1f}% of total variance "
        f"({magnitude} effect). η² is positively biased; prefer ω² for small samples."
    )

    return EffectSizeResult(
        method="Eta-squared (η²)",
        value=round(eta2, 4),
        magnitude=magnitude,
        interpretation=interpretation,
        benchmarks=b,
        inputs={"ss_effect": ss_effect, "ss_total": ss_total},
        power_note="η² is biased upward. Consider ω² for samples under N = 50.",
    )


# ---------------------------------------------------------------------------
# Omega-squared (ω²)
# ---------------------------------------------------------------------------


def omega_squared(
    ss_effect: float, ss_error: float, df_effect: int, n: int
) -> EffectSizeResult:
    """Omega-squared (ω²): bias-corrected η² for one-way ANOVA."""
    if ss_effect < 0 or ss_error < 0:
        raise ValueError("Sum of squares values must be non-negative.")
    if df_effect < 1:
        raise ValueError("df_effect must be >= 1.")
    if n <= df_effect + 1:
        raise ValueError("n must be > df_effect + 1.")

    df_error = n - df_effect - 1
    ms_error = ss_error / df_error
    ss_total = ss_effect + ss_error
    omega2 = (ss_effect - df_effect * ms_error) / (ss_total + ms_error)
    omega2 = max(0.0, omega2)  # clamp: ω² is defined to be ≥ 0

    b = BENCHMARKS["omega2"]
    magnitude = classify_magnitude(omega2, b["small"], b["medium"], b["large"])

    interpretation = (
        f"ω² = {omega2:.3f} — bias-corrected estimate: {omega2 * 100:.1f}% of variance "
        f"attributable to the effect ({magnitude}). "
        f"Preferred over η² for smaller or unbalanced designs."
    )

    return EffectSizeResult(
        method="Omega-squared (ω²)",
        value=round(omega2, 4),
        magnitude=magnitude,
        interpretation=interpretation,
        benchmarks=b,
        inputs={"ss_effect": ss_effect, "ss_error": ss_error, "df_effect": df_effect, "n": n},
    )


# ---------------------------------------------------------------------------
# Pearson r
# ---------------------------------------------------------------------------


def pearson_r(r: float, n: int, confidence: float = 0.95) -> EffectSizeResult:
    """Pearson r as an effect size with Fisher-z CI."""
    if abs(r) >= 1.0:
        raise ValueError("r must be strictly between -1 and 1.")
    if n < 4:
        raise ValueError("n must be >= 4 for Fisher z CI computation.")

    b = BENCHMARKS["r"]
    magnitude = classify_magnitude(r, b["small"], b["medium"], b["large"])
    ci_l, ci_u = _r_ci(r, n, confidence)

    direction = "positive" if r >= 0 else "negative"
    r_sq = r**2
    interpretation = (
        f"r = {r:.2f} ({direction} {magnitude} association, N = {n}). "
        f"Explains {r_sq * 100:.1f}% of shared variance (r² = {r_sq:.3f}). "
        f"{confidence * 100:.0f}% CI [{ci_l:.2f}, {ci_u:.2f}]."
    )

    return EffectSizeResult(
        method="Pearson r",
        value=round(r, 4),
        magnitude=magnitude,
        interpretation=interpretation,
        benchmarks=b,
        inputs={"r": r, "n": n},
        ci_lower=round(ci_l, 4),
        ci_upper=round(ci_u, 4),
    )
