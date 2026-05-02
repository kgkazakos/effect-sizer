"""Unit tests for effect_sizer.calculators."""

from __future__ import annotations

import math

import pytest

from effect_sizer.calculators import (
    classify_magnitude,
    cohens_d,
    cohens_f,
    cohens_h,
    cohens_w,
    cramers_v,
    eta_squared,
    glass_delta,
    hedges_g,
    omega_squared,
    pearson_r,
)
from effect_sizer.models import EffectSizeResult


# ---------------------------------------------------------------------------
# classify_magnitude
# ---------------------------------------------------------------------------


class TestClassifyMagnitude:
    def test_negligible(self):
        assert classify_magnitude(0.1, 0.2, 0.5, 0.8) == "negligible"

    def test_small_at_boundary(self):
        assert classify_magnitude(0.2, 0.2, 0.5, 0.8) == "small"

    def test_small_interior(self):
        assert classify_magnitude(0.35, 0.2, 0.5, 0.8) == "small"

    def test_medium_at_boundary(self):
        assert classify_magnitude(0.5, 0.2, 0.5, 0.8) == "medium"

    def test_medium_interior(self):
        assert classify_magnitude(0.65, 0.2, 0.5, 0.8) == "medium"

    def test_large_at_boundary(self):
        assert classify_magnitude(0.8, 0.2, 0.5, 0.8) == "large"

    def test_large_interior(self):
        assert classify_magnitude(1.2, 0.2, 0.5, 0.8) == "large"

    def test_negative_values_use_abs(self):
        assert classify_magnitude(-0.6, 0.2, 0.5, 0.8) == "medium"

    def test_zero_is_negligible(self):
        assert classify_magnitude(0.0, 0.2, 0.5, 0.8) == "negligible"


# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------


class TestCohensD:
    def test_returns_result_type(self):
        r = cohens_d(6.0, 1.0, 30, 5.0, 1.0, 30)
        assert isinstance(r, EffectSizeResult)

    def test_method_name(self):
        r = cohens_d(6.0, 1.0, 30, 5.0, 1.0, 30)
        assert r.method == "Cohen's d"

    def test_large_effect(self):
        r = cohens_d(6.0, 1.0, 30, 4.0, 1.0, 30)
        assert r.magnitude == "large"
        assert r.value == pytest.approx(2.0, abs=0.01)

    def test_medium_effect(self):
        r = cohens_d(5.5, 1.0, 30, 5.0, 1.0, 30)
        assert r.magnitude == "medium"
        assert r.value == pytest.approx(0.5, abs=0.001)

    def test_small_effect(self):
        r = cohens_d(5.3, 1.0, 30, 5.0, 1.0, 30)
        assert r.magnitude == "small"

    def test_negligible_effect(self):
        r = cohens_d(5.1, 1.0, 30, 5.0, 1.0, 30)
        assert r.magnitude == "negligible"

    def test_negative_direction(self):
        r = cohens_d(4.0, 1.0, 30, 6.0, 1.0, 30)
        assert r.value < 0
        assert r.magnitude == "large"

    def test_zero_difference(self):
        r = cohens_d(5.0, 1.0, 30, 5.0, 1.0, 30)
        assert r.value == 0.0
        assert r.magnitude == "negligible"

    def test_ci_lower_less_than_upper(self):
        r = cohens_d(6.0, 1.0, 30, 5.0, 1.0, 30)
        assert r.ci_lower < r.ci_upper

    def test_ci_contains_value(self):
        r = cohens_d(6.0, 1.0, 30, 5.0, 1.0, 30)
        assert r.ci_lower < r.value < r.ci_upper

    def test_ci_width_decreases_with_n(self):
        r_small = cohens_d(6.0, 1.0, 20, 5.0, 1.0, 20)
        r_large = cohens_d(6.0, 1.0, 200, 5.0, 1.0, 200)
        width_small = r_small.ci_upper - r_small.ci_lower
        width_large = r_large.ci_upper - r_large.ci_lower
        assert width_large < width_small

    def test_90_ci_narrower_than_95(self):
        r95 = cohens_d(6.0, 1.0, 30, 5.0, 1.0, 30, confidence=0.95)
        r90 = cohens_d(6.0, 1.0, 30, 5.0, 1.0, 30, confidence=0.90)
        width95 = r95.ci_upper - r95.ci_lower
        width90 = r90.ci_upper - r90.ci_lower
        assert width90 < width95

    def test_unequal_n(self):
        r = cohens_d(6.0, 1.0, 40, 5.0, 1.0, 20)
        assert isinstance(r, EffectSizeResult)
        assert r.value == pytest.approx(1.0, abs=0.01)

    def test_unequal_sd(self):
        # Pooled SD should weight by n
        r = cohens_d(6.0, 2.0, 30, 5.0, 1.0, 30)
        assert isinstance(r, EffectSizeResult)
        assert r.value > 0

    def test_value_rounded_to_4dp(self):
        r = cohens_d(5.3, 1.0, 30, 5.0, 1.0, 30)
        assert len(str(abs(r.value)).split(".")[-1]) <= 4

    def test_invalid_sd_raises(self):
        with pytest.raises(ValueError, match="Standard deviations"):
            cohens_d(5.0, 0.0, 30, 4.0, 1.0, 30)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="Group sizes"):
            cohens_d(5.0, 1.0, 1, 4.0, 1.0, 30)

    def test_inputs_stored(self):
        r = cohens_d(6.0, 1.0, 30, 5.0, 1.0, 30)
        assert r.inputs["m1"] == 6.0
        assert r.inputs["n1"] == 30


# ---------------------------------------------------------------------------
# hedges_g
# ---------------------------------------------------------------------------


class TestHedgesG:
    def test_returns_result_type(self):
        r = hedges_g(6.0, 1.0, 30, 5.0, 1.0, 30)
        assert isinstance(r, EffectSizeResult)

    def test_method_name(self):
        r = hedges_g(6.0, 1.0, 30, 5.0, 1.0, 30)
        assert r.method == "Hedges' g"

    def test_less_than_d_for_small_n(self):
        d = cohens_d(6.0, 1.0, 5, 5.0, 1.0, 5)
        g = hedges_g(6.0, 1.0, 5, 5.0, 1.0, 5)
        assert abs(g.value) < abs(d.value)

    def test_close_to_d_for_large_n(self):
        d = cohens_d(6.0, 1.0, 500, 5.0, 1.0, 500)
        g = hedges_g(6.0, 1.0, 500, 5.0, 1.0, 500)
        assert abs(g.value - d.value) < 0.01

    def test_large_effect(self):
        r = hedges_g(7.0, 1.0, 30, 5.0, 1.0, 30)
        assert r.magnitude == "large"

    def test_ci_present(self):
        r = hedges_g(6.0, 1.0, 30, 5.0, 1.0, 30)
        assert r.ci_lower is not None
        assert r.ci_upper is not None

    def test_correction_applied(self):
        # For n1=n2=5, df=8, j=1-3/31=0.9032
        r = hedges_g(6.0, 1.0, 5, 5.0, 1.0, 5)
        expected_j = 1.0 - 3.0 / (4.0 * 8 - 1.0)
        expected_g = 1.0 * expected_j  # d=1.0 with these inputs
        assert r.value == pytest.approx(expected_g, abs=0.001)


# ---------------------------------------------------------------------------
# glass_delta
# ---------------------------------------------------------------------------


class TestGlassDelta:
    def test_uses_control_sd(self):
        # Changing sd1 (treatment) should NOT change the result
        r1 = glass_delta(5.5, 1.0, 20, 5.0, 1.0, 20)
        r2 = glass_delta(5.5, 2.0, 20, 5.0, 1.0, 20)
        assert r1.value == r2.value

    def test_different_from_cohens_d_when_sds_differ(self):
        d_r = cohens_d(5.5, 2.0, 30, 5.0, 1.0, 30)
        g_r = glass_delta(5.5, 2.0, 30, 5.0, 1.0, 30)
        assert d_r.value != g_r.value

    def test_method_name(self):
        r = glass_delta(5.5, 1.0, 30, 5.0, 1.0, 30)
        assert r.method == "Glass's Δ"

    def test_magnitude_small(self):
        r = glass_delta(5.2, 1.0, 30, 5.0, 1.0, 30)
        assert r.magnitude == "small"

    def test_ci_present(self):
        r = glass_delta(6.0, 1.0, 30, 5.0, 1.0, 30)
        assert r.ci_lower is not None

    def test_invalid_control_sd_raises(self):
        with pytest.raises(ValueError, match="Control group SD"):
            glass_delta(5.5, 1.0, 30, 5.0, 0.0, 30)


# ---------------------------------------------------------------------------
# cohens_h
# ---------------------------------------------------------------------------


class TestCohensH:
    def test_zero_difference(self):
        r = cohens_h(0.5, 0.5)
        assert r.value == pytest.approx(0.0, abs=1e-9)
        assert r.magnitude == "negligible"

    def test_positive_direction(self):
        r = cohens_h(0.7, 0.5)
        assert r.value > 0

    def test_negative_direction(self):
        r = cohens_h(0.3, 0.5)
        assert r.value < 0

    def test_large_effect(self):
        # p1=0.8, p2=0.2 -> large
        r = cohens_h(0.8, 0.2)
        assert r.magnitude == "large"

    def test_small_effect(self):
        # p1=0.55, p2=0.50 -> small
        r = cohens_h(0.55, 0.50)
        assert r.magnitude in {"small", "negligible"}

    def test_negligible_effect(self):
        r = cohens_h(0.51, 0.50)
        assert r.magnitude == "negligible"

    def test_method_name(self):
        r = cohens_h(0.6, 0.5)
        assert r.method == "Cohen's h"

    def test_symmetry_magnitude(self):
        # |h(p1, p2)| == |h(p2, p1)|
        r1 = cohens_h(0.7, 0.4)
        r2 = cohens_h(0.4, 0.7)
        assert abs(r1.value) == pytest.approx(abs(r2.value), abs=1e-9)
        assert r1.magnitude == r2.magnitude

    def test_arcsine_transform_applied(self):
        # phi1 = 2*arcsin(sqrt(p))
        p1, p2 = 0.6, 0.4
        phi1 = 2.0 * math.asin(math.sqrt(p1))
        phi2 = 2.0 * math.asin(math.sqrt(p2))
        expected = phi1 - phi2
        r = cohens_h(p1, p2)
        assert r.value == pytest.approx(expected, abs=0.0001)

    def test_invalid_proportion_raises(self):
        with pytest.raises(ValueError, match="Proportions"):
            cohens_h(1.2, 0.5)


# ---------------------------------------------------------------------------
# cohens_f
# ---------------------------------------------------------------------------


class TestCohensF:
    def test_large(self):
        # ss_effect=100, ss_error=300 -> eta2=0.25, f=0.577 -> large
        r = cohens_f(100.0, 300.0)
        assert r.magnitude == "large"
        assert r.value == pytest.approx(0.5774, abs=0.001)

    def test_medium(self):
        # ss_effect=20, ss_error=280 -> eta2=0.0667, f=0.267 -> medium
        r = cohens_f(20.0, 280.0)
        assert r.magnitude == "medium"

    def test_small(self):
        # ss_effect=5, ss_error=495 -> f~0.1005 -> small
        r = cohens_f(5.0, 495.0)
        assert r.magnitude == "small"

    def test_negligible(self):
        r = cohens_f(1.0, 999.0)
        assert r.magnitude == "negligible"

    def test_method_name(self):
        r = cohens_f(100.0, 300.0)
        assert r.method == "Cohen's f"

    def test_eta2_in_inputs(self):
        r = cohens_f(100.0, 300.0)
        assert "eta2_derived" in r.inputs

    def test_invalid_ss_raises(self):
        with pytest.raises(ValueError):
            cohens_f(100.0, 0.0)


# ---------------------------------------------------------------------------
# cohens_w
# ---------------------------------------------------------------------------


class TestCohensW:
    def test_small(self):
        # w = sqrt(chi2/n) = sqrt(2/200) = 0.1 -> exactly small
        r = cohens_w(2.0, 200, 1)
        assert r.magnitude == "small"
        assert r.value == pytest.approx(0.1, abs=0.001)

    def test_medium(self):
        # w = sqrt(18/200) = 0.3 -> medium
        r = cohens_w(18.0, 200, 3)
        assert r.magnitude == "medium"

    def test_method_name(self):
        r = cohens_w(10.0, 100, 2)
        assert r.method == "Cohen's w"

    def test_relationship_to_chi2(self):
        # Doubling chi2 multiplies w by sqrt(2)
        r1 = cohens_w(10.0, 100, 2)
        r2 = cohens_w(20.0, 100, 2)
        assert r2.value == pytest.approx(r1.value * math.sqrt(2), abs=0.001)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="N must be positive"):
            cohens_w(10.0, 0, 2)


# ---------------------------------------------------------------------------
# cramers_v
# ---------------------------------------------------------------------------


class TestCramersV:
    def test_2x2_large(self):
        # k=2, df_min=1, thresholds: small=0.10, medium=0.30, large=0.50
        # V = sqrt(25/100) = 0.5 -> large
        r = cramers_v(25.0, 100, 2, 2)
        assert r.magnitude == "large"
        assert r.value == pytest.approx(0.5, abs=0.001)

    def test_2x2_small(self):
        # V = sqrt(4/100) = 0.2 -> small
        r = cramers_v(4.0, 100, 2, 2)
        assert r.magnitude == "small"

    def test_3x4_thresholds(self):
        # k=3, df_min=2, thresholds: small=0.07, medium=0.21, large=0.35
        # V = sqrt(10/200) = 0.2236 -> medium (>= 0.21)
        r = cramers_v(10.0, 100, 3, 4)
        assert r.magnitude == "medium"

    def test_method_name(self):
        r = cramers_v(10.0, 100, 2, 2)
        assert r.method == "Cramér's V"

    def test_symmetric_table(self):
        # V same regardless of rows/cols orientation
        r1 = cramers_v(10.0, 100, 3, 4)
        r2 = cramers_v(10.0, 100, 4, 3)
        assert r1.value == r2.value

    def test_df_min_in_benchmarks(self):
        r = cramers_v(10.0, 100, 3, 4)
        assert r.benchmarks["df_min"] == 2

    def test_invalid_rows_raises(self):
        with pytest.raises(ValueError, match="rows and cols"):
            cramers_v(10.0, 100, 1, 4)


# ---------------------------------------------------------------------------
# eta_squared
# ---------------------------------------------------------------------------


class TestEtaSquared:
    def test_large(self):
        r = eta_squared(120.0, 600.0)
        assert r.magnitude == "large"
        assert r.value == pytest.approx(0.2, abs=0.001)

    def test_medium(self):
        # eta2 = 0.07 -> medium (>= 0.06, < 0.14)
        r = eta_squared(7.0, 100.0)
        assert r.magnitude == "medium"

    def test_small(self):
        # eta2 = 0.01 -> small
        r = eta_squared(1.0, 100.0)
        assert r.magnitude == "small"

    def test_negligible(self):
        r = eta_squared(0.5, 100.0)
        assert r.magnitude == "negligible"

    def test_power_note_present(self):
        r = eta_squared(100.0, 500.0)
        assert r.power_note is not None
        assert "biased" in r.power_note.lower() or "ω" in r.power_note

    def test_value_correct(self):
        r = eta_squared(80.0, 400.0)
        assert r.value == pytest.approx(0.2, abs=0.001)

    def test_method_name(self):
        r = eta_squared(80.0, 400.0)
        assert "η²" in r.method or "Eta" in r.method

    def test_invalid_ss_total_raises(self):
        with pytest.raises(ValueError):
            eta_squared(100.0, 0.0)

    def test_ss_effect_exceeds_total_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            eta_squared(200.0, 100.0)


# ---------------------------------------------------------------------------
# omega_squared
# ---------------------------------------------------------------------------


class TestOmegaSquared:
    def test_less_than_eta_squared(self):
        # For same data, ω² < η²
        eta = eta_squared(120.0, 600.0)
        omega = omega_squared(120.0, 480.0, 3, 60)
        assert omega.value < eta.value

    def test_clamped_to_zero(self):
        # Small effect, many groups, small n → ω² can go negative → clamp
        r = omega_squared(5.0, 500.0, 10, 30)
        assert r.value >= 0.0

    def test_large_effect(self):
        r = omega_squared(120.0, 480.0, 3, 60)
        assert r.magnitude in {"medium", "large"}

    def test_method_name(self):
        r = omega_squared(120.0, 480.0, 3, 60)
        assert "ω²" in r.method or "Omega" in r.method

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="n must be"):
            omega_squared(100.0, 400.0, 3, 4)

    def test_invalid_df_raises(self):
        with pytest.raises(ValueError, match="df_effect"):
            omega_squared(100.0, 400.0, 0, 60)


# ---------------------------------------------------------------------------
# pearson_r
# ---------------------------------------------------------------------------


class TestPearsonR:
    def test_large_positive(self):
        r = pearson_r(0.55, 100)
        assert r.magnitude == "large"

    def test_large_negative(self):
        r = pearson_r(-0.55, 100)
        assert r.magnitude == "large"

    def test_medium(self):
        r = pearson_r(0.35, 100)
        assert r.magnitude == "medium"

    def test_small(self):
        r = pearson_r(0.15, 100)
        assert r.magnitude == "small"

    def test_negligible(self):
        r = pearson_r(0.05, 100)
        assert r.magnitude == "negligible"

    def test_ci_lower_less_than_upper(self):
        r = pearson_r(0.4, 100)
        assert r.ci_lower < r.ci_upper

    def test_ci_contains_value(self):
        r = pearson_r(0.5, 100)
        assert r.ci_lower < r.value < r.ci_upper

    def test_ci_narrower_with_larger_n(self):
        r_small = pearson_r(0.3, 30)
        r_large = pearson_r(0.3, 300)
        w_small = r_small.ci_upper - r_small.ci_lower
        w_large = r_large.ci_upper - r_large.ci_lower
        assert w_large < w_small

    def test_r_squared_mentioned_in_interpretation(self):
        r = pearson_r(0.4, 100)
        assert "r²" in r.interpretation or "r2" in r.interpretation.lower()

    def test_invalid_r_raises(self):
        with pytest.raises(ValueError, match="strictly between"):
            pearson_r(1.0, 100)

    def test_invalid_r_negative_one_raises(self):
        with pytest.raises(ValueError, match="strictly between"):
            pearson_r(-1.0, 100)

    def test_small_n_raises(self):
        with pytest.raises(ValueError, match="n must be"):
            pearson_r(0.3, 3)

    def test_method_name(self):
        r = pearson_r(0.3, 100)
        assert r.method == "Pearson r"

    def test_ci_lower_upper_in_result(self):
        r = pearson_r(0.3, 100)
        assert r.ci_lower is not None
        assert r.ci_upper is not None

    def test_to_dict_has_ci(self):
        r = pearson_r(0.3, 100)
        d = r.to_dict()
        assert "confidence_interval" in d
        assert d["confidence_interval"]["lower"] is not None


# ---------------------------------------------------------------------------
# to_dict serialisation
# ---------------------------------------------------------------------------


class TestToDict:
    def test_cohens_d_to_dict_fields(self):
        r = cohens_d(6.0, 1.0, 30, 5.0, 1.0, 30)
        d = r.to_dict()
        assert "method" in d
        assert "value" in d
        assert "magnitude" in d
        assert "interpretation" in d
        assert "benchmarks" in d
        assert "inputs" in d

    def test_eta_squared_no_ci_in_dict(self):
        r = eta_squared(100.0, 500.0)
        d = r.to_dict()
        assert "confidence_interval" not in d

    def test_power_note_in_dict_when_present(self):
        r = eta_squared(100.0, 500.0)
        d = r.to_dict()
        assert "power_note" in d

    def test_narrative_absent_when_none(self):
        r = cohens_d(6.0, 1.0, 30, 5.0, 1.0, 30)
        d = r.to_dict()
        assert "narrative" not in d
