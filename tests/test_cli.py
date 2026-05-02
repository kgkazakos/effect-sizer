"""CLI integration tests for effect-sizer."""

from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner

from effect_sizer.cli import cli

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Help / smoke
# ---------------------------------------------------------------------------


def test_cli_help(runner):
    r = runner.invoke(cli, ["--help"])
    assert r.exit_code == 0
    assert "effect-sizer" in r.output.lower() or "Usage" in r.output


def test_cohens_d_help(runner):
    r = runner.invoke(cli, ["cohens-d", "--help"])
    assert r.exit_code == 0
    assert "--m1" in r.output


def test_pearson_r_help(runner):
    r = runner.invoke(cli, ["pearson-r", "--help"])
    assert r.exit_code == 0
    assert "--r" in r.output


# ---------------------------------------------------------------------------
# cohens-d
# ---------------------------------------------------------------------------


def test_cohens_d_basic_output(runner):
    r = runner.invoke(cli, [
        "cohens-d",
        "--m1", "6.0", "--sd1", "1.0", "--n1", "30",
        "--m2", "5.0", "--sd2", "1.0", "--n2", "30",
    ])
    assert r.exit_code == 0
    assert "Cohen" in r.output


def test_cohens_d_json_output(runner):
    r = runner.invoke(cli, [
        "cohens-d",
        "--m1", "6.0", "--sd1", "1.0", "--n1", "30",
        "--m2", "5.0", "--sd2", "1.0", "--n2", "30",
        "--json",
    ])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert data["method"] == "Cohen's d"


def test_cohens_d_json_value_field(runner):
    r = runner.invoke(cli, [
        "cohens-d",
        "--m1", "6.0", "--sd1", "1.0", "--n1", "30",
        "--m2", "5.0", "--sd2", "1.0", "--n2", "30",
        "--json",
    ])
    data = json.loads(r.output)
    assert data["value"] == pytest.approx(1.0, abs=0.01)


def test_cohens_d_json_magnitude_field(runner):
    r = runner.invoke(cli, [
        "cohens-d",
        "--m1", "6.0", "--sd1", "1.0", "--n1", "30",
        "--m2", "5.0", "--sd2", "1.0", "--n2", "30",
        "--json",
    ])
    data = json.loads(r.output)
    assert data["magnitude"] == "large"


def test_cohens_d_json_ci_field(runner):
    r = runner.invoke(cli, [
        "cohens-d",
        "--m1", "6.0", "--sd1", "1.0", "--n1", "30",
        "--m2", "5.0", "--sd2", "1.0", "--n2", "30",
        "--json",
    ])
    data = json.loads(r.output)
    assert "confidence_interval" in data
    assert data["confidence_interval"]["lower"] is not None


def test_cohens_d_confidence_90(runner):
    r = runner.invoke(cli, [
        "cohens-d",
        "--m1", "6.0", "--sd1", "1.0", "--n1", "30",
        "--m2", "5.0", "--sd2", "1.0", "--n2", "30",
        "--confidence", "0.90",
        "--json",
    ])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert "confidence_interval" in data


def test_cohens_d_json_has_inputs(runner):
    r = runner.invoke(cli, [
        "cohens-d",
        "--m1", "6.0", "--sd1", "1.0", "--n1", "30",
        "--m2", "5.0", "--sd2", "1.0", "--n2", "30",
        "--json",
    ])
    data = json.loads(r.output)
    assert "inputs" in data
    assert data["inputs"]["m1"] == 6.0


# ---------------------------------------------------------------------------
# hedges-g
# ---------------------------------------------------------------------------


def test_hedges_g_basic(runner):
    r = runner.invoke(cli, [
        "hedges-g",
        "--m1", "6.0", "--sd1", "1.0", "--n1", "30",
        "--m2", "5.0", "--sd2", "1.0", "--n2", "30",
    ])
    assert r.exit_code == 0
    assert "Hedges" in r.output


def test_hedges_g_json(runner):
    r = runner.invoke(cli, [
        "hedges-g",
        "--m1", "6.0", "--sd1", "1.0", "--n1", "30",
        "--m2", "5.0", "--sd2", "1.0", "--n2", "30",
        "--json",
    ])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert data["method"] == "Hedges' g"


# ---------------------------------------------------------------------------
# glass-delta
# ---------------------------------------------------------------------------


def test_glass_delta_basic(runner):
    r = runner.invoke(cli, [
        "glass-delta",
        "--m1", "6.0", "--sd1", "1.0", "--n1", "30",
        "--m2", "5.0", "--sd2-control", "1.0", "--n2", "30",
    ])
    assert r.exit_code == 0
    assert "Glass" in r.output


def test_glass_delta_json(runner):
    r = runner.invoke(cli, [
        "glass-delta",
        "--m1", "6.0", "--sd1", "1.0", "--n1", "30",
        "--m2", "5.0", "--sd2-control", "1.0", "--n2", "30",
        "--json",
    ])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert "Glass" in data["method"]


# ---------------------------------------------------------------------------
# cohens-h
# ---------------------------------------------------------------------------


def test_cohens_h_basic(runner):
    r = runner.invoke(cli, ["cohens-h", "--p1", "0.65", "--p2", "0.50"])
    assert r.exit_code == 0
    assert "Cohen" in r.output


def test_cohens_h_json(runner):
    r = runner.invoke(cli, ["cohens-h", "--p1", "0.65", "--p2", "0.50", "--json"])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert data["method"] == "Cohen's h"


def test_cohens_h_json_value_positive(runner):
    r = runner.invoke(cli, ["cohens-h", "--p1", "0.65", "--p2", "0.50", "--json"])
    data = json.loads(r.output)
    assert data["value"] > 0


# ---------------------------------------------------------------------------
# cohens-f
# ---------------------------------------------------------------------------


def test_cohens_f_basic(runner):
    r = runner.invoke(cli, ["cohens-f", "--ss-effect", "100.0", "--ss-error", "300.0"])
    assert r.exit_code == 0


def test_cohens_f_json(runner):
    r = runner.invoke(cli, [
        "cohens-f", "--ss-effect", "100.0", "--ss-error", "300.0", "--json",
    ])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert data["method"] == "Cohen's f"
    assert data["magnitude"] == "large"


# ---------------------------------------------------------------------------
# cohens-w
# ---------------------------------------------------------------------------


def test_cohens_w_basic(runner):
    r = runner.invoke(cli, ["cohens-w", "--chi2", "18.0", "--n", "200", "--df", "3"])
    assert r.exit_code == 0


def test_cohens_w_json(runner):
    r = runner.invoke(cli, [
        "cohens-w", "--chi2", "18.0", "--n", "200", "--df", "3", "--json",
    ])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert data["method"] == "Cohen's w"


# ---------------------------------------------------------------------------
# cramers-v
# ---------------------------------------------------------------------------


def test_cramers_v_basic(runner):
    r = runner.invoke(cli, [
        "cramers-v", "--chi2", "25.0", "--n", "100", "--rows", "2", "--cols", "2",
    ])
    assert r.exit_code == 0


def test_cramers_v_json(runner):
    r = runner.invoke(cli, [
        "cramers-v", "--chi2", "25.0", "--n", "100", "--rows", "2", "--cols", "2", "--json",
    ])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert data["method"] == "Cramér's V"


def test_cramers_v_json_large(runner):
    r = runner.invoke(cli, [
        "cramers-v", "--chi2", "25.0", "--n", "100", "--rows", "2", "--cols", "2", "--json",
    ])
    data = json.loads(r.output)
    assert data["magnitude"] == "large"


# ---------------------------------------------------------------------------
# eta-squared
# ---------------------------------------------------------------------------


def test_eta_squared_basic(runner):
    r = runner.invoke(cli, ["eta-squared", "--ss-effect", "120.0", "--ss-total", "600.0"])
    assert r.exit_code == 0


def test_eta_squared_json(runner):
    r = runner.invoke(cli, [
        "eta-squared", "--ss-effect", "120.0", "--ss-total", "600.0", "--json",
    ])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert "Eta" in data["method"] or "η²" in data["method"]


def test_eta_squared_json_power_note(runner):
    r = runner.invoke(cli, [
        "eta-squared", "--ss-effect", "120.0", "--ss-total", "600.0", "--json",
    ])
    data = json.loads(r.output)
    assert "power_note" in data


# ---------------------------------------------------------------------------
# omega-squared
# ---------------------------------------------------------------------------


def test_omega_squared_basic(runner):
    r = runner.invoke(cli, [
        "omega-squared",
        "--ss-effect", "120.0", "--ss-error", "480.0",
        "--df-effect", "3", "--n", "60",
    ])
    assert r.exit_code == 0


def test_omega_squared_json(runner):
    r = runner.invoke(cli, [
        "omega-squared",
        "--ss-effect", "120.0", "--ss-error", "480.0",
        "--df-effect", "3", "--n", "60",
        "--json",
    ])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert "Omega" in data["method"] or "ω²" in data["method"]


# ---------------------------------------------------------------------------
# pearson-r
# ---------------------------------------------------------------------------


def test_pearson_r_basic(runner):
    r = runner.invoke(cli, ["pearson-r", "--r", "0.35", "--n", "100"])
    assert r.exit_code == 0


def test_pearson_r_json(runner):
    r = runner.invoke(cli, ["pearson-r", "--r", "0.35", "--n", "100", "--json"])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert data["method"] == "Pearson r"


def test_pearson_r_json_ci_field(runner):
    r = runner.invoke(cli, ["pearson-r", "--r", "0.35", "--n", "100", "--json"])
    data = json.loads(r.output)
    assert "confidence_interval" in data
    assert data["confidence_interval"]["lower"] < data["confidence_interval"]["upper"]


def test_pearson_r_json_magnitude(runner):
    r = runner.invoke(cli, ["pearson-r", "--r", "0.35", "--n", "100", "--json"])
    data = json.loads(r.output)
    assert data["magnitude"] == "medium"


def test_pearson_r_invalid_exits_nonzero(runner):
    r = runner.invoke(cli, ["pearson-r", "--r", "1.0", "--n", "100"])
    assert r.exit_code != 0


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


def test_batch_basic(runner):
    fixture = os.path.join(FIXTURE_DIR, "batch_effects.json")
    r = runner.invoke(cli, ["batch", "--file", fixture])
    assert r.exit_code == 0


def test_batch_json_output(runner):
    fixture = os.path.join(FIXTURE_DIR, "batch_effects.json")
    r = runner.invoke(cli, ["batch", "--file", fixture, "--json"])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert isinstance(data, list)


def test_batch_json_length(runner):
    fixture = os.path.join(FIXTURE_DIR, "batch_effects.json")
    r = runner.invoke(cli, ["batch", "--file", fixture, "--json"])
    data = json.loads(r.output)
    assert len(data) == 4  # fixture has 4 items


def test_batch_json_first_method(runner):
    fixture = os.path.join(FIXTURE_DIR, "batch_effects.json")
    r = runner.invoke(cli, ["batch", "--file", fixture, "--json"])
    data = json.loads(r.output)
    assert data[0]["method"] == "Cohen's d"


def test_batch_with_temp_file(runner, tmp_path):
    batch_data = [
        {
            "method": "cohens_h",
            "params": {"p1": 0.7, "p2": 0.5},
        }
    ]
    f = tmp_path / "test_batch.json"
    f.write_text(json.dumps(batch_data))
    r = runner.invoke(cli, ["batch", "--file", str(f), "--json"])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert data[0]["method"] == "Cohen's h"


def test_batch_unknown_method_continues(tmp_path):
    batch_data = [
        {"method": "unknown_method", "params": {}},
        {"method": "cohens_h", "params": {"p1": 0.6, "p2": 0.4}},
    ]
    f = tmp_path / "mixed.json"
    f.write_text(json.dumps(batch_data))
    r = CliRunner().invoke(cli, ["batch", "--file", str(f)])
    assert r.exit_code == 0
    # Valid item should still be processed and rendered
    assert "Cohen" in r.output
