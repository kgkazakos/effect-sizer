"""
effect-sizer CLI.

Commands:
  cohens-d      Cohen's d (two independent means)
  hedges-g      Hedges' g (bias-corrected d)
  glass-delta   Glass's Δ (control-SD denominator)
  cohens-h      Cohen's h (two proportions)
  cohens-f      Cohen's f (ANOVA)
  cohens-w      Cohen's w (chi-square)
  cramers-v     Cramér's V (contingency tables)
  eta-squared   Eta-squared η² (ANOVA variance)
  omega-squared Omega-squared ω² (bias-corrected η²)
  pearson-r     Pearson r (correlation)
  batch         Run multiple calculations from a JSON file
"""

from __future__ import annotations

import json
import sys
from typing import List

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import calculators
from .models import EffectSizeResult

console = Console()

MAGNITUDE_COLORS = {
    "negligible": "dim",
    "small": "yellow",
    "medium": "blue",
    "large": "green",
}

CALCULATORS = {
    "cohens_d": lambda p: calculators.cohens_d(**p),
    "hedges_g": lambda p: calculators.hedges_g(**p),
    "glass_delta": lambda p: calculators.glass_delta(**p),
    "cohens_h": lambda p: calculators.cohens_h(**p),
    "cohens_f": lambda p: calculators.cohens_f(**p),
    "cohens_w": lambda p: calculators.cohens_w(**p),
    "cramers_v": lambda p: calculators.cramers_v(**p),
    "eta_squared": lambda p: calculators.eta_squared(**p),
    "omega_squared": lambda p: calculators.omega_squared(**p),
    "pearson_r": lambda p: calculators.pearson_r(**p),
}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _display(result: EffectSizeResult) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Field", style="dim", no_wrap=True)
    table.add_column("Value")

    table.add_row("Method", result.method)
    table.add_row("Value", str(result.value))

    mag_text = Text(result.magnitude, style=MAGNITUDE_COLORS.get(result.magnitude, "white"))
    table.add_row("Magnitude", mag_text)

    if result.ci_lower is not None:
        table.add_row("CI", f"[{result.ci_lower}, {result.ci_upper}]")

    table.add_row("Interpretation", result.interpretation)

    b = result.benchmarks
    if "small" in b:
        bench_str = (
            f"small ≥ {b['small']}  "
            f"medium ≥ {b['medium']}  "
            f"large ≥ {b['large']}"
        )
        table.add_row("Benchmarks", bench_str)

    if result.power_note:
        table.add_row("Note", Text(result.power_note, style="italic dim"))

    if result.narrative:
        table.add_row("Narrative", result.narrative)

    console.print(Panel(table, title="[bold]effect-sizer[/bold]", border_style="blue"))


def _emit(result: EffectSizeResult, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        _display(result)


def _enrich_maybe(result: EffectSizeResult, llm_backend: str, context: str) -> EffectSizeResult:
    if llm_backend:
        from .llm import enrich  # noqa: PLC0415
        result = enrich(result, backend=llm_backend, context=context)
    return result


def _run(fn, kwargs: dict, as_json: bool, llm_backend: str, context: str) -> None:
    try:
        result = fn(**kwargs)
        result = _enrich_maybe(result, llm_backend, context)
        _emit(result, as_json)
    except (ValueError, TypeError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="effect-sizer")
def cli() -> None:
    """effect-sizer: calculate and contextualise research effect sizes."""


# Common options reused across subcommands
_CONFIDENCE = click.option(
    "--confidence", default=0.95, show_default=True,
    help="Confidence level for CI (e.g. 0.90, 0.95).",
)
_JSON = click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
_LLM = click.option(
    "--llm-backend", type=click.Choice(["openai", "anthropic", "gemini"]),
    default=None, show_default=False,
    help="Enable LLM narrative (openai/anthropic/gemini — set API key env var).",
)
_CONTEXT = click.option("--context", default="", help="Research context for LLM narrative.")


# ---------------------------------------------------------------------------
# cohens-d
# ---------------------------------------------------------------------------


@cli.command("cohens-d")
@click.option("--m1", required=True, type=float, help="Mean of group 1.")
@click.option("--sd1", required=True, type=float, help="SD of group 1.")
@click.option("--n1", required=True, type=int, help="N of group 1.")
@click.option("--m2", required=True, type=float, help="Mean of group 2.")
@click.option("--sd2", required=True, type=float, help="SD of group 2.")
@click.option("--n2", required=True, type=int, help="N of group 2.")
@_CONFIDENCE
@_JSON
@_LLM
@_CONTEXT
def cohens_d_cmd(
    m1: float, sd1: float, n1: int,
    m2: float, sd2: float, n2: int,
    confidence: float, as_json: bool, llm_backend: str, context: str,
) -> None:
    """Cohen's d for two independent groups."""
    _run(
        calculators.cohens_d,
        {"m1": m1, "sd1": sd1, "n1": n1, "m2": m2, "sd2": sd2, "n2": n2, "confidence": confidence},
        as_json, llm_backend, context,
    )


# ---------------------------------------------------------------------------
# hedges-g
# ---------------------------------------------------------------------------


@cli.command("hedges-g")
@click.option("--m1", required=True, type=float)
@click.option("--sd1", required=True, type=float)
@click.option("--n1", required=True, type=int)
@click.option("--m2", required=True, type=float)
@click.option("--sd2", required=True, type=float)
@click.option("--n2", required=True, type=int)
@_CONFIDENCE
@_JSON
@_LLM
@_CONTEXT
def hedges_g_cmd(
    m1: float, sd1: float, n1: int,
    m2: float, sd2: float, n2: int,
    confidence: float, as_json: bool, llm_backend: str, context: str,
) -> None:
    """Hedges' g — bias-corrected Cohen's d."""
    _run(
        calculators.hedges_g,
        {"m1": m1, "sd1": sd1, "n1": n1, "m2": m2, "sd2": sd2, "n2": n2, "confidence": confidence},
        as_json, llm_backend, context,
    )


# ---------------------------------------------------------------------------
# glass-delta
# ---------------------------------------------------------------------------


@cli.command("glass-delta")
@click.option("--m1", required=True, type=float)
@click.option("--sd1", required=True, type=float)
@click.option("--n1", required=True, type=int)
@click.option("--m2", required=True, type=float)
@click.option("--sd2-control", required=True, type=float, help="Control group SD.")
@click.option("--n2", required=True, type=int)
@_CONFIDENCE
@_JSON
@_LLM
@_CONTEXT
def glass_delta_cmd(
    m1: float, sd1: float, n1: int,
    m2: float, sd2_control: float, n2: int,
    confidence: float, as_json: bool, llm_backend: str, context: str,
) -> None:
    """Glass's Δ — uses control group SD as denominator."""
    _run(
        calculators.glass_delta,
        {
            "m1": m1, "sd1": sd1, "n1": n1, "m2": m2,
            "sd2_control": sd2_control, "n2": n2, "confidence": confidence,
        },
        as_json, llm_backend, context,
    )


# ---------------------------------------------------------------------------
# cohens-h
# ---------------------------------------------------------------------------


@cli.command("cohens-h")
@click.option("--p1", required=True, type=float, help="Proportion 1 (0–1).")
@click.option("--p2", required=True, type=float, help="Proportion 2 (0–1).")
@_JSON
@_LLM
@_CONTEXT
def cohens_h_cmd(
    p1: float, p2: float, as_json: bool, llm_backend: str, context: str,
) -> None:
    """Cohen's h for two proportions."""
    _run(calculators.cohens_h, {"p1": p1, "p2": p2}, as_json, llm_backend, context)


# ---------------------------------------------------------------------------
# cohens-f
# ---------------------------------------------------------------------------


@cli.command("cohens-f")
@click.option("--ss-effect", required=True, type=float, help="Sum of squares: effect.")
@click.option("--ss-error", required=True, type=float, help="Sum of squares: error.")
@_JSON
@_LLM
@_CONTEXT
def cohens_f_cmd(
    ss_effect: float, ss_error: float, as_json: bool, llm_backend: str, context: str,
) -> None:
    """Cohen's f from ANOVA SS components."""
    _run(
        calculators.cohens_f, {"ss_effect": ss_effect, "ss_error": ss_error},
        as_json, llm_backend, context,
    )


# ---------------------------------------------------------------------------
# cohens-w
# ---------------------------------------------------------------------------


@cli.command("cohens-w")
@click.option("--chi2", required=True, type=float, help="Chi-square statistic.")
@click.option("--n", required=True, type=int, help="Total sample size.")
@click.option("--df", required=True, type=int, help="Degrees of freedom.")
@_JSON
@_LLM
@_CONTEXT
def cohens_w_cmd(
    chi2: float, n: int, df: int, as_json: bool, llm_backend: str, context: str,
) -> None:
    """Cohen's w for chi-square tests."""
    _run(
        calculators.cohens_w, {"chi2": chi2, "n": n, "df": df},
        as_json, llm_backend, context,
    )


# ---------------------------------------------------------------------------
# cramers-v
# ---------------------------------------------------------------------------


@cli.command("cramers-v")
@click.option("--chi2", required=True, type=float)
@click.option("--n", required=True, type=int)
@click.option("--rows", required=True, type=int, help="Number of rows in contingency table.")
@click.option("--cols", required=True, type=int, help="Number of columns.")
@_JSON
@_LLM
@_CONTEXT
def cramers_v_cmd(
    chi2: float, n: int, rows: int, cols: int,
    as_json: bool, llm_backend: str, context: str,
) -> None:
    """Cramér's V for contingency tables (df-adjusted benchmarks)."""
    _run(
        calculators.cramers_v, {"chi2": chi2, "n": n, "rows": rows, "cols": cols},
        as_json, llm_backend, context,
    )


# ---------------------------------------------------------------------------
# eta-squared
# ---------------------------------------------------------------------------


@cli.command("eta-squared")
@click.option("--ss-effect", required=True, type=float)
@click.option("--ss-total", required=True, type=float)
@_JSON
@_LLM
@_CONTEXT
def eta_squared_cmd(
    ss_effect: float, ss_total: float, as_json: bool, llm_backend: str, context: str,
) -> None:
    """Eta-squared (η²) — proportion of total variance explained."""
    _run(
        calculators.eta_squared, {"ss_effect": ss_effect, "ss_total": ss_total},
        as_json, llm_backend, context,
    )


# ---------------------------------------------------------------------------
# omega-squared
# ---------------------------------------------------------------------------


@cli.command("omega-squared")
@click.option("--ss-effect", required=True, type=float)
@click.option("--ss-error", required=True, type=float)
@click.option("--df-effect", required=True, type=int, help="Effect degrees of freedom.")
@click.option("--n", required=True, type=int, help="Total sample size.")
@_JSON
@_LLM
@_CONTEXT
def omega_squared_cmd(
    ss_effect: float, ss_error: float, df_effect: int, n: int,
    as_json: bool, llm_backend: str, context: str,
) -> None:
    """Omega-squared (ω²) — bias-corrected η² for ANOVA."""
    _run(
        calculators.omega_squared,
        {"ss_effect": ss_effect, "ss_error": ss_error, "df_effect": df_effect, "n": n},
        as_json, llm_backend, context,
    )


# ---------------------------------------------------------------------------
# pearson-r
# ---------------------------------------------------------------------------


@cli.command("pearson-r")
@click.option("--r", required=True, type=float, help="Pearson r value (-1 < r < 1).")
@click.option("--n", required=True, type=int, help="Sample size (N ≥ 4).")
@_CONFIDENCE
@_JSON
@_LLM
@_CONTEXT
def pearson_r_cmd(
    r: float, n: int, confidence: float,
    as_json: bool, llm_backend: str, context: str,
) -> None:
    """Pearson r as an effect size with Fisher-z CI."""
    _run(
        calculators.pearson_r, {"r": r, "n": n, "confidence": confidence},
        as_json, llm_backend, context,
    )


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


@cli.command("batch")
@click.option(
    "--file", "input_file", required=True, type=click.Path(exists=True),
    help="JSON file with list of {method, params} objects.",
)
@_JSON
def batch_cmd(input_file: str, as_json: bool) -> None:
    """Run multiple effect size calculations from a JSON file.

    \b
    File format:
      [
        {"method": "cohens_d", "params": {"m1": 5.2, "sd1": 1.1, "n1": 30, ...}},
        {"method": "pearson_r", "params": {"r": 0.35, "n": 100}}
      ]
    """
    with open(input_file) as fh:
        items = json.load(fh)

    results: List[EffectSizeResult] = []
    errors: List[str] = []

    for i, item in enumerate(items):
        method = item.get("method", "")
        params = item.get("params", {})
        if method not in CALCULATORS:
            errors.append(f"Item {i}: unknown method '{method}'.")
            continue
        try:
            result = CALCULATORS[method](params)
            results.append(result)
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"Item {i} ({method}): {exc}")

    if as_json:
        click.echo(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        for result in results:
            _display(result)

    if errors:
        for err in errors:
            click.echo(f"Warning: {err}", err=True)
