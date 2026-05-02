"""
LLM enrichment for effect-sizer.

Reference-only — not tested in CI.
Requires: pip install 'effect-sizer[llm]'

Backends:
  openai    → openai/gpt-4o-mini
  anthropic → anthropic/claude-haiku-4-5-20251001
  gemini    → gemini/gemini-2.0-flash

Usage:
    from effect_sizer.llm import enrich
    result = enrich(result, backend="anthropic", context="SaaS onboarding A/B test")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import EffectSizeResult

_MODELS: dict = {
    "openai": "openai/gpt-4o-mini",
    "anthropic": "anthropic/claude-haiku-4-5-20251001",
    "gemini": "gemini/gemini-2.0-flash",
}


def enrich(result: "EffectSizeResult", backend: str, context: str = "") -> "EffectSizeResult":
    """Add a plain-language LLM narrative to an EffectSizeResult.

    Args:
        result: A computed EffectSizeResult (rule-based fields already populated).
        backend: One of "openai", "anthropic", "gemini".
        context: Optional free-text research context (e.g. study description).

    Returns:
        The same result with the ``narrative`` field populated.
    """
    try:
        import dspy  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "dspy-ai is required for LLM enrichment. "
            "Install with: pip install effect-sizer"
        ) from exc

    if backend not in _MODELS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(_MODELS)}")

    lm = dspy.LM(_MODELS[backend])
    dspy.configure(lm=lm)

    class _Sig(dspy.Signature):  # type: ignore[misc]
        """Generate a plain-language narrative for a research effect size result."""

        method: str = dspy.InputField(desc="Effect size method name (e.g. Cohen's d)")
        value: str = dspy.InputField(desc="Numeric effect size value")
        magnitude: str = dspy.InputField(desc="negligible / small / medium / large")
        interpretation: str = dspy.InputField(desc="Rule-based interpretation string")
        context: str = dspy.InputField(desc="Optional research context")
        narrative: str = dspy.OutputField(
            desc=(
                "2–3 sentence plain-language summary suitable for a stakeholder brief. "
                "Avoid jargon. Ground the magnitude in practical terms the audience understands."
            )
        )

    class _Mod(dspy.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            self.narrator = dspy.ChainOfThought(_Sig)

        def forward(self, **kwargs: str) -> dspy.Prediction:  # type: ignore[return]
            return self.narrator(**kwargs)

    pred = _Mod().forward(
        method=result.method,
        value=str(result.value),
        magnitude=result.magnitude,
        interpretation=result.interpretation,
        context=context,
    )
    result.narrative = pred.narrative
    return result
