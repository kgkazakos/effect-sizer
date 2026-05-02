# effect-sizer

**Calculate and contextualise research effect sizes with plain-language rationale.**

Part of the [CPR Orbital series](https://github.com/kgkazakos) — 40 open-source tools for AI-native product research infrastructure.

---

## The problem

Your study results are in. The p-value is < 0.05. Now what?

Significance tells you the finding is probably real. Effect size tells you whether it *matters*. Most product researchers either skip this step or report raw numbers without context. Reviewers, PMs, and leadership need to know: is this a small nudge or a meaningful shift?

`effect-sizer` calculates the right effect size for your design — with benchmarks, CIs, and plain-language interpretation baked in. No R required. No manual lookup of Cohen (1988). Works rule-based from the CLI; add an LLM backend for one-paragraph stakeholder narratives.

---

## Features

- **10 effect size methods** — Cohen's d, Hedges' g, Glass's Δ, Cohen's h, Cohen's f, Cohen's w, Cramér's V, η², ω², Pearson r
- **Confidence intervals** for d, g, Δ (Hedges & Olkin SE), and r (Fisher z)
- **df-adjusted Cramér's V benchmarks** (Lakens, 2013) — not the naïve one-size-fits-all thresholds
- **Bias correction** — ω² over η² for small samples, Hedges' correction factor displayed
- **Rich terminal output** with magnitude colour-coding
- **JSON output** for CI pipelines: `--json`
- **Batch mode** — run multiple calculations from a single JSON file
- **LLM enrichment** (optional) — OpenAI, Anthropic, Gemini via `--llm-backend`; zero API calls without it

---

## Install

```bash
pip install effect-sizer
```

LLM enrichment (OpenAI, Anthropic, Gemini) is included by default. Set the relevant API key as an environment variable and use `--llm-backend` to activate it.

---

## Quick start

```bash
# Cohen's d: did the redesign move task completion time?
effect-sizer cohens-d \
  --m1 142 --sd1 28 --n1 45 \
  --m2 165 --sd2 31 --n2 43

# Proportion shift: feature adoption before vs after
effect-sizer cohens-h --p1 0.62 --p2 0.45

# Correlation between SUS score and session time
effect-sizer pearson-r --r -0.42 --n 88

# Cramér's V: device × error category (3×4 table)
effect-sizer cramers-v --chi2 18.4 --n 240 --rows 3 --cols 4

# Bias-corrected ANOVA variance explained
effect-sizer omega-squared \
  --ss-effect 220 --ss-error 880 --df-effect 3 --n 80

# JSON output for CI or downstream tooling
effect-sizer cohens-d \
  --m1 142 --sd1 28 --n1 45 \
  --m2 165 --sd2 31 --n2 43 \
  --json

# Add a plain-language narrative (requires [llm] extra)
effect-sizer pearson-r --r -0.42 --n 88 \
  --llm-backend anthropic \
  --context "SaaS product: correlation between onboarding SUS score and 30-day retention"
```

---

## Batch mode

Run multiple effect sizes from a single JSON file — useful for synthesis or automated reporting.

```json
[
  {"method": "cohens_d",  "params": {"m1": 5.2, "sd1": 1.1, "n1": 30, "m2": 4.8, "sd2": 1.0, "n2": 28}},
  {"method": "pearson_r", "params": {"r": 0.35, "n": 100}},
  {"method": "cohens_h",  "params": {"p1": 0.65, "p2": 0.50}}
]
```

```bash
effect-sizer batch --file effects.json
effect-sizer batch --file effects.json --json
```

---

## Methods reference

| Command | Method | CI | Benchmarks |
|---|---|---|---|
| `cohens-d` | Cohen's d (pooled SD) | ✓ Hedges & Olkin | 0.2 / 0.5 / 0.8 |
| `hedges-g` | Hedges' g (bias-corrected d) | ✓ Hedges & Olkin | 0.2 / 0.5 / 0.8 |
| `glass-delta` | Glass's Δ (control SD) | ✓ Hedges & Olkin | 0.2 / 0.5 / 0.8 |
| `cohens-h` | Cohen's h (proportions) | — | 0.2 / 0.5 / 0.8 |
| `cohens-f` | Cohen's f (ANOVA) | — | 0.1 / 0.25 / 0.4 |
| `cohens-w` | Cohen's w (chi-square) | — | 0.1 / 0.3 / 0.5 |
| `cramers-v` | Cramér's V (contingency) | — | df-adjusted |
| `eta-squared` | η² (ANOVA) | — | 0.01 / 0.06 / 0.14 |
| `omega-squared` | ω² (bias-corrected η²) | — | 0.01 / 0.06 / 0.14 |
| `pearson-r` | Pearson r (correlation) | ✓ Fisher z | 0.1 / 0.3 / 0.5 |

---

## Python API

```python
from effect_sizer.calculators import cohens_d, pearson_r, cramers_v

# Cohen's d with 95% CI
result = cohens_d(m1=142, sd1=28, n1=45, m2=165, sd2=31, n2=43)
print(result.value)         # -0.7641
print(result.magnitude)     # medium
print(result.interpretation)
print(result.ci_lower, result.ci_upper)

# JSON-serialisable
import json
print(json.dumps(result.to_dict(), indent=2))
```

---

## LLM enrichment (optional)

```python
from effect_sizer.calculators import cohens_d
from effect_sizer.llm import enrich

result = cohens_d(m1=142, sd1=28, n1=45, m2=165, sd2=31, n2=43)
result = enrich(
    result,
    backend="anthropic",      # openai | anthropic | gemini
    context="SaaS onboarding A/B test — 88 participants, within-subjects"
)
print(result.narrative)  # 2–3 sentence stakeholder summary
```

Requires: `pip install effect-sizer`

LLM scaffolding runs on top of fully computed rule-based fields. No API key is needed for any of the calculators.

---

## Development

```bash
git clone https://github.com/kgkazakos/effect-sizer
cd effect-sizer
pip install -e '.[dev]'
pytest
ruff check .
```

---

## References

- Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.).
- Hedges, L. V., & Olkin, I. (1985). *Statistical methods for meta-analysis*.
- Lakens, D. (2013). Calculating and reporting effect sizes to facilitate cumulative science. *Frontiers in Psychology*, 4, 863.

---

## Part of the CPR Ecosystem

`effect-sizer` is Tool 16 of 40 in the **CPR Orbital series** — a portfolio of open-source Python tools for AI-native product research infrastructure, shipping weekly on GitHub.

**Related tools:**
- [`research-rubric`](https://github.com/kgkazakos/research-rubric) — qualitative research quality evaluation
- [`survey-cleaner`](https://github.com/kgkazakos/survey-cleaner) — open-end quality scoring
- [`bias-audit`](https://github.com/kgkazakos/bias-audit) — interviewer bias detection across transcript corpora
- [`uxr-bench`](https://github.com/kgkazakos/uxr-bench) — benchmark dataset for LLM research task evaluation

**CPR core projects:** [CausalTrack](https://github.com/kgkazakos/CausalTrack) · Synthetic User Council · Co-Reasoning Canvas

---

## License

MIT © Kostas Kazakos
