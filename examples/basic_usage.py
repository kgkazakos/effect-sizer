"""
effect-sizer basic usage examples.

Run: python examples/basic_usage.py
"""

from effect_sizer.calculators import (
    cohens_d,
    cohens_h,
    cramers_v,
    eta_squared,
    hedges_g,
    omega_squared,
    pearson_r,
)


def main() -> None:
    print("=" * 60)
    print("effect-sizer — basic usage examples")
    print("=" * 60)

    # Cohen's d: did the onboarding redesign move task completion time?
    print("\n1. Cohen's d — onboarding task completion time")
    d = cohens_d(m1=142.0, sd1=28.0, n1=45, m2=165.0, sd2=31.0, n2=43)
    print(f"   d = {d.value}  |  {d.magnitude}  |  {d.interpretation}")

    # Hedges' g: prefer for small samples
    print("\n2. Hedges' g — same data, bias-corrected")
    g = hedges_g(m1=142.0, sd1=28.0, n1=45, m2=165.0, sd2=31.0, n2=43)
    print(f"   g = {g.value}  |  {g.magnitude}")

    # Cohen's h: did feature adoption rate change?
    print("\n3. Cohen's h — feature adoption rate (45% vs 62%)")
    h = cohens_h(p1=0.62, p2=0.45)
    print(f"   h = {h.value:.3f}  |  {h.magnitude}")

    # Pearson r: correlation between SUS score and task time
    print("\n4. Pearson r — SUS score vs task time")
    r = pearson_r(r=-0.42, n=88)
    print(f"   r = {r.value}  |  {r.magnitude}  |  r² = {r.value ** 2:.3f}")
    print(f"   95% CI [{r.ci_lower}, {r.ci_upper}]")

    # Cramér's V: association between device type and error category
    print("\n5. Cramér's V — device × error category (3×4 table)")
    v = cramers_v(chi2=18.4, n=240, rows=3, cols=4)
    print(f"   V = {v.value}  |  {v.magnitude}")
    print(f"   {v.interpretation}")

    # η² and ω² for an ANOVA
    print("\n6. η² and ω² — ANOVA on SUS scores across 4 persona segments")
    eta = eta_squared(ss_effect=220.0, ss_total=1100.0)
    omega = omega_squared(ss_effect=220.0, ss_error=880.0, df_effect=3, n=80)
    print(f"   η² = {eta.value}  ({eta.magnitude})")
    print(f"   ω² = {omega.value}  ({omega.magnitude})  ← bias-corrected")

    # JSON output for CI pipelines
    print("\n7. JSON output (for CI / downstream tooling)")
    import json

    print(json.dumps(d.to_dict(), indent=2)[:400], "...")

    print("\n✓ All examples complete.")


if __name__ == "__main__":
    main()
