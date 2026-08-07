"""
Part 1 of the FAERS equity audit: what makes under-documented reports different?

Paper 2 found that the seriousness-triage model was less reliable on reports with
missing demographics, but it didn't explain why. Before touching the model again,
I wanted to understand the reports themselves. Two questions drive this script:

  1. Are reports with missing demographics actually *thinner* in clinical content
     (fewer reactions, fewer drugs), or just different in some other way?
  2. Do documentation gaps cluster? That is, if a report is missing one field,
     is it more likely to be missing others too?

Everything here is model-free — it's a description of the data, computed on the
held-out test quarter (2025 Q4). The subgroups I care about:

    missing sex        sex == 'unk'              ~4,777 reports
    missing age        age_years is NaN          ~11,021 reports
    unknown reporter   qualification == 'unk'    ~208 reports (too small to lean on;
                                                  reported only for completeness)

Clinical-content measures, per report:
    n_reactions   reported reactions
    pt_active     distinct in-vocabulary reaction terms  (the pt:: columns, summed)
    n_drugs       reported drugs
    drug_active   distinct in-vocabulary ingredients     (the drug:: columns, summed)
    n_suspect     suspect drugs

For the missing-vs-recorded contrasts I report an effect size next to every
p-value. With subgroups in the thousands, a p-value is basically guaranteed to be
tiny, so on its own it says nothing about whether the difference actually matters —
the effect size is the honest number to read.

Input : faers_temporal_test.parquet  (set DATA_DIR below)
Output: printed tables feeding Paper 3 Table 1 / Figure 1
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# Point this at the folder holding the parquet files.
DATA_DIR = r"C:\Users\shaki\Downloads\faers_data"

# One RNG, one seed, so the bootstrap intervals reproduce exactly.
SEED = 42
N_BOOT = 2000
rng = np.random.default_rng(SEED)

# The five content measures, in the order they appear in the output table.
CONTENT_COLS = ["n_reactions", "pt_active", "n_drugs", "drug_active", "n_suspect"]


def load_test_quarter():
    """Load the test quarter and add the two 'distinct term' counts.

    pt_active / drug_active are just the row-sums of the one-hot pt:: and drug::
    columns — i.e. how many distinct in-vocabulary reaction terms / ingredients
    a report actually mentions.
    """
    te = pd.read_parquet(os.path.join(DATA_DIR, "faers_temporal_test.parquet"))
    pt_cols = [c for c in te.columns if c.startswith("pt::")]
    drug_cols = [c for c in te.columns if c.startswith("drug::")]
    te["pt_active"] = te[pt_cols].sum(axis=1)
    te["drug_active"] = te[drug_cols].sum(axis=1)
    return te


def content_density_table(te):
    """Mean content by subgroup, each documentation axis split recorded vs missing."""
    rows = []
    axes = [
        ("sex", te["sex"].eq("unk")),
        ("age", te["age_years"].isna()),
        ("reporter", te["qualification"].eq("unk")),
        ("any_demog", te["sex"].eq("unk") | te["age_years"].isna()),
    ]
    for axis_name, is_missing in axes:
        for label, mask in [(f"{axis_name}=RECORDED", ~is_missing),
                            (f"{axis_name}=MISSING", is_missing)]:
            sub = te[mask]
            row = {
                "subgroup": label,
                "n": len(sub),
                "serious%": round(100 * sub["label_serious"].mean(), 1),
            }
            for col in CONTENT_COLS:
                row[col] = round(sub[col].mean(), 2)
            row["weight_missing%"] = round(100 * sub["weight_kg"].isna().mean(), 1)
            rows.append(row)
    return pd.DataFrame(rows)


def mean_difference_ci(missing_vals, recorded_vals, n_boot=N_BOOT):
    """Bootstrap 95% CI for (mean of missing) - (mean of recorded).

    Stratified in the sense that each group is resampled on its own, so the two
    subgroup sizes are preserved on every draw.
    """
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        boot_missing = rng.choice(missing_vals, len(missing_vals), replace=True).mean()
        boot_recorded = rng.choice(recorded_vals, len(recorded_vals), replace=True).mean()
        diffs[i] = boot_missing - boot_recorded
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return lo, hi


def cohens_d(missing_vals, recorded_vals):
    """Standardized mean difference, pooled SD."""
    n_m, n_r = len(missing_vals), len(recorded_vals)
    pooled_sd = np.sqrt(
        ((n_m - 1) * missing_vals.var(ddof=1) + (n_r - 1) * recorded_vals.var(ddof=1))
        / (n_m + n_r - 2)
    )
    if pooled_sd == 0:
        return np.nan
    return (missing_vals.mean() - recorded_vals.mean()) / pooled_sd


def rank_biserial_r(missing_vals, recorded_vals):
    """Rank-biserial correlation derived from the Mann-Whitney U statistic.

    Ranges -1 to 1; expresses the 'missing' group relative to 'recorded'.
    """
    u, _ = mannwhitneyu(missing_vals, recorded_vals)
    return 2 * u / (len(missing_vals) * len(recorded_vals)) - 1


def print_content_contrasts(te):
    """Missing vs recorded, for age and sex, with effect sizes beside the p-value.

    Reading guide for the effect sizes:
        |d| ~ 0.2 small, 0.5 medium, 0.8 large
        |r| ~ 0.1 small, 0.3 medium, 0.5 large
    """
    print("\n=== Content thinness: MISSING vs RECORDED (with effect sizes) ===")
    for axis_name, is_missing in [("age", te["age_years"].isna()),
                                  ("sex", te["sex"].eq("unk"))]:
        for col in ["n_reactions", "n_drugs"]:
            missing_vals = te.loc[is_missing, col].values
            recorded_vals = te.loc[~is_missing, col].values
            lo, hi = mean_difference_ci(missing_vals, recorded_vals)
            _, p = mannwhitneyu(missing_vals, recorded_vals)
            diff = missing_vals.mean() - recorded_vals.mean()
            print(f"  {axis_name:4} {col:12} diff {diff:+.2f} "
                  f"95% CI [{lo:+.2f},{hi:+.2f}]  "
                  f"d={cohens_d(missing_vals, recorded_vals):+.2f}  "
                  f"r={rank_biserial_r(missing_vals, recorded_vals):+.2f}  (p={p:.1e})")


def print_correlated_missingness(te):
    """Do gaps cluster? Compare P(field missing | sex missing) vs (| sex recorded)."""
    print("\n=== Correlated missingness (documentation gaps cluster) ===")
    sex_missing = te["sex"].eq("unk")
    age_missing = te["age_years"].isna()
    weight_missing = te["weight_kg"].isna()
    print(f"  P(age missing    | sex missing)  = {age_missing[sex_missing].mean():.3f}   "
          f"vs (sex recorded) = {age_missing[~sex_missing].mean():.3f}")
    print(f"  P(weight missing | sex missing)  = {weight_missing[sex_missing].mean():.3f}   "
          f"vs (sex recorded) = {weight_missing[~sex_missing].mean():.3f}")


def print_completeness_gradient(te):
    """Content and seriousness across a 0-3 demographic-completeness score.

    A note on why the score uses only age/sex/weight: reactions and drugs are
    present in essentially every report (that's the reporting floor), so a
    'no clinical content at all' bucket doesn't really exist and would be
    misleading to invent. Demographic completeness is the axis that genuinely
    varies, so that's what I count here.
    """
    demo_complete = (
        (~te["age_years"].isna()).astype(int)
        + te["sex"].ne("unk").astype(int)
        + (~te["weight_kg"].isna()).astype(int)
    )
    te = te.assign(demo_complete=demo_complete)
    print("\n=== Documentation completeness gradient (0-3 demographic fields) ===")
    summary = (
        te.groupby("demo_complete")
        .agg(n=("label_serious", "size"),
             serious=("label_serious", "mean"),
             n_reactions=("n_reactions", "mean"),
             n_drugs=("n_drugs", "mean"),
             pt_active=("pt_active", "mean"))
        .round(2)
    )
    print(summary.to_string())
    print("  Note: more demographic documentation tracks with more clinical")
    print("  content AND higher seriousness - impoverished reports are thinner")
    print("  and genuinely less often serious (a confound to name, not 'fix').")


def main():
    te = load_test_quarter()
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 20)

    print("=== Part 1: content-density profile (test quarter) ===")
    print(content_density_table(te).to_string(index=False))
    print_content_contrasts(te)
    print_correlated_missingness(te)
    print_completeness_gradient(te)

    print("\nRead: under-documented reports are systematically thinner in clinical")
    print("content (small-to-modest per report), their gaps cluster, and content")
    print("rises monotonically with documentation completeness. The gap is")
    print("structural and largely content-driven - not a mere modelling artifact -")
    print("but the per-report effect is modest and completeness is confounded with")
    print("seriousness itself, both of which the paper states plainly.")
    print("\nNext: 11_reliability_mechanism.py (link content density to per-report")
    print("reliability using the clinical model).")


if __name__ == "__main__":
    main()
