"""
Generate the three tables for Paper 3, straight from the parquet files.

  Table 1  cohort characteristics and documentation subgroups
  Table 2  content density by documentation status, with effect sizes
  Table 3  conformal coverage and abstention (marginal and conditional),
           averaged over 100 stratified calibration splits

Self-contained: the clinical model is trained inline for Table 3. Each table is
printed and also written to a CSV so it can be pasted into the manuscript.

Input  : faers_temporal_{train,test}.parquet  (set DATA_DIR below)
Output : table1_cohort.csv, table2_content.csv, table3_conformal.csv
Runtime: ~3-5 min (one XGBoost fit + 100 conformal splits)
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

DATA_DIR = r"C:\Users\shaki\Downloads\faers_data"   # <-- set to your data path
OUT_DIR = "."

ALPHA = 0.10        # conformal target coverage = 90%
N_SPLITS = 100      # calibration splits to average Table 3 over
N_BOOT = 2000       # bootstrap resamples for the Table 2 difference CIs
SEED = 42
rng = np.random.default_rng(SEED)

NUMERIC = ["n_drugs", "n_suspect", "n_concomitant", "n_reactions", "age_years", "weight_kg"]
# Reporter-qualification codes -> readable stream labels (for Table 1).
REPORTER_LABELS = {"1": "Physician", "2": "Pharmacist",
                   "3": "Other health professional", "5": "Consumer / self-report"}


def load():
    """Load both quarters; add the two 'distinct term' counts to the test set."""
    tr = pd.read_parquet(os.path.join(DATA_DIR, "faers_temporal_train.parquet"))
    te = pd.read_parquet(os.path.join(DATA_DIR, "faers_temporal_test.parquet")).copy()
    te["pt_active"] = te[[c for c in te.columns if c.startswith("pt::")]].sum(axis=1)
    te["drug_active"] = te[[c for c in te.columns if c.startswith("drug::")]].sum(axis=1)
    return tr, te


# ---------------------------------------------------------------- Table 1
def table1(te):
    """One row per subgroup: size, share, % serious, and % missing per field."""
    total = len(te)
    age_missing = te["age_years"].isna()
    sex_missing = te["sex"].eq("unk")
    reporter = te["qualification"].map(REPORTER_LABELS).fillna("Other or unknown")

    def summarise(label, mask):
        sub = te[mask]
        return dict(
            Subgroup=label,
            n=int(mask.sum()),
            pct_of_total=round(100 * mask.sum() / total, 1),
            pct_serious=round(100 * sub["label_serious"].mean(), 1),
            pct_age_missing=round(100 * sub["age_years"].isna().mean(), 1),
            pct_sex_missing=round(100 * sub["sex"].eq("unk").mean(), 1),
            pct_weight_missing=round(100 * sub["weight_kg"].isna().mean(), 1),
        )

    rows = [
        summarise("All reports", pd.Series(True, index=te.index)),
        summarise("Age recorded", ~age_missing),
        summarise("Age missing", age_missing),
        summarise("Sex recorded", ~sex_missing),
        summarise("Sex missing", sex_missing),
        summarise("Complete demographics (age and sex)", ~(age_missing | sex_missing)),
        summarise("Any demographic missing", age_missing | sex_missing),
    ]
    for stream in ["Physician", "Pharmacist", "Other health professional",
                   "Consumer / self-report", "Other or unknown"]:
        rows.append(summarise(stream, reporter.eq(stream)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Table 2
def cohens_d(missing_vals, recorded_vals):
    """Standardized mean difference, pooled SD."""
    n_m, n_r = len(missing_vals), len(recorded_vals)
    pooled_sd = np.sqrt(
        ((n_m - 1) * missing_vals.var(ddof=1) + (n_r - 1) * recorded_vals.var(ddof=1))
        / (n_m + n_r - 2))
    return (missing_vals.mean() - recorded_vals.mean()) / pooled_sd if pooled_sd > 0 else np.nan


def rank_biserial(missing_vals, recorded_vals):
    """Rank-biserial correlation from the Mann-Whitney U statistic."""
    u, _ = mannwhitneyu(missing_vals, recorded_vals)
    return 2 * u / (len(missing_vals) * len(recorded_vals)) - 1


def difference_ci(missing_vals, recorded_vals, n_boot=N_BOOT):
    """Bootstrap 95% CI for (mean missing) - (mean recorded), each group resampled
    on its own so subgroup sizes are preserved every draw.
    """
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = (rng.choice(missing_vals, len(missing_vals), True).mean()
                    - rng.choice(recorded_vals, len(recorded_vals), True).mean())
    return np.percentile(diffs, [2.5, 97.5])


def table2(te):
    """Content density missing-vs-recorded, for age and sex, with effect sizes."""
    metrics = [("n_reactions", "Reactions"),
               ("pt_active", "Distinct in-vocabulary reaction terms"),
               ("n_drugs", "Drugs"),
               ("drug_active", "Distinct in-vocabulary ingredients"),
               ("n_suspect", "Suspect drugs")]
    rows = []
    for field, is_missing in [("Age", te["age_years"].isna()), ("Sex", te["sex"].eq("unk"))]:
        for col, label in metrics:
            missing_vals = te.loc[is_missing, col].values
            recorded_vals = te.loc[~is_missing, col].values
            lo, hi = difference_ci(missing_vals, recorded_vals)
            _, p = mannwhitneyu(missing_vals, recorded_vals)
            rows.append(dict(
                Missing_field=field, Metric=label,
                Missing_mean=round(missing_vals.mean(), 2),
                Recorded_mean=round(recorded_vals.mean(), 2),
                Difference=round(missing_vals.mean() - recorded_vals.mean(), 2),
                CI=f"({lo:+.2f}, {hi:+.2f})",
                Cohens_d=round(cohens_d(missing_vals, recorded_vals), 2),
                rank_biserial_r=round(rank_biserial(missing_vals, recorded_vals), 2),
                p=("<1e-300" if p == 0 else f"{p:.1e}")))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Table 3
def clinical_probs(tr, te):
    """Fit the clinical model, return test-set seriousness probabilities."""
    binary = [c for c in tr.columns if "::" in c]
    features = NUMERIC + ["sex"] + binary
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["sex"]),
        ("bin", "passthrough", binary)])
    model = Pipeline([("prep", prep), ("xgb", XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
        tree_method="hist", n_jobs=2, random_state=SEED))])
    model.fit(tr[features], tr["label_serious"])
    return model.predict_proba(te[features])[:, 1]


def conformal_threshold(scores, alpha):
    """(1 - alpha) conformal quantile with the finite-sample correction."""
    n = len(scores)
    return np.sort(scores)[min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1]


def nonconformity(p, y):
    """1 - p(true class)."""
    return np.where(y == 1, 1 - p, p)


def table3(te, p, y):
    """Coverage and abstention by subgroup, marginal and conditional, over N_SPLITS.

    For each random stratified calibration split we compute, per subgroup, coverage
    and abstention under a single global threshold (marginal) and under a
    per-subgroup threshold (conditional/Mondrian). The table reports the mean and
    empirical 95% range across splits.
    """
    age_missing = te["age_years"].isna().values
    sex_missing = te["sex"].eq("unk").values
    groups = [("All reports", np.ones(len(y), bool)),
              ("Age recorded", ~age_missing), ("Age missing", age_missing),
              ("Sex recorded", ~sex_missing), ("Sex missing", sex_missing)]

    marginal = {g: {"cov": [], "ab": [], "sc": []} for g, _ in groups}
    conditional = {g: {"cov": [], "ab": []} for g, _ in groups}
    idx = np.arange(len(y))

    for seed in range(N_SPLITS):
        split_rng = np.random.default_rng(seed)
        cal = np.zeros(len(y), bool)
        for cls in (0, 1):
            class_idx = idx[y == cls]
            cal[split_rng.choice(class_idx, len(class_idx) // 2, replace=False)] = True
        ev = ~cal
        q_global = conformal_threshold(nonconformity(p[cal], y[cal]), ALPHA)

        for label, group in groups:
            e = ev & group
            # Marginal: one global threshold.
            in_s = (1 - p[e]) <= q_global
            in_n = p[e] <= q_global
            size = in_s.astype(int) + in_n.astype(int)
            marginal[label]["cov"].append(np.where(y[e] == 1, in_s, in_n).mean())
            marginal[label]["ab"].append((size == 2).mean())
            marginal[label]["sc"].append((size == 1).mean())
            # Conditional: threshold calibrated inside this subgroup.
            q_group = conformal_threshold(nonconformity(p[cal & group], y[cal & group]), ALPHA)
            in_sc = (1 - p[e]) <= q_group
            in_nc = p[e] <= q_group
            size_c = in_sc.astype(int) + in_nc.astype(int)
            conditional[label]["cov"].append(np.where(y[e] == 1, in_sc, in_nc).mean())
            conditional[label]["ab"].append((size_c == 2).mean())

    def fmt_frac(values):
        v = np.array(values)
        return f"{v.mean():.3f} ({np.percentile(v, 2.5):.3f}-{np.percentile(v, 97.5):.3f})"

    def fmt_pct(values):
        v = np.array(values) * 100
        return f"{v.mean():.1f} ({np.percentile(v, 2.5):.1f}-{np.percentile(v, 97.5):.1f})"

    rows = []
    for label, group in groups:
        rows.append(dict(
            Subgroup=label, n_evaluation=int(group.sum() // 2),
            Coverage_marginal=fmt_frac(marginal[label]["cov"]),
            Abstained_pct_marginal=fmt_pct(marginal[label]["ab"]),
            Scoreable_pct_marginal=fmt_pct(marginal[label]["sc"]),
            Coverage_conditional=fmt_frac(conditional[label]["cov"]),
            Abstained_pct_conditional=fmt_pct(conditional[label]["ab"])))
    table = pd.DataFrame(rows)

    # The two headline abstention ratios (missing / recorded), printed separately.
    ratio_age = (np.array(marginal["Age missing"]["ab"])
                 / np.array(marginal["Age recorded"]["ab"]))
    ratio_sex = (np.array(marginal["Sex missing"]["ab"])
                 / np.array(marginal["Sex recorded"]["ab"]))
    print(f"\nAbstention ratio age (missing/recorded): {ratio_age.mean():.2f} "
          f"({np.percentile(ratio_age, 2.5):.2f}-{np.percentile(ratio_age, 97.5):.2f})")
    print(f"Abstention ratio sex (missing/recorded): {ratio_sex.mean():.2f} "
          f"({np.percentile(ratio_sex, 2.5):.2f}-{np.percentile(ratio_sex, 97.5):.2f})")
    return table


def main():
    tr, te = load()
    y = te["label_serious"].values
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 25)

    t1 = table1(te)
    print("=== TABLE 1 ===\n", t1.to_string(index=False))
    t1.to_csv(os.path.join(OUT_DIR, "table1_cohort.csv"), index=False)

    t2 = table2(te)
    print("\n=== TABLE 2 ===\n", t2.to_string(index=False))
    t2.to_csv(os.path.join(OUT_DIR, "table2_content.csv"), index=False)

    p = clinical_probs(tr, te)
    t3 = table3(te, p, y)
    print("\n=== TABLE 3 ===\n", t3.to_string(index=False))
    t3.to_csv(os.path.join(OUT_DIR, "table3_conformal.csv"), index=False)
    print("\nCSVs written to", OUT_DIR)


if __name__ == "__main__":
    main()
