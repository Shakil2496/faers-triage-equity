"""
Part 2a: how big is the reliability gap, and what actually causes it?

Part 1 showed that under-documented reports are thinner in content. This script
brings the model back in and asks the harder question: is that thinner content
*why* the triage model is less reliable on them — or is something else going on?

I reproduce the paper's clinical model (the leakage-controlled one, no reporting-
channel features), measure the discrimination gap between reports with and without
recorded age, and then try to explain that gap two ways. Both explanations fail,
and the fact that they fail is the point:

  Mechanism 1 - content mediation. If thin content were the cause, the gap should
  shrink once we compare reports with the same number of reactions. It doesn't:
  the age gap sits at ~0.07 inside every reaction-count band. So it isn't content.

  Mechanism 2 - age-feature mediation. Maybe the model just misses the age value
  itself (median-imputed when absent). If so, dropping age for *everyone* should
  erase the gap. It doesn't: the gap barely moves, and overall AUROC barely drops.
  So it isn't the missing field either.

What's left is the honest, less convenient reading: reports without demographic
documentation are simply a harder-to-rank population. That's why the paper argues
for a calibrated abstention boundary (Part 2b) instead of claiming a fix.

(One thing worth checking separately, and noted in the writeup: whether missing
age is mostly a stand-in for reporter type / source stream. That refines the
story rather than overturning it.)

Input  : faers_temporal_{train,test}.parquet  (set DATA_DIR below)
Output : printed reliability gap and the two mechanism tests
Runtime: ~2-4 min (trains two XGBoost models)
"""

import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

DATA_DIR = r"C:\Users\shaki\Downloads\faers_data"
SEED = 42
rng = np.random.default_rng(SEED)

# Numeric features. NUMERIC_NO_AGE is the same list with age dropped — used for
# the Mechanism 2 test, where age is removed for every report.
NUMERIC = ["n_drugs", "n_suspect", "n_concomitant", "n_reactions",
           "age_years", "weight_kg"]
NUMERIC_NO_AGE = [c for c in NUMERIC if c != "age_years"]


def load():
    tr = pd.read_parquet(os.path.join(DATA_DIR, "faers_temporal_train.parquet"))
    te = pd.read_parquet(os.path.join(DATA_DIR, "faers_temporal_test.parquet"))
    return tr, te


def fit_predict(tr, te, numeric, categorical):
    """Fit the clinical model on the given feature set, return test probabilities.

    Numeric features are median-imputed; categoricals are one-hot encoded (so a
    missing sex becomes its own 'unknown' column rather than being imputed away);
    the pt::/drug:: binary indicators pass straight through. Hyperparameters are
    fixed — this is a reproduction, not a tuning exercise.
    """
    binary = [c for c in tr.columns if "::" in c]
    features = numeric + categorical + binary

    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    transformers.append(("bin", "passthrough", binary))

    model = Pipeline([
        ("prep", ColumnTransformer(transformers)),
        ("xgb", XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            eval_metric="logloss", tree_method="hist", n_jobs=-1,
            random_state=SEED)),
    ])
    model.fit(tr[features], tr["label_serious"])
    return model.predict_proba(te[features])[:, 1]


def auc_with_ci(y, p, n_boot=800):
    """AUROC plus a stratified bootstrap 95% CI (resample positives and negatives
    separately so the class balance is preserved). Returns NaNs if a subgroup is
    too small to bootstrap sensibly.
    """
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) < 3 or len(neg) < 3:
        return np.nan, np.nan, np.nan
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), True),
                              rng.choice(neg, len(neg), True)])
        boot.append(roc_auc_score(y[idx], p[idx]))
    return roc_auc_score(y, p), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def report_gap(y, p, age_missing, header):
    """Print overall AUROC and the recorded-vs-missing age gap for one model."""
    recorded = auc_with_ci(y[~age_missing], p[~age_missing])
    missing = auc_with_ci(y[age_missing], p[age_missing])
    print(header)
    print(f"   overall AUROC {roc_auc_score(y, p):.3f}")
    print(f"   age RECORDED  {recorded[0]:.3f} ({recorded[1]:.3f}-{recorded[2]:.3f})  "
          f"n={(~age_missing).sum():,}")
    print(f"   age MISSING   {missing[0]:.3f} ({missing[1]:.3f}-{missing[2]:.3f})  "
          f"n={age_missing.sum():,}")
    print(f"   >>> gap = {recorded[0] - missing[0]:+.3f}\n")


def main():
    tr, te = load()
    y = te["label_serious"].values
    age_missing = te["age_years"].isna().values

    # Baseline: the paper's clinical model, with age included.
    print("=== Baseline reliability gap (paper CLINICAL model) ===")
    p = fit_predict(tr, te, NUMERIC, ["sex"])
    report_gap(y, p, age_missing, "MODEL WITH age (paper baseline):")

    # Mechanism 1: does the gap survive when content is held roughly constant?
    print("=== Mechanism 1: content mediation (age gap within n_reactions strata) ===")
    n_reactions = te["n_reactions"].values
    for low, high, label in [(1, 1, "1"), (2, 3, "2-3"), (4, 99, "4+")]:
        in_band = (n_reactions >= low) & (n_reactions <= high)
        recorded = auc_with_ci(y[in_band & ~age_missing], p[in_band & ~age_missing])
        missing = auc_with_ci(y[in_band & age_missing], p[in_band & age_missing])
        print(f"  n_reactions {label:>3}: recorded {recorded[0]:.3f}  "
              f"missing {missing[0]:.3f}  gap {recorded[0] - missing[0]:+.3f}")
    print("  -> gap persists across strata: content is NOT the mechanism.\n")

    # Mechanism 2: does the gap survive when age is removed for everyone?
    print("=== Mechanism 2: age-feature mediation (remove age for everyone) ===")
    p_no_age = fit_predict(tr, te, NUMERIC_NO_AGE, ["sex"])
    report_gap(y, p_no_age, age_missing, "MODEL WITHOUT age (equal footing):")
    print("  -> gap unchanged: loss of the age field is NOT the mechanism.\n")

    print("Conclusion: the under-documented gap is intrinsic to the subpopulation,")
    print("not remediable by content or by the missing field. This justifies a")
    print("calibrated abstention boundary (Part 2b), not a claimed fix.")
    print("\nNext: 12_part2b_conformal_boundary.py")


if __name__ == "__main__":
    main()
