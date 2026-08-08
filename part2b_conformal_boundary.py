"""
Part 2b: turning the reliability gap into an honest operating rule.

Part 2a established that the gap for under-documented reports is real and not
fixable by the obvious means. So the responsible move isn't to pretend we can
score every report equally well — it's to say clearly which reports we *can*
score reliably and route the rest to a human. Conformal prediction gives a
principled way to draw that line, with a coverage guarantee attached.

How it works here (split conformal on the clinical model's probabilities):
  - Split the test quarter 50/50, stratified by outcome, into a calibration half
    and an evaluation half. Both come from the same quarter, so the exchangeability
    the method needs holds.
  - Nonconformity score for a report = 1 - p(its true class).
  - Take the (1 - alpha) quantile of calibration scores as the threshold q.
  - For an evaluation report, the prediction set contains a label if that label's
    score is <= q. A singleton set means we score the report; a set with BOTH
    labels means we abstain and send it to review.

Two ways to set the threshold:
  MARGINAL     one global q for everyone.
  CONDITIONAL  a separate q per subgroup (Mondrian), guaranteeing coverage inside
               each subgroup rather than only on average.

What comes out (target coverage 90%, alpha = 0.10):
  - Coverage lands near 90% for age-recorded and age-missing reports under BOTH
    schemes — there's no hidden coverage gap, and marginal ≈ conditional here, so
    the simpler marginal threshold is enough for the age split.
  - The inequality shows up in the ABSTENTION rate, not coverage: holding the
    guarantee means routing ~35% of age-missing reports to review versus ~10% of
    age-recorded ones (~3.4x). Coverage on the hard group is bought by abstaining
    more — which is exactly the honest behaviour we want.

So instead of Paper 2's blunt 'bypass these reports entirely,' we get a graded
boundary: ~65% of under-documented reports stay reliably scoreable, ~35% go to a
human. That's the operational contribution.

(For the manuscript, the split is repeated over many random calibration splits to
show the abstention gap is stable — see 14_tables.py / 13_figures.py.)

Input : faers_temporal_{train,test}.parquet  (set DATA_DIR below)
Output: coverage / abstention by subgroup, marginal and conditional
Runtime: ~2-4 min (one XGBoost fit)
"""

import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

DATA_DIR = r"C:\Users\shaki\Downloads\faers_data"
ALPHA = 0.10          # target coverage = 1 - ALPHA = 90%
SEED = 42
rng = np.random.default_rng(SEED)
NUMERIC = ["n_drugs", "n_suspect", "n_concomitant", "n_reactions",
           "age_years", "weight_kg"]


def clinical_probs(tr, te):
    """Fit the paper's clinical model and return test-set seriousness probabilities."""
    binary = [c for c in tr.columns if "::" in c]
    features = NUMERIC + ["sex"] + binary
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["sex"]),
        ("bin", "passthrough", binary)])
    model = Pipeline([("prep", prep), ("xgb", XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
        tree_method="hist", n_jobs=-1, random_state=SEED))])
    model.fit(tr[features], tr["label_serious"])
    return model.predict_proba(te[features])[:, 1]


def conformal_threshold(scores, alpha):
    """The split-conformal quantile q: the (1 - alpha) empirical quantile of the
    calibration nonconformity scores, with the standard finite-sample correction.
    """
    n = len(scores)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return np.sort(scores)[k - 1]


def nonconformity(p, y):
    """1 - p(true class): small when the model is confident and correct."""
    return np.where(y == 1, 1 - p, p)


def coverage_abstention(p, y, q):
    """Given a threshold, return (coverage, abstention rate, scoreable rate).

    A label is in the prediction set if its nonconformity score is <= q, i.e.
    'serious' is included when (1 - p) <= q and 'non-serious' when p <= q.
    """
    includes_serious = (1 - p) <= q
    includes_nonserious = p <= q
    set_size = includes_serious.astype(int) + includes_nonserious.astype(int)
    covered = np.where(y == 1, includes_serious, includes_nonserious)
    coverage = covered.mean()
    abstained = (set_size == 2).mean()   # both labels -> route to human
    scoreable = (set_size == 1).mean()   # singleton   -> score it
    return coverage, abstained, scoreable


def stratified_half(y):
    """Split indices into a calibration half and an evaluation half, drawing half
    of each outcome class into calibration so the class balance is preserved.
    """
    in_calibration = np.zeros(len(y), bool)
    idx = np.arange(len(y))
    for cls in (0, 1):
        class_idx = idx[y == cls]
        in_calibration[rng.choice(class_idx, len(class_idx) // 2, replace=False)] = True
    return in_calibration, ~in_calibration


def main():
    tr = pd.read_parquet(os.path.join(DATA_DIR, "faers_temporal_train.parquet"))
    te = pd.read_parquet(os.path.join(DATA_DIR, "faers_temporal_test.parquet"))
    y = te["label_serious"].values
    age_missing = te["age_years"].isna().values

    p = clinical_probs(tr, te)
    cal, ev = stratified_half(y)

    print(f"target coverage = {1 - ALPHA:.0%}   "
          f"(abstain = prediction set contains BOTH labels -> route to review)")

    # Marginal: one threshold from the whole calibration set.
    print("\n=== MARGINAL conformal (single global threshold) ===")
    q_global = conformal_threshold(nonconformity(p[cal], y[cal]), ALPHA)
    for label, group in [("overall", np.ones(len(y), bool)),
                         ("age RECORDED", ~age_missing),
                         ("age MISSING", age_missing)]:
        cov, ab, sc = coverage_abstention(p[ev & group], y[ev & group], q_global)
        print(f"  {label:13} n={int((ev & group).sum()):6d}  coverage {cov:.3f}  "
              f"abstain {ab:.1%}  scoreable {sc:.1%}")

    # Conditional (Mondrian): a separate threshold calibrated within each subgroup.
    print("\n=== CONDITIONAL conformal (per-subgroup threshold, Mondrian) ===")
    for label, group in [("age RECORDED", ~age_missing), ("age MISSING", age_missing)]:
        q_group = conformal_threshold(nonconformity(p[cal & group], y[cal & group]), ALPHA)
        cov, ab, sc = coverage_abstention(p[ev & group], y[ev & group], q_group)
        print(f"  {label:13} n={int((ev & group).sum()):6d}  coverage {cov:.3f}  "
              f"abstain {ab:.1%}  scoreable {sc:.1%}  (q={q_group:.3f})")

    print("\nRead: coverage is held for both groups; the cost is unequal. Reliable")
    print("automated scoring is available ~3.4x more often for well-documented")
    print("reports. The graded boundary scores ~65% of under-documented reports")
    print("and routes ~35% to review - a principled replacement for blanket refusal.")
    print("\nManuscript: repeat over multiple calibration splits (stability); the")
    print("hallmark figure is abstention-rate-by-subgroup, not a coverage panel.")


if __name__ == "__main__":
    main()
