"""
Generate the three figures for Paper 3, straight from the parquet files.

  fig1_completeness_gradient.png  content and seriousness rising with documentation
  fig2_three_diagnostics.png      the mechanism story: two refutations + a refinement
  fig3_calibrated_boundary.png    coverage held for both groups, abstention unequal

The script is self-contained: it trains the clinical model inline, so it doesn't
depend on any cached predictions. The one value that isn't recomputed here is the
pair of "without age" AUROCs in Figure 2, panel B — those come from
11_reliability_mechanism.py and are pasted in as WITHOUT_AGE so the panel matches
that script exactly.

Input  : faers_temporal_{train,test}.parquet  (set DATA_DIR below)
Output : three PNGs written to OUT_DIR
Runtime: ~2-4 min (one XGBoost fit)
"""

import os

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")   # no display needed; render straight to file
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

DATA_DIR = r"C:\Users\shaki\Downloads\faers_data"   # <-- set to your data path
OUT_DIR = "."

# Panel B's "without age" subgroup AUROCs, reproduced by 11_reliability_mechanism.py.
WITHOUT_AGE = {"recorded": 0.912, "missing": 0.842}
ALPHA = 0.10   # conformal target coverage = 90%

# Shared look: drop the top/right spines, bump the DPI, bold titles.
mpl.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200,
                     "axes.titleweight": "bold"})
# Consistent colours: blue for "recorded/well-documented", orange for "missing".
REC, MISS = "#2c7fb8", "#d95f0e"


def clinical_probs(tr, te):
    """Fit the paper's clinical model and return test-set seriousness probabilities."""
    binary = [c for c in tr.columns if "::" in c]
    num = ["n_drugs", "n_suspect", "n_concomitant", "n_reactions", "age_years", "weight_kg"]
    feats = num + ["sex"] + binary
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["sex"]), ("bin", "passthrough", binary)])
    clf = Pipeline([("pre", pre), ("xgb", XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
        tree_method="hist", n_jobs=2, random_state=42))])
    clf.fit(tr[feats], tr["label_serious"])
    return clf.predict_proba(te[feats])[:, 1]


def fig1(te, y):
    """Figure 1: mean content and % serious across a 0-3 demographic-completeness score."""
    demo = ((~te["age_years"].isna()).astype(int) + te["sex"].ne("unk").astype(int)
            + (~te["weight_kg"].isna()).astype(int)).values
    lv = [0, 1, 2, 3]
    drugs = [te.loc[demo == k, "n_drugs"].mean() for k in lv]
    reacts = [te.loc[demo == k, "n_reactions"].mean() for k in lv]
    ser = [100 * y[demo == k].mean() for k in lv]
    fig, ax1 = plt.subplots(figsize=(6.2, 4))
    ax1.plot(lv, drugs, "-o", color=REC, label="mean drugs")
    ax1.plot(lv, reacts, "-s", color="#31a354", label="mean reactions")
    ax1.set_xlabel("Demographic documentation completeness (age, sex, weight)")
    ax1.set_ylabel("Mean count per report"); ax1.set_xticks(lv)
    # % serious lives on a second y-axis so it shares the x but keeps its own scale.
    ax2 = ax1.twinx(); ax2.spines["top"].set_visible(False)
    ax2.plot(lv, ser, "--^", color=MISS, label="% serious")
    ax2.set_ylabel("% serious", color=MISS); ax2.tick_params(axis="y", labelcolor=MISS)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=9)
    ax1.set_title("Figure 1. Documentation completeness tracks content and seriousness")
    fig.tight_layout(); fig.savefig(OUT_DIR + "/fig1_completeness_gradient.png"); plt.close(fig)


def fig2(te, p, y, am):
    """Figure 2: the mechanism spine across four panels.

    A: content mediation refuted (gap persists within reaction-count strata).
    B: feature mediation refuted (gap unchanged with vs without the age feature).
    C: reporter-stream refinement (gap concentrated in consumer reports).
    D: a plain-text inference summary tying the three together.
    """
    def auc(m): return roc_auc_score(y[m], p[m])
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.4)); w = 0.36

    # Panel A - AUROC by reaction-count band, recorded vs missing age.
    strata = [("1", te["n_reactions"].values == 1),
              ("2-3", (te["n_reactions"].values >= 2) & (te["n_reactions"].values <= 3)),
              ("4+", te["n_reactions"].values >= 4)]
    xr = np.arange(len(strata)); axA = axs[0, 0]
    axA.bar(xr - w/2, [auc(b & ~am) for _, b in strata], w, color=REC, label="age recorded")
    axA.bar(xr + w/2, [auc(b & am) for _, b in strata], w, color=MISS, label="age missing")
    axA.set_xticks(xr); axA.set_xticklabels([s for s, _ in strata]); axA.set_ylim(0.75, 0.95)
    axA.set_ylabel("AUROC"); axA.set_xlabel("n reactions")
    axA.set_title("A  Content mediation - refuted\n(gap persists within strata)", fontsize=10)
    axA.legend(frameon=False, fontsize=8)

    # Panel B - subgroup AUROC with age vs with age removed for everyone.
    axB = axs[0, 1]; xb = np.arange(2)
    axB.bar(xb - w/2, [auc(~am), WITHOUT_AGE["recorded"]], w, color=REC, label="age recorded")
    axB.bar(xb + w/2, [auc(am), WITHOUT_AGE["missing"]], w, color=MISS, label="age missing")
    axB.set_xticks(xb); axB.set_xticklabels(["with age", "without age"]); axB.set_ylim(0.75, 0.95)
    axB.set_ylabel("AUROC")
    axB.set_title("B  Feature mediation - refuted\n(gap unchanged when age removed)", fontsize=10)
    axB.legend(frameon=False, fontsize=8)

    # Panel C - gap within each reporter stream.
    REPLAB = {"1": "physician", "2": "pharmacist", "3": "other HP", "5": "consumer"}
    rep = te["qualification"].map(REPLAB).fillna("other/unk").values
    order = ["pharmacist", "physician", "other HP", "consumer"]; xc = np.arange(len(order))
    axC = axs[1, 0]
    axC.bar(xc - w/2, [auc((rep == g) & ~am) for g in order], w, color=REC, label="age recorded")
    axC.bar(xc + w/2, [auc((rep == g) & am) for g in order], w, color=MISS, label="age missing")
    axC.set_xticks(xc); axC.set_xticklabels(order, rotation=20, ha="right"); axC.set_ylim(0.75, 0.97)
    axC.set_ylabel("AUROC")
    axC.set_title("C  Reporter-stream refinement\n(gap concentrated in consumers)", fontsize=10)
    axC.legend(frameon=False, fontsize=8)

    # Panel D - text summary (no axes).
    axD = axs[1, 1]; axD.axis("off"); axD.set_title("D  Inference", fontsize=10, loc="left")
    axD.text(0, 0.95,
             "The under-documented reliability gap (0.913 -> 0.846)\nis NOT explained by:\n"
             "   - thinner content       (Panel A)\n   - loss of the age field  (Panel B)\n\n"
             "It is intrinsic to an under-documented subpopulation,\nconcentrated in the consumer/self-reporting\n"
             "stream (Panel C).\n\n=> Largely irreducible: the response is a calibrated\nabstention boundary (Fig 3), not a claimed fix.",
             va="top", ha="left", fontsize=9.3, family="monospace")
    fig.suptitle("Figure 2. Two mechanisms refuted, one refinement: the gap is intrinsic", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT_DIR + "/fig2_three_diagnostics.png"); plt.close(fig)


def fig3(p, y, am):
    """Figure 3: conformal coverage (held) vs abstention (unequal), over 100 splits.

    Repeats the split-conformal boundary across 100 random stratified calibration
    splits and plots the mean with an empirical 95% range as error bars: coverage
    on the left (both groups near the 90% target), abstention on the right (the
    ~3.4x gap that is the paper's operational finding).
    """
    idx = np.arange(len(y))
    def qhat(s, a): n = len(s); return np.sort(s)[min(int(np.ceil((n+1)*(1-a))), n)-1]
    def nc(pp, yy): return np.where(yy == 1, 1-pp, pp)
    cov_r, cov_m, ab_r, ab_m = [], [], [], []
    for seed in range(100):
        rng = np.random.default_rng(seed); cal = np.zeros(len(y), bool)
        for cls in (0, 1):
            c = idx[y == cls]; cal[rng.choice(c, len(c)//2, replace=False)] = True
        ev = ~cal; q = qhat(nc(p[cal], y[cal]), ALPHA)
        for mask, cl, al in [(~am, cov_r, ab_r), (am, cov_m, ab_m)]:
            e = ev & mask; s1 = (1-p[e]) <= q; s0 = p[e] <= q; size = s1.astype(int)+s0.astype(int)
            cl.append(np.where(y[e] == 1, s1, s0).mean()); al.append((size == 2).mean())
    def msr(v): return np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.5, 4))

    # Left: empirical coverage, both groups, with the 90% target line.
    cr, mr = msr(cov_r), msr(cov_m)
    axL.bar([0, 1], [cr[0], mr[0]], color=[REC, MISS],
            yerr=[[cr[0]-cr[1], mr[0]-mr[1]], [cr[2]-cr[0], mr[2]-mr[0]]], capsize=5)
    axL.axhline(0.90, ls="--", color="grey"); axL.text(1.35, 0.905, "target 90%", fontsize=8, color="grey")
    axL.set_xticks([0, 1]); axL.set_xticklabels(["age recorded", "age missing"]); axL.set_ylim(0.8, 0.95)
    axL.set_ylabel("Empirical coverage"); axL.set_title("Coverage is held for both groups", fontsize=10)

    # Right: abstention rate (% routed to review), the unequal cost.
    ar, amb = msr(ab_r), msr(ab_m)
    axR.bar([0, 1], [100*ar[0], 100*amb[0]], color=[REC, MISS],
            yerr=[[100*(ar[0]-ar[1]), 100*(amb[0]-amb[1])], [100*(ar[2]-ar[0]), 100*(amb[2]-amb[0])]], capsize=5)
    axR.set_xticks([0, 1]); axR.set_xticklabels(["age recorded", "age missing"])
    axR.set_ylabel("% routed to human review"); axR.set_title("...but the cost is unequal (3.4x)", fontsize=10)
    for i, v in enumerate([ar[0], amb[0]]):
        axR.text(i, 100*v+1.2, f"{100*v:.0f}%", ha="center", fontweight="bold")
    fig.suptitle("Figure 3. The calibrated boundary: equal reliability, unequal abstention", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT_DIR + "/fig3_calibrated_boundary.png"); plt.close(fig)


def main():
    tr = pd.read_parquet(os.path.join(DATA_DIR, "faers_temporal_train.parquet"))
    te = pd.read_parquet(os.path.join(DATA_DIR, "faers_temporal_test.parquet"))
    y = te["label_serious"].values
    am = te["age_years"].isna().values
    p = clinical_probs(tr, te)
    fig1(te, y); fig2(te, p, y, am); fig3(p, y, am)
    print("figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
