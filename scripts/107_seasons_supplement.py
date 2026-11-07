"""
107_seasons_supplement.py

Supplementary numbers for the seasons write-up, read from script 104's outputs
(outputs/revision_nov8/seasons_district.csv, seasons_regressions.csv). No new
construction. Adds: rank correlations, the curvature table with SEs, movers on the
exclusive-partition index A_alpha, season D1 by system without Whole Year crops,
intensity-tertile table, and two extra inverted-U checks (LUS cropping intensity as
a control; GCA weights).

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/107_seasons_supplement.py
Output: outputs/revision_nov8/seasons_supplement_log.txt
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

OUT = r"D:/crop-diversity/outputs/revision_nov8"
L = []


def w(s=""):
    print(s)
    L.append(str(s))


T = pd.read_csv(OUT + "/seasons_district.csv")
S = T[T["in_final"] == True].copy()
w("n = {}".format(len(S)))
for v in ["D1_ws", "D1_ws_noW", "A_alpha", "D1_K", "D1_R"]:
    s = S.dropna(subset=[v])
    r = stats.spearmanr(s["D1_ann"], s[v])
    w("spearman D1_ann vs {}: {:.3f} (n {})".format(v, r.correlation, len(s)))

R = pd.read_csv(OUT + "/seasons_regressions.csv")
w(R[["sample", "y", "n", "b1", "se1", "b2", "se2", "p2", "tp", "tp_lo", "tp_hi",
     "slope_p2", "slope_p2_p", "slope_p98", "slope_p98_p"]].round(4).to_string())

S["rk_ann"] = S["D1_ann"].rank(ascending=False)
S["rk_A"] = S["A_alpha"].rank(ascending=False)
S["mv"] = S["rk_A"] - S["rk_ann"]
cols = ["district_name", "state_name", "sys", "D1_ann", "A_alpha", "B_beta", "beta_share",
        "ci_lus", "rk_ann", "rk_A", "mv"]
m = S.sort_values("mv")[cols]
w("RISERS on A_alpha (rank improves within season)")
w(m.head(12).round(2).to_string())
w("FALLERS on A_alpha")
w(m.tail(12).round(2).to_string())
w("median |move| {:.0f}, >100: {}".format(S["mv"].abs().median(), int((S["mv"].abs() > 100).sum())))

g = []
for f, s in S.groupby("sys"):
    if len(s) < 6:
        continue
    row = {"sys": f, "n": len(s)}
    for c in ["D1_ann", "D1_K_noW", "D1_R_noW", "D1_ws_noW", "A_alpha", "B_beta", "beta_share",
              "ci_lus", "irr_share"]:
        x = s[c].dropna()
        row[c] = "{:.2f} ({:.2f})".format(x.mean(), x.std() / np.sqrt(len(x)))
    g.append(row)
w(pd.DataFrame(g).sort_values("n", ascending=False).to_string())

s = S.dropna(subset=["ci_lus"]).copy()
s["ci_t"] = pd.qcut(s["ci_lus"], 3, labels=["low", "mid", "high"])
w(s.groupby("ci_t", observed=True).agg(n=("D1_ann", "size"), ci=("ci_lus", "mean"),
                                       D1_ann=("D1_ann", "mean"), A=("A_alpha", "mean"),
                                       B=("B_beta", "mean"), bsh=("beta_share", "mean")).round(3).to_string())

def u(d, y, extra="", wt=None):
    d = d.dropna(subset=[y, "irr_share", "shrug_key"] + ([extra] if extra else [])).copy()
    f = "{} ~ irr_share + I(irr_share**2) + C(state_name)".format(y) + (" + " + extra if extra else "")
    kw = dict(cov_type="cluster", cov_kwds={"groups": d["shrug_key"]})
    m = smf.wls(f, data=d, weights=d[wt]).fit(**kw) if wt else smf.ols(f, data=d).fit(**kw)
    b1, b2 = m.params["irr_share"], m.params["I(irr_share ** 2)"]
    return "b2 {:.3f} (se {:.3f}, p {:.2g}), tp {:.3f}, n {}".format(
        b2, m.bse["I(irr_share ** 2)"], m.pvalues["I(irr_share ** 2)"], -b1 / (2 * b2), int(m.nobs))

for y in ["D1_ann", "D1_ws", "A_alpha", "H_alpha", "H_beta"]:
    w("{} + LUS CI control: {}".format(y, u(S, y, "ci_lus")))
    w("{} GCA-weighted: {}".format(y, u(S, y, wt="gca_mean")))
open(OUT + "/seasons_supplement_log.txt", "w", encoding="utf-8").write("\n".join(L))
