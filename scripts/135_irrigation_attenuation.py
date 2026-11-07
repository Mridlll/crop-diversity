"""
135_irrigation_attenuation.py

Is the flat within-district slope of the effective number of crops on irrigation (B12:
-0.225 per unit share, SE 0.192, district FE + state x year FE) a real null, or attenuation
from measurement error in the yearly land-use statistics (LUS) irrigated share? District
fixed effects throw away the between-district signal and keep the year-to-year wobble,
which is where reporting noise lives.

  A. Reproduce the within slope (D1 and D0 at 0.5 percent) on the 104 panel.
  B. Yearly noise in LUS irrigation: within a district, if the irrigated share is
     x_t = s_t + e_t with e_t classical (iid) and true year-on-year changes uncorrelated,
     Cov(dx_t, dx_{t-1}) = -Var(e). Implied reliability of the variation the FE slope uses
     (residual of irrigation after district and state x year FE).
  C. Decadal signal share: LUS change 2000-02 to 2010-12 against the census village
     directory change 2001 to 2011 (constant 2011 boundaries, 121), both within state.
     With independent errors, Cov(dLUS, dCensus) = Var(true change), so
     reliability(dLUS) = Cov / Var(dLUS) and reliability(dCensus) = Cov / Var(dCensus).
     Cluster bootstrap by state (999 draws) for the intervals.
  D. Long difference of D1 (and D0 at 0.5 percent) over the same windows on the LUS change,
     OLS and 2SLS with the census change as the instrument, state FE; first-stage F
     (cluster-robust), IV estimate and 95% interval, Anderson-Rubin interval (valid under
     a weak instrument); clusters: 2011 parent and state.
  E. Adversarial: other windows (1999-2003 to 2009-2013), drop districts sharing a 2011
     parent, drop districts with an LUS territory break, drop spike-flagged years, drop the
     states with thin LUS coverage, hold the 2001 census unattributed share constant (the
     2001 directory has no "well with electricity" field), reduced form alone.
  F. What the within slope can and cannot rule out, set against the cross-sectional slope
     above the knot (134).

Outputs (outputs/revision_nov8/irrigation_v3/):
  attenuation_results.csv, attenuation_longdiff_data.csv, attenuation_raw.md

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/135_irrigation_attenuation.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = r"D:/crop-diversity"
sys.path.insert(0, REPO + "/scripts")
from _years import FIRST_YEAR, LAST_YEAR  # noqa: E402
from _revision_fe import fe_ols, fe_iv, demean, pval  # noqa: E402

OUT = REPO + "/outputs/revision_nov8/irrigation_v3"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20261110)
LOG, RES = [], []


def w(s=""):
    print(s)
    LOG.append(str(s))


def rec(**k):
    RES.append(k)


# ---------------------------------------------------------------- panel
P = pd.read_csv(REPO + "/outputs/revision_nov8/irrigation_panel_district_year.csv")
P = P[P["in_final"] & P["year_start"].between(FIRST_YEAR, LAST_YEAR)].copy()
P["irr"] = P["irr_nsa"]
P["sy"] = P["state_name"] + "_" + P["year_start"].astype(str)
P = P.sort_values(["district_key", "year_start"])
P["spell"] = P["district_key"] + "_" + P.groupby("district_key")["brk_nsa"].cumsum().astype(str)
E = P[P["irr"].notna()].copy()
NESTED = ("district_key", "spell")

w("# Irrigation within districts: attenuation check")
w("")
w("Window rule {}-{} to {}-{} (scripts/_years.py). Panel: {:,} district-years with LUS irrigation, "
  "{} districts.".format(FIRST_YEAR, str(FIRST_YEAR + 1)[-2:], LAST_YEAR, str(LAST_YEAR + 1)[-2:],
                         len(E), E["district_key"].nunique()))

# ---------------------------------------------------------------- A
w("")
w("## A. Within slope (district FE + state x year FE, clusters = 2011 parents)")
WITHIN = {}
for y in ["D1", "D0_05"]:
    r = fe_ols(E, y, ["irr"], ["district_key", "sy"], "shrug_key", nested=NESTED)
    b, se = r["params"]["irr"], r["se"]["irr"]
    WITHIN[y] = (b, se, r["n"], r["G"])
    w("- {}: {:+.3f} ({:.3f}), n {:,}, G {}".format(y, b, se, r["n"], r["G"]))
    rec(part="A", label="within slope " + y, b=b, se=se, n=r["n"], G=r["G"])
r = fe_ols(E, "D1", ["irr"], ["district_key", "sy"], "shrug_key", nested=NESTED)
dmx = demean(r["data"], ["irr"], ["district_key", "sy"])["irr"].to_numpy()
var_res = float(dmx.var())
w("Residual SD of irrigation after district and state x year FE: {:.4f} (variance {:.5f}); "
  "total SD {:.3f}.".format(np.sqrt(var_res), var_res, E["irr"].std()))

# ---------------------------------------------------------------- B yearly noise
w("")
w("## B. Yearly noise in the LUS share (first-difference autocovariance)")
Q = E[["district_key", "year_start", "irr", "state_name", "irr_spike", "brk_nsa"]].copy()
Q["dx"] = Q.groupby("district_key")["irr"].diff()
Q.loc[Q.groupby("district_key")["year_start"].diff() != 1, "dx"] = np.nan
Q.loc[Q["brk_nsa"], "dx"] = np.nan
Q["dx_l"] = Q.groupby("district_key")["dx"].shift(1)
Q.loc[Q.groupby("district_key")["year_start"].diff() != 1, "dx_l"] = np.nan


def noise_var(Qs):
    q = Qs.dropna(subset=["dx", "dx_l"]).copy()
    # remove state x year means of the changes (state-year FE remove common shocks)
    q["sy"] = q["state_name"] + "_" + q["year_start"].astype(str)
    q["dxr"] = q["dx"] - q.groupby("sy")["dx"].transform("mean")
    q["dxlr"] = q["dx_l"] - q.groupby("sy")["dx_l"].transform("mean")
    c = float((q["dxr"] * q["dxlr"]).mean())
    v = float((q["dxr"] ** 2).mean())
    return -c, v, len(q), q


T_bar = E.groupby("district_key")["irr"].count().mean()
NOISE = {}
for lab, Qs in [("all years", Q), ("spike-flagged years dropped", Q[~Q["irr_spike"]])]:
    s2e, vdx, nq, q = noise_var(Qs)
    lam_w = 1 - s2e * (1 - 1 / T_bar) / var_res
    # cluster bootstrap by district for the noise variance
    keys = q["district_key"].unique()
    grp = {k: g for k, g in q.groupby("district_key")}
    bs = []
    for _ in range(499):
        pick = RNG.choice(keys, len(keys))
        qq = pd.concat([grp[k] for k in pick])
        bs.append(-(qq["dxr"] * qq["dxlr"]).mean())
    lo, hi = np.quantile(bs, [.025, .975])
    NOISE[lab] = (s2e, lam_w)
    w("- {}: n {:,} pairs of changes; Var(change) {:.5f}; implied noise variance {:.5f} (SD {:.3f}; "
      "district bootstrap 95% {:.5f} to {:.5f}); first-order autocorrelation of changes {:.2f}. "
      "Mean years per district {:.1f}. Implied reliability of the FE-residual variation {:.2f} "
      "(noise share {:.2f})".format(lab, nq, vdx, s2e, np.sqrt(max(s2e, 0)), lo, hi, -s2e / vdx, T_bar,
                                    lam_w, 1 - lam_w))
    rec(part="B", label="yearly noise " + lab, noise_var=s2e, noise_lo=lo, noise_hi=hi, var_change=vdx,
        var_fe_resid=var_res, reliability_within=lam_w, n=nq)
s2e, lam_w = NOISE["all years"]
b, se, n, G = WITHIN["D1"]
w("Within D1 slope corrected for this reliability: {:+.3f} (SE {:.3f}), 95% {:+.2f} to {:+.2f} per unit share.".format(
    b / lam_w, se / lam_w, (b - 1.96 * se) / lam_w, (b + 1.96 * se) / lam_w))
rec(part="B", label="within D1 corrected by yearly reliability", b=b / lam_w, se=se / lam_w,
    lo=(b - 1.96 * se) / lam_w, hi=(b + 1.96 * se) / lam_w)

# ---------------------------------------------------------------- C/D long differences
cen = pd.read_csv(REPO + "/outputs/revision_nov8/census_irrigation/census_irrigation_panel.csv")
cw = cen.pivot_table(index="shrug_key", columns="round",
                     values=["irr_share", "district_usable", "sh_unattributed", "sh_groundwater"], aggfunc="first")
cw.columns = ["{}_{}".format(a, b) for a, b in cw.columns]
cw = cw.reset_index()
for r_ in (2001, 2011):
    cw["district_usable_%d" % r_] = cw["district_usable_%d" % r_].astype(object).eq(True)
cw["dcen"] = (cw["irr_share_2011"] - cw["irr_share_2001"]).where(cw["district_usable_2001"] & cw["district_usable_2011"])
cw["unatt01"] = cw["sh_unattributed_2001"]


def ld(lo0, hi0, lo1, hi1, minyrs, Ps=P):
    def per(lo, hi):
        q = Ps[Ps["year_start"].between(lo, hi)]
        g = q.groupby("district_key")
        o = g[["D1", "D0_05", "irr"]].mean()
        o["n_irr"], o["n_y"] = g["irr"].count(), g["D1"].count()
        o["brk"] = g["brk_nsa"].sum()
        return o
    a, b_ = per(lo0, hi0), per(lo1, hi1)
    L = a.join(b_, lsuffix="_0", rsuffix="_1", how="inner")
    L = L[(L[["n_irr_0", "n_irr_1", "n_y_0", "n_y_1"]] >= minyrs).all(axis=1)]
    # breaks between and inside the windows
    br = Ps[Ps["year_start"].between(lo0 + 1, hi1)].groupby("district_key")["brk_nsa"].sum().rename("brk_any")
    meta = Ps.groupby("district_key")[["state_name", "shrug_key", "flag_shared_parent"]].first()
    L = L.join(meta).join(br).reset_index()
    L["dlus"] = L["irr_1"] - L["irr_0"]
    L["dD1"] = L["D1_1"] - L["D1_0"]
    L["dD0_05"] = L["D0_05_1"] - L["D0_05_0"]
    L = L.merge(cw[["shrug_key", "dcen", "unatt01", "irr_share_2001", "irr_share_2011"]], on="shrug_key", how="left")
    return L.dropna(subset=["dlus", "dcen"])


def reliab(L):
    g = pd.factorize(L["state_name"])[0]
    a = L["dlus"].to_numpy() - L.groupby("state_name")["dlus"].transform("mean").to_numpy()
    c = L["dcen"].to_numpy() - L.groupby("state_name")["dcen"].transform("mean").to_numpy()
    cov = (a * c).mean()
    return dict(cov=cov, var_lus=(a ** 2).mean(), var_cen=(c ** 2).mean(), lam_lus=cov / (a ** 2).mean(),
                lam_cen=cov / (c ** 2).mean(), corr=cov / np.sqrt((a ** 2).mean() * (c ** 2).mean()))


def reliab_boot(L, B_=999):
    states = L["state_name"].unique()
    grp = {s: g for s, g in L.groupby("state_name")}
    out = []
    for _ in range(B_):
        pick = RNG.choice(states, len(states))
        parts = []
        for j, s in enumerate(pick):
            gg = grp[s].copy()
            gg["state_name"] = "{}_{}".format(s, j)
            parts.append(gg)
        out.append(reliab(pd.concat(parts)))
    o = pd.DataFrame(out)
    return {k: tuple(np.quantile(o[k], [.025, .975])) for k in ["lam_lus", "lam_cen", "corr"]}


def ar_ci(L, y, cl, grid=np.linspace(-40, 40, 1601)):
    """Anderson-Rubin set: values b0 for which the reduced form of (y - b0 dlus) on dcen is not
    rejected at 5% (cluster-robust t, state FE)."""
    keep = []
    for b0 in grid:
        Lb = L.assign(yb=L[y] - b0 * L["dlus"])
        r = fe_ols(Lb, "yb", ["dcen"], ["state_name"], cl)
        G = r["G"]
        crit = stats.t.ppf(.975, G - 1) if cl == "state_name" else 1.96
        if abs(r["params"]["dcen"] / r["se"]["dcen"]) <= crit:
            keep.append(b0)
    if not keep:
        return (np.nan, np.nan, "empty")
    keep = np.array(keep)
    connected = np.all(np.diff(keep) < (grid[1] - grid[0]) * 1.5)
    edge = keep.min() <= grid[0] or keep.max() >= grid[-1]
    return (keep.min(), keep.max(), ("unbounded" if edge else ("connected" if connected else "not connected")))


def ld_block(L, lab, full=True):
    rl = reliab(L)
    out = dict(label=lab, n=len(L), G_state=L["state_name"].nunique(), G_par=L["shrug_key"].nunique(),
               mean_dlus=L["dlus"].mean(), mean_dcen=L["dcen"].mean(), **rl)
    txt = ["### " + lab, "",
           "n {} districts ({} parents, {} states). Mean change: LUS {:+.3f}, census {:+.3f}. Within state: "
           "Var(dLUS) {:.4f}, Var(dCensus) {:.4f}, Cov {:.4f}, correlation {:.2f}; signal share of the LUS change "
           "{:.2f}, of the census change {:.2f}.".format(len(L), out["G_par"], out["G_state"], out["mean_dlus"],
                                                          out["mean_dcen"], rl["var_lus"], rl["var_cen"], rl["cov"],
                                                          rl["corr"], rl["lam_lus"], rl["lam_cen"])]
    if full:
        bb = reliab_boot(L)
        out.update(lam_lus_lo=bb["lam_lus"][0], lam_lus_hi=bb["lam_lus"][1],
                   lam_cen_lo=bb["lam_cen"][0], lam_cen_hi=bb["lam_cen"][1])
        txt.append("State cluster bootstrap 95%: LUS signal share {:.2f} to {:.2f}; census {:.2f} to {:.2f}; "
                   "correlation {:.2f} to {:.2f}.".format(*bb["lam_lus"], *bb["lam_cen"], *bb["corr"]))
    for y in ["dD1", "dD0_05"]:
        for cl in ["shrug_key", "state_name"]:
            if cl == "state_name" and not full:
                continue
            o = fe_ols(L, y, ["dlus"], ["state_name"], cl)
            rf = fe_ols(L, y, ["dcen"], ["state_name"], cl)
            iv = fe_iv(L, y, ["dlus"], [], ["dcen"], ["state_name"], cl)
            G = iv["G"]
            crit = stats.t.ppf(.975, G - 1) if cl == "state_name" else 1.96
            b, se = iv["params"]["dlus"], iv["se"]["dlus"]
            F = iv["first"][0]["F"]
            fsb = iv["first"][0]["coefs"]["dcen"]
            ar = ar_ci(L, y, cl) if (full or y == "dD1") else (np.nan, np.nan, "")
            row = dict(label=lab, y=y, cluster=cl, n=iv["n"], G=G, ols=o["params"]["dlus"], ols_se=o["se"]["dlus"],
                       rf=rf["params"]["dcen"], rf_se=rf["se"]["dcen"], fs=fsb, F=F, iv=b, iv_se=se,
                       iv_lo=b - crit * se, iv_hi=b + crit * se, iv_p=pval(b, se, G if cl == "state_name" else None),
                       ar_lo=ar[0], ar_hi=ar[1], ar_shape=ar[2])
            RES.append(dict(part="D", **row))
            txt.append("- {} (clusters {}): OLS {:+.3f} ({:.3f}); reduced form on census change {:+.3f} ({:.3f}); "
                       "first stage {:+.3f}, F {:.1f}; IV {:+.3f} ({:.3f}), 95% {:+.2f} to {:+.2f}, p {:.3f}; "
                       "AR 95% {} to {} ({})".format(
                           y, "2011 parent" if cl == "shrug_key" else "state", row["ols"], row["ols_se"], row["rf"],
                           row["rf_se"], fsb, F, b, se, row["iv_lo"], row["iv_hi"], row["iv_p"],
                           "{:+.2f}".format(ar[0]) if np.isfinite(ar[0]) else "na",
                           "{:+.2f}".format(ar[1]) if np.isfinite(ar[1]) else "na", ar[2]))
    RES.append(dict(part="C", **out))
    for t in txt:
        w(t)
    w("")
    return out


w("")
w("## C-D. Long differences, census change as the instrument for the LUS change")
w("")
L0 = ld(2000, 2002, 2010, 2012, 2)
L0.to_csv(OUT + "/attenuation_longdiff_data.csv", index=False)
main_ld = ld_block(L0, "main: 2000-02 to 2010-12 (at least 2 of 3 years each end)")

w("## E. Adversarial")
w("")
ld_block(ld(1999, 2003, 2009, 2013, 3), "windows 1999-2003 to 2009-2013 (at least 3 of 5 years)", full=False)
ld_block(L0[~L0["flag_shared_parent"].astype(bool)], "drop districts sharing a 2011 parent", full=False)
ld_block(L0[L0["brk_any"].fillna(0) == 0], "drop districts with an LUS territory break", full=False)
ld_block(ld(2000, 2002, 2010, 2012, 2, Ps=P.assign(irr=P["irr"].where(~P["irr_spike"]))),
         "spike-flagged LUS years set to missing", full=False)
thin = ["WEST BENGAL", "MAHARASHTRA", "MANIPUR"]
ld_block(L0[~L0["state_name"].isin(thin)], "drop West Bengal, Maharashtra, Manipur (thin LUS coverage)", full=False)
Lu = L0[L0["unatt01"].notna()]
Lu = Lu[Lu["unatt01"] <= Lu["unatt01"].median()]
ld_block(Lu, "2001 census unattributed share at or below its median", full=False)

# ---------------------------------------------------------------- F. what the within slope rules out
w("## F. What the within slope can and cannot rule out")
w("")
spl = pd.read_csv(OUT + "/threshold_spline.csv")
sp = spl[(spl["window"] == "full") & spl["chosen"] & (spl["y"] == "D1")].iloc[0]
b, se, n, G = WITHIN["D1"]
lam_dec = main_ld["lam_lus"]
w("Cross-sectional slope above the knot ({:.2f}): {:+.2f} per unit (95% {:+.2f} to {:+.2f}), "
  "i.e. {:+.2f} per 10 points.".format(sp["knot"], sp["above"], sp["above_lo"], sp["above_hi"], sp["above"] / 10))
w("Within slope as estimated: {:+.3f} per unit, 95% {:+.2f} to {:+.2f}.".format(b, b - 1.96 * se, b + 1.96 * se))
for lab, lam in [("yearly-noise reliability", lam_w), ("decadal signal share (census comparison)", lam_dec)]:
    lo, hi = (b - 1.96 * se) / lam, (b + 1.96 * se) / lam
    w("- corrected by the {} {:.2f}: {:+.2f} per unit, 95% {:+.2f} to {:+.2f}; the cross-sectional slope {:+.2f} "
      "is {} this interval.".format(lab, lam, b / lam, lo, hi, sp["above"], "outside" if sp["above"] < lo else "inside"))
    rec(part="F", label="within corrected by " + lab, lam=lam, b=b / lam, lo=lo, hi=hi, xs=sp["above"])
lam_need = (b - 1.96 * se) / sp["above"]
w("Reliability at which the lower end of the within interval would reach the cross-sectional slope: {:.2f}.".format(lam_need))
rec(part="F", label="reliability needed to reach the cross-sectional slope", lam=lam_need)
r = [x for x in RES if x.get("part") == "D" and x["label"].startswith("main") and x["y"] == "dD1" and x["cluster"] == "shrug_key"][0]
w("Long-difference IV (main, parent clusters): {:+.2f}, 95% {:+.2f} to {:+.2f}; AR {} to {} ({}).".format(
    r["iv"], r["iv_lo"], r["iv_hi"], r["ar_lo"], r["ar_hi"], r["ar_shape"]))

pd.DataFrame(RES).to_csv(OUT + "/attenuation_results.csv", index=False)
with open(OUT + "/attenuation_raw.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG) + "\n")
print("wrote " + OUT)
