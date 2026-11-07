"""
137_market_haat_mechanism.py

What the weekly haat result is made of, plus the referee checks asked for on the
Markets page.

  A  haat coefficient on the count of crops at area floors of 0 (all crops), 0.1, 0.5
     and 1 percent of gross cropped area, and on the effective number D1; all crops and
     with the six break crops removed (drop6). Site model: population shares (version W),
     all site controls, state FE; SE on the 2011 parent, then state clustering t(G-1)
     and a restricted wild cluster bootstrap by state (Webb, 9,999 draws).
  B  crop-level presence: for each crop, the share of the district's years 1998-99 to
     2019-20 in which it is grown (any area), regressed on haat reach with the same model;
     BH q across crops. Also each crop's share of cropped area (percentage points).
  C  sugar: haat on sugar-crop share (solo and the output-market block that gave
     +10.1 points, q 0.003), leaving out one state at a time; without Maharashtra and Uttar
     Pradesh together; state clustering and wild bootstrap.
  D  the "producer organisation" item: Mission Antyodaya's question is "Does the village
     have any Farmers Collective", coded FPO 1, PACS 2, both 3, none 4; SHRUG's 0/1 `fpo`
     is 1 for codes 1 to 3. Checks: correlation with Census 2011 agricultural credit
     society reach, and the site's coefficient re-estimated with that census measure.
  E  the haat result re-estimated with the Census 2011 weekly haat item (a multi-answer
     item) in place of Mission Antyodaya's single-answer market item.

Outputs (outputs/revision_nov8/markets_v2/):
  haat_thresholds.csv, haat_crop_presence.csv, sugar_leave_one_state_out.csv,
  fpo_checks.csv, haat_census2011_check.csv, haat_mechanism_log.md
  docs/data/site_markets_v2.json (chart data for the Markets page)

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/137_market_haat_mechanism.py
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _mkt2_common import REPO, apy_base, fit_par, load_104, wcr  # noqa: E402

warnings.filterwarnings("ignore")
OUT = REPO + "/outputs/revision_nov8/markets_v2"
os.makedirs(OUT, exist_ok=True)
DKEY = ["pc11_state_id", "pc11_district_id"]
LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


ns = load_104()
X = ns["prep"]("W")
CTRL = ns["CTRL"]
X = X[["district_key", "shrug_key", "state_name"] + DKEY +
      ["mean_holding_ha", "log_viirs", "estab_per_1000pop", "idx_connectivity_use", "pca_st_share",
       "irr_share", "D0_richness", "D1_exp_shannon", "share_sugar", "share_cereals", "share_pulses"] +
      ["F_" + s for _, s in ns["FAC"]]].copy()
assert len(X) == 606
BASEKEEP = ["mean_holding_ha", "log_viirs", "shrug_key", "state_name"]

# ================================================================ crop record, full window
a = apy_base()
base = a["base"]
DROP6 = a["DROP6"]
base = base[base["year_start"].between(1998, 2019)]
base = base[base["district_key"].isin(X["district_key"])]


def idx(dy, tag):
    g = dy.groupby(["district_key", "year_start"])["area"]
    p = dy["area"] / g.transform("sum")
    t = dy.assign(p=p, plnp=-p * np.log(p),
                  f01=(p >= 0.001).astype(int), f05=(p >= 0.005).astype(int), f1=(p >= 0.01).astype(int))
    y = t.groupby(["district_key", "year_start"]).agg(
        D0=("crop_name", "size"), H=("plnp", "sum"), D0_01=("f01", "sum"),
        D0_05=("f05", "sum"), D0_1=("f1", "sum")).reset_index()
    y["D1"] = np.exp(y["H"])
    m = y.groupby("district_key")[["D0", "D0_01", "D0_05", "D0_1", "D1"]].mean()
    m.columns = [c + tag for c in m.columns]
    return m


TH = idx(base, "").join(idx(base[~base["crop_name"].isin(DROP6)], "_d6"))
X = X.merge(TH, left_on="district_key", right_index=True, how="left")
dmax = (X["D0"] - X["D0_richness"]).abs().max()
w("# Haat mechanism and referee checks: raw log")
w("")
w("Check: rebuilt full-window count of crops against the site's D0_richness, max abs "
  "difference {:.2e}; D1 {:.2e}.".format(dmax, (X["D1"] - X["D1_exp_shannon"]).abs().max()))
assert dmax < 1e-6
w("Means over the 606: D0 {:.2f}; at 0.1% {:.2f}; at 0.5% {:.2f}; at 1% {:.2f}; D1 {:.3f}.".format(
    *[X[c].mean() for c in ["D0", "D0_01", "D0_05", "D0_1", "D1"]]))
w("Crops lost when moving the floor: 0 to 0.1% {:.2f}; 0.1% to 0.5% {:.2f}; 0.5% to 1% {:.2f}.".format(
    (X.D0 - X.D0_01).mean(), (X.D0_01 - X.D0_05).mean(), (X.D0_05 - X.D0_1).mean()))

sd_haat = X["F_weekly_haat"].std()
w("SD of haat population share: {:.4f}.".format(sd_haat))


def full_inf(df, y, term, extra_rhs=None):
    rhs = extra_rhs if extra_rhs else term
    f = "{} ~ {} + {}".format(y, rhs, CTRL)
    keep = [y] + [v.strip() for v in rhs.split("+")] + BASEKEEP
    m, s = fit_par(df, f, keep)
    wb = wcr(df, f, term, keep)
    ci = m.conf_int().loc[term]
    return dict(b=m.params[term], se=m.bse[term], p=m.pvalues[term], lo=ci[0], hi=ci[1],
                n=int(m.nobs), se_state=wb["se_st"], p_state=wb["p_st"], p_wild=wb["p_wcr"],
                G=wb["G"])


# ================================================================ A thresholds
w("")
w("## A. Haat coefficient by area floor (0 to all rural people with a haat)")
w("")
w("| outcome | crops | n | b | SE parent | p parent | p state t(G-1) | p wild | one SD | mean |")
w("|---|---|---|---|---|---|---|---|---|---|")
TROWS = []
for y, lab in [("D0", "count, any area"), ("D0_01", "count, 0.1% floor"), ("D0_05", "count, 0.5% floor"),
               ("D0_1", "count, 1% floor"), ("D1", "effective number")]:
    for v, suf in [("all", ""), ("drop6", "_d6")]:
        r = full_inf(X, y + suf, "F_weekly_haat")
        r.update(outcome=y, label=lab, crops=v, mean=X[y + suf].mean(), sd_effect=r["b"] * sd_haat)
        TROWS.append(r)
        w("| {label} | {crops} | {n} | {b:+.3f} | {se:.3f} | {p:.4f} | {p_state:.4f} | {p_wild:.4f} | "
          "{sd_effect:+.3f} | {mean:.2f} |".format(**r))
pd.DataFrame(TROWS).to_csv(OUT + "/haat_thresholds.csv", index=False)

# ================================================================ B crop presence
w("")
w("## B. Which crops haat-rich districts add (presence = share of years grown)")
w("")
ny = base.groupby("district_key")["year_start"].nunique()
pres = (base[base["area"] > 0].groupby(["district_key", "crop_name"])["year_start"].nunique()
        .unstack(fill_value=0).div(ny, axis=0))
area = base.groupby(["district_key", "crop_name"])["area"].sum().unstack(fill_value=0)
share = area.div(area.sum(axis=1), axis=0)
crops = [c for c in pres.columns if (pres[c] > 0).sum() >= 60]
w("Crop names in 1998-99 to 2019-20: {}; grown in at least 60 of the 606 districts: {}.".format(
    pres.shape[1], len(crops)))
CROWS = []
for c in crops:
    t = X.merge(pd.DataFrame({"pres": pres[c], "shr": share[c] * 100}), left_on="district_key",
                right_index=True, how="left")
    t[["pres", "shr"]] = t[["pres", "shr"]].fillna(0)
    m, s = fit_par(t, "pres ~ F_weekly_haat + " + CTRL, ["pres", "F_weekly_haat"] + BASEKEEP)
    m2, _ = fit_par(t, "shr ~ F_weekly_haat + " + CTRL, ["shr", "F_weekly_haat"] + BASEKEEP)
    ci = m.conf_int().loc["F_weekly_haat"]
    CROWS.append(dict(crop=c, crop_type=a["crop_type"].get(c, ""), n_grown=int((pres[c] > 0).sum()),
                      mean_presence=t["pres"].mean(), mean_share_pct=t["shr"].mean(),
                      b=m.params["F_weekly_haat"], se=m.bse["F_weekly_haat"],
                      p=m.pvalues["F_weekly_haat"], lo=ci[0], hi=ci[1],
                      b_share_pp=m2.params["F_weekly_haat"], p_share=m2.pvalues["F_weekly_haat"],
                      drop6=c in DROP6))
C = pd.DataFrame(CROWS)
C["q"] = multipletests(C["p"], method="fdr_bh")[1]
C["q_share"] = multipletests(C["p_share"], method="fdr_bh")[1]
C = C.sort_values("b", ascending=False)
# state-clustered wild p for the crops that pass BH
C["p_wild"] = np.nan
for i, r in C[C.q < 0.05].iterrows():
    t = X.merge(pd.DataFrame({"pres": pres[r.crop]}), left_on="district_key", right_index=True, how="left")
    t["pres"] = t["pres"].fillna(0)
    C.loc[i, "p_wild"] = wcr(t, "pres ~ F_weekly_haat + " + CTRL, "F_weekly_haat",
                             ["pres", "F_weekly_haat"] + BASEKEEP)["p_wcr"]
C.to_csv(OUT + "/haat_crop_presence.csv", index=False)
w("| crop | type | districts | mean presence | mean share % | b presence | q | p wild | b share pp | q share |")
w("|---|---|---|---|---|---|---|---|---|---|")
for _, r in C.iterrows():
    w("| {crop} | {crop_type} | {n_grown} | {mean_presence:.2f} | {mean_share_pct:.2f} | {b:+.3f} | "
      "{q:.3f} | {pw} | {b_share_pp:+.2f} | {q_share:.3f} |".format(
          pw="" if np.isnan(r.p_wild) else "{:.3f}".format(r.p_wild), **r))
pos = C[(C.q < 0.05) & (C.b > 0)]
neg = C[(C.q < 0.05) & (C.b < 0)]
w("")
w("Crops more often grown where haats reach more people (q < 0.05): {} ({}). Less often: {} ({}).".format(
    len(pos), ", ".join(pos.crop), len(neg), ", ".join(neg.crop)))
w("Sum of presence coefficients over all crops: {:+.2f} (compare the count-of-crops coefficient, "
  "which is the same sum over every crop name).".format(C["b"].sum()))
w("Of the positive q < 0.05 crops: mean area share {:.2f}%, median {:.2f}%; mean share coefficient "
  "{:+.2f} pp.".format(pos.mean_share_pct.mean(), pos.mean_share_pct.median(), pos.b_share_pp.mean()))

# ================================================================ C sugar leave-one-state-out
w("")
w("## C. Haat and sugar-crop share: leave one state out")
w("")
OB = "F_weekly_haat + F_mandi + F_regular_market"
w("Mean sugar share over the 606: {:.2f}%; districts with any sugar crop: {}.".format(
    X.share_sugar.mean() * 100, int((X.share_sugar > 0).sum())))
top = (X.assign(s=X.share_sugar * X.groupby("state_name")["share_sugar"].transform("size"))
       .groupby("state_name")["share_sugar"].agg(["mean", "size"]).sort_values("mean", ascending=False).head(6))
w("States with the highest mean sugar share: " + ", ".join(
    "{} {:.1f}% ({} districts)".format(s, r["mean"] * 100, int(r["size"])) for s, r in top.iterrows()))
SROWS = []
for spec, rhs in [("solo", "F_weekly_haat"), ("output block", OB)]:
    full = full_inf(X, "share_sugar", "F_weekly_haat", rhs)
    full.update(spec=spec, dropped="none")
    SROWS.append(full)
    for st in sorted(X.state_name.unique()):
        sub = X[X.state_name != st]
        f = "share_sugar ~ {} + {}".format(rhs, CTRL)
        m, s = fit_par(sub, f, ["share_sugar"] + rhs.split(" + ") + BASEKEEP)
        SROWS.append(dict(spec=spec, dropped=st, b=m.params["F_weekly_haat"], se=m.bse["F_weekly_haat"],
                          p=m.pvalues["F_weekly_haat"], n=int(m.nobs)))
    for drop in [["MAHARASHTRA", "UTTAR PRADESH"], ["MAHARASHTRA", "UTTAR PRADESH", "UTTARAKHAND"],
                 ["UTTAR PRADESH", "UTTARAKHAND", "GUJARAT"]]:
        sub = X[~X.state_name.isin(drop)]
        r = full_inf(sub, "share_sugar", "F_weekly_haat", rhs)
        r.update(spec=spec, dropped=" + ".join(drop))
        SROWS.append(r)
S = pd.DataFrame(SROWS)
S.to_csv(OUT + "/sugar_leave_one_state_out.csv", index=False)
for spec in ["solo", "output block"]:
    t = S[(S.spec == spec)]
    f0 = t[t.dropped == "none"].iloc[0]
    one = t[~t.dropped.isin(["none"]) & ~t.dropped.str.contains(r"\+")]
    mn, mx = one.loc[one.b.idxmin()], one.loc[one.b.idxmax()]
    w("{}: all states b {:+.4f} (SE {:.4f}, p {:.4f}; state t(G-1) p {:.4f}; wild p {:.4f}). Leave one out: "
      "lowest {:+.4f} without {} (p {:.4f}); highest {:+.4f} without {}; max p {:.4f}.".format(
          spec, f0.b, f0.se, f0.p, f0.p_state, f0.p_wild, mn.b, mn.dropped, mn.p, mx.b, mx.dropped,
          one.p.max()))
    for _, r in t[t.dropped.str.contains(r"\+")].iterrows():
        w("   without {}: b {:+.4f} (SE {:.4f}, p {:.4f}; state p {:.4f}; wild p {:.4f}; n {})".format(
            r.dropped, r.b, r.se, r.p, r.p_state, r.p_wild, int(r.n)))
w("")
w("Leave-one-out table (output block):")
w("| state dropped | b | SE | p |")
w("|---|---|---|---|")
for _, r in S[S.spec == "output block"].sort_values("b").iterrows():
    w("| {} | {:+.4f} | {:.4f} | {:.4f} |".format(r.dropped, r.b, r.se, r.p))

# ================================================================ D farmer collectives
w("")
w("## D. The 'producer organisation' item is a farmer collective: FPO or PACS")
w("")
P11 = pd.read_csv(OUT + "/census_market_district_panel.csv")
for c in DKEY:
    P11[c] = pd.to_numeric(P11[c], errors="coerce")
X = X.merge(P11[DKEY + ["wk11_all", "dl11_all", "acs11_all"]], on=DKEY, how="left")
FR = []
r_acs = X[["F_fpo", "acs11_all"]].corr().iloc[0, 1]
others = {s: X[["F_" + s, "acs11_all"]].corr().iloc[0, 1] for _, s in ns["FAC"] if s != "fpo"}
w("Correlation across the 606 of MA 'farmer collective' reach with Census 2011 agricultural credit "
  "society reach: {:.2f}. Other MA facilities with that census measure: {}.".format(
      r_acs, ", ".join("{} {:.2f}".format(k, v) for k, v in sorted(others.items(), key=lambda kv: -kv[1]))))
w("Mean reach: MA farmer collective {:.3f}; Census 2011 agricultural credit society {:.3f}.".format(
    X.F_fpo.mean(), X.acs11_all.mean()))
for y in ["D1_exp_shannon", "D0_richness"]:
    for term in ["F_fpo", "acs11_all"]:
        r = full_inf(X, y, term)
        r.update(outcome=y, term=term)
        FR.append(r)
        w("{} on {}: b {:+.3f} (SE {:.3f}, p {:.4f}; state p {:.4f}; wild p {:.4f}; n {}).".format(
            term, y, r["b"], r["se"], r["p"], r["p_state"], r["p_wild"], r["n"]))
pd.DataFrame(FR).assign(r_fpo_acs11=r_acs).to_csv(OUT + "/fpo_checks.csv", index=False)

# ================================================================ E census 2011 haat
w("")
w("## E. The haat result with the Census 2011 weekly haat item")
w("")
w("Correlation across the 606: MA haat with census 2011 haat {:.2f}; MA haat with census 2011 "
  "mandi/regular market {:.2f}; census haat with census mandi/regular {:.2f}.".format(
      X[["F_weekly_haat", "wk11_all"]].corr().iloc[0, 1], X[["F_weekly_haat", "dl11_all"]].corr().iloc[0, 1],
      X[["wk11_all", "dl11_all"]].corr().iloc[0, 1]))
ER = []
for y in ["D0", "D0_05", "D1"]:
    for term in ["F_weekly_haat", "wk11_all"]:
        r = full_inf(X, y, term)
        r.update(outcome=y, term=term, sd_x=X[term].std())
        ER.append(r)
        w("{} on {}: b {:+.3f} (SE {:.3f}, p {:.4f}; state p {:.4f}; wild p {:.4f}); one SD ({:.3f}) = {:+.3f}.".format(
            term, y, r["b"], r["se"], r["p"], r["p_state"], r["p_wild"], r["sd_x"], r["b"] * r["sd_x"]))
for y in ["D0", "D0_05", "D1"]:
    for term in ["wk11_all", "dl11_all"]:
        r = full_inf(X, y, term, "wk11_all + dl11_all")
        r.update(outcome=y, term=term + " (joint)", sd_x=X[term].std())
        ER.append(r)
        w("joint census haat + census mandi/regular, {} on {}: b {:+.3f} (SE {:.3f}, p {:.4f}; state p {:.4f}; "
          "wild p {:.4f}).".format(term, y, r["b"], r["se"], r["p"], r["p_state"], r["p_wild"]))
    r = full_inf(X, y, "F_weekly_haat", "F_weekly_haat + wk11_all")
    r.update(outcome=y, term="F_weekly_haat (with census haat)", sd_x=X["F_weekly_haat"].std())
    ER.append(r)
    w("MA haat holding census haat constant, on {}: b {:+.3f} (SE {:.3f}, p {:.4f}; wild p {:.4f}).".format(
        y, r["b"], r["se"], r["p"], r["p_wild"]))
pd.DataFrame(ER).to_csv(OUT + "/haat_census2011_check.csv", index=False)

open(OUT + "/haat_mechanism_log.md", "w", encoding="utf-8").write("\n".join(LOG) + "\n")

# ================================================================ site data
T = pd.DataFrame(TROWS)
NAME = {"Soyabean": "soybean", "Other  Rabi Pulses": "other winter pulses", "Urad": "black gram (urad)",
        "Masoor": "lentil (masoor)", "Horse-Gram": "horse gram", "Arhar/Tur": "pigeon pea (arhar)",
        "Moong(Green Gram)": "green gram (moong)", "Khesari": "grass pea (khesari)",
        "Guar Seed": "guar", "Moth": "moth bean", "Mesta": "mesta",
        "Dry Chillies": "dry chillies"}
site = {
    "sd_haat": sd_haat,
    "thresholds": [dict(key=r.outcome, label=r.label, b=r.b, lo=r.lo, hi=r.hi, p=r.p, p_wild=r.p_wild,
                        mean=r["mean"]) for _, r in T[T.crops == "all"].iterrows()],
    "crops": [dict(crop=r.crop, name=NAME.get(r.crop, r.crop.lower()), type=r.crop_type,
                   p_wild=None if np.isnan(r.p_wild) else r.p_wild, b=r.b, lo=r.lo, hi=r.hi, q=r.q, p=r.p,
                   presence=r.mean_presence, share=r.mean_share_pct, b_share=r.b_share_pp,
                   drop6=bool(r.drop6))
              for _, r in C[C.q < 0.05].iterrows()],
}
json.dump(site, open(REPO + "/docs/data/site_markets_v2.json", "w", encoding="utf-8"),
          indent=1, default=float)
