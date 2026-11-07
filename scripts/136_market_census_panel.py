"""
136_market_census_panel.py

Market panel from the Census village directories, and a first-difference test of
whether districts that gained market reach also gained crops.

What the census directories carry (SHRUG 2.1 variable names and labels):
  1991  pc91_vd_mrkt_fac "Market Facility"; pc91_vd_mrkt_day_1..7 "Market Day Is
        Monday".."Sunday", pc91_vd_mrkt_day_d "Market Is Daily", _f fortnightly, _m monthly
  2001  NO market item. Only credit institutions: pc01_vd_ac_soc "Number Of Agricultural
        Credit Societies", pc01_vd_crsoc_fac "Credit Societies"
  2011  pc11_vd_wkl_haat "Weekly Haat"; pc11_vd_mrkt "Mandis/Regular Market";
        pc11_vd_acs "Agricultural Credit Societies"; pc11_vd_ams "Agricultural Marketing Society"

So the 2001-2011 panel exists only for agricultural credit societies. For haats and
markets the nearest pair is 1991-2011:
  weekly market   1991: a market on a named weekday (any of day_1..7 > 0)   2011: weekly haat
  daily market    1991: market is daily (day_d > 0)                          2011: mandi or regular market
  any market      1991: market facility                                      2011: haat or mandi/regular market

Reach = share of the district's Census 2011 rural population (pc11_pca_tot_p, fixed
weights in both rounds) living in a village that reports the facility. A village enters
a pair only if both rounds report the item (balanced village set), so a change in reach
is a change in facilities, not in coverage. 2011 districts with less than 75 percent of
their 2011 rural population in the balanced set are dropped from that pair.

Outcomes: change in district means between 1998-99..2002-03 and 2009-10..2013-14
(div_windows.csv, windows e9802 and cs0913, from script 122; at least 3 of 5 years in
each): count of crops D0, D0 at the 0.5 percent floor, effective number D1; variants
all crops and drop6 (six crops lost in the 2020-21 crop-list break removed in every year).

Model: dY = b dReach + c dIrrigation(census 2001->2011) + state FE, crop-record districts
in the 606 analysis set, facility of the 2011 parent district, SE clustered on the 2011
parent; state clustering with t(G-1) and a restricted wild cluster bootstrap by state
(Webb, 9,999 draws) as robustness.

Outputs (outputs/revision_nov8/markets_v2/):
  census_market_district_panel.csv, census_market_panel_fd.csv,
  census_market_comparability.csv, census_market_panel_log.md (raw log)

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/136_market_census_panel.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _mkt2_common import REPO, fit_par, wcr  # noqa: E402

warnings.filterwarnings("ignore")
SHRUG = r"D:/SHRUG_2.1_Data/extracted"
OUT = REPO + "/outputs/revision_nov8/markets_v2"
os.makedirs(OUT, exist_ok=True)
DKEY = ["pc11_state_id", "pc11_district_id"]
LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


# ================================================================ villages
key = pd.read_stata(SHRUG + "/shrug-pc-keys-dta/shrid_pc11dist_key.dta")
key = key.dropna(subset=DKEY).drop_duplicates("shrid2")
for c in DKEY:
    key[c] = pd.to_numeric(key[c].astype(str), errors="coerce")
pop = pd.read_stata(SHRUG + "/shrug-pca11-dta/pc11_pca_clean_shrid.dta",
                    columns=["shrid2", "pc11_pca_tot_p"])
DAYS = ["pc91_vd_mrkt_day_" + s for s in "1234567"]
v91 = pd.read_stata(SHRUG + "/shrug-vd91-dta/pc91_vd_clean_shrid.dta",
                    columns=["shrid2", "pc91_vd_mrkt_fac", "pc91_vd_mrkt_day_d",
                             "pc91_vd_mrkt_day_f", "pc91_vd_mrkt_day_m"] + DAYS)
v01 = pd.read_stata(SHRUG + "/shrug-vd01-dta/pc01_vd_clean_shrid.dta",
                    columns=["shrid2", "pc01_vd_ac_soc", "pc01_vd_crsoc_fac"])
v11 = pd.read_stata(SHRUG + "/shrug-vd11-dta/pc11_vd_clean_shrid.dta",
                    columns=["shrid2", "pc11_vd_wkl_haat", "pc11_vd_mrkt", "pc11_vd_acs",
                             "pc11_vd_ams"])
ma = pd.read_stata(SHRUG + "/shrug-antyodaya-dta/antyodaya_shrid.dta",
                   columns=["shrid2", "weekly_haat", "mandi", "regular_market"])

w("# Census market panel: raw log")
w("")
w("## Villages")
w("")
w("SHRUG village rows: 1991 {:,}; 2001 {:,}; 2011 {:,}; Mission Antyodaya {:,}.".format(
    len(v91), len(v01), len(v11), len(ma)))

# shrids are aggregates of villages, so 0/1 items can be fractional; >0 means present
v91["wk91"] = (v91[DAYS].fillna(0).gt(0).any(axis=1)).astype(float)
v91.loc[v91[DAYS].isna().all(axis=1), "wk91"] = np.nan
v91["dl91"] = (v91["pc91_vd_mrkt_day_d"] > 0).astype(float).where(v91["pc91_vd_mrkt_day_d"].notna())
v91["any91"] = (v91["pc91_vd_mrkt_fac"] > 0).astype(float).where(v91["pc91_vd_mrkt_fac"].notna())
v01["acs01"] = (v01["pc01_vd_ac_soc"] > 0).astype(float).where(v01["pc01_vd_ac_soc"].notna())
v11["wk11"] = (v11["pc11_vd_wkl_haat"] > 0).astype(float).where(v11["pc11_vd_wkl_haat"].notna())
v11["dl11"] = (v11["pc11_vd_mrkt"] > 0).astype(float).where(v11["pc11_vd_mrkt"].notna())
v11["any11"] = ((v11["pc11_vd_wkl_haat"] > 0) | (v11["pc11_vd_mrkt"] > 0)).astype(float).where(
    v11[["pc11_vd_wkl_haat", "pc11_vd_mrkt"]].notna().all(axis=1))
v11["acs11"] = (v11["pc11_vd_acs"] > 0).astype(float).where(v11["pc11_vd_acs"].notna())
ma["wkMA"] = (ma["weekly_haat"] > 0).astype(float)
ma["dlMA"] = ((ma["mandi"] > 0) | (ma["regular_market"] > 0)).astype(float)

# 1991 internal consistency
m1 = v91.dropna(subset=["any91", "wk91"])
w("1991 villages with both the market flag and day fields: {:,}. Market flag = 1: {:.1%}. "
  "Of those, a weekday market day: {:.1%}; daily: {:.1%}; fortnightly or monthly only: {:.1%}; "
  "no frequency at all: {:.1%}. Weekday day recorded where flag = 0: {:,} villages.".format(
      len(m1), m1["any91"].mean(),
      m1.loc[m1.any91 == 1, "wk91"].mean(), (m1.loc[m1.any91 == 1, "pc91_vd_mrkt_day_d"] > 0).mean(),
      ((m1.any91 == 1) & (m1.wk91 == 0) & (m1.pc91_vd_mrkt_day_d.fillna(0) == 0) &
       ((m1.pc91_vd_mrkt_day_f.fillna(0) > 0) | (m1.pc91_vd_mrkt_day_m.fillna(0) > 0))).sum() /
      max(1, (m1.any91 == 1).sum()),
      ((m1.any91 == 1) & (m1.wk91 == 0) & (m1.pc91_vd_mrkt_day_d.fillna(0) == 0) &
       (m1.pc91_vd_mrkt_day_f.fillna(0) == 0) & (m1.pc91_vd_mrkt_day_m.fillna(0) == 0)).sum() /
      max(1, (m1.any91 == 1).sum()),
      int(((m1.any91 == 0) & (m1.wk91 == 1)).sum())))
w("1991 day fields missing while the flag is present: {:,} villages ({:.1%} of flagged rows).".format(
    int(v91["pc91_vd_mrkt_fac"].notna().sum() - len(m1)),
    1 - len(m1) / v91["pc91_vd_mrkt_fac"].notna().sum()))
x11 = v11.dropna(subset=["wk11", "dl11"])
w("2011: weekly haat {:.1%} of villages, mandi or regular market {:.1%}, both {:.1%}.".format(
    x11.wk11.mean(), x11.dl11.mean(), ((x11.wk11 == 1) & (x11.dl11 == 1)).mean()))
w("Mission Antyodaya 2019-20: the market item is ONE answer per village (mandi 1, regular "
  "market 2, weekly haat 3, none 4; question 71 of the MA 2020 data mapping), so a village "
  "never carries two market types. Villages with more than one market type in the SHRUG file: "
  "{:,}.".format(int(((ma.weekly_haat > 0).astype(int) + (ma.mandi > 0).astype(int) +
                      (ma.regular_market > 0).astype(int) > 1).sum())))

V = (key.merge(pop, on="shrid2", how="left")
     .merge(v91[["shrid2", "wk91", "dl91", "any91"]], on="shrid2", how="left")
     .merge(v01[["shrid2", "acs01"]], on="shrid2", how="left")
     .merge(v11[["shrid2", "wk11", "dl11", "any11", "acs11"]], on="shrid2", how="left")
     .merge(ma[["shrid2", "wkMA", "dlMA"]], on="shrid2", how="left"))
V["pw"] = V["pc11_pca_tot_p"].where(V["pc11_pca_tot_p"] > 0, 0).fillna(0)
V = V[V[["wk11", "acs11"]].notna().any(axis=1)]          # rural villages in the 2011 directory
w("Villages in the 2011 directory with a district key: {:,}; 2011 rural population {:.0f} million.".format(
    len(V), V.pw.sum() / 1e6))

PAIRS = {  # name: (early column, late column)
    "haat": ("wk91", "wk11"), "daily": ("dl91", "dl11"), "anymkt": ("any91", "any11"),
    "acs": ("acs01", "acs11"), "haatMA": ("wk11", "wkMA"), "dailyMA": ("dl11", "dlMA")}
g = V.groupby(DKEY)
tot = g["pw"].sum()
D = pd.DataFrame(index=tot.index)
D["pop11"] = tot
for nm, (a, b) in PAIRS.items():
    ok = V[a].notna() & V[b].notna()
    pw = V["pw"].where(ok, 0)
    den = pw.groupby([V[k] for k in DKEY]).sum()
    D["cov_" + nm] = den / tot
    D[nm + "_e"] = (V[a].fillna(0) * pw).groupby([V[k] for k in DKEY]).sum() / den
    D[nm + "_l"] = (V[b].fillna(0) * pw).groupby([V[k] for k in DKEY]).sum() / den
    D.loc[D["cov_" + nm] < 0.75, [nm + "_e", nm + "_l"]] = np.nan
    D["d_" + nm] = D[nm + "_l"] - D[nm + "_e"]
# single-round 2011 reach over every 2011 village (for cross-sectional checks in 137)
for c in ["wk11", "dl11", "acs11"]:
    ok = V[c].notna()
    pw = V["pw"].where(ok, 0)
    D[c + "_all"] = ((V[c].fillna(0) * pw).groupby([V[k] for k in DKEY]).sum() /
                     pw.groupby([V[k] for k in DKEY]).sum())
D = D.reset_index()

# ================================================================ comparability table
w("")
w("## Comparability of the measures (district level, 2011 population weights)")
w("")
w("| pair | districts with >= 75% coverage | mean early | mean late | mean change | SD change | r(early, late) |")
w("|---|---|---|---|---|---|---|")
CMP = []
for nm in PAIRS:
    t = D.dropna(subset=[nm + "_e", nm + "_l"])
    r = dict(pair=nm, n=len(t), mean_early=t[nm + "_e"].mean(), mean_late=t[nm + "_l"].mean(),
             mean_change=t["d_" + nm].mean(), sd_change=t["d_" + nm].std(),
             r_early_late=t[nm + "_e"].corr(t[nm + "_l"]),
             cov_median=D["cov_" + nm].median())
    CMP.append(r)
    w("| {pair} | {n} | {mean_early:.3f} | {mean_late:.3f} | {mean_change:+.3f} | {sd_change:.3f} | {r_early_late:.2f} |".format(**r))
pd.DataFrame(CMP).to_csv(OUT + "/census_market_comparability.csv", index=False)

# state-level coverage of the 1991 market items
st_names = pd.read_csv(REPO + "/outputs/shrug_covariates/final_panel.csv",
                       usecols=DKEY + ["state_name"]).drop_duplicates(DKEY)
for c in DKEY:
    st_names[c] = pd.to_numeric(st_names[c], errors="coerce")
Ds = D.merge(st_names, on=DKEY, how="left")
cv = Ds.groupby("state_name")[["cov_haat", "cov_acs"]].median().sort_values("cov_haat")
w("")
w("States with median 1991-2011 market coverage below 0.75: " + ", ".join(
    "{} {:.2f}".format(s, v) for s, v in cv["cov_haat"].items() if v < 0.75))
w("States with median 2001-2011 credit-society coverage below 0.75: " + ", ".join(
    "{} {:.2f}".format(s, v) for s, v in cv["cov_acs"].items() if v < 0.75))
D.to_csv(OUT + "/census_market_district_panel.csv", index=False)

# ================================================================ outcomes and controls
fp = pd.read_csv(REPO + "/outputs/shrug_covariates/final_panel.csv")
for c in DKEY:
    fp[c] = pd.to_numeric(fp[c], errors="coerce")
fp = fp[fp["in_final"]][["district_key", "state_name", "shrug_key"] + DKEY]
DV = pd.read_csv(REPO + "/outputs/revision_nov8/irrigation_v2/div_windows.csv")
Y = ["D0", "D0_05", "D1"]
dd = []
for v in ["all", "drop6"]:
    e = DV[(DV.window == "e9802") & (DV.variant == v)].set_index("district_key")[Y]
    l_ = DV[(DV.window == "cs0913") & (DV.variant == v)].set_index("district_key")[Y]
    ch = (l_ - e).dropna()
    ch.columns = ["d" + c + ("" if v == "all" else "_d6") for c in Y]
    dd.append(ch)
    if v == "all":
        e.columns = [c + "_early" for c in Y]
        dd.append(e)
dd = pd.concat(dd, axis=1).reset_index()

ci = pd.read_csv(REPO + "/outputs/revision_nov8/census_irrigation/census_irrigation_panel_wide.csv")
ci["d_irr0111"] = (ci["irr_share_2011"] - ci["irr_share_2001"]).where(
    ci["district_usable_2001"] & ci["district_usable_2011"])
ci["d_irr9111"] = (ci["irr_share_2011"] - ci["irr_share_1991"]).where(
    ci["district_usable_1991"] & ci["district_usable_2011"])
ci = ci[DKEY + ["d_irr0111", "d_irr9111", "irr_share_2001"]]

P = (fp.merge(D, on=DKEY, how="left").merge(ci, on=DKEY, how="left")
     .merge(dd, on="district_key", how="left"))
w("")
w("## Analysis frame")
w("")
w("Crop-record districts in the 606: {}. With both crop windows: {}. With the 1991-2011 haat "
  "change: {}. With the 2001-2011 credit-society change: {}. With the census irrigation change "
  "2001-2011: {}.".format(len(P), P["dD1"].notna().sum(), P["d_haat"].notna().sum(),
                          P["d_acs"].notna().sum(), P["d_irr0111"].notna().sum()))
w("Mean early-window values: D0 {:.2f}, D0 at 0.5% {:.2f}, D1 {:.3f}. Mean changes: dD0 {:+.2f}, "
  "dD0_05 {:+.2f}, dD1 {:+.3f}.".format(P.D0_early.mean(), P.D0_05_early.mean(), P.D1_early.mean(),
                                         P.dD0.mean(), P.dD0_05.mean(), P.dD1.mean()))

# ================================================================ regressions
w("")
w("## First-difference regressions (state FE; SE on 2011 parent; state-clustered t(G-1); "
  "wild cluster bootstrap by state, Webb, 9,999 draws)")
w("")
SPECS = [  # tag, regressors, extra controls
    ("haat + daily", ["d_haat", "d_daily"], "d_irr0111"),
    ("haat only", ["d_haat"], "d_irr0111"),
    ("any market", ["d_anymkt"], "d_irr0111"),
    ("credit societies", ["d_acs"], "d_irr0111"),
    ("haat + daily + credit", ["d_haat", "d_daily", "d_acs"], "d_irr0111"),
    ("haat + daily, irrigation 1991-2011", ["d_haat", "d_daily"], "d_irr9111"),
    ("haat + daily, with 1991 levels", ["d_haat", "d_daily"], "d_irr0111 + haat_e + daily_e"),
    ("haat + daily, no irrigation control", ["d_haat", "d_daily"], ""),
]
OUTS = [("dD0", "count of crops"), ("dD0_05", "count at 0.5% floor"), ("dD1", "effective number"),
        ("dD0_d6", "count, drop6"), ("dD0_05_d6", "count 0.5%, drop6"), ("dD1_d6", "effective, drop6")]
ROWS = []
w("| spec | outcome | term | n | b | SE parent | p parent | p state t(G-1) | p wild | G |")
w("|---|---|---|---|---|---|---|---|---|---|")
for tag, regs, ext in SPECS:
    for y, ylab in OUTS:
        rhs = " + ".join(regs + ([ext] if ext else []) + ["C(state_name)"])
        f = "{} ~ {}".format(y, rhs)
        keep = [y, "shrug_key", "state_name"] + regs + [v.strip() for v in ext.split("+") if v.strip()]
        m, s = fit_par(P, f, keep)
        for t in regs:
            wb = wcr(P, f, t, keep)
            r = dict(spec=tag, outcome=y, outcome_label=ylab, term=t, n=int(m.nobs),
                     n_states=s.state_name.nunique(), b=m.params[t], se=m.bse[t],
                     p=m.pvalues[t], lo=m.conf_int().loc[t, 0], hi=m.conf_int().loc[t, 1],
                     se_state=wb["se_st"], p_state=wb["p_st"], p_wild=wb["p_wcr"], G=wb["G"],
                     sd_x=s[t].std())
            ROWS.append(r)
            w("| {spec} | {outcome_label} | {term} | {n} | {b:+.3f} | {se:.3f} | {p:.3f} | "
              "{p_state:.3f} | {p_wild:.3f} | {G} |".format(**r))
R = pd.DataFrame(ROWS)
R.to_csv(OUT + "/census_market_panel_fd.csv", index=False)

# one-SD effects for the main spec
w("")
w("One-SD effects, main spec (haat + daily, all crops):")
for _, r in R[(R.spec == "haat + daily") & R.outcome.isin(["dD0", "dD0_05", "dD1"])].iterrows():
    w("  {} on {}: SD of change {:.3f}; one SD -> {:+.3f} (95% CI {:+.3f} to {:+.3f})".format(
        r.term, r.outcome, r.sd_x, r.b * r.sd_x, r.lo * r.sd_x, r.hi * r.sd_x))

# ================================================================ cross-section with census 2011 haat
w("")
w("## Cross-check: the site's cross-section with the CENSUS 2011 haat in place of Mission Antyodaya")
w("")
w("Correlation across the 606 of 2011 census haat reach with MA 2019-20 haat reach "
  "(same village set): {:.2f}; census 2011 mandi/regular with MA mandi or regular: {:.2f}.".format(
      P["haatMA_e"].corr(P["haatMA_l"]), P["dailyMA_e"].corr(P["dailyMA_l"])))
open(OUT + "/census_market_panel_log.md", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
P.to_csv(OUT + "/census_market_analysis_frame.csv", index=False)
