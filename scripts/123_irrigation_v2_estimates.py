"""
123_irrigation_v2_estimates.py

Questions 1-4 of the irrigation v2 note (outputs/revision_nov8/irrigation_v2/).
Diversity windows come from 122_irrigation_v2_diversity.py; the census source panel from
121_census_irrigation_panel.py; market regressors (population shares, version W) are
rebuilt by running the data-construction part of 104_market_pop_weighted.py unchanged.

Every coefficient carries two standard errors: clustered on 2011 parent district
(shrug_key, the site's choice) and clustered on state. Within-state = state fixed effects.

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/123_irrigation_v2_estimates.py
"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
REPO = r"D:/crop-diversity"
OUT = REPO + "/outputs/revision_nov8/irrigation_v2"
CEN = REPO + "/outputs/revision_nov8/census_irrigation/census_irrigation_panel.csv"
DK = ["pc11_state_id", "pc11_district_id"]
LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


# ================================================================ data
fp = pd.read_csv(REPO + "/outputs/shrug_covariates/final_panel.csv")
for c in DK:
    fp[c] = pd.to_numeric(fp[c], errors="coerce")
d0 = fp[fp["in_final"] == True].copy()
assert len(d0) == 606
d0["st_id"] = pd.factorize(d0["state_name"])[0]
DV = pd.read_csv(OUT + "/div_windows.csv")
YV = ["D1", "D0", "D0_05", "D2", "E"]
YLAB = {"D1": "D1 effective number", "D0": "D0 count", "D0_05": "D0 at 0.5%", "D2": "D2",
        "E": "evenness", "share_cereals": "cereal share", "share_pulses": "pulse share"}


def win(tag, variant):
    t = DV[(DV["window"] == tag) & (DV["variant"] == variant)].set_index("district_key")
    return t[YV + ["share_cereals", "share_pulses", "gca", "n_years"]]


# census, wide with composition shares
cl = pd.read_csv(CEN)
for c in DK:
    cl[c] = pd.to_numeric(cl[c], errors="coerce")
cv = ["irr_share", "sh_canal", "sh_groundwater", "sh_gw_elec", "sh_tank_lake", "sh_river_other",
      "comp_canal", "comp_groundwater", "comp_gw_elec", "comp_tank_lake", "comp_river_other",
      "district_usable"]
cw = cl.pivot_table(index=DK, columns="round", values=cv, aggfunc="first")
cw.columns = ["c{}_{}".format(b, a) for a, b in cw.columns]
cw = cw.reset_index()
ter = cl.drop_duplicates(DK)[DK + ["elevation_mean_m", "tri_mean"]]
cw = cw.merge(ter, on=DK)
d0 = d0.merge(cw, on=DK, how="left")
assert len(d0) == 606
d0["elev_km"] = d0["elevation_mean_m"] / 1000
d0["tri"] = d0["tri_mean"]
for r in (1991, 2001, 2011):
    d0["c%d_district_usable" % r] = d0["c%d_district_usable" % r].astype(object).eq(True)
TER = " + elev_km + tri"
w("# Irrigation v2: estimates")
w("")
w("Site sample: 606 districts, {} 2011 parents, {} states. Census district usable: 1991 {}, "
  "2001 {}, 2011 {}. Terrain present: {}.".format(
      d0["shrug_key"].nunique(), d0["state_name"].nunique(),
      int(d0["c1991_district_usable"].sum()), int(d0["c2001_district_usable"].sum()),
      int(d0["c2011_district_usable"].sum()), int(d0["elev_km"].notna().sum())))

ROWS = []


def fit(df, f, keep):
    """OLS with parent and state clustered SEs; returns models and sample."""
    vars_ = [v for v in keep if v in df]
    s = df.dropna(subset=vars_).copy()
    m1 = smf.ols(f, s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s["shrug_key"])[0]})
    m2 = smf.ols(f, s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s["state_name"])[0]})
    return m1, m2, s


def peak(m, x, x2):
    b1, b2 = m.params[x], m.params[x2]
    V = m.cov_params().loc[[x, x2], [x, x2]].values
    tp = -b1 / (2 * b2)
    g = np.array([-1 / (2 * b2), b1 / (2 * b2 ** 2)])
    return tp, float(np.sqrt(g @ V @ g))


def quad(df, y, x, extra="", fe=True, q="", tag="", **meta):
    x2 = "I({} ** 2)".format(x)
    f = "{} ~ {} + I({}**2){}{}".format(y, x, x, extra, " + C(state_name)" if fe else "")
    keep = [y, x, "shrug_key"] + [v.strip() for v in extra.replace("+", " ").split()
                                  if v.strip().isidentifier()]
    m1, m2, s = fit(df, f, keep)
    tp, tse1 = peak(m1, x, x2)
    _, tse2 = peak(m2, x, x2)
    r = dict(q=q, tag=tag, y=y, x=x, n=int(m1.nobs), n_par=s["shrug_key"].nunique(),
             n_st=s["state_name"].nunique(),
             b1=m1.params[x], b1_se_par=m1.bse[x], b1_se_st=m2.bse[x],
             b2=m1.params[x2], b2_se_par=m1.bse[x2], b2_se_st=m2.bse[x2],
             b2_p_par=m1.pvalues[x2], b2_p_st=m2.pvalues[x2],
             peak=tp if m1.params[x2] < 0 else np.nan,
             peak_se_par=tse1 if m1.params[x2] < 0 else np.nan,
             peak_se_st=tse2 if m1.params[x2] < 0 else np.nan, **meta)
    ROWS.append(r)
    return r, m1, m2


def lin(df, y, xs, extra="", q="", tag="", fe=True, **meta):
    f = "{} ~ {}{}{}".format(y, " + ".join(xs), extra, " + C(state_name)" if fe else "")
    keep = [y, "shrug_key"] + [v for v in xs if v.isidentifier()] + \
           [v.strip() for v in extra.replace("+", " ").split() if v.strip().isidentifier()]
    m1, m2, s = fit(df, f, keep)
    out = []
    for x in xs:
        r = dict(q=q, tag=tag, y=y, x=x, n=int(m1.nobs), n_par=s["shrug_key"].nunique(),
                 n_st=s["state_name"].nunique(), b1=m1.params[x], b1_se_par=m1.bse[x],
                 b1_se_st=m2.bse[x], b1_p_par=m1.pvalues[x], b1_p_st=m2.pvalues[x], **meta)
        ROWS.append(r)
        out.append(r)
    return out, m1, m2


def st(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def fmt_quad(r):
    pk = "{:.2f} ({:.2f}; {:.2f})".format(r["peak"], r["peak_se_par"], r["peak_se_st"]) \
        if pd.notna(r["peak"]) else "none (b2 > 0)"
    return "| {} | {} | {:.3f} ({:.3f}; {:.3f}) | {:.3f} ({:.3f}; {:.3f}){} | {:.3f} / {:.3f} | {} |".format(
        r["tag"], r["n"], r["b1"], r["b1_se_par"], r["b1_se_st"], r["b2"], r["b2_se_par"],
        r["b2_se_st"], st(r["b2_p_par"]), r["b2_p_par"], r["b2_p_st"], pk)


QH = ("| sample / window | n | linear (SE parent; SE state) | squared (SE parent; SE state) | "
      "p squared, parent / state | peak (SE parent; SE state) |")
QS = "|---|---|---|---|---|---|"


def with_win(base, tag, variant):
    t = win(tag, variant)
    x = base.drop(columns=[c for c in YV + ["share_cereals", "share_pulses"] if c in base])
    return x.merge(t, left_on="district_key", right_index=True, how="left")


# ================================================================ Q1 irrigation bend
w("")
w("## Q1. Recent window: irrigation bend (Antyodaya 2019-20 share, state FE)")
WINS = [("(c) full 1998-99 to 2019-20, all crops [site]", "full", "all"),
        ("(c) full, six crops removed", "full", "drop6"),
        ("(a) 2015-16 to 2019-20, all crops", "w1517", "all"),
        ("(a) 2015-16 to 2019-20, six crops removed", "w1517", "drop6"),
        ("(a) 2015-16 to 2019-20, state-stable crop list", "w1517", "stable"),
        ("(b) 2017-18 to 2021-22, raw (no flags, all crops)", "w1721", "all"),
        ("(b) 2017-18 to 2021-22, flags dropped, all crops", "w1721", "flags"),
        ("(b) 2017-18 to 2021-22, flags dropped, six crops removed [main b]", "w1721", "flags_drop6"),
        ("(b) 2017-18 to 2021-22, flags dropped, state-stable crop list", "w1721", "flags_stable")]
DW = {}
for lab, t, v in WINS:
    DW[(t, v)] = with_win(d0, t, v)
# check: full/all reproduces the site
r, _, _ = quad(DW[("full", "all")], "D1", "irr_share", q="chk", tag="site check")
ref = pd.read_csv(REPO + "/outputs/shrug_covariates/final_robustness.csv").set_index("spec")
assert abs(r["b2"] - ref.loc["D1: state fixed effects", "b2"]) < 1e-9
ROWS.pop()
w("Check: full-period D1 reproduces the site's within-state b2 ({:.4f}).".format(r["b2"]))
# common sample: districts with every window
common = set(d0["district_key"])
for k, x in DW.items():
    common &= set(x.dropna(subset=["D1"])["district_key"])
w("Districts present in every window: {} of 606.".format(len(common)))
for y in YV:
    w("")
    w("### {}".format(YLAB[y]))
    w("")
    w(QH)
    w(QS)
    for lab, t, v in WINS:
        r, _, _ = quad(DW[(t, v)], y, "irr_share", q="Q1", tag=lab, window=t, variant=v, sample="own")
        w(fmt_quad(r))
    for lab, t, v in [WINS[0], WINS[2], WINS[7]]:
        x = DW[(t, v)][DW[(t, v)]["district_key"].isin(common)]
        r, _, _ = quad(x, y, "irr_share", q="Q1", tag=lab + ", common sample", window=t,
                       variant=v, sample="common")
        w(fmt_quad(r))

# ================================================================ Q1 markets
w("")
w("## Q1. Recent window: market headline results (population shares, all controls, state FE)")
w("")
src = open(REPO + "/scripts/104_market_pop_weighted.py", encoding="utf-8").read()
cut = src.index("VERS = [")
ns = {"__name__": "m104"}
exec(compile(src[:cut], "104_head", "exec"), ns)        # data construction only, writes nothing
XW = ns["prep"]("W")
CTRL = ns["CTRL"]
XW = XW[["district_key", "shrug_key", "state_name", "mean_holding_ha", "log_viirs",
         "estab_per_1000pop", "idx_connectivity_use", "pca_st_share", "irr_share"] +
        ["F_" + s for _, s in ns["FAC"]]]
MK = [("weekly haat", "weekly_haat", "D0"), ("weekly haat", "weekly_haat", "D1"),
      ("FPO", "fpo", "D1"), ("FPO", "fpo", "D0"),
      ("fertiliser shop", "fert_shop", "share_pulses"), ("mandi", "mandi", "share_cereals")]
coef = pd.read_csv(REPO + "/outputs/revision_nov8/market_pw_coefs.csv")
SITE = {"D0": "D0_richness", "D1": "D1_exp_shannon", "share_pulses": "share_pulses",
        "share_cereals": "share_cereals"}
MROWS = []
w("| facility | outcome | window | n | b | SE parent | SE state | p parent / state |")
w("|---|---|---|---|---|---|---|---|")
for lab, s, y in MK:
    for wl, t, v in [WINS[0], WINS[2], WINS[3], WINS[5], WINS[7], WINS[8]]:
        x = XW.merge(win(t, v), left_on="district_key", right_index=True, how="left")
        f = "{} ~ F_{} + {}".format(y, s, CTRL)
        m1, m2, smp = fit(x, f, [y, "F_" + s, "mean_holding_ha", "log_viirs", "shrug_key"])
        k = "F_" + s
        if (t, v) == ("full", "all"):
            ref_b = coef[(coef.block == "solo") & (coef.version == "W") & (coef.facility == s) &
                         (coef.outcome == SITE[y])]["b"].iloc[0]
            assert abs(ref_b - m1.params[k]) < 1e-6, (s, y, ref_b, m1.params[k])
        rr = dict(q="Q1m", tag=wl, facility=lab, y=y, window=t, variant=v, n=int(m1.nobs),
                  b1=m1.params[k], b1_se_par=m1.bse[k], b1_se_st=m2.bse[k],
                  b1_p_par=m1.pvalues[k], b1_p_st=m2.pvalues[k])
        MROWS.append(rr)
        w("| {} | {} | {} | {} | {:.3f}{} | {:.3f} | {:.3f} | {:.3f} / {:.3f} |".format(
            lab, YLAB[y], wl, rr["n"], rr["b1"], st(rr["b1_p_par"]), rr["b1_se_par"],
            rr["b1_se_st"], rr["b1_p_par"], rr["b1_p_st"]))
w("")
w("Check: the full-period, all-crops rows reproduce market_pw_coefs.csv (solo, version W) to 1e-6.")

# ================================================================ Q2 canal vs groundwater
w("")
w("## Q2. Canal or groundwater: 2011 census sources, diversity 2009-10 to 2013-14")
c2 = with_win(d0, "cs0913", "all")
c2 = c2[c2["c2011_district_usable"]].copy()
c2["irr"] = c2["c2011_irr_share"]
c2["gwc"] = c2["c2011_comp_groundwater"]
c2["canc"] = c2["c2011_comp_canal"]
c2["sh_can"] = c2["c2011_sh_canal"]
c2["sh_gw"] = c2["c2011_sh_groundwater"]
c2["sh_tk"] = c2["c2011_sh_tank_lake"]
c2["sh_ro"] = c2["c2011_sh_river_other"]
w("")
w("Usable 2011 census districts in the site sample with 2009-13 diversity: {}. Mean irrigated share "
  "{:.3f}; mean groundwater share of irrigated area {:.3f}, canal {:.3f}.".format(
      int(c2["D1"].notna().sum()), c2["irr"].mean(), c2["gwc"].mean(), c2["canc"].mean()))
Q2 = []
for y in ["D1", "D0", "D0_05", "E"]:
    w("")
    w("### {}".format(YLAB[y]))
    w("")
    w(QH)
    w(QS)
    r, _, _ = quad(c2, y, "irr", TER, q="Q2", tag="A total share, quadratic", spec="A")
    w(fmt_quad(r))
    r, m1, m2 = quad(c2, y, "irr", " + gwc" + TER, q="Q2", tag="B + groundwater share of irrigated", spec="B")
    w(fmt_quad(r))
    w("")
    w("Spec B, groundwater share of irrigated area: {:.3f} (SE {:.3f} parent; {:.3f} state), p {:.3f} / {:.3f}.".format(
        m1.params["gwc"], m1.bse["gwc"], m2.bse["gwc"], m1.pvalues["gwc"], m2.pvalues["gwc"]))
    ROWS.append(dict(q="Q2", tag="B gw share coefficient", y=y, x="gwc", n=int(m1.nobs),
                     b1=m1.params["gwc"], b1_se_par=m1.bse["gwc"], b1_se_st=m2.bse["gwc"],
                     b1_p_par=m1.pvalues["gwc"], b1_p_st=m2.pvalues["gwc"], spec="B"))
    # C: curvature by source mix
    f = ("{} ~ irr + I(irr**2) + gwc + irr:gwc + I(irr**2):gwc{} + C(state_name)").format(y, TER)
    m1, m2, s = fit(c2, f, [y, "irr", "gwc", "elev_km", "tri", "shrug_key"])
    b2c, b2g = m1.params["I(irr ** 2)"], m1.params["I(irr ** 2)"] + m1.params["I(irr ** 2):gwc"]
    # SE of b2 at gwc = 1
    def se_comb(m, a, b):
        V = m.cov_params()
        return float(np.sqrt(V.loc[a, a] + V.loc[b, b] + 2 * V.loc[a, b]))
    sg1 = se_comb(m1, "I(irr ** 2)", "I(irr ** 2):gwc")
    sg2 = se_comb(m2, "I(irr ** 2)", "I(irr ** 2):gwc")
    w("")
    w("Spec C, curvature by source mix (n {}): squared term in an all-canal-and-other district "
      "(groundwater share 0) {:.3f} (SE {:.3f}; {:.3f}); in an all-groundwater district {:.3f} "
      "(SE {:.3f}; {:.3f}); difference {:.3f}, p {:.3f} parent / {:.3f} state.".format(
          int(m1.nobs), b2c, m1.bse["I(irr ** 2)"], m2.bse["I(irr ** 2)"], b2g, sg1, sg2,
          m1.params["I(irr ** 2):gwc"], m1.pvalues["I(irr ** 2):gwc"], m2.pvalues["I(irr ** 2):gwc"]))
    ROWS.append(dict(q="Q2", tag="C b2 at gw=0", y=y, x="irr^2", n=int(m1.nobs), b2=b2c,
                     b2_se_par=m1.bse["I(irr ** 2)"], b2_se_st=m2.bse["I(irr ** 2)"],
                     b2_p_par=m1.pvalues["I(irr ** 2)"], b2_p_st=m2.pvalues["I(irr ** 2)"], spec="C"))
    ROWS.append(dict(q="Q2", tag="C b2 at gw=1", y=y, x="irr^2", n=int(m1.nobs), b2=b2g,
                     b2_se_par=sg1, b2_se_st=sg2, spec="C"))
    ROWS.append(dict(q="Q2", tag="C interaction irr^2 x gw", y=y, x="irr^2:gwc", n=int(m1.nobs),
                     b2=m1.params["I(irr ** 2):gwc"], b2_se_par=m1.bse["I(irr ** 2):gwc"],
                     b2_se_st=m2.bse["I(irr ** 2):gwc"], b2_p_par=m1.pvalues["I(irr ** 2):gwc"],
                     b2_p_st=m2.pvalues["I(irr ** 2):gwc"], spec="C"))
    # D: separate source shares with squares
    f = ("{} ~ sh_can + I(sh_can**2) + sh_gw + I(sh_gw**2) + sh_tk + sh_ro{} + C(state_name)").format(y, TER)
    m1, m2, s = fit(c2, f, [y, "sh_can", "sh_gw", "sh_tk", "sh_ro", "elev_km", "tri", "shrug_key"])
    w("")
    w("Spec D, separate source shares of sown area (n {}):".format(int(m1.nobs)))
    w("")
    w("| term | b | SE parent | SE state | p parent / state |")
    w("|---|---|---|---|---|")
    for k in ["sh_can", "I(sh_can ** 2)", "sh_gw", "I(sh_gw ** 2)", "sh_tk", "sh_ro"]:
        w("| {} | {:.3f}{} | {:.3f} | {:.3f} | {:.3f} / {:.3f} |".format(
            k, m1.params[k], st(m1.pvalues[k]), m1.bse[k], m2.bse[k], m1.pvalues[k], m2.pvalues[k]))
        ROWS.append(dict(q="Q2", tag="D " + k, y=y, x=k, n=int(m1.nobs), b1=m1.params[k],
                         b1_se_par=m1.bse[k], b1_se_st=m2.bse[k], b1_p_par=m1.pvalues[k],
                         b1_p_st=m2.pvalues[k], spec="D"))
    for k, kk in [("sh_can", "I(sh_can ** 2)"), ("sh_gw", "I(sh_gw ** 2)")]:
        if m1.params[kk] < 0:
            tp, s1 = peak(m1, k, kk)
            _, s2 = peak(m2, k, kk)
            w("Peak in {}: {:.2f} (SE {:.2f}; {:.2f}); share of districts above it {:.1%}; max {:.2f}.".format(
                k, tp, s1, s2, (s[k] > tp).mean(), s[k].max()))
    # E: split sample by dominant source
    for nm, cond in [("groundwater-majority (gw share of irrigated > 0.5)", c2["gwc"] > 0.5),
                     ("canal-majority (canal share of irrigated > 0.5)", c2["canc"] > 0.5),
                     ("neither source above 0.5", (c2["gwc"] <= 0.5) & (c2["canc"] <= 0.5))]:
        r, _, _ = quad(c2[cond], y, "irr", TER, q="Q2", tag="E " + nm, spec="E")
        w("")
        w("Spec E, " + nm + ":")
        w("")
        w(QH)
        w(QS)
        w(fmt_quad(r))

# Antyodaya version: recent window, village shares by source
w("")
w("### Antyodaya 2019-20 sources, diversity 2015-16 to 2019-20 (all crops) and full period")
w("")
w("Antyodaya records sources as the share of villages using each, not irrigated area; "
  "groundwater share below is ground / (ground + canal + surface + other) village shares.")
for tagw in ["w1517", "full"]:
    a = with_win(d0, tagw, "all")
    tot = a[["ay_src_ground_vshare", "ay_src_canal_vshare", "ay_src_surface_vshare",
             "ay_src_other_vshare"]].sum(axis=1)
    a["gwa"] = a["ay_src_ground_vshare"] / tot.replace(0, np.nan)
    a["ay_can"] = a["ay_src_canal_vshare"]
    a["ay_gw"] = a["ay_src_ground_vshare"]
    for y in ["D1", "D0", "D0_05", "E"]:
        w("")
        w("{} , window {}:".format(YLAB[y], tagw))
        w("")
        r, m1, m2 = quad(a, y, "irr_share", " + gwa" + TER, q="Q2ay", tag="B Antyodaya + gw mix, " + tagw)
        w(QH)
        w(QS)
        w(fmt_quad(r))
        w("groundwater mix coefficient {:.3f} (SE {:.3f}; {:.3f}).".format(
            m1.params["gwa"], m1.bse["gwa"], m2.bse["gwa"]))
        f = ("{} ~ irr_share + I(irr_share**2) + gwa + irr_share:gwa + I(irr_share**2):gwa{} + "
             "C(state_name)").format(y, TER)
        m1, m2, s = fit(a, f, [y, "irr_share", "gwa", "elev_km", "tri", "shrug_key"])
        k2, ki = "I(irr_share ** 2)", "I(irr_share ** 2):gwa"
        V = m1.cov_params()
        sg = float(np.sqrt(V.loc[k2, k2] + V.loc[ki, ki] + 2 * V.loc[k2, ki]))
        w("Curvature at groundwater mix 0: {:.3f} (SE {:.3f}); at mix 1: {:.3f} (SE {:.3f}); "
          "interaction p {:.3f} parent / {:.3f} state; n {}.".format(
              m1.params[k2], m1.bse[k2], m1.params[k2] + m1.params[ki], sg,
              m1.pvalues[ki], m2.pvalues[ki], int(m1.nobs)))
        ROWS.append(dict(q="Q2ay", tag="C interaction, " + tagw, y=y, x=ki, n=int(m1.nobs),
                         b2=m1.params[ki], b2_se_par=m1.bse[ki], b2_se_st=m2.bse[ki],
                         b2_p_par=m1.pvalues[ki], b2_p_st=m2.pvalues[ki]))
        for nm, cond in [("groundwater-majority villages", a["gwa"] > 0.5),
                         ("canal-majority villages", a["ay_can"] / tot > 0.5)]:
            r, _, _ = quad(a[cond], y, "irr_share", TER, q="Q2ay", tag="E " + nm + ", " + tagw)
            w("  " + nm + ": " + fmt_quad(r))

# ================================================================ Q3 predetermined 1991
w("")
w("## Q3. Irrigation measured in 1991 (before the crop record)")
c3b = d0[d0["c1991_district_usable"]].copy()
c3b["irr91"] = c3b["c1991_irr_share"]
c3b["gw91"] = c3b["c1991_comp_groundwater"]
c3b["can91s"] = c3b["c1991_sh_canal"]
c3b["gw91s"] = c3b["c1991_sh_groundwater"]
c3b["el91s"] = c3b["c1991_sh_gw_elec"]
c3b["tk91s"] = c3b["c1991_sh_tank_lake"]
c3b["ro91s"] = c3b["c1991_sh_river_other"]
w("")
w("Usable 1991 districts in the site sample: {}; correlation of 1991 share with Antyodaya "
  "2019-20 share {:.3f}; mean 1991 share {:.3f}.".format(
      len(c3b), c3b[["irr91", "irr_share"]].corr().iloc[0, 1], c3b["irr91"].mean()))
for wl, t, v in [WINS[0], WINS[2], WINS[7]]:
    c3 = with_win(c3b, t, v)
    for y in ["D1", "D0", "D0_05", "E"]:
        w("")
        w("### {}, {}".format(YLAB[y], wl))
        w("")
        w(QH)
        w(QS)
        r, _, _ = quad(c3, y, "irr91", TER, q="Q3", tag="1991 share, quadratic", window=t, variant=v)
        w(fmt_quad(r))
        r, _, _ = quad(c3, y, "irr_share", TER, q="Q3", tag="Antyodaya 2019-20 share, same districts",
                       window=t, variant=v)
        w(fmt_quad(r))
        r, m1, m2 = quad(c3, y, "irr91", " + gw91" + TER, q="Q3", tag="1991 share + 1991 groundwater share of irrigated",
                         window=t, variant=v)
        w(fmt_quad(r))
        w("")
        w("1991 groundwater share of irrigated: {:.3f} (SE {:.3f}; {:.3f}), p {:.3f} / {:.3f}.".format(
            m1.params["gw91"], m1.bse["gw91"], m2.bse["gw91"], m1.pvalues["gw91"], m2.pvalues["gw91"]))
        f = ("{} ~ can91s + I(can91s**2) + gw91s + I(gw91s**2) + tk91s + ro91s{} + C(state_name)").format(y, TER)
        m1, m2, s = fit(c3, f, [y, "can91s", "gw91s", "tk91s", "ro91s", "elev_km", "tri", "shrug_key"])
        w("Separate 1991 source shares (n {}): canal {:.3f} / sq {:.3f}{} (SE {:.3f}; {:.3f}); "
          "groundwater {:.3f} / sq {:.3f}{} (SE {:.3f}; {:.3f}).".format(
              int(m1.nobs), m1.params["can91s"], m1.params["I(can91s ** 2)"], st(m1.pvalues["I(can91s ** 2)"]),
              m1.bse["I(can91s ** 2)"], m2.bse["I(can91s ** 2)"],
              m1.params["gw91s"], m1.params["I(gw91s ** 2)"], st(m1.pvalues["I(gw91s ** 2)"]),
              m1.bse["I(gw91s ** 2)"], m2.bse["I(gw91s ** 2)"]))
        for k in ["can91s", "I(can91s ** 2)", "gw91s", "I(gw91s ** 2)"]:
            ROWS.append(dict(q="Q3", tag="sources " + k, y=y, x=k, window=t, variant=v, n=int(m1.nobs),
                             b1=m1.params[k], b1_se_par=m1.bse[k], b1_se_st=m2.bse[k],
                             b1_p_par=m1.pvalues[k], b1_p_st=m2.pvalues[k]))
        f = ("{} ~ irr91 + I(irr91**2) + el91s{} + C(state_name)").format(y, TER)
        m1, m2, s = fit(c3, f, [y, "irr91", "el91s", "elev_km", "tri", "shrug_key"])
        w("Adding the 1991 electrified wells and tube wells share of sown area (n {}): {:.3f} "
          "(SE {:.3f}; {:.3f}), p {:.3f} / {:.3f}.".format(
              int(m1.nobs), m1.params["el91s"], m1.bse["el91s"], m2.bse["el91s"],
              m1.pvalues["el91s"], m2.pvalues["el91s"]))
        ROWS.append(dict(q="Q3", tag="electrified gw share", y=y, x="el91s", window=t, variant=v,
                         n=int(m1.nobs), b1=m1.params["el91s"], b1_se_par=m1.bse["el91s"],
                         b1_se_st=m2.bse["el91s"], b1_p_par=m1.pvalues["el91s"], b1_p_st=m2.pvalues["el91s"]))

# ================================================================ Q4 long differences
w("")
w("## Q4. Long differences")
e = win("e9802", "all").add_suffix("_e")
L = d0.merge(e, left_on="district_key", right_index=True, how="left")
for tg in ["cs0913", "w1517"]:
    t = win(tg, "all").add_suffix("_" + tg)
    L = L.merge(t, left_on="district_key", right_index=True, how="left")
for y in ["D1", "D0", "D0_05", "E"]:
    for tg in ["cs0913", "w1517"]:
        L["d_{}_{}".format(y, tg)] = L["{}_{}".format(y, tg)] - L[y + "_e"]
for tg in ["cs0913", "w1517"]:
    rat = L["gca_" + tg] / L["gca_e"]
    L["area_stable_" + tg] = rat.between(2 / 3, 1.5)
for a, b in [(1991, 2011), (2001, 2011)]:
    for k, c in [("irr", "irr_share"), ("can", "sh_canal"), ("gw", "sh_groundwater"),
                 ("tk", "sh_tank_lake"), ("ro", "sh_river_other")]:
        L["d{}_{}_{}".format(k, a, b)] = L["c{}_{}".format(b, c)] - L["c{}_{}".format(a, c)]
    L["dirr2_{}_{}".format(a, b)] = L["c{}_irr_share".format(b)] ** 2 - L["c{}_irr_share".format(a)] ** 2
    L["ok_{}_{}".format(a, b)] = L["c%d_district_usable" % a] & L["c%d_district_usable" % b]
w("")
w("Districts with diversity in 1998-99 to 2002-03 (at least 3 years): {}. Usable in both 1991 and 2011: {}; "
  "both 2001 and 2011: {}.".format(int(L["D1_e"].notna().sum()), int(L["ok_1991_2011"].sum()),
                                    int(L["ok_2001_2011"].sum())))
for a, b in [(1991, 2011), (2001, 2011)]:
    sub = L[L["ok_{}_{}".format(a, b)]]
    w("Mean change {}-{}: irrigated share {:+.3f}, canal {:+.3f}, groundwater {:+.3f} (share of sown area).".format(
        a, b, sub["dirr_{}_{}".format(a, b)].mean(), sub["dcan_{}_{}".format(a, b)].mean(),
        sub["dgw_{}_{}".format(a, b)].mean()))
LDH = "| change in | n | slope (SE parent; SE state) | p parent / state |"
LDS = "|---|---|---|---|"
for a, b in [(1991, 2011), (2001, 2011)]:
    sfx = "{}_{}".format(a, b)
    for tg, tl in [("cs0913", "1998-02 to 2009-13"), ("w1517", "1998-02 to 2015-19")]:
        for filt, fl in [(None, "all"), ("area_stable_" + tg, "gross area within 2/3 to 1.5x")]:
            sub = L[L["ok_" + sfx]]
            if filt:
                sub = sub[sub[filt]]
            for y in ["D1", "D0", "D0_05", "E"]:
                dy = "d_{}_{}".format(y, tg)
                w("")
                w("### Census {}-{}, diversity {}, {}, outcome change in {}".format(a, b, tl, fl, YLAB[y]))
                w("")
                w(LDH)
                w(LDS)
                meta = dict(census=sfx, div=tg, filt=fl)
                o, _, _ = lin(sub, dy, ["dirr_" + sfx], TER, q="Q4", tag="total", **meta)
                for r in o:
                    w("| total irrigated share | {} | {:.3f}{} ({:.3f}; {:.3f}) | {:.3f} / {:.3f} |".format(
                        r["n"], r["b1"], st(r["b1_p_par"]), r["b1_se_par"], r["b1_se_st"], r["b1_p_par"], r["b1_p_st"]))
                o, _, _ = lin(sub, dy, ["dirr_" + sfx, "dirr2_" + sfx], TER, q="Q4", tag="total + square", **meta)
                for r in o:
                    w("| {} (with change in square) | {} | {:.3f}{} ({:.3f}; {:.3f}) | {:.3f} / {:.3f} |".format(
                        r["x"], r["n"], r["b1"], st(r["b1_p_par"]), r["b1_se_par"], r["b1_se_st"], r["b1_p_par"], r["b1_p_st"]))
                xs = ["dcan_" + sfx, "dgw_" + sfx, "dtk_" + sfx, "dro_" + sfx]
                o, m1, m2 = lin(sub, dy, xs, TER, q="Q4", tag="by source", **meta)
                for r in o:
                    w("| {} | {} | {:.3f}{} ({:.3f}; {:.3f}) | {:.3f} / {:.3f} |".format(
                        r["x"], r["n"], r["b1"], st(r["b1_p_par"]), r["b1_se_par"], r["b1_se_st"], r["b1_p_par"], r["b1_p_st"]))
                tt1 = m1.t_test("dcan_{0} - dgw_{0} = 0".format(sfx))
                tt2 = m2.t_test("dcan_{0} - dgw_{0} = 0".format(sfx))
                w("canal minus groundwater slope: {:.3f}, p {:.3f} parent / {:.3f} state.".format(
                    float(tt1.effect), float(tt1.pvalue), float(tt2.pvalue)))
                ROWS.append(dict(q="Q4", tag="canal - gw", y=dy, x="diff", n=int(m1.nobs),
                                 b1=float(tt1.effect), b1_p_par=float(tt1.pvalue),
                                 b1_p_st=float(tt2.pvalue), **meta))

pd.DataFrame(ROWS).to_csv(OUT + "/irrigation_v2_coefs.csv", index=False)
pd.DataFrame(MROWS).to_csv(OUT + "/irrigation_v2_markets.csv", index=False)
L.to_csv(OUT + "/irrigation_v2_longdiff_data.csv", index=False)
open(OUT + "/irrigation_v2_raw.md", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
print("done")
