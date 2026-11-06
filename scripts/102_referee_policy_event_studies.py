"""
102_referee_policy_event_studies.py

Adversarial referee checks on scripts/99_crop_groups_policy_event_studies.py.
Reuses its estimator (copied, unchanged) and its sample construction. Does not overwrite
any of script 99's outputs. Writes outputs/crop_groups/referee_policy_es_*.csv and
referee_policy_es_log.txt.

Checks
  R1  the "treated linear trend" spec in script 99 is collinear with the saturated event
      dummies + unit FE (pinv returns an arbitrary min-norm split). Show rank, then
      replace with identified versions: (a) post dummy + treated trend, (b) trend fitted
      on pre-period only and extrapolated.
  R2  reference-period sensitivity of the P07 pulse result; raw treated-control gap by year.
  R3  short-window DiD (drop the early convergence years).
  R4  leave-one-state-out for P07 pulse share.
  R5  mixed-treatment units (2011 parent with treated and untreated children).
  R6  NFSM-Rice log rice area and NFSM-Wheat pulse share (positive DiDs not in headline).
  R7  OMM: raw log-area changes by cohort, untreated controls have ~0 millets, power/MDE,
      2017 cohort pre-trend.
  R8  MSP C1: 2019 alone; Holm/BH over 9 groups; Whole-Year double counting in exposure.

USAGE: python scripts/102_referee_policy_event_studies.py
No APCNF / PMDS / RySS data are used.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")
REPO = r"D:/crop-diversity"
OUT = REPO + "/outputs/crop_groups"
RAW = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


# ---------------------------------------------------------------- estimator (from 99)
def demean(df, cols, fes, tol=1e-10, maxit=2000):
    X = df[cols].to_numpy(dtype=float).copy()
    codes = [pd.factorize(df[f])[0] for f in fes]
    counts = [np.bincount(c) for c in codes]
    for it in range(maxit):
        Xold = X.copy()
        for c, n in zip(codes, counts):
            for j in range(X.shape[1]):
                m = np.bincount(c, weights=X[:, j]) / n
                X[:, j] -= m[c]
        if np.max(np.abs(X - Xold)) < tol:
            break
    return X


def feols(df, y, xcols, fes, cluster):
    d = df.dropna(subset=[y] + xcols).copy()
    for _ in range(10):
        keep = np.ones(len(d), bool)
        for f in fes:
            keep &= d.groupby(f)[y].transform("size").to_numpy() > 1
        if keep.all():
            break
        d = d[keep]
    Z = demean(d, [y] + xcols, fes)
    yy, X = Z[:, 0], Z[:, 1:]
    keepc = np.abs(X).sum(0) > 1e-9
    xc = [c for c, k in zip(xcols, keepc) if k]
    X = X[:, keepc]
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ yy
    u = yy - X @ b
    g = pd.factorize(d[cluster])[0]
    G = g.max() + 1
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, g, X * u[:, None])
    N, K = X.shape
    V = XtX_inv @ (S.T @ S) @ XtX_inv * G / (G - 1) * (N - 1) / (N - K)
    return dict(b=pd.Series(b, xc), V=pd.DataFrame(V, xc, xc), n=N, G=G, data=d,
                rank=np.linalg.matrix_rank(X, tol=1e-8), k=X.shape[1], X=X)


def lincom(r, w_):
    w_ = pd.Series(w_).reindex(r["b"].index).fillna(0)
    return float(w_ @ r["b"]), float(np.sqrt(w_ @ r["V"] @ w_))


def wald(r, names):
    names = [n for n in names if n in r["b"].index]
    b = r["b"][names].to_numpy()
    V = r["V"].loc[names, names].to_numpy()
    F = float(b @ np.linalg.pinv(V) @ b) / len(names)
    return F, 1 - stats.f.cdf(F, len(names), r["G"] - 1)


def fmt(e, s):
    return "{:+.4f} ({:.4f})".format(e, s)


# ---------------------------------------------------------------- data (as in 99)
A = pd.read_csv(OUT + "/crop_group_panel_annual.csv")
A["state"] = A["unit_key"].str.split("|").str[0]
A = A[A["in_606"] == 1].copy()
A["yr"] = A["year_start"]
A_ok = A[A["flag_annual_unreliable"] == 0].copy()
cmap = pd.read_csv(OUT + "/crop_group_child_map.csv")[["div_key", "unit_key"]]
nf = pd.read_csv(OUT + "/policy_nfsm_district_year.csv")


def unit_treat(component):
    d = nf[nf["component"] == component].merge(cmap, left_on="district_key", right_on="div_key")
    d["yr"] = d["year"].str[:4].astype(int)
    return d.groupby(["unit_key", "yr"])["treated"].agg(
        lambda s: np.nan if s.isna().all() else s.max()).unstack()


def build_stack(panel, unit_first, cohort, ctrl_rule, lo, hi, name, state_fe=True,
                exclude_states=()):
    p = panel[(panel.yr >= lo) & (panel.yr <= hi)].copy()
    f = p["unit_key"].map(unit_first)
    p["treat"] = (f == cohort).astype(int)
    p = p[(p["treat"] == 1) | ctrl_rule(f)]
    p = p[~p["state"].isin(exclude_states)]
    if state_fe:
        st = p.groupby("state")["treat"].agg(["min", "max"])
        p = p[p["state"].isin(st[(st["min"] == 0) & (st["max"] == 1)].index)]
    p["e"] = np.where(p["treat"] == 1, p["yr"] - cohort, -999)
    p["stack"] = name
    p["u_s"] = p["unit_key"] + "#" + name
    p["t_s"] = p["state"] + "#" + p["yr"].astype(str)
    p["t_y"] = p["yr"].astype(str)
    return p


def add_events(p, ref=-1):
    es = sorted(e for e in p.loc[p.treat == 1, "e"].unique() if e != ref)
    cols = []
    for e in es:
        c = "ev_{}".format(e).replace("-", "m")
        p[c] = (p["e"] == e).astype(float)
        cols.append(c)
    return es, cols


uP = unit_treat("pulses")
firstP = uP.apply(lambda r: r[r == 1].index.min(), axis=1)
P07 = build_stack(A_ok, firstP, 2007, lambda f: f >= 2010, 1997, 2009, "P07",
                  exclude_states=("ASSAM", "JHARKHAND"))
es, cols = add_events(P07)
P07["post"] = ((P07.treat == 1) & (P07.e >= 0)).astype(float)
P07["tr_lin"] = P07["treat"] * (P07["yr"] - 1997)
rows = []

w("=" * 78)
w("Referee checks on script 99 (policy event studies)")
w("=" * 78)

# ---------------------------------------------------------------- R0 reproduce
for y in ["share_gca_pulses", "ln_area_ha_pulses"]:
    r = feols(P07, y, ["post"], ["u_s", "t_s"], "unit_key")
    w("R0 reproduce P07 DiD {}: {}  n {} G {}".format(y, fmt(r["b"]["post"],
      np.sqrt(r["V"].loc["post", "post"])), r["n"], r["G"]))

# ---------------------------------------------------------------- R1 trend collinearity
w("")
w("R1  treated linear trend in script 99")
for y in ["share_gca_pulses", "ln_area_ha_pulses"]:
    r = feols(P07, y, cols + ["tr_lin"], ["u_s", "t_s"], "unit_key")
    w("  {}: design has {} columns, numerical rank {} -> {}".format(
        y, r["k"], r["rank"], "RANK DEFICIENT (trend spanned by event dummies + unit FE)"
        if r["rank"] < r["k"] else "full rank"))
    # proof: regress demeaned trend on demeaned event dummies
    Xe = r["X"][:, :-1]
    t = r["X"][:, -1]
    bb = np.linalg.lstsq(Xe, t, rcond=None)[0]
    w("    R^2 of demeaned treated-trend on demeaned event dummies: {:.10f}".format(
        1 - np.sum((t - Xe @ bb) ** 2) / np.sum(t ** 2)))
    post = [c for c, e in zip(cols, es) if e >= 0]
    est = lincom(r, {c: 1 / len(post) for c in post})
    w("    script-99 'avg post with trend' (pinv min-norm, arbitrary): {}".format(fmt(*est)))
    # (a) identified: post dummy + treated linear trend
    ra = feols(P07, y, ["post", "tr_lin"], ["u_s", "t_s"], "unit_key")
    w("    (a) DiD with treated linear trend (post + treat*t): post {}  trend {}".format(
        fmt(ra["b"]["post"], np.sqrt(ra["V"].loc["post", "post"])),
        fmt(ra["b"]["tr_lin"], np.sqrt(ra["V"].loc["tr_lin", "tr_lin"]))))
    rows.append(dict(check="R1a post + treated trend, full pre", y=y, est=ra["b"]["post"],
                     se=np.sqrt(ra["V"].loc["post", "post"]), n=ra["n"]))
    # (a') same but pre window 2002+ (after the early convergence)
    sub = P07[P07.yr >= 2002]
    ra2 = feols(sub, y, ["post", "tr_lin"], ["u_s", "t_s"], "unit_key")
    w("    (a') same, window 2002-2009: post {}".format(
        fmt(ra2["b"]["post"], np.sqrt(ra2["V"].loc["post", "post"]))))
    rows.append(dict(check="R1a' post + treated trend, 2002-09", y=y, est=ra2["b"]["post"],
                     se=np.sqrt(ra2["V"].loc["post", "post"]), n=ra2["n"]))
    # (b) pre-period-only trend, extrapolated (event study on detrended outcome);
    # SE via unit cluster bootstrap over both steps
    def two_step(df):
        pre = df[(df.treat == 0) | (df.e < 0)]
        rp = feols(pre, y, ["tr_lin"], ["u_s", "t_s"], "unit_key")
        d2 = df.copy()
        d2["yd"] = d2[y] - rp["b"]["tr_lin"] * d2["tr_lin"]
        rr = feols(d2, "yd", ["post"], ["u_s", "t_s"], "unit_key")
        return rr["b"]["post"], rp["b"]["tr_lin"]
    est_b, slope = two_step(P07)
    units = P07.unit_key.unique()
    rng = np.random.default_rng(7)
    bs = []
    for _ in range(199):
        dr = rng.choice(units, len(units), replace=True)
        parts = []
        for i, u in enumerate(dr):
            x = P07[P07.unit_key == u].copy()
            x["unit_key"] = u + "#" + str(i)
            x["u_s"] = x["unit_key"]
            parts.append(x)
        try:
            bs.append(two_step(pd.concat(parts))[0])
        except Exception:
            pass
    w("    (b) pre-period trend {:+.5f}/yr extrapolated; post DiD on detrended y: {} "
      "(bootstrap 199)".format(slope, fmt(est_b, np.std(bs))))
    rows.append(dict(check="R1b pre-only trend extrapolated", y=y, est=est_b, se=np.std(bs),
                     n=len(P07)))

# ---------------------------------------------------------------- R2 reference sensitivity
w("")
w("R2  P07 pulse share: mean post coefficient minus alternative pre baselines")
for y in ["share_gca_pulses", "ln_area_ha_pulses"]:
    r = feols(P07, y, cols, ["u_s", "t_s"], "unit_key")
    post = [c for c, e in zip(cols, es) if e >= 0]
    for lab, rng_ in [("e=-1 only", (-1, -1)), ("e=-2..-1", (-2, -1)), ("e=-3..-1", (-3, -1)),
                      ("e=-3..-2", (-3, -2)), ("e=-5..-1", (-5, -1)), ("e=-4..-1", (-4, -1)),
                      ("e=-10..-1 (all)", (-10, -1))]:
        base = [e for e in range(rng_[0], rng_[1] + 1)]
        wv = {c: 1 / len(post) for c in post}
        for e in base:
            if e != -1:
                wv["ev_m{}".format(-e)] = wv.get("ev_m{}".format(-e), 0) - 1 / len(base)
        est = lincom(r, wv)
        w("  {:<18} {:<18} {}".format(y, lab, fmt(*est)))
        rows.append(dict(check="R2 post minus " + lab, y=y, est=est[0], se=est[1], n=r["n"]))
# raw within-state gap by year
g = P07.copy()
g["dev"] = g["share_gca_pulses"] - g.groupby("t_s")["share_gca_pulses"].transform("mean")
gap = g.groupby(["yr", "treat"])["dev"].mean().unstack()
w("  raw within-state-year deviation of pulse share, treated minus control, by year:")
w("  " + ", ".join("{} {:+.4f}".format(t, v) for t, v in (gap[1] - gap[0]).items()))

# ---------------------------------------------------------------- R3 short windows
w("")
w("R3  short-window DiD (post vs pre within window), P07")
for lo in [1997, 2000, 2002, 2003, 2004]:
    sub = P07[P07.yr >= lo]
    for y in ["share_gca_pulses", "ln_area_ha_pulses"]:
        r = feols(sub, y, ["post"], ["u_s", "t_s"], "unit_key")
        est = (r["b"]["post"], np.sqrt(r["V"].loc["post", "post"]))
        w("  window {}-2009 {:<18} DiD {}".format(lo, y, fmt(*est)))
        rows.append(dict(check="R3 DiD window {}-2009".format(lo), y=y, est=est[0],
                         se=est[1], n=r["n"]))

# ---------------------------------------------------------------- R4 leave one state out
w("")
w("R4  leave-one-state-out, P07 pulse share: DiD and post-vs-(e-4..-1)")
post = [c for c, e in zip(cols, es) if e >= 0]
near = [c for c, e in zip(cols, es) if -4 <= e < -1]
for s in sorted(P07.state.unique()):
    sub = P07[P07.state != s]
    r = feols(sub, "share_gca_pulses", ["post"], ["u_s", "t_s"], "unit_key")
    re_ = feols(sub, "share_gca_pulses", cols, ["u_s", "t_s"], "unit_key")
    wv = {c: 1 / len(post) for c in post}
    for c in near:
        wv[c] = -1 / (len(near) + 1)
    pv = lincom(re_, wv)
    w("  drop {:<16} DiD {}  post-vs-pre4 {}  (treated units {})".format(
        s, fmt(r["b"]["post"], np.sqrt(r["V"].loc["post", "post"])), fmt(*pv),
        P07[(P07.state == s) & (P07.treat == 1)].unit_key.nunique()))

# ---------------------------------------------------------------- R5 mixed units
w("")
w("R5  2011-parent units with mixed child treatment in NFSM-Pulses 2007")
d = nf[(nf.component == "pulses") & (nf.year == "2007-08")].merge(
    cmap, left_on="district_key", right_on="div_key")
mix = d.groupby("unit_key")["treated"].agg(["min", "max", "size"])
mixed = mix[(mix["size"] > 1) & (mix["min"] != mix["max"])]
w("  mixed units: {} ({})".format(len(mixed), ", ".join(mixed.index[:10])))
sub = P07[~P07.unit_key.isin(mixed.index)]
r = feols(sub, "share_gca_pulses", ["post"], ["u_s", "t_s"], "unit_key")
w("  P07 pulse share DiD dropping mixed units: {}".format(
    fmt(r["b"]["post"], np.sqrt(r["V"].loc["post", "post"]))))
w("  confidence of 2007-08 pulses list rows: " +
  str(nf[(nf.component == "pulses") & (nf.year == "2007-08")].confidence.value_counts().to_dict()))

# ---------------------------------------------------------------- R6 rice / wheat lists
w("")
w("R6  NFSM-Rice and NFSM-Wheat 2007: event coefficients for outcomes not in the headline")
uR, uW = unit_treat("rice"), unit_treat("wheat")
for comp, u_, ys in [("rice", uR, ["share_gca_rice", "ln_area_ha_rice"]),
                     ("wheat", uW, ["share_gca_wheat", "ln_area_ha_wheat", "share_gca_pulses"])]:
    first = u_.apply(lambda r: r[r == 1].index.min(), axis=1)
    S = build_stack(A_ok, first, 2007, lambda f: f.isna() | (f >= 2014), 1997, 2013,
                    comp[0].upper() + "07")
    es_s, cols_s = add_events(S)
    S["post"] = ((S.treat == 1) & (S.e >= 0)).astype(float)
    S["tr_lin"] = S["treat"] * (S["yr"] - 1997)
    for y in ys:
        r = feols(S, y, cols_s, ["u_s", "t_s"], "unit_key")
        co = ", ".join("{}:{:+.3f}".format(e, r["b"].get(c, np.nan)) for e, c in zip(es_s, cols_s))
        ra = feols(S, y, ["post", "tr_lin"], ["u_s", "t_s"], "unit_key")
        w("  {}07 {:<18} {}".format(comp, y, co))
        w("      DiD with treated linear trend: {}".format(
            fmt(ra["b"]["post"], np.sqrt(ra["V"].loc["post", "post"]))))
        rows.append(dict(check="R6 {}07 post + treated trend".format(comp), y=y,
                         est=ra["b"]["post"], se=np.sqrt(ra["V"].loc["post", "post"]), n=ra["n"]))
    if comp == "rice":
        sub = S[S.state != "TAMIL NADU"]
        w("      rice07 TN units in stack: {}".format(S[S.state == "TAMIL NADU"].unit_key.nunique()))
        for y in ["share_gca_rice", "ln_area_ha_rice"]:
            r = feols(sub, y, ["post"], ["u_s", "t_s"], "unit_key")
            w("      rice07 drop Tamil Nadu, {} DiD {}".format(
                y, fmt(r["b"]["post"], np.sqrt(r["V"].loc["post", "post"]))))

# ---------------------------------------------------------------- R7 OMM
w("")
w("R7  Odisha Millets Mission")
om = pd.read_csv(OUT + "/policy_omm_district_year.csv").merge(
    cmap, left_on="district_key", right_on="div_key")
gmap = om.groupby("unit_key")["omm_entry_year"].first().str[:4].astype(float).fillna(np.inf)
gmap[gmap > 2019] = np.inf
O = A_ok[(A_ok.state == "ODISHA") & (A_ok.yr >= 2005)].copy()
O["g"] = O.unit_key.map(gmap)
O["c"] = O.g.replace(np.inf, 0)
tab = O.pivot_table(index="yr", columns="c", values="area_ha_millets", aggfunc="sum")
w("  total millet area (ha) by cohort and year:")
w(tab.round(0).to_string())
tabg = O.pivot_table(index="yr", columns="c", values="gca_ha", aggfunc="sum")
w("  total GCA (ha) by cohort, 2016-2019:")
w(tabg.loc[2016:2019].round(0).to_string())
cnt0 = O[(O.c == 0)].groupby("yr").apply(lambda d: (d.area_ha_millets > 0).sum())
w("  untreated districts with any millet area, by year: " + cnt0.to_dict().__str__())
# simple 2x2 on log area of TOTAL cohort area (2016 vs 2019), treated cohorts pooled
for c in [2017, 2018, 2019, 0]:
    a16, a18, a19 = tab.loc[2016, c], tab.loc[2018, c], tab.loc[2019, c]
    w("  cohort {}: ln area 2019 - 2016 = {:+.3f}; 2019 - 2018 = {:+.3f}".format(
        c, np.log(a19) - np.log(a16), np.log(a19) - np.log(a18)))
# MDE for millet share given bootstrap SE 0.0035
w("  MDE (80% power, 5%) for millet share ATT with SE 0.0035: {:.4f}; 2017-cohort pre mean "
  "share {:.4f}".format(2.8 * 0.0035, O[(O.g == 2017) & (O.yr < 2017)].share_gca_millets.mean()))
# 2017 cohort raw trend 2005-2016 relative to untreated
sl = O[(O.yr < 2017) & (O.g == 2017)].groupby("yr").share_gca_millets.mean()
w("  2017 cohort millet share 2005-2016 linear slope: {:+.5f}/yr".format(
    np.polyfit(sl.index, sl.values, 1)[0]))

# ---------------------------------------------------------------- R8 MSP
w("")
w("R8  MSP C1 checks")
c1 = pd.read_csv(OUT + "/policy_es_msp_c1_exposure.csv")
from statsmodels.stats.multitest import multipletests
p_ = c1["p"].to_numpy()
w("  raw p: " + ", ".join("{} {:.3f}".format(g_, p) for g_, p in zip(c1.group, p_)))
w("  Holm-adjusted min p {:.3f}; BH min q {:.3f}".format(
    multipletests(p_, method="holm")[1].min(), multipletests(p_, method="fdr_bh")[1].min()))
# C1 2019 alone and pre-trend-adjusted post (post - linear extrapolation of 2008-2016 leads)
GROUPS = ["rice", "wheat", "maize", "millets", "pulses", "oilseeds", "cotton", "sugarcane", "other"]
X_d = pd.read_csv(OUT + "/policy_es_msp_exposure.csv", index_col=0)["msp_exposure"]
C = A_ok[A_ok.yr >= 2008].copy()
C["X"] = C.unit_key.map((X_d - X_d.mean()) / X_d.std())
C = C.dropna(subset=["X"])
C["st_yr"] = C.state + "#" + C.yr.astype(str)
yrs = sorted(C.yr.unique())
xc = []
for t in yrs:
    if t == 2017:
        continue
    C["x_{}".format(t)] = C.X * (C.yr == t)
    xc.append("x_{}".format(t))
C["post"] = C.X * (C.yr >= 2018)
C["xtr"] = C.X * (C.yr - 2008)
for g_ in GROUPS:
    y = "share_gca_" + g_
    r = feols(C, y, xc, ["unit_key", "st_yr"], "unit_key")
    e19 = lincom(r, {"x_2019": 1})
    ra = feols(C, y, ["post", "xtr"], ["unit_key", "st_yr"], "unit_key")
    w("  {:<9} 2019 alone {}   post with exposure x linear trend {}".format(
        g_, fmt(*e19), fmt(ra["b"]["post"], np.sqrt(ra["V"].loc["post", "post"]))))
    rows.append(dict(check="R8 C1 post + exposure trend", y=y, est=ra["b"]["post"],
                     se=np.sqrt(ra["V"].loc["post", "post"]), n=ra["n"]))
# Whole-Year rows: how much 2013-16 area sits in Whole Year rows for MSP crops
df = pd.read_csv(RAW, usecols=["year", "season", "state_name", "district_name", "crop_name", "area"])
for c_ in ["year", "season", "state_name", "district_name", "crop_name"]:
    df[c_] = df[c_].astype(str).str.strip()
df["ys"] = df.year.str[:4].astype(int)
sub = df[(df.ys >= 2013) & (df.ys <= 2016) & (df.area > 0)]
wy = sub[sub.season == "Whole Year"]
both = wy.merge(sub[sub.season != "Whole Year"][["year", "state_name", "district_name", "crop_name"]]
                .drop_duplicates(), on=["year", "state_name", "district_name", "crop_name"])
w("  2013-16 raw APY: Whole Year area {:.2f} Mha of {:.2f} Mha; Whole Year rows that "
  "duplicate a seasonal row of the same crop: {} rows, {:.2f} Mha".format(
      wy.area.sum() / 1e6, sub.area.sum() / 1e6, len(both), both.area.sum() / 1e6))
w("  top crops in those overlaps: " + both.groupby("crop_name").area.sum().sort_values(
    ascending=False).head(6).div(1e3).round(0).to_dict().__str__() + " (kha)")

# ---------------------------------------------------------------- R9 Odisha coverage break
w("")
w("R9  Odisha APY coverage by year (all Odisha units, kha)")
od = pd.read_csv(OUT + "/crop_group_panel_annual.csv")
od = od[od.unit_key.str.startswith("ODISHA|")]
w(od.groupby("year_start")[["gca_ha", "area_ha_rice", "area_ha_millets", "area_ha_pulses",
                            "area_ha_oilseeds", "area_ha_cotton", "area_ha_other"]]
  .sum().div(1e3).round(0).loc[2004:2019].to_string())
odr = df[(df.state_name == "Odisha") & (df.ys >= 2017) & (df.ys <= 2019) & (df.area > 0)]
w("  Odisha raw crops reported, by year (count of crop names): " +
  odr.groupby("ys").crop_name.nunique().to_dict().__str__())
new = set(odr[odr.ys == 2019].crop_name) - set(odr[odr.ys == 2018].crop_name)
w("  crop names present in 2019-20 but not 2018-19: " + ", ".join(sorted(new)))

# ---------------------------------------------------------------- R10 OMM without 2019-20
w("")
w("R10 OMM ATT dropping 2019-20 (coverage break); only the 2017 and 2018 cohorts remain")


def cs_att(df_, y):
    wide = df_.pivot(index="unit_key", columns="yr", values=y)
    gg_ = df_.groupby("unit_key")["g"].first().reindex(wide.index)
    res = []
    for gg in sorted(set(gg_[np.isfinite(gg_)])):
        base = gg - 1
        for t in wide.columns:
            if t == base or base not in wide.columns:
                continue
            tr = gg_ == gg
            ct = (gg_ > max(t, base)) & (gg_ != gg)
            dd = wide[t] - wide[base]
            a, b = dd[tr].dropna(), dd[ct].dropna()
            if len(a) == 0 or len(b) < 2:
                continue
            res.append(dict(g=gg, t=t, e=t - gg, att=a.mean() - b.mean(), n_t=len(a)))
    return pd.DataFrame(res)


O["ln1p_millets"] = np.log1p(O["area_ha_millets"])
O18 = O[O.yr <= 2018].copy()
O18.loc[O18.g == 2019, "g"] = np.inf
for y in ["share_gca_millets", "ln1p_millets"]:
    for lab, dfx in [("with 2019-20", O), ("without 2019-20", O18)]:
        att = cs_att(dfx, y)
        post = att[att.e >= 0]
        est = np.average(post.att, weights=post.n_t)
        units = dfx.unit_key.unique()
        rng = np.random.default_rng(11)
        bs = []
        for _ in range(299):
            dr = rng.choice(units, len(units), replace=True)
            parts = []
            for i, u in enumerate(dr):
                x = dfx[dfx.unit_key == u].copy()
                x["unit_key"] = u + "#" + str(i)
                parts.append(x)
            a = cs_att(pd.concat(parts), y)
            if len(a) and (a.e >= 0).any():
                pa = a[a.e >= 0]
                bs.append(np.average(pa.att, weights=pa.n_t))
        w("  {:<18} {:<16} ATT {}  95% CI {:+.4f} to {:+.4f}".format(
            y, lab, fmt(est, np.std(bs)), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
        rows.append(dict(check="R10 OMM " + lab, y=y, est=est, se=np.std(bs), n=len(dfx)))

pd.DataFrame(rows).to_csv(OUT + "/referee_policy_es_checks.csv", index=False)
with open(OUT + "/referee_policy_es_log.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG))
w("done")
