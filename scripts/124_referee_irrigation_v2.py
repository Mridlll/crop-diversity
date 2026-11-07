"""
124_referee_irrigation_v2.py

Referee checks on the irrigation v2 note (scripts 122/123,
outputs/revision_nov8/irrigation_v2/irrigation_v2_results.md). Reads 122/123/121 code
heads for data construction only (no writes from them); writes only to
outputs/revision_nov8/irrigation_v2/referee/.

Sections (run all, or name some on the command line):
  repro   reproduce key numbers from 123
  infer   state clustering with t(G-1) and wild cluster restricted bootstrap (Webb weights)
  q1x     Q1: extension-only window (2020-21, 2021-22), leave-one-state-out on the bend
  q2      Q2: tank leave-one-state-out, per-SD sizes, canal-majority support and power
  q3      Q3: 1991 repair variants, dropping weak 1991 state-rounds, scale/rank/bin checks,
          other early measures (DES 1998-2000, census 2001/2011) on the same districts
  q4      Q4: leave-one-state-out, fixed-denominator count, GCA control, repair variants
  mt      multiple testing across the source-split tests in irrigation_v2_coefs.csv

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/124_referee_irrigation_v2.py [sections]
"""
import io
import os
import sys
import contextlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
REPO = r"D:/crop-diversity"
SRC = REPO + "/scripts/"
OUTV = REPO + "/outputs/revision_nov8/irrigation_v2"
OUT = OUTV + "/referee"
os.makedirs(OUT, exist_ok=True)
SECTIONS = sys.argv[1:] or ["repro", "infer", "q1x", "q2", "q3", "q4", "mt"]
LOG = []
RES = []
RNG = np.random.default_rng(20261107)
B = 9999


def w(s=""):
    print(s)
    LOG.append(str(s))


def head_exec(path, marker, extra=None):
    src = open(path, encoding="utf-8").read()
    ns = {"__name__": "ref_" + os.path.basename(path)}
    if extra:
        ns.update(extra)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src[:src.index(marker)], path, "exec"), ns)
    return ns


# ------------------------------------------------------------------ data from 123 (head only)
N123 = head_exec(SRC + "123_irrigation_v2_estimates.py",
                 "# ================================================================ Q1 irrigation bend")
d0 = N123["d0"]
with_win = N123["with_win"]
win = N123["win"]
TER = " + elev_km + tri"


# ------------------------------------------------------------------ inference tools
def design(df, formula, keep):
    s = df.dropna(subset=[k for k in keep if k in df]).copy()
    m = smf.ols(formula, s)
    return m, s


def cr1(X, u, h, g):
    """cluster-robust SE of a'beta where h = X (X'X)^-1 a (n vector); u residuals (n x B)."""
    G = g.max() + 1
    n, k = X.shape
    S = np.zeros((G,) + u.shape[1:])
    np.add.at(S, g, h[:, None] * u if u.ndim == 2 else h * u)
    c = G / (G - 1) * (n - 1) / (n - k)
    return np.sqrt(c * (S ** 2).sum(axis=0))


def webb(G, B_):
    vals = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
    return vals[RNG.integers(0, 6, size=(G, B_))]


def wcr(df, formula, term, cluster, keep, B_=B):
    """Wild cluster restricted bootstrap p for H0: term = 0 (Webb 6-point weights),
    plus CR1 t and the t(G-1) p. Returns dict."""
    m, s = design(df, formula, keep)
    X = m.exog
    y = m.endog
    names = m.exog_names
    j = names.index(term)
    g = pd.factorize(s[cluster])[0]
    G = g.max() + 1
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    u = y - X @ b
    h = X @ XtXi[:, j]
    se = cr1(X, u, h, g)
    t0 = b[j] / se
    # restricted fit
    Xr = np.delete(X, j, axis=1)
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    ur = y - Xr @ br
    fr = Xr @ br
    V = webb(G, B_)
    Ys = fr[:, None] + ur[:, None] * V[g, :]
    bs = XtXi @ (X.T @ Ys)
    us = Ys - X @ bs
    ses = cr1(X, us, h, g)
    ts = bs[j] / ses
    p_wcr = float(np.mean(np.abs(ts) >= abs(t0)))
    return dict(term=term, b=b[j], se=se, t=t0, G=G, n=len(y), p_norm=2 * stats.norm.sf(abs(t0)),
                p_tG1=2 * stats.t.sf(abs(t0), G - 1), p_wcr=p_wcr)


def fmt_inf(r):
    return ("b {:.3f}, state SE {:.3f}, t {:.2f}, G {}, n {}: p normal {:.3f}, p t(G-1) {:.3f}, "
            "p wild-cluster (Webb, {} draws) {:.3f}").format(r["b"], r["se"], r["t"], r["G"], r["n"],
                                                             r["p_norm"], r["p_tG1"], B, r["p_wcr"])


def quadfit(df, y, x, extra="", fe=True, cl="state_name"):
    f = "{} ~ {} + I({}**2){}{}".format(y, x, x, extra, " + C(state_name)" if fe else "")
    keep = [y, x, "shrug_key", "state_name"] + [v.strip() for v in extra.replace("+", " ").split()
                                                 if v.strip().isidentifier()]
    s = df.dropna(subset=[k for k in keep if k in df]).copy()
    m1 = smf.ols(f, s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s["shrug_key"])[0]})
    m2 = smf.ols(f, s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s[cl])[0]})
    x2 = "I({} ** 2)".format(x)
    b1, b2 = m1.params[x], m1.params[x2]
    pk = -b1 / (2 * b2) if b2 < 0 else np.nan
    return dict(n=int(m1.nobs), G=s["state_name"].nunique(), b1=b1, b2=b2, se2_par=m1.bse[x2],
                se2_st=m2.bse[x2], p2_par=m1.pvalues[x2], p2_st=m2.pvalues[x2],
                p1_par=m1.pvalues[x], p1_st=m2.pvalues[x], peak=pk, f=f, keep=keep, s=s)


def rq(r):
    return "n {} (G {}): b1 {:.2f} (p {:.3f}/{:.3f}), b2 {:.2f} (SE {:.2f}; {:.2f}, p {:.3f}/{:.3f}), peak {}".format(
        r["n"], r["G"], r["b1"], r["p1_par"], r["p1_st"], r["b2"], r["se2_par"], r["se2_st"],
        r["p2_par"], r["p2_st"], "{:.2f}".format(r["peak"]) if pd.notna(r["peak"]) else "none")


def rec(section, label, **kw):
    RES.append(dict(section=section, label=label, **{k: v for k, v in kw.items() if not isinstance(v, (pd.DataFrame, list))}))


# ------------------------------------------------------------------ shared samples
c2 = with_win(d0, "cs0913", "all")
c2 = c2[c2["c2011_district_usable"]].copy()
c2["irr"] = c2["c2011_irr_share"]
c2["gwc"] = c2["c2011_comp_groundwater"]
c2["canc"] = c2["c2011_comp_canal"]
c2["sh_can"] = c2["c2011_sh_canal"]
c2["sh_gw"] = c2["c2011_sh_groundwater"]
c2["sh_tk"] = c2["c2011_sh_tank_lake"]
c2["sh_ro"] = c2["c2011_sh_river_other"]
FD = "{} ~ sh_can + I(sh_can**2) + sh_gw + I(sh_gw**2) + sh_tk + sh_ro" + TER + " + C(state_name)"
KD = ["sh_can", "sh_gw", "sh_tk", "sh_ro", "elev_km", "tri", "shrug_key", "state_name"]

c3b = d0[d0["c1991_district_usable"]].copy()
c3b["irr91"] = c3b["c1991_irr_share"]


def ld_frame():
    e = win("e9802", "all").add_suffix("_e")
    L = d0.merge(e, left_on="district_key", right_index=True, how="left")
    for tg in ["cs0913", "w1517"]:
        t = win(tg, "all").add_suffix("_" + tg)
        L = L.merge(t, left_on="district_key", right_index=True, how="left")
        for y in ["D1", "D0", "D0_05", "E"]:
            L["d_{}_{}".format(y, tg)] = L["{}_{}".format(y, tg)] - L[y + "_e"]
        L["area_stable_" + tg] = (L["gca_" + tg] / L["gca_e"]).between(2 / 3, 1.5)
        L["dlgca_" + tg] = np.log(L["gca_" + tg] / L["gca_e"])
    L["dirr_1991_2011"] = L["c2011_irr_share"] - L["c1991_irr_share"]
    L["ok_1991_2011"] = L["c1991_district_usable"] & L["c2011_district_usable"]
    for k, c in [("can", "sh_canal"), ("gw", "sh_groundwater"), ("tk", "sh_tank_lake"), ("ro", "sh_river_other")]:
        L["d{}_1991_2011".format(k)] = L["c2011_" + c] - L["c1991_" + c]
    return L


# ================================================================== repro
if "repro" in SECTIONS:
    w("# Referee checks on irrigation v2")
    w("")
    w("## R0. Reproduction")
    w("")
    for lab, t, v, y, exp in [("Q1 D1 (b) flags drop6", "w1721", "flags_drop6", "D1", -6.46),
                              ("Q1 D1 (a) all", "w1517", "all", "D1", -6.78),
                              ("Q1 D0_05 (b)", "w1721", "flags_drop6", "D0_05", -12.81)]:
        r = quadfit(with_win(d0, t, v), y, "irr_share")
        w("- {}: {} [note says {}]".format(lab, rq(r), exp))
        rec("repro", lab, **{k: r[k] for k in ["n", "b2", "p2_par", "p2_st", "peak"]})
    for y in ["D1"]:
        m = smf.ols(FD.format(y), c2.dropna(subset=KD + [y])).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(c2.dropna(subset=KD + [y])["state_name"])[0]})
        w("- Q2 spec D tank, D1: {:.2f} (state SE {:.2f}) n {} [note -9.28 (0.93)]".format(
            m.params["sh_tk"], m.bse["sh_tk"], int(m.nobs)))
    for nm, cond in [("gw-majority", c2["gwc"] > 0.5), ("canal-majority", c2["canc"] > 0.5)]:
        r = quadfit(c2[cond], "D1", "irr", TER)
        w("- Q2 E {} D1: {}".format(nm, rq(r)))
    for t, v in [("full", "all"), ("w1517", "all"), ("w1721", "flags_drop6")]:
        r = quadfit(with_win(c3b, t, v), "D1", "irr91", TER)
        w("- Q3 D1 1991 {} : {}".format(t, rq(r)))
    L = ld_frame()
    for tg in ["cs0913", "w1517"]:
        sub = L[L["ok_1991_2011"]]
        f = "d_D0_05_{} ~ dirr_1991_2011{} + C(state_name)".format(tg, TER)
        s = sub.dropna(subset=["d_D0_05_" + tg, "dirr_1991_2011", "elev_km", "tri"])
        m = smf.ols(f, s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s["state_name"])[0]})
        w("- Q4 dD0_05 to {}: {:.2f} (state SE {:.2f}) n {}".format(tg, m.params["dirr_1991_2011"],
                                                                     m.bse["dirr_1991_2011"], int(m.nobs)))

# ================================================================== inference
if "infer" in SECTIONS:
    w("")
    w("## R1. Few-cluster inference: state clusters, t(G-1), wild cluster restricted bootstrap")
    w("")
    jobs = []
    for lab, t, v, y in [("Q1 D1 (a) 2015-20 all", "w1517", "all", "D1"),
                         ("Q1 D1 (b) flags drop6", "w1721", "flags_drop6", "D1"),
                         ("Q1 D1 (b) raw", "w1721", "all", "D1"),
                         ("Q1 D0_05 (b) flags drop6", "w1721", "flags_drop6", "D0_05"),
                         ("Q1 D2 (b) raw", "w1721", "all", "D2"),
                         ("Q1 E (b) flags drop6", "w1721", "flags_drop6", "E")]:
        df = with_win(d0, t, v)
        jobs.append((lab, df, "{} ~ irr_share + I(irr_share**2) + C(state_name)".format(y),
                     "I(irr_share ** 2)", [y, "irr_share", "state_name"]))
    for y in ["D1", "D0_05", "D0"]:
        for nm, cond in [("gw-majority", c2["gwc"] > 0.5), ("canal-majority", c2["canc"] > 0.5)]:
            jobs.append(("Q2 E {} {}".format(nm, y), c2[cond],
                         "{} ~ irr + I(irr**2){} + C(state_name)".format(y, TER), "I(irr ** 2)",
                         [y, "irr", "elev_km", "tri", "state_name"]))
        jobs.append(("Q2 C interaction irr^2 x gw, " + y, c2,
                     "{} ~ irr + I(irr**2) + gwc + irr:gwc + I(irr**2):gwc{} + C(state_name)".format(y, TER),
                     "I(irr ** 2):gwc", [y, "irr", "gwc", "elev_km", "tri", "state_name"]))
    for y in ["D1", "D0_05"]:
        jobs.append(("Q2 D tank share, " + y, c2, FD.format(y), "sh_tk", KD + [y]))
        jobs.append(("Q2 D gw square, " + y, c2, FD.format(y), "I(sh_gw ** 2)", KD + [y]))
        jobs.append(("Q2 D canal square, " + y, c2, FD.format(y), "I(sh_can ** 2)", KD + [y]))
    for t, v, tl in [("full", "all", "full"), ("w1517", "all", "(a)"), ("w1721", "flags_drop6", "(b)")]:
        df = with_win(c3b, t, v)
        for y in ["D1", "D0_05", "D0"]:
            jobs.append(("Q3 1991 {} {}".format(tl, y), df,
                         "{} ~ irr91 + I(irr91**2){} + C(state_name)".format(y, TER), "I(irr91 ** 2)",
                         [y, "irr91", "elev_km", "tri", "state_name"]))
    L = ld_frame()
    for tg in ["cs0913", "w1517"]:
        for filt in [None, "area_stable_" + tg]:
            sub = L[L["ok_1991_2011"]]
            if filt:
                sub = sub[sub[filt]]
            for y in ["D0_05", "D1"]:
                jobs.append(("Q4 d{} to {} {}".format(y, tg, "area-stable" if filt else "all"), sub,
                             "d_{}_{} ~ dirr_1991_2011{} + C(state_name)".format(y, tg, TER), "dirr_1991_2011",
                             ["d_{}_{}".format(y, tg), "dirr_1991_2011", "elev_km", "tri", "state_name"]))
    w("| test | b | state SE | G | n | p normal | p t(G-1) | p wild cluster |")
    w("|---|---|---|---|---|---|---|---|")
    for lab, df, f, term, keep in jobs:
        r = wcr(df, f, term, "state_name", keep)
        w("| {} | {:.3f} | {:.3f} | {} | {} | {:.3f} | {:.3f} | {:.3f} |".format(
            lab, r["b"], r["se"], r["G"], r["n"], r["p_norm"], r["p_tG1"], r["p_wcr"]))
        rec("infer", lab, **r)

# ================================================================== Q1 extra
if "q1x" in SECTIONS:
    w("")
    w("## R2. Q1 extras")
    dy = pd.read_csv(OUTV + "/div_district_year.csv")
    w("")
    w("Extension-only window: 2020-21 and 2021-22, flagged state-years dropped, six crops removed, "
      "district needs both years.")
    x = dy[dy["year_start"].isin([2020, 2021]) & ~dy["flag_sy"]]
    g = x.groupby("district_key").agg(D1=("D1_d6", "mean"), D0=("D0_d6", "mean"), D0_05=("D0_05_d6", "mean"),
                                      E=("E_d6", "mean"), ny=("year_start", "nunique"))
    g = g[g["ny"] == 2]
    xx = d0.drop(columns=[c for c in ["D1", "D0", "D0_05", "E"] if c in d0]).merge(
        g, left_on="district_key", right_index=True, how="inner")
    for y in ["D1", "D0_05", "D0", "E"]:
        r = quadfit(xx, y, "irr_share")
        w("- {}: {}".format(y, rq(r)))
        rec("q1x", "ext-only " + y, **{k: r[k] for k in ["n", "b2", "p2_par", "p2_st", "peak"]})
    # 2020-21 alone vs 2019-20 alone on the same districts
    for yr in [2019, 2020, 2021]:
        x = dy[(dy["year_start"] == yr) & ~dy["flag_sy"]].set_index("district_key")[["D1_d6"]].rename(
            columns={"D1_d6": "D1"})
        xx = d0.drop(columns=["D1"], errors="ignore").merge(x, left_on="district_key", right_index=True, how="inner")
        r = quadfit(xx, "D1", "irr_share")
        w("- single year {}-{}, D1 six removed: {}".format(yr, str(yr + 1)[-2:], rq(r)))
    w("")
    w("Leave-one-state-out on the main (b) D1 bend (flags dropped, six removed):")
    df = with_win(d0, "w1721", "flags_drop6")
    base = quadfit(df, "D1", "irr_share")
    out = []
    for s in sorted(df["state_name"].unique()):
        r = quadfit(df[df["state_name"] != s], "D1", "irr_share")
        out.append((s, r["b2"], r["p2_st"], r["peak"]))
    o = pd.DataFrame(out, columns=["state", "b2", "p_st", "peak"])
    w("b2 range {:.2f} to {:.2f}; max state p {:.3f} (dropping {}); peak range {:.2f} to {:.2f}".format(
        o.b2.min(), o.b2.max(), o.p_st.max(), o.loc[o.p_st.idxmax(), "state"], o.peak.min(), o.peak.max()))
    w("Most influential: " + "; ".join("{} {:.2f}".format(a, b) for a, b in
                                        o.reindex((o.b2 - base["b2"]).abs().sort_values(ascending=False).index)
                                        [["state", "b2"]].head(4).values))
    w("")
    w("Binned means (state FE + nothing else; * state p < .05), Antyodaya share, site sample:")
    for t, v in [("full", "all"), ("w1721", "flags_drop6")]:
        dfb = with_win(d0, t, v).dropna(subset=["D1", "irr_share"]).copy()
        for nb in [5, 10]:
            dfb["q_"] = pd.qcut(dfb["irr_share"], nb, labels=False, duplicates="drop")
            edges = dfb.groupby("q_")["irr_share"].max().round(2).tolist()
            for y in ["D1", "D0_05"]:
                mq = smf.ols("{} ~ C(q_) + C(state_name)".format(y), dfb).fit(
                    cov_type="cluster", cov_kwds={"groups": pd.factorize(dfb["state_name"])[0]})
                w("- {} {} bins={} (upper edges {}): {}".format(t, y, nb, edges, ", ".join(
                    "{:+.2f}{}".format(mq.params["C(q_)[T.{}]".format(k)],
                                       "*" if mq.pvalues["C(q_)[T.{}]".format(k)] < .05 else "") for k in range(1, nb))))
    # linear spline with a knot at 0.30: slope below and above, state clusters
    for t, v in [("full", "all"), ("w1721", "flags_drop6")]:
        dfb = with_win(d0, t, v)
        dfb["lo"] = dfb["irr_share"].clip(upper=0.3)
        dfb["hi"] = (dfb["irr_share"] - 0.3).clip(lower=0)
        for y in ["D1", "D0_05"]:
            r1 = wcr(dfb, "{} ~ lo + hi + C(state_name)".format(y), "lo", "state_name", [y, "lo", "hi", "state_name"])
            r2 = wcr(dfb, "{} ~ lo + hi + C(state_name)".format(y), "hi", "state_name", [y, "lo", "hi", "state_name"])
            w("- spline knot 0.30, {} {}: slope below {:.2f} (t(G-1) p {:.3f}, wild {:.3f}); above {:.2f} (t(G-1) p {:.3f}, wild {:.3f})".format(
                t, y, r1["b"], r1["p_tG1"], r1["p_wcr"], r2["b"], r2["p_tG1"], r2["p_wcr"]))
            rec("q1x", "spline {} {} below".format(t, y), **r1)
            rec("q1x", "spline {} {} above".format(t, y), **r2)
    # does the D0_05 bend survive a common-district, peak-in-the-sample check (share above peak)
    for y in ["D1", "D0_05"]:
        r = quadfit(df, y, "irr_share")
        w("Share of districts above the {} peak ({:.2f}): {:.0%}".format(y, r["peak"],
                                                                     (r["s"]["irr_share"] > r["peak"]).mean()))

# ================================================================== Q2
if "q2" in SECTIONS:
    w("")
    w("## R3. Q2: canal vs groundwater, tank")
    s = c2.dropna(subset=KD + ["D1"])
    w("")
    w("Spread of the source shares (share of sown area) in the spec D sample, n {}:".format(len(s)))
    for k in ["sh_can", "sh_gw", "sh_tk", "sh_ro"]:
        w("- {}: mean {:.3f}, sd {:.3f}, within-state sd {:.3f}, p90 {:.3f}, max {:.3f}".format(
            k, s[k].mean(), s[k].std(), (s[k] - s.groupby("state_name")[k].transform("mean")).std(),
            s[k].quantile(.9), s[k].max()))
    m = smf.ols(FD.format("D1"), s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s["state_name"])[0]})
    ws = {k: (s[k] - s.groupby("state_name")[k].transform("mean")).std() for k in ["sh_can", "sh_gw", "sh_tk", "sh_ro"]}
    # marginal effect of a 1 within-state-SD change at the mean for each source
    for k in ["sh_can", "sh_gw"]:
        me = m.params[k] + 2 * m.params["I({} ** 2)".format(k)] * s[k].mean()
        w("- D1 effect of +1 within-state SD of {} at its mean: {:.3f}".format(k, me * ws[k]))
    w("- D1 effect of +1 within-state SD of sh_tk: {:.3f}; of sh_ro: {:.3f}".format(
        m.params["sh_tk"] * ws["sh_tk"], m.params["sh_ro"] * ws["sh_ro"]))
    w("")
    w("Tank share by state (spec D sample): n with tank share > 0.05, state mean tank share")
    tk = s.groupby("state_name")["sh_tk"].agg(["size", "mean", lambda v: (v > 0.05).sum()])
    tk.columns = ["n", "mean_tank", "n_gt_05"]
    w(tk.sort_values("mean_tank", ascending=False).head(8).round(3).to_string())
    w("")
    w("Leave-one-state-out, spec D tank coefficient:")
    for y in ["D1", "D0_05"]:
        rows = []
        sy = c2.dropna(subset=KD + [y])
        for st_ in sorted(sy["state_name"].unique()):
            ss = sy[sy["state_name"] != st_]
            mm = smf.ols(FD.format(y), ss).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(ss["state_name"])[0]})
            rows.append((st_, mm.params["sh_tk"], mm.pvalues["sh_tk"]))
        o = pd.DataFrame(rows, columns=["state", "b", "p"])
        w("- {}: range {:.2f} to {:.2f}; max state p {:.3f}; biggest moves: {}".format(
            y, o.b.min(), o.b.max(), o.p.max(),
            "; ".join("{} {:.2f}".format(a, b) for a, b in o.reindex(o.b.sub(o.b.median()).abs()
                                                                    .sort_values(ascending=False).index)[["state", "b"]].head(4).values)))
        for drop in [["TAMIL NADU"], ["ANDHRA PRADESH", "TELANGANA"], ["KARNATAKA"],
                     ["TAMIL NADU", "ANDHRA PRADESH", "TELANGANA", "KARNATAKA"]]:
            ss = sy[~sy["state_name"].isin(drop)]
            r = wcr(ss, FD.format(y), "sh_tk", "state_name", KD + [y])
            w("  drop {}: {}".format("+".join(d.title() for d in drop), fmt_inf(r)))
            rec("q2", "tank drop {} {}".format("+".join(drop), y), **r)
    for thr in [0.3, 0.15]:
        ss = c2[c2["sh_tk"] <= thr]
        r = wcr(ss, FD.format("D1"), "sh_tk", "state_name", KD + ["D1"])
        w("- tank, D1, districts with tank share <= {} only ({} dropped): {}".format(thr, int((c2["sh_tk"] > thr).sum()), fmt_inf(r)))
    ss = c2[~c2["state_name"].isin(["TAMIL NADU", "ANDHRA PRADESH", "TELANGANA", "WEST BENGAL", "JHARKHAND"])]
    r = wcr(ss, FD.format("D1"), "sh_tk", "state_name", KD + ["D1"])
    w("- tank, D1, without the four highest-tank states (TN, AP+TG, WB, JH): {}".format(fmt_inf(r)))
    # tank alone with state FE + terrain (no other sources), and tank among irrigated (composition)
    for y in ["D1"]:
        r = wcr(c2, "{} ~ sh_tk{} + C(state_name)".format(y, TER), "sh_tk", "state_name", [y, "sh_tk", "elev_km", "tri", "state_name"])
        w("- tank share alone + terrain + state FE, {}: {}".format(y, fmt_inf(r)))
        r = wcr(c2, "{} ~ sh_tk + irr + I(irr**2){} + C(state_name)".format(y, TER), "sh_tk", "state_name",
                [y, "sh_tk", "irr", "elev_km", "tri", "state_name"])
        w("- tank share + total irrigated quadratic, {}: {}".format(y, fmt_inf(r)))
    w("")
    w("Canal-majority vs groundwater-majority districts: support of the irrigated share")
    for nm, cond in [("gw-majority", c2["gwc"] > 0.5), ("canal-majority", c2["canc"] > 0.5),
                     ("neither", (c2["gwc"] <= 0.5) & (c2["canc"] <= 0.5))]:
        x = c2[cond]
        w("- {}: n {}, states {}, irr share p10/p50/p90 {:.2f}/{:.2f}/{:.2f}, within-state sd {:.3f}; "
          "share above 0.5 {:.0%}".format(nm, len(x), x["state_name"].nunique(), *x["irr"].quantile([.1, .5, .9]).values,
                                           (x["irr"] - x.groupby("state_name")["irr"].transform("mean")).std(),
                                           (x["irr"] > 0.5).mean()))
    # pooled interaction on the dichotomy, and minimum detectable difference
    c2["canmaj"] = (c2["canc"] > 0.5).astype(float)
    for y in ["D1", "D0_05"]:
        f = "{} ~ irr + I(irr**2) + canmaj + irr:canmaj + I(irr**2):canmaj{} + C(state_name)".format(y, TER)
        r = wcr(c2, f, "I(irr ** 2):canmaj", "state_name", [y, "irr", "canmaj", "elev_km", "tri", "state_name"])
        w("- {}: canal-majority minus rest, squared term: {}; minimum detectable difference (80% power, 5%) about {:.1f}".format(
            y, fmt_inf(r), 2.8 * r["se"]))
        rec("q2", "canmaj interaction " + y, **r)

# ================================================================== Q3
if "q3" in SECTIONS:
    w("")
    w("## R4. Q3: irrigation measured in 1991")
    # ---- 1991 village variants
    N121 = head_exec(SRC + "121_census_irrigation_panel.py",
                     "# ------------------------------------------------------------------ village tables")
    key, univ, R_, PM = N121["key"], N121["univ"], N121["R"], N121["PLAUS_MULT"]
    d = pd.read_stata(R_ + "shrug-vd91-dta/pc91_vd_clean_shrid.dta")
    p = "pc91_vd_"
    cols = ["canal_govt", "canal_pvt", "well_wo_el", "well_w_el", "tw_wo_el", "tw_w_el", "tank_irr",
            "lake_irr", "river_irr", "w_fall", "oth_irr"]
    S = pd.DataFrame({c: pd.to_numeric(d[p + c], errors="coerce") for c in cols})
    tot_f = pd.to_numeric(d[p + "tot_irr"], errors="coerce")
    un = pd.to_numeric(d[p + "un_irr"], errors="coerce")
    own = pd.to_numeric(d[p + "area"], errors="coerce")
    ssum = S.sum(axis=1, min_count=1)
    w("")
    w("1991 villages: total field present {:.1%}; source sum present {:.1%}; total 0 or blank with sources > 0: {:.1%}; "
      "both present and source sum > total + 0.5: {:.1%}; both present and total > source sum + 0.5: {:.1%}".format(
          tot_f.notna().mean(), ssum.notna().mean(), ((tot_f.fillna(0) == 0) & (ssum > 0)).mean(),
          (tot_f.notna() & (ssum > tot_f + 0.5) & (tot_f > 0)).mean(), (ssum.notna() & (tot_f > ssum + 0.5)).mean()))
    VAR = {"main (max of total, sources)": pd.Series(np.fmax(tot_f.values, ssum.values)),
           "reported total only": tot_f,
           "sum of sources only": ssum.where(ssum.notna(), tot_f.where(tot_f == 0))}
    v0 = pd.DataFrame({"shrid2": d.shrid2, "un": un, "own": own}).merge(
        key[["shrid2", "st", "dt", "ref_ha_pc11"]], on="shrid2", how="left")
    alt = {}
    for nm, tot in VAR.items():
        v = v0.copy()
        v["tot"] = tot.values
        v = v.dropna(subset=["st"])
        ref = np.fmax(v.own.fillna(0), v.ref_ha_pc11.fillna(0))
        has = v.tot.notna() & v.un.notna()
        a = v.tot.fillna(0) + v.un.fillna(0)
        valid = has & ~((a > PM * ref) & (a > 5))
        vv = v[valid]
        g = vv.groupby(["st", "dt"]).agg(irr=("tot", "sum"), un=("un", "sum"))
        g["share"] = g.irr / (g.irr + g.un)
        alt[nm] = g["share"]
        w("- {}: national share {:.3f} (valid villages {:,})".format(nm, vv.tot.sum() / (vv.tot.sum() + vv.un.sum()), int(valid.sum())))
    A = pd.DataFrame(alt).reset_index().rename(columns={"st": "pc11_state_id", "dt": "pc11_district_id"})
    A.columns = ["pc11_state_id", "pc11_district_id", "i_main", "i_rep", "i_src"]
    c3 = c3b.merge(A, on=["pc11_state_id", "pc11_district_id"], how="left")
    w("- check main variant vs panel c1991_irr_share: max abs diff {:.2e}".format((c3.i_main - c3.irr91).abs().max()))
    A.to_csv(OUT + "/alt1991_shares.csv", index=False)
    w("- correlation with Antyodaya 2019-20 share: main {:.3f}, reported-only {:.3f}, sources-only {:.3f}; "
      "with census 2001 share: {:.3f} / {:.3f} / {:.3f}".format(
          *[c3[[k, "irr_share"]].corr().iloc[0, 1] for k in ["i_main", "i_rep", "i_src"]],
          *[c3[[k, "c2001_irr_share"]].corr().iloc[0, 1] for k in ["i_main", "i_rep", "i_src"]]))
    w("- share of usable districts where the repair raises the 1991 share by > 0.10: {:.0%}; by state (top): {}".format(
        ((c3.i_main - c3.i_rep) > 0.10).mean(),
        "; ".join("{} {:.2f}".format(a, b) for a, b in
                  c3.assign(gap=c3.i_main - c3.i_rep).groupby("state_name")["gap"].mean()
                  .sort_values(ascending=False).head(6).items())))
    W3 = [("full", "all", "full"), ("w1517", "all", "(a)"), ("w1721", "flags_drop6", "(b)")]
    w("")
    w("| 1991 measure | window | outcome | n | G | linear (p par/st) | squared (SE par; st) p par/st | peak | p wild (squared) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for xv, xl in [("i_main", "main"), ("i_rep", "reported total only"), ("i_src", "sum of sources only")]:
        for t, v, tl in W3:
            df = with_win(c3, t, v)
            for y in ["D1", "D0_05", "D0"]:
                r = quadfit(df, y, xv, TER)
                wb = wcr(df, r["f"], "I({} ** 2)".format(xv), "state_name", r["keep"])
                w("| {} | {} | {} | {} | {} | {:.2f} ({:.3f}/{:.3f}) | {:.2f} ({:.2f}; {:.2f}) {:.3f}/{:.3f} | {} | {:.3f} |".format(
                    xl, tl, y, r["n"], r["G"], r["b1"], r["p1_par"], r["p1_st"], r["b2"], r["se2_par"], r["se2_st"],
                    r["p2_par"], r["p2_st"], "{:.2f}".format(r["peak"]) if pd.notna(r["peak"]) else "none", wb["p_wcr"]))
                rec("q3", "{} {} {}".format(xl, tl, y), n=r["n"], b1=r["b1"], b2=r["b2"], p2_st=r["p2_st"],
                    peak=r["peak"], p_wcr=wb["p_wcr"])
    # ---- dropping weak 1991 state-rounds
    w("")
    w("Dropping weak 1991 state-rounds (main 1991 measure, D1 and D0_05):")
    sr = pd.read_csv(REPO + "/outputs/revision_nov8/census_irrigation/census_irrigation_state_round.csv")
    piv = sr.pivot_table(index="state_name", columns="round", values=["irr_share", "cov_area"])
    ratio = (piv[("irr_share", 1991)] / piv[("irr_share", 2001)])
    weak = sorted(ratio[ratio < 0.67].index.tolist())
    lowcov = sorted(piv[piv[("cov_area", 1991)] < 0.8].index.tolist())
    w("- states with 1991/2001 share ratio < 0.67: {}".format(", ".join("{} ({:.2f})".format(s_, ratio[s_]) for s_ in weak)))
    w("- states with 1991 area coverage < 80%: {}".format(", ".join(lowcov)))
    up = lambda L_: [s_.upper().replace("JAMMU KASHMIR", "JAMMU AND KASHMIR") for s_ in L_]
    for dl, drop in [("Rajasthan", ["RAJASTHAN"]), ("Rajasthan + Madhya Pradesh", ["RAJASTHAN", "MADHYA PRADESH"]),
                     ("all ratio < 0.67", up(weak)), ("ratio < 0.67 or coverage < 80%", up(weak + lowcov))]:
        for t, v, tl in W3:
            df = with_win(c3[~c3["state_name"].isin(drop)], t, v)
            for y in ["D1", "D0_05"]:
                r = quadfit(df, y, "irr91", TER)
                wb = wcr(df, r["f"], "I(irr91 ** 2)", "state_name", r["keep"])
                w("- drop {}, {}, {}: {}; p wild {:.3f}".format(dl, tl, y, rq(r), wb["p_wcr"]))
                rec("q3", "drop {} {} {}".format(dl, tl, y), n=r["n"], b2=r["b2"], p2_st=r["p2_st"], peak=r["peak"], p_wcr=wb["p_wcr"])
    # ---- scale checks: ranks, bins, other measures on the same districts
    w("")
    w("Is the missing rise a units artefact? Same districts, full and (b) windows.")
    des = pd.read_csv(REPO + "/outputs/revision_nov8/irrigation_panel_district_year.csv")
    dd = des[des["year_start"].between(1998, 2000)].groupby("district_key")["irr_nsa"].mean().rename("irr_des98")
    c3 = c3.merge(dd, left_on="district_key", right_index=True, how="left")
    c3["irr_des98"] = c3["irr_des98"].clip(upper=1)
    MEAS = [("irr91", "census 1991"), ("c2001_irr_share", "census 2001"), ("c2011_irr_share", "census 2011"),
            ("irr_des98", "DES 1998-2000"), ("irr_share", "Antyodaya 2019-20")]
    cc_all = c3.dropna(subset=[m_ for m_, _ in MEAS])
    w("Common districts with all five measures: {} (1991 sample {}). States losing districts: {}".format(
        len(cc_all), len(c3), "; ".join("{} {}".format(a, b) for a, b in
                                        (c3["state_name"].value_counts() - cc_all["state_name"].value_counts())
                                        .fillna(c3["state_name"].value_counts()).sort_values(ascending=False).head(8).items())))
    SAMPLES = [("1991 sample, census 1991 and Antyodaya", c3.dropna(subset=["irr91", "irr_share"]),
                [MEAS[0], MEAS[4], MEAS[2]]),
               ("common five-measure sample", cc_all, MEAS)]
    for sl, cc, meas in SAMPLES:
        cc = cc.copy()
        for m_, _ in meas:
            cc["r_" + m_] = cc[m_].rank(pct=True)
        w("")
        w("Sample: {} (n {}).".format(sl, len(cc)))
        w("")
        w("| measure | window | outcome | level: b2, p st, peak | percentile rank: b1 (p st), b2 (p st), peak pct | quintile means vs Q1 (state FE, * state p<.05) |")
        w("|---|---|---|---|---|---|")
        for t, v, tl in [("full", "all", "full"), ("w1721", "flags_drop6", "(b)")]:
            df0 = with_win(cc, t, v)
            for y in ["D1", "D0_05"]:
                for m_, ml in meas:
                    df = df0.dropna(subset=[y, m_, "elev_km", "tri"]).copy()
                    r = quadfit(df, y, m_, TER)
                    rr = quadfit(df, y, "r_" + m_, TER)
                    df["q_"] = pd.qcut(df[m_], 5, labels=False, duplicates="drop")
                    mq = smf.ols("{} ~ C(q_){} + C(state_name)".format(y, TER), df).fit(
                        cov_type="cluster", cov_kwds={"groups": pd.factorize(df["state_name"])[0]})
                    qs = ", ".join("{:+.2f}{}".format(mq.params.get("C(q_)[T.{}]".format(k), np.nan),
                                                      "*" if mq.pvalues.get("C(q_)[T.{}]".format(k), 1) < .05 else "")
                                   for k in range(1, 5))
                    w("| {} | {} | {} | {:.2f}, {:.3f}, {} | {:.2f} ({:.3f}), {:.2f} ({:.3f}), {} | {} |".format(
                        ml, tl, y, r["b2"], r["p2_st"], "{:.2f}".format(r["peak"]) if pd.notna(r["peak"]) else "none",
                        rr["b1"], rr["p1_st"], rr["b2"], rr["p2_st"],
                        "{:.2f}".format(rr["peak"]) if pd.notna(rr["peak"]) else "none", qs))
                    rec("q3scale", "{} | {} {} {}".format(sl, ml, tl, y), b2=r["b2"], p2_st=r["p2_st"], peak=r["peak"],
                        rank_b1=rr["b1"], rank_p1=rr["p1_st"], rank_b2=rr["b2"], rank_p2=rr["p2_st"],
                        rank_peak=rr["peak"], quint=qs)
    cc = cc_all.copy()
    # mapping 1991 share to the 2019-20 scale
    mp = smf.ols("irr_share ~ irr91 + I(irr91**2)", cc).fit()
    cc["irr91_on19"] = mp.fittedvalues
    for y in ["D1", "D0_05"]:
        r = quadfit(with_win(cc, "full", "all"), y, "irr91_on19", TER)
        w("- 1991 share rescaled to the 2019-20 scale (fitted quadratic map), full window {}: {}".format(y, rq(r)))

# ================================================================== Q4
if "q4" in SECTIONS:
    w("")
    w("## R5. Q4: long differences")
    L = ld_frame()
    sub0 = L[L["ok_1991_2011"]].copy()
    w("")
    w("Correlation of the 1991-2011 change in irrigated share with the change in log gross cropped area: "
      "to 2009-13 {:.3f}, to 2015-19 {:.3f}; with the 1991 level {:.3f}".format(
          sub0[["dirr_1991_2011", "dlgca_cs0913"]].corr().iloc[0, 1],
          sub0[["dirr_1991_2011", "dlgca_w1517"]].corr().iloc[0, 1],
          sub0[["dirr_1991_2011", "c1991_irr_share"]].corr().iloc[0, 1]))
    # fixed-denominator count from crop-level APY
    N122 = head_exec(SRC + "122_irrigation_v2_diversity.py", "v_all = indices(base)")
    base = N122["base"]
    WIN = {"e": (1998, 2002), "cs0913": (2009, 2013), "w1517": (2015, 2019)}
    b = base[base["year_start"].between(1998, 2019)].copy()
    gy = b.groupby(["district_key", "year_start"])["area"].transform("sum")
    b["gca_y"] = gy
    early = b[b["year_start"].between(1998, 2002)].groupby("district_key")["gca_y"].agg(
        lambda s_: s_.groupby(level=0).first().mean() if False else s_.mean())
    # mean early GCA per district (district-year level)
    dyg = b.drop_duplicates(["district_key", "year_start"])
    early = dyg[dyg["year_start"].between(1998, 2002)].groupby("district_key")["gca_y"].mean()
    b["thr_fix"] = b["district_key"].map(early) * 0.005
    b["ge_fix"] = (b["area"] >= b["thr_fix"]).astype(int)
    b["ge_rel"] = (b["area"] >= 0.005 * b["gca_y"]).astype(int)
    b["any"] = 1
    cnt = b.groupby(["district_key", "year_start"]).agg(fix=("ge_fix", "sum"), rel=("ge_rel", "sum"),
                                                         d0=("any", "sum")).reset_index()
    for tg, (y0, y1) in WIN.items():
        m = cnt[cnt["year_start"].between(y0, y1)].groupby("district_key").agg(
            fix=("fix", "mean"), rel=("rel", "mean"), ny=("year_start", "nunique"))
        m = m[m["ny"] >= 3]
        L = L.merge(m[["fix", "rel"]].add_suffix("_" + tg), left_on="district_key", right_index=True, how="left")
    for tg in ["cs0913", "w1517"]:
        L["d_fix_" + tg] = L["fix_" + tg] - L["fix_e"]
        L["d_rel_" + tg] = L["rel_" + tg] - L["rel_e"]
    sub0 = L[L["ok_1991_2011"]].copy()
    w("Check: rebuilt relative-threshold change equals d_D0_05 to 2009-13: max abs diff {:.2e}".format(
        (sub0["d_rel_cs0913"] - sub0["d_D0_05_cs0913"]).abs().max()))
    w("")
    w("| outcome | diversity to | sample | control | b dirr | state SE | p t(G-1) | p wild | n |")
    w("|---|---|---|---|---|---|---|---|---|")
    for tg in ["cs0913", "w1517"]:
        for filt in [None, "area_stable_" + tg]:
            sub = sub0[sub0[filt]] if filt else sub0
            for y, yl, ctl in [("d_D0_05_" + tg, "D0 at 0.5% (as in note)", ""),
                               ("d_D0_05_" + tg, "D0 at 0.5%", " + dlgca_" + tg),
                               ("d_fix_" + tg, "count >= 0.5% of EARLY area (fixed ha)", ""),
                               ("d_fix_" + tg, "count >= 0.5% of EARLY area (fixed ha)", " + dlgca_" + tg),
                               ("dlgca_" + tg, "change in log GCA", "")]:
                f = "{} ~ dirr_1991_2011{}{} + C(state_name)".format(y, ctl, TER)
                keep = [y, "dirr_1991_2011", "elev_km", "tri", "state_name"] + ([ctl.strip(" +")] if ctl else [])
                r = wcr(sub, f, "dirr_1991_2011", "state_name", keep)
                w("| {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
                    yl, tg, "area-stable" if filt else "all", ctl.strip(" +") or "-", r["b"], r["se"], r["p_tG1"], r["p_wcr"], r["n"]))
                rec("q4", "{} {} {} {}".format(yl, tg, "area-stable" if filt else "all", ctl.strip(" +") or "-"), **r)
    w("")
    w("Leave-one-state-out, dD0_05 on dirr 1991-2011 (all districts):")
    for tg in ["cs0913", "w1517"]:
        rows = []
        y = "d_D0_05_" + tg
        ss0 = sub0.dropna(subset=[y, "dirr_1991_2011", "elev_km", "tri"])
        for st_ in sorted(ss0["state_name"].unique()):
            ss = ss0[ss0["state_name"] != st_]
            mm = smf.ols("{} ~ dirr_1991_2011{} + C(state_name)".format(y, TER), ss).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(ss["state_name"])[0]})
            rows.append((st_, mm.params["dirr_1991_2011"], mm.pvalues["dirr_1991_2011"]))
        o = pd.DataFrame(rows, columns=["state", "b", "p"])
        w("- to {}: b range {:.2f} to {:.2f}; max state p {:.3f}; states whose removal moves b most: {}".format(
            tg, o.b.min(), o.b.max(), o.p.max(),
            "; ".join("{} {:.2f} (p {:.3f})".format(a, b_, c) for a, b_, c in
                      o.reindex(o.b.sub(o.b.median()).abs().sort_values(ascending=False).index).head(4).values)))
    # baseline-level control and 1991 repair variants
    w("")
    for tg in ["cs0913", "w1517"]:
        y = "d_D0_05_" + tg
        r = wcr(sub0, "{} ~ dirr_1991_2011 + c1991_irr_share{} + C(state_name)".format(y, TER), "dirr_1991_2011",
                "state_name", [y, "dirr_1991_2011", "c1991_irr_share", "elev_km", "tri", "state_name"])
        w("- adding the 1991 level, to {}: {}".format(tg, fmt_inf(r)))
        r = wcr(sub0, "{} ~ dirr_1991_2011 + D0_05_e{} + C(state_name)".format(y, TER), "dirr_1991_2011",
                "state_name", [y, "dirr_1991_2011", "D0_05_e", "elev_km", "tri", "state_name"])
        w("- adding the early D0 at 0.5% (regression to the mean), to {}: {}".format(tg, fmt_inf(r)))
        r = wcr(sub0, "{} ~ dirr_1991_2011{} + C(state_name)".format(y, TER), "dirr_1991_2011", "state_name",
                [y, "dirr_1991_2011", "elev_km", "tri", "state_name"])
    # 1991 repair variants in the long difference (needs alt1991_shares.csv from section q3)
    if os.path.exists(OUT + "/alt1991_shares.csv"):
        A = pd.read_csv(OUT + "/alt1991_shares.csv")
        LA = sub0.merge(A, on=["pc11_state_id", "pc11_district_id"], how="left")
        for xv, xl in [("i_rep", "reported total only"), ("i_src", "sum of sources only")]:
            LA["dx"] = LA["c2011_irr_share"] - LA[xv]
            for tg in ["cs0913", "w1517"]:
                y = "d_D0_05_" + tg
                r = wcr(LA, "{} ~ dx{} + C(state_name)".format(y, TER), "dx", "state_name",
                        [y, "dx", "elev_km", "tri", "state_name"])
                w("- 1991 measured as {}, dD0_05 to {}: {}".format(xl, tg, fmt_inf(r)))
                rec("q4", "repair {} {}".format(xl, tg), **r)
                ss = LA[~LA["state_name"].isin(["ODISHA", "ASSAM", "SIKKIM", "GOA"])]
                r = wcr(ss, "{} ~ dx{} + C(state_name)".format(y, TER), "dx", "state_name",
                        [y, "dx", "elev_km", "tri", "state_name"])
                w("  same, without Odisha/Assam/Sikkim/Goa (where the variants differ): {}".format(fmt_inf(r)))
                ss = ss.assign(dx2=ss["dirr_1991_2011"])
                r = wcr(ss, "{} ~ dx2{} + C(state_name)".format(y, TER), "dx2", "state_name",
                        [y, "dx2", "elev_km", "tri", "state_name"])
                w("  main 1991 measure on that same reduced sample: {}".format(fmt_inf(r)))
    for dl, drop in [("Rajasthan", ["RAJASTHAN"]), ("Rajasthan + Madhya Pradesh", ["RAJASTHAN", "MADHYA PRADESH"])]:
        for tg in ["cs0913", "w1517"]:
            y = "d_D0_05_" + tg
            r = wcr(sub0[~sub0["state_name"].isin(drop)], "{} ~ dirr_1991_2011{} + C(state_name)".format(y, TER),
                    "dirr_1991_2011", "state_name", [y, "dirr_1991_2011", "elev_km", "tri", "state_name"])
            w("- drop {}, dD0_05 to {}: {}".format(dl, tg, fmt_inf(r)))
    # BH across the 32 total-change slopes in the note
    COEF = pd.read_csv(OUTV + "/irrigation_v2_coefs.csv")
    T = COEF[(COEF["q"] == "Q4") & (COEF["tag"] == "total")].copy()
    for cl in ["b1_p_par", "b1_p_st"]:
        pv = T[cl].values
        o = np.argsort(pv)
        qv = np.empty(len(pv))
        qv[o] = np.minimum.accumulate((pv[o] * len(pv) / np.arange(1, len(pv) + 1))[::-1])[::-1]
        T["q_" + cl] = np.minimum(qv, 1)
    w("")
    w("BH across the {} total-change slopes (4 outcomes x 2 census spans x 2 windows x 2 samples):".format(len(T)))
    for _, r in T.sort_values("b1_p_st").head(6).iterrows():
        w("- {} {} {} {}: b {:.2f}, p {:.3f}/{:.3f}, q {:.3f}/{:.3f}".format(
            r["y"], r["census"], r["div"], r["filt"], r["b1"], r["b1_p_par"], r["b1_p_st"],
            r["q_b1_p_par"], r["q_b1_p_st"]))
    # 2001-2011 total change with the clean 2001 measure (no repair issue)
    L["dirr_2001_2011"] = L["c2011_irr_share"] - L["c2001_irr_share"]
    s01 = L[L["c2001_district_usable"] & L["c2011_district_usable"]]
    for tg in ["cs0913", "w1517"]:
        y = "d_D0_05_" + tg
        r = wcr(s01, "{} ~ dirr_2001_2011{} + C(state_name)".format(y, TER), "dirr_2001_2011", "state_name",
                [y, "dirr_2001_2011", "elev_km", "tri", "state_name"])
        w("- 2001-2011 change (no 1991 repair involved), dD0_05 to {}: {}".format(tg, fmt_inf(r)))

# ================================================================== multiple testing
if "mt" in SECTIONS:
    w("")
    w("## R6. Multiple testing across source-split tests")
    COEF = pd.read_csv(OUTV + "/irrigation_v2_coefs.csv")
    rows = []
    for _, r in COEF.iterrows():
        q, tag = r["q"], str(r["tag"])
        if q == "Q2" and (tag.startswith("D sh_") or tag.startswith("D I(") or tag.startswith("C interaction")
                          or tag.startswith("B gw share")):
            p_par = r["b1_p_par"] if pd.notna(r.get("b1_p_par")) else r["b2_p_par"]
            p_st = r["b1_p_st"] if pd.notna(r.get("b1_p_st")) else r["b2_p_st"]
            rows.append(("Q2", tag + " | " + r["y"], p_par, p_st))
        elif q == "Q2ay" and tag.startswith("C interaction"):
            rows.append(("Q2ay", tag + " | " + r["y"], r["b2_p_par"], r["b2_p_st"]))
        elif q == "Q3" and tag.startswith("sources"):
            rows.append(("Q3", "{} | {} | {}".format(tag, r["y"], r["window"]), r["b1_p_par"], r["b1_p_st"]))
        elif q == "Q4" and (tag == "by source" or tag == "canal - gw"):
            if tag == "by source" and not any(str(r["x"]).startswith(k) for k in ["dcan", "dgw", "dtk"]):
                continue
            rows.append(("Q4", "{} {} | {} | {} {}".format(tag, r["x"], r["y"], r["census"], r["filt"]),
                         r["b1_p_par"], r["b1_p_st"]))
    M = pd.DataFrame(rows, columns=["block", "test", "p_par", "p_st"]).dropna(subset=["p_par"])

    def bh(p):
        p = np.asarray(p, float)
        n = len(p)
        o = np.argsort(p)
        q = np.empty(n)
        q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
        return np.minimum(q, 1)

    M["q_par"] = bh(M["p_par"])
    M["q_st"] = bh(M["p_st"].fillna(1))
    M["bonf_st"] = np.minimum(1, M["p_st"].fillna(1) * len(M))
    w("Source-split tests collected: {} (Q2 {}, Q2ay {}, Q3 {}, Q4 {}).".format(
        len(M), *(M["block"].eq(b_).sum() for b_ in ["Q2", "Q2ay", "Q3", "Q4"])))
    w("Raw p < 0.05: parent {} , state {}. Expected under the global null: {:.1f}.".format(
        int((M.p_par < .05).sum()), int((M.p_st < .05).sum()), .05 * len(M)))
    w("BH q < 0.05: parent {}, state {}; BH q < 0.10: parent {}, state {}.".format(
        int((M.q_par < .05).sum()), int((M.q_st < .05).sum()), int((M.q_par < .10).sum()), int((M.q_st < .10).sum())))
    w("")
    w("Tests surviving BH q < 0.10 on either clustering:")
    for _, r in M[(M.q_par < .10) | (M.q_st < .10)].sort_values("p_st").iterrows():
        w("- {} | {}: p {:.4f}/{:.4f}, q {:.3f}/{:.3f}".format(r["block"], r["test"], r["p_par"], r["p_st"], r["q_par"], r["q_st"]))
    for blk in ["Q4"]:
        mm = M[M.block == blk].copy()
        mm["q_st_blk"] = bh(mm["p_st"].fillna(1))
        mm["q_par_blk"] = bh(mm["p_par"])
        w("Within {} alone ({} tests): BH q<0.10 parent {}, state {}.".format(
            blk, len(mm), int((mm.q_par_blk < .1).sum()), int((mm.q_st_blk < .1).sum())))
    M.to_csv(OUT + "/referee_multiple_testing.csv", index=False)

pd.DataFrame(RES).to_csv(OUT + "/referee_results_{}.csv".format("_".join(SECTIONS)), index=False)
open(OUT + "/referee_raw_{}.md".format("_".join(SECTIONS)), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
print("done")
