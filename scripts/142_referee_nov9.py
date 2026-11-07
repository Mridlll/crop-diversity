"""
142_referee_nov9.py

Hostile referee pass on the 8 Nov revision (commit a4cde1d). Read-only on every earlier
output; writes only to outputs/revision_nov8/referee_nov9/.

  H  PRIMARY. Census 2011 weekly haat reach and the crop mix (ANALYSIS_INDEX C11).
     H0 the variable: label and coding of pc11_vd_wkl_haat in shrug-vd11, shrid
        aggregation, rebuild of wk11_all from villages and check against script 136.
     H1 reproduce: D0, D0 at 0.1 / 0.5 / 1 percent, D1, D2, evenness; parent SE,
        state t(G-1), restricted wild cluster bootstrap by state (Webb, 9,999).
     H2 leave one state out (D0 at 0.5 percent, D1).
     H3 controls on top of the site set (irrigation and its square, log holding, ST share,
        night lights, non-farm establishments, connectivity, state FE): census
        mandi/regular reach, log rural population density, village size (population-
        weighted mean log village population, share of rural people in villages of
        5,000+), Mission Antyodaya mandi and regular-market reach, MA haat reach; all.
     H4 drop districts with weak 2011 village-directory coverage (< 0.95, < 0.90 of
        rural population).
     H5 the six crops lost in the 2020-21 crop-list break removed.
     H6 alternative definitions: share of villages (unweighted); haat-only villages (haat
        and no mandi/regular market); haat-and-mandi villages; the two jointly with
        mandi-only.
     H7 windows: 1998-99..2002-03, 2009-10..2013-14, 2015-16..2017-18, 2017-18..2021-22
        (flags dropped, drop6).
     H8 Benjamini-Hochberg across the outcome family (wild p).
  S1 B18 irrigation knot: reproduce knot 0.43 and the slope above 0.50; break: add the
     markets-page controls (holding, ST share, night lights, non-farm, connectivity).
  S2 B19 attenuation: reproduce the within-state correlation and LUS signal share;
     break: winsorised changes, rank correlation, trimming the largest LUS changes.
  S3 C10 haat threshold collapse at 0.5 percent: reproduce; break: floors between 0.2
     and 0.5 percent and absolute-hectare floors.
  S4 E11 soybean within-minus-across edge: reproduce 0.094; break: leave one event unit
     out, restricted wild cluster bootstrap by state on the contrast.
  S5 E13 common-crop border gap: reproduce M1-M3; break: leave one state out, drop
     cotton, the 1998-2002 window with wild p.

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/142_referee_nov9.py [H S1 S2 S3 S4 S5]
"""
import contextlib
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = r"D:/crop-diversity"
OUT = REPO + "/outputs/revision_nov8/referee_nov9"
os.makedirs(OUT, exist_ok=True)
SHRUG = r"D:/SHRUG_2.1_Data/extracted"
DKEY = ["pc11_state_id", "pc11_district_id"]
PARTS = sys.argv[1:] or ["H", "S1", "S2", "S3", "S4", "S5"]
LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


def flush(tag):
    open(OUT + "/raw_{}.md".format(tag), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    LOG.clear()


WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])


def wcr_arrays(X, y, j, g, B_=9999, seed=20261109):
    """Restricted wild cluster bootstrap on arrays (X already FE-partialled or with
    dummies). Returns b, CR1 se, t(G-1) p, wild p."""
    rng = np.random.default_rng(seed)
    g = pd.factorize(g)[0]
    G = g.max() + 1
    n, k = X.shape

    def cr1(u, h):
        S = np.zeros((G,) + u.shape[1:])
        np.add.at(S, g, h[:, None] * u if u.ndim == 2 else h * u)
        return np.sqrt(G / (G - 1) * (n - 1) / (n - k) * (S ** 2).sum(axis=0))

    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    h = X @ XtXi[:, j]
    se = cr1(y - X @ b, h)
    t0 = b[j] / se
    Xr = np.delete(X, j, axis=1)
    fr = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
    ur = y - fr
    V = WEBB[rng.integers(0, 6, size=(G, B_))]
    Ys = fr[:, None] + ur[:, None] * V[g, :]
    bs = XtXi @ (X.T @ Ys)
    ts = bs[j] / cr1(Ys - X @ bs, h)
    return dict(b=float(b[j]), se=float(se), G=int(G), n=int(n),
                p_st=float(2 * stats.t.sf(abs(t0), G - 1)), p_wcr=float(np.mean(np.abs(ts) >= abs(t0))))


# =====================================================================================
# H  census 2011 haat
# =====================================================================================
if "H" in PARTS:
    from _mkt2_common import apy_base, fit_par, load_104, wcr  # noqa: E402

    with contextlib.redirect_stdout(io.StringIO()):
        ns = load_104()
    X = ns["prep"]("W")
    CTRL = ns["CTRL"]
    BASEKEEP = ["mean_holding_ha", "log_viirs", "shrug_key", "state_name"]
    with contextlib.redirect_stdout(io.StringIO()):
        a = apy_base()
    base = a["base"]
    DROP6 = a["DROP6"]
    base = base[base["year_start"].between(1998, 2019)]
    base = base[base["district_key"].isin(X["district_key"])]

    def idx(dy, tag):
        g = dy.groupby(["district_key", "year_start"])["area"]
        p = dy["area"] / g.transform("sum")
        t = dy.assign(p=p, plnp=-p * np.log(p), f01=(p >= 0.001).astype(int),
                      f05=(p >= 0.005).astype(int), f1=(p >= 0.01).astype(int), p2=p ** 2)
        y = t.groupby(["district_key", "year_start"]).agg(
            D0=("crop_name", "size"), H=("plnp", "sum"), D0_01=("f01", "sum"),
            D0_05=("f05", "sum"), D0_1=("f1", "sum"), HHI=("p2", "sum")).reset_index()
        y["D1"] = np.exp(y["H"])
        y["D2"] = 1 / y["HHI"]
        y["E"] = np.where(y["D0"] > 1, y["H"] / np.log(y["D0"].clip(lower=2)), np.nan)
        m = y.groupby("district_key")[["D0", "D0_01", "D0_05", "D0_1", "D1", "D2", "E"]].mean()
        m.columns = [c + tag for c in m.columns]
        return m

    X = X.merge(idx(base, "").join(idx(base[~base["crop_name"].isin(DROP6)], "_d6")),
                left_on="district_key", right_index=True, how="left")
    assert (X["D0"] - X["D0_richness"]).abs().max() < 1e-6
    gca = base.groupby(["district_key", "year_start"])["area"].sum().groupby(level=0).mean().rename("gca")
    X = X.merge(gca, left_on="district_key", right_index=True, how="left")
    X["lgca"] = np.log(X["gca"])

    # ---------------------------------------------------------------- H0 the variable
    w("# H. Census 2011 weekly haat reach: referee log")
    w("")
    w("## H0. The variable")
    with pd.io.stata.StataReader(SHRUG + "/shrug-vd11-dta/pc11_vd_clean_shrid.dta") as r:
        lab = r.variable_labels()
    w("shrug-vd11 labels: pc11_vd_wkl_haat = '{}'; pc11_vd_mrkt = '{}'; pc11_vd_area = '{}'.".format(
        lab["pc11_vd_wkl_haat"], lab["pc11_vd_mrkt"], lab["pc11_vd_area"]))
    v11 = pd.read_stata(SHRUG + "/shrug-vd11-dta/pc11_vd_clean_shrid.dta",
                        columns=["shrid2", "pc11_vd_wkl_haat", "pc11_vd_mrkt", "pc11_vd_area"])
    vc = v11["pc11_vd_wkl_haat"].value_counts(dropna=False).sort_index()
    w("Values of pc11_vd_wkl_haat over {:,} shrids: {}.".format(
        len(v11), ", ".join("{}: {:,}".format(k, int(n)) for k, n in vc.items())))
    w("Distinct values of pc11_vd_mrkt: {}.".format(sorted(v11["pc11_vd_mrkt"].dropna().unique().tolist())))
    rk = pd.read_stata(SHRUG + "/shrug-pc-keys-dta/pc11r_shrid_key.dta")
    vcol = [c for c in rk.columns if c != "shrid2"]
    nv = rk.groupby("shrid2").size()
    w("Rural key columns: {}. Census villages per rural shrid: 1 in {:.1%} of shrids; more than one in "
      "{:,} shrids ({:.1%}).".format(vcol, (nv == 1).mean(), int((nv > 1).sum()), (nv > 1).mean()))
    mult = v11[v11.shrid2.isin(nv[nv > 1].index)]
    w("Haat share among multi-village shrids {:.1%} against {:.1%} overall (values stay 0/1, so a shrid "
      "counts as a haat shrid if any of its villages has one).".format(
          mult["pc11_vd_wkl_haat"].mean(), v11["pc11_vd_wkl_haat"].mean()))

    key = pd.read_stata(SHRUG + "/shrug-pc-keys-dta/shrid_pc11dist_key.dta").dropna(subset=DKEY)
    key = key.drop_duplicates("shrid2")
    for c in DKEY:
        key[c] = pd.to_numeric(key[c].astype(str), errors="coerce")
    pop = pd.read_stata(SHRUG + "/shrug-pca11-dta/pc11_pca_clean_shrid.dta", columns=["shrid2", "pc11_pca_tot_p"])
    V = key.merge(pop, on="shrid2", how="left").merge(v11, on="shrid2", how="left")
    V["rural"] = V.shrid2.isin(set(rk.shrid2))
    V["pw"] = V["pc11_pca_tot_p"].where(V["pc11_pca_tot_p"] > 0, 0).fillna(0)
    V["hk"] = (V.pc11_vd_wkl_haat > 0).astype(float).where(V.pc11_vd_wkl_haat.notna())
    V["mk"] = (V.pc11_vd_mrkt > 0).astype(float).where(V.pc11_vd_mrkt.notna())
    V["h_only"] = ((V.hk == 1) & (V.mk == 0)).astype(float).where(V[["hk", "mk"]].notna().all(axis=1))
    V["h_both"] = ((V.hk == 1) & (V.mk == 1)).astype(float).where(V[["hk", "mk"]].notna().all(axis=1))
    V["m_only"] = ((V.hk == 0) & (V.mk == 1)).astype(float).where(V[["hk", "mk"]].notna().all(axis=1))
    V["lpop"] = np.log(V["pw"].where(V["pw"] > 0))
    V["big"] = (V["pw"] >= 5000).astype(float)
    gk = [V[k] for k in DKEY]

    def pshare(col):
        ok = V[col].notna()
        pw = V["pw"].where(ok, 0)
        return (V[col].fillna(0) * pw).groupby(gk).sum() / pw.groupby(gk).sum()

    D = pd.DataFrame({"wk11_re": pshare("hk"), "dl11_re": pshare("mk"), "h_only": pshare("h_only"),
                      "h_both": pshare("h_both"), "m_only": pshare("m_only")})
    okh = V.hk.notna()
    D["wk11_vs"] = V.loc[okh, "hk"].groupby([V.loc[okh, k] for k in DKEY]).mean()
    rur = V[V.rural]
    D["vd_cov"] = (rur.pw.where(rur.hk.notna(), 0).groupby([rur[k] for k in DKEY]).sum() /
                   rur.pw.groupby([rur[k] for k in DKEY]).sum())
    va = V[okh & (V.pc11_vd_area > 0)]
    D["log_dens"] = np.log(va.pw.groupby([va[k] for k in DKEY]).sum() /
                           (va.pc11_vd_area.groupby([va[k] for k in DKEY]).sum() / 100))
    vv = V[okh & (V.pw > 0)]
    D["vsize"] = (vv.lpop * vv.pw).groupby([vv[k] for k in DKEY]).sum() / vv.pw.groupby([vv[k] for k in DKEY]).sum()
    D["big5k"] = (vv.big * vv.pw).groupby([vv[k] for k in DKEY]).sum() / vv.pw.groupby([vv[k] for k in DKEY]).sum()
    D = D.reset_index()
    P11 = pd.read_csv(REPO + "/outputs/revision_nov8/markets_v2/census_market_district_panel.csv")
    for c in DKEY:
        P11[c] = pd.to_numeric(P11[c], errors="coerce")
    X = X.merge(P11[DKEY + ["wk11_all", "dl11_all"]], on=DKEY, how="left").merge(D, on=DKEY, how="left")
    w("Rebuilt census haat reach against script 136 wk11_all: max abs diff {:.2e}; mandi/regular {:.2e}; "
      "missing over the 606: {}.".format((X.wk11_all - X.wk11_re).abs().max(), (X.dl11_all - X.dl11_re).abs().max(),
                                       int(X.wk11_all.isna().sum())))
    w("Means over the 606: census haat {:.3f} (SD {:.3f}); haat-only {:.3f}; haat and mandi/regular {:.3f}; "
      "mandi/regular only {:.3f}; village share {:.3f}; MA haat {:.3f}.".format(
          X.wk11_all.mean(), X.wk11_all.std(), X.h_only.mean(), X.h_both.mean(), X.m_only.mean(),
          X.wk11_vs.mean(), X.F_weekly_haat.mean()))
    cr = lambda u, v: X[[u, v]].corr().iloc[0, 1]  # noqa: E731
    w("Correlations: census haat with MA haat {:.2f}; census haat-only with MA haat {:.2f}; census haat-and-"
      "mandi with MA haat {:.2f}; census haat with log density {:.2f}, village size {:.2f}, irrigation {:.2f}, "
      "ST share {:.2f}, census mandi/regular {:.2f}.".format(
          cr("wk11_all", "F_weekly_haat"), cr("h_only", "F_weekly_haat"), cr("h_both", "F_weekly_haat"),
          cr("wk11_all", "log_dens"), cr("wk11_all", "vsize"), cr("wk11_all", "irr_share"),
          cr("wk11_all", "pca_st_share"), cr("wk11_all", "dl11_all")))
    w("2011 village-directory coverage of rural population: median {:.3f}; districts below 0.95: {}; "
      "below 0.90: {}.".format(X.vd_cov.median(), int((X.vd_cov < 0.95).sum()), int((X.vd_cov < 0.90).sum())))

    def run(df, y, term, extra=(), wild=True):
        rhs = " + ".join([term] + list(extra))
        f = "{} ~ {} + {}".format(y, rhs, CTRL)
        keep = [y, term] + list(extra) + BASEKEEP
        m, s = fit_par(df, f, keep)
        out = dict(b=m.params[term], se=m.bse[term], p=m.pvalues[term], n=int(m.nobs))
        if wild:
            r = wcr(df, f, term, keep)
            out.update(p_state=r["p_st"], p_wild=r["p_wcr"], G=r["G"])
        return out

    ROWS = []

    def rec(block, spec, y, r):
        r = dict(block=block, spec=spec, outcome=y, **r)
        ROWS.append(r)
        w("| {} | {} | {} | {} | {:+.3f} | {:.3f} | {:.4f} | {} | {} |".format(
            block, spec, y, r["n"], r["b"], r["se"], r["p"],
            "{:.4f}".format(r["p_state"]) if "p_state" in r else "",
            "{:.4f}".format(r["p_wild"]) if "p_wild" in r else ""))

    HDR = ("| block | spec | outcome | n | b | SE parent | p parent | p state t(G-1) | p wild |\n"
           "|---|---|---|---|---|---|---|---|---|")
    OUTS = ["D0", "D0_01", "D0_05", "D0_1", "D1", "D2", "E"]
    w("")
    w("## H1. Reproduce, all outcomes (census haat reach, all site controls, state FE)")
    w(HDR)
    for y in OUTS:
        rec("H1", "base", y, run(X, y, "wk11_all"))
    w("Check: D1 is the site's D1_exp_shannon (max diff {:.1e}).".format((X.D1 - X.D1_exp_shannon).abs().max()))

    # ---------------------------------------------------------------- H8 BH
    h1 = pd.DataFrame([r for r in ROWS if r["block"] == "H1"])
    h1["q_wild"] = multipletests(h1.p_wild, method="fdr_bh")[1]
    h1["q_parent"] = multipletests(h1.p, method="fdr_bh")[1]
    h1["holm_wild"] = multipletests(h1.p_wild, method="holm")[1]
    w("")
    w("## H8. Multiplicity over the seven outcomes")
    for _, r in h1.iterrows():
        w("- {}: wild p {:.4f}, BH q {:.4f}, Holm {:.4f}; parent p {:.4f}, BH q {:.4f}".format(
            r.outcome, r.p_wild, r.q_wild, r.holm_wild, r.p, r.q_parent))

    # ---------------------------------------------------------------- H2 LOSO
    w("")
    w("## H2. Leave one state out")
    LO = []
    for y in ["D0_05", "D1"]:
        for st in sorted(X.state_name.unique()):
            sub = X[X.state_name != st]
            f = "{} ~ wk11_all + {}".format(y, CTRL)
            keep = [y, "wk11_all"] + BASEKEEP
            m, s = fit_par(sub, f, keep)
            # state-clustered t(G-1) without bootstrap (fast)
            r = wcr(sub, f, "wk11_all", keep, B_=1)
            LO.append(dict(outcome=y, dropped=st, b=m.params["wk11_all"], p=m.pvalues["wk11_all"],
                           p_state=r["p_st"], n=int(m.nobs)))
    LO = pd.DataFrame(LO)
    LO.to_csv(OUT + "/H2_census_haat_loso.csv", index=False)
    for y in ["D0_05", "D1"]:
        t = LO[LO.outcome == y]
        lo, hi = t.loc[t.b.idxmin()], t.loc[t.b.idxmax()]
        w("{}: b from {:+.3f} (without {}, p {:.4f}, state p {:.4f}) to {:+.3f} (without {}); max parent p {:.4f} "
          "(without {}); max state p {:.4f} (without {}); states where state p > 0.10: {}.".format(
              y, lo.b, lo.dropped, lo.p, lo.p_state, hi.b, hi.dropped, t.p.max(), t.loc[t.p.idxmax(), "dropped"],
              t.p_state.max(), t.loc[t.p_state.idxmax(), "dropped"],
              ", ".join(t.loc[t.p_state > 0.10, "dropped"]) or "none"))

    # ---------------------------------------------------------------- H3-H6
    w("")
    w("## H3-H6. Controls, coverage, drop6, definitions")
    w(HDR)
    MA = ["F_mandi", "F_regular_market"]
    SPECS = [("H3", "+ census mandi/regular", ["dl11_all"]),
             ("H3", "+ log rural density", ["log_dens"]),
             ("H3", "+ village size (log pop, share 5k+)", ["vsize", "big5k"]),
             ("H3", "+ MA mandi + regular market", MA),
             ("H3", "+ MA haat", ["F_weekly_haat"]),
             ("H3", "+ log cropped area", ["lgca"]),
             ("H3", "+ all of the above", ["dl11_all", "log_dens", "vsize", "big5k", "lgca"] + MA + ["F_weekly_haat"]),
             ("H3", "MA haat, + log cropped area (term is census haat; MA haat as control)", ["F_weekly_haat", "lgca"])]
    for y in ["D0", "D0_05", "D1"]:
        for blk, nm, ex in SPECS:
            rec(blk, nm, y, run(X, y, "wk11_all", ex))
        for cut in [0.95, 0.90]:
            rec("H4", "VD coverage >= {:.2f}".format(cut), y, run(X[X.vd_cov >= cut], y, "wk11_all"))
        rec("H5", "drop6", y + "_d6", run(X, y + "_d6", "wk11_all"))
        rec("H6", "village share (unweighted)", y, run(X, y, "wk11_vs"))
        rec("H6", "haat-only villages (no mandi/regular)", y, run(X, y, "h_only"))
        rec("H6", "haat-and-mandi villages", y, run(X, y, "h_both"))
        rec("H6", "joint: haat-only term", y, run(X, y, "h_only", ["h_both", "m_only"]))
        rec("H6", "joint: haat-and-mandi term", y, run(X, y, "h_both", ["h_only", "m_only"]))
        rec("H6", "haat-only, all H3 controls", y, run(X, y, "h_only", ["h_both", "m_only", "log_dens", "vsize",
                                                                       "big5k", "lgca"] + MA + ["F_weekly_haat"]))
        rec("H3x", "MA haat term, + log cropped area", y, run(X, y, "F_weekly_haat", ["lgca"]))
        rec("H3x", "MA haat term, + log cropped area + census haat", y, run(X, y, "F_weekly_haat", ["lgca", "wk11_all"]))
    # difference haat-only minus haat-and-mandi in the joint model
    import statsmodels.formula.api as smf
    for y in ["D0_05", "D1"]:
        s = X.dropna(subset=[y, "h_only", "h_both", "m_only"] + BASEKEEP)
        m = smf.ols("{} ~ h_only + h_both + m_only + {}".format(y, CTRL), s).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(s.shrug_key)[0]})
        tt = m.t_test("h_only - h_both = 0")
        w("Joint model {}: haat-only minus haat-and-mandi {:+.3f} (SE {:.3f}, p {:.4f}).".format(
            y, float(np.squeeze(tt.effect)), float(np.squeeze(tt.sd)), float(np.squeeze(tt.pvalue))))

    # ---------------------------------------------------------------- H7 windows
    w("")
    w("## H7. Windows (div_windows.csv)")
    w(HDR)
    DW = pd.read_csv(REPO + "/outputs/revision_nov8/irrigation_v2/div_windows.csv")
    for win, var in [("full", "all"), ("e9802", "all"), ("cs0913", "all"), ("w1517", "all"), ("w1721", "flags_drop6")]:
        t = DW[(DW.window == win) & (DW.variant == var)][["district_key", "D0", "D0_05", "D1"]]
        t.columns = ["district_key", "wD0", "wD0_05", "wD1"]
        Xw = X.merge(t, on="district_key", how="left")
        for y in ["wD0", "wD0_05", "wD1"]:
            rec("H7", "{} {}".format(win, var), y, run(Xw, y, "wk11_all"))
    R = pd.DataFrame(ROWS)
    R.to_csv(OUT + "/H_census_haat.csv", index=False)
    h1.to_csv(OUT + "/H8_bh.csv", index=False)
    flush("H")

# =====================================================================================
# S1  irrigation knot
# =====================================================================================
if "S1" in PARTS:
    src = open(HERE + "/134_irrigation_threshold.py", encoding="utf-8").read()
    head = src[:src.index('w("# Irrigation threshold')]
    ns1 = {"__name__": "s134", "__file__": HERE + "/134_irrigation_threshold.py"}
    _argv = sys.argv
    sys.argv = [sys.argv[0]]
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(head, "134_head", "exec"), ns1)
    sys.argv = _argv
    LOG.clear()
    d0, with_win, wcr124, threshold_run = ns1["d0"], ns1["with_win"], ns1["wcr"], ns1["threshold_run"]
    FULL = with_win(d0, "full", "all")
    w("# S1. Irrigation knot (B18)")
    o, _, s = threshold_run(FULL, "D1", "irr_share", "repro", boot=False)
    w("Reproduce: n {}, knot {:.2f}, LR set {:.2f} to {:.2f}, slope above {:+.2f}.".format(
        o["n"], o["knot"], o["lr_ci95_lo"], o["lr_ci95_hi"], o["slope_above"]))

    def slopes(df, ctrl, tag):
        df = df.copy()
        df["lo"] = df["irr_share"].clip(upper=0.5)
        df["hi"] = (df["irr_share"] - 0.5).clip(lower=0)
        f = "D1 ~ lo + hi + {}C(state_name)".format("".join(c + " + " for c in ctrl))
        keep = ["D1", "lo", "hi", "state_name"] + ctrl
        r = wcr124(df, f, "hi", "state_name", keep, B_=9999)
        r2 = wcr124(df, f, "lo", "state_name", keep, B_=9999)
        w("{}: knot 0.50, slope below {:+.2f} (wild p {:.3f}), above {:+.2f} (state SE {:.2f}, t(G-1) p {:.3f}, "
          "wild p {:.3f}), n {}".format(tag, r2["b"], r2["p_wcr"], r["b"], r["se"], r["p_tG1"], r["p_wcr"], r["n"]))
        return dict(spec=tag, below=r2["b"], below_p=r2["p_wcr"], above=r["b"], above_se=r["se"],
                    above_p_tG1=r["p_tG1"], above_p_wild=r["p_wcr"], n=r["n"])

    S1 = [slopes(FULL, [], "reproduce, state FE only")]
    # break: markets-page controls
    with contextlib.redirect_stdout(io.StringIO()):
        from _mkt2_common import load_104  # noqa: E402
        ns = load_104()
    Xm = ns["prep"]("W")[["district_key", "mean_holding_ha", "pca_st_share", "log_viirs", "estab_per_1000pop",
                          "idx_connectivity_use"]].copy()
    Xm["lhold"] = np.log(Xm["mean_holding_ha"])
    F2 = FULL.drop(columns=[c for c in Xm.columns if c != "district_key" and c in FULL]).merge(
        Xm, on="district_key", how="inner")
    CT = ["lhold", "pca_st_share", "log_viirs", "estab_per_1000pop", "idx_connectivity_use"]
    w("Merged with the markets-page controls: {} of {} districts.".format(len(F2.dropna(subset=CT + ["D1"])), len(FULL)))
    o0, _, _ = threshold_run(F2.dropna(subset=CT), "D1", "irr_share", "same sample, no controls", boot=False)
    w("Same sample without controls: knot {:.2f}, slope above {:+.2f}.".format(o0["knot"], o0["slope_above"]))
    o2, _, _ = threshold_run(F2, "D1", "irr_share", "with controls", ctrl=tuple(CT), boot=True)
    w("With controls: knot {:.2f}; LR set {:.2f} to {:.2f}; Hansen set {:.2f} to {:.2f}; bootstrap {:.2f} to {:.2f}; "
      "sup-Wald p {:.3f}; slope below {:+.2f}, above {:+.2f}.".format(
          o2["knot"], o2["lr_ci95_lo"], o2["lr_ci95_hi"], o2["hansen735_lo"], o2["hansen735_hi"],
          o2["boot_lo"], o2["boot_hi"], o2["supwald_p"], o2["slope_below"], o2["slope_above"]))
    S1.append(slopes(F2.dropna(subset=CT), [], "same sample, no controls"))
    S1.append(slopes(F2, CT, "markets-page controls"))
    pd.DataFrame(S1).assign(knot_ctrl=o2["knot"], boot_lo=o2["boot_lo"], boot_hi=o2["boot_hi"],
                            supwald_p=o2["supwald_p"]).to_csv(OUT + "/S1_knot.csv", index=False)
    flush("S1")

# =====================================================================================
# S2  attenuation
# =====================================================================================
if "S2" in PARTS:
    w("# S2. Attenuation (B19)")
    L = pd.read_csv(REPO + "/outputs/revision_nov8/irrigation_v3/attenuation_longdiff_data.csv")

    def rel(L, a_col="dlus", c_col="dcen"):
        a = L[a_col] - L.groupby("state_name")[a_col].transform("mean")
        c = L[c_col] - L.groupby("state_name")[c_col].transform("mean")
        cov = (a * c).mean()
        return dict(n=len(L), corr=cov / np.sqrt((a ** 2).mean() * (c ** 2).mean()), lam_lus=cov / (a ** 2).mean(),
                    lam_cen=cov / (c ** 2).mean())

    rows = []
    r = rel(L)
    rows.append(dict(spec="reproduce", **r))
    w("Reproduce: n {n}, within-state correlation {corr:.3f}, LUS signal share {lam_lus:.3f}, census {lam_cen:.3f}.".format(**r))
    for q in [0.01, 0.025, 0.05]:
        Lw = L.copy()
        for c in ["dlus", "dcen"]:
            lo, hi = Lw[c].quantile([q, 1 - q])
            Lw[c] = Lw[c].clip(lo, hi)
        r = rel(Lw)
        rows.append(dict(spec="winsorised {:.1%}".format(q), **r))
        w("Winsorised at {:.1%}/{:.1%}: correlation {corr:.3f}, LUS share {lam_lus:.3f}, census {lam_cen:.3f}.".format(q, 1 - q, **r))
    for q in [0.95, 0.90]:
        cut = L.dlus.abs().quantile(q)
        r = rel(L[L.dlus.abs() <= cut])
        rows.append(dict(spec="drop |dLUS| above p{:.0f}".format(100 * q), **r))
        w("Drop LUS changes above the {:.0f}th percentile of |dLUS| ({:.3f}): n {n}, correlation {corr:.3f}, "
          "LUS share {lam_lus:.3f}.".format(100 * q, cut, **r))
    Lr = L.copy()
    Lr["rl"] = Lr.groupby("state_name").dlus.rank(pct=True)
    Lr["rc"] = Lr.groupby("state_name").dcen.rank(pct=True)
    r = rel(Lr, "rl", "rc")
    rows.append(dict(spec="within-state ranks", n=r["n"], corr=r["corr"]))
    w("Within-state rank correlation: {:.3f}.".format(r["corr"]))
    r = rel(L[L.brk_any == 0])
    rows.append(dict(spec="no LUS territory break", **r))
    w("No LUS territory break: n {n}, correlation {corr:.3f}, LUS share {lam_lus:.3f}.".format(**r))
    pd.DataFrame(rows).to_csv(OUT + "/S2_attenuation.csv", index=False)
    flush("S2")

# =====================================================================================
# S3  haat threshold collapse
# =====================================================================================
if "S3" in PARTS:
    from _mkt2_common import apy_base, load_104, wcr  # noqa: E402
    w("# S3. Haat threshold collapse at 0.5 percent (C10)")
    with contextlib.redirect_stdout(io.StringIO()):
        ns = load_104()
        a = apy_base()
    X = ns["prep"]("W")
    CTRL = ns["CTRL"]
    BASEKEEP = ["mean_holding_ha", "log_viirs", "shrug_key", "state_name"]
    base = a["base"]
    base = base[base["year_start"].between(1998, 2019) & base["district_key"].isin(X["district_key"])]
    p = base["area"] / base.groupby(["district_key", "year_start"])["area"].transform("sum")
    FL = {}
    for f in [0.0, 0.001, 0.002, 0.003, 0.004, 0.005]:
        FL["share>={:.1%}".format(f)] = (p >= f) if f > 0 else (p > -1)
    for ha in [100, 500, 1000, 2000]:
        FL["area>={}ha".format(ha)] = base["area"] >= ha
    gca = base.groupby(["district_key", "year_start"])["area"].sum().groupby(level=0).mean().rename("gca")
    X = X.merge(gca, left_on="district_key", right_index=True, how="left")
    X["lgca"] = np.log(X["gca"])
    w("Mean yearly cropped area {:,.0f} ha (median {:,.0f}); correlation of MA haat reach with log cropped area "
      "{:.2f}, within state {:.2f}.".format(X.gca.mean(), X.gca.median(), X[["F_weekly_haat", "lgca"]].corr().iloc[0, 1],
                                           (X.F_weekly_haat - X.groupby("state_name").F_weekly_haat.transform("mean")).corr(
                                               X.lgca - X.groupby("state_name").lgca.transform("mean"))))
    rows = []
    for nm, msk in FL.items():
        cnt = msk.groupby([base.district_key, base.year_start]).sum().groupby(level=0).mean().rename("y")
        t = X.merge(cnt, left_on="district_key", right_index=True, how="left")
        for tag, ex in [("site controls", ""), ("+ log cropped area", " + lgca")]:
            f = "y ~ F_weekly_haat + " + CTRL + ex
            r = wcr(t, f, "F_weekly_haat", ["y", "F_weekly_haat", "lgca"] + BASEKEEP)
            rows.append(dict(floor=nm, spec=tag, mean=t.y.mean(), b=r["b"], se_state=r["se_st"], p_state=r["p_st"],
                             p_wild=r["p_wcr"]))
            w("{} [{}]: mean {:.2f}; MA haat b {:+.3f} (state SE {:.3f}; state p {:.4f}; wild p {:.4f}).".format(
                nm, tag, t.y.mean(), r["b"], r["se_st"], r["p_st"], r["p_wcr"]))
    pd.DataFrame(rows).to_csv(OUT + "/S3_haat_floors.csv", index=False)
    flush("S3")

# =====================================================================================
# S4  soybean within-minus-across edge
# =====================================================================================
if "S4" in PARTS:
    src = open(HERE + "/140_pe_within_cross_hazard.py", encoding="utf-8").read()
    head = src[:src.index("coef_rows, test_rows = [], []")]
    ns4 = {"__name__": "s140", "__file__": HERE + "/140_pe_within_cross_hazard.py"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(head, "140_head", "exec"), ns4)
    LOG.clear()
    build, fit, contrast, CTRL4, demean = ns4["build"], ns4["fit"], ns4["contrast"], ns4["CTRL"], ns4["demean"]
    w("# S4. Soybean entry: edge within minus edge across (E11)")
    XS = ["nbr_same", "nbr_other", "ring_same", "ring_other"]
    WT = {"nbr_same": 1, "ring_same": -1, "nbr_other": -1, "ring_other": 1}
    rows = []
    for ep in [("soybean", "entry"), ("chickpea", "surge")]:
        d, T, status = build(*ep)
        o = fit(d, XS, "st")
        b, se, pv = contrast(o, "unit", WT)
        bs, ses, pvs = contrast(o, "state", WT)
        w("{} {}: n {}, events {}; contrast {:+.4f} (unit SE {:.4f}, p {:.4f}; state SE {:.4f}, p {:.4f})".format(
            *ep, o["n"], o["n_events"], b, se, pv, ses, pvs))
        # restricted wild cluster bootstrap by state on the contrast (reparametrised)
        dd = d.dropna(subset=["entry"] + XS + CTRL4).copy()
        sy = pd.factorize(dd.st.astype(str).values + "_" + dd.year.astype(str).values)[0]
        keep = np.bincount(sy)[sy] > 1
        dd, sy = dd[keep], pd.factorize(sy[keep])[0]
        Z = demean(dd[["entry"] + XS + CTRL4].values.astype(float), [sy])
        y, x = Z[:, 0], Z[:, 1:]
        ns_, no, rs, ro = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        Xr = np.column_stack([ns_, rs + ns_, no + ns_, ro - ns_, x[:, 4:]])
        wb = wcr_arrays(Xr, y, 0, dd.st.values)
        w("  reparametrised check b {:+.4f}; state CR1 t(G-1) p {:.4f}; restricted wild bootstrap by state p {:.4f} "
          "(G {})".format(wb["b"], wb["p_st"], wb["p_wcr"], wb["G"]))
        wbu = wcr_arrays(Xr, y, 0, dd.unit_key.values)
        rows.append(dict(episode=ep[0], mode=ep[1], drop="none", b=b, se_unit=se, p_unit=pv, p_state=pvs,
                         p_wild_state=wb["p_wcr"], p_wild_unit=wbu["p_wcr"], n_events=o["n_events"]))
        w("  restricted wild bootstrap by unit p {:.4f}".format(wbu["p_wcr"]))
        if ep[0] == "soybean":
            ev_units = dd.loc[dd.entry == 1, "unit_key"].unique()
            w("  event units: " + ", ".join("{} ({}, {})".format(u, dd.loc[(dd.unit_key == u) & (dd.entry == 1), "st"].iloc[0],
                                                              int(dd.loc[(dd.unit_key == u) & (dd.entry == 1), "year"].iloc[0]))
                                             for u in ev_units))
            for u in ev_units:
                o2 = fit(d[d.unit_key != u], XS, "st")
                if o2 is None:
                    w("  without {}: fewer than 5 events, not estimable".format(u))
                    rows.append(dict(episode=ep[0], mode=ep[1], drop=u, n_events=4))
                    continue
                b2, se2, p2 = contrast(o2, "unit", WT)
                _, _, p2s = contrast(o2, "state", WT)
                rows.append(dict(episode=ep[0], mode=ep[1], drop=u, b=b2, se_unit=se2, p_unit=p2, p_state=p2s,
                                 n_events=o2["n_events"]))
                w("  without event unit {}: contrast {:+.4f} (SE {:.4f}, p {:.4f}; state p {:.4f}), events {}".format(
                    u, b2, se2, p2, p2s, o2["n_events"]))
            # coefficients behind the contrast
            r, _ = o["res"]["unit"]
            w("  terms: " + ", ".join("{} {:+.4f} ({:.4f})".format(k, r.params[i], r.bse[i]) for i, k in enumerate(XS)))
    pd.DataFrame(rows).to_csv(OUT + "/S4_soybean_edge.csv", index=False)
    flush("S4")

# =====================================================================================
# S5  common-crop border gap
# =====================================================================================
if "S5" in PARTS:
    src = open(HERE + "/139_pe_common_crops_border.py", encoding="utf-8").read()
    head = src[:src.index("Aadj = Pn[Pn.adjacent == 1].copy()")]
    ns5 = {"__name__": "s139", "__file__": HERE + "/139_pe_common_crops_border.py"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(head, "139_head", "exec"), ns5)
    LOG.clear()
    Pn, reg, wild_state, GEO, win, C8 = ns5["Pn"], ns5["reg"], ns5["wild_state"], ns5["GEO"], ns5["win"], ns5["C8"]
    w("# S5. Common-crop border gap (E13)")
    for lab in ["w9802", "w1519"]:
        Ai = win[lab].reindex(Pn.unit_i).values
        Bj = win[lab].reindex(Pn.unit_j).values
        keep = [i for i, c in enumerate(C8) if c != "cotton"]
        Ai, Bj = Ai[:, keep], Bj[:, keep]
        sa, sb = Ai.sum(1, keepdims=True), Bj.sum(1, keepdims=True)
        an = np.where(sa > 0, Ai / np.where(sa > 0, sa, 1), np.nan)
        bn = np.where(sb > 0, Bj / np.where(sb > 0, sb, 1), np.nan)
        Pn["sim7_" + lab] = (1 - 0.5 * np.abs(an - bn).sum(1)) * 100
    A = Pn[Pn.adjacent == 1].copy()
    rows = []
    for lab in ["w1519", "w9802"]:
        for meas in ["sim8", "sim7"]:
            yc = "{}_{}".format(meas, lab)
            b0 = A.dropna(subset=[yc, "sim9_" + lab, "sim8_" + lab] + GEO).copy()
            for spec in ["M1", "M2", "M3"]:
                ou, os_ = reg(b0, yc, spec, "unit"), reg(b0, yc, spec, "state")
                pw = wild_state(b0, yc, spec)
                rows.append(dict(window=lab, measure=meas, drop="none", spec=spec, n=ou["_n"], b=ou["cross"][0],
                                 se_unit=ou["cross"][1], p_unit=ou["cross"][2], p_state=os_["cross"][2], p_wild=pw))
                w("{} {} {}: n {}, b {:+.2f} (unit SE {:.2f}, p {:.4f}; state p {:.4f}; wild p {:.3f})".format(
                    lab, meas, spec, ou["_n"], ou["cross"][0], ou["cross"][1], ou["cross"][2], os_["cross"][2], pw))
    w("")
    w("Leave one state out (sim8, 2015-19; pairs touching the state dropped):")
    yc = "sim8_w1519"
    b0 = A.dropna(subset=[yc, "sim9_w1519"] + GEO).copy()
    states = sorted(set(b0.st_i) | set(b0.st_j))
    lo = []
    for s in states:
        sub = b0[(b0.st_i != s) & (b0.st_j != s)]
        if sub.cross.sum() < 10:
            continue
        for spec in ["M1", "M2", "M3"]:
            ou, os_ = reg(sub, yc, spec, "unit"), reg(sub, yc, spec, "state")
            lo.append(dict(dropped=s, spec=spec, b=ou["cross"][0], p_unit=ou["cross"][2], p_state=os_["cross"][2],
                           n=ou["_n"]))
    LO = pd.DataFrame(lo)
    for spec in ["M1", "M2", "M3"]:
        t = LO[LO.spec == spec]
        w("{}: b {:+.2f} (without {}) to {:+.2f} (without {}); max unit p {:.4f} (without {}); max state p {:.4f} "
          "(without {}).".format(spec, t.b.min(), t.loc[t.b.idxmin(), "dropped"], t.b.max(),
                                 t.loc[t.b.idxmax(), "dropped"], t.p_unit.max(), t.loc[t.p_unit.idxmax(), "dropped"],
                                 t.p_state.max(), t.loc[t.p_state.idxmax(), "dropped"]))
    pd.DataFrame(rows).to_csv(OUT + "/S5_border_gap.csv", index=False)
    LO.to_csv(OUT + "/S5_border_gap_loso.csv", index=False)
    flush("S5")
