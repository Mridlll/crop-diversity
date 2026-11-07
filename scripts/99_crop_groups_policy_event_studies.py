"""
99_crop_groups_policy_event_studies.py

Q2 policy event studies on the crop-group panel (2011-parent units, 1998-99 to 2019-20; 1997-98 is dropped through scripts/_years.py).

  A. NFSM-Pulses entry -> pulse share of GCA, log pulse area, rice and wheat share.
     Stacked event study: one stack per entry cohort, each with only clean controls
     (units not yet treated anywhere in the stack window). Within a stack there is a
     single treatment date, so the TWFE event study inside it has no forbidden
     comparisons (Cengiz et al. 2019; equivalent to a CS ATT(g,t) with not-yet-treated
     controls). Unit-by-stack FE and state-by-year-by-stack FE; SEs clustered on the
     2011-parent unit (= the shared-parent groups, 605 in the 606 sample).
       stack P07: NFSM-Pulses 2007-08 list vs same-state units entering 2010-11 or later,
                  window 1998-99 .. 2009-10.  Main estimate.
       stack P10: 2010-11 state-wide expansion vs units entering 2014-15/2015-16.
                  All entry is state-wide, so only year FE are possible (no state-by-year);
                  reported as weak / descriptive.
     Also NFSM-Rice and NFSM-Wheat 2007-08 lists (rice / wheat share), window to 2013-14,
     controls never listed for that component through 2013-14.
  B. Odisha Millets Mission: Callaway-Sant'Anna ATT(g,t) within Odisha, not-yet-treated
     controls, cohorts 2017-18 / 2018-19 / 2019-20, district + year FE logic (long
     differences against g-1). Inference by cluster bootstrap and by randomisation
     inference over cohort assignment (30 districts).
  C. 2018-19 MSP increase, shift-share.
       C1 district exposure  X_d = sum_c s_dc(2013-16) * dlnMSP_c(excess), event study of
          group shares on X_d x year, unit FE + state-by-year FE.
       C2 group-level design  y_dgt on s_dg(pre) * dlnMSP_g x year, with d x t, g x t,
          d x g FE and s_dg(pre) x year controls; randomisation inference permuting the
          group shocks.

USAGE:  python scripts/99_crop_groups_policy_event_studies.py
Inputs: outputs/crop_groups/crop_group_panel_annual.csv, crop_group_panel.csv,
        crop_group_child_map.csv, policy_nfsm_district_year.csv,
        policy_omm_district_year.csv, policy_msp_crop_year.csv, raw APY (for crop-level
        pre-period shares in C1).
Outputs: outputs/crop_groups/policy_es_*.csv, policy_es_*.png, policy_es_log.txt
No APCNF / PMDS / RySS data are used.
"""
import itertools
import os
import warnings

import numpy as np
import pandas as pd
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _years import keep_years, FIRST_YEAR

REPO = r"D:/crop-diversity"
OUT = REPO + "/outputs/crop_groups"
RAW = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
RNG = np.random.default_rng(20261107)
LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


# ====================================================================== estimator
def demean(df, cols, fes, tol=1e-10, maxit=2000):
    """Alternating projections: sweep out each FE in turn until convergence."""
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
    # drop FE singletons iteratively
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
                resid=u, X=X)


def lincom(r, w_):
    w_ = pd.Series(w_).reindex(r["b"].index).fillna(0)
    est = float(w_ @ r["b"])
    se = float(np.sqrt(w_ @ r["V"] @ w_))
    return est, se


def wald(r, names):
    names = [n for n in names if n in r["b"].index]
    if not names:
        return np.nan, np.nan, 0
    b = r["b"][names].to_numpy()
    V = r["V"].loc[names, names].to_numpy()
    F = float(b @ np.linalg.pinv(V) @ b) / len(names)
    p = 1 - stats.f.cdf(F, len(names), r["G"] - 1)
    return F, p, len(names)


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


# ====================================================================== data
w("=" * 78)
w("Q2 policy event studies on crop-group shares")
w("=" * 78)
A = pd.read_csv(OUT + "/crop_group_panel_annual.csv")
A["state"] = A["unit_key"].str.split("|").str[0]
A = keep_years(A[A["in_606"] == 1]).copy()  # analysis window, scripts/_years.py
A["yr"] = A["year_start"]
GROUPS = ["rice", "wheat", "maize", "millets", "pulses", "oilseeds", "cotton",
          "sugarcane", "other"]
w("606-sample annual unit-years: {:,} in {} units".format(len(A), A.unit_key.nunique()))
A_ok = A[A["flag_annual_unreliable"] == 0].copy()
w("after dropping flag_annual_unreliable: {:,}".format(len(A_ok)))

cmap = pd.read_csv(OUT + "/crop_group_child_map.csv")[["div_key", "unit_key"]]


def unit_treat(component):
    d = pd.read_csv(OUT + "/policy_nfsm_district_year.csv")
    d = d[d["component"] == component].merge(cmap, left_on="district_key", right_on="div_key")
    d["yr"] = d["year"].str[:4].astype(int)
    u = d.groupby(["unit_key", "yr"])["treated"].agg(
        lambda s: np.nan if s.isna().all() else s.max()).unstack()
    return u


# ====================================================================== stacked event study
def build_stack(panel, unit_first, cohort, ctrl_rule, lo, hi, name, state_fe=True,
                exclude_states=()):
    """treated = units first treated in `cohort`; controls = units with first
    treatment > hi (not yet treated in window) or never."""
    p = panel[(panel.yr >= lo) & (panel.yr <= hi)].copy()
    f = p["unit_key"].map(unit_first)
    p["treat"] = (f == cohort).astype(int)
    ctrl = ctrl_rule(f)
    p = p[(p["treat"] == 1) | ctrl]
    p = p[~p["state"].isin(exclude_states)]
    if state_fe:  # keep only states that have both treated and control units
        st = p.groupby("state")["treat"].agg(["min", "max"])
        p = p[p["state"].isin(st[(st["min"] == 0) & (st["max"] == 1)].index)]
    p["e"] = np.where(p["treat"] == 1, p["yr"] - cohort, -999)
    p["stack"] = name
    return p


def event_study(p, y, ref=-1, state_fe=True, extra_x=(), trend=False, label=""):
    p = p.copy()
    es = sorted(e for e in p.loc[p.treat == 1, "e"].unique() if e != ref)
    cols = []
    for e in es:
        c = "ev_{}".format(e).replace("-", "m")
        p[c] = (p["e"] == e).astype(float)
        cols.append(c)
    xs = cols + list(extra_x)
    if trend:  # treated-specific linear trend, identified off the pre-period shape
        p["tr_lin"] = p["treat"] * (p["yr"] - p["yr"].min())
        xs = xs + ["tr_lin"]
    p["u_s"] = p["unit_key"] + "#" + p["stack"]
    p["t_s"] = (p["state"] + "#" if state_fe else "") + p["yr"].astype(str) + "#" + p["stack"]
    r = feols(p, y, xs, ["u_s", "t_s"], "unit_key")
    # DiD: post vs pre, same FE
    p["post"] = ((p["treat"] == 1) & (p["e"] >= 0)).astype(float)
    r2 = feols(p, y, ["post"] + list(extra_x), ["u_s", "t_s"], "unit_key")
    pre = [c for c, e in zip(cols, es) if e < 0]
    post = [c for c, e in zip(cols, es) if e >= 0]
    Fp, pp, kp = wald(r, pre)
    avg_post = lincom(r, {c: 1 / len(post) for c in post})
    # short-window contrast: mean post minus mean of e = -4..-1 (e=-1 is the zero reference)
    near = [c for c, e in zip(cols, es) if -4 <= e < -1]
    wv = {c: 1 / len(post) for c in post}
    for c in near:
        wv[c] = -1 / (len(near) + 1)
    pv4 = lincom(r, wv)
    ev = pd.DataFrame({"e": es, "coef": [r["b"].get(c, np.nan) for c in cols],
                       "se": [np.sqrt(r["V"].loc[c, c]) if c in r["b"].index else np.nan
                              for c in cols]})
    ev = pd.concat([ev, pd.DataFrame({"e": [ref], "coef": [0.0], "se": [0.0]})]).sort_values("e")
    d = r2["data"]
    out = dict(label=label, y=y, did=r2["b"]["post"], did_se=np.sqrt(r2["V"].loc["post", "post"]),
               avgpost=avg_post[0], avgpost_se=avg_post[1],
               post_vs_pre4=pv4[0], post_vs_pre4_se=pv4[1], pre_F=Fp, pre_p=pp, pre_k=kp,
               n=r2["n"], G=r2["G"], n_treat_units=d.loc[d.treat == 1, "unit_key"].nunique(),
               n_ctrl_units=d.loc[d.treat == 0, "unit_key"].nunique(),
               ymean_treat_pre=d.loc[(d.treat == 1) & (d.e < 0), y].mean())
    if trend:
        out["trend"] = r["b"].get("tr_lin", np.nan)
    return out, ev


def plot_es(evs, title, fname, ylab):
    n = len(evs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), squeeze=False)
    for ax, (lab, ev) in zip(axes[0], evs.items()):
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(-0.5, color="0.6", lw=0.8, ls=":")
        ax.errorbar(ev["e"], ev["coef"], yerr=1.96 * ev["se"], fmt="o", color="0.1",
                    ms=3.5, capsize=2, lw=1)
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("years since entry", fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(labelsize=7)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT + "/" + fname, dpi=160)
    plt.close(fig)


rows = []
w("")
w("-" * 78)
w("A. NFSM entry, stacked event studies")
w("-" * 78)
uP = unit_treat("pulses")
firstP = uP.apply(lambda r: r[r == 1].index.min(), axis=1)
w("NFSM-Pulses unit first-entry year (606 sample units):")
w(firstP.loc[firstP.index.isin(A.unit_key)].value_counts(dropna=False).sort_index().to_string())

# units whose 2010-13 pulses status is unknown (Assam, Jharkhand unnamed districts)
unk = uP.index[uP.loc[:, 2010:2013].isna().any(axis=1)]
w("units with unknown 2010-13 pulses status (Assam, Jharkhand): {}".format(len(unk)))

ev_store = {}
OUTC_P = [("share_gca_pulses", "pulse share of GCA"),
          ("ln_area_ha_pulses", "log pulse area"),
          ("share_gca_rice", "rice share of GCA"),
          ("share_gca_wheat", "wheat share of GCA")]

# P07 main: 2007 list vs units first entering >= 2010, within state
P07 = build_stack(A_ok, firstP, 2007, lambda f: f >= 2010, FIRST_YEAR, 2009, "P07",
                  exclude_states=("ASSAM", "JHARKHAND"))
w("P07 stack: {} treated units, {} control units, states {}".format(
    P07[P07.treat == 1].unit_key.nunique(), P07[P07.treat == 0].unit_key.nunique(),
    P07.state.nunique()))
for y, lab in OUTC_P:
    o, ev = event_study(P07, y, label="NFSM-Pulses 2007 [P07]")
    rows.append(o)
    ev_store[("P07", y)] = ev
plot_es({lab: ev_store[("P07", y)] for y, lab in OUTC_P},
        "NFSM-Pulses entry in 2007-08, same-state units entering 2010-11 as controls",
        "policy_es_nfsm_pulses_2007.png", "coef (ref e = -1)")

# robustness for P07 pulse share
for y in ["share_gca_pulses", "ln_area_ha_pulses"]:
    bal = P07.groupby("unit_key").yr.transform("size") == P07.yr.nunique()
    o, _ = event_study(P07[bal], y, label="P07 balanced units")
    rows.append(o)
    o, _ = event_study(P07, y, trend=True, label="P07 + treated linear trend")
    rows.append(o)
    o, _ = event_study(P07, y, state_fe=False, label="P07 year FE only")
    rows.append(o)
    pw = P07.copy()
    pw["e"] = np.where(pw.treat == 1, pw.e, -999)
    o, _ = event_study(A[(A.yr >= FIRST_YEAR) & (A.yr <= 2009)].assign(
        treat=lambda d: (d.unit_key.map(firstP) == 2007).astype(int))
        .pipe(lambda d: d[(d.treat == 1) | (d.unit_key.map(firstP) >= 2010)])
        .pipe(lambda d: d[~d.state.isin(["ASSAM", "JHARKHAND"])])
        .assign(e=lambda d: np.where(d.treat == 1, d.yr - 2007, -999), stack="P07"),
        y, label="P07 incl. flagged unit-years")
    rows.append(o)
# overlap with NFSM-Rice 2007 list: control for it
uR = unit_treat("rice")
uW = unit_treat("wheat")
P07c = P07.copy()
P07c["rice07"] = P07c.unit_key.map(uR[2007]).fillna(0) * (P07c.yr >= 2007)
P07c["wheat07"] = P07c.unit_key.map(uW[2007]).fillna(0) * (P07c.yr >= 2007)
for y in ["share_gca_pulses", "ln_area_ha_pulses"]:
    o, _ = event_study(P07c, y, extra_x=["rice07", "wheat07"],
                       label="P07 + NFSM rice/wheat 2007 controls")
    rows.append(o)

# P10: 2010 state-wide expansion vs 2014/2015 entrants, year FE only
P10 = build_stack(A_ok, firstP, 2010, lambda f: f >= 2014, 2003, 2013, "P10",
                  state_fe=False, exclude_states=("ASSAM", "JHARKHAND"))
w("P10 stack: {} treated, {} control units (controls are hill / NE / Kerala states)".format(
    P10[P10.treat == 1].unit_key.nunique(), P10[P10.treat == 0].unit_key.nunique()))
for y, lab in OUTC_P[:2]:
    o, ev = event_study(P10, y, state_fe=False, label="NFSM-Pulses 2010 [P10, year FE]")
    rows.append(o)
    ev_store[("P10", y)] = ev
plot_es({lab: ev_store[("P10", y)] for y, lab in OUTC_P[:2]},
        "NFSM-Pulses state-wide expansion 2010-11 vs hill, NE and Kerala units (year FE only)",
        "policy_es_nfsm_pulses_2010.png", "coef (ref e = -1)")

# Rice and wheat 2007 lists
firstR = uR.apply(lambda r: r[r == 1].index.min(), axis=1)
firstW = uW.apply(lambda r: r[r == 1].index.min(), axis=1)
for comp, first, ys in [("rice", firstR, [("share_gca_rice", "rice share"),
                                          ("ln_area_ha_rice", "log rice area"),
                                          ("share_gca_pulses", "pulse share")]),
                        ("wheat", firstW, [("share_gca_wheat", "wheat share"),
                                           ("ln_area_ha_wheat", "log wheat area"),
                                           ("share_gca_pulses", "pulse share")])]:
    S = build_stack(A_ok, first, 2007, lambda f: f.isna() | (f >= 2014), FIRST_YEAR, 2013,
                    comp[0].upper() + "07")
    w("{}07 stack: {} treated, {} controls, {} states".format(
        comp, S[S.treat == 1].unit_key.nunique(), S[S.treat == 0].unit_key.nunique(),
        S.state.nunique()))
    evs = {}
    for y, lab in ys:
        o, ev = event_study(S, y, label="NFSM-{} 2007".format(comp.capitalize()))
        rows.append(o)
        evs[lab] = ev
        ev_store[(comp + "07", y)] = ev
    plot_es(evs, "NFSM-{} entry in 2007-08, never-listed same-state units as controls".format(
        comp.capitalize()), "policy_es_nfsm_{}_2007.png".format(comp), "coef (ref e = -1)")

# ====================================================================== B. OMM
w("")
w("-" * 78)
w("B. Odisha Millets Mission, Callaway-Sant'Anna within Odisha")
w("-" * 78)
om = pd.read_csv(OUT + "/policy_omm_district_year.csv")
om = om.merge(cmap, left_on="district_key", right_on="div_key")
gmap = om.groupby("unit_key")["omm_entry_year"].first().str[:4].astype(float)
gmap = gmap.fillna(np.inf)  # never entered
gmap[gmap > 2019] = np.inf  # not treated within panel window  # not treated within panel window
O = A_ok[A_ok.state == "ODISHA"].copy()
O = O[O.yr >= 2005]
O["g"] = O.unit_key.map(gmap)
O["ln1p_millets"] = np.log1p(O["area_ha_millets"])
w("Odisha units: {}; cohorts: {}".format(O.unit_key.nunique(),
                                          gmap.value_counts().sort_index().to_dict()))
# kharif millet share (season split reliable in Odisha from 2005 except 2012 which is dropped)
P = pd.read_csv(OUT + "/crop_group_panel.csv",
                usecols=["unit_key", "year_start", "season", "share_season_millets",
                         "flag_season_unreliable"])
K = P[(P.season == "kharif") & (P.unit_key.str.startswith("ODISHA|"))]
K = K[K.flag_season_unreliable == 0].set_index(["unit_key", "year_start"])["share_season_millets"]
O["kharif_share_millets"] = [K.get((u, t), np.nan) for u, t in zip(O.unit_key, O.yr)]


def cs_att(df, y, gcol="g"):
    """ATT(g,t) with not-yet-treated controls and base period g-1 (CS 2021, unconditional)."""
    wide = df.pivot(index="unit_key", columns="yr", values=y)
    g = df.groupby("unit_key")[gcol].first().reindex(wide.index)
    res = []
    for gg in sorted(set(g[np.isfinite(g)])):
        base = gg - 1
        for t in wide.columns:
            if t == base:
                continue
            tr = g == gg
            ct = (g > max(t, base)) & (g != gg)
            d = wide[t] - wide[base]
            a, b = d[tr].dropna(), d[ct].dropna()
            if len(a) == 0 or len(b) < 2:
                continue
            res.append(dict(g=gg, t=t, e=t - gg, att=a.mean() - b.mean(), n_t=len(a),
                            n_c=len(b)))
    return pd.DataFrame(res)


def cs_aggregate(att):
    post = att[att.e >= 0]
    overall = np.average(post.att, weights=post.n_t) if len(post) else np.nan
    dyn = att.groupby("e").apply(lambda d: np.average(d.att, weights=d.n_t))
    return overall, dyn


def cs_with_inference(df, y, B=499):
    att = cs_att(df, y)
    overall, dyn = cs_aggregate(att)
    units = df.unit_key.unique()
    bo, bd = [], []
    for _ in range(B):  # cluster (unit) bootstrap
        draw = RNG.choice(units, len(units), replace=True)
        parts = []
        for i, u in enumerate(draw):
            x = df[df.unit_key == u].copy()
            x["unit_key"] = u + "#" + str(i)
            parts.append(x)
        bdf = pd.concat(parts)
        a = cs_att(bdf, y)
        if len(a) == 0 or (a.e >= 0).sum() == 0:
            continue
        o, dd = cs_aggregate(a)
        bo.append(o)
        bd.append(dd)
    bd = pd.DataFrame(bd)
    se_dyn = bd.std()
    # randomisation inference: reshuffle cohort labels over units
    gs = df.groupby("unit_key")["g"].first()
    ri = []
    for _ in range(B):
        perm = pd.Series(RNG.permutation(gs.to_numpy()), gs.index)
        x = df.copy()
        x["g"] = x.unit_key.map(perm)
        o, _ = cs_aggregate(cs_att(x, y))
        ri.append(o)
    ri = np.array(ri)
    p_ri = (np.sum(np.abs(ri) >= abs(overall)) + 1) / (len(ri) + 1)
    pre = dyn[dyn.index < -1]
    pre_mean = pre.mean() if len(pre) else np.nan
    pre_se = (bd[[c for c in bd.columns if c < -1]].mean(axis=1).std()
              if len(pre) else np.nan)
    return dict(att=overall, se=np.std(bo), ci_lo=np.percentile(bo, 2.5),
                ci_hi=np.percentile(bo, 97.5), p_ri=p_ri, pre_mean=pre_mean, pre_se=pre_se,
                n_units=df.unit_key.nunique(), n=len(df.dropna(subset=[y])),
                n_treat=int(np.isfinite(gs).sum())), \
        pd.DataFrame({"e": dyn.index, "coef": dyn.values,
                      "se": se_dyn.reindex(dyn.index).values}), att


omm_rows = []
omm_ev = {}
for y, lab in [("share_gca_millets", "millet share of GCA"),
               ("ln1p_millets", "log(1 + millet area)"),
               ("kharif_share_millets", "kharif millet share")]:
    dfy = O.dropna(subset=[y])
    r, ev, att = cs_with_inference(dfy, y)
    r.update(y=y)
    omm_rows.append(r)
    omm_ev[lab] = ev
    att.to_csv(OUT + "/policy_es_omm_attgt_{}.csv".format(y), index=False)
    w("OMM {:<22} ATT {:+.4f} (boot se {:.4f}, 95% CI {:+.4f} to {:+.4f}, RI p {:.3f}); "
      "mean pre-period placebo {:+.4f}; n {} unit-years, {} units, {} treated".format(
          y, r["att"], r["se"], r["ci_lo"], r["ci_hi"], r["p_ri"], r["pre_mean"], r["n"],
          r["n_units"], r["n_treat"]))
# also TWFE-in-Odisha event study via the stacked machinery for comparison (2017 cohort only)
O17 = O[(O.g == 2017) | (O.g > 2019)].copy()
O17["treat"] = (O17.g == 2017).astype(int)
O17["e"] = np.where(O17.treat == 1, O17.yr - 2017, -999)
O17["stack"] = "O17"
for y in ["share_gca_millets", "ln1p_millets"]:
    o, ev = event_study(O17, y, state_fe=False, label="OMM 2017 cohort vs not-in-window")
    rows.append(o)
plot_es(omm_ev, "Odisha Millets Mission entry, Callaway-Sant'Anna, not-yet-treated controls",
        "policy_es_omm.png", "ATT (ref e = -1)")
pd.DataFrame(omm_rows).to_csv(OUT + "/policy_es_omm_summary.csv", index=False)
# raw means for the record
w("Odisha mean millet share by cohort and year:")
w(O.assign(c=O.g.replace(np.inf, 0)).pivot_table(index="yr", columns="c",
                                                  values="share_gca_millets").round(4).to_string())

# ====================================================================== C. MSP shift-share
w("")
w("-" * 78)
w("C. 2018-19 MSP increase, shift-share")
w("-" * 78)
msp = pd.read_csv(OUT + "/policy_msp_crop_year.csv")
msp = msp[msp.variety.isna() | msp.variety.isin(["common", "medium_staple", "yellow"])]
msp = msp[msp.crop != "toria"]
msp["yr"] = msp.crop_year.str[:4].astype(int)
lm = msp.pivot_table(index="crop", columns="yr", values="msp_rs_per_qtl").apply(np.log)
jump = lm[2018] - lm[2017]
trend = (lm[2017] - lm[2013]) / 4
dmsp = (jump - trend).rename("dlnmsp_excess")
MSP2APY = {"paddy": "Rice", "wheat": "Wheat", "maize": "Maize", "jowar": "Jowar",
           "bajra": "Bajra", "ragi": "Ragi", "arhar": "Arhar/Tur", "moong": "Moong(Green Gram)",
           "urad": "Urad", "gram": "Gram", "masur": "Masoor", "groundnut": "Groundnut",
           "rapeseed_mustard": "Rapeseed &Mustard", "soybean": "Soyabean",
           "sunflower": "Sunflower", "sesamum": "Sesamum", "nigerseed": "Niger Seed",
           "safflower": "Safflower", "cotton": "Cotton(Lint)", "sugarcane": "Sugarcane",
           "barley": "Barley", "jute": "Jute"}
mt = pd.DataFrame({"jump_2018": jump, "trend_13_17": trend, "excess": dmsp})
mt["apy_crop"] = mt.index.map(MSP2APY)
w("MSP log change 2017-18 -> 2018-19, trend 2013-17, excess:")
w(mt.round(4).to_string())

# crop-level pre-period (2013-2016) shares from raw APY, script-76/98 cleaning
df = pd.read_csv(RAW, usecols=["year", "season", "state_name", "district_name", "crop_name",
                               "area"])
for c in ["year", "season", "state_name", "district_name", "crop_name"]:
    df[c] = df[c].astype(str).str.strip()
df = df.dropna(subset=["area"])
df = df[df.area > 0]
df["year_start"] = df.year.str.split("-").str[0].astype(int)
df = df[(df.year_start >= 2013) & (df.year_start <= 2016)]
REMAP = {("Uttar Pradesh", d): "Uttarakhand" for d in
         ["Almora", "Bageshwar", "Chamoli", "Champawat", "Dehradun", "Haridwar",
          "Nainital", "Pauri Garhwal", "Pithoragarh"]}
REMAP.update({("Madhya Pradesh", d): "Chhattisgarh" for d in
              ["Bastar", "Bilaspur", "Dhamtari", "Durg", "Jashpur", "Kanker",
               "Korba", "Mahasamund", "Raipur", "Rajnandgaon", "Surguja"]})
REMAP.update({("Bihar", d): "Jharkhand" for d in ["Deoghar", "Dhanbad", "Garhwa"]})
for (s_, d_), new in REMAP.items():
    m = (df.state_name == s_) & (df.district_name == d_)
    df.loc[m, "state_name"] = new
df["div_key"] = df.state_name.str.upper() + "|" + df.district_name.str.upper()
df = df.sort_values("area", ascending=False).drop_duplicates(
    ["div_key", "year", "season", "crop_name"])
df = df.merge(cmap, on="div_key", how="inner")
# drop flagged unit-years (annual) before forming pre shares
flag = A.set_index(["unit_key", "yr"])["flag_annual_unreliable"]
df = df[[flag.get((u, t), 1) == 0 for u, t in zip(df.unit_key, df.year_start)]]
cy = df.groupby(["unit_key", "year_start", "crop_name"]).area.sum()
tot = cy.groupby(level=[0, 1]).transform("sum")
sh = (cy / tot).groupby(level=[0, 2]).mean().unstack(fill_value=0)  # mean over pre years
apy_d = mt.set_index("apy_crop")["excess"]
X_d = (sh.reindex(columns=apy_d.index, fill_value=0) * apy_d).sum(axis=1)
w("district exposure X_d: mean {:.4f}, sd {:.4f}, p10 {:.4f}, p90 {:.4f}, n {}".format(
    X_d.mean(), X_d.std(), X_d.quantile(.1), X_d.quantile(.9), len(X_d)))
X_d.rename("msp_exposure").to_csv(OUT + "/policy_es_msp_exposure.csv")
# top-exposure units
w("10 most exposed units: " + ", ".join(X_d.sort_values().index[-10:][::-1]))

# group-level shocks: national pre-area-weighted mean of crop excess changes (0 for non-MSP)
crop_group = pd.read_csv(OUT + "/crop_group_crop_mapping.csv").set_index("crop_name")["group"]
nat = df.groupby("crop_name").area.sum()
gshock = {}
for g in GROUPS:
    crops = crop_group[crop_group == g].index
    a = nat.reindex(crops).fillna(0)
    e = apy_d.reindex(crops).fillna(0)
    gshock[g] = float((a * e).sum() / a.sum())
gshock = pd.Series(gshock)
w("group shocks (area-weighted excess dlnMSP): " + ", ".join(
    "{} {:+.3f}".format(k, v) for k, v in gshock.items()))

# C1: district exposure event study, years 2008-2019, ref 2017
C = A_ok[(A_ok.yr >= 2008)].copy()
C["X"] = C.unit_key.map((X_d - X_d.mean()) / X_d.std())
C = C.dropna(subset=["X"])
yrs = sorted(C.yr.unique())
xc = []
for t in yrs:
    if t == 2017:
        continue
    C["x_{}".format(t)] = C.X * (C.yr == t)
    xc.append("x_{}".format(t))
c1 = []
c1_ev = {}
for g in GROUPS:
    y = "share_gca_" + g
    C["st_yr"] = C.state + "#" + C.yr.astype(str)
    r = feols(C, y, xc, ["unit_key", "st_yr"], "unit_key")
    post = ["x_2018", "x_2019"]
    pre = [c for c in xc if int(c[2:]) < 2017]
    est, se = lincom(r, {c: 0.5 for c in post})
    wv = {c: 0.5 for c in post}
    wv.update({"x_2014": -0.25, "x_2015": -0.25, "x_2016": -0.25})
    e4, s4 = lincom(r, wv)
    F, p, k = wald(r, pre)
    c1.append(dict(group=g, avg_post_per_sd=est, se=se, post_vs_1416=e4, post_vs_1416_se=s4, p=2 * (1 - stats.norm.cdf(abs(est / se))),
                   pre_F=F, pre_p=p, n=r["n"], G=r["G"],
                   mean_share=C[y].mean()))
    c1_ev[g] = pd.DataFrame({"e": [int(c[2:]) for c in xc] + [2017],
                             "coef": [r["b"].get(c, np.nan) for c in xc] + [0],
                             "se": [np.sqrt(r["V"].loc[c, c]) for c in xc] + [0]})
c1 = pd.DataFrame(c1)
c1.to_csv(OUT + "/policy_es_msp_c1_exposure.csv", index=False)
w("C1 (share on standardised exposure x post; unit + state-by-year FE):")
w(c1.round(5).to_string(index=False))
fig, axes = plt.subplots(3, 3, figsize=(10, 7.5), sharex=True)
for ax, g in zip(axes.flat, GROUPS):
    ev = c1_ev[g].sort_values("e")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(2017.5, color="0.6", lw=0.8, ls=":")
    ax.errorbar(ev.e, ev.coef, yerr=1.96 * ev.se, fmt="o", color="0.1", ms=3, capsize=2, lw=1)
    ax.set_title(g, fontsize=9)
    ax.tick_params(labelsize=7)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
fig.suptitle("Share of GCA on 1 SD of 2018-19 MSP exposure x year (ref 2017-18)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT + "/policy_es_msp_exposure.png", dpi=160)
plt.close(fig)

# C2: group-level design
L = []
pre_sh = A_ok[(A_ok.yr >= 2013) & (A_ok.yr <= 2016)].groupby("unit_key")[
    ["share_gca_" + g for g in GROUPS]].mean()
for g in GROUPS:
    x = C[["unit_key", "state", "yr", "share_gca_" + g]].rename(columns={"share_gca_" + g: "y"})
    x["group"] = g
    x["s_pre"] = x.unit_key.map(pre_sh["share_gca_" + g])
    L.append(x)
L = pd.concat(L).dropna(subset=["s_pre"])
L["dg"] = L.unit_key + "#" + L.group
L["gt"] = L.group + "#" + L.yr.astype(str)
L["dt"] = L.unit_key + "#" + L.yr.astype(str)


# demean the basis once: s_pre x 1[group=g] x 1[t] and the controls s_pre x 1[t];
# any shock vector then gives z_t = sum_g shock_g * basis(g, t) by linearity.
tyrs = [t for t in yrs if t != 2017]
L = L.reset_index(drop=True)
for t in tyrs:
    L["c_{}".format(t)] = L.s_pre * (L.yr == t)
    for g in GROUPS:
        L["b_{}_{}".format(g, t)] = L.s_pre * (L.yr == t) * (L.group == g)
bcols = ["b_{}_{}".format(g, t) for t in tyrs for g in GROUPS]
ccols = ["c_{}".format(t) for t in tyrs]
w("C2: demeaning {} basis columns on {:,} rows with 3 FE".format(len(bcols) + len(ccols) + 1,
                                                                  len(L)))
Zdm = demean(L, ["y"] + ccols + bcols, ["dg", "gt", "dt"], tol=1e-9)
ydm = Zdm[:, 0]
Cdm = Zdm[:, 1:1 + len(ccols)]
Bdm = Zdm[:, 1 + len(ccols):].reshape(len(L), len(tyrs), len(GROUPS))
cl = pd.factorize(L.unit_key)[0]
G2 = cl.max() + 1


def c2_fit(shock, return_full=False):
    sv = shock.reindex(GROUPS).to_numpy()
    Zt = Bdm @ sv  # n x T
    X = np.hstack([Zt, Cdm])
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ ydm
    k = len(tyrs)
    wv = np.zeros(X.shape[1])
    wv[tyrs.index(2018)] = 0.5
    wv[tyrs.index(2019)] = 0.5
    est = float(wv @ b)
    if not return_full:
        return est
    u = ydm - X @ b
    S = np.zeros((G2, X.shape[1]))
    np.add.at(S, cl, X * u[:, None])
    N, K = X.shape
    V = XtXi @ (S.T @ S) @ XtXi * G2 / (G2 - 1) * (N - 1) / (N - K)
    names = ["z_{}".format(t) for t in tyrs] + ccols
    r = dict(b=pd.Series(b, names), V=pd.DataFrame(V, names, names), n=N, G=G2)
    return r, est, float(np.sqrt(wv @ V @ wv))


r2, est2, se2 = c2_fit(gshock, True)
pre_z = ["z_{}".format(t) for t in yrs if t < 2017]
F2, p2, k2 = wald(r2, pre_z)
perm_est = []
vals = gshock.to_numpy()
for _ in range(2000):
    perm_est.append(c2_fit(pd.Series(RNG.permutation(vals), gshock.index)))
perm_est = np.array(perm_est)
p_ri2 = (np.sum(np.abs(perm_est) >= abs(est2)) + 1) / (len(perm_est) + 1)
w("C2 group-level shift-share: avg post coef on s_pre x shock = {:+.4f} (se {:.4f}); "
  "pre-trend F {:.2f} p {:.3f} ({} leads); RI p over shock permutations {:.3f}; n {} G {}".format(
      est2, se2, F2, p2, k2, p_ri2, r2["n"], r2["G"]))
w("  interpretation: 0.1 excess log-MSP gain on a group with pre share 0.3 -> {:+.4f} share".format(
    est2 * 0.3 * 0.1))
# leave-one-group-out: zero each group's shock in turn
for g in GROUPS:
    sh_ = gshock.copy()
    sh_[g] = 0.0
    _, e_, s_ = c2_fit(sh_, True)
    w("  C2 with {:<9} shock set to 0: {:+.4f} (se {:.4f})".format(g, e_, s_))
c2ev = pd.DataFrame({"yr": [t for t in yrs if t != 2017],
                     "coef": [r2["b"].get("z_{}".format(t), np.nan) for t in yrs if t != 2017],
                     "se": [np.sqrt(r2["V"].loc["z_{}".format(t), "z_{}".format(t)])
                            for t in yrs if t != 2017]})
c2ev.to_csv(OUT + "/policy_es_msp_c2_event.csv", index=False)
pd.DataFrame([dict(avg_post=est2, se=se2, pre_F=F2, pre_p=p2, p_ri=p_ri2, n=r2["n"],
                   G=r2["G"])]).to_csv(OUT + "/policy_es_msp_c2_summary.csv", index=False)
mt.to_csv(OUT + "/policy_es_msp_shocks_crop.csv")
gshock.rename("excess_dlnmsp").to_csv(OUT + "/policy_es_msp_shocks_group.csv")

# ====================================================================== write
R = pd.DataFrame(rows)
R["did_p"] = 2 * (1 - stats.norm.cdf(np.abs(R.did / R.did_se)))
R.to_csv(OUT + "/policy_es_nfsm_summary.csv", index=False)
w("")
w("NFSM / stacked summary:")
pd.set_option("display.width", 250)
w(R[["label", "y", "did", "did_se", "did_p", "avgpost", "avgpost_se", "post_vs_pre4",
     "post_vs_pre4_se", "pre_F", "pre_p",
     "pre_k", "n", "G", "n_treat_units", "n_ctrl_units", "ymean_treat_pre"]
    ].round(4).to_string(index=False))
for k, ev in ev_store.items():
    ev.assign(stack=k[0], y=k[1]).to_csv(OUT + "/policy_es_event_{}_{}.csv".format(*k),
                                          index=False)
with open(OUT + "/policy_es_log.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG))
w("done")
