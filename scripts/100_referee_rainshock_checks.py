"""
100_referee_rainshock_checks.py

Adversarial referee checks on scripts/99_crop_groups_rainfall_shocks.py (Q3, monsoon
shocks and the crop-group mix). Lines up to the sample builder are copied unchanged from
script 99 (estimator, rainfall, panel); the only change in that block is the log file
name and extra panel flag columns. Checks are appended below.
Output: outputs/crop_groups/referee_rainshock_log.txt, referee_rainshock_checks.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "D:/crop-diversity"
OUT = REPO + "/outputs/crop_groups"
G = ["rice", "wheat", "maize", "millets", "pulses", "oilseeds", "cotton", "sugarcane", "other"]
FOCUS = ["millets", "pulses", "rice"]
LOG = open(OUT + "/referee_rainshock_log.txt", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


# ----------------------------------------------------------------------------- rainfall
rain = pd.read_csv(OUT + "/rainfall_district_year.csv")
cmap = pd.read_csv(OUT + "/crop_group_child_map.csv")
rain = rain.merge(cmap[["div_key", "unit_key"]], left_on="district_key", right_on="div_key",
                  how="left")
assert rain["unit_key"].notna().all()

SEAS = ["jjas", "ond", "jfmam_next"]


def unit_rain(df):
    out = {}
    for s in SEAS:
        ok = df["rain_" + s].notna()
        w = df.loc[ok, "area_km2"]
        out["rain_" + s] = np.average(df.loc[ok, "rain_" + s], weights=w) if ok.any() else np.nan
        out["qc_" + s] = int(df["qc_" + s].fillna(0).max() > 0)
    return pd.Series(out)


ur = rain.groupby(["unit_key", "year"]).apply(unit_rain).reset_index()
for s in SEAS:
    base = ur[ur["qc_" + s] == 0].groupby("unit_key")["rain_" + s].agg(["mean", "std"])
    ur = ur.merge(base.rename(columns={"mean": "m_" + s, "std": "sd_" + s}),
                  left_on="unit_key", right_index=True, how="left")
    ur["pct_" + s] = 100 * (ur["rain_" + s] / ur["m_" + s] - 1)
    ur["z_" + s] = (ur["rain_" + s] - ur["m_" + s]) / ur["sd_" + s]
    small = ur["m_" + s] < 10
    ur.loc[small, "pct_" + s] = np.nan
say("unit rainfall rows", len(ur), "units", ur["unit_key"].nunique())

BINS = [-np.inf, -40, -20, -10, 10, 20, 40, np.inf]
BLAB = ["le_m40", "m40_m20", "m20_m10", "normal", "p10_p20", "p20_p40", "ge_p40"]
BTXT = ["<= -40", "-40 to -20", "-20 to -10", "-10 to +10", "+10 to +20", "+20 to +40",
        ">= +40"]


def shock_cols(d, pre):
    """Add deficient / excess dummies, bins and piecewise z for a JJAS column prefix."""
    p = d[pre + "pct"]
    d[pre + "def"] = (p <= -20).astype(float)
    d[pre + "exc"] = (p >= 20).astype(float)
    b = pd.cut(p, BINS, labels=BLAB, right=False)
    for lab in BLAB:
        d[pre + "b_" + lab] = (b == lab).astype(float)
    d[pre + "zdry"] = (-d[pre + "z"]).clip(lower=0)
    d[pre + "zwet"] = d[pre + "z"].clip(lower=0)
    nanrow = p.isna()
    for c in [pre + "def", pre + "exc"] + [pre + "b_" + l for l in BLAB]:
        d.loc[nanrow, c] = np.nan
    return d


R = ur.rename(columns={"year": "yr", "pct_jjas": "s_pct", "z_jjas": "s_z", "qc_jjas": "s_qc",
                       "pct_ond": "ond_pct", "z_ond": "ond_z", "qc_ond": "ond_qc",
                       "pct_jfmam_next": "jfm_pct", "z_jfmam_next": "jfm_z",
                       "qc_jfmam_next": "jfm_qc"})
R = shock_cols(R, "s_")
R = R[["unit_key", "yr", "s_pct", "s_z", "s_qc", "s_def", "s_exc", "s_zdry", "s_zwet"]
      + ["s_b_" + l for l in BLAB] + ["ond_z", "ond_qc", "jfm_z", "jfm_qc", "rain_jjas"]]

# ----------------------------------------------------------------------------- irrigation
fp = pd.read_csv(REPO + "/outputs/shrug_covariates/final_panel.csv",
                 usecols=["shrug_key", "in_final", "irr_share", "mean_annual_cropped_area"])
fp = fp[fp["in_final"] == True]
irr = fp.groupby("shrug_key").apply(
    lambda x: np.average(x["irr_share"], weights=x["mean_annual_cropped_area"])).rename("irr")
irr = irr.reset_index().rename(columns={"shrug_key": "unit_key"})
irr["irr_t"] = pd.qcut(irr["irr"], 3, labels=[1, 2, 3]).astype(int)
say("irrigation units", len(irr), "tercile cut points",
    irr.groupby("irr_t")["irr"].agg(["min", "max", "mean", "size"]).round(3).to_dict("index"))

# ----------------------------------------------------------------------------- crop panel
pan = pd.read_csv(OUT + "/crop_group_panel.csv", low_memory=False)
pan = pan[pan["in_606"] == 1]
keep = ["unit_key", "state", "year_start", "season_area_ha", "gca_ha", "flag_season_unreliable",
        "flag_annual_unreliable", "f_whole_year_ge20", "f_tn_rice_single_season", "f_gca_jump_severe", "f_gca_jump_mild", "f_pre_split_donor", "f_before_entry", "f_child_gap", "f_late_entrant"]
keep += ["share_season_" + g for g in G] + ["area_ha_" + g for g in G]


def season(s):
    d = pan[pan["season"] == s][keep].copy()
    for g in G:
        d["sh_" + g] = 100 * d["share_season_" + g]           # percentage points
        d["la_" + g] = np.log1p(d["area_ha_" + g])
    d["la_total"] = np.log(d["season_area_ha"].where(d["season_area_ha"] > 0))
    return d


rabi = season("rabi")
khar = season("kharif")
ann = season("annual")
meangca = ann.groupby("unit_key")["gca_ha"].mean().rename("w_gca")
nyrs = ann.groupby("unit_key")["year_start"].nunique().rename("n_years")

# ----------------------------------------------------------------------------- estimator


def demean(M, groups, tol=1e-10, maxit=500):
    """Alternating projections: sweep out each FE in turn until the change is tiny."""
    M = M.copy()
    codes = [pd.factorize(g)[0] for g in groups]
    cnt = [np.bincount(c) for c in codes]
    for it in range(maxit):
        old = M.copy()
        for c, n in zip(codes, cnt):
            sums = np.zeros((n.size, M.shape[1]))
            np.add.at(sums, c, M)
            M -= (sums / n[:, None])[c]
        if np.max(np.abs(M - old)) < tol:
            break
    return M


def demean_w(M, groups, w, tol=1e-10, maxit=500):
    M = M.copy()
    codes = [pd.factorize(g)[0] for g in groups]
    for it in range(maxit):
        old = M.copy()
        for c in codes:
            sw = np.bincount(c, weights=w)
            sums = np.zeros((sw.size, M.shape[1]))
            np.add.at(sums, c, M * w[:, None])
            M -= (sums / sw[:, None])[c]
        if np.max(np.abs(M - old)) < tol:
            break
    return M


def fe_ols(d, y, X, fe=("unit_key", "sy"), cluster="unit_key", w=None):
    d = d.dropna(subset=[y] + X + list(fe) + [cluster]).copy()
    # drop singletons on each FE (repeat until stable)
    while True:
        n0 = len(d)
        for f in fe:
            d = d[d.groupby(f)[f].transform("size") > 1]
        if len(d) == n0:
            break
    M = d[[y] + X].to_numpy(float)
    groups = [d[f].to_numpy() for f in fe]
    if w is None:
        Md = demean(M, groups)
        ww = np.ones(len(d))
    else:
        ww = d[w].to_numpy(float)
        Md = demean_w(M, groups, ww)
    yy, XX = Md[:, 0], Md[:, 1:]
    sw = np.sqrt(ww)
    Xs, ys = XX * sw[:, None], yy * sw
    XtX_inv = np.linalg.pinv(Xs.T @ Xs)
    b = XtX_inv @ Xs.T @ ys
    e = ys - Xs @ b
    cl = pd.factorize(d[cluster])[0]
    Gc = cl.max() + 1
    S = np.zeros((Gc, XX.shape[1]))
    np.add.at(S, cl, Xs * e[:, None])
    N, K = len(d), XX.shape[1]
    adj = Gc / (Gc - 1) * (N - 1) / (N - K)
    V = adj * XtX_inv @ (S.T @ S) @ XtX_inv
    se = np.sqrt(np.diag(V))
    ybar = np.average(d[y], weights=ww)
    return dict(b=pd.Series(b, X), se=pd.Series(se, X), V=pd.DataFrame(V, X, X), n=N,
                g=Gc, units=d["unit_key"].nunique(), ymean=ybar, d=d)


from scipy import stats


def row(res, var, **meta):
    b, s = res["b"][var], res["se"][var]
    t = b / s
    p = 2 * stats.t.sf(abs(t), res["g"] - 1)
    return dict(meta, term=var, coef=b, se=s, ci_lo=b - 1.96 * s, ci_hi=b + 1.96 * s, p=p,
                n=res["n"], clusters=res["g"], ymean=res["ymean"])


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


# ----------------------------------------------------------------------------- samples


def build(outcome_df, shock_lag, ctrl_same=False, drop_flag=True, drop_qc=True):
    """outcome at crop year t, shock JJAS(t - shock_lag)."""
    d = outcome_df.copy()
    d["yr"] = d["year_start"] - shock_lag
    d = d.merge(R, on=["unit_key", "yr"], how="left")
    if ctrl_same:   # kharif(t+1): also control for the monsoon of t+1 itself
        c = R[["unit_key", "yr", "s_def", "s_exc", "s_qc", "s_zdry", "s_zwet"]].rename(
            columns={"yr": "year_start", "s_def": "c_def", "s_exc": "c_exc", "s_qc": "c_qc",
                     "s_zdry": "c_zdry", "s_zwet": "c_zwet"})
        d = d.merge(c, on=["unit_key", "year_start"], how="left")
    d = d.merge(irr, on="unit_key", how="left").merge(meangca, on="unit_key", how="left")
    d = d.merge(nyrs, on="unit_key", how="left")
    d["sy"] = d["state"] + "_" + d["year_start"].astype(str)
    d["yfe"] = d["year_start"].astype(str)
    if drop_flag:
        d = d[d["flag_season_unreliable"] == 0]
    if drop_qc:
        d = d[d["s_qc"] == 0]
        if ctrl_same:
            d = d[d["c_qc"] == 0]
    d = d[d["season_area_ha"] > 0]
    for t in (1, 2, 3):
        d[f"def_T{t}"] = d["s_def"] * (d["irr_t"] == t)
        d[f"exc_T{t}"] = d["s_exc"] * (d["irr_t"] == t)
    return d


# rabi(t) <- JJAS(t);  kharif(t+1) <- JJAS(t);  kharif(t) <- JJAS(t) reference
SAMPLES = {
    "rabi_t": (rabi, 0, False),
    "kharif_t1": (khar, 1, True),
    "kharif_t": (khar, 0, False),
}
OUTC = [("sh_" + g, "share") for g in G] + [("la_" + g, "logarea") for g in G] + \
       [("la_total", "logarea")]

# =========================================================================== REFEREE CHECKS
rows = []


def rec(check, sample, y, res, var, **kw):
    r = row(res, var, check=check, sample=sample, outcome=y, **kw)
    rows.append(r)
    say(f"{check:45s} {sample:9s} {y:12s} {var:10s} b={r['coef']:+.3f} se={r['se']:.3f} "
        f"p={r['p']:.3f} n={r['n']} g={r['clusters']}")
    return r


D = {name: build(df, lag, ctrl) for name, (df, lag, ctrl) in SAMPLES.items()}
CX = {"rabi_t": [], "kharif_t1": ["c_def", "c_exc"], "kharif_t": []}
KEY = [("rabi_t", "sh_millets", "s_def"), ("rabi_t", "sh_rice", "s_def"),
       ("rabi_t", "la_total", "s_def"), ("rabi_t", "sh_pulses", "s_def"),
       ("rabi_t", "la_pulses", "s_def"), ("rabi_t", "la_millets", "s_def"),
       ("rabi_t", "la_rice", "s_def"),
       ("kharif_t1", "sh_millets", "s_exc"), ("kharif_t1", "la_millets", "s_exc"),
       ("kharif_t1", "la_total", "s_exc"), ("kharif_t1", "sh_pulses", "s_def")]


def run_key(check, transform=None, **kw):
    for name, y, v in KEY:
        d = D[name] if transform is None else transform(D[name], name)
        r = fe_ols(d, y, ["s_def", "s_exc"] + CX[name], **kw)
        rec(check, name, y, r, v)


say("\n### 1. reproduce baseline")
run_key("baseline")

say("\n### 2. also drop annual-unreliable unit-years (GCA jumps, pre-split donors, entry)")
run_key("drop flag_annual_unreliable", lambda d, n: d[d["flag_annual_unreliable"] == 0])
run_key("drop annual-unreliable + mild jump + 1997",
        lambda d, n: d[(d["flag_annual_unreliable"] == 0) & (d["f_gca_jump_mild"] == 0)
                       & (d["year_start"] > 1997)])

say("\n### 3. clustering on state (shocks are spatially correlated)")
run_key("cluster state", cluster="state")

say("\n### 4. unit-specific linear trends (with state-year FE)")


def demean_trend(M, unit, sy, yr, tol=1e-9, maxit=2000):
    M = M.copy()
    uc = pd.factorize(unit)[0]
    sc = pd.factorize(sy)[0]
    t = yr.astype(float)
    nu = uc.max() + 1
    ns = np.bincount(sc)
    n = np.bincount(uc, minlength=nu).astype(float)
    st = np.bincount(uc, t, nu)
    stt = np.bincount(uc, t * t, nu)
    det = n * stt - st ** 2
    for it in range(maxit):
        old = M.copy()
        for j in range(M.shape[1]):
            sy_ = np.bincount(uc, M[:, j], nu)
            sty = np.bincount(uc, M[:, j] * t, nu)
            b1 = np.where(det > 1e-9, (n * sty - st * sy_) / np.where(det > 1e-9, det, 1), 0)
            b0 = (sy_ - b1 * st) / n
            M[:, j] -= b0[uc] + b1[uc] * t
        sums = np.zeros((ns.size, M.shape[1]))
        np.add.at(sums, sc, M)
        M -= (sums / ns[:, None])[sc]
        if np.max(np.abs(M - old)) < tol:
            break
    return M


def fe_trend(d, y, X):
    d = d.dropna(subset=[y] + X).copy()
    d = d[d.groupby("unit_key")["unit_key"].transform("size") > 2]
    d = d[d.groupby("sy")["sy"].transform("size") > 1]
    Md = demean_trend(d[[y] + X].to_numpy(float), d["unit_key"].to_numpy(), d["sy"].to_numpy(),
                      d["year_start"].to_numpy())
    yy, XX = Md[:, 0], Md[:, 1:]
    XtX_inv = np.linalg.pinv(XX.T @ XX)
    b = XtX_inv @ XX.T @ yy
    e = yy - XX @ b
    cl = pd.factorize(d["unit_key"])[0]
    Gc = cl.max() + 1
    S = np.zeros((Gc, XX.shape[1]))
    np.add.at(S, cl, XX * e[:, None])
    N, K = len(d), XX.shape[1]
    V = Gc / (Gc - 1) * (N - 1) / (N - K) * XtX_inv @ (S.T @ S) @ XtX_inv
    return dict(b=pd.Series(b, X), se=pd.Series(np.sqrt(np.diag(V)), X), n=N, g=Gc,
                ymean=d[y].mean())


for name, y, v in KEY:
    rec("unit linear trends + state-year FE", name, y,
        fe_trend(D[name], y, ["s_def", "s_exc"] + CX[name]), v)

say("\n### 5. placebos: lag JJAS(t-1) on rabi(t); lead JJAS(t+2) on kharif(t+1)")
Rl = R[["unit_key", "yr", "s_def", "s_exc", "s_qc"]]
lag1 = Rl.rename(columns={"yr": "year_start", "s_def": "lag_def", "s_exc": "lag_exc",
                          "s_qc": "lag_qc"})
lag1["year_start"] = lag1["year_start"] + 1
d = D["rabi_t"].merge(lag1, on=["unit_key", "year_start"], how="left")
d = d[d["lag_qc"] == 0]
for y in ["sh_millets", "sh_rice", "la_total", "sh_pulses"]:
    r = fe_ols(d, y, ["s_def", "s_exc", "lag_def", "lag_exc"])
    rec("placebo lag JJAS(t-1), with JJAS(t)", "rabi_t", y, r, "lag_def")
    rec("placebo lag JJAS(t-1), with JJAS(t)", "rabi_t", y, r, "lag_exc")
lead2 = Rl.rename(columns={"yr": "year_start", "s_def": "ld_def", "s_exc": "ld_exc",
                           "s_qc": "ld_qc"})
lead2["year_start"] = lead2["year_start"] - 1
d = D["kharif_t1"].merge(lead2, on=["unit_key", "year_start"], how="left")
d = d[d["ld_qc"] == 0]
for y in ["sh_millets", "la_millets", "la_total"]:
    r = fe_ols(d, y, ["s_def", "s_exc", "c_def", "c_exc", "ld_def", "ld_exc"])
    rec("placebo lead JJAS(t+2) on kharif(t+1)", "kharif_t1", y, r, "ld_exc")
    rec("placebo lead JJAS(t+2) on kharif(t+1)", "kharif_t1", y, r, "ld_def")
for y in ["sh_millets", "la_millets"]:
    r = fe_ols(D["kharif_t"], y, ["s_def", "s_exc"])
    rec("same-year kharif(t)", "kharif_t", y, r, "s_exc")

say("\n### 6. intensive margin: log area among units that actually grow the crop that season")
for name, g, v in [("rabi_t", "millets", "s_def"), ("rabi_t", "rice", "s_def"),
                   ("rabi_t", "pulses", "s_def"), ("kharif_t1", "millets", "s_exc")]:
    d = D[name]
    msh = d.groupby("unit_key")["sh_" + g].transform("mean")
    for thr in (2, 5):
        dd = d[(msh >= thr) & (d["area_ha_" + g] > 0)].copy()
        dd["lnA"] = np.log(dd["area_ha_" + g])
        r = fe_ols(dd, "lnA", ["s_def", "s_exc"] + CX[name])
        rec(f"log {g} area, mean share>={thr}pp, A>0", name, "ln_" + g, r, v)
        r = fe_ols(dd, "la_total", ["s_def", "s_exc"] + CX[name])
        rec(f"log total area, same units ({g},{thr})", name, "la_total", r, v)
        r = fe_ols(dd, "sh_" + g, ["s_def", "s_exc"] + CX[name])
        rec(f"share, same units ({g},{thr})", name, "sh_" + g, r, v)

say("\n### 7. mechanical check: share change implied by flat crop area and shrinking total")
for name, g, v in [("rabi_t", "millets", "s_def"), ("rabi_t", "rice", "s_def"),
                   ("kharif_t1", "millets", "s_exc")]:
    d = D[name]
    rt = fe_ols(d, "la_total", ["s_def", "s_exc"] + CX[name])
    rs = fe_ols(d, "sh_" + g, ["s_def", "s_exc"] + CX[name])
    say(f"  {name} {g}: observed share {rs['b'][v]:+.3f}; pure-denominator prediction "
        f"-mean share x dlog total = {-rs['ymean'] * rt['b'][v]:+.3f}")

say("\n### 8. leave-one-state-out and leave-one-year-out")
for name, y, v in [("rabi_t", "sh_millets", "s_def"), ("rabi_t", "sh_rice", "s_def"),
                   ("rabi_t", "la_total", "s_def"), ("kharif_t1", "sh_millets", "s_exc")]:
    d = D[name]
    lo = []
    for s in sorted(d["state"].unique()):
        r = fe_ols(d[d["state"] != s], y, ["s_def", "s_exc"] + CX[name])
        lo.append((s, r["b"][v], r["se"][v]))
    lo = pd.DataFrame(lo, columns=["state", "b", "se"])
    i0, i1 = lo.b.idxmin(), lo.b.idxmax()
    say(f"  LOSO {name} {y} {v}: range {lo.b[i0]:+.3f} ({lo.state[i0]}) "
        f"to {lo.b[i1]:+.3f} ({lo.state[i1]}); min |t| {np.min(np.abs(lo.b / lo.se)):.2f}")
    ly = []
    for t in sorted(d["year_start"].unique()):
        r = fe_ols(d[d["year_start"] != t], y, ["s_def", "s_exc"] + CX[name])
        ly.append((t, r["b"][v], r["se"][v]))
    ly = pd.DataFrame(ly, columns=["yr", "b", "se"])
    i0, i1 = ly.b.idxmin(), ly.b.idxmax()
    say(f"  LOYO {name} {y} {v}: range {ly.b[i0]:+.3f} ({ly.yr[i0]}) "
        f"to {ly.b[i1]:+.3f} ({ly.yr[i1]}); min |t| {np.min(np.abs(ly.b / ly.se)):.2f}")
    say("   shock unit-years by state (top 6):",
        d.groupby("state")[v].sum().sort_values(ascending=False).head(6).astype(int).to_dict())
    say("   shock unit-years by year (top 6):",
        d.groupby("year_start")[v].sum().sort_values(ascending=False).head(6).astype(int).to_dict())

say("\n### 9. randomization inference: permute whole rainfall years (keeps spatial correlation)")
rng = np.random.default_rng(20261107)
yrs_all = np.array(sorted(R["yr"].unique()))
PERM = 300
Rsave = R
for name, y, v in [("rabi_t", "sh_millets", "s_def"), ("rabi_t", "sh_rice", "s_def"),
                   ("rabi_t", "la_total", "s_def"), ("kharif_t1", "sh_millets", "s_exc")]:
    df0, lag, ctrl = SAMPLES[name]
    base = fe_ols(D[name], y, ["s_def", "s_exc"] + CX[name])["b"][v]
    draws = []
    for k in range(PERM):
        mp = dict(zip(yrs_all, rng.permutation(yrs_all)))
        Rp = Rsave.copy()
        Rp["yr"] = Rp["yr"].map(mp)
        R = Rp
        dp = build(df0, lag, ctrl)
        draws.append(fe_ols(dp, y, ["s_def", "s_exc"] + CX[name])["b"][v])
    R = Rsave
    draws = np.array(draws)
    p = (np.sum(np.abs(draws) >= abs(base)) + 1) / (PERM + 1)
    say(f"  RI {name} {y} {v}: b={base:+.3f}; permutation sd {draws.std():.3f}; RI p={p:.3f}")
    rows.append(dict(check="randomization inference (year permutation)", sample=name,
                     outcome=y, term=v, coef=base, se=draws.std(), p=p, n=len(D[name]),
                     clusters=np.nan))

say("\n### 10. irrigation tercile: low-tercile rabi pulses and wheat")
d = D["rabi_t"].dropna(subset=["irr_t"])
Xi = [f"def_T{t}" for t in (1, 2, 3)] + [f"exc_T{t}" for t in (1, 2, 3)]
for y in ["sh_pulses", "sh_wheat", "la_pulses", "sh_millets", "la_total"]:
    r = fe_ols(d, y, Xi)
    rec("irrigation terciles baseline", "rabi_t", y, r, "def_T1")
    r = fe_ols(d, y, Xi, cluster="state")
    rec("irrigation terciles, cluster state", "rabi_t", y, r, "def_T1")
    lo = []
    for s in sorted(d.loc[d["irr_t"] == 1, "state"].unique()):
        rr = fe_ols(d[d["state"] != s], y, Xi)
        lo.append((s, rr["b"]["def_T1"], rr["se"]["def_T1"]))
    lo = pd.DataFrame(lo, columns=["state", "b", "se"])
    i0, i1 = lo.b.idxmin(), lo.b.idxmax()
    say(f"  LOSO low-tercile {y}: range {lo.b[i0]:+.3f} ({lo.state[i0]}) to "
        f"{lo.b[i1]:+.3f} ({lo.state[i1]})")
say("  low-tercile deficient rabi unit-years by state:",
    d[d["irr_t"] == 1].groupby("state")["s_def"].sum().sort_values(ascending=False).head(8)
    .astype(int).to_dict())
say("  low-tercile mean rabi wheat share by state:",
    d[d["irr_t"] == 1].groupby("state")["sh_wheat"].mean().sort_values(ascending=False)
    .head(8).round(1).to_dict())

pd.DataFrame(rows).to_csv(OUT + "/referee_rainshock_checks.csv", index=False)
LOG.close()
