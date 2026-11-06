"""
99_crop_groups_rainfall_shocks.py

Q3. Does a deficient or excess monsoon (June-September rainfall against the district's
own 1990-2020 history) move land between crop groups in the rabi that follows it and in
the next year's kharif?

Inputs (all produced earlier in this repo):
  outputs/crop_groups/crop_group_panel.csv        unit x crop year x season, 2011 parents
  outputs/crop_groups/crop_group_child_map.csv    APY district key -> 2011 parent unit
  outputs/crop_groups/rainfall_district_year.csv  IMD 0.25 deg district rainfall 1990-2020
  outputs/shrug_covariates/final_panel.csv        irrigation share (Mission Antyodaya)

Timing. Crop year t is "t-(t+1)". Its kharif is sown in the June-September monsoon of
calendar year t; its rabi is sown in Oct-Dec of t on the moisture that monsoon left. So
the shock JJAS(t) is matched to rabi(t) and to kharif(t+1). Kharif(t) is shown as the
same-season reference, where drought acts on sowing and harvest directly.

Estimator. OLS with unit FE and state-by-year FE, absorbed by alternating projections.
Standard errors clustered on the 2011-parent unit (605 groups, which is the shared-parent
clustering used across the repo). No external FE package is needed.

Run:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/99_crop_groups_rainfall_shocks.py
Outputs go to outputs/crop_groups/ with the prefix rainshock_.
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
LOG = open(OUT + "/rainshock_log.txt", "w", encoding="utf-8")


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
        "flag_annual_unreliable", "f_whole_year_ge20", "f_tn_rice_single_season"]
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

main_rows, bin_rows, irr_rows, rob_rows = [], [], [], []
datasets = {}
for name, (df, lag, ctrl) in SAMPLES.items():
    d = build(df, lag, ctrl)
    datasets[name] = d
    say(f"\n== {name}: rows {len(d)}, units {d['unit_key'].nunique()}, years "
        f"{d['year_start'].min()}-{d['year_start'].max()}, deficient {int(d['s_def'].sum())}, "
        f"excess {int(d['s_exc'].sum())}")
    ctrlX = ["c_def", "c_exc"] if ctrl else []
    for y, kind in OUTC:
        X = ["s_def", "s_exc"] + ctrlX
        r = fe_ols(d, y, X)
        for v in ["s_def", "s_exc"]:
            main_rows.append(row(r, v, sample=name, outcome=y, kind=kind))
        # dose-response bins (reference: -10 to +10)
        Xb = ["s_b_" + l for l in BLAB if l != "normal"] + ctrlX
        rb = fe_ols(d, y, Xb)
        for l in BLAB:
            if l == "normal":
                bin_rows.append(dict(sample=name, outcome=y, kind=kind, term="s_b_normal",
                                     coef=0.0, se=0.0, ci_lo=0, ci_hi=0, p=np.nan,
                                     n=rb["n"], clusters=rb["g"], ymean=rb["ymean"]))
            else:
                bin_rows.append(row(rb, "s_b_" + l, sample=name, outcome=y, kind=kind))
        # irrigation terciles
        Xi = [f"def_T{t}" for t in (1, 2, 3)] + [f"exc_T{t}" for t in (1, 2, 3)] + ctrlX
        ri = fe_ols(d.dropna(subset=["irr_t"]), y, Xi)
        for v in Xi[:6]:
            irr_rows.append(row(ri, v, sample=name, outcome=y, kind=kind))
        for sh in ("def", "exc"):
            a, c = f"{sh}_T1", f"{sh}_T3"
            diff = ri["b"][a] - ri["b"][c]
            sd = np.sqrt(ri["V"].loc[a, a] + ri["V"].loc[c, c] - 2 * ri["V"].loc[a, c])
            irr_rows.append(dict(sample=name, outcome=y, kind=kind, term=f"{sh}_T1_minus_T3",
                                 coef=diff, se=sd, ci_lo=diff - 1.96 * sd,
                                 ci_hi=diff + 1.96 * sd,
                                 p=2 * stats.t.sf(abs(diff / sd), ri["g"] - 1), n=ri["n"],
                                 clusters=ri["g"], ymean=ri["ymean"]))

main = pd.DataFrame(main_rows)
# BH within each sample over the 9 share outcomes x 2 shocks (18 tests)
main["q_bh_shares"] = np.nan
for s in SAMPLES:
    m = (main["sample"] == s) & (main["kind"] == "share")
    main.loc[m, "q_bh_shares"] = bh(main.loc[m, "p"])
main.to_csv(OUT + "/rainshock_main.csv", index=False)
bins = pd.DataFrame(bin_rows)
bins["bin"] = bins["term"].str.replace("s_b_", "", regex=False)
bins.to_csv(OUT + "/rainshock_bins.csv", index=False)
irrd = pd.DataFrame(irr_rows)
irrd.to_csv(OUT + "/rainshock_irrigation.csv", index=False)

# bin counts
for name, d in datasets.items():
    cnt = {l: int(d["s_b_" + l].sum()) for l in BLAB}
    say(name, "bin counts", cnt)

# ----------------------------------------------------------------------------- robustness
ROB_Y = ["sh_millets", "sh_pulses", "sh_rice", "sh_wheat", "la_total"]


def rob(label, name, d, X, fe=("unit_key", "sy"), cluster="unit_key", w=None, terms=None):
    for y in ROB_Y:
        r = fe_ols(d, y, X, fe=fe, cluster=cluster, w=w)
        for v in (terms or ["s_def", "s_exc"]):
            rob_rows.append(row(r, v, sample=name, outcome=y, spec=label))


for name, (df, lag, ctrl) in SAMPLES.items():
    if name == "kharif_t":
        continue
    d = datasets[name]
    cx = ["c_def", "c_exc"] if ctrl else []
    rob("baseline", name, d, ["s_def", "s_exc"] + cx)
    rob("unit + year FE", name, d, ["s_def", "s_exc"] + cx, fe=("unit_key", "yfe"))
    rob("cluster on state", name, d, ["s_def", "s_exc"] + cx, cluster="state")
    rob("weighted by mean GCA", name, d, ["s_def", "s_exc"] + cx, w="w_gca")
    rob("piecewise z (per SD)", name, d, ["s_zdry", "s_zwet"] + cx, terms=["s_zdry", "s_zwet"])
    d2 = build(df, lag, ctrl, drop_qc=False)
    rob("keep rainfall-QC rows", name, d2, ["s_def", "s_exc"] + cx)
    rob("drop Whole-Year >=20% rows", name, d[d["f_whole_year_ge20"] == 0],
        ["s_def", "s_exc"] + cx)
    rob("drop 1997-98", name, d[d["year_start"] > 1997], ["s_def", "s_exc"] + cx)
    rob("units seen >= 20 years", name, d[d["n_years"] >= 20], ["s_def", "s_exc"] + cx)
    if name == "rabi_t":
        dd = d.merge(R[["unit_key", "yr", "ond_z", "ond_qc", "jfm_z", "jfm_qc"]].rename(
            columns={"ond_z": "oz", "ond_qc": "oq", "jfm_z": "jz", "jfm_qc": "jq"}),
            on=["unit_key", "yr"], how="left")
        dd = dd[(dd["oq"] == 0) & (dd["jq"] == 0)]
        rob("control Oct-Dec and Jan-May rain (z)", name, dd, ["s_def", "s_exc", "oz", "jz"])
        # lead placebo: next year's monsoon cannot act on this rabi
        lead = R[["unit_key", "yr", "s_def", "s_exc", "s_qc"]].rename(
            columns={"yr": "ly", "s_def": "lead_def", "s_exc": "lead_exc", "s_qc": "lead_qc"})
        dl = d.assign(ly=d["year_start"] + 1).merge(lead, on=["unit_key", "ly"], how="left")
        dl = dl[dl["lead_qc"] == 0]
        rob("placebo: JJAS(t+1) lead, with JJAS(t)", name, dl,
            ["s_def", "s_exc", "lead_def", "lead_exc"], terms=["lead_def", "lead_exc"])
    else:
        rob("no control for JJAS(t+1)", name, d, ["s_def", "s_exc"])

robd = pd.DataFrame(rob_rows)
robd.to_csv(OUT + "/rainshock_robustness.csv", index=False)

# ----------------------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
FIGY = [("sh_millets", "Millets share of season area (pp)"),
        ("sh_pulses", "Pulses share of season area (pp)"),
        ("sh_rice", "Rice share of season area (pp)"),
        ("sh_wheat", "Wheat share of season area (pp)"),
        ("la_total", "Log season area (all crops)"),
        ("sh_oilseeds", "Oilseeds share of season area (pp)")]
TITLE = {"rabi_t": "Rabi of the same crop year",
         "kharif_t1": "Kharif of the next crop year",
         "kharif_t": "Kharif of the same crop year (reference)"}
for name in SAMPLES:
    fig, axs = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
    for ax, (y, lab) in zip(axs.flat, FIGY):
        b = bins[(bins["sample"] == name) & (bins["outcome"] == y)].set_index("bin").loc[BLAB]
        x = np.arange(len(BLAB))
        ax.axhline(0, color="0.6", lw=0.8)
        ax.errorbar(x, b["coef"], yerr=1.96 * b["se"], fmt="o", color="black", ms=4,
                    capsize=2, lw=1)
        ax.set_title(lab, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(BTXT, rotation=45, ha="right", fontsize=7)
    fig.suptitle(f"{TITLE[name]}: response to June-September rainfall, percent departure from "
                 "the district's own mean\n(reference bin -10 to +10; bars are 95% intervals)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT + f"/rainshock_bins_{name}.png", dpi=170)
    plt.close(fig)

# irrigation tercile figure: deficient-monsoon coefficient by tercile
fig, axs = plt.subplots(1, 4, figsize=(11, 3.2))
for ax, (y, lab) in zip(axs, FIGY[:4]):
    for j, (name, mk) in enumerate([("rabi_t", "o"), ("kharif_t1", "s")]):
        sub = irrd[(irrd["sample"] == name) & (irrd["outcome"] == y)].set_index("term")
        xs = np.arange(3) + (j - 0.5) * 0.25
        cs = [sub.loc[f"def_T{t}", "coef"] for t in (1, 2, 3)]
        es = [1.96 * sub.loc[f"def_T{t}", "se"] for t in (1, 2, 3)]
        ax.errorbar(xs, cs, yerr=es, fmt=mk, color="black" if j == 0 else "0.55", ms=4,
                    capsize=2, lw=1, label="rabi, same year" if j == 0 else "kharif, next year")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["low irr.", "mid irr.", "high irr."])
    ax.set_title(lab, fontsize=8.5)
axs[0].legend(fontsize=7, frameon=False)
fig.suptitle("Deficient monsoon (<= -20%) by irrigation tercile, 95% intervals", fontsize=10)
fig.tight_layout()
fig.savefig(OUT + "/rainshock_irrigation_terciles.png", dpi=170)
plt.close(fig)

# ----------------------------------------------------------------------------- print summary
pd.set_option("display.width", 220)
fmt = lambda df: df.assign(**{c: df[c].round(3) for c in ["coef", "se", "p", "ymean"]
                              if c in df})
say("\nMAIN (shares in pp, log areas in log points)")
say(fmt(main[["sample", "outcome", "term", "coef", "se", "p", "q_bh_shares", "n", "ymean"]])
    .to_string())
say("\nIRRIGATION")
say(fmt(irrd[irrd["outcome"].isin(ROB_Y)][["sample", "outcome", "term", "coef", "se", "p",
                                             "n"]]).to_string())
say("\nROBUSTNESS")
say(fmt(robd[["sample", "spec", "outcome", "term", "coef", "se", "p", "n"]]).to_string())
say("\nBINS (focus)")
say(fmt(bins[bins["outcome"].isin(ROB_Y)][["sample", "outcome", "bin", "coef", "se", "p",
                                           "n"]]).to_string())
LOG.close()
