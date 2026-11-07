"""
104_season_separated_diversity.py

Seasons and double counting. The annual Hill numbers pool every season's area by
crop, so a district that grows rice in kharif and wheat in rabi on the same land
scores as two crops. This script separates the two sources of annual diversity.

  (1) Within-season Hill numbers, per district-year:
        kharif pool  Kharif + Autumn + Winter rows
        rabi pool    Rabi rows
        summer pool  Summer rows
      MAIN: Whole Year crops (sugarcane, coconut, banana...) stand in the field in
      every season, so they are added to each season pool that has its own area.
      ALT:  Whole Year rows left out of the season pools.
      D1_ws = area-weighted mean of the season pools' D1 (weights = pool area).

  (2) Exact Shannon decomposition on the exclusive partition {kharif, rabi,
      summer, whole year}: H_annual = H_alpha + H_beta with H_alpha = sum_s w_s H_s,
      w_s the block's share of gross cropped area. D1_annual = A x B, where
      A = exp(H_alpha) and B = exp(H_beta) >= 1 is the between-season multiplier.
      H_beta / H_annual is the share of annual entropy that is between seasons.

  (3) Cropping intensity, four ways:
        ci_lus     DES Land Use Statistics GCA / net sown area, district-year, read from
                   outputs/revision_nov8/irrigation_panel_district_year.csv (script
                   104_irrigation_panel_build.py must run first). Kept if 0.9-3.0.
        ci_apy     APY gross cropped area / (Whole Year + largest single season).
                   The denominator is a LOWER bound on net sown area, so this is an
                   UPPER bound on intensity.
        ci_ay      Mission Antyodaya 2019-20, (kharif + rabi + other sown) / net sown.
        secc_2crop SECC 2011 share of household land that is irrigated and carries
                   two or more crops.

  (4) Rankings: annual D1 against D1_ws, biggest movers.
  (5) Irrigation inverted U on within-season measures, the site's spec:
      y ~ irr + irr^2 + C(state), clustered on the 2011 parent (shrug_key), on the
      606-district sample.

Season reliability: unit-year flag_season_unreliable from
outputs/crop_groups/crop_group_flags.csv (Odisha 2001-03 collapsed seasons, Odisha
2012 summer rice, Bihar 1998-2001 rice swap, Tamil Nadu rice before 2017-18,
Whole Year >= 50 percent of GCA). Districts inherit their 2011 parent's flag.

Cleaning is script 76's, copied verbatim.

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/104_season_separated_diversity.py
Outputs (outputs/revision_nov8/):
  seasons_district_year.csv, seasons_district.csv, seasons_regressions.csv,
  seasons_movers.csv, seasons_log.txt
"""
import json
import os

import numpy as np
import pandas as pd

from _years import keep_years
import statsmodels.formula.api as smf
from scipy import stats

RAW = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
REPO = r"D:/crop-diversity"
DIV = REPO + "/outputs/crop_diversity_analysis"
COV = REPO + "/outputs/shrug_covariates"
CG = REPO + "/outputs/crop_groups"
OUT = REPO + "/outputs/revision_nov8"
os.makedirs(OUT, exist_ok=True)
MIN_YEARS = 5

LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


# ------------------------------------------------------------------ load, script 76 cleaning
df = pd.read_csv(RAW)
for c in ["year", "season", "state_name", "district_name", "crop_name", "crop_type"]:
    df[c] = df[c].astype(str).str.strip()
df = df.dropna(subset=["area"])
df = df[df["area"] > 0]
BOGUS = [("Delhi", "Chandigarh"), ("Goa", "Chandigarh"),
         ("The Dadra And Nagar Haveli And Daman And Diu", "Chandigarh"),
         ("Delhi", "Surguja"), ("Goa", "Surguja"),
         ("The Dadra And Nagar Haveli And Daman And Diu", "Surguja"),
         ("Andaman And Nicobar Islands", "Purulia")]
REMAP = {("Uttar Pradesh", d): "Uttarakhand" for d in
         ["Almora", "Bageshwar", "Chamoli", "Champawat", "Dehradun", "Haridwar",
          "Nainital", "Pauri Garhwal", "Pithoragarh"]}
REMAP.update({("Madhya Pradesh", d): "Chhattisgarh" for d in
              ["Bastar", "Bilaspur", "Dhamtari", "Durg", "Jashpur", "Kanker",
               "Korba", "Mahasamund", "Raipur", "Rajnandgaon", "Surguja"]})
REMAP.update({("Bihar", d): "Jharkhand" for d in ["Deoghar", "Dhanbad", "Garhwa"]})
pair = list(zip(df["state_name"], df["district_name"]))
df = df[[p not in BOGUS for p in pair]]
for (s_, d_), new in REMAP.items():
    m = (df["state_name"] == s_) & (df["district_name"] == d_)
    df.loc[m, "state_name"] = new
df["year_start"] = df["year"].str.split("-").str[0].astype(int)
df["district_key"] = df["state_name"].str.upper() + "|" + df["district_name"].str.upper()
df = keep_years(df)  # analysis window, scripts/_years.py
K = ["district_key", "year", "season", "crop_name"]
df = df.sort_values("area", ascending=False).drop_duplicates(subset=K, keep="first")

SLOT = {"Kharif": "K", "Autumn": "K", "Winter": "K", "Rabi": "R", "Summer": "S",
        "Whole Year": "W"}
df["slot"] = df["season"].map(SLOT)
assert df["slot"].notna().all()

# ------------------------------------------------------------------ helpers


def hill(a):
    a = np.asarray(a, float)
    a = a[a > 0]
    if a.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    p = a / a.sum()
    H = float(-(p * np.log(p)).sum())
    return float(a.size), float(np.exp(H)), float(1.0 / (p ** 2).sum()), H


# crop x slot area per district-year
cs = (df.groupby(["district_key", "year_start", "slot", "crop_name"])["area"].sum()
        .unstack("slot", fill_value=0.0))
for s in "KRSW":
    if s not in cs.columns:
        cs[s] = 0.0
cs = cs[list("KRSW")]

rec = []
for (dk, y), g in cs.groupby(level=[0, 1]):
    g = g.droplevel([0, 1])
    tot = g.values.sum()
    annual = g.sum(axis=1)
    D0a, D1a, D2a, Ha = hill(annual.values)
    r = {"district_key": dk, "year_start": y, "gca": tot,
         "D0_ann": D0a, "D1_ann": D1a, "D2_ann": D2a,
         "area_K": g["K"].sum(), "area_R": g["R"].sum(), "area_S": g["S"].sum(),
         "area_W": g["W"].sum()}
    # (1) season pools
    for mode, addW in [("", True), ("_noW", False)]:
        pools = {}
        for s in "KRS":
            if g[s].sum() > 0:
                pools[s] = g[s] + (g["W"] if addW else 0.0)
        wsum, acc = 0.0, {"D0": 0.0, "D1": 0.0, "D2": 0.0}
        for s in "KRS":
            if s in pools:
                d0, d1, d2, _ = hill(pools[s].values)
                a = pools[s].sum()
                wsum += a
                acc["D0"] += a * d0
                acc["D1"] += a * d1
                acc["D2"] += a * d2
            else:
                d0 = d1 = d2 = np.nan
            r["D0_" + s + mode], r["D1_" + s + mode], r["D2_" + s + mode] = d0, d1, d2
        for k_ in acc:
            r[k_ + "_ws" + mode] = acc[k_] / wsum if wsum > 0 else np.nan
    # (2) exact decomposition on the exclusive partition
    Halpha = 0.0
    for s in "KRSW":
        a = g[s].sum()
        if a > 0:
            Halpha += (a / tot) * hill(g[s].values)[3]
    r["H_ann"], r["H_alpha"] = Ha, Halpha
    r["H_beta"] = Ha - Halpha
    r["A_alpha"], r["B_beta"] = np.exp(Halpha), np.exp(Ha - Halpha)
    # (3) APY cropping intensity, upper bound
    r["ci_apy"] = tot / (g["W"].sum() + max(g["K"].sum(), g["R"].sum(), g["S"].sum()))
    rec.append(r)
DY = pd.DataFrame(rec)
DY["w_share"] = DY["area_W"] / DY["gca"]
DY["dry_share"] = (DY["area_R"] + DY["area_S"]) / DY["gca"]
assert (DY["B_beta"] >= 1 - 1e-9).all()
assert np.allclose(DY["A_alpha"] * DY["B_beta"], DY["D1_ann"])
ws_ok = DY["D1_ws"].notna()
w("# seasons log")
w("District-years: {:,}; with a season pool: {:,}".format(len(DY), int(ws_ok.sum())))

# check annual D1 reproduces script 76
p76 = pd.read_csv(DIV + "/district_year_diversity_panel_corrected.csv")
chk = DY.merge(p76[["district_key", "year_start", "D1_exp_shannon"]], on=["district_key", "year_start"])
w("Reproduces script 76 annual D1: max abs diff {:.2e} on {:,} rows".format(
    (chk["D1_ann"] - chk["D1_exp_shannon"]).abs().max(), len(chk)))

# ------------------------------------------------------------------ season reliability flags
cmap = pd.read_csv(CG + "/crop_group_child_map.csv")[["div_key", "unit_key"]]
fl = pd.read_csv(CG + "/crop_group_flags.csv")[["unit_key", "year_start", "flag_season_unreliable",
                                                 "flag_annual_unreliable", "f_tn_rice_single_season"]]
DY = DY.merge(cmap, left_on="district_key", right_on="div_key", how="left")
w("District-years with no 2011 unit: {}".format(int(DY["unit_key"].isna().sum())))
DY = DY.merge(fl, on=["unit_key", "year_start"], how="left")
for c in ["flag_season_unreliable", "flag_annual_unreliable", "f_tn_rice_single_season"]:
    DY[c] = DY[c].fillna(0).astype(int)
# district-level Whole Year share too: a child can be dominated even where the unit is not
DY["season_ok"] = (DY["flag_season_unreliable"] == 0) & (DY["w_share"] < 0.5) & ws_ok
w("Season-reliable district-years: {:,} of {:,}".format(int(DY["season_ok"].sum()), len(DY)))
# district Land Use Statistics (GCA, NSA), assembled for the irrigation panel task
LUSP = OUT + "/irrigation_panel_district_year.csv"
lus = pd.read_csv(LUSP)[["district_key", "year_start", "gca", "nsa"]].rename(
    columns={"gca": "lus_gca", "nsa": "lus_nsa"})
DY = DY.merge(lus, on=["district_key", "year_start"], how="left")
DY["ci_lus"] = DY["lus_gca"] / DY["lus_nsa"]
DY.loc[~DY["ci_lus"].between(0.9, 3.0), "ci_lus"] = np.nan
DY["apy_over_lus_gca"] = DY["gca"] / DY["lus_gca"]
w("LUS cropping intensity available for {:,} district-years (0.9-3.0 kept)".format(
    int(DY["ci_lus"].notna().sum())))
DY.to_csv(OUT + "/seasons_district_year.csv", index=False)

# ------------------------------------------------------------------ district means
V = ["D0_ann", "D1_ann", "D2_ann", "D0_K", "D1_K", "D2_K", "D0_R", "D1_R", "D2_R",
     "D1_S", "D0_ws", "D1_ws", "D2_ws", "D1_K_noW", "D1_R_noW", "D1_ws_noW",
     "A_alpha", "B_beta", "H_ann", "H_alpha", "H_beta", "ci_apy", "ci_lus", "apy_over_lus_gca", "w_share", "dry_share"]


def dmean(t):
    g = t.groupby("district_key")
    out = g[V].mean()
    out["n_years_season"] = g.size()
    out["beta_share"] = g["H_beta"].sum() / g["H_ann"].sum()
    out["gca_mean"] = g["gca"].mean()
    return out


DD = dmean(DY[DY["season_ok"]])
DD_all = dmean(DY[ws_ok])  # flags ignored, robustness
DD = DD[DD["n_years_season"] >= MIN_YEARS]
DD_all = DD_all[DD_all["n_years_season"] >= MIN_YEARS]

fp = pd.read_csv(COV + "/final_panel.csv")
fpk = fp[["div_key", "state_name", "district_name", "shrug_key", "irr_share", "in_final",
          "D1_exp_shannon", "D0_richness", "D2_inv_simpson", "n_years", "cropping_intensity",
          "flag_shared_parent", "secc_two_crop_acre_sum", "secc_unirr_land_acre_sum",
          "secc_other_irr_acre_sum", "mean_annual_cropped_area"]].copy()
fpk["secc_2crop"] = fpk["secc_two_crop_acre_sum"] / fpk[
    ["secc_two_crop_acre_sum", "secc_unirr_land_acre_sum", "secc_other_irr_acre_sum"]].sum(axis=1)
fpk.loc[fpk["cropping_intensity"].between(0.5, 3.5) == False, "cropping_intensity"] = np.nan
fpk = fpk.rename(columns={"cropping_intensity": "ci_ay"})

mj = json.load(open(REPO + "/docs/data/site_monoculture.json"))["districts"]
sysmap = pd.DataFrame([{"div_key": v["dk"], "sys": v["sys"]} for v in mj.values()])


def attach(dd):
    t = dd.reset_index().merge(fpk, left_on="district_key", right_on="div_key", how="left")
    t = t.merge(sysmap, on="div_key", how="left")
    return t


T = attach(DD)
T_all = attach(DD_all)
S = T[T["in_final"] == True].copy()
S_all = T_all[T_all["in_final"] == True].copy()
w("606 sample districts with >= {} season-reliable years: {}".format(MIN_YEARS, len(S)))
miss = fp.loc[fp["in_final"] == True, ["div_key", "state_name"]]
miss = miss[~miss["div_key"].isin(S["div_key"])]
w("Dropped from the 606 for want of season-reliable years: {} ({})".format(
    len(miss), miss["state_name"].value_counts().to_dict()))
T.to_csv(OUT + "/seasons_district.csv", index=False)

R = {}  # numbers for the results file

# ------------------------------------------------------------------ 1. descriptives


def mci(x):
    x = pd.Series(x).dropna()
    se = x.std() / np.sqrt(len(x))
    return x.mean(), se, len(x)


desc = []
for c in ["D0_ann", "D1_ann", "D2_ann", "D0_K", "D1_K", "D2_K", "D0_R", "D1_R", "D2_R",
          "D0_ws", "D1_ws", "D2_ws", "D1_ws_noW", "A_alpha", "B_beta", "beta_share", "ci_apy",
          "ci_ay", "ci_lus", "apy_over_lus_gca", "secc_2crop"]:
    m, se, n = mci(S[c])
    desc.append(dict(var=c, mean=m, se=se, n=n, median=S[c].median(),
                     p10=S[c].quantile(.1), p90=S[c].quantile(.9)))
DESC = pd.DataFrame(desc)
w(DESC.to_string())
gap = S["D1_ann"] - S["D1_ws"]
R["gap"] = mci(gap)
R["ratio"] = mci(S["D1_ann"] / S["D1_ws"])
# national, area-weighted
R["aw"] = {c: np.average(S[c], weights=S["gca_mean"]) for c in
           ["D1_ann", "D1_ws", "B_beta", "beta_share", "ci_apy"]}
w("gap annual - within-season D1: {}".format(R["gap"]))
w("area-weighted: {}".format(R["aw"]))

# ------------------------------------------------------------------ 2. by system
SYS = []
for f in ["rice-wheat", "rice-rice", "rice-only", "maize-wheat", "soybean-wheat",
          "rice-wheat-light", "cotton-only", "cotton-other", "millet-based", "whole"]:
    s = S[S["sys"] == f]
    if len(s) == 0:
        continue
    row = {"sys": f, "n": len(s)}
    for c in ["D1_ann", "D1_K", "D1_R", "D1_ws", "D1_ws_noW", "A_alpha", "B_beta", "beta_share", "ci_apy", "ci_ay"]:
        m, se, _ = mci(s[c])
        row[c] = m
        row[c + "_se"] = se
    SYS.append(row)
SYS = pd.DataFrame(SYS)
w(SYS.round(3).to_string())
rest = S[~S["sys"].isin(["rice-wheat", "rice-rice"])]
for f in ["rice-wheat", "rice-rice"]:
    a = S.loc[S["sys"] == f, "beta_share"]
    t_ = stats.ttest_ind(a, rest["beta_share"], equal_var=False)
    R["beta_diff_" + f] = (a.mean() - rest["beta_share"].mean(), t_.pvalue, len(a), len(rest))

# ------------------------------------------------------------------ 3. rankings
S["rk_ann"] = S["D1_ann"].rank(ascending=False)
S["rk_ws"] = S["D1_ws"].rank(ascending=False)
S["rk_move"] = S["rk_ws"] - S["rk_ann"]  # positive = falls down the ranking within season
R["spearman_ann_ws"] = stats.spearmanr(S["D1_ann"], S["D1_ws"])
R["spearman_ann_K"] = stats.spearmanr(S["D1_ann"], S["D1_K"])
R["spearman_ann_R"] = stats.spearmanr(S.dropna(subset=["D1_R"])["D1_ann"],
                                      S.dropna(subset=["D1_R"])["D1_R"])
R["abs_move_median"] = S["rk_move"].abs().median()
R["move_gt100"] = int((S["rk_move"].abs() > 100).sum())
mv = S[["district_name", "state_name", "sys", "D1_ann", "D1_ws", "D1_K", "D1_R", "B_beta",
        "beta_share", "ci_apy", "rk_ann", "rk_ws", "rk_move"]].sort_values("rk_move")
mv.to_csv(OUT + "/seasons_movers.csv", index=False)
w("Top risers (more diverse within season than annual rank says):")
w(mv.head(12).round(2).to_string())
w("Top fallers:")
w(mv.tail(12).round(2).to_string())
R["fall_sys"] = mv.tail(50)["sys"].value_counts().head(5).to_dict()
R["rise_sys"] = mv.head(50)["sys"].value_counts().head(5).to_dict()
R["fall_state"] = mv.tail(50)["state_name"].value_counts().head(5).to_dict()
R["rise_state"] = mv.head(50)["state_name"].value_counts().head(5).to_dict()

# ------------------------------------------------------------------ 4. cropping intensity
CI = []
for civ in ["ci_lus", "ci_apy", "ci_ay", "secc_2crop"]:
    for y in ["D1_ann", "D1_ws", "A_alpha", "B_beta", "beta_share"]:
        s = S.dropna(subset=[civ, y, "shrug_key"])
        rho = stats.spearmanr(s[civ], s[y])
        z = s[[y, civ]].copy()
        z = (z - z.mean()) / z.std()
        s = s.assign(zy=z[y].values, zx=z[civ].values)
        m = smf.ols("zy ~ zx + C(state_name)", data=s).fit(
            cov_type="cluster", cov_kwds={"groups": s["shrug_key"]})
        CI.append(dict(ci=civ, y=y, n=len(s), spearman=rho.correlation, sp_p=rho.pvalue,
                       beta_std_stateFE=m.params["zx"], se=m.bse["zx"], p=m.pvalues["zx"]))
CI = pd.DataFrame(CI)
w(CI.round(4).to_string())
R["ci_cross"] = {
    "lus_apy": stats.spearmanr(*S.dropna(subset=["ci_lus", "ci_apy"])[["ci_lus", "ci_apy"]].T.values),
    "lus_ay": stats.spearmanr(*S.dropna(subset=["ci_lus", "ci_ay"])[["ci_lus", "ci_ay"]].T.values),
    "lus_secc": stats.spearmanr(*S.dropna(subset=["ci_lus", "secc_2crop"])[["ci_lus", "secc_2crop"]].T.values),
    "apy_ay": stats.spearmanr(*S.dropna(subset=["ci_apy", "ci_ay"])[["ci_apy", "ci_ay"]].T.values),
    "apy_secc": stats.spearmanr(*S.dropna(subset=["ci_apy", "secc_2crop"])[["ci_apy", "secc_2crop"]].T.values),
    "ay_secc": stats.spearmanr(*S.dropna(subset=["ci_ay", "secc_2crop"])[["ci_ay", "secc_2crop"]].T.values)}
w("CI cross-validation: {}".format(R["ci_cross"]))
# does conditioning on CI absorb the annual-vs-within gap?
s = S.dropna(subset=["ci_lus"])
m = smf.ols("D1_ann ~ A_alpha + ci_lus + C(state_name)", data=s).fit(
    cov_type="cluster", cov_kwds={"groups": s["shrug_key"]})
R["ann_on_alpha_ci"] = (m.params["A_alpha"], m.bse["A_alpha"], m.params["ci_lus"], m.bse["ci_lus"],
                        m.pvalues["ci_lus"], int(m.nobs), m.rsquared)
# within district over time: does the between-season share track LUS intensity?
pz = DY[DY["season_ok"] & DY["ci_lus"].notna() & DY["district_key"].isin(S["div_key"])].copy()
pz = pz.merge(S[["div_key", "shrug_key"]], left_on="district_key", right_on="div_key")
pz["bshare_y"] = pz["H_beta"] / pz["H_ann"]
pz = pz[pz["H_ann"] > 0].copy()
for y in ["H_beta", "bshare_y", "H_alpha", "H_ann"]:
    m = smf.ols(y + " ~ ci_lus + C(district_key) + C(year_start)", data=pz).fit(
        cov_type="cluster", cov_kwds={"groups": pz["shrug_key"]})
    R["fe_" + y + "_on_ci_lus"] = (m.params["ci_lus"], m.bse["ci_lus"], m.pvalues["ci_lus"],
                                   int(m.nobs), int(pz["district_key"].nunique()))
    w("district+year FE, {} on LUS CI: {:.4f} (se {:.4f}, p {:.2g}, n {}, districts {})".format(
        y, *R["fe_" + y + "_on_ci_lus"]))

# ------------------------------------------------------------------ 5. irrigation inverted U


def ufit(df, y, x="irr_share", fe=True, wt=None):
    df = df.dropna(subset=[y, x, "shrug_key"]).copy()
    f = "{} ~ {} + I({}**2)".format(y, x, x) + (" + C(state_name)" if fe else "")
    kw = dict(cov_type="cluster", cov_kwds={"groups": df["shrug_key"]})
    m = (smf.wls(f, data=df, weights=df[wt]).fit(**kw) if wt else smf.ols(f, data=df).fit(**kw))
    b1n, b2n = x, "I({} ** 2)".format(x)
    b1, b2 = m.params[b1n], m.params[b2n]
    V_ = m.cov_params().loc[[b1n, b2n], [b1n, b2n]].values
    tp = -b1 / (2 * b2)
    g = np.array([-1 / (2 * b2), b1 / (2 * b2 ** 2)])
    tpse = float(np.sqrt(g @ V_ @ g))
    # Lind-Mehlum style check: slope at the observed min and max
    lo, hi = df[x].quantile(.02), df[x].quantile(.98)
    sl_lo = b1 + 2 * b2 * lo
    sl_hi = b1 + 2 * b2 * hi
    gl = np.array([1, 2 * lo]); gh = np.array([1, 2 * hi])
    return dict(y=y, n=int(m.nobs), b1=b1, se1=m.bse[b1n], b2=b2, se2=m.bse[b2n],
                p2=m.pvalues[b2n], tp=tp, tp_lo=tp - 1.96 * tpse, tp_hi=tp + 1.96 * tpse,
                slope_p2=sl_lo, slope_p2_p=2 * stats.norm.sf(abs(sl_lo / np.sqrt(gl @ V_ @ gl))),
                slope_p98=sl_hi, slope_p98_p=2 * stats.norm.sf(abs(sl_hi / np.sqrt(gh @ V_ @ gh))),
                r2=m.rsquared, ymean=df[y].mean())


REG = []
for lab, data in [("main (flags applied)", S), ("flags ignored", S_all),
                  ("drop shared parents", S[S["flag_shared_parent"] != True])]:
    for y in ["D1_exp_shannon", "D1_ann", "D1_ws", "D1_ws_noW", "D1_K", "D1_R",
              "D0_ws", "D2_ws", "A_alpha", "B_beta", "beta_share", "H_ann", "H_alpha", "H_beta"]:
        d_ = data
        if y == "D1_R":
            d_ = data[data["dry_share"] >= 0.05]
        r = ufit(d_, y)
        r["sample"] = lab
        REG.append(r)
REG = pd.DataFrame(REG)
# exact additive check: H_ann = H_alpha + H_beta, so the coefficients add
for lab in REG["sample"].unique():
    q = REG[REG["sample"] == lab].set_index("y")
    w("{}: b2 H_ann {:.4f} = H_alpha {:.4f} + H_beta {:.4f}".format(
        lab, q.loc["H_ann", "b2"], q.loc["H_alpha", "b2"], q.loc["H_beta", "b2"]))
# predicted fall from the turning point to the 98th percentile, as a share of the mean
for _, r in REG.iterrows():
    pass
REG.to_csv(OUT + "/seasons_regressions.csv", index=False)
w(REG[["sample", "y", "n", "b1", "b2", "p2", "tp", "tp_lo", "tp_hi", "slope_p2", "slope_p98",
       "slope_p98_p", "r2"]].round(4).to_string())

# irrigation vs CI, linear, state FE
s = S.dropna(subset=["ci_lus", "irr_share"])
m = smf.ols("ci_lus ~ irr_share + C(state_name)", data=s).fit(cov_type="cluster",
                                                              cov_kwds={"groups": s["shrug_key"]})
R["ci_on_irr"] = (m.params["irr_share"], m.bse["irr_share"], m.pvalues["irr_share"], int(m.nobs))

with open(OUT + "/seasons_log.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG))
DESC.to_csv(OUT + "/seasons_descriptives.csv", index=False)
SYS.to_csv(OUT + "/seasons_by_system.csv", index=False)
CI.to_csv(OUT + "/seasons_intensity.csv", index=False)
with open(OUT + "/seasons_numbers.json", "w") as fh:
    json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in R.items()
               if not hasattr(v, "correlation")}, fh, default=float, indent=1)
print("\nR:", R)

# ------------------------------------------------------------------ extras for the write-up
x98 = S["irr_share"].quantile(.98)
w("irr p98 = {:.3f}, median {:.3f}".format(x98, S["irr_share"].median()))
for y in ["D1_ann", "D1_ws", "D1_ws_noW", "D1_K", "D1_R", "H_ann", "H_alpha", "H_beta"]:
    q = REG[(REG["sample"] == "main (flags applied)") & (REG["y"] == y)].iloc[0]
    f = lambda v: q["b1"] * v + q["b2"] * v * v
    tp = min(max(q["tp"], 0), x98)
    w("{}: fall from turning point to p98 {:.3f} ({:.1%} of mean {:.3f})".format(
        y, f(x98) - f(tp), (f(x98) - f(tp)) / q["ymean"], q["ymean"]))
asm = S[S["state_name"] == "ASSAM"]
w("Assam rabi D1 with Whole Year {:.2f}, without {:.2f}; rabi share of GCA {:.3f}".format(
    asm["D1_R"].mean(), asm["D1_R_noW"].mean(), asm["dry_share"].mean()))
w("Share of districts with D1_ws > D1_ann: {:.3f}; A_alpha > D1_ws: {:.3f}".format(
    (S["D1_ws"] > S["D1_ann"]).mean(), (S["A_alpha"] > S["D1_ws"]).mean()))
w("Mean Whole Year share of GCA (sample): {:.3f}".format(S["w_share"].mean()))
st = S.groupby("state_name").agg(n=("D1_ann", "size"), D1_ann=("D1_ann", "mean"),
                                 D1_ws=("D1_ws", "mean"), A=("A_alpha", "mean"),
                                 B=("B_beta", "mean"), bshare=("beta_share", "mean"),
                                 ci=("ci_apy", "mean"))
st = st[st["n"] >= 8].sort_values("bshare")
w(st.round(3).to_string())
st.to_csv(OUT + "/seasons_by_state.csv")
with open(OUT + "/seasons_log.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG))

# ------------------------------------------------------------------ ranking on the exclusive partition
# D1_ws adds Whole Year crops to every season pool; where a season pool is small (Assam
# rabi) that inflates it (Assam rabi D1 8.3 with, 3.7 without). A_alpha = exp(sum w_s H_s)
# on the exclusive partition has no such problem, so the movers list is also written on it.
for v in ["D1_ws_noW", "A_alpha"]:
    S["rk_" + v] = S[v].rank(ascending=False)
    S["mv_" + v] = S["rk_" + v] - S["rk_ann"]
    w("{}: spearman with annual {:.3f}; median |rank move| {:.0f}; moves > 100: {}".format(
        v, stats.spearmanr(S["D1_ann"], S[v])[0], S["mv_" + v].abs().median(),
        int((S["mv_" + v].abs() > 100).sum())))
ma = S.sort_values("mv_A_alpha")[["district_name", "state_name", "sys", "D1_ann", "A_alpha",
                                  "B_beta", "beta_share", "ci_apy", "rk_ann", "rk_A_alpha",
                                  "mv_A_alpha"]]
ma.to_csv(OUT + "/seasons_movers_alpha.csv", index=False)
w("risers (A_alpha) states {} systems {}".format(ma.head(50)["state_name"].value_counts().head(4).to_dict(),
                                                 ma.head(50)["sys"].value_counts().head(3).to_dict()))
w("fallers (A_alpha) states {} systems {}".format(ma.tail(50)["state_name"].value_counts().head(4).to_dict(),
                                                  ma.tail(50)["sys"].value_counts().head(3).to_dict()))
for y in ["H_beta", "beta_share", "B_beta"]:
    m = smf.ols(y + " ~ irr_share + C(state_name)", data=S).fit(
        cov_type="cluster", cov_kwds={"groups": S["shrug_key"]})
    w("{} linear on irrigation, state FE: {:.4f} (se {:.4f}, p {:.2g}, n {})".format(
        y, m.params["irr_share"], m.bse["irr_share"], m.pvalues["irr_share"], int(m.nobs)))
with open(OUT + "/seasons_log.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG))
