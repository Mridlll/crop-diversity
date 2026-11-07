"""
107_irrigation_iv_recentred_cgwb.py

Follow-up to 106. Two things 106 left open.

  IV-1b Shift-share on irrigation source, RECENTRED. In 106 the just-identified
        first stage was negative (-0.21) and the instrument correlated 0.92 with
        baseline tubewell share. A sum-of-shares instrument moves with the
        district's baseline irrigation level times time, which inside district FE
        is a convergence trend. Following Borusyak, Hull and Jaravel (incomplete
        shares), we control for baseline total irrigation (1998-2000 NIA/NSA)
        interacted with year dummies, so only the SOURCE MIX at baseline
        identifies. Also the pure-composition version: source shares of baseline
        irrigated area (sum to one) times leave-state-out source growth.

  IV-3  CGWB depth to water table x trend. India Data Portal redistribution of
        CGWB well readings (outputs/revision_nov8/raw_cgwb/, 2013-2021 only).
        Stations are placed in 2011 districts by their coordinates (SHRUG pc11
        district polygons). Pre-monsoon (April-June) depth, 2013-2015 median of
        station means. Measured AFTER most of the panel, so it is partly an
        outcome of the pumping it is meant to predict: a first stage only, not a
        credible exclusion restriction.

All with district FE and state x year FE, errors clustered on shared 2011 parents.
Outputs: outputs/revision_nov8/irrigation_iv2_raw.md, irrigation_iv2_results.csv,
         irrigation_cgwb_district_depth.csv
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, r"D:/crop-diversity/scripts")
from _revision_fe import fe_iv, fe_ols, turning_point, pval  # noqa: E402

REPO = r"D:/crop-diversity"
OUT = REPO + "/outputs/revision_nov8"
SHR = r"D:/SHRUG_2.1_Data/extracted"
R = []
ROWS = []


def w(s=""):
    print(s)
    R.append(str(s))


P = pd.read_csv(OUT + "/irrigation_panel_district_year.csv")
P = P[P["year_start"].between(1998, 2019)].copy()
SRC = ["nia_canal", "nia_tank", "nia_tubewell", "nia_otherwell", "nia_other"]
for s in SRC:
    P[s] = np.where(P["nia"].notna(), P[s].fillna(0), np.nan)

# ------------------------------------------------- leave-state-out source growth (as 106)
P["src_ok"] = P["nia"] > 0
nyr = P.groupby("district_key")["src_ok"].transform("sum")
Bal = P[(nyr >= 20) & P["src_ok"]].copy()
tot = Bal.groupby(["state_name", "year_start"])[SRC].sum()
nat = tot.groupby("year_start").sum()
grow = {}
for st in P["state_name"].unique():
    o = nat - (tot.xs(st, level="state_name") if st in tot.index.get_level_values(0) else 0)
    o = o.reindex(nat.index)
    grow[st] = o / o.loc[1998:2000].mean()

b0 = P[P["year_start"].between(1998, 2000)].groupby("district_key")
base = b0.apply(lambda g: pd.Series(
    {**{"sh_" + s[4:]: g[s].sum() / g["nsa"][g[s].notna()].sum() if g[s].notna().any() else np.nan
        for s in SRC},
     "base_irr": g["nia"].sum() / g["nsa"][g["nia"].notna()].sum() if g["nia"].notna().any() else np.nan}))
P = P.join(base, on="district_key")
G = {s: np.array([grow[st].loc[y, s] if y in grow[st].index else np.nan
                  for st, y in zip(P["state_name"], P["year_start"])]) for s in SRC}
z = np.zeros(len(P))
zc = np.zeros(len(P))
ok = P["base_irr"].notna().to_numpy() & (P["base_irr"] > 0).to_numpy()
for s in SRC:
    sh = P["sh_" + s[4:]].to_numpy()
    z += np.nan_to_num(sh * G[s])
    zc += np.nan_to_num(sh / P["base_irr"].to_numpy() * G[s])  # composition: shares of irrigated area
    ok &= ~np.isnan(sh)
P["z_ss"] = np.where(ok, z, np.nan)
P["z_comp"] = np.where(ok, zc, np.nan)
P["z_ss2"] = P["z_ss"] ** 2
P["z_comp2"] = P["z_comp"] ** 2
P["tw_frac"] = P["sh_tubewell"] / P["base_irr"]

# ------------------------------------------------- CGWB depth to water, by 2011 district
cg = pd.read_csv(OUT + "/raw_cgwb/cgwb-changes-in-depth-to-water-level.csv",
                 usecols=["date", "station_name", "latitude", "longitude", "currentlevel"],
                 parse_dates=["date"])
cg = cg[cg["date"].dt.year.between(2013, 2015) & cg["date"].dt.month.between(4, 6)]
cg = cg.dropna(subset=["latitude", "longitude", "currentlevel"])
st_ = cg.groupby(["station_name", "latitude", "longitude"], as_index=False)["currentlevel"].mean()
poly = gpd.read_file(SHR + "/shrug-pc11dist-poly-gpkg/" +
                     [f for f in __import__("os").listdir(SHR + "/shrug-pc11dist-poly-gpkg")
                      if f.endswith(".gpkg")][0])
pts = gpd.GeoDataFrame(st_, geometry=gpd.points_from_xy(st_["longitude"], st_["latitude"]),
                       crs="EPSG:4326").to_crs(poly.crs)
j = gpd.sjoin(pts, poly[["pc11_district_id", "geometry"]], how="inner", predicate="within")
j["pc11_district_id"] = j["pc11_district_id"].astype(int)
dep = j.groupby("pc11_district_id").agg(depth=("currentlevel", "median"),
                                        n_wells=("currentlevel", "size")).reset_index()
w("# IV follow-up: recentred shift-share and CGWB depth")
w("")
w("CGWB pre-monsoon readings 2013-15: {:,} stations, {:,} placed in a 2011 district; "
  "{} districts with a depth".format(len(st_), len(j), len(dep)))
fp = pd.read_csv(REPO + "/outputs/shrug_covariates/final_panel.csv",
                 usecols=["div_key", "pc11_district_id"]).dropna()
fp["pc11_district_id"] = fp["pc11_district_id"].astype(int)
fp = fp.merge(dep, on="pc11_district_id", how="left")
fp[["div_key", "pc11_district_id", "depth", "n_wells"]].to_csv(
    OUT + "/irrigation_cgwb_district_depth.csv", index=False)
P = P.merge(fp[["div_key", "depth", "n_wells"]], left_on="district_key", right_on="div_key", how="left")
P.loc[P["n_wells"] < 3, "depth"] = np.nan
P["t"] = (P["year_start"] - 1998) / 10
dsd = P.drop_duplicates("district_key")["depth"]
P["ldep"] = np.log(P["depth"])
lsd = P.drop_duplicates("district_key")["ldep"]
P["z_dep_t"] = (P["ldep"] - lsd.mean()) / lsd.std() * P["t"]
P["z_dep_t2"] = P["z_dep_t"] ** 2
w("District pre-monsoon depth (m), analysis districts with >= 3 wells: median {:.1f}, p10 {:.1f}, p90 {:.1f}".format(
    dsd.median(), dsd.quantile(.1), dsd.quantile(.9)))

# ------------------------------------------------- estimation sample
E = P[P["in_final"] & P["irr_nsa"].notna() & (P["year_start"] >= 2001)].copy()
E["irr"] = E["irr_nsa"]
E["irr2"] = E["irr"] ** 2
E["sy"] = E["state_name"] + "_" + E["year_start"].astype(str)
YRS = sorted(E["year_start"].unique())[1:]
BX = []
for y in YRS:
    c = "bx_{}".format(y)
    E[c] = E["base_irr"] * (E["year_start"] == y)
    BX.append(c)
FES = ["district_key", "sy"]
NEST = ("district_key",)
OUTC = ["D1", "D0_05", "D0", "D2", "even", "cereal_share"]
w("")
w("Correlation, across districts, of baseline tubewell fraction of irrigated area with baseline irrigation: {:.3f}".format(
    E.drop_duplicates("district_key")[["tw_frac", "base_irr"]].corr().iloc[0, 1]))


def run(label, inst, exog, sample_cols, quad=True):
    d = E.dropna(subset=inst + exog + sample_cols)
    fs = fe_ols(d, "irr", inst + exog, FES, "shrug_key", nested=NEST)
    b = fs["params"][inst].to_numpy()
    V = fs["vcov"].loc[inst, inst].to_numpy()
    F = float(b @ np.linalg.inv(V) @ b) / len(inst)
    w("")
    w("## " + label)
    w("First stage (n={:,}, districts {}, clusters {}): ".format(fs["n"], fs["n_units"], fs["G"])
      + "; ".join("{} {:+.4f} ({:.4f})".format(i, fs["params"][i], fs["se"][i]) for i in inst)
      + "; cluster-robust F = {:.2f}".format(F))
    ROWS.append(dict(iv=label, stage="first", y="irr", n=fs["n"], G=fs["G"], F=F,
                     b=fs["params"][inst[0]], se=fs["se"][inst[0]]))
    for y in OUTC:
        rf = fe_ols(d, y, inst + exog, FES, "shrug_key", nested=NEST)
        ols = fe_ols(d, y, ["irr"] + exog, FES, "shrug_key", nested=NEST)
        r = fe_iv(d, y, ["irr"], exog, inst, FES, "shrug_key", nested=NEST)
        p = pval(r["params"]["irr"], r["se"]["irr"], r["G"])
        lo, hi = r["params"]["irr"] - 1.96 * r["se"]["irr"], r["params"]["irr"] + 1.96 * r["se"]["irr"]
        line = "- {}: reduced form {} {:+.3f} ({:.3f}); 2SLS {:+.3f} ({:.3f}, p={:.3f}, 95% CI {:+.2f} to {:+.2f}); OLS {:+.3f} ({:.3f}); n={:,}".format(
            y, inst[0], rf["params"][inst[0]], rf["se"][inst[0]], r["params"]["irr"], r["se"]["irr"], p,
            lo, hi, ols["params"]["irr"], ols["se"]["irr"], r["n"])
        row = dict(iv=label, stage="2sls linear", y=y, n=r["n"], G=r["G"], F=F,
                   b=r["params"]["irr"], se=r["se"]["irr"], p=p,
                   rf=rf["params"][inst[0]], rf_se=rf["se"][inst[0]],
                   ols=ols["params"]["irr"], ols_se=ols["se"]["irr"])
        if quad and len(inst) >= 2:
            rq = fe_iv(d, y, ["irr", "irr2"], exog, inst, FES, "shrug_key", nested=NEST)
            tp, tse = turning_point(rq, "irr", "irr2")
            line += "\n  quadratic: irr {:+.3f} ({:.3f}), irr2 {:+.3f} ({:.3f}), p(irr2)={:.3f}, peak {:.3f} ({:.3f}); first-stage F irr {:.2f}, irr2 {:.2f}".format(
                rq["params"]["irr"], rq["se"]["irr"], rq["params"]["irr2"], rq["se"]["irr2"],
                pval(rq["params"]["irr2"], rq["se"]["irr2"], rq["G"]), tp, tse,
                rq["first"][0]["F"], rq["first"][1]["F"])
            ROWS.append(dict(iv=label, stage="2sls quadratic", y=y, n=rq["n"], G=rq["G"],
                             b=rq["params"]["irr"], se=rq["se"]["irr"], b2=rq["params"]["irr2"],
                             se2=rq["se"]["irr2"], tp=tp, tp_se=tse,
                             F=rq["first"][0]["F"], F2=rq["first"][1]["F"]))
        w(line)
        ROWS.append(row)


run("shift-share, just identified, NO baseline control (106 replication)", ["z_ss"], [], [], quad=False)
run("shift-share, recentred: baseline irrigation x year controls", ["z_ss"], BX, [], quad=False)
run("shift-share, recentred, with square", ["z_ss", "z_ss2"], BX, [])
run("source composition (shares of irrigated area) x leave-state-out growth, baseline x year controls",
    ["z_comp"], BX, [], quad=False)
run("CGWB depth x trend (2013-15 depth; NOT a valid exclusion restriction)", ["z_dep_t"], [], [], quad=False)
run("CGWB depth x trend and its square", ["z_dep_t", "z_dep_t2"], [], [])
run("CGWB depth x trend, baseline irrigation x year controls", ["z_dep_t"], BX, [], quad=False)

pd.DataFrame(ROWS).to_csv(OUT + "/irrigation_iv2_results.csv", index=False)
open(OUT + "/irrigation_iv2_raw.md", "w", encoding="utf-8").write("\n".join(R))
