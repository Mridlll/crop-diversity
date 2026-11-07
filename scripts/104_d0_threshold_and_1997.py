"""
104_d0_threshold_and_1997.py

Two checks on the diversity trend.

(1) Is the rise in the count of crops (D0) real diversification, or crops entering
    the records at tiny areas? D0 is recomputed per district-year counting only
    crops at or above a minimum share of the district's gross cropped area
    (0.1, 0.5 and 1 percent) and compared with raw D0 and the effective number of
    crops (D1). Also: which crops drive raw D0's rise, whether crop names reported
    per state jump in particular years, and whether D0 rises within districts when
    D1 does not.

(2) Why 1997-98 sits far below every later year (D1 about 3.8 against 5.2+).
    Established from the raw APY rows.

Cleaning is exactly script 76's. Samples: the 725 crop-record districts (724 once 1997-98
is dropped; one district exists only in 1997-98), and the balanced panel of districts
present in all 23 years including 1997-98 (429). The site's national series now uses
the 500 districts present in all 22 years from 1998-99 (script 110). Standard errors cluster on the 2011 SHRUG parent district
(district_crosswalk.csv shrug_key; own key where unmatched); bootstrap resamples
those clusters.

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/104_d0_threshold_and_1997.py

Outputs (outputs/revision_nov8/):
  d0_results.md                 the write-up skeleton with every number
  d0_threshold_panel.csv        district-year D0 at each threshold, D1, area
  d0_threshold_series.csv       national series per year and threshold
  d0_crop_drivers.csv           per-crop presence change, balanced panel
  d0_state_crop_names.csv       crop names reported per state per year
  d0_state_jumps.csv            state-years where crop names jump
  y1997_year_profile.csv        rows, districts, states, crops, area by year
  y1997_crop_area.csv           national area by crop, 1997 against 1998-2000
  y1997_season_area.csv         area by season and year
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RAW = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
REPO = r"D:/crop-diversity"
DIV = REPO + "/outputs/crop_diversity_analysis"
COV = REPO + "/outputs/shrug_covariates"
OUT = REPO + "/outputs/revision_nov8"

import os
os.makedirs(OUT, exist_ok=True)

R = []


def w(s=""):
    print(s)
    R.append(str(s))


TH = [0.0, 0.001, 0.005, 0.01]
TN = {0.0: "raw", 0.001: "0.1%", 0.005: "0.5%", 0.01: "1%"}
EARLY = list(range(1998, 2005))
LATE = list(range(2013, 2020))
rng = np.random.default_rng(104)
NB = 1000

# ------------------------------------------------------------- load + clean (76)
raw = pd.read_csv(RAW)
for c in ["year", "season", "state_name", "district_name", "crop_name", "crop_type"]:
    raw[c] = raw[c].astype(str).str.strip()
raw = raw.dropna(subset=["area"])
raw = raw[raw["area"] > 0]
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
pair = list(zip(raw["state_name"], raw["district_name"]))
raw = raw[[p not in BOGUS for p in pair]].copy()
for (s_, d_), new in REMAP.items():
    m = (raw["state_name"] == s_) & (raw["district_name"] == d_)
    raw.loc[m, "state_name"] = new
raw["year_start"] = raw["year"].str.split("-").str[0].astype(int)
raw["district_key"] = raw["state_name"].str.upper() + "|" + raw["district_name"].str.upper()
raw_all_years = raw.copy()          # includes 2020-21 for the year profile
raw = raw[raw["year_start"] <= 2019]
K = ["district_key", "year", "season", "crop_name"]
raw = raw.sort_values("area", ascending=False).drop_duplicates(subset=K, keep="first")
df = raw

# ------------------------------------------------------------- samples
corr = pd.read_csv(DIV + "/district_diversity_indices_corrected.csv")
keys725 = set(corr["district_key"])
cw = pd.read_csv(COV + "/district_crosswalk.csv")[["div_key", "shrug_key"]]
clmap = dict(zip(cw["div_key"], cw["shrug_key"]))

dy = (df.groupby(["district_key", "year_start", "crop_name"])["area"].sum()
        .reset_index())
dy["gca"] = dy.groupby(["district_key", "year_start"])["area"].transform("sum")
dy["share"] = dy["area"] / dy["gca"]


def hill1(s):
    p = s.values / s.values.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


g = dy.groupby(["district_key", "year_start"])
pan = pd.DataFrame({"gca": g["area"].sum(), "D1": g["area"].apply(hill1)})
for t in TH:
    pan["D0_" + TN[t]] = (dy[dy["share"] >= t] if t > 0 else dy).groupby(
        ["district_key", "year_start"]).size()
pan = pan.fillna(0).reset_index()
pan = pan[pan["district_key"].isin(keys725)]
pan["cl"] = pan["district_key"].map(clmap).fillna(pan["district_key"])

# check against the corrected panel
ref = pd.read_csv(DIV + "/district_year_diversity_panel_corrected.csv")
chk = pan.merge(ref, on=["district_key", "year_start"])
assert (chk["D0_raw"] == chk["D0_richness"]).all(), "raw D0 does not reproduce"
assert np.allclose(chk["D1"], chk["D1_exp_shannon"]), "D1 does not reproduce"

YRS = sorted(pan["year_start"].unique())
cnt = pan.groupby("district_key")["year_start"].nunique()
bal = set(cnt[cnt == len(YRS)].index)
bp = pan[pan["district_key"].isin(bal)].copy()
pan.to_csv(OUT + "/d0_threshold_panel.csv", index=False)

MEAS = ["D0_raw", "D0_0.1%", "D0_0.5%", "D0_1%", "D1"]

w("# D0 noise and the 1997-98 break")
w("")
w("Sample: {} crop-record districts; balanced panel {} districts x {} years "
  "(1997-98 to 2019-20); {} cluster groups in the balanced panel.".format(
      pan["district_key"].nunique(), len(bal), len(YRS), bp["cl"].nunique()))
w("Raw D0 and D1 reproduce script 76's corrected panel exactly "
  "({:,} district-years checked).".format(len(chk)))
w("")

# ------------------------------------------------------------- national series
rows = []
for y, gg in bp.groupby("year_start"):
    r = {"year": y, "n": len(gg)}
    for m_ in MEAS:
        r[m_ + "_aw"] = np.average(gg[m_], weights=gg["gca"])
        r[m_ + "_mean"] = gg[m_].mean()
        r[m_ + "_median"] = gg[m_].median()
    rows.append(r)
ser = pd.DataFrame(rows)
ser.to_csv(OUT + "/d0_threshold_series.csv", index=False)

w("## 1. National series, balanced panel")
w("")
w("Area-weighted mean (the site's weighting), unweighted mean, and district median.")
w("")
w("| measure | aw 1998 | aw 2019 | aw change | mean 1998 | mean 2019 | median 1998 | median 2019 |")
w("|---|---|---|---|---|---|---|---|")
s98 = ser.set_index("year").loc[1998]
s19 = ser.set_index("year").loc[2019]
for m_ in MEAS:
    w("| {} | {:.2f} | {:.2f} | {:+.2f} | {:.2f} | {:.2f} | {:.1f} | {:.1f} |".format(
        m_, s98[m_ + "_aw"], s19[m_ + "_aw"], s19[m_ + "_aw"] - s98[m_ + "_aw"],
        s98[m_ + "_mean"], s19[m_ + "_mean"], s98[m_ + "_median"],
        s19[m_ + "_median"]))
w("")


# --- cluster bootstrap of changes
def stats_of(d):
    out = {}
    a = d[d["year_start"] == 1998]
    b = d[d["year_start"] == 2019]
    e = d[d["year_start"].isin(EARLY)].groupby("district_key")
    l_ = d[d["year_start"].isin(LATE)].groupby("district_key")
    for m_ in MEAS:
        out[m_ + "|aw98_19"] = (np.average(b[m_], weights=b["gca"])
                                - np.average(a[m_], weights=a["gca"]))
        dd = l_[m_].mean() - e[m_].mean()
        out[m_ + "|mean_win"] = dd.mean()
        out[m_ + "|median_win"] = dd.median()
        # area-weighted window change: weight by district mean area
        wa = d.groupby("district_key")["gca"].mean().reindex(dd.index)
        out[m_ + "|aw_win"] = np.average(dd, weights=wa)
    return out


point = stats_of(bp)
cls = bp["cl"].unique()
byc = {c: x for c, x in bp.groupby("cl")}
boots = []
for b_ in range(NB):
    pick = rng.choice(cls, size=len(cls), replace=True)
    parts = []
    for i, c in enumerate(pick):
        x = byc[c].copy()
        x["district_key"] = x["district_key"] + "#" + str(i)
        parts.append(x)
    boots.append(stats_of(pd.concat(parts)))
bt = pd.DataFrame(boots)


def fmt(key):
    v = point[key]
    se = bt[key].std()
    lo, hi = bt[key].quantile([0.025, 0.975])
    return v, se, lo, hi


w("### Changes with cluster-bootstrap SEs and 95% CIs ({} draws, {} clusters)".format(
    NB, len(cls)))
w("")
w("aw98_19: area-weighted, 1998-99 vs 2019-20 single years (site's 21.2 -> 24.9).")
w("mean_win / median_win / aw_win: district change between 1998-2004 and 2013-2019 "
  "seven-year means; unweighted mean, median, and weighted by district mean area.")
w("")
w("| measure | statistic | change | SE | 95% CI | share of raw D0 change |")
w("|---|---|---|---|---|---|")
for st_ in ["aw98_19", "aw_win", "mean_win", "median_win"]:
    base = point["D0_raw|" + st_]
    for m_ in MEAS:
        v, se, lo, hi = fmt(m_ + "|" + st_)
        sh = "{:.0%}".format(v / base) if m_.startswith("D0") and abs(base) > 1e-9 else ""
        w("| {} | {} | {:+.2f} | {:.2f} | [{:+.2f}, {:+.2f}] | {} |".format(
            m_, st_, v, se, lo, hi, sh))
w("")

# --- FE trend regressions
w("### Within-district linear trend, per decade (district FE, 1998-2019)")
w("")
w("OLS of the measure on year with district fixed effects, clustered on 2011 parent. "
  "Weighted version uses district mean cropped area.")
w("")
w("| measure | sample | weighting | slope per decade | SE | p | n district-years |")
w("|---|---|---|---|---|---|---|")
trend = {}
for samp, dsub in [("balanced 429", bp), ("all 725", pan)]:
    d = dsub[dsub["year_start"] >= 1998].copy()
    d["yr10"] = (d["year_start"] - 2008) / 10
    d["wa"] = d.groupby("district_key")["gca"].transform("mean")
    for m_ in MEAS:
        for wt in ["unweighted", "area"]:
            dm = d.copy()
            for c in [m_, "yr10"]:
                if wt == "area":
                    dm[c + "_dm"] = dm[c] - dm.groupby("district_key")[c].transform("mean")
                else:
                    dm[c + "_dm"] = dm[c] - dm.groupby("district_key")[c].transform("mean")
            yv = "Q('" + m_ + "_dm')"
            if wt == "area":
                f = smf.wls(yv + " ~ yr10_dm - 1", data=dm, weights=dm["wa"])
            else:
                f = smf.ols(yv + " ~ yr10_dm - 1", data=dm)
            r = f.fit(cov_type="cluster", cov_kwds={"groups": dm["cl"]})
            trend[(samp, m_, wt)] = r
            w("| {} | {} | {} | {:+.3f} | {:.3f} | {:.4f} | {:,} |".format(
                m_, samp, wt, r.params.iloc[0], r.bse.iloc[0], r.pvalues.iloc[0],
                int(r.nobs)))
w("")

# --- is the rise a few steps or gradual
w("### Year-on-year steps in raw D0 (balanced panel, unweighted mean)")
w("")
st = ser.set_index("year")
yoy = st["D0_raw_mean"].diff().loc[1999:]
tot = st.loc[2019, "D0_raw_mean"] - st.loc[1998, "D0_raw_mean"]
top = yoy.sort_values(ascending=False).head(4)
w("Total 1998->2019 change in unweighted mean raw D0: {:+.2f}. Largest single-year "
  "steps: {}.".format(tot, "; ".join("{}-{} {:+.2f}".format(y - 1, str(y)[-2:], v)
                                     for y, v in top.items())))
yoy_aw = st["D0_raw_aw"].diff().loc[1999:]
top_aw = yoy_aw.sort_values(ascending=False).head(4)
w("Area-weighted: total {:+.2f}; largest steps: {}.".format(
    st.loc[2019, "D0_raw_aw"] - st.loc[1998, "D0_raw_aw"],
    "; ".join("{}-{} {:+.2f}".format(y - 1, str(y)[-2:], v) for y, v in top_aw.items())))
w("")
w("Full series (balanced, area-weighted):")
w("")
w("| year | D0 raw | D0 0.1% | D0 0.5% | D0 1% | D1 | D0 raw mean | D0 0.5% mean |")
w("|---|---|---|---|---|---|---|---|")
for y, r in st.iterrows():
    w("| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
        y, r["D0_raw_aw"], r["D0_0.1%_aw"], r["D0_0.5%_aw"], r["D0_1%_aw"],
        r["D1_aw"], r["D0_raw_mean"], r["D0_0.5%_mean"]))
w("")

# ------------------------------------------------------------- 2. by share bin
w("## 2. Where the raw D0 change sits, by the crop's share of district area")
w("")
w("Mean number of crops per district-year in each share bin, early (1998-2004) and "
  "late (2013-2019) windows, balanced panel. Bins sum to raw D0.")
w("")
bdy = dy[dy["district_key"].isin(bal)].copy()
BINS = [0, 0.001, 0.005, 0.01, 0.05, 1.01]
BL = ["<0.1%", "0.1-0.5%", "0.5-1%", "1-5%", ">=5%"]
bdy["bin"] = pd.cut(bdy["share"], BINS, labels=BL, right=False)
cb = (bdy.groupby(["district_key", "year_start", "bin"], observed=False).size()
         .rename("n").reset_index())
cb = cb.merge(bp[["district_key", "year_start", "gca", "cl"]],
              on=["district_key", "year_start"])
w("| share bin | early mean | late mean | change (unweighted) | SE | early aw | late aw | change (aw) |")
w("|---|---|---|---|---|---|---|---|")
binrows = []
for bl in BL:
    x = cb[cb["bin"] == bl]
    e = x[x["year_start"].isin(EARLY)].groupby("district_key").agg(n=("n", "mean"), a=("gca", "mean"), cl=("cl", "first"))
    l_ = x[x["year_start"].isin(LATE)].groupby("district_key").agg(n=("n", "mean"))
    dd = (l_["n"] - e["n"]).rename("d").to_frame().join(e[["a", "cl"]])
    r = smf.ols("d ~ 1", data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["cl"]})
    eaw = np.average(e["n"], weights=e["a"])
    law = np.average(l_["n"].reindex(e.index), weights=e["a"])
    w("| {} | {:.2f} | {:.2f} | {:+.2f} | {:.2f} | {:.2f} | {:.2f} | {:+.2f} |".format(
        bl, e["n"].mean(), l_["n"].mean(), r.params.iloc[0], r.bse.iloc[0],
        eaw, law, law - eaw))
    binrows.append((bl, r.params.iloc[0], law - eaw))
w("")

# ------------------------------------------------------------- 3. crop drivers
w("## 3. Which crops drive the raw D0 change")
w("")
w("Per crop, the share of balanced-panel district-years in which it is recorded at "
  "any area, early vs late window. The change in mean raw D0 equals the sum of these "
  "changes. Also shown: median share of district area when present (late), and the "
  "share of late presences below 0.5% of district area.")
w("")
early = bdy[bdy["year_start"].isin(EARLY)]
late = bdy[bdy["year_start"].isin(LATE)]
nE = len(bal) * len(EARLY)
nL = len(bal) * len(LATE)
# area-weighted presence: weight each district-year by its gca
gE = bp[bp["year_start"].isin(EARLY)].set_index(["district_key", "year_start"])["gca"]
gL = bp[bp["year_start"].isin(LATE)].set_index(["district_key", "year_start"])["gca"]
cr = pd.DataFrame({
    "pres_early": early.groupby("crop_name").size() / nE,
    "pres_late": late.groupby("crop_name").size() / nL,
    "aw_pres_early": early.set_index(["district_key", "year_start"]).assign(
        g=gE).groupby("crop_name")["g"].sum() / gE.sum(),
    "aw_pres_late": late.set_index(["district_key", "year_start"]).assign(
        g=gL).groupby("crop_name")["g"].sum() / gL.sum(),
    "median_share_late": late.groupby("crop_name")["share"].median(),
    "frac_late_below_0.5%": late.groupby("crop_name")["share"].apply(lambda s: (s < 0.005).mean()),
    "area_early_kha": early.groupby("crop_name")["area"].sum() / len(EARLY) / 1e3,
    "area_late_kha": late.groupby("crop_name")["area"].sum() / len(LATE) / 1e3,
}).fillna(0)
cr["d_pres"] = cr["pres_late"] - cr["pres_early"]
cr["d_aw_pres"] = cr["aw_pres_late"] - cr["aw_pres_early"]
cr = cr.sort_values("d_aw_pres", ascending=False)
cr.to_csv(OUT + "/d0_crop_drivers.csv")
w("Sum of presence changes: unweighted {:+.2f} crops, area-weighted {:+.2f}.".format(
    cr["d_pres"].sum(), cr["d_aw_pres"].sum()))
w("Gross gains (crops with rising presence) {:+.2f} aw; gross losses {:+.2f} aw.".format(
    cr.loc[cr["d_aw_pres"] > 0, "d_aw_pres"].sum(), cr.loc[cr["d_aw_pres"] < 0, "d_aw_pres"].sum()))
w("")
w("| crop | presence early | late | change | aw change | median share late | late presences <0.5% | area early kha | area late kha |")
w("|---|---|---|---|---|---|---|---|---|")
for c, r in pd.concat([cr.head(15), cr.tail(8)]).iterrows():
    w("| {} | {:.2f} | {:.2f} | {:+.3f} | {:+.3f} | {:.3%} | {:.0%} | {:,.0f} | {:,.0f} |".format(
        c, r["pres_early"], r["pres_late"], r["d_pres"], r["d_aw_pres"],
        r["median_share_late"], r["frac_late_below_0.5%"], r["area_early_kha"],
        r["area_late_kha"]))
w("")
# crops absent nationally from early window
first_year = df.groupby("crop_name")["year_start"].min().sort_values()
newc = first_year[first_year > 1998]
w("Crop names first appearing anywhere in the records after 1998-99: {}.".format(
    "; ".join("{} ({})".format(c, y) for c, y in newc.items()) if len(newc) else "none"))
lastc = df.groupby("crop_name")["year_start"].max()
gone = lastc[lastc < 2019]
w("Crop names last appearing before 2019-20: {}.".format(
    "; ".join("{} ({})".format(c, y) for c, y in gone.items()) if len(gone) else "none"))
w("")
nat = df.groupby("year_start")["crop_name"].nunique()
w("Distinct crop names nationally per year: " + ", ".join(
    "{} {}".format(y, n) for y, n in nat.items()) + ".")
w("")

# ------------------------------------------------------------- 4. state coverage
w("## 4. Reporting coverage: crop names per state per year")
w("")
sdf = df[df["district_key"].isin(keys725)]
sc = sdf.groupby(["state_name", "year_start"]).agg(
    crop_names=("crop_name", "nunique"), districts=("district_key", "nunique"),
    rows=("area", "size")).reset_index()
# per-district mean D0 in the state-year
sdd = pan.assign(state=pan["district_key"].str.split("|").str[0])
sc["mean_D0_district"] = sc.apply(lambda r: np.nan, axis=1)
sc.to_csv(OUT + "/d0_state_crop_names.csv", index=False)
sc = sc.sort_values(["state_name", "year_start"])
sc["d_names"] = sc.groupby("state_name")["crop_names"].diff()
sc["prev_year"] = sc.groupby("state_name")["year_start"].shift()
sc["d_dist"] = sc.groupby("state_name")["districts"].diff()
# crops added to a state's list in a jump, with their share in that state-year
jumps = sc[(sc["d_names"] >= 4) & (sc["year_start"] >= 1999)].copy()
st_crop = sdf.groupby(["state_name", "year_start", "crop_name"])["area"].sum().reset_index()
st_crop["st_area"] = st_crop.groupby(["state_name", "year_start"])["area"].transform("sum")
added = []
for _, r in jumps.iterrows():
    a = set(st_crop[(st_crop["state_name"] == r["state_name"]) & (st_crop["year_start"] == r["year_start"])]["crop_name"])
    b = set(st_crop[(st_crop["state_name"] == r["state_name"]) & (st_crop["year_start"] == r["prev_year"])]["crop_name"])
    new = sorted(a - b)
    sub = st_crop[(st_crop["state_name"] == r["state_name"]) & (st_crop["year_start"] == r["year_start"]) & st_crop["crop_name"].isin(new)]
    persist = sc[(sc["state_name"] == r["state_name"]) & (sc["year_start"] > r["year_start"])]
    later = st_crop[(st_crop["state_name"] == r["state_name"]) & (st_crop["year_start"] > r["year_start"]) & st_crop["crop_name"].isin(new)]
    yrs_after = len(persist)
    pers = later.groupby("crop_name")["year_start"].nunique().reindex(new).fillna(0)
    added.append({"state": r["state_name"], "year": int(r["year_start"]),
                  "prev_names": int(r["crop_names"] - r["d_names"]),
                  "names": int(r["crop_names"]), "jump": int(r["d_names"]),
                  "district_change": int(r["d_dist"]) if pd.notna(r["d_dist"]) else None,
                  "new_crops": "; ".join(new),
                  "new_crops_share_of_state_area": sub["area"].sum() / sub["st_area"].iloc[0] if len(sub) else 0,
                  "mean_years_new_crops_persist": pers.mean() if len(pers) else 0,
                  "years_after": yrs_after})
aj = pd.DataFrame(added)
aj.to_csv(OUT + "/d0_state_jumps.csv", index=False)
w("State-years where the number of distinct crop names rises by 4 or more over the "
  "previous reported year (1999 onward): {} state-years in {} states.".format(
      len(aj), aj["state"].nunique() if len(aj) else 0))
w("")
w("| state | year | names before -> after | district change | new crop names | their share of state area | mean later years they persist / years after |")
w("|---|---|---|---|---|---|---|")
for _, r in aj.sort_values("jump", ascending=False).iterrows():
    w("| {} | {} | {} -> {} | {} | {} | {:.2%} | {:.1f} / {} |".format(
        r["state"], r["year"], r["prev_names"], r["names"], r["district_change"],
        r["new_crops"], r["new_crops_share_of_state_area"],
        r["mean_years_new_crops_persist"], r["years_after"]))
w("")
# how much of balanced raw D0 rise sits in districts of jump state-years
# decomposition: year-on-year D0 change in balanced districts in jump state-years
bp["state"] = bp["district_key"].str.split("|").str[0]
bp = bp.sort_values(["district_key", "year_start"])
bp["dD0"] = bp.groupby("district_key")["D0_raw"].diff()
bp["dD0_05"] = bp.groupby("district_key")["D0_0.5%"].diff()
bp["dD1"] = bp.groupby("district_key")["D1"].diff()
js = set(zip(aj["state"].str.upper(), aj["year"])) if len(aj) else set()
bp["jump_sy"] = [(s, y) in js for s, y in zip(bp["state"], bp["year_start"])]
x = bp[bp["year_start"] >= 1999]
tot_d = x["dD0"].sum()
w("Balanced panel, sum of district year-on-year raw D0 changes 1998->2019: {:+.0f} "
  "crop-district units; of which in jump state-years: {:+.0f} ({:.0%}), which are "
  "{:.1%} of district-years. Same for D0 at 0.5%: {:+.0f} total, {:+.0f} in jump "
  "state-years. D1: {:+.1f} total, {:+.1f} in jump state-years.".format(
      tot_d, x.loc[x["jump_sy"], "dD0"].sum(),
      x.loc[x["jump_sy"], "dD0"].sum() / tot_d if tot_d else np.nan,
      x["jump_sy"].mean(), x["dD0_05"].sum(), x.loc[x["jump_sy"], "dD0_05"].sum(),
      x["dD1"].sum(), x.loc[x["jump_sy"], "dD1"].sum()))
w("")

# ------------------------------------------------------------- 5. within district
w("## 5. Within district: does D0 rise when D1 does not?")
w("")
e = bp[bp["year_start"].isin(EARLY)].groupby("district_key")[MEAS].mean()
l_ = bp[bp["year_start"].isin(LATE)].groupby("district_key")[MEAS].mean()
dd = (l_ - e)
dd["cl"] = bp.groupby("district_key")["cl"].first()
dd["a"] = bp.groupby("district_key")["gca"].mean()
n = len(dd)
for m_ in ["D0_raw", "D0_0.5%"]:
    up0 = dd[m_] > 0
    w("{}: rose in {} of {} districts; rose while D1 did not (D1 change <= 0) in {} "
      "({:.0%}); rose while D1 rose in {}; fell while D1 rose in {}.".format(
          m_, int(up0.sum()), n, int((up0 & (dd["D1"] <= 0)).sum()),
          (up0 & (dd["D1"] <= 0)).mean(), int((up0 & (dd["D1"] > 0)).sum()),
          int((~up0 & (dd["D1"] > 0)).sum())))
w("")
# flat-D1 districts
flat = dd[dd["D1"].abs() < 0.25]
r = smf.ols("Q('D0_raw') ~ 1", data=flat).fit(cov_type="cluster", cov_kwds={"groups": flat["cl"]})
r5 = smf.ols("Q('D0_0.5%') ~ 1", data=flat).fit(cov_type="cluster", cov_kwds={"groups": flat["cl"]})
w("Districts whose D1 moved by less than 0.25 between windows (n = {}): mean raw D0 "
  "change {:+.2f} (SE {:.2f}, p {:.3f}); D0 at 0.5% {:+.2f} (SE {:.2f}, p {:.3f}).".format(
      len(flat), r.params.iloc[0], r.bse.iloc[0], r.pvalues.iloc[0],
      r5.params.iloc[0], r5.bse.iloc[0], r5.pvalues.iloc[0]))
# regression of dD0 on dD1
for m_ in ["D0_raw", "D0_0.5%"]:
    rr = smf.ols("Q('{}') ~ D1".format(m_), data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["cl"]})
    w("Window change {} on window change D1: slope {:.2f} (SE {:.2f}), intercept {:+.2f} "
      "(SE {:.2f}, p {:.4f}) = the D0 change at zero D1 change; R2 {:.2f}; n {}.".format(
          m_, rr.params["D1"], rr.bse["D1"], rr.params["Intercept"], rr.bse["Intercept"],
          rr.pvalues["Intercept"], rr.rsquared, int(rr.nobs)))
w("Correlation of window changes: raw D0 with D1 {:.2f}; D0 0.5% with D1 {:.2f}; "
  "raw D0 with D0 0.5% {:.2f}.".format(
      dd["D0_raw"].corr(dd["D1"]), dd["D0_0.5%"].corr(dd["D1"]),
      dd["D0_raw"].corr(dd["D0_0.5%"])))
w("")
# per-district trend slopes
sl = []
for k, gg in bp[bp["year_start"] >= 1998].groupby("district_key"):
    t = gg["year_start"].values - 2008
    o = {"district_key": k}
    for m_ in ["D0_raw", "D0_0.5%", "D1"]:
        X = np.c_[np.ones(len(t)), t]
        yv = gg[m_].values
        beta, res, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        s2 = (resid ** 2).sum() / (len(t) - 2)
        se = np.sqrt(s2 / ((t - t.mean()) ** 2).sum())
        o[m_ + "_b"] = beta[1] * 10
        o[m_ + "_t"] = beta[1] / se if se > 0 else np.nan
    sl.append(o)
sl = pd.DataFrame(sl)
a0 = (sl["D0_raw_t"] > 1.96)
a1 = (sl["D1_t"] > 1.96)
a05 = (sl["D0_0.5%_t"] > 1.96)
w("Per-district OLS trends 1998-2019 (|t| > 1.96, naive iid SEs, so read as descriptive): "
  "raw D0 significantly up in {} of {}; of those, D1 not significantly up in {} and "
  "significantly down in {}. D0 at 0.5% significantly up in {}; of those D1 not up in {}. "
  "D1 significantly up in {}, down in {}.".format(
      int(a0.sum()), len(sl), int((a0 & ~a1).sum()), int((a0 & (sl["D1_t"] < -1.96)).sum()),
      int(a05.sum()), int((a05 & ~a1).sum()), int(a1.sum()), int((sl["D1_t"] < -1.96).sum())))
w("")
# two-way FE: D0 on D1
tw = bp[bp["year_start"] >= 1998].copy()
for m_ in ["D0_raw", "D0_0.5%"]:
    rr = smf.ols("Q('{}') ~ D1 + C(district_key) + C(year_start)".format(m_), data=tw).fit(
        cov_type="cluster", cov_kwds={"groups": tw["cl"]})
    w("Two-way FE (district + year), {} on D1: {:.2f} (SE {:.2f}), n {}.".format(
        m_, rr.params["D1"], rr.bse["D1"], int(rr.nobs)))
w("")

# ============================================================== PART 2: 1997-98
w("# Part 2. The 1997-98 break")
w("")
ra = raw_all_years.sort_values("area", ascending=False).drop_duplicates(subset=K, keep="first")
prof = ra.groupby("year_start").agg(
    rows=("area", "size"), states=("state_name", "nunique"),
    districts=("district_key", "nunique"), crop_names=("crop_name", "nunique"),
    seasons=("season", "nunique"), area_mha=("area", lambda s: s.sum() / 1e6))
cpd = ra.groupby(["year_start", "district_key"])["crop_name"].nunique().groupby("year_start").mean()
prof["crops_per_district"] = cpd
rpd = ra.groupby(["year_start", "district_key"]).size().groupby("year_start").mean()
prof["rows_per_district"] = rpd
prof["area_per_district_kha"] = prof["area_mha"] * 1e3 / prof["districts"]
prof.to_csv(OUT + "/y1997_year_profile.csv")
w("## Year profile, raw APY after cleaning (all districts)")
w("")
w("| year | rows | states | districts | crop names | seasons | area Mha | crops/district | rows/district | area/district kha |")
w("|---|---|---|---|---|---|---|---|---|---|")
for y, r in prof.iterrows():
    w("| {} | {:,} | {} | {} | {} | {} | {:.1f} | {:.1f} | {:.1f} | {:.0f} |".format(
        y, int(r["rows"]), int(r["states"]), int(r["districts"]), int(r["crop_names"]),
        int(r["seasons"]), r["area_mha"], r["crops_per_district"], r["rows_per_district"],
        r["area_per_district_kha"]))
w("")
s97 = set(ra[ra["year_start"] == 1997]["state_name"])
s98 = set(ra[ra["year_start"] == 1998]["state_name"])
a98 = ra[ra["year_start"] == 1998].groupby("state_name")["area"].sum()
w("States in 1998-99 absent in 1997-98: {}.".format(
    "; ".join("{} ({:.2f} Mha in 1998)".format(s, a98[s] / 1e6) for s in sorted(s98 - s97)) or "none"))
w("States in 1997-98 absent in 1998-99: {}.".format("; ".join(sorted(s97 - s98)) or "none"))
w("")
# state area 1997 vs 1998
sa = ra[ra["year_start"].isin([1997, 1998, 1999])].groupby(["state_name", "year_start"])["area"].sum().unstack()
sa = sa / 1e6
sa["ratio_97_98"] = sa[1997] / sa[1998]
sd = ra[ra["year_start"].isin([1997, 1998])].groupby(["state_name", "year_start"])["district_key"].nunique().unstack()
sa["dist97"] = sd[1997]
sa["dist98"] = sd[1998]
scn = ra[ra["year_start"].isin([1997, 1998])].groupby(["state_name", "year_start"])["crop_name"].nunique().unstack()
sa["crops97"] = scn[1997]
sa["crops98"] = scn[1998]
sa = sa.sort_values(1998, ascending=False)
w("### By state: area (Mha), districts and crop names, 1997-98 against 1998-99")
w("")
w("| state | area 97 | area 98 | area 99 | ratio 97/98 | districts 97/98 | crop names 97/98 |")
w("|---|---|---|---|---|---|---|")
for s, r in sa.iterrows():
    w("| {} | {} | {:.2f} | {} | {} | {}/{} | {}/{} |".format(
        s, "{:.2f}".format(r[1997]) if pd.notna(r[1997]) else "-", r[1998] if pd.notna(r[1998]) else 0,
        "{:.2f}".format(r[1999]) if pd.notna(r[1999]) else "-",
        "{:.2f}".format(r["ratio_97_98"]) if pd.notna(r["ratio_97_98"]) else "-",
        int(r["dist97"]) if pd.notna(r["dist97"]) else 0,
        int(r["dist98"]) if pd.notna(r["dist98"]) else 0,
        int(r["crops97"]) if pd.notna(r["crops97"]) else 0,
        int(r["crops98"]) if pd.notna(r["crops98"]) else 0))
w("")
# season
se_ = ra[ra["year_start"].isin([1997, 1998, 1999, 2000, 2010, 2019])].groupby(["year_start", "season"])["area"].sum().unstack() / 1e6
se_.to_csv(OUT + "/y1997_season_area.csv")
w("### Area by season (Mha)")
w("")
w("| year | " + " | ".join(se_.columns) + " |")
w("|---|" + "---|" * len(se_.columns))
for y, r in se_.iterrows():
    w("| {} | ".format(y) + " | ".join("{:.1f}".format(v) if pd.notna(v) else "-" for v in r.values) + " |")
w("")
# crops
ca = ra[ra["year_start"].isin([1997, 1998, 1999, 2000])].groupby(["crop_name", "year_start"])["area"].sum().unstack() / 1e3
cd = ra[ra["year_start"].isin([1997, 1998])].groupby(["crop_name", "year_start"])["district_key"].nunique().unstack()
ca["dist97"] = cd[1997]
ca["dist98"] = cd[1998]
ca["ratio_97_98"] = ca[1997] / ca[1998]
ca = ca.sort_values(1998, ascending=False)
ca.to_csv(OUT + "/y1997_crop_area.csv")
w("### National area by crop (thousand ha), and districts reporting it")
w("")
w("| crop | 1997 | 1998 | 1999 | 2000 | ratio 97/98 | districts 97 | districts 98 |")
w("|---|---|---|---|---|---|---|---|")
for c, r in ca.head(30).iterrows():
    w("| {} | {} | {:,.0f} | {:,.0f} | {:,.0f} | {} | {} | {} |".format(
        c, "{:,.0f}".format(r[1997]) if pd.notna(r[1997]) else "-",
        r[1998] if pd.notna(r[1998]) else 0, r[1999] if pd.notna(r[1999]) else 0,
        r[2000] if pd.notna(r[2000]) else 0,
        "{:.2f}".format(r["ratio_97_98"]) if pd.notna(r["ratio_97_98"]) else "-",
        int(r["dist97"]) if pd.notna(r["dist97"]) else 0,
        int(r["dist98"]) if pd.notna(r["dist98"]) else 0))
absent = ca[ca[1997].isna() & ca[1998].notna()]
w("")
w("Crops reported in 1998-99 but absent everywhere in 1997-98: {}.".format(
    "; ".join("{} ({:,.0f} kha in 1998)".format(c, r[1998]) for c, r in absent.iterrows()) or "none"))
w("")
# same-district comparison
p97 = pan[pan["year_start"] == 1997].set_index("district_key")
p98 = pan[pan["year_start"] == 1998].set_index("district_key")
both = p97.index.intersection(p98.index)
cmp_ = pd.DataFrame({"D0_97": p97.loc[both, "D0_raw"], "D0_98": p98.loc[both, "D0_raw"],
                     "D1_97": p97.loc[both, "D1"], "D1_98": p98.loc[both, "D1"],
                     "gca97": p97.loc[both, "gca"], "gca98": p98.loc[both, "gca"],
                     "cl": p98.loc[both, "cl"]})
cmp_["gr"] = cmp_["gca97"] / cmp_["gca98"]
w("### Same districts, 1997-98 against 1998-99 (n = {}, of which {} in balanced panel)".format(
    len(both), len(set(both) & bal)))
w("")
for a_, b_ in [("D0_97", "D0_98"), ("D1_97", "D1_98")]:
    dif = cmp_[a_] - cmp_[b_]
    rr = smf.ols("d ~ 1", data=pd.DataFrame({"d": dif, "cl": cmp_["cl"]})).fit(
        cov_type="cluster", cov_kwds={"groups": cmp_["cl"]})
    w("{} vs {}: means {:.2f} vs {:.2f}, aw {:.2f} vs {:.2f}; mean paired difference "
      "{:+.2f} (SE {:.2f}); lower in 1997 in {} of {} districts.".format(
          a_, b_, cmp_[a_].mean(), cmp_[b_].mean(),
          np.average(cmp_[a_], weights=cmp_["gca97"]), np.average(cmp_[b_], weights=cmp_["gca98"]),
          rr.params.iloc[0], rr.bse.iloc[0], int((dif < 0).sum()), len(dif)))
w("Cropped area ratio 1997/1998 per district: median {:.2f}; below 0.8 in {} of {}; "
  "below 0.5 in {}; total {:.1f} vs {:.1f} Mha.".format(
      cmp_["gr"].median(), int((cmp_["gr"] < 0.8).sum()), len(cmp_), int((cmp_["gr"] < 0.5).sum()),
      cmp_["gca97"].sum() / 1e6, cmp_["gca98"].sum() / 1e6))
w("")
# seasons missing per district
ds = ra[ra["year_start"].isin([1997, 1998])].groupby(["district_key", "year_start"])["season"].apply(set).unstack()
ds = ds.dropna()
miss = Counter = {}
for k, r in ds.iterrows():
    for s in r[1998] - r[1997]:
        miss[s] = miss.get(s, 0) + 1
w("Districts in both years: seasons present in 1998-99 but missing in 1997-98: " +
  ", ".join("{} {}".format(s, n) for s, n in sorted(miss.items(), key=lambda x: -x[1])) +
  " (of {} districts).".format(len(ds)))
# crops missing per district
dc = ra[ra["year_start"].isin([1997, 1998])].groupby(["district_key", "year_start"])["crop_name"].apply(set).unstack().dropna()
mc = {}
for k, r in dc.iterrows():
    for c in r[1998] - r[1997]:
        mc[c] = mc.get(c, 0) + 1
w("Crops present in 1998-99 but missing in 1997-98 within the same district, most "
  "frequent: " + ", ".join("{} {}".format(c, n) for c, n in sorted(mc.items(), key=lambda x: -x[1])[:15]) + ".")
w("")
# largest crop share 1997 vs 1998 in same districts
top97 = dy[dy["year_start"] == 1997].groupby("district_key")["share"].max()
top98 = dy[dy["year_start"] == 1998].groupby("district_key")["share"].max()
tt = pd.DataFrame({"t97": top97, "t98": top98}).dropna()
w("Largest crop's share of district area, same districts: mean {:.3f} in 1997 vs {:.3f} "
  "in 1998 (n {}).".format(tt["t97"].mean(), tt["t98"].mean(), len(tt)))
dom97 = dy[dy["year_start"] == 1997].sort_values("share").groupby("district_key").tail(1)["crop_name"].value_counts().head(6)
dom98 = dy[dy["year_start"] == 1998].sort_values("share").groupby("district_key").tail(1)["crop_name"].value_counts().head(6)
w("Most common largest crop 1997: " + ", ".join("{} {}".format(a, b) for a, b in dom97.items()))
w("Most common largest crop 1998: " + ", ".join("{} {}".format(a, b) for a, b in dom98.items()))
w("")

with open(OUT + "/d0_results_raw.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(R))
print("wrote", OUT + "/d0_results_raw.md")
