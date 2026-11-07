"""
96_monoculture_baseline.py

Data for two additions to the site: a reference scale for reading the diversity
measures, and a monoculture and cropping-system layer.

  (A) scale       the count of crops (D0), the effective number of crops (D1), the
                  number of main crops (D2) and evenness (D1/D0) at fixed reference
                  points: one crop on all the land, two equal crops, the least
                  diverse district, p10, median, mean, p90, the most diverse
                  district. For the 725 crop-record districts and the 606-district
                  analysis set.

  (B) monoculture per district, the average share of gross cropped area held by
                  its largest crop and by its two largest crops, and the identity
                  of each. Classes, in priority order:
                    mono   largest crop holds at least two thirds of the land
                    pair   two largest hold at least 80 percent, neither alone 2/3
                    few    fewer than 10 crops a year, and neither of the above
                    mixed  everything else
                  Also the 50 and 80 percent thresholds, correlations with D0 and
                  D1, and the trend 1998-99 to 2019-20.

  (C) systems     the season-by-season cropping system. Seasons are grouped into
                  three slots: monsoon (Kharif, Autumn, Winter), rabi (Rabi),
                  summer (Summer). 'Whole Year' rows count in every annual measure
                  but lead no slot. Each district-year is classified, and the
                  district takes its most frequent class across years:
                    rice-wheat  rice leads the monsoon slot, wheat leads rabi, and
                                each holds at least 20 percent of the year's land
                    rice-rice   rice leads the monsoon slot and leads rabi or
                                summer too, and that second rice crop holds at
                                least 10 percent of the year's land
                    X-Y         monsoon leader X, leader Y of rabi plus summer,
                                where Y's dry-season area is at least 10 percent of
                                the year's land
                    X only      monsoon leader X with no dry-season crop that large
                    whole year  'Whole Year' rows are at least half the land

Cleaning is exactly script 76's: bogus state-district pairs dropped, Uttarakhand,
Chhattisgarh and Jharkhand districts remapped, 2020-21 dropped, duplicate
(district, year, season, crop) rows collapsed by maximum.

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/96_monoculture_baseline.py

Output:
  docs/data/site_monoculture.json
"""
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from _years import keep_years
from scipy import stats

RAW = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
REPO = r"D:/crop-diversity"
DIV = REPO + "/outputs/crop_diversity_analysis"
COV = REPO + "/outputs/shrug_covariates"
OUT = REPO + "/docs/data/site_monoculture.json"

MONO = 2 / 3          # near-monoculture
PAIR = 0.80           # two crops hold the year
FEW = 10              # fewer than this many crops a year
ALT = [0.50, 0.80]    # alternative single-crop thresholds
RW_MIN = 0.20         # rice and wheat each at least this share of the year
SECOND_MIN = 0.10     # second-season crop at least this share of the year
WHOLE_MIN = 0.50      # 'Whole Year' rows at least this share: seasons not separated

ALIAS = {"orissa": "odisha", "pondicherry": "puducherry", "uttaranchal": "uttarakhand",
         "jammu and kashmir": "jammu kashmir", "nct of delhi": "delhi",
         "andaman and nicobar islands": "andaman nicobar islands",
         "andaman and nicobar": "andaman nicobar islands",
         "dadra and nagar haveli": "dadra nagar haveli", "daman and diu": "daman diu",
         "the dadra and nagar haveli and daman and diu": "dadra nagar haveli"}


def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
    return re.sub(r"\s+", " ", s)


def key(state, dist):
    return ALIAS.get(norm(state), norm(state)) + "|" + norm(dist)


def title(s):
    return " ".join(x if x in ("and", "of") else x.capitalize() for x in str(s).lower().split())


def r(v, k=4):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), k)


SHORT = {"Cotton(Lint)": "Cotton", "Rapeseed &Mustard": "Mustard",
         "Moong(Green Gram)": "Moong", "Peas & Beans (Pulses)": "Peas and beans",
         "Other  Rabi Pulses": "Other rabi pulses", "Other Kharif Pulses": "Other kharif pulses",
         "Soyabean": "Soybean", "Arhar/Tur": "Tur", "Cowpea(Lobia)": "Cowpea",
         "Castor Seed": "Castor", "Guar Seed": "Guar", "Niger Seed": "Niger"}


def short(c):
    return SHORT.get(c, c)


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

# ------------------------------------------------------------------ index tables
ind = pd.read_csv(DIV + "/district_diversity_indices_corrected.csv")
ind = ind.rename(columns={"D0_richness": "D0", "D1_exp_shannon": "D1",
                          "D2_inv_simpson": "D2", "evenness_D1_D0": "E", "n_years": "y"})
fp = pd.read_csv(COV + "/final_panel.csv")
S606 = set(fp.loc[fp["in_final"] == True, "div_key"])
assert len(S606) == 606
ind["in606"] = ind["district_key"].isin(S606)
assert len(ind) == 724 and ind["in606"].sum() == 606  # Mumbai has crop rows only in 1997-98
ann = pd.read_csv(DIV + "/district_year_diversity_panel_corrected.csv")

# ================================================================== (A) scale
IDX = ["D0", "D1", "D2", "E"]


def place(row):
    return {"district": title(row["district_name"]), "state": title(row["state_name"]),
            "k": key(row["state_name"], row["district_name"]), "years": int(row["y"])}


def scale(t):
    rows = [{"id": "one", "label": "One crop on all the land", "D0": 1, "D1": 1, "D2": 1, "E": 1},
            {"id": "two", "label": "Two crops, equal land", "D0": 2, "D1": 2, "D2": 2, "E": 1}]
    lo = t.loc[t["D1"].idxmin()]
    hi = t.loc[t["D1"].idxmax()]
    q = lambda p: {c: r(t[c].quantile(p), 3) for c in IDX}
    lo_r = {"id": "min", "label": "Least diverse district"}
    lo_r.update({c: r(lo[c], 3) for c in IDX}); lo_r.update(place(lo))
    hi_r = {"id": "max", "label": "Most diverse district"}
    hi_r.update({c: r(hi[c], 3) for c in IDX}); hi_r.update(place(hi))
    rows += [lo_r,
             dict(id="p10", label="10th percentile", **q(.10)),
             dict(id="median", label="Median district", **q(.50)),
             dict(id="mean", label="National average (district mean)",
                  **{c: r(t[c].mean(), 3) for c in IDX}),
             dict(id="p90", label="90th percentile", **q(.90)),
             hi_r]
    # per-index extremes with names, since the least diverse district on one measure
    # need not be the least on another
    ext = {}
    for c in IDX:
        a, b = t.loc[t[c].idxmin()], t.loc[t[c].idxmax()]
        ext[c] = {"min": dict(value=r(a[c], 3), **place(a)),
                  "max": dict(value=r(b[c], 3), **place(b))}
    w = t["mean_annual_cropped_area"]
    aw = {c: r(np.average(t[c], weights=w), 3) for c in IDX}
    return {"n": int(len(t)), "rows": rows, "extremes": ext, "area_weighted_mean": aw}


SCALE = {"all": scale(ind), "sample": scale(ind[ind["in606"]]),
         "all_10y": scale(ind[ind["y"] >= 10])}

# ================================================================== (B) monoculture
dy = df.groupby(["district_key", "year_start", "crop_name"])["area"].sum().reset_index()
dy["sh"] = dy["area"] / dy.groupby(["district_key", "year_start"])["area"].transform("sum")
ny = dy.groupby("district_key")["year_start"].nunique()

# average over years of each crop's annual share, zero in years it is absent.
# The largest crop is the one with the highest AVERAGE SHARE. Ranking by mean annual
# area instead (as dominant_crop in script 76 does) lets one year with an absurd
# area figure crown a minor crop: Nadia's Niger seed is the worked example.
wide = dy.pivot_table(index=["district_key", "year_start"], columns="crop_name",
                      values="sh", fill_value=0.0)
avs = wide.groupby(level=0).mean()
rk = avs.stack().rename("avs").reset_index()
rk = rk[rk["avs"] > 0].sort_values(["district_key", "avs"], ascending=[True, False])
rk["rank"] = rk.groupby("district_key").cumcount() + 1
t1 = rk[rk["rank"] == 1].set_index("district_key")
t2 = rk[rk["rank"] == 2].set_index("district_key")
M = pd.DataFrame({"top": t1["crop_name"], "tsh": t1["avs"]})
M["second"] = t2["crop_name"].reindex(M.index).fillna("")
M["ssh"] = t2["avs"].reindex(M.index).fillna(0.0)
M["t2sh"] = M["tsh"] + M["ssh"]
# for reference: average of each year's own largest share (leader may change by year)
yr_top = dy.groupby(["district_key", "year_start"])["sh"].max()
M["tsh_yearly_leader"] = yr_top.groupby("district_key").mean()
M = M.reset_index().rename(columns={"index": "district_key"})
I = ind.merge(M, on="district_key", how="left")
assert I["tsh"].notna().all()

# check against the dominant_crop fields of script 76
top_agree = float((I["top"] == I["dominant_crop"]).mean())
top_disagree = [title(d) + ", " + title(s) + ": " + short(a) + " not " + short(b) for d, s, a, b in
                I.loc[I["top"] != I["dominant_crop"], ["district_name", "state_name", "top", "dominant_crop"]].values]
diff_domsh = float((I["tsh"] - I["dominant_crop_share"]).abs().median())

I["cls"] = np.select(
    [I["tsh"] >= MONO, I["t2sh"] >= PAIR, I["D0"] < FEW],
    ["mono", "pair", "few"], default="mixed")


def cls_summary(t):
    out = {}
    for c in ["mono", "pair", "few", "mixed"]:
        s = t[t["cls"] == c]
        out[c] = {"n": int(len(s)),
                  "states": {title(k): int(v) for k, v in s["state_name"].value_counts().items()},
                  "top_crops": {short(k): int(v) for k, v in s["top"].value_counts().items()},
                  "D0_median": r(s["D0"].median(), 2), "D1_median": r(s["D1"].median(), 2),
                  "D2_median": r(s["D2"].median(), 2), "E_median": r(s["E"].median(), 3),
                  "D1_mean": r(s["D1"].mean(), 2), "D0_mean": r(s["D0"].mean(), 2),
                  "tsh_median": r(s["tsh"].median(), 3), "t2sh_median": r(s["t2sh"].median(), 3)}
    # the three ideas overlap before priority is applied; report the overlaps
    out["overlap"] = {
        "few_and_mono": int(((t["D0"] < FEW) & (t["tsh"] >= MONO)).sum()),
        "few_and_pair": int(((t["D0"] < FEW) & (t["tsh"] < MONO) & (t["t2sh"] >= PAIR)).sum()),
        "few_any": int((t["D0"] < FEW).sum())}
    out["thresholds"] = {"{:.0f}".format(th * 100): int((t["tsh"] >= th).sum())
                         for th in [0.5, MONO, 0.8]}
    out["thresholds_states"] = {"{:.0f}".format(th * 100):
                                {title(k): int(v) for k, v in
                                 t.loc[t["tsh"] >= th, "state_name"].value_counts().items()}
                                for th in [0.5, MONO, 0.8]}
    out["corr"] = {}
    for c in ["D0", "D1", "D2", "E"]:
        out["corr"]["tsh_" + c] = {"pearson": r(t["tsh"].corr(t[c]), 3),
                                   "spearman": r(stats.spearmanr(t["tsh"], t[c])[0], 3)}
        out["corr"]["t2sh_" + c] = {"pearson": r(t["t2sh"].corr(t[c]), 3),
                                    "spearman": r(stats.spearmanr(t["t2sh"], t[c])[0], 3)}
    out["tsh_dist"] = {p: r(t["tsh"].quantile(q), 3) for p, q in
                       [("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9)]}
    out["tsh_mean"] = r(t["tsh"].mean(), 3)
    out["top_crop_counts"] = {short(k): int(v) for k, v in t["top"].value_counts().head(12).items()}
    return out


MONO_SUM = {"all": cls_summary(I), "sample": cls_summary(I[I["in606"]])}

# ------------------------------------------------------------------ trend
yt = yr_top.reset_index().rename(columns={"sh": "ysh"})
yt = yt.merge(dy.groupby(["district_key", "year_start"])["area"].sum().rename("gca").reset_index(),
              on=["district_key", "year_start"])
yt = yt[yt["district_key"].isin(set(ind["district_key"]))]
by_year = []
for y, g in yt.groupby("year_start"):
    by_year.append({"year": "{}-{}".format(y, str(y + 1)[-2:]), "n": int(len(g)),
                    "share_mono": r((g["ysh"] >= MONO).mean(), 4),
                    "share_50": r((g["ysh"] >= .5).mean(), 4),
                    "share_80": r((g["ysh"] >= .8).mean(), 4),
                    "area_share_mono": r(g.loc[g["ysh"] >= MONO, "gca"].sum() / g["gca"].sum(), 4),
                    "median_top_share": r(g["ysh"].median(), 4)})
W1, W2 = range(1998, 2005), range(2013, 2020)
a = yt[yt["year_start"].isin(W1)].groupby("district_key")["ysh"].agg(["mean", "size"])
b = yt[yt["year_start"].isin(W2)].groupby("district_key")["ysh"].agg(["mean", "size"])
bal = a.join(b, lsuffix="_1", rsuffix="_2", how="inner")
bal = bal[(bal["size_1"] >= 5) & (bal["size_2"] >= 5)]
d_ = bal["mean_2"] - bal["mean_1"]
TREND = {
    "by_year": by_year,
    "balanced": {
        "window_1": "1998-99 to 2004-05", "window_2": "2013-14 to 2019-20",
        "rule": "district observed in at least 5 of the 7 years of each window",
        "n": int(len(bal)),
        "mono_window_1": int((bal["mean_1"] >= MONO).sum()),
        "mono_window_2": int((bal["mean_2"] >= MONO).sum()),
        "n50_window_1": int((bal["mean_1"] >= .5).sum()),
        "n50_window_2": int((bal["mean_2"] >= .5).sum()),
        "n80_window_1": int((bal["mean_1"] >= .8).sum()),
        "n80_window_2": int((bal["mean_2"] >= .8).sum()),
        "median_top_share_1": r(bal["mean_1"].median(), 4),
        "median_top_share_2": r(bal["mean_2"].median(), 4),
        "mean_change": r(d_.mean(), 4),
        "rose_5pt": int((d_ >= .05).sum()), "fell_5pt": int((d_ <= -.05).sum()),
        "wilcoxon_p": r(stats.wilcoxon(bal["mean_2"], bal["mean_1"]).pvalue, 6)}}

# ================================================================== (C) systems
SLOT = {"Kharif": "mon", "Autumn": "mon", "Winter": "mon", "Rabi": "rabi",
        "Summer": "sum", "Whole Year": "whole"}
df["slot"] = df["season"].map(SLOT)
assert df["slot"].notna().all(), df.loc[df["slot"].isna(), "season"].unique()
sy = df.groupby(["district_key", "year_start", "slot", "crop_name"])["area"].sum()
gca = df.groupby(["district_key", "year_start"])["area"].sum()


def lead(s):
    """leader crop and its area in a slot series indexed by crop."""
    if s is None or len(s) == 0:
        return None, 0.0
    c = s.idxmax()
    return c, float(s.max())


rec = []
for (dk, y), g in sy.groupby(level=[0, 1]):
    g = g.droplevel([0, 1])
    tot = gca.loc[(dk, y)]
    sl = {k: g.xs(k, level=0) if k in g.index.get_level_values(0) else pd.Series(dtype=float)
          for k in ["mon", "rabi", "sum", "whole"]}
    dry = sl["rabi"].add(sl["sum"], fill_value=0)
    mon_c, mon_a = lead(sl["mon"])
    rabi_c, _ = lead(sl["rabi"])
    sum_c, _ = lead(sl["sum"])
    dry_c, dry_a = lead(dry)
    rice_mon = float(sl["mon"].get("Rice", 0.0)) / tot
    rice_dry = float(dry.get("Rice", 0.0)) / tot
    wheat_rabi = float(sl["rabi"].get("Wheat", 0.0)) / tot
    whole = float(sl["whole"].sum()) / tot
    is_rw = (mon_c == "Rice") and (rabi_c == "Wheat") and rice_mon >= RW_MIN and wheat_rabi >= RW_MIN
    is_rr = (mon_c == "Rice") and ("Rice" in (rabi_c, sum_c)) and rice_dry >= SECOND_MIN
    if whole >= WHOLE_MIN:
        sysy = "whole"
    elif is_rw and is_rr:
        sysy = "rice-wheat" if wheat_rabi >= rice_dry else "rice-rice"
    elif is_rw:
        sysy = "rice-wheat"
    elif is_rr:
        sysy = "rice-rice"
    elif mon_c is None:
        sysy = "{} (dry season only)".format(dry_c) if dry_c else "none"
    elif dry_c is not None and dry_a / tot >= SECOND_MIN:
        sysy = "{}-{}".format(mon_c, dry_c)
    else:
        sysy = "{} only".format(mon_c)
    rec.append({"district_key": dk, "year": y, "sys": sysy, "mon": mon_c, "dry": dry_c,
                "rice_mon": rice_mon, "rice_dry": rice_dry, "wheat_rabi": wheat_rabi,
                "whole": whole, "mon_sh": float(sl["mon"].sum()) / tot,
                "dry_sh": float(dry.sum()) / tot})
SY = pd.DataFrame(rec)

modal = SY.groupby("district_key")["sys"].agg(lambda s: Counter(s).most_common(1)[0][0])
agree = SY.groupby("district_key")["sys"].agg(lambda s: Counter(s).most_common(1)[0][1] / len(s))
ever_rr = SY.groupby("district_key")["sys"].agg(lambda s: int((s == "rice-rice").sum()))
ever_rw = SY.groupby("district_key")["sys"].agg(lambda s: int((s == "rice-wheat").sum()))
means = SY.groupby("district_key")[["rice_mon", "rice_dry", "wheat_rabi", "whole",
                                    "mon_sh", "dry_sh"]].mean()
I = I.merge(modal.rename("sys_raw").reset_index(), on="district_key", how="left")
I = I.merge(agree.rename("sys_agree").reset_index(), on="district_key", how="left")
I = I.merge(ever_rr.rename("yrs_rr").reset_index(), on="district_key", how="left")
I = I.merge(ever_rw.rename("yrs_rw").reset_index(), on="district_key", how="left")
I = I.merge(means.reset_index(), on="district_key", how="left")

# ------------------------------------------------------------------ name the systems
raw_counts = I["sys_raw"].value_counts()
print("raw modal systems (725):")
print(raw_counts.head(40).to_string())


def family(s):
    """group raw X-Y labels into the families the page names."""
    if s in ("rice-wheat", "rice-rice", "whole", "none"):
        return s
    a, _, b = s.partition("-")
    if s.endswith(" only"):
        a, b = s[:-5], None
    if "(dry season only)" in s:
        return "other"
    a = a.strip()
    if a == "Rice":
        if b == "Wheat":
            return "rice-wheat-light"
        if b in ("Gram", "Urad", "Moong(Green Gram)", "Khesari", "Masoor",
                 "Other  Rabi Pulses", "Peas & Beans (Pulses)", "Horse-Gram"):
            return "rice-pulse"
        if b in ("Rapeseed &Mustard", "Linseed", "Sesamum", "Groundnut", "Niger Seed"):
            return "rice-oilseed"
        if b is None:
            return "rice-only"
        return "rice-other"
    if a == "Soyabean":
        return "soybean-wheat" if b == "Wheat" else ("soybean-gram" if b == "Gram" else "soybean-other")
    if a == "Cotton(Lint)":
        return "cotton-only" if b is None else "cotton-other"
    if a == "Maize":
        return "maize-wheat" if b == "Wheat" else ("maize-only" if b is None else "maize-other")
    if a == "Bajra":
        return "bajra-wheat" if b in ("Wheat",) else ("bajra-mustard" if b == "Rapeseed &Mustard"
                                                     else ("bajra-only" if b is None else "bajra-other"))
    if a in ("Jowar", "Ragi", "Small Millets"):
        return "millet-based"
    if a == "Groundnut":
        return "groundnut-based"
    if a in ("Arhar/Tur", "Moong(Green Gram)", "Urad", "Moth", "Guar Seed", "Horse-Gram"):
        return "kharif-pulse-based"
    return "other"


I["sys"] = I["sys_raw"].map(family)
fam_counts = I["sys"].value_counts()
print("\nfamilies (725):")
print(fam_counts.to_string())

LAB = {"rice-wheat": "Rice then wheat", "rice-rice": "Rice twice a year",
       "rice-pulse": "Rice then a pulse", "rice-oilseed": "Rice then an oilseed",
       "rice-only": "Rice, little second season",
       "rice-wheat-light": "Rice then wheat, one under a fifth of the land", "rice-other": "Rice then another crop",
       "soybean-wheat": "Soybean then wheat", "soybean-gram": "Soybean then gram",
       "soybean-other": "Soybean then another crop",
       "cotton-only": "Cotton, little second season", "cotton-other": "Cotton then another crop",
       "maize-wheat": "Maize then wheat", "maize-only": "Maize, little second season",
       "maize-other": "Maize then another crop", "bajra-wheat": "Bajra then wheat",
       "bajra-mustard": "Bajra then mustard", "bajra-only": "Bajra, little second season",
       "bajra-other": "Bajra then another crop", "millet-based": "Jowar, ragi or small millets lead",
       "groundnut-based": "Groundnut leads the monsoon", "kharif-pulse-based": "A pulse or guar leads the monsoon",
       "whole": "Seasons not reported separately", "other": "Other", "none": "No data"}
for f in fam_counts.index:
    assert f in LAB, f


def sys_summary(t):
    out = []
    for f, n in t["sys"].value_counts().items():
        s = t[t["sys"] == f]
        out.append({"code": f, "label": LAB[f], "n": int(n),
                    "states": {title(k): int(v) for k, v in s["state_name"].value_counts().items()},
                    "D0_median": r(s["D0"].median(), 2), "D1_median": r(s["D1"].median(), 2),
                    "D2_median": r(s["D2"].median(), 2), "E_median": r(s["E"].median(), 3),
                    "D0_mean": r(s["D0"].mean(), 2), "D1_mean": r(s["D1"].mean(), 2),
                    "D2_mean": r(s["D2"].mean(), 2),
                    "tsh_median": r(s["tsh"].median(), 3), "t2sh_median": r(s["t2sh"].median(), 3),
                    "mono_n": int((s["cls"] == "mono").sum()), "pair_n": int((s["cls"] == "pair").sum()),
                    "agree_median": r(s["sys_agree"].median(), 3),
                    "rice_mon_median": r(s["rice_mon"].median(), 3),
                    "rice_dry_median": r(s["rice_dry"].median(), 3),
                    "wheat_rabi_median": r(s["wheat_rabi"].median(), 3),
                    "examples": [title(x) + ", " + title(z) for x, z in
                                 s.sort_values("mean_annual_cropped_area", ascending=False)
                                 [["district_name", "state_name"]].head(5).values]})
    return out


SYS_SUM = {"all": sys_summary(I), "sample": sys_summary(I[I["in606"]])}

# rice-rice vs rice-wheat head to head, the comparison the page draws
H2H = {}
for f in ["rice-rice", "rice-wheat", "rice-only"]:
    for nm, t in [("all", I), ("sample", I[I["in606"]])]:
        s = t[t["sys"] == f]
        H2H.setdefault(f, {})[nm] = {
            "n": int(len(s)), "D0": r(s["D0"].median(), 2), "D1": r(s["D1"].median(), 2),
            "D2": r(s["D2"].median(), 2), "E": r(s["E"].median(), 3),
            "tsh": r(s["tsh"].median(), 3), "t2sh": r(s["t2sh"].median(), 3),
            "top_is_rice": int((s["top"] == "Rice").sum())}

# ------------------------------------------------------------------ data caveats, measured
cav = {}
st_rice = df[df["crop_name"] == "Rice"].groupby(["state_name", "slot"])["area"].sum().unstack(fill_value=0)
cav["rice_one_slot_states"] = sorted(title(s) for s, row in st_rice.iterrows()
                                     if row.sum() > 1e5 * 22 and (row[["rabi", "sum"]].sum() / row.sum()) < 0.01)
tn = df[(df["state_name"] == "Tamil Nadu") & (df["crop_name"] == "Rice")]
cav["tamil_nadu_rice_split_years"] = sorted(
    "{}-{}".format(y, str(y + 1)[-2:]) for y in tn.loc[tn["season"] != "Kharif", "year_start"].unique())
cav["whole_dominant_districts"] = int((I["whole"] >= WHOLE_MIN).sum())
cav["whole_dominant_states"] = {title(k): int(v) for k, v in
                                I.loc[I["whole"] >= WHOLE_MIN, "state_name"].value_counts().items()}
od = df[(df["state_name"] == "Odisha") & (df["crop_name"] == "Rice") & (df["season"] == "Summer")]
od_y = od.groupby("year_start")["area"].sum()
cav["odisha_summer_rice_2012_mha"] = r(od_y.get(2012, 0) / 1e6, 2)
cav["odisha_summer_rice_median_mha"] = r(od_y.drop(2012, errors="ignore").median() / 1e6, 2)
cav["sys_agree_median"] = r(I["sys_agree"].median(), 3)
cav["sys_agree_below_half"] = int((I["sys_agree"] < 0.5).sum())
print("\ncaveats:", cav)

# ================================================================== write
I["k"] = [key(s, d) for s, d in zip(I["state_name"], I["district_name"])]
dupk = int(I["k"].duplicated().sum())
D = {}
for _, x in I.drop_duplicates("k").iterrows():
    D[x["k"]] = {"dk": x["district_key"], "d": title(x["district_name"]), "s": title(x["state_name"]),
                 "y": int(x["y"]), "in606": bool(x["in606"]),
                 "top": short(x["top"]), "tsh": r(x["tsh"], 4),
                 "second": short(x["second"]) if x["second"] else None, "ssh": r(x["ssh"], 4),
                 "t2sh": r(x["t2sh"], 4), "cls": x["cls"],
                 "sys": x["sys"], "sys_detail": x["sys_raw"].replace("(Lint)", ""),
                 "sys_agree": r(x["sys_agree"], 3),
                 "yrs_rr": int(x["yrs_rr"]), "yrs_rw": int(x["yrs_rw"]),
                 "rice_mon": r(x["rice_mon"], 3), "rice_dry": r(x["rice_dry"], 3),
                 "wheat_rabi": r(x["wheat_rabi"], 3), "whole": r(x["whole"], 3),
                 "D0": r(x["D0"], 3), "D1": r(x["D1"], 3), "D2": r(x["D2"], 3), "E": r(x["E"], 4)}

out = {
    "definitions": {
        "D0": "count of crops grown in a year, averaged over years",
        "D1": "effective number of crops, exp(Shannon), per year then averaged",
        "D2": "number of main crops, inverse Simpson, per year then averaged",
        "E": "evenness, D1 / D0",
        "tsh": "average over years of the largest crop's share of gross cropped area; "
               "the largest crop is the one with the most land on average",
        "t2sh": "the same for the two largest crops together",
        "cls": {"mono": "largest crop holds at least two thirds of the land",
                "pair": "two largest crops hold at least 80 percent, neither alone two thirds",
                "few": "fewer than 10 crops a year and neither of the above",
                "mixed": "everything else"},
        "slots": {"monsoon": ["Kharif", "Autumn", "Winter"], "rabi": ["Rabi"], "summer": ["Summer"],
                  "not a slot": ["Whole Year"]},
        "systems": {"rice-wheat": "rice leads the monsoon, wheat leads rabi, each at least 20 percent of the year's land",
                    "rice-rice": "rice leads the monsoon and leads rabi or summer as well, with that second rice crop "
                                 "at least 10 percent of the year's land",
                    "pairs": "monsoon leader then the leader of rabi plus summer, when that crop is at least "
                             "10 percent of the year's land; otherwise 'little second season'",
                    "whole": "'Whole Year' rows are at least half the land",
                    "district": "each district-year is classified; the district takes its most frequent class"},
        "thresholds": {"mono": MONO, "pair": PAIR, "few": FEW, "alt": ALT, "rw_min": RW_MIN,
                       "second_min": SECOND_MIN, "whole_min": WHOLE_MIN},
        "keys": "districts keyed like the 'k' property of districts.geojson; dk is the analysis key "
                "used by site_indices.json",
        "sample": "'sample' is the 606-district analysis set; 'all' is the 725 crop-record districts; "
                  "'all_10y' is the 725 restricted to districts observed for at least 10 years"},
    "scale": SCALE,
    "monoculture": MONO_SUM,
    "trend": TREND,
    "systems": SYS_SUM,
    "system_labels": LAB,
    "head_to_head": H2H,
    "caveats": cav,
    "checks": {"tsh_vs_dominant_crop_share_median_abs_diff": r(diff_domsh, 4),
               "top_matches_dominant_crop": r(top_agree, 4),
               "top_differs_from_dominant_crop": top_disagree,
               "tsh_vs_yearly_leader_corr": r(I["tsh"].corr(I["tsh_yearly_leader"]), 4),
               "duplicate_k_dropped": dupk, "districts": len(D)},
    "districts": D}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"))

# ------------------------------------------------------------------ console report
print("\nSCALE")
for nm in ["all", "sample", "all_10y"]:
    print(nm, SCALE[nm]["n"])
    for row in SCALE[nm]["rows"]:
        print("  {:<36} D0 {:>7} D1 {:>7} D2 {:>7} E {:>7}  {}".format(
            row["label"], row["D0"], row["D1"], row["D2"], row["E"],
            (row.get("district", "") + ", " + row.get("state", "") + " y" + str(row.get("years", "")))
            if "district" in row else ""))
    for c in IDX:
        e = SCALE[nm]["extremes"][c]
        print("   {} min {} {} {} y{} | max {} {} {} y{}".format(
            c, e["min"]["value"], e["min"]["district"], e["min"]["state"], e["min"]["years"],
            e["max"]["value"], e["max"]["district"], e["max"]["state"], e["max"]["years"]))
    print("   area-weighted", SCALE[nm]["area_weighted_mean"])
print("\nMONOCULTURE")
print(json.dumps(MONO_SUM, indent=1))
print("\nTREND")
print(json.dumps(TREND["balanced"], indent=1))
for b_ in by_year:
    print(" ", b_)
print("\nSYSTEMS")
print(json.dumps(SYS_SUM, indent=1))
print("\nH2H", json.dumps(H2H, indent=1))
print("checks", out["checks"])
print("WROTE", OUT)
