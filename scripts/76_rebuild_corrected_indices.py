"""
76_rebuild_corrected_indices.py

Rebuilds the district diversity indices with the four defects the audit
(script 75) found in the original construction fixed.

DEFECT 1 - the last year is empty.
  The file is labelled 1997-2021 and the README claims 24 agricultural years.
  2020-21 holds 319 rows, 13 districts and 0.9 million hectares against 194.9
  million in 2019-20. It is a partial year, not a year. Dropped.
  Usable range is 1997-98 to 2019-20, which is 23 years.

DEFECT 2 - duplicate rows are silently summed.
  91 exact duplicates on (district, year, season, crop) survive the bogus-pair
  cleaning in script 57, mostly Niger Seed in Andhra Pradesh. groupby sums them,
  which inflates area for those cells. Collapsed by taking the maximum, since
  the pattern is a small stub row alongside the real one.

DEFECT 3 - richness is pooled over 24 years, not measured per year.
  Script 57 groups by district only, so `crop_richness` counts crops grown at
  ANY point in the period. That is 1.49x the mean annual richness, and it
  correlates with a district's years of coverage at r = 0.545 against r = 0.309
  for annual richness. Since 27% of districts have under 20 years of data, a
  third of the ABI was partly measuring how long a district was observed.
  Rebuilt as the mean of annual richness.

DEFECT 4 - Shannon and Simpson pooled across years.
  Less serious: rank correlation between pooled and annual-mean is 0.975 and
  0.972, and the level gap is +0.105 and +0.025. Still rebuilt on annual means
  so all three ABI components are measured the same way.

NOT a defect, checked and cleared:
  - Summing seasons does NOT double count. Only 441 crop-district-years (0.64%
    of Whole Year cells) appear under both Whole Year and a named season, and
    only 7.7% of those look like a genuine total. Reconstructed gross cropped
    area is 180-195 million hectares, matching published national figures.

Outputs:
  outputs/crop_diversity_analysis/district_diversity_indices_corrected.csv
  outputs/shrug_covariates/corrected_vs_original.md
"""
import numpy as np
import pandas as pd
from scipy import stats

RAW = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
REPO = r"D:/crop-diversity"
DIV = REPO + "/outputs/crop_diversity_analysis"
OUT = REPO + "/outputs/shrug_covariates"

R = []


def w(s=""):
    print(s)
    R.append(str(s))


w("# Corrected diversity indices, and what changes")
w("")

# ------------------------------------------------------------------ load
df = pd.read_csv(RAW)
for c in ["year", "season", "state_name", "district_name", "crop_name", "crop_type"]:
    df[c] = df[c].astype(str).str.strip()
df = df.dropna(subset=["area"])
df = df[df["area"] > 0]
n0 = len(df)

# --- script 57's own cleaning, kept as is
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
w("Dropped {} rows on {} bogus state-district pairs.".format(n0 - len(df), len(BOGUS)))
for (s_, d_), new in REMAP.items():
    m = (df["state_name"] == s_) & (df["district_name"] == d_)
    df.loc[m, "state_name"] = new

df["year_start"] = df["year"].str.split("-").str[0].astype(int)
df["district_key"] = df["state_name"].str.upper() + "|" + df["district_name"].str.upper()

# ------------------------------------------------------------- defect 1
yr = df.groupby("year_start").agg(rows=("area", "size"),
                                  mha=("area", lambda s: s.sum() / 1e6),
                                  dist=("district_name", "nunique"))
w("")
w("## Defect 1: the last year is a stub")
w("")
w("| year | rows | million ha | districts |")
w("|---|---|---|---|")
for y in sorted(yr.index)[-4:]:
    w("| {}-{} | {:,} | {:.1f} | {} |".format(
        y, str(y + 1)[-2:], int(yr.loc[y, "rows"]), yr.loc[y, "mha"],
        int(yr.loc[y, "dist"])))
LAST = 2019
before = len(df)
df = df[df["year_start"] <= LAST]
w("")
w("Dropped {} rows in 2020-21. Usable range is 1997-98 to {}-{}, {} years.".format(
    before - len(df), LAST, str(LAST + 1)[-2:], df["year_start"].nunique()))

# ------------------------------------------------------------- defect 2
K = ["district_key", "year", "season", "crop_name"]
ndup = int(df.duplicated(subset=K).sum())
w("")
w("## Defect 2: duplicate rows")
w("")
w("{} exact duplicates on (district, year, season, crop) remain after the".format(ndup))
w("bogus-pair cleaning. Collapsed by taking the maximum rather than the sum.")
df = (df.sort_values("area", ascending=False)
        .drop_duplicates(subset=K, keep="first"))
w("Rows after collapsing: {:,}.".format(len(df)))

# ------------------------------------------------------ defects 3 and 4
w("")
w("## Defects 3 and 4: measure each index per year, then average")
w("")


def shannon(a):
    p = a[a > 0]
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def simpson(a):
    p = a[a > 0]
    p = p / p.sum()
    return float(1 - (p ** 2).sum())


dy = df.groupby(["district_key", "year_start", "crop_name"])["area"].sum().reset_index()
g = dy.groupby(["district_key", "year_start"])["area"]
ann = pd.DataFrame({
    "shannon_index": g.apply(lambda s: shannon(s.values)),
    "simpson_index": g.apply(lambda s: simpson(s.values)),
    "crop_richness": g.size(),
    "cropped_area": g.sum(),
}).reset_index()
w("District-year observations: {:,}.".format(len(ann)))

# --- Hill numbers, the defensible index family
#
# Shannon and Simpson are not independent quantities to be averaged into a
# composite. They are the same family at different sensitivities to rare crops,
# and their Hill transforms all carry ONE unit: the effective number of crops,
# meaning the number of equally-common crops that would give the observed
# diversity. That makes them comparable to each other and interpretable on their
# own, which a min-max composite is not.
#
#   D0  q=0  richness                    counts every crop equally
#   D1  q=1  exp(Shannon)                weights crops by their area share
#   D2  q=2  1 / sum(p^2)                dominated by the common crops
#   E    evenness  D1 / D0               how evenly area is spread, net of count
#
# Computed per district-year then averaged, because exp(mean) != mean(exp).
ann["D0_richness"] = ann["crop_richness"]
ann["D1_exp_shannon"] = np.exp(ann["shannon_index"])
ann["D2_inv_simpson"] = 1.0 / (1.0 - ann["simpson_index"]).clip(lower=1e-12)
ann["evenness_D1_D0"] = ann["D1_exp_shannon"] / ann["D0_richness"]
w("")
w("Hill numbers added, all in units of effective number of crops:")
w("  D0 richness        mean {:.1f}".format(ann["D0_richness"].mean()))
w("  D1 exp(Shannon)    mean {:.1f}".format(ann["D1_exp_shannon"].mean()))
w("  D2 inverse Simpson mean {:.1f}".format(ann["D2_inv_simpson"].mean()))
w("  evenness D1/D0     mean {:.3f}".format(ann["evenness_D1_D0"].mean()))
w("")
w("D0 >= D1 >= D2 must hold for every observation. Violations: {}.".format(
    int(((ann["D0_richness"] < ann["D1_exp_shannon"] - 1e-9) |
         (ann["D1_exp_shannon"] < ann["D2_inv_simpson"] - 1e-9)).sum())))

dist = ann.groupby("district_key").agg(
    shannon_index=("shannon_index", "mean"),
    simpson_index=("simpson_index", "mean"),
    crop_richness=("crop_richness", "mean"),
    D0_richness=("D0_richness", "mean"),
    D1_exp_shannon=("D1_exp_shannon", "mean"),
    D2_inv_simpson=("D2_inv_simpson", "mean"),
    evenness_D1_D0=("evenness_D1_D0", "mean"),
    mean_annual_cropped_area=("cropped_area", "mean"),
    n_years=("year_start", "nunique")).reset_index()

# dominant crop and category shares, on mean ANNUAL area not a 24-year sum
ca = (df.groupby(["district_key", "crop_name"])["area"].sum() /
      df.groupby("district_key")["year_start"].nunique())
ca = ca.reset_index(name="mean_annual_area")
tot = ca.groupby("district_key")["mean_annual_area"].transform("sum")
ca["share"] = ca["mean_annual_area"] / tot
top = ca.sort_values("share", ascending=False).groupby("district_key")
dom = top.head(1).set_index("district_key")[["crop_name", "share"]]
dom.columns = ["dominant_crop", "dominant_crop_share"]
top3 = top.head(3).groupby("district_key")["share"].sum().rename("top3_crops_share")
dist = dist.merge(dom.reset_index(), on="district_key", how="left")
dist = dist.merge(top3.reset_index(), on="district_key", how="left")

cts = df.groupby(["district_key", "crop_type"])["area"].sum().reset_index()
cts["share"] = cts["area"] / cts.groupby("district_key")["area"].transform("sum")
piv = cts.pivot_table(index="district_key", columns="crop_type",
                      values="share", fill_value=0)
piv.columns = ["share_" + str(c).lower().replace(" ", "_") for c in piv.columns]
dist = dist.merge(piv.reset_index(), on="district_key", how="left")

# ABI, same recipe as script 57 but on the corrected components
for c in ["shannon_index", "simpson_index", "crop_richness"]:
    v = dist[c]
    dist[c + "_norm"] = (v - v.min()) / (v.max() - v.min())
dist["agro_biodiversity_index"] = (
    dist[["shannon_index_norm", "simpson_index_norm", "crop_richness_norm"]]
    .mean(axis=1).round(4))
dist["abi_category"] = pd.cut(dist["agro_biodiversity_index"],
                              [0, .25, .5, .75, 1.0],
                              labels=["Very Low", "Low", "Moderate", "High"],
                              include_lowest=True)

# ------------------------------------------------- is the ABI defensible?
w("")
w("## Why the ABI is kept but demoted")
w("")
w("The ABI is the equal-weighted mean of min-max normalised Shannon, Simpson and")
w("richness. Three problems, the first of which is easy to demonstrate.")
w("")
w("**1. It is sample-dependent.** Min-max normalisation rescales against whichever")
w("districts happen to be in the file, so a district's ABI changes when other")
w("districts are added or removed, without anything about that district changing.")
rng = np.random.default_rng(11)
moves = []
for _ in range(200):
    sub = dist.sample(frac=0.80, random_state=int(rng.integers(1e9)))
    a = sub.copy()
    for c in ["shannon_index", "simpson_index", "crop_richness"]:
        v = a[c]
        a[c + "_n2"] = (v - v.min()) / (v.max() - v.min())
    a["abi2"] = a[[c + "_n2" for c in
                   ["shannon_index", "simpson_index", "crop_richness"]]].mean(axis=1)
    j = a[["district_key", "abi2"]].merge(
        dist[["district_key", "agro_biodiversity_index"]], on="district_key")
    moves.append((j["abi2"] - j["agro_biodiversity_index"]).abs().mean())
w("")
w("Recomputing the ABI on 200 random 80 percent subsamples moves a district's own")
w("score by {:.4f} on average, up to {:.4f}. The index is not a property of the".format(
    float(np.mean(moves)), float(np.max(moves))))
w("district alone.")
w("")
w("**2. Equal weighting is arbitrary.** There is no argument for one third each.")
w("")
w("**3. It double counts evenness.** Shannon and Simpson are the same family at")
w("different sensitivities and correlate at r = {:.3f} here, so two of the three".format(
    float(dist[["shannon_index", "simpson_index"]].corr().iloc[0, 1])))
w("components measure nearly the same thing, and richness is outvoted two to one.")
w("")
w("The Hill numbers have none of these problems. They are absolute, they need no")
w("normalisation, they share one unit, and D0, D1 and D2 are a deliberate ladder")
w("rather than three things to average. Results are reported on all of them; the")
w("ABI is kept only so this rebuild can be compared against the original file.")
dist[["state_name", "district_name"]] = dist["district_key"].str.split("|", expand=True)
w("Districts in the corrected file: {}.".format(len(dist)))

# ------------------------------------------------------------- compare
orig = pd.read_csv(DIV + "/district_diversity_indices.csv")
m = orig.merge(dist, on="district_key", how="inner", suffixes=("_orig", "_corr"))
w("")
w("## What changes")
w("")
w("| index | original (pooled) | corrected (annual mean) | Pearson r | Spearman rho |")
w("|---|---|---|---|---|")
for a, b, lab in [("shannon_index_orig", "shannon_index_corr", "Shannon"),
                  ("simpson_index_orig", "simpson_index_corr", "Simpson"),
                  ("crop_richness_orig", "crop_richness_corr", "richness"),
                  ("agro_biodiversity_index_orig", "agro_biodiversity_index_corr", "ABI")]:
    k = m[a].notna() & m[b].notna()
    w("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
        lab, m.loc[k, a].mean(), m.loc[k, b].mean(),
        stats.pearsonr(m.loc[k, a], m.loc[k, b])[0],
        stats.spearmanr(m.loc[k, a], m.loc[k, b])[0]))

m["abi_rank_orig"] = m["agro_biodiversity_index_orig"].rank(ascending=False)
m["abi_rank_corr"] = m["agro_biodiversity_index_corr"].rank(ascending=False)
m["rank_move"] = (m["abi_rank_orig"] - m["abi_rank_corr"]).abs()
w("")
w("ABI rank movement between the two: median {:.0f} places, "
  "90th percentile {:.0f}, max {:.0f}.".format(
      m["rank_move"].median(), m["rank_move"].quantile(.9), m["rank_move"].max()))
w("Districts moving more than 100 rank places: {} of {}.".format(
    int((m["rank_move"] > 100).sum()), len(m)))
w("")
w("Biggest movers:")
w("")
w("| district | years | original ABI | corrected ABI | rank move |")
w("|---|---|---|---|---|")
for _, r_ in m.nlargest(10, "rank_move").iterrows():
    w("| {} | {} | {:.3f} | {:.3f} | {:.0f} |".format(
        r_["district_key"], int(r_["n_years"]),
        r_["agro_biodiversity_index_orig"], r_["agro_biodiversity_index_corr"],
        r_["rank_move"]))

w("")
w("The state ranking the README reports (Karnataka most diverse, Punjab least):")
w("")
st = m.groupby("state_name_orig").agg(
    n=("district_key", "size"),
    abi_orig=("agro_biodiversity_index_orig", "mean"),
    abi_corr=("agro_biodiversity_index_corr", "mean")).query("n >= 5")
st["rank_orig"] = st["abi_orig"].rank(ascending=False)
st["rank_corr"] = st["abi_corr"].rank(ascending=False)
w("| state | n | original | corrected | rank orig | rank corr |")
w("|---|---|---|---|---|---|")
for s_, r_ in st.sort_values("abi_corr", ascending=False).head(8).iterrows():
    w("| {} | {} | {:.3f} | {:.3f} | {:.0f} | {:.0f} |".format(
        s_, int(r_["n"]), r_["abi_orig"], r_["abi_corr"],
        r_["rank_orig"], r_["rank_corr"]))
for s_, r_ in st.sort_values("abi_corr").head(3).iterrows():
    w("| {} | {} | {:.3f} | {:.3f} | {:.0f} | {:.0f} |".format(
        s_, int(r_["n"]), r_["abi_orig"], r_["abi_corr"],
        r_["rank_orig"], r_["rank_corr"]))

keep = ["state_name", "district_name", "district_key", "n_years",
        "shannon_index", "simpson_index", "crop_richness",
        "D0_richness", "D1_exp_shannon", "D2_inv_simpson", "evenness_D1_D0",
        "mean_annual_cropped_area", "dominant_crop", "dominant_crop_share",
        "top3_crops_share", "shannon_index_norm", "simpson_index_norm",
        "crop_richness_norm", "agro_biodiversity_index", "abi_category"]
keep += [c for c in dist.columns if c.startswith("share_")]
dist[keep].to_csv(DIV + "/district_diversity_indices_corrected.csv", index=False)
ann.to_csv(DIV + "/district_year_diversity_panel_corrected.csv", index=False)
with open(OUT + "/corrected_vs_original.md", "w", encoding="utf-8") as f:
    f.write("\n".join(R))
print("\nWROTE district_diversity_indices_corrected.csv ({} districts), "
      "district_year_diversity_panel_corrected.csv ({} rows), "
      "corrected_vs_original.md".format(len(dist), len(ann)))
