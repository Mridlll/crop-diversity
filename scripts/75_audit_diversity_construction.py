"""
75_audit_diversity_construction.py

Audit of how the diversity indices in `district_diversity_indices.csv` are built,
against the raw APY file. Nothing here changes any output; it only reports.

Reading script 57, three things need checking before the indices can be trusted:

  A. `district_avg` groups by district ONLY, pooling all 24 years and all seasons
     into a single call of compute_diversity. So `crop_richness` counts crops grown
     at any point in 24 years, not in a year, and `total_cropped_area` is a 24-year
     sum. If districts have unequal year coverage, richness is mechanically biased
     by coverage, and richness is one third of the ABI.

  B. Seasons are summed. In APY, "Whole Year" can be a separate reporting line for
     perennials OR a total that already contains Kharif and Rabi. If the latter,
     every area is double counted.

  C. Shannon and Simpson computed on 24-year pooled shares are the diversity of the
     long-run average cropping pattern, not the average of annual diversity. Crops
     that swap in and out across years inflate the pooled figure.

Checks run:
  A1  year coverage per district, and whether it correlates with richness
  A2  pooled 24-year richness vs mean annual richness
  B1  season values present, and whether crops overlap between Whole Year and
      Kharif/Rabi within the same district-year
  B2  exact duplicate rows on (district, year, season, crop)
  B3  reconstructed gross cropped area vs a published national benchmark
  C1  pooled Shannon vs mean annual Shannon
  D1  does the SHRUG inverted-U survive a corrected, annual-mean diversity measure?

Output: outputs/shrug_covariates/diversity_construction_audit.md
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

RAW = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
REPO = r"D:/crop-diversity"
OUT = REPO + "/outputs/shrug_covariates"
DIV = REPO + "/outputs/crop_diversity_analysis"

R = []


def w(s=""):
    print(s)
    R.append(str(s))


def head(t):
    w("")
    w("## " + t)
    w("")


w("# Audit: how the crop diversity indices are constructed")
w("")
w("Raw source: `all_crops_apy_1997_2021_india_data_portal.csv`")
w("Compared against: `district_diversity_indices.csv` (725 districts)")

df = pd.read_csv(RAW)
w("")
w("Raw file: {:,} rows, {} columns.".format(*df.shape))
w("Columns: " + ", ".join(df.columns))

for c in ["year", "season", "state_name", "district_name", "crop_name", "crop_type"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()
df = df.dropna(subset=["area"])
df = df[df["area"] > 0]
df["year_start"] = df["year"].str.split("-").str[0].astype(int)
df["district_key"] = df["state_name"].str.upper() + "|" + df["district_name"].str.upper()
w("")
w("After dropping missing and non-positive area: {:,} rows, {} districts, "
  "{} crops, {} years.".format(len(df), df["district_key"].nunique(),
                               df["crop_name"].nunique(), df["year"].nunique()))


# ------------------------------------------------------------------ B1
head("B1. Seasons, and whether Whole Year double counts")
sv = df["season"].value_counts()
w("| season | rows | share of area |")
w("|---|---|---|")
ar = df.groupby("season")["area"].sum()
for s_, n in sv.items():
    w("| {} | {:,} | {:.3f} |".format(s_, n, ar[s_] / ar.sum()))

# does the same crop appear under Whole Year AND a seasonal label
# in the same district-year?
seasonal = {"Kharif", "Rabi", "Summer", "Autumn", "Winter"}
key = ["district_key", "year_start", "crop_name"]
wy = df[df["season"].str.lower().str.startswith("whole")][key].drop_duplicates()
se = df[df["season"].isin(seasonal)][key].drop_duplicates()
ov = wy.merge(se, on=key, how="inner")
w("")
w("Crop-district-years appearing under BOTH Whole Year and a named season: "
  "{:,}".format(len(ov)))
w("As a share of all Whole Year crop-district-years: {:.4f}".format(
    len(ov) / max(len(wy), 1)))
if len(ov):
    w("")
    w("Example overlaps:")
    for _, r_ in ov.head(8).iterrows():
        sub = df[(df["district_key"] == r_["district_key"]) &
                 (df["year_start"] == r_["year_start"]) &
                 (df["crop_name"] == r_["crop_name"])]
        w("  {} {} {}: ".format(r_["district_key"], r_["year_start"], r_["crop_name"]) +
          "; ".join("{}={:.0f}".format(a, b) for a, b in
                    zip(sub["season"], sub["area"])))
    w("")
    w("If a Whole Year row equals the sum of that crop's seasonal rows, summing all")
    w("of them double counts. Checking whether Whole Year equals the seasonal sum:")
    chk = df.groupby(key + ["season"])["area"].sum().reset_index()
    piv = chk.pivot_table(index=key, columns="season", values="area")
    have = [c for c in piv.columns if c in seasonal]
    wy_col = [c for c in piv.columns if str(c).lower().startswith("whole")]
    if wy_col and have:
        p2 = piv.dropna(subset=wy_col)
        p2 = p2[p2[have].notna().any(axis=1)]
        ssum = p2[have].sum(axis=1)
        wv = p2[wy_col[0]]
        close = ((wv - ssum).abs() / wv.replace(0, np.nan) < 0.02)
        w("  of {:,} overlapping cases, Whole Year is within 2% of the seasonal "
          "sum in {:,} ({:.1%})".format(len(p2), int(close.sum()),
                                        close.mean() if len(p2) else 0))
else:
    w("")
    w("No overlap. Whole Year is a separate reporting line, mostly perennials, so")
    w("summing seasons is a legitimate gross cropped area and does NOT double count.")


# ------------------------------------------------------------------ B2
head("B2. Duplicate rows")
dupk = ["district_key", "year", "season", "crop_name"]
nd = int(df.duplicated(subset=dupk).sum())
w("Exact duplicates on (district, year, season, crop): {:,} of {:,} rows.".format(
    nd, len(df)))
if nd:
    d2 = df[df.duplicated(subset=dupk, keep=False)].sort_values(dupk)
    w("")
    w("These are summed by the groupby, which inflates area for the affected cells.")
    w("Example:")
    w("```")
    w(d2.head(6)[dupk + ["area"]].to_string(index=False))
    w("```")


# ------------------------------------------------------------------ B3
head("B3. Does reconstructed area look like real gross cropped area?")
nat = df.groupby("year_start")["area"].sum() / 1e6
w("India gross cropped area implied by this file, million hectares:")
w("")
w("| year | implied GCA (m ha) |")
w("|---|---|")
for y in [1997, 2000, 2005, 2010, 2015, 2019, 2020]:
    if y in nat.index:
        w("| {} | {:.1f} |".format(y, nat[y]))
w("")
w("> Published Indian gross cropped area is about 195-200 million hectares.")
w("> A figure far below that means the file is a partial crop or district set;")
w("> far above means double counting.")


# ------------------------------------------------------------------ A1
head("A1. Year coverage per district, and its effect on richness")
cov = df.groupby("district_key")["year_start"].nunique().rename("n_years")
w("Years of data per district: min {}, p25 {}, median {}, p75 {}, max {}.".format(
    cov.min(), int(cov.quantile(.25)), int(cov.median()),
    int(cov.quantile(.75)), cov.max()))
w("Districts with fewer than 20 years: {} of {} ({:.1%}).".format(
    int((cov < 20).sum()), len(cov), (cov < 20).mean()))

pooled = df.groupby("district_key").agg(
    rich_pooled=("crop_name", "nunique"),
    area_pooled=("area", "sum"))
ann = (df.groupby(["district_key", "year_start"])["crop_name"].nunique()
         .groupby("district_key").mean().rename("rich_annual_mean"))
cmp_ = pooled.join(ann).join(cov)
r_cov = stats.pearsonr(cmp_["n_years"], cmp_["rich_pooled"])
w("")
w("Correlation between years of coverage and POOLED richness: r = {:.3f} (p = {:.2g}).".format(
    r_cov[0], r_cov[1]))
r_cov2 = stats.pearsonr(cmp_["n_years"], cmp_["rich_annual_mean"])
w("Correlation between years of coverage and MEAN ANNUAL richness: r = {:.3f} (p = {:.2g}).".format(
    r_cov2[0], r_cov2[1]))
w("")
w("If the first is much larger than the second, pooled richness is partly measuring")
w("how long a district was observed rather than how diverse it is.")


# ------------------------------------------------------------------ A2
head("A2. Pooled richness vs mean annual richness")
w("| statistic | pooled over 24 years | mean annual |")
w("|---|---|---|")
for lab, f in [("mean", np.mean), ("median", np.median),
               ("p10", lambda x: np.percentile(x, 10)),
               ("p90", lambda x: np.percentile(x, 90)), ("max", np.max)]:
    w("| {} | {:.1f} | {:.1f} |".format(
        lab, f(cmp_["rich_pooled"]), f(cmp_["rich_annual_mean"])))
w("")
w("Ratio of pooled to annual, median: {:.2f}x.".format(
    (cmp_["rich_pooled"] / cmp_["rich_annual_mean"]).median()))
w("Rank correlation between the two: rho = {:.3f}.".format(
    stats.spearmanr(cmp_["rich_pooled"], cmp_["rich_annual_mean"])[0]))


# ------------------------------------------------------------------ C1
head("C1. Pooled Shannon vs mean annual Shannon")


def shannon(a):
    a = a[a > 0]
    p = a / a.sum()
    return float(-(p * np.log(p)).sum())


def simpson(a):
    a = a[a > 0]
    p = a / a.sum()
    return float(1 - (p ** 2).sum())


pool_sh = (df.groupby(["district_key", "crop_name"])["area"].sum()
             .groupby("district_key").apply(lambda s: shannon(s.values))
             .rename("shannon_pooled"))
ann_sh = (df.groupby(["district_key", "year_start", "crop_name"])["area"].sum()
            .groupby(["district_key", "year_start"]).apply(lambda s: shannon(s.values))
            .groupby("district_key").mean().rename("shannon_annual_mean"))
pool_si = (df.groupby(["district_key", "crop_name"])["area"].sum()
             .groupby("district_key").apply(lambda s: simpson(s.values))
             .rename("simpson_pooled"))
ann_si = (df.groupby(["district_key", "year_start", "crop_name"])["area"].sum()
            .groupby(["district_key", "year_start"]).apply(lambda s: simpson(s.values))
            .groupby("district_key").mean().rename("simpson_annual_mean"))
S = pd.concat([pool_sh, ann_sh, pool_si, ann_si], axis=1)
w("| index | pooled mean | annual mean | difference | rank correlation |")
w("|---|---|---|---|---|")
for a, b, lab in [("shannon_pooled", "shannon_annual_mean", "Shannon"),
                  ("simpson_pooled", "simpson_annual_mean", "Simpson")]:
    k = S[a].notna() & S[b].notna()
    w("| {} | {:.3f} | {:.3f} | {:+.3f} | {:.3f} |".format(
        lab, S.loc[k, a].mean(), S.loc[k, b].mean(),
        S.loc[k, a].mean() - S.loc[k, b].mean(),
        stats.spearmanr(S.loc[k, a], S.loc[k, b])[0]))

# reproduce the published file to confirm we understand the pipeline
pub = pd.read_csv(DIV + "/district_diversity_indices.csv")
chk = pub.merge(S.reset_index(), left_on="district_key", right_on="district_key", how="left")
k = chk["shannon_index"].notna() & chk["shannon_pooled"].notna()
w("")
w("Reproducing the published `shannon_index` from raw with the pooled method: "
  "r = {:.4f} on {} districts.".format(
      stats.pearsonr(chk.loc[k, "shannon_index"], chk.loc[k, "shannon_pooled"])[0],
      int(k.sum())))
w("Median absolute difference: {:.4f}.".format(
    (chk.loc[k, "shannon_index"] - chk.loc[k, "shannon_pooled"]).abs().median()))
w("")
w("A near-perfect match confirms the published indices ARE the 24-year pooled")
w("version, and that this audit is reading the pipeline correctly.")

corrected = pd.concat([cmp_["rich_annual_mean"], ann_sh, ann_si,
                       cmp_["rich_pooled"], pool_sh, pool_si, cov], axis=1).reset_index()
corrected.to_csv(OUT + "/diversity_corrected_annual.csv", index=False)


# ------------------------------------------------------------------ D1
head("D1. Does the inverted U survive a corrected diversity measure?")
import statsmodels.formula.api as smf

panel = pd.read_csv(OUT + "/analysis_panel.csv")
p = panel.merge(corrected, left_on="div_key", right_on="district_key",
                how="left", suffixes=("", "_c"))
p = p[p["in_analysis"]].copy()

# rebuild ABI the same way script 57 does, but on annual-mean components
for src, dst in [("shannon_annual_mean", "sh_n"), ("simpson_annual_mean", "si_n"),
                 ("rich_annual_mean", "ri_n")]:
    v = pd.to_numeric(p[src], errors="coerce")
    p[dst] = (v - v.min()) / (v.max() - v.min())
p["abi_annual"] = (p["sh_n"] + p["si_n"] + p["ri_n"]) / 3
w("Districts with a corrected measure: {} of {}.".format(
    int(p["abi_annual"].notna().sum()), len(p)))
w("Correlation between published ABI and the annual-mean ABI: r = {:.3f}, rho = {:.3f}.".format(
    *[f(p["agro_biodiversity_index"], p["abi_annual"])[0] for f in
      (lambda a, b: stats.pearsonr(*[x[(a.notna()) & (b.notna())] for x in (a, b)]),
       lambda a, b: stats.spearmanr(*[x[(a.notna()) & (b.notna())] for x in (a, b)]))]))
w("")
w("| outcome | n | linear | squared | p(squared) | turning point |")
w("|---|---|---|---|---|---|")
for y, lab in [("agro_biodiversity_index", "published ABI (24-yr pooled)"),
               ("abi_annual", "ABI on annual means"),
               ("shannon_annual_mean", "Shannon, annual mean"),
               ("simpson_annual_mean", "Simpson, annual mean"),
               ("rich_annual_mean", "richness, annual mean")]:
    s2 = p.dropna(subset=[y, "irr_share", "shrug_key"])
    m = smf.ols(y + " ~ irr_share + I(irr_share**2) + C(state_name)", data=s2).fit(
        cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})
    b1, b2 = m.params["irr_share"], m.params["I(irr_share ** 2)"]
    p2 = m.pvalues["I(irr_share ** 2)"]
    star = "***" if p2 < 0.01 else ("**" if p2 < 0.05 else ("*" if p2 < 0.10 else ""))
    w("| {} | {} | {:+.3f} | {:+.3f}{} | {:.4f} | {:.3f} |".format(
        lab, int(m.nobs), b1, b2, star, p2, -b1 / (2 * b2)))

with open(OUT + "/diversity_construction_audit.md", "w", encoding="utf-8") as f:
    f.write("\n".join(R))
print("\nWROTE diversity_construction_audit.md and diversity_corrected_annual.csv")
