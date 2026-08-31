"""
78_generate_shrug_notebook.py

Generates notebooks/shrug_covariates_analysis.ipynb, the readable walkthrough of
the SHRUG covariate build, the diversity-construction audit, and the results.

The notebook reads the CSVs the pipeline produces, so it runs in seconds and does
not need the 14 GB SHRUG extract or the raw APY file to be present. Re-running the
underlying build is `python scripts/run_all.py`.

Figures follow one rule set: single ink colour plus one accent, thin marks,
recessive grid, no dual axes, no legend where there is one series.
"""
import os
import nbformat as nbf

REPO = r"D:/crop-diversity"
OUTNB = REPO + "/notebooks/shrug_covariates_analysis.ipynb"

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# Agrobiodiversity and rural structure: SHRUG covariates for the crop-diversity panel

This notebook does two things.

1. It **audits how the diversity indices in this repository were built**, against the
   raw APY file, and rebuilds them with the defects fixed.
2. It **replaces the scraped irrigation variable** the repository's headline finding
   rested on with a measure built from SHRUG 2.1, and re-tests that finding along with
   four other dimensions of rural structure.

Everything below reads CSVs the pipeline has already produced. To regenerate them:

```
python scripts/run_all.py
```

That needs `pandas`, `numpy`, `scipy` and `statsmodels`, the SHRUG 2.1 extract at
`D:/SHRUG_2.1_Data/extracted`, and the raw APY file at
`E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv`.

**The short version.** The headline finding survives, and is better identified than
before. But three real construction defects were found in the diversity indices, and
the biggest of them was inflating crop richness by about 50 percent and partly
measuring how long a district had been observed rather than how diverse it is.
""")

md("## 0. Setup")

code(r"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")

REPO = r"D:/crop-diversity"
COV  = REPO + "/outputs/shrug_covariates"
DIV  = REPO + "/outputs/crop_diversity_analysis"

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)

# One ink colour plus one accent. No categorical palette is needed because every
# figure here carries a single series.
INK, ACCENT, GRID = "#1f2328", "#0b6e6e", "#e3e6ea"
plt.rcParams.update({
    "figure.figsize": (7.2, 4.2), "figure.dpi": 120,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8b949e", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "text.color": INK,
    "xtick.color": "#57606a", "ytick.color": "#57606a",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "600",
    "legend.frameon": False,
})
print("ok")
""")

md(r"""
---
# Part 1. Auditing the diversity indices

The indices in `district_diversity_indices.csv` come from `57_crop_diversity_agro_biodiversity.py`.
Reading that script, the district-level table is built by

```python
district_avg = df.groupby(['state_name','district_name','district_key']).apply(compute_diversity)
```

which groups by district **only**. Every year and every season falls into one call of
`compute_diversity`, where `total_area = group['area'].sum()`. Four things follow, and
each needed checking against the raw file rather than assuming.
""")

md("### 1.1 Does summing seasons double count?")

md(r"""
This was the biggest worry. In APY, `Whole Year` is sometimes a separate reporting line
for perennials and sometimes a total that already contains Kharif and Rabi. If it is the
latter, summing every season double counts all the area.

It does not. Only 441 crop-district-years (0.64 percent of Whole Year cells) appear under
both Whole Year and a named season, and of those only 7.7 percent look like a genuine
total. The reconstructed national gross cropped area lands where published figures put it.
""")

code(r"""
audit = open(COV + "/diversity_construction_audit.md", encoding="utf-8").read()

def section(name, text=audit):
    # pull one section out of a generated markdown report
    i = text.index("## " + name)
    j = text.find("\n## ", i + 1)
    return text[i:j if j > 0 else len(text)]

print(section("B1."))
print(section("B3."))
""")

md(r"""
Published Indian gross cropped area is around 195-200 million hectares. The file
reconstructs to 182-195 million across the period, which is the right order and rules out
gross double counting. **Summing seasons is fine.**
""")

md("### 1.2 Defect: the last year is a stub")

code(r"""
corr_report = open(COV + "/corrected_vs_original.md", encoding="utf-8").read()
print(section("Defect 1: the last year is a stub", corr_report))
""")

md(r"""
The repository title, README and timeline page all say **1997-2021** and **24 agricultural
years**. 2020-21 contains 319 rows covering 13 districts and 0.9 million hectares, against
19,256 rows and 194.9 million hectares in 2019-20. It is a partial year, not a year.

The usable range is **1997-98 to 2019-20, which is 23 years**. This barely moves the pooled
indices, because 0.9 million hectares is nothing against a 24-year pooled total, but the
coverage claim is wrong and the animated timeline ends on a frame built from 13 districts.
""")

md("### 1.3 Defect: crop richness is pooled over the whole period")

md(r"""
This is the serious one. Because the groupby has no year in it, `crop_richness` counts the
crops a district grew **at any point in 23 years**, not the crops it grows in a year. Two
consequences:

- It is about 1.5 times the mean annual richness.
- It rewards districts that were observed for longer, because more years means more chances
  to record a rare crop.

Richness is one third of the ABI, so this propagates into the headline index.
""")

code(r"""
print(section("A1."))
print(section("A2."))
""")

code(r"""
cd = pd.read_csv(COV + "/diversity_corrected_annual.csv")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

ax[0].scatter(cd["rich_annual_mean"], cd["rich_pooled"], s=9, alpha=.45,
              color=INK, linewidths=0)
lim = [0, cd["rich_pooled"].max() * 1.05]
ax[0].plot(lim, lim, color=ACCENT, lw=1.6, zorder=3)
ax[0].set(xlim=lim, ylim=lim, xlabel="Mean annual richness",
          ylabel="Richness pooled over 23 years",
          title="Pooling inflates richness by about half")
ax[0].annotate("45 degree line", xy=(lim[1]*.62, lim[1]*.62),
               xytext=(lim[1]*.66, lim[1]*.40), color=ACCENT, fontsize=9,
               arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1))

b = cd.groupby("n_years").agg(pooled=("rich_pooled", "mean"),
                              annual=("rich_annual_mean", "mean"),
                              n=("district_key", "size"))
b = b[b["n"] >= 5]
ax[1].plot(b.index, b["pooled"], "o-", color=INK, lw=1.8, ms=5, label="pooled over period")
ax[1].plot(b.index, b["annual"], "s--", color=ACCENT, lw=1.8, ms=5, label="mean annual")
ax[1].set(xlabel="Years of data the district has", ylabel="Crop richness",
          title="Pooled richness tracks how long a district was observed")
ax[1].legend(loc="upper left")
plt.tight_layout(); plt.show()

print("correlation with years of coverage")
print("  pooled richness : r = {:.3f}".format(stats.pearsonr(cd.n_years, cd.rich_pooled)[0]))
print("  annual richness : r = {:.3f}".format(stats.pearsonr(cd.n_years, cd.rich_annual_mean)[0]))
""")

md(r"""
The left panel shows every district sitting above the 45 degree line. The right panel shows
why it matters: pooled richness climbs steeply with years of coverage while annual richness
is much flatter. 27 percent of districts have under 20 years of data, so a third of the ABI
was partly a coverage artefact.

Shannon and Simpson are far less affected, because they are share-based and a crop grown in
one year out of 23 contributes almost nothing to a pooled share. Their rank correlation
between pooled and annual is 0.97.
""")

md("### 1.4 Defect: duplicate rows are summed")

code(r"""
print(section("B2."))
""")

md(r"""
91 exact duplicates on (district, year, season, crop) survive the bogus-pair cleaning that
script 57 already does, mostly Niger Seed in Andhra Pradesh. `groupby().sum()` adds them
together, inflating area for those cells. It is 91 rows in 345,000 so it changes nothing
national, but it is wrong and cheap to fix.
""")

md("### 1.5 The corrected indices, and what moves")

code(r"""
print(section("What changes", corr_report))
""")

code(r"""
o = pd.read_csv(DIV + "/district_diversity_indices.csv")
c = pd.read_csv(DIV + "/district_diversity_indices_corrected.csv")
m = o.merge(c, on="district_key", suffixes=("_o", "_c"))
m["rank_o"] = m["agro_biodiversity_index_o"].rank(ascending=False)
m["rank_c"] = m["agro_biodiversity_index_c"].rank(ascending=False)
m["move"]   = (m["rank_o"] - m["rank_c"]).abs()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
big = m["move"] > 100
ax[0].scatter(m.loc[~big, "agro_biodiversity_index_o"], m.loc[~big, "agro_biodiversity_index_c"],
              s=9, alpha=.4, color=INK, linewidths=0)
ax[0].scatter(m.loc[big, "agro_biodiversity_index_o"], m.loc[big, "agro_biodiversity_index_c"],
              s=16, alpha=.9, color=ACCENT, linewidths=0)
ax[0].plot([0, 1], [0, 1], color="#8b949e", lw=1, ls="--")
ax[0].set(xlabel="Original ABI (pooled)", ylabel="Corrected ABI (annual mean)",
          title="Corrected vs original ABI, {} districts".format(len(m)))
ax[0].annotate("{} districts move more\nthan 100 rank places".format(int(big.sum())),
               xy=(.04, .88), xycoords="axes fraction", color=ACCENT, fontsize=9)

ax[1].hist(m["move"], bins=40, color=INK, alpha=.85)
ax[1].axvline(m["move"].median(), color=ACCENT, lw=1.8)
ax[1].annotate("median {:.0f}".format(m["move"].median()),
               xy=(m["move"].median(), ax[1].get_ylim()[1]*.86),
               xytext=(m["move"].median()+22, ax[1].get_ylim()[1]*.86),
               color=ACCENT, fontsize=9)
ax[1].set(xlabel="Absolute change in ABI rank", ylabel="Districts",
          title="How far districts move when the construction is fixed")
plt.tight_layout(); plt.show()
""")

md(r"""
The two versions correlate at r = 0.95, so the broad map of Indian agrobiodiversity is not
overturned. But 95 of 725 districts move more than 100 rank places, and the movement is not
random: Jharkhand falls from 18th among states to 25th, because its districts grew many
crops across the period without growing many in any one year. Karnataka stays first, so that
README claim holds.
""")

md(r"""
### 1.6 Is the ABI defensible? Mostly not

The repository's headline index is the equal-weighted mean of min-max normalised
Shannon, Simpson and richness. Three problems:

1. **It is sample-dependent.** Min-max rescales against whichever districts are in the
   file, so a district's score moves when *other* districts are added or removed.
2. **Equal weighting is arbitrary.** There is no argument for one third each.
3. **It double counts evenness.** Shannon and Simpson correlate at 0.94 here, so the
   composite is really two parts evenness to one part richness, not three equal parts.

The fix is the **Hill numbers**, the standard diversity family. All three share one unit,
the *effective number of crops*, meaning the number of equally-common crops that would
produce the observed diversity:

| index | definition | sensitivity |
|---|---|---|
| **D0** | crop richness | counts every crop equally, however tiny |
| **D1** | exp(Shannon) | weights each crop by its area share |
| **D2** | 1 / sum(p squared) | dominated by the common crops |
| **evenness** | D1 / D0 | how evenly area is spread, net of the count |

They are absolute, need no normalisation, and D0 >= D1 >= D2 always holds, which is a
free correctness check. Every result from here on is reported on all of them, with the
ABI kept only as a comparison row.
""")

code(r"""
print(section("Why the ABI is kept but demoted", corr_report))
""")

code(r"""
cc = pd.read_csv(DIV + "/district_diversity_indices_corrected.csv")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].scatter(cc["D0_richness"], cc["D1_exp_shannon"], s=10, alpha=.45,
              color=INK, linewidths=0)
ax[0].plot([0, cc["D0_richness"].max()], [0, cc["D0_richness"].max()],
           color="#8b949e", lw=1, ls="--")
ax[0].set(xlabel="D0, crops grown", ylabel="D1, effective crops",
          title="Districts grow about 21 crops, effectively about 5")
ax[0].annotate("45 degree line\n(perfectly even cropping)", xy=(.35, .80),
               xycoords="axes fraction", fontsize=8.5, color="#57606a")

o = cc["agro_biodiversity_index"]
ax[1].scatter(o, cc["D1_exp_shannon"], s=10, alpha=.45, color=INK, linewidths=0)
ax[1].set(xlabel="ABI (unitless composite)", ylabel="D1, effective crops",
          title="What the composite is standing in for")
r = stats.spearmanr(o, cc["D1_exp_shannon"], nan_policy="omit")[0]
ax[1].annotate("Spearman rho = {:.2f}".format(r), xy=(.05, .90),
               xycoords="axes fraction", fontsize=9)
plt.tight_layout(); plt.show()

print(cc[["D0_richness", "D1_exp_shannon", "D2_inv_simpson",
          "evenness_D1_D0", "agro_biodiversity_index"]].describe().round(3).to_string())
""")

md(r"""
The left panel is the substantive point. A district grows about 21 crops but is
effectively growing about 5. The gap between D0 and D1 *is* cropping concentration, and
it is a quantity with a unit that anyone can interpret. The ABI collapses that into a
0-to-1 score where 0.63 means nothing on its own.
""")

md(r"""
---
# Part 2. Building the SHRUG covariates

SHRUG 2.1 has no crop-area data at all. Every variable in all 174 files was searched and the
only hit was `two_crop_acre` from SECC. So the diversity outcome stays district-level from
the APY panel, and SHRUG supplies the right-hand side.

Four modules, all at village or shrid level, aggregated to PC11 district:

| module | what it gives |
|---|---|
| Mission Antyodaya 2019-20 | irrigation, seasonal sown area, markets, farmer counts |
| PC11 Village Directory | land use, irrigation by source, an independent 2011 check |
| PC11 Population Abstract | SC/ST population, cultivators vs agricultural labourers |
| SECC 2012 rural | land ownership, owned acres, caste shares |
""")

code(r"""
sh   = pd.read_csv(COV + "/shrug_district_covariates.csv")
prov = pd.read_csv(COV + "/provenance.csv")
print("covariate table: {} districts, {} columns".format(*sh.shape))
print("documented variables: {}".format(len(prov)))
print()
print(prov.groupby("module").size().rename("variables").to_string())
print()
prov[prov.module == "derived"].head(12)[["variable", "formula", "note"]]
""")

md("### 2.1 An error the checks caught")

md(r"""
The first build divided Antyodaya's `area_irrigated_in_hac` by gross cropped area. That was
wrong, and the validation caught it: Tarn Taran in Punjab came out at 0.29 when it is close
to fully irrigated.

Two pieces of evidence settle what the field means:

- Irrigated plus unirrigated reconstructs **net sown** area (ratio 1.15), not gross (0.66).
- Against published state figures, `irr / (irr + unirr)` gives mean absolute error 0.086 and
  bias +0.051. `irr / GCA` gives 0.181 and bias -0.129.

So `area_irrigated_in_hac` is **net** irrigated area, and the correct share is
`irr / (irr + unirr)`, which is internally closed and needs no other field. The wrong columns
are kept as `DEPRECATED_*` so the change stays auditable.
""")

code(r"""
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ok = sh["irr_share"].notna() & sh["irr_share_vd11"].notna()
ax.scatter(sh.loc[ok, "irr_share_vd11"], sh.loc[ok, "irr_share"], s=10, alpha=.45,
           color=INK, linewidths=0)
ax.plot([0, 1], [0, 1], color=ACCENT, lw=1.5)
ax.set(xlabel="PC11 Village Directory, 2011", ylabel="Mission Antyodaya, 2019",
       xlim=(0, 1), ylim=(0, 1),
       title="Two independent irrigation measures, eight years apart")
r = stats.spearmanr(sh.loc[ok, "irr_share"], sh.loc[ok, "irr_share_vd11"])[0]
ax.annotate("Spearman rho = {:.3f}\nn = {} districts".format(r, int(ok.sum())),
            xy=(.04, .84), xycoords="axes fraction", fontsize=9)
plt.tight_layout(); plt.show()
""")

md("### 2.2 The district crosswalk")

md(r"""
SHRUG sits on 2011 census boundaries; the diversity panel uses post-2011 names. **Telangana
does not exist in SHRUG** - all 33 of its current districts trace to 10 pre-2014 parents
filed under Andhra Pradesh. Post-2011 carve-outs add another 60 or so mismatches.

The crosswalk goes exact match, then an explicit parent map, then fuzzy matching within
state. Fuzzy auto-accepts at 0.86 and logs everything above 0.75 for review. Four false
positives were caught by eye and hard-blocked: Pauri Garhwal would have grabbed Tehri
Garhwal, Tirupathur would have grabbed Tiruppur, and two Garo Hills districts would have
grabbed the wrong sibling. Every manual target is asserted to exist in SHRUG before use.
""")

code(r"""
cw = pd.read_csv(COV + "/district_crosswalk.csv")
print(cw["method"].value_counts().to_string())
print("\nmatched {} of {} ({:.1%})".format(cw.shrug_key.notna().sum(), len(cw),
                                           cw.shrug_key.notna().mean()))
print("\ndiversity districts sharing one pre-2011 parent: {}".format(
      int(cw["n_sharing_parent"].fillna(1).gt(1).sum())))
print("\nunmatched: " + ", ".join(cw.loc[cw.method == "unmatched", "div_key"]))
""")

md(r"""
162 diversity districts share a parent, so they carry identical SHRUG covariates. That is
handled throughout by inverse-parent weighting and standard errors clustered on the parent,
and by a specification that drops them entirely.
""")

md("### 2.3 The validation battery")

code(r"""
vr = open(COV + "/validation_report.md", encoding="utf-8").read()
print(vr[vr.index("## Summary"):])
""")

md(r"""
Three warnings are worth knowing about and none is a blocker.

- **C5** Antyodaya cropping intensity only reaches rank agreement 0.65 against published
  state figures. Fine as a control, not something to report as a finding.
- **C7** is not a defect in our work. It is the finding that the scraped `irrigation_pct`
  the original analysis used agrees with the SHRUG measure at only rho = 0.66.
- **C8** 26 districts have Mission Antyodaya covering under a quarter of their villages,
  Tripura and Kerala worst. They are flagged and excluded, not silently averaged in.

One check deserves a note. **C9** compares our shrid-level aggregation against SHRUG's own
pre-aggregated district file. On *levels* they disagree by about 12 percent, because
`antyodaya_shrid.dta` covers 521,223 shrids carrying 857 million people while the district
file aggregates a wider village set carrying 1,007 million. On *ratios* they agree at
r = 0.992 to 0.996. **The rule that came out of this: use SHRUG-derived ratios, never levels.**
""")

md(r"""
---
# Part 3. Results

Everything here uses the **corrected** diversity indices, with **D1, the effective number
of crops**, as the primary outcome. Districts with under 10 years of data are excluded,
along with the thin-coverage districts. Results are shown on D0, D2 and evenness too, so
nothing depends on one index choice.
""")

code(r"""
fr = open(COV + "/final_results.md", encoding="utf-8").read()
print(fr[:fr.index("## 2.")])
""")

md("### 3.1 The shape")

code(r"""
fp = pd.read_csv(COV + "/final_panel.csv")
d  = fp[fp["in_final"]].copy()
d["dec"] = pd.qcut(d["irr_share"], 10, labels=False, duplicates="drop") + 1
g = d.groupby("dec").agg(irr=("irr_share", "mean"),
                         abi=("D1_exp_shannon", "mean"),
                         se=("D1_exp_shannon", lambda s: s.std()/np.sqrt(len(s))))

fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.errorbar(g["irr"], g["abi"], yerr=g["se"], fmt="o", color=INK, ms=6,
            lw=0, elinewidth=1.2, capsize=3, ecolor="#8b949e", zorder=3)
xs = np.linspace(d["irr_share"].min(), d["irr_share"].max(), 200)
cf = np.polyfit(d["irr_share"], d["D1_exp_shannon"], 2)
ax.plot(xs, np.polyval(cf, xs), color=ACCENT, lw=2, zorder=2)
tp = -cf[1] / (2 * cf[0])
ax.axvline(tp, color=ACCENT, lw=1, ls=":")
ax.annotate("peak at {:.0%} irrigated".format(tp), xy=(tp, ax.get_ylim()[0] + .012),
            xytext=(tp + .03, ax.get_ylim()[0] + .012), color=ACCENT, fontsize=9)
ax.set(xlabel="Share of sown area irrigated", ylabel="D1, effective number of crops",
       title="Effective crop diversity rises with irrigation, then falls")
plt.tight_layout(); plt.show()
""")

md("### 3.2 Does it survive being attacked?")

code(r"""
print(fr[fr.index("## 1."):fr.index("## 2.")])
""")

code(r"""
rb = pd.read_csv(COV + "/robustness.csv")
rb = rb[rb["p2"].notna()].copy()
rb["lab"] = rb["spec"].str.replace(r"^\d+\s+", "", regex=True)
rb = rb.iloc[::-1]

fig, ax = plt.subplots(figsize=(7.6, 5.0))
sig = rb["p2"] < 0.05
ax.scatter(rb.loc[sig, "b2"], np.arange(len(rb))[sig.values], s=42, color=INK, zorder=3)
ax.scatter(rb.loc[~sig, "b2"], np.arange(len(rb))[~sig.values], s=42,
           facecolors="none", edgecolors="#8b949e", linewidths=1.4, zorder=3)
ax.axvline(0, color="#8b949e", lw=1)
ax.set_yticks(np.arange(len(rb)))
ax.set_yticklabels(rb["lab"], fontsize=9)
ax.set(xlabel="Coefficient on irrigation squared",
       title="Every specification puts the squared term below zero")
ax.annotate("hollow = not significant at 5%", xy=(.42, .05), xycoords="axes fraction",
            fontsize=8.5, color="#57606a")
plt.tight_layout(); plt.show()
""")

md(r"""
**8 of 8 sample specifications** on D1 keep a negative squared term at p below 0.0001, and
**6 of 6 alternative indices agree** (D0, D2, evenness, raw Shannon, raw Simpson, and the
ABI). So the hump is neither a sample artefact nor an index artefact.

Two things worth carrying forward. First, the turning point is index-dependent: about 0.27
on D1 against 0.37 on the ABI, so quote a range rather than a point. Second, the hump shows
in evenness *on its own* (p = 0.013) as well as in the crop count, which says irrigation
changes both how many crops a district grows and how evenly area is spread across them.

The one row that failed on the original pooled indices was crop richness, at p = 0.19. On
corrected indices it shows the hump at p = 0.0001, so that null was an artefact of pooling
over years rather than a real result.
""")

md("### 3.3 Irrigation source, and the other four dimensions")

code(r"""
print(fr[fr.index("## 3."):])
""")

md(r"""
Holding the level of irrigation and its square constant, with state fixed effects and
groundwater as the omitted source:

- **Surface water** is strongly bad for diversity. ABI -0.168, Shannon -0.543, cereal share
  +0.300, all at 1 percent.
- **Canal against groundwater** cuts crop richness by 2.6 crops and nudges ABI down, but
  leaves Shannon alone and *raises* pulse share.

So "canals force monoculture" is too simple. Canals narrow the crop list. Surface-water
dependence is what concentrates area onto cereals.

On the other dimensions: FPO presence is positive and holds up; regular-market access is
negative raw but does not survive adjustment; mandi presence is null once state and
irrigation are in; SC population share is positive and robust; ST share is null once
controlled; and **mean holding size is null throughout**, so the inverse farm-size-diversity
relationship does not appear here. That last one I read as much as a warning about the
Antyodaya holding proxy as a result.
""")

md(r"""
---
# Part 4. The market layer

Market access does not have one effect on crop diversity. It has two opposite ones, and
which wins depends on what the market is **for**.

- A **regulated mandi** exists to move assured-price cereals in bulk.
- A **weekly haat** clears small lots of perishables: vegetables, fruit, spices, minor
  millets. It is the market a diverse smallholder actually uses.
- A **fertiliser shop** is the retail end of the purchased-input package that travels with
  cereal intensification. It marks input-market penetration, not output-market access.
- **Cold storage and farm-gate processing** make perishables sellable at all.

Every adjusted model below carries log nightlights, non-farm establishment density, a
connectivity index, log mean holding, ST share, irrigation and its square, and state fixed
effects, because all of this infrastructure gets built where there is already surplus to
trade.
""")

code(r"""
mkt = open(COV + "/market_analysis.md", encoding="utf-8").read()
print(section("A. What exists where", mkt))
""")

md(r"""
### The haat is a different animal

Read the correlation matrix above carefully. Every facility correlates 0.4 to 0.83 with
every other one, so they are essentially one "agri-commercial infrastructure" factor. Except
the weekly haat, which correlates **negatively** with mandis (-0.15), regular markets (-0.21)
and custom hiring (-0.09), near zero with nightlights (+0.04), and negatively with non-farm
establishment density (-0.16).

That is not a nuisance in the data. It says the haat is a genuinely different institution
rather than a lesser version of a mandi, and it is what makes the layer worth running.
""")

code(r"""
mv = pd.read_csv(COV + "/market_typology.csv")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].scatter(mv["m_mandi_vshare"], mv["m_weekly_haat_vshare"], s=11, alpha=.5,
              color=INK, linewidths=0)
ax[0].set(xlabel="Mandi village share", ylabel="Weekly haat village share",
          title="Mandis and haats do not go together")
r = stats.pearsonr(mv["m_mandi_vshare"], mv["m_weekly_haat_vshare"])[0]
ax[0].annotate("r = {:+.2f}".format(r), xy=(.75, .90), xycoords="axes fraction", fontsize=9)

ax[1].scatter(mv["m_fert_shop_vshare"], mv["m_weekly_haat_vshare"], s=11, alpha=.5,
              color=INK, linewidths=0)
ax[1].set(xlabel="Fertiliser shop village share", ylabel="Weekly haat village share",
          title="Nor do haats and input supply")
r2 = stats.pearsonr(mv["m_fert_shop_vshare"], mv["m_weekly_haat_vshare"])[0]
ax[1].annotate("r = {:+.2f}".format(r2), xy=(.75, .90), xycoords="axes fraction", fontsize=9)
plt.tight_layout(); plt.show()
""")

md("### What each facility does, entered on its own")

code(r"""
print(section("F. Checks on the market layer", mkt)[:2600])
""")

code(r"""
solo = pd.read_csv(COV + "/market_results_solo.csv")
piv = solo.pivot(index="facility", columns="outcome", values="b")
pp  = solo.pivot(index="facility", columns="outcome", values="p")

fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.6))
for k, (col, lab) in enumerate([("crop_richness", "Effect on crop richness"),
                                ("share_pulses", "Effect on pulse share")]):
    o = piv[col].sort_values()
    sig = pp.loc[o.index, col] < 0.05
    y = np.arange(len(o))
    ax[k].scatter(o[sig.values], y[sig.values], s=46, color=INK, zorder=3)
    ax[k].scatter(o[~sig.values], y[~sig.values], s=46, facecolors="none",
                  edgecolors="#8b949e", linewidths=1.4, zorder=3)
    ax[k].axvline(0, color="#8b949e", lw=1)
    ax[k].set_yticks(y); ax[k].set_yticklabels(o.index, fontsize=9)
    ax[k].set(title=lab, xlabel="Coefficient")
ax[0].annotate("hollow = not significant at 5%", xy=(.30, .04),
               xycoords="axes fraction", fontsize=8.5, color="#57606a")
plt.tight_layout(); plt.show()
""")

md(r"""
Two things come out of this, and both survive entering all ten facilities together.

**The haat goes with more crops, but not with more effective crops.** D0 rises +7.3 per
unit of haat village share, stable from +6.9 to +8.0 across every specification including
one carrying the other nine facilities. But D1 barely moves (+0.40, p = 0.63) and evenness
is *negative* (-0.071, p = 0.05). Haat districts grow more crops, on small patches, around
the same dominant staple. Two honest caveats on magnitude: one standard deviation of haat
share is 0.117, so this is about **0.9 extra crops on a mean of 21**, and it is a change in
the count rather than in the balance. An index that averages richness and evenness together
would have reported this as "haats make districts diverse", which is the strong version and
is not what the data says.

**Commercial and input infrastructure goes with fewer pulses.** Fertiliser shops -0.174,
cold storage -0.172, regular markets -0.094, all at 1 percent, and the fertiliser-shop
result survives a Benjamini-Hochberg correction across all ten crop categories. Pulses are
the crop that gets displaced when the purchased-input economy arrives, which is the one
result here that lines up cleanly with what the natural-farming literature would predict.

**FPOs run the opposite way to haats.** They lift D1 (+2.39, p < 0.05) without lifting D0
(+0.47, ns) and cut cereal share (-0.222, p < 0.01). So they go with area spread more evenly
across the crops a district already grows, rather than extra crops at the margin. Haats add
crops; FPOs rebalance area. Only an index family that separates count from evenness can
tell those two apart, which is the practical case for dropping the composite.
""")

md("### Two hypotheses that failed")

code(r"""
print(section("D. Does market infrastructure explain the irrigation downslope?", mkt))
print(section("F4. What did not hold", mkt) if "F4" in mkt else "")
""")

md(r"""
Worth stating plainly rather than burying.

1. **Mandis do not push districts toward cereals.** The coefficient is negative (-0.240),
   so within a state and at a given level of irrigation, denser mandi networks go with
   *fewer* cereals. The opposite of the hypothesis.
2. **Infrastructure does not steepen the irrigation downslope.** Every interaction in
   section D is positive, meaning the curve is *flatter* where mandis and fertiliser shops
   are dense, and only the mandi one approaches significance at p = 0.059. The Punjab story,
   that assured procurement plus dense input supply is what turns high irrigation into
   monoculture, does not show up in this cross-section.
3. **The market typology adds nothing.** Raw differences across the four types look large
   (ABI 0.53 to 0.68) and every one of them vanishes once the controls go in.

On collinearity: the output-market block is clean, with all variance inflation factors under
2. The input-supply block runs 3.4 to 4.1, below the usual warning line of 5 but high enough
that the joint coefficients in section C should be read alongside the one-at-a-time table.
""")

md(r"""
---
# Part 5. What not to claim

Written down so nobody has to rediscover it.

1. **This is descriptive.** Nothing here is identified. Irrigation is not randomly assigned
   and everything correlates with agro-ecology. The quadratic is a shape, not a dose-response
   curve.
2. **The published state benchmarks in the validation script are entered from memory** of the
   DES Land Use Statistics tables. They are used for rank agreement only, never level
   matching, and must be re-checked against the published source before they appear anywhere.
3. **Mission Antyodaya over-reports irrigation** in Maharashtra (0.54 against a published
   0.21), Jharkhand (0.45 against 0.12) and Assam (0.35 against 0.11). It is self-reported
   panchayat data. Dropping those three states does not change the result, but their levels
   are not trustworthy.
4. **162 districts share a pre-2011 parent** and carry identical covariates. Always cluster
   on the parent.
5. **SHRUG levels are not usable here, only ratios.** The shrid file undercovers villages by
   about 15 percent relative to SHRUG's own district aggregation.
6. **Antyodaya is a single 2019-20 cross-section** matched to diversity averaged over
   1997-2020. The timing is defensible for slow-moving irrigation infrastructure and not for
   anything you want to call a change.
7. **The repository still says 24 years and 1997-2021 in several places.** It is 23 usable
   years, 1997-98 to 2019-20.
8. **Market infrastructure is not randomly placed.** Mandis, cold stores and fertiliser
   shops get built where there is surplus to trade, and the causality plausibly runs both
   ways with cropping pattern. The development controls narrow that gap and do not close it.
9. **The haat effect is robust but small, and it is about counts not balance.** One
   standard deviation buys about 0.9 crops on a mean of 21, on D0 only. D1 is flat and
   evenness falls. Quote the standardised magnitude and say which index it is on.
10. **Do not quote the ABI.** It is sample-dependent, arbitrarily weighted, and two of its
   three components correlate at 0.94. Use D0, D1, D2 and evenness, which carry a unit.
11. **The turning point is index-dependent**, about 0.27 on D1 and 0.37 on the ABI. Quote
   a range, not a point estimate.
""")

code(r"""
files = {
 "district_diversity_indices_corrected.csv": DIV + "/district_diversity_indices_corrected.csv",
 "district_year_diversity_panel_corrected.csv": DIV + "/district_year_diversity_panel_corrected.csv",
 "shrug_district_covariates.csv": COV + "/shrug_district_covariates.csv",
 "final_panel.csv":              COV + "/final_panel.csv",
 "district_crosswalk.csv":       COV + "/district_crosswalk.csv",
 "provenance.csv":               COV + "/provenance.csv",
}
print("{:<46} {:>7} {:>6}".format("file", "rows", "cols"))
print("-" * 62)
for n, p_ in files.items():
    if os.path.exists(p_):
        t = pd.read_csv(p_)
        print("{:<46} {:>7} {:>6}".format(n, len(t), t.shape[1]))
print()
for r_ in ["diversity_construction_audit.md", "corrected_vs_original.md",
           "validation_report.md", "final_results.md", "robustness.md", "findings.md"]:
    print("report: outputs/shrug_covariates/" + r_)
""")

nb["cells"] = C
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
os.makedirs(os.path.dirname(OUTNB), exist_ok=True)
with open(OUTNB, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("WROTE {} with {} cells".format(OUTNB, len(C)))
