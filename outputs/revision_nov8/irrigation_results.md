# Irrigation panel: within-district irrigation and crop diversity, 1998-2019

## Status after referee (Nov 2026)

Written 7 November 2026. Referee: `irrigation_referee_raw.md` (script 108). Site replacement: commit bcf97f7, `docs/data/site_irrigation_v2.json`, `irrigation_v2/referee/referee_raw_repro_infer_q1x_q2_q3_q4_mt.md`.

- Cross-sectional inverted U with a peak at 0.273 (LUS) "matching the site's 0.279": **superseded**. The site no longer reports a turning point. Within state the effective number of crops is flat across the less irrigated districts and falls once irrigation passes about half of sown area: fifths of irrigated share against the least irrigated fifth **+0.12 (SE 0.16), -0.55 (0.36), -1.05 (wild p 0.072), -2.14 (wild p 0.013)**; a two-piece line with a knot at 0.30 has slope **+2.91 below (p 0.119)** and **-4.10 per unit above (p 0.022)**, about -0.4 crops per 10 points. None of the first seven deciles differs significantly from the first. The squared term survives every check (-5.44 to -8.18); the rising arm, and with it the interior peak, does not. Corrected in place below.
- Decomposition of the within curvature: irr x baseline -3.726 (0.934) -> **-3.673 (0.945)**; own-path curvature -1.266 (0.636), p = 0.047 -> **-1.061 (0.731), p = 0.148**, not statistically significant (`irrigation_referee_raw.md` R2). Most of the within hump is the cross-sectional hump reappearing as slopes that differ by baseline irrigation.
- "A within-district quadratic is significant for every index": not under state clustering (R7). D1 squared term **p 0.105**, D2 0.163, evenness 0.291; D0 0.017 and D0 at 0.5 percent 0.003 stay significant.
- Fall in the count of main crops (D0 at 0.5 percent) within districts, -0.815 (0.290): **-0.130 (0.271), p 0.633** once each district has its own linear trend (R5). Refereed-weakened; the long difference (-2.52) is unchanged but not on the page.
- Unchanged: the within linear slope of D1 is a precise zero (-0.225, SE 0.192); no instrument works; the lag-one-year version weakens the curvature to -0.638 (0.416).

The original analysis below is kept unchanged for the record, apart from the sentences marked [corrected Nov 2026]. Where it disagrees with this block, this block and `outputs/ANALYSIS_INDEX.md` hold.

---

Scripts: `scripts/104_irrigation_panel_build.py` (panel), `105_irrigation_panel_estimates.py`
(estimates), `106_irrigation_iv_feasibility.py` and `107_irrigation_iv_recentred_cgwb.py`
(instruments), helper `_revision_fe.py`. Raw tables: `irrigation_panel_results_raw.md`,
`irrigation_iv_raw.md`, `irrigation_iv2_raw.md`, plus the CSVs beside them. 105 was re-run
for this note and reproduces its raw table exactly.

## 1. Data

**Irrigation.** DES Land Use Statistics, district-wise, from the India Data Portal CKAN
package `land-use-statistics` (resources "Sources of Irrigation" and "Classification of
Area", raw copies in `raw_lus/`). It gives net irrigated area (NIA) by source (canal, tank,
tubewell, other well, other), gross irrigated area, net sown area (NSA) and gross cropped
area, 1998-99 to 2023-24. The share used is NIA / NSA, the same concept as the site's
Antyodaya irr / (irr + unirr). GIA / GCA is kept as a robustness variant. ICRISAT DLD and
state abstracts were not needed; the APY file has no irrigated-area column.

**Diversity.** Rebuilt per district-year from the raw APY file with script 76's cleaning.
D0, D1, D2, evenness D1/D0, cereal share, plus D0_05 (crops holding at least 0.5 percent of
the district-year's cropped area). D0 >= D1 >= D2 holds in every district-year.

**Matching.** APY districts to LUS districts by normalised name within state (669 exact,
29 fuzzy at difflib >= 0.85, 8 across the Telangana and Ladakh splits, 19 unmatched);
2011 parents from the existing crosswalk.

**Coverage (honest version).**

| | value |
|---|---|
| analysis districts (of 606) with any LUS irrigation | 600 |
| analysis district-years with diversity, 1997-2019 | 13,233 |
| of which with irrigation | 11,099 (83.9 percent) |
| 1997-98 | no LUS data at all |
| states with under half their district-years covered | West Bengal 4%, Manipur 9%, Maharashtra 21%, Sikkim 0%, Andaman and Nicobar 0% |
| district-years flagged as irrigation spikes | 100 |
| district-years with an NSA territory break / APY area break | 334 / 1,247 |

Agreement with the site's measure: LUS 2017-19 mean against Antyodaya 2019-20, r = 0.871
(n = 600; means 0.528 and 0.546). LUS 1998-2019 mean against Antyodaya, r = 0.847.
Summed over matched districts, national NIA/NSA rises from 0.373 (1998) to 0.564 (2019).

Within-district variation is thin. Irrigation share has mean 0.477 and SD 0.311; the
within-district SD is 0.089 (8 percent of the variance), and after district and
state-by-year fixed effects the residual SD is 0.068. 142 of 600 districts never leave one
10-point bin.

## 2. Method

Quadratic in NIA/NSA. Three nested specifications: pooled with state-by-year FE (between
plus within), district FE plus year FE, and the main one, district FE plus state-by-year FE.
Standard errors clustered on shared 2011 parents (599 clusters pooled, 581 in the FE
specification after singletons drop). Turning points by the delta method. Then a binned
version (10-point bins, reference 0-10 percent), long differences between 1998-2000 and
2017-2019 with state FE, and two families of instruments.

## 3. Headline numbers (D1, effective number of crops, unless stated)

### Cross-section on the same districts, state FE

| measure | n | irr | irr^2 | peak (se) |
|---|---|---|---|---|
| Antyodaya 2019-20 (site) | 606 | +3.811 (1.408) | -6.836 (1.379) | 0.279 (0.054) |
| LUS 1998-2019 mean | 600 | +3.455 (1.104) | -6.322 (1.053) | 0.273 (0.047), 95% CI 0.18 to 0.37 |

The new data reproduce the site's cross-sectional hump almost exactly.

### Panel, quadratic

| spec | outcome | n | irr | irr^2 | p(irr^2) | peak (se) |
|---|---|---|---|---|---|---|
| pooled, state x year FE | D1 | 11,092 | +3.552 (0.956) | -6.116 (0.909) | <0.0001 | 0.290 (0.041) |
| district FE + year FE | D1 | 11,081 | +2.756 (0.586) | -2.860 (0.493) | <0.0001 | 0.482 (0.038) |
| **district FE + state x year FE** | **D1** | **11,074** | **+1.636 (0.537)** | **-1.777 (0.464)** | **0.0001** | **0.460 (0.057)** |
| district FE + state x year FE | D0 | 11,074 | +1.527 (0.931) | -2.199 (0.910) | 0.016 | 0.347 (0.109) |
| district FE + state x year FE | D0_05 | 11,074 | +2.322 (0.711) | -2.995 (0.627) | <0.0001 | 0.388 (0.056) |
| district FE + state x year FE | D2 | 11,074 | +1.463 (0.483) | -1.327 (0.422) | 0.002 | 0.551 (0.059) |
| district FE + state x year FE | evenness | 11,074 | +0.052 (0.028) | -0.047 (0.023) | 0.043 | 0.560 (0.099) |
| district FE + state x year FE | cereal share | 11,074 | -0.090 (0.036) | +0.114 (0.032) | 0.0003 | 0.396 (0.076) |

Main D1: 95 percent CI on irr^2 is -2.69 to -0.87; on the peak 0.35 to 0.57. Districts 582,
clusters 581.

Size. The within curvature is a third of the cross-sectional one (-1.78 against -6.32).
The height of the within hump, from zero irrigation to the peak, is 0.38 effective crops on
a mean of 4.9. The **linear** within slope is -0.225 (0.192), p = 0.24, 95% CI -0.60 to
+0.15 per unit of share, so a 10-point rise in irrigation moves D1 by between -0.06 and
+0.02 crops. Linear slopes for the other indices: D0 -0.776 (0.465), D0_05 -0.815 (0.290,
p = 0.005), D2 +0.073 (0.161), evenness +0.003 (0.010).

### Binned, district FE + state x year FE (reference 0-10 percent)

| bin | n | districts moving through | D1 | D0_05 |
|---|---|---|---|---|
| 10-20 | 1,329 | 197 | +0.169 (0.068) | +0.15 (0.09) |
| 20-30 | 1,165 | 218 | +0.376 (0.088) | +0.51 (0.13) |
| 30-40 | 1,162 | 215 | +0.476 (0.118) | +0.44 (0.15) |
| 40-50 | 1,046 | 192 | +0.388 (0.125) | +0.35 (0.17) |
| 50-60 | 863 | 187 | +0.312 (0.129) | +0.15 (0.18) |
| 60-70 | 956 | 171 | +0.135 (0.140) | +0.13 (0.20) |
| 70-80 | 731 | 164 | +0.128 (0.137) | -0.06 (0.22) |
| 80-100 | 2,389 | 133 | +0.058 (0.143) | -0.26 (0.22) |

The binned profile rises to the 30-40 percent bin and falls back to the reference level by
60 percent. No bin is significantly below the driest one for D1, so the within downslope
returns diversity to where it started; it does not push it lower.

### Long differences, 1998-2000 to 2017-2019, state FE

407 districts have both windows (406 in the regressions). Mean change in NIA/NSA +0.117
(SD 0.163); 179 districts rose by more than 10 points and 16 fell by more than 10.

| outcome | n | dIrr | d(Irr^2) | p(quad) | linear dIrr (se), p |
|---|---|---|---|---|---|
| D1 | 406 | -0.663 (1.499) | +0.195 (1.214) | 0.87 | -0.435 (0.440), 0.32 |
| D0 | 406 | +1.950 (2.558) | -3.768 (1.958) | 0.055 | -2.445 (1.105), 0.027 |
| D0_05 | 406 | -0.482 (2.118) | -1.747 (1.696) | 0.30 | -2.520 (0.777), 0.001 |
| D2 | 406 | -0.479 (1.333) | +0.552 (1.088) | 0.61 | +0.166 (0.370), 0.66 |
| evenness | 406 | -0.060 (0.072) | +0.067 (0.056) | 0.23 | +0.017 (0.022), 0.43 |

No inverted U over twenty years for any index. D1 by baseline irrigation group: below 20
percent -0.43 (1.44, n = 89); 20-40 percent +1.92 (1.40, n = 94); 40-60 percent -2.12
(1.09, p = 0.054, n = 72); 60 percent and above -0.48 (0.77, n = 137). The signs follow the
hump; none is precise. Dropping districts with an NSA territory break (n = 288) leaves the
D0_05 slope at -2.48 (0.93, p = 0.008) and everything else null.

## 4. Robustness of the main within quadratic (D1)

| variant | n | irr^2 (se) | p | peak (se) | linear slope (se) |
|---|---|---|---|---|---|
| main | 11,074 | -1.777 (0.464) | 0.0001 | 0.460 (0.057) | -0.225 (0.192) |
| drop irrigation spikes | 10,973 | -1.736 (0.497) | 0.0005 | 0.398 (0.070) | -0.429 (0.209) |
| district-spell FE (split at NSA breaks) | 10,965 | -1.735 (0.469) | 0.0002 | 0.494 (0.053) | -0.145 (0.187) |
| drop APY area breaks | 10,007 | -1.344 (0.474) | 0.005 | 0.403 (0.082) | -0.338 (0.185) |
| 3-year centred mean of irrigation | 11,073 | -1.561 (0.572) | 0.007 | 0.362 (0.095) | -0.509 (0.240) |
| **irrigation lagged one year** | 10,520 | **-0.638 (0.416)** | **0.13** | 0.235 (0.218) | -0.366 (0.172) |
| GIA / GCA | 10,036 | -2.781 (0.582) | <0.0001 | 0.461 (0.054) | -0.170 (0.299) |
| >= 15 irrigation years | 10,089 | -1.956 (0.520) | 0.0002 | 0.476 (0.055) | -0.238 (0.202) |
| cropped-area weighted | 11,074 | -2.508 (1.048) | 0.017 | 0.538 (0.058) | -0.106 (0.293) |
| drop shared 2011 parents | 9,929 | -1.902 (0.510) | 0.0002 | 0.486 (0.057) | -0.166 (0.213) |
| fixed denominator (NIA / own mean NSA) | 11,074 | -0.842 (0.369) | 0.023 | 0.584 (0.107) | |
| plus rainfall z-scores (JJAS, OND, next JFMAM) | 11,017 | -1.819 (0.468) | 0.0001 | | |
| D0, spell FE, drop spikes | 10,871 | -1.075 (0.987) | 0.28 | 0.426 (0.277) | -0.229 (0.602) |

Where the curvature comes from (D1, main specification):

- Decomposing into a baseline-dependent slope and curvature around a district's own
  baseline: irr x baseline -3.726 (0.934, p < 0.001), own-path curvature -1.266 (0.636,
  p = 0.047). Most of the within "hump" is the slope differing across districts by how
  irrigated they already were, which is the cross-sectional hump seen again in changes.
- Splitting within variation into a slow part (7-year mean, SD 0.046) and a fast part
  (deviation, SD 0.046): the slow quadratic is null, irr_slow^2 -1.205 (0.724, p = 0.097),
  peak 0.233 (0.216). The fast part carries it: irr_fast +2.260 (0.492), fast x baseline
  -4.135 (0.805).
- Next year's irrigation added as a lead: lead -0.670 (0.389, p = 0.085), lead^2 +0.198
  (0.335); the contemporaneous terms are unchanged.

## 5. Identification

### (a) Duflo-Pande river gradient: not feasible here

SHRUG 2.1 holds elevation (mean, median, percentiles, SD) and terrain ruggedness at
pc11-district, subdistrict and shrid level. It has no river network, no dam locations and
no gradient-within-basin measure. The nearest thing on disk, terrain x trend, was tried as
a stand-in: first-stage cluster-robust F = 1.57 (n = 9,736, 575 districts). Too weak to use.

Plan for the real instrument:
1. DEM: SRTM 90 m (CGIAR-CSI) or HydroSHEDS conditioned DEM; compute slope per cell.
2. Rivers: HydroSHEDS HydroRIVERS (or WRIS river layer); keep cells within a buffer of
   rivers of order >= 3.
3. District gradient shares on 2011 polygons: fraction of river-adjacent area in slope
   classes 0-1.5, 1.5-3, 3-6 and above 6 percent, as in Duflo and Pande (2007).
4. Dams: CWC National Register of Large Dams (completion year, location, purpose), so
   upstream and own-district dam counts vary by year; instrument = gradient shares x
   state-level dam count in each year, with state x year FE.
5. Outcome window: NRLD completions after 1998 are fewer than in the DP period, so the
   first stage must be checked before anything else.

### (b) Groundwater: data obtained, instrument not credible

CGWB well readings (India Data Portal `groundwater` package, raw copy in `raw_cgwb/`) cover
2013 to 2021 only. 18,684 stations, placed in 2011 districts by coordinates; 521 districts
have a pre-monsoon depth (median 7.4 m, p10 4.1, p90 12.7). Log depth in 2013-15 x trend,
district FE and state x year FE:

| instrument | n | first stage | F | D1 2SLS (se) |
|---|---|---|---|---|
| depth x trend | 8,100 | -0.0072 (0.0042) | 2.87 | -10.77 (8.15) |
| depth x trend and square | 8,100 | | 6.47 | -3.11 (2.21) |
| depth x trend, baseline irrigation x year controls | 7,032 | -0.0075 (0.0043) | 3.02 | -10.45 (7.39) |

Weak in every version, and invalid by construction: depth in 2013 is partly the result of
the 1998-2013 pumping it would be meant to predict. A usable version needs pre-period
depth (CGWB yearbooks or India-WRIS station series from the 1990s) or a fixed
hydrogeology layer (CGWB/NAQUIM principal aquifer type, alluvial against hard rock),
interacted with a pump-cost shifter that varies over time: state agricultural feeder
separation dates (Gujarat Jyotigram 2003-06 and later copies), farm tariff or free-power
episodes, or national diesel price for diesel-pump districts.

### (c) Shift-share on irrigation source: implemented, fails

Z = sum over sources of the district's 1998-2000 source share of NSA times leave-state-out
growth of that source (tubewell area grew 1.51x nationally, canal 1.01x, tank 0.80x).
Estimated 2001-2019 so the baseline years are not used (n = 8,073, 463 districts, 463
clusters; districts formed after 2000 have no baseline).

| version | first stage (se) | F | D1 2SLS (se) |
|---|---|---|---|
| just identified, no baseline control | -0.213 (0.056) | 14.30 | +2.97 (2.44) |
| recentred: baseline irrigation x year controls | -0.139 (0.074) | 3.53 | -1.17 (5.04) |
| recentred, Z and Z^2 | | 13.86 | quadratic irr^2 +6.50 (16.02) |
| source composition only, baseline x year controls | -0.011 (0.050) | 0.05 | uninformative |

The uncontrolled first stage has the wrong sign: districts whose baseline sources grew more
nationally saw their irrigation share fall against state peers. Z correlates 0.92 with
baseline tubewell share, so it is really baseline irrigation times time, which inside
district FE picks up convergence toward a ceiling. Once baseline irrigation x year is
controlled the instrument is weak (F 3.5), and the pure source-mix version has no first
stage at all (F 0.05). None of the 2SLS estimates is informative.

**Bottom line on identification:** no instrument available in this session has a credible
first stage and exclusion restriction. Every within-district number above is a
conditional correlation. Irrigation in these districts expands alongside roads, markets,
power, credit and state crop programmes that also act on the crop mix; state x year FE
remove only the state-wide part of those.

## 6. What holds and what does not

Holds:
- The cross-sectional inverted U, on new and independent irrigation data. LUS
  1998-2019 means give a peak of 0.273 (0.047), close to the 0.279 then quoted from Antyodaya. [corrected Nov 2026: the site no longer reports a turning point; see the status block.]
- A within-district quadratic is significant for every index in the main specification
  and survives spikes, territory splits, area breaks, weighting, the GIA/GCA measure,
  rainfall controls and dropping shared parents.
- The fall in the count of crops above 0.5 percent of area as irrigation rises: within
  linear slope -0.815 (0.290) and long-difference slope -2.52 (0.78), n = 406. Irrigation
  expansion thins the list of crops that hold a meaningful area.

Does not hold, or holds only weakly:
- The within hump peaks later than the cross-section: 0.46 (0.057) for D1 against 0.27.
  Within estimates span 0.35 (D0) to 0.56 (D2, evenness).
- It is a third the curvature of the cross-section, and the linear within slope of D1 is
  a precise zero (a 10-point rise moves D1 by between -0.06 and +0.02 crops).
- It is fragile to timing. Lagging irrigation one year cuts curvature to -0.64 (0.42,
  p = 0.13); the slow, multi-year component of irrigation carries no significant hump; the
  curvature lives in year-to-year co-movement of the irrigated share and the crop mix, which
  is where shared reporting noise between LUS and APY (both state agricultural statistics)
  and fixed-denominator effects would also sit. The fixed-denominator version halves it.
- Long differences show no inverted U for D1, D2 or evenness (p 0.23 to 0.87).
- The binned within profile rises then returns to its starting level; it never falls
  significantly below the driest bin.
- No instrument works.

## 7. Plain findings

1. The district irrigation panel built from official land-use statistics reproduces the
   site's cross-sectional pattern, with diversity highest around a quarter to a third of
   net sown area irrigated.
2. Inside a district, over two decades, irrigation gains are associated with very small
   changes in the effective number of crops. The year-on-year hump is real in the data but
   a third the size of the cross-section, peaks nearer half irrigated, and fades when
   irrigation is lagged or smoothed or measured over long differences.
3. The one change that shows up both within districts and over twenty years is a shorter
   list of crops that each hold a meaningful share of area as irrigation spreads, about 2.5
   fewer such crops for each full unit of share (a quarter of a crop per 10 points).
4. None of this is causal. The irrigation-source shift-share and the groundwater-depth
   instrument both fail their first stage; a river-gradient design needs a DEM, a river
   network and the dam register, none of which is in hand.
