# Markets page, second revision: results

Scripts: `scripts/136_market_census_panel.py`, `scripts/137_market_haat_mechanism.py`, shared tools in `scripts/_mkt2_common.py`. Raw logs: `census_market_panel_log.md`, `haat_mechanism_log.md` (this folder). Site data: `docs/data/site_markets_v2.json`. Inference throughout: SE clustered on the 2011 parent district; state clustering with t(G-1); restricted wild cluster bootstrap by state, Webb weights, 9,999 draws.

## 1. What the "producer organisation" variable measures

SHRUG label of `fpo`: "Are there any Farmer produce organizations?". The underlying Mission Antyodaya 2019-20 item, from the MoRD "Mission Antyodaya Data Download" supporting document (data column mapping, question 12, variable `availability_of_fpos_pacs`):

> 12. Does the village has any Farmers Collective. Farmers Produce Organization (FPOs) = 1, Primary Agriculture Cooperative Society (PACS) = 2, Both = 3, None = 4.

Source PDF: data.opencity.in, dataset "Mission Antyodaya Data 2020", resource "Mission Antyodaya Supporting Document 2020" (https://data.opencity.in/dataset/6cd880b6-12ca-4b18-a4db-720f34fc249a/resource/28863e17-f537-4bd0-8305-7cb5e66c0c58/download/ac75e1ca-6f85-4f6b-a4a6-b70b2763fd94.pdf).

SHRUG's 0/1 `fpo` equals 1 for codes 1 to 3. Check against the raw MA 2020 village file for Pune (1,901 villages, same portal): codes 1/2/3/4 = 230/148/238/1,285, so codes 1 to 3 = 32.4 percent of villages; SHRUG `fpo` for Pune (PC11 district 521) = 32.6 percent. The same file confirms the market item (question 71, `availability_of_market`) is a single answer: mandi 1, regular market 2, weekly haat 3, none 4 (Pune 50/65/267/1,519; SHRUG 2.3/3.0/14.3 percent against 2.6/3.4/14.0 raw).

Across the 606 districts the MA "farmer collective" reach correlates 0.63 with Census 2011 agricultural credit society reach (means 0.279 and 0.267), higher than any other MA facility (fertiliser shop 0.60 next, weekly haat -0.01). With the census credit-society measure in its place the D1 coefficient is +2.673 (SE 0.696; state-clustered p 0.082; wild p 0.140) against +1.886 (SE 0.790; state p 0.167; wild p 0.182) for the MA item. D0: +4.218 (wild p 0.098) against +3.380 (wild p 0.105).

Renamed on the site to **farmer collective** (a producer organisation or a primary agricultural credit society); sections 01 and 09 say what the question asks, and the method drawer quotes it.

## 2. Census market panel

Comparability (SHRUG village directories):

| round | market items | credit items |
|---|---|---|
| 1991 | `pc91_vd_mrkt_fac` market facility; market day Monday to Sunday, daily, fortnightly, monthly | none used |
| 2001 | **none** | `pc01_vd_ac_soc` number of agricultural credit societies |
| 2011 | `pc11_vd_wkl_haat` weekly haat; `pc11_vd_mrkt` mandis/regular market (one item) | `pc11_vd_acs` agricultural credit societies |
| MA 2019-20 | one answer per village: mandi, regular market, weekly haat or none | farmer collective (FPO or PACS) |

A 2001-2011 haat or mandi panel cannot be built, because the 2001 directory has no market item. Pairs used: weekly market 1991 (a named weekday market day) against weekly haat 2011; daily market 1991 against mandi/regular market 2011 (control only); credit societies 2001 against 2011. Fixed Census 2011 village population weights, balanced village set, district needs 75 percent of its 2011 rural population covered.

In 1991, 85.7 percent of villages with a market have a weekday market day and 13.2 percent a daily market; day fields are missing for 20.9 percent of flagged rows. District means: weekly market 0.132 (1991) to 0.251 (2011), r = 0.48 across 435 districts; daily 0.033 to 0.257 (r 0.37; not comparable in level); credit societies 0.263 to 0.281 (r 0.83, 568 districts). 1991 market coverage is below 0.75 for Gujarat, Haryana, Maharashtra, Odisha, Assam, Jammu and Kashmir, Arunachal Pradesh. In 2011, 46 percent of haat villages (7.3 of 15.9 percent) also have a mandi or regular market, which the single-answer MA item cannot record; MA and census 2011 haat reach correlate only 0.43 across the 606.

First differences, 1998-99..2002-03 to 2009-10..2013-14, state FE, control for census irrigation change 2001-2011 (338 districts, 17 states):

| change in | haat b (SE) | p parent / state / wild | one SD (15 pts) with 95% CI | daily b | credit societies (431 districts, 22 states) |
|---|---|---|---|---|---|
| D0 | +1.004 (1.183) | 0.396 / 0.442 / 0.433 | +0.15 (-0.19 to +0.48) | -0.750 (p 0.475) | -1.071 (p 0.275) |
| D0 at 0.5% | +0.246 (0.670) | 0.714 / 0.735 / 0.728 | +0.04 (-0.16 to +0.23) | -0.591 (p 0.304) | -1.288 (p 0.040; state 0.112; wild 0.258) |
| D1 | +0.624 (0.455) | 0.170 / 0.290 / 0.296 | +0.09 (-0.04 to +0.22) | -0.283 (p 0.400) | -0.701 (p 0.118; wild 0.271) |

Robustness (all in `census_market_panel_fd.csv`): drop6 crop list (haat on D0 +1.245, p 0.218; D1 +0.612, p 0.133); haat alone; any market; irrigation change 1991-2011; adding 1991 levels; no irrigation control (384 districts, 22 states). No specification gives a haat or daily-market coefficient with p below 0.09 at the parent level or 0.2 under state inference. Reading: the panel neither confirms nor rules out the cross-sectional haat link; intervals are wide.

## 3. Fertiliser shops and pulses/cotton

No new estimate. Section 12 now states the reverse-causality reading: dealers locate where input demand is, and cotton is input-hungry.

## 4. Haats and sugar crops, leave one state out

Sugar share (full window, W, all controls): solo +0.1101 (SE 0.0271; state p 0.006; wild p 0.006); output-market block +0.1014 (SE 0.0280; state p 0.006; wild p 0.005).

Leave one state out (block): range +0.0811 (without Uttarakhand) to +0.1124 (without Madhya Pradesh); largest p 0.0022. Without Uttar Pradesh +0.0840; without Maharashtra +0.1003. Joint drops: without Maharashtra and Uttar Pradesh +0.0809 (wild p 0.030); without Maharashtra, Uttar Pradesh and Uttarakhand +0.0509 (wild p 0.036); without Uttar Pradesh, Uttarakhand and Gujarat (the three highest-share states with sugar) +0.0350 (SE 0.0139, p 0.012; wild p 0.018). No single state carries it; Maharashtra does not. Its size leans on the cane states. Kept on the page with that statement.

## 5. Mandi curvature

Moved out of the scene sequence into a note in the method drawer: curvature -8.21 (low mandi) against -4.06 (high), p 0.060 on population shares and 0.030 on village shares, one of five measures tried; no measure steepens the fall.

## 6. Haat mechanism

Haat coefficient (0 to all rural people), full window, all controls, state FE:

| outcome | mean | b (SE) | p parent | p state t(G-1) | p wild | one SD |
|---|---|---|---|---|---|---|
| count, any area | 21.38 | +6.276 (1.231) | <0.0001 | 0.0002 | 0.005 | +0.94 |
| count, 0.1% floor | 14.13 | +2.460 (1.200) | 0.040 | 0.145 | 0.159 | +0.37 |
| count, 0.5% floor | 9.69 | -0.058 (0.997) | 0.954 | 0.961 | 0.963 | -0.01 |
| count, 1% floor | 7.70 | -0.219 (0.875) | 0.803 | 0.808 | 0.808 | -0.03 |
| effective number | 4.95 | +0.553 (0.717) | 0.441 | 0.487 | 0.498 | +0.08 |

Drop6: any area +5.170 (wild p 0.006); 0.1% +2.184 (p 0.059); 0.5% +0.216; 1% -0.023; D1 +0.562. The effect is gone at 0.5 percent: about 60 percent of the extra crops each hold under 0.1 percent of cropped area.

Crop-level presence (share of years 1998-99 to 2019-20 grown; 53 crops grown in at least 60 districts; BH across crops): 20 crops more often grown, 2 less often at q < 0.05. Largest (b on share of years, 0 to all): sunflower +0.52, safflower +0.47, wheat +0.46, turmeric +0.44, linseed +0.41, sugarcane +0.39, banana +0.35, mesta +0.32, soybean +0.29, coriander +0.28; then other rabi pulses, jute, urad, masoor, horse gram, arhar, onion, moong, khesari, dry chillies. Less often: guar -0.13, moth -0.15. The 20 cover a median 0.39 percent of cropped area (mean 1.49); only sugarcane (+11.0 pts, q 0.001) and sunflower (+1.9 pts, q 0.040) also gain area share. Presence coefficients sum to +6.26, matching the count coefficient. 11 of the 22 also pass the state wild bootstrap at 0.05 (filled on the chart): sunflower, safflower, wheat, turmeric, linseed, sugarcane, banana, coriander, urad, masoor, onion. Neither negative crop passes.

Adversarial check: the Census 2011 haat item (multi-answer) gives a different picture. Census haat reach on D0 +3.258 (wild p 0.022), on D0 at 0.5% +2.108 (wild p 0.028), on D1 +1.857 (SE 0.493; state p 0.044; wild p 0.025; one SD 0.197 gives +0.37). With census mandi/regular market reach in the same model, census haat on D1 is +2.133 (wild p 0.033) and mandi/regular -0.550 (p 0.35). Holding census haat constant, the MA haat item still adds +5.48 crops (p < 0.0001) but nothing at 0.5% (-1.12, p 0.29) or on D1 (-0.30). The thin-crop pattern is specific to villages whose only recorded market is a haat; census haat reach, which includes villages that also have a mandi or regular market, goes with a fuller crop mix too. This census result has had no further referee pass.
