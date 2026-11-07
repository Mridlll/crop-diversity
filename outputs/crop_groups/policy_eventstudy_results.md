# Q2 policy event studies: NFSM, Odisha Millets Mission, 2018-19 MSP

## Status after referee (Nov 2026)

Written 7 November 2026. Referee: `referee_policy_es_results.md`, `referee_policy_es_checks.csv` (script 102). Site: Crop groups page, section 04.

- NFSM-Pulses 2007 with a treated-specific linear trend, -0.0088 (0.0032) on share and -0.057 (0.029) on log area: **refuted as arbitrary**. A treated trend plus saturated event dummies is collinear (R2 1.0000, rank 12 of 13 columns), so the number is not an estimate and should not be quoted.
- NFSM-Pulses 2007 DiD +0.0103 (0.0038): survives as **no detectable increase**. It comes from catch-up before entry (-0.034 at e = -10 to about zero by e = -4); short window +0.0036 (0.0035); post against e = -4 to -1 +0.0042 (0.0032); area +0.009 (0.041). Rises above about 0.010 of GCA (6 percent of the 0.18 mean) are ruled out. Rice and wheat 2007 unchanged (-0.0015, -0.0047).
- Odisha Millets Mission ATT -0.0011 (CI -0.008 to +0.006): **refuted as reported**. APY lists 13 to 14 crops for Odisha from 2005-06 to 2018-19 and 39 in 2019-20, so the millet group is ragi alone until 2019-20, and the 16 untreated districts hold 28 to 462 ha of millets in total. With a fixed basket the ragi share ATT is -0.0052 (0.0030), reshuffle p 0.04, but there is no valid control group, so no effect is identified in either direction.
- 2018-19 MSP increase: null survives. Smallest raw p 0.06, Holm 0.56, BH q 0.42, permutation p 0.60.
- Window: the "Drop 1997-98" row is now the default; 1997-98 is excluded everywhere by `_years.py`.

The original analysis below is kept unchanged for the record, apart from the sentences marked [corrected Nov 2026]. Where it disagrees with this block, this block and `outputs/ANALYSIS_INDEX.md` hold.

---

Script: `scripts/99_crop_groups_policy_event_studies.py` (runs in about 4 minutes). Log: `policy_es_log.txt`.
Policy inputs, sources and confidence are documented in `policy_results.md`. No APCNF, PMDS or RySS data are used.

## Common setup
- **Units:** 2011-parent units in the 606 sample (605 units), from the annual rows of `crop_group_panel_annual.csv`.
- **Dropped:** unit-years with `flag_annual_unreliable` = 1. 12,373 of 13,291 are kept.
- **Outcomes:** shares of gross cropped area, and log area (zero areas dropped).
- **Inference:** SEs are clustered on the unit, which is the shared-2011-parent group. 95% CIs are the estimate plus or minus 1.96 SE.

## A. NFSM entry (stacked event study)

### Design
- **Stacks:** one stack per entry cohort. Each stack holds only clean controls: units not yet treated anywhere in its window.
- **Why this is robust:** each stack has a single treatment date, so the event study inside it never uses an already-treated unit as a control (Cengiz et al. 2019). This is the same as Callaway-Sant'Anna ATT(g,t) with not-yet-treated controls.
- **Fixed effects:** unit x stack and state x year x stack.
- **Reference year:** e = -1, which is 2006-07.

### Only the 2007-08 list is identified with state-by-year fixed effects
- **Entry turns state-wide from 2010-11.** All districts of the 14 ISOPOM states entered in 2010-11, then the NE states, hill states and Kerala in 2014-15 and 2015-16. No within-state controls remain after that.
- **Assam and Jharkhand are dropped (51 units).** 25 of their pulses districts entered in 2010-11 but are not named in any source.
- **The 2010-11 expansion is descriptive only.** Its list is medium confidence, it can only be run with year fixed effects, and its controls are hill, NE and Kerala units.

| Stack | Window | Treated units | Control units | States | n | Clusters |
|---|---|---|---|---|---|---|
| Pulses 2007 (P07) | 1997-2009 | 167 | 265 | 14 | 4,944 | 432 |
| Pulses 2010 (P10), year FE only | 2003-2013 | 271 | 110 | n/a | 3,890 | 381 |
| Rice 2007 | 1997-2013 | 133 | 257 never listed by 2013 | 12 | 5,822 | 390 |
| Wheat 2007 | 1997-2013 | 138 | 163 never listed by 2013 | 9 | 4,653 | 301 |

### Three summaries per outcome
- **DiD:** post years against all pre years.
- **Avg event post:** the mean post coefficient, relative to e = -1.
- **Post vs e-4..-1:** the mean post coefficient minus the mean of e = -4 to -1. This one is the least affected by the long early pre-trend.

The pre-trend p-value is a joint F test on all leads.

| Outcome (stack) | DiD (SE) | Avg event post (SE) | Post vs e-4..-1 (SE) | Pre-trend p |
|---|---|---|---|---|
| Pulse share (P07) | +0.0103 (0.0038) | -0.0030 (0.0027) | +0.0042 (0.0032) | <0.001 |
| Log pulse area (P07) | +0.144 (0.039) | +0.023 (0.022) | -0.011 (0.045) | <0.001 |
| Rice share (P07) | +0.0019 (0.0035) | -0.0010 (0.0025) | -0.0018 (0.0025) | 0.61 |
| Wheat share (P07) | -0.0075 (0.0027) | +0.0019 (0.0025) | -0.0045 (0.0024) | 0.010 |
| Pulse share (P10, year FE) | -0.0042 (0.0026) | +0.0041 (0.0036) | +0.0014 (0.0027) | <0.001 |
| Log pulse area (P10, year FE) | +0.020 (0.085) | +0.210 (0.062) | -0.005 (0.069) | <0.001 |
| Rice share (Rice 2007) | -0.0015 (0.0048) | -0.0021 (0.0039) | -0.0071 (0.0038) | 0.017 |
| Log rice area (Rice 2007) | +0.082 (0.034) | +0.014 (0.027) | +0.024 (0.027) | <0.001 |
| Pulse share (Rice 2007) | +0.0044 (0.0039) | +0.0022 (0.0028) | +0.0048 (0.0033) | 0.21 |
| Wheat share (Wheat 2007) | -0.0047 (0.0043) | +0.0018 (0.0041) | -0.0038 (0.0038) | 0.001 |
| Log wheat area (Wheat 2007) | -0.046 (0.028) | +0.005 (0.023) | -0.024 (0.023) | 0.037 |
| Pulse share (Wheat 2007) | +0.0096 (0.0048) | -0.0021 (0.0033) | +0.0059 (0.0041) | 0.025 |

### Reading the pulses result
The positive pulse DiD comes from the pre-trend, not from entry. The event plot is `policy_es_nfsm_pulses_2007.png`.
- **Pulse share before entry:** the 2007 districts climb relative to controls, from -0.034 at e = -10 to about zero by e = -4. The coefficients are -0.014 at e = -3 and e = -2.
- **Pulse share after entry:** +0.003, -0.004 and -0.008.
- **Log pulse area:** the same catch-up, from -0.60 at e = -10 to about zero by e = -5, then flat.
- **Where the DiD comes from:** the DiD (+0.010, p = 0.006; +14% on area) is driven entirely by that convergence before entry.
- **With a treated-specific linear trend**, the post effect turns negative: -0.0088 (0.0032) on share and -0.057 (0.029) on log area.
- **Why the convergence is expected:** NFSM chose districts with large pulse area and low yields.

### Robustness, pulse share (P07)

| Variant | DiD (SE) | Post vs e-4..-1 (SE) |
|---|---|---|
| Drop 1997-98 | +0.0088 (0.0037) | +0.0042 (0.0032) |
| Balanced units (87 vs 126) | +0.0101 (0.0050) | +0.0039 (0.0042) |
| Year FE only | +0.0110 (0.0039) | +0.0013 (0.0033) |
| Keep flagged unit-years | +0.0066 (0.0037) | +0.0024 (0.0032) |
| Control for NFSM rice/wheat 2007 | +0.0092 (0.0037) | +0.0032 (0.0031) |

### Limitation
P07 has only three post years before its controls also enter in 2010-11. A slow effect that builds after 2010 cannot be separated from the national expansion.

### Files
- Figures: `policy_es_nfsm_pulses_2007.png`, `policy_es_nfsm_pulses_2010.png`, `policy_es_nfsm_rice_2007.png`, `policy_es_nfsm_wheat_2007.png`.
- Event coefficients: `policy_es_event_*.csv`.
- Summary table: `policy_es_nfsm_summary.csv`.

## B. Odisha Millets Mission (Callaway-Sant'Anna, within Odisha)

### Design
- **Sample:** 30 Odisha units, 2005-06 to 2019-20, with 2012-13 dropped because it is flagged. 420 unit-years.
- **Cohorts:** 7 districts in 2017-18, 4 in 2018-19 and 3 in 2019-20. 16 districts are untreated in the window; Angul enters in 2021-22.
- **Estimator:** ATT(g,t) with not-yet-treated controls and base year g-1, aggregated over post cells weighted by cohort size.
- **Inference:** a unit-cluster bootstrap (499 draws) and randomisation inference (499 reshuffles of the cohort labels across the 30 districts).

| Outcome | ATT | Bootstrap SE | 95% CI (percentile) | RI p | Mean pre-period placebo |
|---|---|---|---|---|---|
| Millet share of GCA | -0.0011 | 0.0035 | -0.0079 to +0.0055 | 0.74 | +0.0037 (0.0024) |
| Kharif millet share | -0.0001 | 0.0037 | -0.0082 to +0.0062 | 0.97 | +0.0036 (0.0021) |
| log(1 + millet area) | +0.64 | 0.37 | -0.05 to +1.36 | 0.13 | -0.07 (0.55) |

### Robustness
The 2017 cohort alone against the 16 untreated districts gives a millet share DiD of -0.0088 (SE 0.0069, 23 clusters).

### Context
- OMM runs in only 3 to 10 blocks per district, so a district-level share dilutes any effect.
- Millets are already small: about 5% of GCA in the 2017 cohort, under 1% elsewhere.
- Only 1 to 3 post years are observed.
- The 2019-20 rise in millet share appears in every cohort, the untreated districts included.

### Files
`policy_es_omm.png`, `policy_es_omm_summary.csv`, `policy_es_omm_attgt_*.csv`.

## C. 2018-19 MSP increase (shift-share)

### The price shock
- **Definition:** the excess log MSP change, i.e. ln MSP 2018-19 minus ln MSP 2017-18, minus the average annual log change over 2013-17.
- **Kharif crops:** the 2018 kharif hike dominates: ragi +0.36, niger +0.34, jowar +0.33, bajra +0.28, sunflower +0.25, cotton +0.23, moong +0.17, maize +0.16, paddy +0.08.
- **Rabi crops, tur and urad:** around zero or slightly negative.
- **Group shocks** (weighted by national 2013-16 area): millets +0.295, cotton +0.227, maize +0.155, rice +0.079, oilseeds +0.059, sugarcane +0.027, wheat +0.005, pulses +0.004, other -0.005.

### C1: district exposure
- **Exposure:** X_d = the sum over crops of the unit's 2013-16 crop share times that crop's excess change.
- **Distribution:** mean 0.073, SD 0.039, across 603 units. The most exposed units are in Karnataka and Marathwada.
- **Specification:** each group's share on standardised X_d x year, with unit and state-by-year fixed effects, 2008-2019, reference year 2017. n = 7,028; 603 clusters.

| Group | Avg 2018-19 per SD (SE) | Post vs 2014-16 (SE) | Pre-trend p |
|---|---|---|---|
| Rice | +0.0041 (0.0028) | +0.0038 (0.0031) | 0.027 |
| Wheat | -0.0018 (0.0015) | +0.0018 (0.0015) | <0.001 |
| Maize | +0.0011 (0.0015) | +0.0002 (0.0018) | 0.013 |
| Millets | -0.0024 (0.0029) | -0.0052 (0.0028) | <0.001 |
| Pulses | +0.0030 (0.0019) | -0.0001 (0.0020) | 0.009 |
| Oilseeds | +0.0005 (0.0019) | +0.0030 (0.0020) | <0.001 |
| Cotton | -0.0004 (0.0016) | -0.0012 (0.0014) | 0.046 |
| Sugarcane | +0.0007 (0.0006) | +0.0007 (0.0004) | 0.075 |
| Other | -0.0048 (0.0026) | -0.0030 (0.0032) | <0.001 |

### C2: group-level design
- **Specification:** the outcome is the unit-group-year share. The regressor is the 2013-16 group share x group shock x year.
- **Fixed effects:** unit x group, group x year and unit x year. Control: 2013-16 share x year.
- **Sample:** n = 63,297; 603 clusters.
- **Estimate:** the average 2018-19 coefficient is -0.141 (SE 0.056).
- **Pre-trends fail:** the joint F test is 5.99 (p < 0.001), and individual leads run from -0.235 to +0.245.
- **Randomisation inference:** permuting the 9 group shocks 2,000 times gives p = 0.60.
- **It rests on rice alone:** setting the rice shock to zero gives +0.025 (0.056).
- **Inference caveat:** with only 9 shocks, conventional clustered SEs overstate precision (Adao-Kolesar-Morales). The permutation p is the honest one.

### Files
- Figure: `policy_es_msp_exposure.png`.
- C1: `policy_es_msp_c1_exposure.csv`.
- C2: `policy_es_msp_c2_event.csv`, `policy_es_msp_c2_summary.csv`.
- Shocks and exposure: `policy_es_msp_shocks_crop.csv`, `policy_es_msp_shocks_group.csv`, `policy_es_msp_exposure.csv`.

## What holds and what does not

**Holds**
- NFSM-Pulses 2007 districts were converging on pulses for a decade before entry.
- NFSM-Rice and NFSM-Wheat 2007 entry left their own crop's share unchanged. The short-window contrasts are within about one to two SE of zero.

**Does not hold**
- Any positive NFSM-Pulses effect on pulse share or area, once the pre-trend is taken seriously.
- Any OMM effect on district millet share.
- Any shift after 2018 toward the crops whose MSP rose most, in the two post years observed.

**Not estimable with these lists**
- **NFSM-Pulses after 2010:** entry is state-wide, and 2010-13 coverage is medium or low confidence.
- **NFSM coarse cereals and nutri-cereals:** the 2015-17 list is missing, and the 2018 list leaves only two post years.
