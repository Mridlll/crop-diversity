# Policy and price data: NFSM districts, Odisha Millets Mission, MSP

Built by `scripts/97_policy_nfsm_omm_msp.py`. Source PDFs are saved in `outputs/crop_groups/raw_sources/`. Every CSV row carries the URL of the document it came from.

## Files

| File | Content |
|---|---|
| `policy_nfsm_district_lists.csv` | 2,185 rows, one per (component, guideline vintage, listed district), with the name as printed and the matched crop-panel key |
| `policy_nfsm_state_rules.csv` | component-years where coverage is "all districts of state S" |
| `policy_nfsm_district_year.csv` | 725 panel districts x 13 years (2007-08 to 2019-20) x 5 components: `treated`, `basis`, `confidence`, `in_606` |
| `policy_nfsm_unmatched.csv` | empty: all 2,185 listed names matched (2,145 exact or alias, 40 fuzzy at score 85 or above) |
| `policy_omm_blocks.csv` | 176 OMM blocks with entry year, basis and confidence |
| `policy_omm_district_year.csv` | 30 Odisha panel districts x 2014-15 to 2021-22, cumulative OMM blocks |
| `policy_msp_crop_year.csv` | MSP, 27 crop-variety series x 1997-98 to 2020-21, crop-year basis |
| `policy_msp_crop_group_map.csv` | MSP crop to crop group |
| `policy_msp_rbi_crosscheck.csv` | 15 RBI Handbook series against the compiled series |

## 1. National Food Security Mission

### What exists, by year

| Years | Rice | Wheat | Pulses | Coarse / nutri-cereals |
|---|---|---|---|---|
| 2007-08, 2008-09 | list, 133 districts | list, 138 | list, 168 | none |
| 2009-10 | list, 136 | list, 141 | list, 171 | none |
| 2010-11 to 2013-14 | 2009 list (138 by 2011; 2 added Jharkhand districts unnamed) | 2009 list (141) | all districts of 14 ISOPOM states, plus 10 Assam and 15 Jharkhand districts not named | none |
| 2014-15 | list, 199 | list, 119 | all districts of 15 states plus all 95 NE districts (557) | coarse cereals list, 182 |
| 2015-16 to 2017-18 | 2018 list used (194), vintage uncertain | 2018 list used (126), vintage uncertain | all districts of 29 states (638), vintage uncertain | 2014 list kept (182; official count rose to 265, list not found) |
| 2018-19, 2019-20 | list, 194 | list, 126 | all districts of 29 states, 638 | maize and barley list (269 incl. 7 common), nutri-cereals list (202, with jowar/bajra/ragi/small-millet flags) |

Within the 606-district sample: rice 133 (2007-08), 136 (2009-10), 185 (2014-15), 181 (2018-19); wheat 137 (2009-10), 118 (2014-15), 124 (2018-19); pulses 168, 171, then 441 of 556 known in 2010-11 to 2013-14 (50 Assam and Jharkhand districts set to NA), 544 in 2014-15, 604 from 2015-16; nutri-cereals 198 in 2018-19.

### Confidence

- **High:** the 2007, 2009, XII Plan (2014-15) and 2018-19 lists. The 2007 and 2009 lists are transcribed by hand because the PDF layout scrambles text extraction; every per-state count printed in the annexure is asserted. The 2014-15 and 2018-19 lists are parsed from word positions and every printed state count is asserted (no mismatches).
- **Medium:** pulses 2010-11 and 2011-12 as all districts of the 14 ISOPOM states. Two secondary sources quoting DAC agree (CSD report; an 8 Mar 2011 PIB-derived note). The DAC annual report itself was not retrieved.
- **Low:** 2012-13 and 2013-14 assumed to continue the 2010-11 coverage. Plausible since the 12th Plan guidelines took effect from 2014-15 (the 2018-19 guidelines say NFSM-Coarse Cereals began "from 2014-15"), but no year-specific list was found.
- **Medium/low:** 2015-16 to 2017-18. The 2018-19 document gives 194/126/638 and an official coarse-cereal count of 265 is cited for the later 12th Plan years, so the 2018 lists are applied. The exact year the 199/119/557 lists became 194/126/638 is not pinned down. The site lists amendment orders from Dec 2014 and Jan 2015 (J&K districts revised, 2 Punjab districts added, Pali added to wheat, Tiruppur replacing Nilgiris for pulses, 7 districts added) that were not applied one by one.

### Source quirks kept in the data
- 2009 wheat annex prints a stray "Gujarat (2): Dahod, Panchmahals" block. It is a copy of the rice entry: totals reach 141 only without it, so it is excluded.
- The 2009 wheat list for UP drops Chitrakoot, Lalitpur, Mahoba, Mirzapur and adds Barabanki, Kushinagar, Mainpuri, Mathura relative to 2007. Both versions are kept as printed.
- The XII Plan wheat list for MP prints both "East Nimar" and "Khandwa", which are the same district. Both rows are kept and both map to KHANDWA.
- Pre-2014 Andhra lists name Telangana districts. Where the crop panel carries a name under both states (Khammam, Medak, Nalgonda, Warangal, Karimnagar, Nizamabad), both keys are recorded, separated by ";".
- Lists name districts as they stood at the time. A district carved out later is not flagged when its parent was listed. Use the 2011-parent key (`shrug_key` in `outputs/shrug_covariates/district_crosswalk.csv`) for parent-level treatment.
- "Jaintia Hills" (pre-2012) is mapped to WEST JAINTIA HILLS; Sikkim's East/West/South/North are mapped to Gangtok/Gyalshing/Namchi/Mangan.

### URLs
- 2007 guidelines: https://agritech.tnau.ac.in/govt_schemes_services/govt_schemes_National%20Food%20Security%20Mission.pdf
- 2009 guidelines: https://www.nfsm.gov.in/Guidelines/ContentE.pdf
- 12th Plan guidelines: https://www.nfsm.gov.in/Guidelines/NFSMXII.pdf
- 2018-19 guidelines: https://www.nfsm.gov.in/Guidelines/Guideline_nfsmandoilseed201819to201920.pdf
- Amendment orders index: https://www.nfsm.gov.in/guidelinesorders.aspx
- 2010-11 expansion: http://www.csdindia.org/pdfs/Project-reports/ROLE%20OF%20NATIONAL%20FOOD%20SECURITY%20MISSION%20(NFSM)%20IN%20IMPROVING%20AGRICULTURAL%20PRODUCTIVITY.pdf and https://currentaffairsappsc.blogspot.com/2011/03/476-districts-covered-under-nfsm.html

### Gaps
- No district list for A3P (2010-11 Accelerated Pulses Production Programme). It ran in blocks/clusters inside NFSM-Pulses districts, so pulses treatment from 2010-11 is state-wide anyway.
- The 10 Assam and 15 Jharkhand pulses districts of 2010-11, and the 2 Jharkhand rice districts added by 2010-11, are not named in anything retrieved. Lok Sabha answers on eparlib.nic.in would likely name them, but the host did not resolve from this machine, and the Wayback Machine was offline during the session.
- The 265-district coarse-cereal list (2015-16 to 2017-18) was not found.

## 2. Odisha Millets Mission (Shree Anna Abhiyan)

District entries by year come from the OMM Annual Report 2020-21 phase table: 30 blocks in 7 districts (2017-18), +25 in 3 old and 4 new districts (2018-19), +17 in 3 new districts (2019-20, Keonjhar's 7 funded by DMF), +4 in Sundargarh (2020-21), 76 total. The 2021-22 total of 84 blocks in 15 districts implies Angul (5) and 3 more Sundargarh blocks.

| District | Entry | Blocks by 2019-20 |
|---|---|---|
| Gajapati, Kalahandi, Malkangiri, Nuapada | 2017-18 | 4, 4, 4, 3 |
| Kandhamal, Koraput, Rayagada | 2017-18 (more in 2018-19) | 7, 10, 8 |
| Balangir, Ganjam, Mayurbhanj, Sundargarh | 2018-19 | 4, 4, 3, 4 |
| Bargarh, Keonjhar, Nabarangpur | 2019-20 | 5, 7, 5 |

**Block names.** 29 of the 30 first-phase blocks are named in the 2017-18 annual report; its table lists 3 Rayagada blocks where every other table gives 4. Blocks for 2018-19 to 2021-22 come from the 84 "existing blocks" in the 2022-26 guideline, assigned to the district's entry year. This is exact for every district except Rayagada (5 blocks: 2017-18 or 2018-19) and Sundargarh (11 blocks spread over 2018-19, 2020-21 and 2021-22, not separable). The 58 new 2022-23 blocks are exact. The 34 blocks in the later 143-to-177 expansion carry an approximate year. They are listed one short of 177 because the 143 baseline includes one block outside the 142 list.

Confidence: high for district-year entry to 2021-22, high for 145 of 176 block-years, low for the rest.

URLs: https://milletsodisha.com/uploads/files/publication/Annual-Report-Odisha-Millets-Mission_2017-18_24.09.2019.pdf ; https://milletsodisha.com/uploads/files/Odisha%20Millets%20Mission%20-%20Annual%20Report%202020-21.pdf ; https://milletsodisha.com/uploads/files/guidelinedocument/Approved%20OMM%20Guideline_Final.pdf ; https://milletsodisha.com/uploads/files/guidelinedocument/Approved_MODIFICATIONS%20OF%20ODISHA%20MILLETS%20MISSION%20GUIDELINE.pdf

Within the crop panel window (to 2019-20) OMM touches 14 of Odisha's 30 districts, with 3 to 10 blocks each.

## 3. Minimum Support Prices, 1997-98 to 2020-21

Crop-year basis throughout: a rabi crop sown in crop year 2018-19 is marketed in RMS 2019-20 and appears as 2018-19. Values are as published by each source.

| Span | Source | URL |
|---|---|---|
| 1997-98 to 1999-00 | Economic Survey 1999-2000, Table 5.5 | https://www.indiabudget.gov.in/budget_archive/es99-2000/app5.5.pdf |
| 1999-00 (rabi) to 2004-05 | DAC/DES MSP table as on 03.11.2004 | https://desagri.gov.in/wp-content/uploads/2021/04/MINIMUM-SUPPORT-PRICES03-11-2004.pdf |
| 2004-05 to 2012-13 | Economic Survey 2012-13, Table 5.5 | https://www.indiabudget.gov.in/budget_archive/es2012-13/estat1.pdf |
| 2013-14 to 2020-21 | CACP "MSPs recommended and fixed", fixed column | https://www.spuvvn.edu/academic_centres/c_c_s/MSP_SINCE_2012_13.pdf |
| check | RBI Handbook 2025-26, Tables 23 and 24 | https://www.rbi.org.in/scripts/PublicationsView.aspx?id=23848 , https://www.rbi.org.in/scripts/PublicationsView.aspx?id=23849 |

Coverage: 24 of 24 years for paddy (common, grade A), wheat, maize, jowar, bajra, ragi, barley, arhar, moong, urad, gram, groundnut, rapeseed-mustard, soybean yellow, sunflower, sesamum, nigerseed, safflower, cotton (long staple), jute and sugarcane. Masur starts 2000-01 (first MSP year). Jowar maldandi starts 2006-07. Soybean black stops after 2014-15. Toria runs to 2011-12 (it is not in the CACP table). Cotton medium staple has all years.

**RBI cross-check.** 347 of 360 crop-years (96.4%) agree within Rs 1. The 13 that differ are bonuses: RBI adds them and Economic Survey tables report the base MSP with a footnote. Paddy 2007-08 to 2009-10 (+Rs 50 to 100), wheat 2006-07 and 2010-11, and the Rs 500 incentive on tur, moong and urad in 2010-11 and 2011-12. Maize 2004-05 and 2005-06 is the one real disagreement: DES and the Economic Survey give 525 and 540, RBI gives 515 and 525. Kept the DES figure. In 2015-16 to 2017-18 the CACP "fixed" column includes the pulse and oilseed bonuses. Those rows carry `bonus_flag_in_source = True`.

**2018-19 change.** `cost_plus_50_regime = 1` from crop year 2018-19: kharif 2018-19 MSPs (announced 4 July 2018) and rabi RMS 2019-20 were set at no less than 1.5 times A2+FL cost. The jumps are visible in the data: jowar 1700 to 2430, bajra 1425 to 1950, ragi 1900 to 2897, moong 5575 to 6975, nigerseed 4050 to 5877, cotton medium 4020 to 5150, paddy 1550 to 1750. Announcement: https://www.business-standard.com/article/news-cm/cabinet-approves-hike-in-msp-for-kharif-crops-for-2018-19-season-118070401007_1.html (a reprint of the PIB release; the PIB page itself was not retrieved).

### MSP crop to crop group

| Crop group | MSP crops |
|---|---|
| rice | paddy (common, grade A) |
| wheat | wheat |
| maize | maize |
| millets | jowar (hybrid, maldandi), bajra, ragi |
| pulses | arhar/tur, moong, urad, gram, masur |
| oilseeds | groundnut, rapeseed-mustard, toria, soybean (black, yellow), sunflower, sesamum, nigerseed, safflower |
| cotton | cotton (medium, long staple) |
| sugarcane | sugarcane (SMP to 2008-09, FRP from 2009-10) |
| other | barley, jute |

Small millets have no MSP. Copra, de-husked coconut and tobacco appear in some tables and are not carried.

## What could not be found
- The DAC Annual Report 2010-11 (primary source for the 2010-11 pulses expansion), and named lists for the 2010-11 Assam/Jharkhand additions.
- The coarse-cereal list for 2015-16 to 2017-18 (265 districts).
- A year-by-year record of the Dec 2014 to Jan 2015 NFSM amendment orders, applied to the lists.
- Block-level entry years in Sundargarh and the fourth 2017-18 Rayagada block.


## Event-study results

Estimates using these lists are in `policy_eventstudy_results.md` (script `scripts/99_crop_groups_policy_event_studies.py`).
