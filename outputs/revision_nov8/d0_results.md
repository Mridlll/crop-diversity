# D0 noise and the 1997-98 break

Scripts: `scripts/104_d0_threshold_and_1997.py` (main), `scripts/108_d0_1997_followups.py` (1997 decomposition, endpoint robustness). Full numeric output in `d0_results_raw.md` and `d0_followups_raw.md`; tables in `d0_*.csv`, `y1997_*.csv`.

## Data and method

- Raw APY (`all_crops_apy_1997_2021_india_data_portal.csv`), cleaned exactly as script 76 (bogus pairs dropped, Uttarakhand/Chhattisgarh/Jharkhand remaps, 2020-21 dropped, duplicate cells collapsed to the max). Raw D0 and D1 reproduce script 76's corrected panel exactly on all 14,089 district-years.
- Thresholded D0: per district-year, crops summed over seasons, share = crop area / district gross cropped area; count crops with share >= 0.1%, 0.5%, 1%.
- Main sample is the site's: balanced panel of 429 districts present in all 23 years, area-weighted by district cropped area. "Endpoint" = 1998-99 vs 2019-20 (the site's 21.2 to 24.9). "Window" = district change between 1998-2004 and 2013-2019 seven-year means. Robustness on the 606-district analysis sample (427 of them balanced) and on all 725 districts.
- Inference: cluster bootstrap on the 2011 parent (2000 draws in script 108, 1000 in 104; 429 clusters in the balanced panel) and cluster-robust OLS on the same groups for FE trends.

## 1. D0 with a minimum-share threshold

Balanced 429, area-weighted, cluster-bootstrap SE and 95% CI:

| measure | 1998-99 | 2019-20 | endpoint change | window change (aw) | window change (unweighted mean) | median district, window |
|---|---|---|---|---|---|---|
| D0 raw | 21.23 | 24.94 | +3.71 (0.43) [+2.83, +4.54] | -0.78 (0.22) [-1.23, -0.35] | -0.65 (0.26) [-1.17, -0.15] | -0.29 (0.22) [-0.71, +0.14] |
| D0 >= 0.1% | 14.28 | 14.21 | -0.07 (0.24) [-0.54, +0.39] | -1.22 (0.17) | -0.87 (0.19) | -0.57 (0.19) |
| D0 >= 0.5% | 10.28 | 9.81 | -0.47 (0.17) [-0.79, -0.14] | -0.89 (0.12) | -0.65 (0.11) | -0.43 (0.13) |
| D0 >= 1% | 8.29 | 7.95 | -0.34 (0.15) [-0.63, -0.03] | -0.61 (0.10) | -0.45 (0.08) | -0.43 (0.09) |
| D1 | 5.46 | 5.42 | -0.04 (0.13) [-0.29, +0.21] | -0.22 (0.08) | -0.14 (0.05) | -0.19 (0.03) |

District medians at the endpoints: raw D0 21 to 25; 0.1% 13 to 14; 0.5% 9 to 9; 1% 7 to 7; D1 4.6 to 4.5.

Within-district linear trend per decade, district FE, 1998-2019, balanced (n = 9,438 district-years), unweighted / area-weighted:
raw D0 -0.11 (0.15) / -0.22 (0.14); 0.1% -0.45 (0.12) / -0.69 (0.11); 0.5% -0.38 (0.07) / -0.53 (0.08); 1% -0.27 (0.06) / -0.37 (0.07); D1 -0.085 (0.036) / -0.124 (0.054). On all 725 districts (n = 13,605) the unweighted raw D0 trend is +0.54 (0.14) per decade while 0.5% is -0.09 (0.07) and D1 +0.01 (0.04).

How much of the rise survives: none. At 0.1% the endpoint change is zero (-2% of the raw rise); at 0.5% and 1% it is negative. Every threshold falls on the window comparison and on the FE trend.

### The endpoint rise depends on the start year

Area-weighted raw D0 change, start year by end year:

| start \ end | 2015 | 2016 | 2017 | 2018 | 2019 |
|---|---|---|---|---|---|
| 1998 | +1.61 | +2.21 | +2.27 | +3.18 | +3.71 |
| 1999 | -0.84 | -0.24 | -0.18 | +0.73 | +1.25 |
| 2000 | -1.32 | -0.72 | -0.66 | +0.25 | +0.78 |
| 2002 | -1.82 | -1.22 | -1.16 | -0.25 | +0.28 |

The single largest step in the whole series is 1998-99 to 1999-2000, +2.45 area-weighted (+2.28 unweighted). 1998-99 is itself a thin reporting year (below).

### Which crops drive raw D0's endpoint rise

By share of district area, the area-weighted count of crops below 0.1% of district area rises from 6.95 to 10.73 (+3.78), more than the whole +3.71. Crops at 0.1-0.5% add +0.40; 0.5-5% lose 0.59; crops at 5% or more +0.12.

Top gainers in area-weighted presence (share of 2019 presences below 0.5% of district area): Coriander +0.36 (93%), Guar Seed +0.35 (69%), Garlic +0.35 (88%), Other Oilseeds +0.35 (67%), Sweet Potato +0.31 (90%), Peas & Beans +0.26 (63%), Onion +0.23 (78%), Moong +0.21, Masoor +0.21, Ginger +0.21 (76%), Potato +0.20, Banana +0.20, Urad +0.18, Mesta +0.15 (91%), Turmeric +0.14 (86%). The ten largest gainers give +2.86 of the +3.71 and hold 7.1% of 2019 area. Median share when present in 2019: Coriander 0.03%, Garlic 0.03%, Sweet Potato 0.03%, Mesta 0.03%, Turmeric 0.06%. Gross gains +4.66, gross losses -0.95 (Sunflower, Castor, Moth, Sannhamp, Tobacco).

Two of these are naming rather than cropping: Guar Seed does not exist anywhere in the records before 1999-2000 (2.98 Mha in its first year, Rajasthan 2.65 Mha), and Other Oilseeds is 7 kha nationally in 1998-99. Cowpea (2006) and Other Summer Pulses (2016) are the only other crop names that enter after 1998. Distinct crop names nationally go 51 (1998) to 54 (2019).

By state, Madhya Pradesh alone contributes +2.92 of the +3.71 (mean district D0 13.9 in 1998-99 to 27.5 in 2019-20); MP's 1998-99 list lacks Moong, Urad, Masoor, Peas & Beans, Potato, Onion, Garlic, Coriander and nine others that it reports later. Then Rajasthan +0.86, Karnataka +0.68, Andhra Pradesh +0.60, Odisha +0.56 (Odisha 1998-99 has 13 crop names against 36 the next year).

### Reporting coverage by state

48 state-years in 20 states have a crop list under 60% of the state's own 1998-2019 median. 12 states collapse in 1997-98; Andhra Pradesh, Chhattisgarh, Madhya Pradesh and Nagaland also in 1998-99; otherwise collapses are scattered (Tripura 1999-2003, Andaman 2001-04, Himachal 2003-04, Tamil Nadu 2001, 2010, 2012, Ladakh often). 68 state-years see the crop-name count rise by 4 or more over the previous year; the largest are recoveries from collapses (Odisha 2019 14 to 39, Himachal 2005 7 to 31, Odisha 1999 13 to 36, Tamil Nadu 2011 14 to 36 and 2002 12 to 34). Lists are in `d0_state_jumps.csv` and `d0_state_collapses.csv`.

Removing collapsed state-years, and also unit-years carrying `flag_annual_unreliable` from the crop-group panel, the area-weighted endpoint rise in raw D0 shrinks to +2.32 (0.38) [+1.61, +3.07]; the window change becomes -1.29 (0.24), and every thresholded measure stays negative (0.5%: endpoint -0.62 (0.21), window -0.97 (0.12)). The 606-sample versions are identical to two decimals (n 427).

### Within district: does D0 rise when D1 does not?

- Window changes, 429 districts: raw D0 rose in 199; in 94 of those (22% of districts) D1 did not rise. For D0 at 0.5%, 147 rose and 40 (9%) without D1.
- Regressing the window change in raw D0 on the change in D1: slope 2.17 (0.24), intercept -0.34 (0.22, p = 0.12), R2 0.22. At zero D1 change raw D0 does not rise. For D0 at 0.5%: slope 1.46 (0.10), intercept -0.44 (0.08), R2 0.48.
- Districts whose D1 moved less than 0.25 (n = 103): raw D0 +0.65 (0.41, p = 0.11); D0 at 0.5% -0.23 (0.11, p = 0.03).
- Year on year (9,009 changes): raw D0 rises in 36%, and D1 falls or holds in 38% of those. Among rises of 3 or more crops (1,147), D1 falls or holds in 26% and D0 at 0.5% does not rise in 45%. Correlation of yearly changes with D1: 0.39 raw, 0.64 at 0.5%.
- Two-way FE (district and year): raw D0 on D1 1.89 (0.14); D0 0.5% on D1 1.44 (0.06).

Answer: raw D0 picks up a layer of movement unrelated to D1 (about half the year-to-year jumps of 3+ crops are crops under 0.5% of area), but across districts there is no systematic D0 rise where D1 is flat. The marginal crops come and go together with the reporting of a state's crop list, which is a year shock rather than a district trend.

## 2. The 1997-98 break

Year profile (all districts, cleaned): 1997-98 has 27 states, 484 districts, 8,532 rows, 49 crop names and 15.0 crops per district, against 30 states, 543 districts, 12,261 rows and 19.5 crops per district in 1998-99 and 20 to 22 per district from 1999 on. Its recorded area is 182.2 Mha, the highest of any year before 2010, against a 1999-2003 median of 164.9 Mha.

What is wrong with it, in order of effect on the area-weighted D1:

1. **One impossible crop in West Bengal.** Niger Seed, rabi, 1997-98: 50.8 Mha in 26 rows (1998-99: 7,587 ha), with production of 38.7 Mt filed alongside. Nadia alone is filed at 8.58 Mha, Murshidabad 5.54, Pashchim Medinipur 5.25. In the 18 balanced West Bengal districts Niger Seed is a median 84% of district area (minimum 54%). It is 28% of all area recorded nationally that year and makes West Bengal 59.0 Mha against 8.1 Mha in 1998-99. In the balanced panel West Bengal carries 33% of the 1997-98 area weight (5% in 1998-99) and every West Bengal district is dominated by one crop, so area-weighted D1 collapses. Dropping those rows alone moves 1997-98 D1 from 3.81 to 4.74. The crop-group panel already flags West Bengal 1997-98 (`f_gca_jump_severe`).
2. **Whole seasons missing.** State-seasons that hold under half of their 1999-2003 median: Odisha kharif (0 against 5.16 Mha; Odisha totals 0.91 Mha against 5.23 in 1998-99), Chhattisgarh kharif (0.48 against 4.56), Tamil Nadu kharif and whole year (absent), Rajasthan, Andhra Pradesh and Madhya Pradesh whole year (0), Karnataka whole year (0.45 against 1.26), Himachal kharif and rabi (absent), Tripura kharif (absent). 25 Mha less kharif nationally than 1998-99 (58.1 against 83.2). 88 of 429 balanced districts have kharif plus autumn under half their later median, 14 have none; their raw D0 is 11.2 against 23.2 later.
3. **States absent:** Tamil Nadu (6.8 Mha in 1998-99), Himachal Pradesh (0.9), Tripura (0.3), Puducherry, Dadra and Nagar Haveli, Chandigarh. Jammu and Kashmir, Ladakh and Goa are present in 1997-98 and absent in 1998-99.
4. **Short crop lists everywhere else.** Twelve states report under 60% of their usual crop names (Maharashtra 13 of 24, Madhya Pradesh 14 of 36, Rajasthan 18 of 38, Karnataka 18 of 42, Haryana 11 of 28, Kerala 5 of 20). Even in the 323 balanced districts outside West Bengal with kharif present, raw D0 is 16.0 against 24.0 and D1 4.84 against 5.89 in 1998-2004. Same-district comparison with 1998-99 (n = 467): D0 lower in 373, mean gap -4.20 (0.23); D1 lower in 309, mean gap -0.43 (0.07).
5. Extra area in a few places: Gujarat whole year 3.6 Mha (0.6 later), Punjab whole year 0.9 (0.1), Haryana whole year 0.8 (0.2).

1997-98 national value (balanced, area-weighted) under each correction:

| version | n | D0 raw | D0 0.5% | D1 | D1 unweighted |
|---|---|---|---|---|---|
| 1998-2004 mean | 429 | 24.01 | 10.64 | 5.56 | 5.02 |
| 1997-98 as published | 429 | 18.67 | 7.54 | 3.81 | 4.40 |
| without West Bengal Niger Seed | 429 | 16.24 | 9.14 | 4.74 | 4.45 |
| without West Bengal | 411 | 15.74 | 9.19 | 4.86 | 4.51 |
| outside West Bengal, kharif present | 323 | 15.96 | 9.24 | 4.84 | 4.56 |
| same 323, 1998-2004 mean | 323 | 23.96 | 10.87 | 5.89 | 5.31 |

Verdict: 1997-98 is a reporting artefact, with no agronomic content in the drop. The 3.8 comes mostly (about 0.9 of the 1.75 gap) from 50.8 Mha of Niger Seed filed for West Bengal's rabi, and the rest from incomplete returns: kharif missing for Odisha and Chhattisgarh, whole-year crops missing for Rajasthan, Andhra Pradesh and Madhya Pradesh, Tamil Nadu, Himachal and Tripura absent, and crop lists in twelve states at half their usual length.

## What holds and what does not

Holds:
- The effective number of crops is flat at the endpoints (-0.04, CI -0.29 to +0.21) and falls slightly on window and FE comparisons (-0.14 to -0.22 crops; -0.09 to -0.12 per decade).
- Every thresholded count is flat or falling under every comparison, sample and cleaning rule.
- 1997-98 is unusable and the reasons are documented row by row.

Does not hold:
- "The count of crops rose from 21.2 to 24.9" as a statement about diversification. It is an endpoint comparison against 1998-99, which is itself under-reported (Madhya Pradesh, Odisha, Andhra Pradesh, Chhattisgarh, Nagaland), and the rise is entirely in crops below 0.1% of district area. Starting from 1999-2000 the rise is +1.25; ending in 2015-17 it turns negative; on seven-year windows raw D0 falls (-0.78 aw, -0.65 unweighted). The site already reports that the median district's count fell by 0.29.
- Raw D0 rising on all 725 districts (+0.54 per decade unweighted) is not mirrored by any thresholded count; it reflects districts with fuller crop lists later in the record.

Limits: the reporting explanation is inferred from the pattern (state-wide list collapses and recoveries, crop names that do not exist before 1999); there is no APY metadata on the coverage of each return to confirm it. The 60% collapse cut-off is a judgement call; at it, 3% of balanced district-years are affected. We could not separate genuine small-area horticulture spread from better recording of it, since both look identical in APY.

## Plain findings

1. The rise in the count of crops (21.2 to 24.9) is a reporting effect. It sits entirely in crops below 0.1% of a district's area and disappears once crops under 0.1%, 0.5% or 1% of area are left out; at 0.5% the count falls slightly, 10.3 to 9.8.
2. Most of that endpoint rise comes from one state-year: Madhya Pradesh in 1998-99 reported 17 crop names against about 36 normally. Comparing seven-year averages, every version of the count, raw or thresholded, has fallen a little since 1998-2004.
3. 1997-98 is dropped because its records are broken, not because anything changed on the ground: 50.8 Mha of Niger Seed filed for West Bengal (a third of the year's recorded area), no kharif returns for Odisha or Chhattisgarh, Tamil Nadu, Himachal Pradesh and Tripura missing, and short crop lists in twelve other states.
4. The effective number of crops, which weights crops by area, is the stable measure: flat across the two decades, with a slight decline on longer-window comparisons.
