# Census irrigation panel by source: audit

Script 121_census_irrigation_panel.py. Source: SHRUG 2.1 village directories (PC91, PC01, PC11), aggregated to constant 2011 districts through shrid_pc11dist_key. Areas are hectares in all three rounds.

## 1. What the panel holds

- `census_irrigation_panel.csv`: 1875 district-round rows (625 districts x 3 rounds), long.
- `census_irrigation_panel_wide.csv`: one row per district, round suffixes _1991, _2001, _2011.
- `census_irrigation_state_round.csv`: the state x round coverage table behind section 3.
- `census_irrigation_national.csv`, `census_irrigation_validation.csv`: tables reproduced below.
- Irrigated share = irrigated / (irrigated + unirrigated), net irrigated over net area sown. `sh_*` columns are source area over that same denominator; `comp_*` are source area over total irrigated area.
- `district_usable` is True when the district's state-round passes the checks in section 3 and the district has at least 50% of its PC11 rural area in valid villages.
- Terrain: `elevation_mean_m` and `tri_mean` are area-weighted (by raster cell count) over the district's shrids; they match the SHRUG district files (correlation 0.9777 for elevation, 0.9884 for TRI).

## 2. Source harmonisation (every mapping)

| category | 1991 fields | 2001 fields | 2011 fields |
|---|---|---|---|
| canal | canal_govt + canal_pvt | canal_govt + canal_pvt | land_canal_irr |
| groundwater | well_wo_el + well_w_el + tw_wo_el + tw_w_el | well_wo_el + tw_wo_el + tw_w_el (no well-with-electricity field exists in the 2001 file) | land_wl_tw_irr (wells and tube wells lumped) |
| gw_elec (electrified wells and tube wells) | well_w_el + tw_w_el | tw_w_el only (partial: electrified open wells are not recorded) | not recorded |
| tank_lake | tank_irr + lake_irr | tank_irr + lake_irr | land_tnk_lk_irr (lumped) |
| river | river_irr | river_irr | not a category (falls in 'other') |
| other_misc (waterfall + other) | w_fall + oth_irr | w_fall + oth_irr | land_w_fall_irr + land_oth_src_irr |
| river_other (river + waterfall + other) | river_irr + w_fall + oth_irr | river_irr + w_fall + oth_irr | land_w_fall_irr + land_oth_src_irr |
| irrigated total | tot_irr; if missing, sum of the source columns | tot_irr; if missing, sum of the source columns | max(land_src_irr, sum of sources) |
| unirrigated | un_irr | un_irr | land_un_irr |
| unattributed | tot - sum(listed sources), floored at 0 | tot - sum(listed sources), floored at 0 | src_irr - sum(listed sources), floored at 0 |

Rules applied at village level:

1. Total irrigated = the larger of the village's total field and the sum of its source fields (blank fields ignored). The 1991 total is blank or zero in many villages whose source fields are filled; sources then exceed the total, and the repair adds the difference to irrigated area. The repair is quantified per round in the log, and the 2001 and 2011 totals are almost never below their sources. A village needs both a total irrigated and an unirrigated value to count as having land data.
2. Missing source fields inside a village that has a total are set to zero (a blank source is a zero in these directories).
3. A village is dropped as implausible when irrigated + unirrigated exceeds 1.5x its reference area (the larger of the round's own village area and the PC11 shrid area) and is above 5 ha. These are mostly merged or split villages and keying errors (the worst 2001 row reports 2.4 million ha of unirrigated land).
4. Shares come from summed hectares, never from averaging village or district ratios.
5. Comparability limits that no mapping can repair: 2011 has no river category (river irrigation sits inside other), 2011 lumps wells with tube wells, 2001 has no electrified open wells, and electrification is not recorded in 2011. The comparable three-round groups are canal, groundwater, tank and lake, and river_other.

## 3. Coverage audit

National coverage:

| round | rows in file | non-missing land (share of PC11 rural shrids) | valid area / PC11 rural area | implausible villages |
|---|---|---|---|---|
| 1991 | 557893 | 0.929 | 0.890 | 6229 |
| 2001 | 522122 | 0.886 | 0.920 | 8057 |
| 2011 | 588973 | 0.999 | 0.996 | 3242 |

Share of PC11 rural villages (shrids) with non-missing land data, and share of PC11 rural area in valid villages, by state and round. `fit` is the verdict from the state-round rules below.

| state_name | area cov 1991 | area cov 2001 | area cov 2011 | village cov 1991 | village cov 2001 | village cov 2011 | share 1991 | share 2001 | share 2011 | fit 1991 | fit 2001 | fit 2011 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| andaman nicobar islands | 73% | 23% | 100% | 80% | 46% | 100% | 0.00 | 0.05 | 0.00 | FLAG | FLAG | FLAG |
| andhra pradesh | 97% | 100% | 100% | 97% | 100% | 100% | 0.30 | 0.34 | 0.33 | ok | ok | ok |
| arunachal pradesh | 27% | 96% | 100% | 42% | 94% | 100% |  |  | 0.30 | FLAG | FLAG | ok |
| assam | 91% | 92% | 100% | 91% | 91% | 100% | 0.05 | 0.09 | 0.06 | ok | ok | ok |
| bihar | 99% | 95% | 100% | 97% | 96% | 100% | 0.56 | 0.59 | 0.64 | ok | ok | ok |
| chandigarh | 0% | 100% | 100% | 0% | 100% | 100% |  | 0.82 | 0.78 | FLAG | ok | ok |
| chhattisgarh | 55% | 100% | 97% | 58% | 100% | 99% | 0.18 | 0.20 | 0.26 | FLAG | ok | ok |
| dadra nagar haveli | 66% | 100% | 100% | 68% | 100% | 100% | 0.27 | 0.06 | 0.12 | FLAG | ok | ok |
| daman diu | 100% | 100% | 100% | 100% | 100% | 100% | 0.07 | 0.14 | 0.12 | ok | ok | ok |
| goa | 98% | 99% | 100% | 100% | 100% | 100% | 0.04 | 0.05 | 0.11 | ok | ok | ok |
| gujarat | 76% | 100% | 100% | 79% | 100% | 100% | 0.28 | 0.33 | 0.42 | ok | ok | ok |
| haryana | 99% | 99% | 100% | 99% | 100% | 100% | 0.72 | 0.80 | 0.88 | ok | ok | ok |
| himachal pradesh | 92% | 97% | 100% | 95% | 99% | 100% | 0.16 | 0.18 | 0.20 | ok | ok | ok |
| jammu kashmir | 0% | 99% | 100% | 0% | 99% | 100% |  | 0.40 | 0.40 | FLAG | ok | ok |
| jharkhand | 99% | 99% | 99% | 98% | 99% | 100% | 0.14 | 0.18 | 0.13 | ok | ok | ok |
| karnataka | 97% | 99% | 100% | 97% | 99% | 100% | 0.16 | 0.22 | 0.27 | ok | ok | ok |
| kerala | 100% | 100% | 99% | 100% | 100% | 100% | 0.23 | 0.30 | 0.28 | ok | ok | ok |
| lakshadweep | 52% | 95% | 100% | 100% | 100% | 100% | 0.05 | 0.00 | 0.00 | FLAG | FLAG | FLAG |
| madhya pradesh | 95% | 100% | 100% | 94% | 99% | 100% | 0.18 | 0.34 | 0.44 | ok | ok | ok |
| maharashtra | 98% | 91% | 98% | 97% | 100% | 100% | 0.11 | 0.17 | 0.83 | ok | ok | FLAG |
| manipur | 89% | 89% | 100% | 85% | 92% | 100% |  |  | 0.07 | FLAG | FLAG | FLAG |
| meghalaya | 91% | 63% | 100% | 91% | 69% | 100% |  |  |  | FLAG | FLAG | FLAG |
| mizoram | 79% | 67% | 100% | 81% | 68% | 100% | 0.13 |  |  | ok | FLAG | FLAG |
| nagaland | 88% | 98% | 100% | 91% | 99% | 100% | 0.32 | 0.22 | 0.16 | ok | FLAG | ok |
| nct of delhi | 0% | 100% | 100% | 0% | 100% | 100% |  | 0.80 | 0.86 | FLAG | ok | ok |
| odisha | 98% | 39% | 100% | 97% | 40% | 100% | 0.16 | 0.19 | 0.22 | ok | FLAG | ok |
| puducherry | 86% | 100% | 100% | 90% | 100% | 100% | 0.74 | 0.79 | 0.89 | ok | ok | ok |
| punjab | 92% | 95% | 100% | 92% | 95% | 100% | 0.85 | 0.92 | 0.98 | ok | ok | ok |
| rajasthan | 92% | 99% | 100% | 93% | 99% | 100% | 0.17 | 0.29 | 0.31 | ok | ok | ok |
| sikkim | 78% | 79% | 99% | 99% | 100% | 100% | 0.12 | 0.12 | 0.15 | ok | ok | ok |
| tamil nadu | 86% | 96% | 100% | 94% | 99% | 100% | 0.41 | 0.46 | 0.57 | ok | ok | ok |
| tripura | 93% | 96% | 100% | 97% | 98% | 100% | 0.29 | 0.28 | 0.28 | ok | ok | ok |
| uttar pradesh | 98% | 96% | 100% | 98% | 99% | 100% | 0.67 | 0.77 | 0.83 | ok | ok | ok |
| uttarakhand | 63% | 92% | 100% | 97% | 99% | 100% | 0.36 | 0.39 | 0.40 | ok | ok | ok |
| west bengal | 95% | 22% | 99% | 98% | 22% | 100% | 0.33 | 0.34 | 0.52 | ok | FLAG | ok |

State-round rules (a state-round is flagged if any applies):

- the round has no land rows for the state;
- every land cell in the state is zero, so no cultivated area is recorded;
- valid villages cover under 60% of the state's PC11 rural area (a coverage-only flag still leaves districts usable where the district itself has at least 50% coverage);
- zeros everywhere: over 97% of cultivated valid villages report zero irrigation, or over 95% do and the state share is under 1%;
- more than 10% of the state's reported area sits in implausible villages;
- the mean census district share differs from the DES mean on the same districts (2000-02 for the 2001 round, 2010-12 for 2011, at least 3 matched districts) by more than 0.25;
- where DES cannot be matched in 2001 or 2011: 2011 differs from Antyodaya 2019-20 on the same districts by more than 0.25, or the share differs by more than 0.30 from the other of 2001/2011 when that one passes its DES check;
- for 1991, where DES cannot help (its panel starts in 1997): the 1991 state share differs from the 2001 share by more than 0.25, or is under 0.5x (or over 2.00x) the 2001 share with a gap above 0.05, when the 2001 state-round passes its own checks. Net irrigated area rose by roughly a sixth between the two censuses nationally, so a 1991 state share far below its 2001 share is under-recording.

Flagged state-rounds:

| state_name | round | cov_area | irr_share | des_state_share | flag_reason |
|---|---|---|---|---|---|
| andaman nicobar islands | 1991 | 73% | 0.003 |  | 97% of cultivated villages report zero irrigation |
| arunachal pradesh | 1991 | 27% |  |  | no cultivated land recorded (all land cells zero); area coverage 27% < 60% |
| chandigarh | 1991 | 0% |  |  | round absent for this state |
| chhattisgarh | 1991 | 55% | 0.177 |  | area coverage 55% < 60% |
| dadra nagar haveli | 1991 | 66% | 0.267 |  | 1991 share 0.27 vs 2001 share 0.06 |
| jammu kashmir | 1991 | 0% |  |  | round absent for this state |
| lakshadweep | 1991 | 52% | 0.050 |  | area coverage 52% < 60% |
| manipur | 1991 | 89% |  |  | no cultivated land recorded (all land cells zero) |
| meghalaya | 1991 | 91% |  |  | no cultivated land recorded (all land cells zero) |
| nct of delhi | 1991 | 0% |  |  | round absent for this state |
| andaman nicobar islands | 2001 | 23% | 0.046 |  | area coverage 23% < 60% |
| arunachal pradesh | 2001 | 96% |  |  | no cultivated land recorded (all land cells zero) |
| lakshadweep | 2001 | 95% | 0.000 |  | 100% of cultivated villages report zero irrigation |
| manipur | 2001 | 89% |  |  | no cultivated land recorded (all land cells zero) |
| meghalaya | 2001 | 63% |  |  | no cultivated land recorded (all land cells zero) |
| mizoram | 2001 | 67% |  |  | no cultivated land recorded (all land cells zero) |
| nagaland | 2001 | 98% | 0.217 | 0.193 | census share 0.56 vs DES 0.19 on the same districts |
| odisha | 2001 | 39% | 0.187 | 0.233 | area coverage 39% < 60% |
| west bengal | 2001 | 22% | 0.342 |  | area coverage 22% < 60% |
| andaman nicobar islands | 2011 | 100% | 0.003 |  | 99% of cultivated villages report zero irrigation; census share 0.00 vs Antyodaya 2019-20 0.41 on the same districts (no DES match) |
| lakshadweep | 2011 | 100% | 0.000 |  | 100% of cultivated villages report zero irrigation |
| maharashtra | 2011 | 98% | 0.827 |  | census share 0.83 vs Antyodaya 2019-20 0.53 on the same districts (no DES match); share 0.83 vs 0.17 in the other census round that passes the DES check (no DES match here) |
| manipur | 2011 | 100% | 0.071 |  | census share 0.09 vs Antyodaya 2019-20 0.53 on the same districts (no DES match) |
| meghalaya | 2011 | 100% |  |  | no cultivated land recorded (all land cells zero) |
| mizoram | 2011 | 100% |  |  | no cultivated land recorded (all land cells zero) |

Coverage of a share above 100% means the round lists more village rows than the PC11 rural shrid universe (villages that split or merge between rounds); those shares are capped at 100% in the table above.

## 4. National shares by round and source

Share of (irrigated + unirrigated) area. `comp_*` columns are shares of total irrigated area.

| round | basis | n_states | irr_unirr_Mha | irr_share | sh_canal | sh_groundwater | sh_gw_elec | sh_tank_lake | sh_river | sh_other_misc | sh_river_other | sh_unattributed | comp_canal | comp_groundwater | comp_tank_lake | comp_river_other | comp_unattributed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1991 | all valid villages | 32 | 145 | 0.297 | 0.096 | 0.144 | 0.078 | 0.026 | 0.010 | 0.015 | 0.025 | 0.008 | 0.322 | 0.483 | 0.086 | 0.083 | 0.026 |
| 1991 | usable state-rounds only | 25 | 141 | 0.300 | 0.095 | 0.147 | 0.079 | 0.026 | 0.010 | 0.015 | 0.025 | 0.008 | 0.316 | 0.488 | 0.086 | 0.083 | 0.026 |
| 2001 | all valid villages | 35 | 139 | 0.368 | 0.124 | 0.135 | 0.066 | 0.021 | 0.011 | 0.017 | 0.028 | 0.061 | 0.336 | 0.367 | 0.056 | 0.076 | 0.164 |
| 2001 | usable state-rounds only | 26 | 135 | 0.372 | 0.124 | 0.137 | 0.067 | 0.021 | 0.011 | 0.017 | 0.028 | 0.062 | 0.334 | 0.369 | 0.055 | 0.075 | 0.167 |
| 2011 | all valid villages | 35 | 144 | 0.508 | 0.124 | 0.224 |  | 0.023 |  | 0.050 | 0.050 | 0.087 | 0.243 | 0.440 | 0.046 | 0.098 | 0.172 |
| 2011 | usable state-rounds only | 29 | 125 | 0.459 | 0.137 | 0.243 |  | 0.025 |  | 0.055 | 0.055 | 0.000 | 0.299 | 0.528 | 0.053 | 0.120 | 0.000 |

Cross-check against the SHRUG district files (sums of district totals vs village-level sums, hectares in millions):

| round | dist_file_irr_Mha | village_irr_all_Mha | dist_file_unirr_Mha | village_unirr_all_Mha | dist_file_share | village_all_share | village_valid_share |
|---|---|---|---|---|---|---|---|
| 1991 | 22.464 | 44.289 | 112 | 105 | 0.168 | 0.297 | 0.297 |
| 2001 | 57.831 | 55.784 | 104 | 97.126 | 0.358 | 0.365 | 0.368 |
| 2011 | 75.170 | 76.099 | 73.491 | 72.412 | 0.506 | 0.512 | 0.508 |

For 2011 the village aggregation reproduces the SHRUG district file at district level: correlation 0.9753, median absolute difference 0.0007 in share. The 1991 district file totals are lower than the village sums because it omits the irrigated total for villages where only the source fields are filled (about 13% of 1991 villages), which the village build recovers.

## 5. Validation

| comparison | n | pearson | spearman | mean_x | mean_y | mean_diff_x_minus_y | mean_abs_diff | slope |
|---|---|---|---|---|---|---|---|---|
| census 2011 vs Antyodaya 2019-20, all districts | 582 | 0.827 | 0.829 | 0.483 | 0.554 | -0.071 | 0.149 | 1.137 |
| census 2011 vs Antyodaya 2019-20, site sample (all districts) | 572 | 0.832 | 0.832 | 0.485 | 0.556 | -0.071 | 0.147 | 1.142 |
| census 2011 vs Antyodaya 2019-20, usable districts | 542 | 0.881 | 0.881 | 0.467 | 0.556 | -0.089 | 0.136 | 1.151 |
| census 2011 vs Antyodaya 2019-20, site sample (usable districts) | 533 | 0.886 | 0.885 | 0.468 | 0.559 | -0.091 | 0.135 | 1.159 |
| census 2011 vs DES 2010-12, usable districts, unshared parents only | 467 | 0.939 | 0.938 | 0.473 | 0.494 | -0.021 | 0.072 | 0.944 |
| census 2011 vs DES 2010-12, all districts, unshared parents only | 467 | 0.939 | 0.938 | 0.473 | 0.494 | -0.021 | 0.072 | 0.944 |
| census 2001 vs DES 2000-02, usable districts, unshared parents only | 381 | 0.942 | 0.939 | 0.471 | 0.478 | -0.007 | 0.070 | 0.916 |
| census 2001 vs DES 2000-02, all districts, unshared parents only | 414 | 0.907 | 0.901 | 0.462 | 0.462 | 0.000 | 0.080 | 0.894 |
| census 2011 vs DES 2010-12, usable districts, all parents, mean of child districts | 520 | 0.939 | 0.937 | 0.467 | 0.489 | -0.022 | 0.073 | 0.940 |
| census 2011 vs DES 2010-12, all districts, all parents, mean of child districts | 520 | 0.939 | 0.937 | 0.467 | 0.489 | -0.022 | 0.073 | 0.940 |
| census 2001 vs DES 2000-02, usable districts, all parents, mean of child districts | 423 | 0.941 | 0.939 | 0.466 | 0.474 | -0.008 | 0.070 | 0.910 |
| census 2001 vs DES 2000-02, all districts, all parents, mean of child districts | 456 | 0.909 | 0.904 | 0.458 | 0.460 | -0.002 | 0.080 | 0.890 |
| census 2001 vs census 2011, usable in both | 496 | 0.952 | 0.951 | 0.439 | 0.480 | -0.041 | 0.065 | 0.895 |
| census 1991 vs census 2001, all | 551 | 0.916 | 0.912 | 0.342 | 0.416 | -0.075 | 0.094 | 0.876 |
| census 1991 vs census 2011, all | 550 | 0.794 | 0.771 | 0.340 | 0.491 | -0.151 | 0.163 | 0.708 |

Reading the table: DES and the census agree closely (r about 0.94) in both overlap rounds, which validates the village build for 2001 and 2011. Antyodaya 2019-20 agrees less (r about 0.82; 0.87 on usable districts) and runs about 9 points higher, as expected from a later year and a self-reported village survey. Census 1991 correlates only about 0.5 with 2001 because the 1991 directory under-records irrigation in many states.

State-level agreement with DES in the two overlapping rounds (district-level correlation within state, n at least 5):

| round | state_name | n | corr | census_mean | des_mean |
|---|---|---|---|---|---|
| 2001 | nagaland | 8 | 0.030 | 0.563 | 0.193 |
| 2001 | odisha | 28 | 0.503 | 0.240 | 0.268 |
| 2001 | haryana | 19 | 0.671 | 0.823 | 0.838 |
| 2001 | madhya pradesh | 45 | 0.756 | 0.361 | 0.322 |
| 2001 | tamil nadu | 28 | 0.771 | 0.487 | 0.538 |
| 2001 | bihar | 36 | 0.825 | 0.578 | 0.610 |
| 2001 | rajasthan | 32 | 0.859 | 0.391 | 0.404 |
| 2001 | punjab | 17 | 0.867 | 0.915 | 0.942 |
| 2001 | kerala | 14 | 0.885 | 0.319 | 0.178 |
| 2001 | gujarat | 19 | 0.902 | 0.335 | 0.307 |
| 2001 | karnataka | 27 | 0.922 | 0.239 | 0.264 |
| 2001 | uttar pradesh | 69 | 0.932 | 0.779 | 0.768 |
| 2001 | maharashtra | 33 | 0.951 | 0.155 | 0.188 |
| 2001 | jammu kashmir | 14 | 0.952 | 0.439 | 0.470 |
| 2001 | andhra pradesh | 22 | 0.959 | 0.386 | 0.442 |
| 2001 | uttarakhand | 13 | 0.972 | 0.308 | 0.305 |
| 2001 | himachal pradesh | 12 | 0.990 | 0.252 | 0.280 |
| 2001 | chhattisgarh | 16 | 0.991 | 0.192 | 0.215 |
| 2001 | chandigarh | 1 |  | 0.825 | 0.617 |
| 2001 | arunachal pradesh | 0 |  |  |  |
| 2001 | mizoram | 0 |  |  |  |
| 2001 | meghalaya | 0 |  |  |  |
| 2001 | dadra nagar haveli | 1 |  | 0.064 | 0.313 |
| 2001 | puducherry | 2 |  | 0.779 | 0.774 |
| 2011 | arunachal pradesh | 8 | 0.302 | 0.276 | 0.286 |
| 2011 | odisha | 30 | 0.421 | 0.241 | 0.275 |
| 2011 | assam | 27 | 0.502 | 0.064 | 0.075 |
| 2011 | jharkhand | 24 | 0.581 | 0.140 | 0.151 |
| 2011 | bihar | 38 | 0.736 | 0.647 | 0.591 |
| 2011 | gujarat | 25 | 0.820 | 0.401 | 0.431 |
| 2011 | karnataka | 30 | 0.824 | 0.282 | 0.329 |
| 2011 | kerala | 14 | 0.849 | 0.306 | 0.199 |
| 2011 | haryana | 21 | 0.854 | 0.892 | 0.860 |
| 2011 | punjab | 20 | 0.875 | 0.972 | 0.983 |
| 2011 | rajasthan | 33 | 0.919 | 0.394 | 0.512 |
| 2011 | andhra pradesh | 22 | 0.928 | 0.367 | 0.475 |
| 2011 | tamil nadu | 31 | 0.929 | 0.560 | 0.573 |
| 2011 | uttar pradesh | 70 | 0.935 | 0.843 | 0.830 |
| 2011 | uttarakhand | 13 | 0.945 | 0.272 | 0.314 |
| 2011 | jammu kashmir | 22 | 0.972 | 0.435 | 0.472 |
| 2011 | madhya pradesh | 50 | 0.973 | 0.430 | 0.495 |
| 2011 | chhattisgarh | 18 | 0.980 | 0.223 | 0.266 |
| 2011 | himachal pradesh | 12 | 0.996 | 0.270 | 0.293 |
| 2011 | nagaland | 4 |  | 0.248 | 0.277 |
| 2011 | mizoram | 0 |  |  |  |
| 2011 | tripura | 4 |  | 0.308 | 0.265 |
| 2011 | meghalaya | 0 |  |  |  |
| 2011 | dadra nagar haveli | 1 |  | 0.117 | 0.215 |
| 2011 | goa | 2 |  | 0.110 | 0.308 |
| 2011 | puducherry | 1 |  | 0.812 | 0.850 |

## 6. Fit for analysis

**1991**: 25 states fit. Not fit: andaman nicobar islands, arunachal pradesh, chandigarh, chhattisgarh, dadra nagar haveli, jammu kashmir, lakshadweep, manipur, meghalaya, nct of delhi.

Fit: andhra pradesh, assam, bihar, daman diu, goa, gujarat, haryana, himachal pradesh, jharkhand, karnataka, kerala, madhya pradesh, maharashtra, mizoram, nagaland, odisha, puducherry, punjab, rajasthan, sikkim, tamil nadu, tripura, uttar pradesh, uttarakhand, west bengal

**2001**: 26 states fit. Not fit: andaman nicobar islands, arunachal pradesh, lakshadweep, manipur, meghalaya, mizoram, nagaland, odisha, west bengal.

Fit: andhra pradesh, assam, bihar, chandigarh, chhattisgarh, dadra nagar haveli, daman diu, goa, gujarat, haryana, himachal pradesh, jammu kashmir, jharkhand, karnataka, kerala, madhya pradesh, maharashtra, nct of delhi, puducherry, punjab, rajasthan, sikkim, tamil nadu, tripura, uttar pradesh, uttarakhand

**2011**: 29 states fit. Not fit: andaman nicobar islands, lakshadweep, maharashtra, manipur, meghalaya, mizoram.

Fit: andhra pradesh, arunachal pradesh, assam, bihar, chandigarh, chhattisgarh, dadra nagar haveli, daman diu, goa, gujarat, haryana, himachal pradesh, jammu kashmir, jharkhand, karnataka, kerala, madhya pradesh, nagaland, nct of delhi, odisha, puducherry, punjab, rajasthan, sikkim, tamil nadu, tripura, uttar pradesh, uttarakhand, west bengal

## 7. Coverage of the site's regression sample

Unique SHRUG parent districts behind the site's 606 districts, and how many carry a usable census share.

| round | site SHRUG parents | with a census share | usable |
|---|---|---|---|
| 1991 | 605 | 553 | 538 |
| 2001 | 605 | 561 | 509 |
| 2011 | 605 | 572 | 533 |
