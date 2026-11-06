# Q1. Crop-group trends 1998-99 to 2019-20, and the Andhra Pradesh / Telangana rice check

Script: `scripts/99_crop_groups_trends.py` (about 1 minute). Full printed output: `trends_log.txt`.
Input is the crop-group panel from scripts 97 and 98 (APY area, 2011 parent districts). The crop-name mapping is the panel's: rice; wheat; maize; millets = jowar, bajra, ragi, small millets; pulses = the 13 APY pulse names; oilseeds = the 10 listed oilseeds; cotton; sugarcane; other = the remaining 22 names, including coconut, guar, barley and other cereals (see `crop_group_panel_results.md`). Outcomes are shares of annual gross cropped area (GCA) and area in hectares. No APCNF, PMDS or RySS data is used.

## Method

- Units: the 605 2011-parent units that hold the 606-district sample. Andhra Pradesh is split into its 13 residual districts and the 9 Telangana parents in the sample (Hyderabad has no usable crop area).
- Years: 1998-99 to 2019-20. 1997-98 is dropped because it is weak in several states.
- A unit-year is used only when the panel's `flag_annual_unreliable` is 0. One repair was made: 5 pre-split donor links cross a state border, which a district split cannot do (for example Chittoor flagged as the donor of Chikkaballapura). Those flags are lifted, which frees 29 unit-years. 12,062 of 12,814 unit-years are usable.
- Endpoint panel: units with at least 2 usable years in both windows, 1998-99 to 2000-01 and 2017-18 to 2019-20. That leaves 467 units. Each window value is the window's summed group area over its summed GCA. Of the 138 units that drop out, 55 do not exist yet in 1998-2000 (late entrants). The other 83 are flagged in that window: 57 as pre-split donors and 42 for a severe GCA jump (a unit can carry both flags).
- Area-weighted change is the change in the aggregate share (sum of area over sum of GCA). Its standard error comes from a bootstrap over units (1,000 draws). District-mean change is the plain mean of unit changes, with SE equal to sd / sqrt(n).
- Trend slopes come from a regression of the share (pp) on year with unit fixed effects, with SE clustered on unit (the 2011 parent, so 467 clusters). They are reported in pp per decade.
- Hot spots use Getis-Ord Gi* on the change in rice share, with queen adjacency plus self from script 97.

## 1. National (467 units)

Change in share of GCA, 1998-2000 to 2017-2019, in percentage points (SE in brackets). Area change is in thousand ha.

| group | share 1998-2000 | share 2017-2019 | area-weighted change | district-mean change | slope, area-wtd, pp/decade | slope, district-mean | area change, kha | districts up / down |
|---|---|---|---|---|---|---|---|---|
| rice | 25.9 | 23.9 | -1.98 (0.45) | -0.47 (0.42) | +0.15 (0.16) | -0.21 (0.22) | +748 (702) | 199 / 254 |
| wheat | 16.1 | 17.5 | +1.34 (0.34) | +0.59 (0.21) | +0.58 (0.15) | +0.39 (0.11) | +4,708 (671) | 203 / 215 |
| maize | 3.5 | 4.7 | +1.27 (0.33) | +0.74 (0.27) | +0.68 (0.17) | +0.45 (0.14) | +2,625 (550) | 220 / 235 |
| millets | 13.4 | 7.9 | -5.55 (0.59) | -3.16 (0.33) | -3.30 (0.32) | -1.88 (0.17) | -7,054 (930) | 72 / 358 |
| pulses | 12.9 | 15.1 | +2.26 (0.64) | +0.60 (0.38) | +0.53 (0.29) | +0.29 (0.18) | +5,724 (1,244) | 230 / 235 |
| oilseeds | 14.4 | 14.3 | -0.14 (0.63) | -1.09 (0.37) | -0.70 (0.36) | -0.59 (0.21) | +2,013 (1,079) | 179 / 288 |
| cotton | 5.5 | 7.2 | +1.73 (0.53) | +0.80 (0.27) | +1.11 (0.31) | +0.56 (0.16) | +3,696 (928) | 128 / 156 |
| sugarcane | 2.5 | 2.8 | +0.26 (0.14) | +0.20 (0.13) | +0.27 (0.10) | +0.12 (0.07) | +814 (264) | 159 / 278 |
| other | 5.9 | 6.7 | +0.81 (0.33) | +1.80 (0.31) | +0.69 (0.18) | +0.87 (0.16) | +2,247 (601) | 268 / 194 |

GCA in these units rose from about 124 to 139 million ha, so a flat area becomes a falling share.

### Robustness: area-weighted change, pp

| group | main (467) | strict balanced, all 22 years usable (352) | 5-year windows (470) | end window 2016-18 (467) | drop units with Whole Year share >= 20% (413) | original flag (464) |
|---|---|---|---|---|---|---|
| rice | -1.98 | -1.42 | -2.08 | -2.25 | -1.72 | -1.94 |
| wheat | +1.34 | +1.18 | +1.21 | +0.93 | +1.37 | +1.31 |
| maize | +1.27 | +1.37 | +1.16 | +1.24 | +1.16 | +1.28 |
| millets | -5.55 | -5.74 | -5.29 | -5.24 | -5.65 | -5.56 |
| pulses | +2.26 | +2.40 | +2.00 | +2.63 | +2.14 | +2.21 |
| oilseeds | -0.14 | -0.77 | +0.48 | +0.02 | +0.09 | -0.13 |
| cotton | +1.72 | +1.85 | +1.60 | +1.42 | +2.03 | +1.74 |
| sugarcane | +0.26 | +0.20 | +0.11 | +0.26 | +0.27 | +0.28 |
| other | +0.81 | +0.93 | +0.81 | +0.98 | +0.31 | +0.82 |

District-mean changes are just as stable across specifications (tables in `trends_national_endpoint.csv`). All-India June-September rainfall departure in the two windows is +1.9, -4.2 and -8.5 percent at the start and -1.9, -6.6 and +12.5 at the end, so neither window is a drought window.

What holds and what does not:
- **Millets are the one large, uniform change.** They lose 5.5 pp of GCA (7.1 million ha), and 358 districts lose against 72 that gain. This holds in every specification and in the slopes.
- **Wheat, maize and cotton gain** 1.3 to 1.7 pp each, robustly.
- **Rice share does not have a robust trend.** The endpoint comparison gives -2.0 pp area-weighted, but the district mean (-0.5, SE 0.4) and both fixed-effect slopes (+0.15 and -0.21 pp per decade) are indistinguishable from zero. Rice area is flat at about 29 million ha in the balanced panel until 2019-20; the share falls because GCA grows.
- **Pulses rise in the aggregate (+2.3 pp, SE 0.6), but the typical district does not.** The district mean is +0.6 (SE 0.4), and 230 districts gain against 235 that lose. The gain is concentrated: Rajasthan +2.5 and Madhya Pradesh +2.4 million ha, with Karnataka +1.2, while Gujarat loses 0.8. In the national series the pulses share is flat at 11 to 13 percent until 2015-16 and then jumps to 15 to 16 percent, so the gain is a post-2015 step rather than a steady trend.
- **Oilseeds have no aggregate change but do decline in the typical district** (-1.1 pp, SE 0.4; 288 districts down against 179 up). Soybean growth in Maharashtra and eastern Madhya Pradesh offsets groundnut losses in the south and Gujarat.

## 2. States

Area-weighted change in share, pp, for states ranked by 1998-2000 GCA. The full table with SEs is in `trends_state_endpoint.csv`; the figure is `fig_trends_state_heatmap.png`.

| state (n units) | rice | wheat | maize | millets | pulses | oilseeds | cotton | sugarcane | other |
|---|---|---|---|---|---|---|---|---|---|
| Uttar Pradesh (67) | -0.4 | +1.2 | -1.1 | -1.0 | -1.2 | +1.0 | 0.0 | +1.0 | +0.6 |
| Maharashtra (31) | -0.3 | 0.0 | +3.6 | -19.1 | +2.1 | +6.7 | +5.0 | +1.8 | +0.2 |
| Rajasthan (29) | -0.1 | -1.6 | -1.2 | -6.0 | +8.2 | +1.2 | -0.3 | -0.1 | -0.1 |
| Madhya Pradesh (40) | -0.1 | +5.5 | +1.3 | -3.7 | +3.1 | -7.7 | -0.5 | +0.1 | +2.1 |
| Karnataka (23) | -4.4 | -0.9 | +7.5 | -12.2 | +11.6 | -9.4 | -1.0 | +1.6 | +7.2 |
| West Bengal (18) | -6.0 | -3.1 | +3.0 | -0.1 | +2.7 | +4.7 | 0.0 | -0.1 | -1.1 |
| Andhra Pradesh, residual (13) | -7.3 | 0.0 | +3.3 | -1.6 | +3.8 | -1.8 | +2.7 | -1.1 | +2.1 |
| Gujarat (12) | +1.0 | +5.9 | -0.8 | -7.6 | -12.0 | -3.0 | +15.1 | -0.1 | +1.6 |
| Bihar (32) | +0.7 | +1.6 | +1.5 | -0.3 | -3.2 | -0.7 | 0.0 | +2.2 | -1.7 |
| Odisha (30) | +10.2 | -0.2 | -0.2 | -1.4 | -2.9 | -4.9 | +0.8 | -0.2 | -1.3 |
| Punjab (14) | +7.0 | -0.8 | -0.9 | 0.0 | -0.5 | -0.4 | -4.2 | -0.2 | 0.0 |
| Haryana (17) | +3.7 | -0.6 | -0.2 | -5.0 | -3.0 | +1.5 | +0.4 | -0.8 | +3.9 |
| Tamil Nadu (23) | +3.3 | 0.0 | +4.3 | -3.4 | +2.4 | -9.2 | -0.2 | -1.8 | +4.7 |
| Telangana (9) | +7.8 | -0.2 | +1.6 | -11.9 | -5.4 | -8.4 | +17.8 | -0.6 | -0.8 |
| Chhattisgarh (14) | -1.1 | +0.8 | +0.5 | -2.2 | +2.5 | -1.4 | 0.0 | +0.7 | +0.2 |

Tamil Nadu rice is reported once a year until 2016-17, which affects its season split but not its annual share. The Odisha +10.2 rests on a 1998-2000 base; Odisha 1997-98 and 2012-13 are flagged, and its pulses area fell sharply after 1999.

## 3. Districts that gained or lost most

Ranked by change in share, among the 439 units with at least 25,000 ha of GCA in both windows. Full lists, including rankings by area, are in `trends_top_districts.csv`, and every unit's change is in `trends_district_endpoint.csv`. Maps are in `fig_trends_change_maps.png`.

| | largest gains, pp (area change, kha) | largest losses, pp (area change, kha) |
|---|---|---|
| pulses | Bijapur KA +47.9 (+528); Pali RJ +37.6 (+296); Gadag KA +27.5; YSR Kadapa AP +27.2; Bundi RJ +26.8; Raichur KA +25.2; Nagaur RJ +24.8 (+422) | Datia MP -20.0; Sabar Kantha GJ -18.4; Jamnagar GJ -17.8; Rajkot GJ -17.0 (-146); Bhind MP -16.5; Amreli GJ -16.3; East Godavari AP -14.7 |
| millets | Dausa RJ +17.1; Morena MP +15.2 (+111); Namakkal TN +11.4; Jaipur RJ +8.5 (+101); Alwar RJ +7.2 | Nashik MH -39.1 (-342); Bijapur KA -34.2 (-258); Bid MH -28.9; Pune MH -28.7 (-397); Rangareddy TS -28.6; Dhule MH -27.9; Aurangabad MH -26.9 |
| oilseeds | Washim MH +40.9; Buldana MH +29.8 (+302); Akola MH +27.6; Bidar KA +26.9; Nanded MH +26.1 (+293); Latur MH +25.8 (+290) | Hoshangabad MP -36.2; YSR Kadapa AP -31.7 (-145); Amreli GJ -30.1 (-187); Sheopur MP -28.9; Narsimhapur MP -25.8; Nagpur MH -23.8 |

The pattern is regional: millets lost across the Deccan (Maharashtra lost 3.45 million ha, Karnataka 1.05 and Telangana 0.55); pulses gained in Rajasthan, Madhya Pradesh and north Karnataka and lost in Saurashtra and north Gujarat (into cotton); oilseeds (soybean) gained in Vidarbha and Marathwada and lost in the Narmada valley (into wheat) and the groundnut belt. Small units carry implausible changes (Leh-Ladakh pulses 76 to 3 percent and millets 0 to 63 percent; Gondiya millets 25 to 0 percent) and are excluded from the ranking by the size filter or should be read as data problems.

## 4. Andhra Pradesh and Telangana: movement into rice?

**The claim holds for Telangana and fails for residual Andhra Pradesh.**

| | residual Andhra Pradesh (13) | Telangana (9) |
|---|---|---|
| rice share, 1998-2000 to 2017-2019 | 38.3 to 31.0, **-7.3 pp (SE 1.9)** | 31.1 to 39.0, **+7.8 pp (SE 1.8)** |
| rice area change | **-442 kha (SE 142)** | **+882 kha (SE 226)** |
| of which kharif / rabi | -345 / -97 kha | +269 / +613 kha |
| GCA change | +237 kha (SE 402) | +1,301 kha (SE 260) |
| difference from rest of India in district change of rice share (HC1) | -4.5 pp (SE 2.3), p = 0.047 | +6.8 pp (SE 2.1), p = 0.001 |
| same, change in log rice area | -0.09 (SE 0.09), p = 0.34 | +0.55 (SE 0.10), p < 0.001 |
| rice share slope, unit FE, pp per decade | -1.85 (SE 1.09) | +3.24 (SE 0.80); +1.70 (SE 0.71) without 2019-20 |

District detail (`trends_apts_districts.csv`):
- **Telangana:** rice share rose in 8 of 9 parents. In pp, with the area change: Karimnagar +17.2 (+177 kha), Nalgonda +12.1 (+243), Nizamabad +8.2 (+146), Warangal +6.9 (+116), Mahbubnagar +6.1, Medak +5.4, Adilabad +4.6 and Khammam +2.7. Only Rangareddy fell (-6.3). Karimnagar and Nalgonda sit in the top 6 percent of all 467 districts.
- **Andhra Pradesh:** rice share fell in 10 of 13 districts. The Godavari-Krishna delta districts lost the most: West Godavari -22.2 pp, Krishna -15.3 (-123 kha), Prakasam -13.6 and Guntur -8.6 (-85 kha). Only Vizianagaram (+8.4, on falling GCA), Nellore (+3.7) and Visakhapatnam (+3.3) rose.

**Where Telangana's rice land came from.** Shares of GCA, pp (SE): millets -11.9 (3.1), oilseeds -8.4 (3.5) and pulses -5.4 (1.8) lost share, while cotton gained more than rice, +17.8 (3.3). In hectares the picture is different: GCA grew by 1.3 million ha, and two thirds of the extra rice was rabi (yasangi) rice. Millets lost 553 kha, oilseeds 326 and pulses 142, about 1.0 million ha in all, against +882 for rice and +1,270 for cotton. Most of the new rice is better described as added rabi cropping than as kharif land taken from other crops. The kharif dryland that millets, oilseeds and pulses gave up went mostly into cotton. Pooled across the 11 AP and Telangana districts whose rice share rose by at least 2 pp, the result is the same: rice +7.8 pp, cotton +15.5, millets -10.0, oilseeds -8.2, pulses -4.9.

**In residual Andhra Pradesh,** the share rice lost went to pulses (+3.8 pp, SE 3.2), maize (+3.3, SE 0.7), cotton (+2.7, SE 1.3) and other crops (+2.1, SE 1.0). The delta districts also lost GCA: Krishna lost 55 kha and Guntur 58. APY cannot show what that land became; aquaculture conversion is the usual explanation, but it is not tested here.

**Timing caveats that matter:**
- Telangana's rice area moves between 0.9 and 2.0 million ha up to 2018-19 (`fig_trends_apts_rice.png`) and then reaches 3.2 million ha in 2019-20. With 2016-18 as the end window, the Telangana gain halves to +3.1 pp and +419 kha. Comparing 2008-10 with 2016-18 gives +2.7 pp. A steady rise is present, but half of the measured endpoint gain is the single year 2019-20. The panel stops there, so the well-known post-2019 expansion is outside the data.
- The start window is not a drought window for these states (district-mean June-September departure: AP +16, -26 and +21 percent; TS +17, -11 and +25), so the baseline is not artificially low. The 2002-04 drought troughs are visible in the series and fall outside both windows.
- **On the national map, AP and Telangana are not where rice expanded most.** Gi* on the change in rice share finds 41 hot-spot units at z > 1.96: 21 in Odisha, 8 in Punjab, and only 2 in Telangana (Karimnagar, Warangal) plus 1 in AP (Visakhapatnam). After a Bonferroni correction one unit survives, in Odisha. On the change in log rice area, the hot spots are in Madhya Pradesh, Haryana and Rajasthan (21 units), and none are in AP or Telangana. Telangana is a clear rice gainer by state comparison, but it is not a spatial cluster that stands out nationally.

## 5. Millets falling, maize rising

Among the 182 units where millets held at least 5 percent of GCA in 1998-2000:
- 149 lost at least 1 pp of millet share. 55 of these also gained at least 1 pp of maize share, and 49 of them saw millet area fall while maize area rose. These substitution districts are in Karnataka (15), Maharashtra (13), Tamil Nadu (7), Madhya Pradesh (6), Telangana (2), Chhattisgarh (2) and 4 other states (`trends_millet_maize_substitution.csv`). The clearest cases: Nashik (millets -39.1 pp, maize +20.7), Rangareddy (-28.6, +17.4), Jalgaon (-18.9, +17.2), Chhindwara (-16.1, +29.3), Betul (-15.3, +17.3), Bellary (-13.6, +19.9), Hassan (-12.5, +19.4) and Haveri (-8.3, +31.5).
- Across districts, the change in millet share and the change in maize share correlate at r = -0.29 (95% CI -0.42 to -0.16, n = 182). Each hectare of millets lost goes with 0.20 ha of maize gained (SE 0.05).
- Within districts over time, with unit and year fixed effects, maize share falls 0.15 pp for each pp of millet share (SE 0.03, 182 clusters, 3,959 unit-years). Shares sum to one, so part of this is mechanical. In log area the relation is null (0.03, SE 0.06).
- **Maize is a minor destination for lost millet land.** In the 141 districts that lost at least 2 pp of millets, millets fell 13.1 pp (7.1 million ha). The share went to cotton (+4.2 pp, SE 1.2), pulses (+3.5, SE 1.4), maize (+3.2, SE 0.6), other crops (+1.4) and wheat (+1.3). Even in the 49 substitution districts, maize gained 2.4 million ha against 4.0 million ha of millets lost. Maharashtra is the extreme case: 3.45 million ha of millets lost against 0.75 of maize gained (the rest went to soybean, cotton and pulses).

## Takeaways

1. Over two decades the main change in what India grows is the fall of millets: -5.5 pp of GCA (SE 0.6), about 7 million ha, in 358 of 467 districts and concentrated in the Deccan. Wheat, maize and cotton each gained 1.3 to 1.7 pp. Rice share has no trend that survives a change of method.
2. Pulses gained in the aggregate (+2.3 pp) through a post-2015 step in a few regions (Rajasthan, Madhya Pradesh, north Karnataka), but the typical district shows no gain. Oilseeds are flat in the aggregate and declining in most districts, with Maharashtra soybean offsetting southern and Gujarat groundnut.
3. The claim that rice is spreading in the region holds for Telangana and is the opposite for residual Andhra Pradesh. Telangana's rice share rose 7.8 pp (SE 1.8) and its rice area by 0.9 million ha, mostly as rabi rice on newly added cropped area, while its dryland millets, oilseeds and pulses went largely into cotton. Half of the gain is 2019-20 alone. Andhra Pradesh lost 7.3 pp of rice share (SE 1.9), led by the delta districts. Nationally, the strongest rice-gain clusters are in Odisha and Punjab, not here.
4. Millets-to-maize substitution is real in about 50 districts of Karnataka, Maharashtra, Tamil Nadu and Madhya Pradesh. Even there, maize took only about 0.6 ha for each hectare of millets lost, and nationally cotton and pulses absorbed more of the lost millet share than maize did.

## Files

- `trends_national_endpoint.csv`, `trends_national_fe_slopes.csv`, `trends_national_series.csv`
- `trends_state_endpoint.csv`, `trends_state_change_pp.csv`
- `trends_district_endpoint.csv`, `trends_endpoint_panel_full.csv` (includes Gi* and national ranks), `trends_top_districts.csv`
- `trends_apts_districts.csv`, `trends_apts_accounting.csv`, `trends_apts_year_series.csv`
- `trends_millet_maize_substitution.csv`, `trends_millet_loss_accounting.csv`
- `fig_trends_national_shares.png`, `fig_trends_state_heatmap.png`, `fig_trends_change_maps.png`, `fig_trends_apts_rice.png`, `fig_trends_millet_maize.png`
- `trends_log.txt`
