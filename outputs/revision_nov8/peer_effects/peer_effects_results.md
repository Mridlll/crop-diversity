# District peer effects: are cropping patterns shaped by the neighbours?

## Status after referee (Nov 2026)

Written 7 November 2026. Referee: `referee/` folder and `referee/referee_log.txt` (script 132). Site: Neighbours page, `docs/data/site_neighbours.json`.

- "What holds" item 4, "the border gap is a rice and wheat gap": **refereed-weakened**. After geography (M2) the rice gap is **+1.95 (state wild p 0.147)** and wheat **+0.97 (0.230)**; BH q across the nine groups is 0.69 to 0.98. Rice and wheat together account for 0.43 (M2) or 0.33 (M3) of the similarity gap, oilseeds for 0.24 or 0.22. The pattern across crops is suggestive only; no single crop gap is significant once geography is held (`referee/h3_group_border_gaps.csv`).
- Item 6, cross-border Moran's I of change 0.11 (D1) and 0.12 (rice) -> **0.18 (p 0.004) and 0.15** with the Andhra Pradesh-Telangana links excluded, the values the page quotes. With state means removed the cross-border I is 0.029 (D1) and 0.007 (rice) (`referee/h5_moran_checks.csv`).
- Item 3, the geography-adjusted border effect -3.42 (1.42 / 2.24): state p 0.140, **wild p 0.056**, the weakest number on the page. It shrinks to **-1.33 to -2.64** when similarity leaves out "other", when Telangana and Andhra Pradesh are dropped, or with a correlation measure (`referee/h2_border_by_measure.csv`, `h2_leave_out.csv`). M1 (-7.23, wild p 0.001) and M3 (-4.07, wild p 0.002) hold.
- Item 2, "about 110 km": border-equivalent distance **109 km [44, 169]**; the interval is wide. 500 random partitions give a mean placebo "border" of +1.54 (SD 1.38) against the real -7.23, none as low.
- Item 7, the wave: against suitability-matched benchmarks (five most similar districts in the same state) only **soybean entry holds, 0.750 against 0.562 (p 0.007)**; cotton entry 0.917 against 0.717 (p 0.058) and chickpea entry 0.820 against 0.768 (p 0.134) do not (`referee/h6_wave_benchmarks.csv`).
- Telangana cotton borders (item 4 and 5): 8 of 8 pairs positive, exact sign-flip p **0.250**, the smallest attainable with 8 pairs; 81 percent of the gain came before the 2014 split (`referee/h4_telangana_cotton_pairs.csv`).
- Unchanged: distance decay (E1), no average widening of borders (+0.14, SE 0.92), and the hazard result, read as no evidence of learning from neighbours that the data also cannot rule out.

The original analysis below is kept unchanged for the record, apart from the sentences marked [corrected Nov 2026]. Where it disagrees with this block, this block and `outputs/ANALYSIS_INDEX.md` hold.

---

## Short answer

Neighbouring districts grow much the same crops. Two adjacent districts in the same state share on average 77% of their crop mix. That falls to 63% for same-state districts 200 to 300 km apart and to 48% at 500 to 600 km. A state border costs about as much similarity as an extra 110 km of distance. Adjacent districts on opposite sides of a border share 67%, the level same-state pairs reach at about 190 km.

About half of that border gap is geography. State borders in India tend to sit where the land changes, and adjacent districts across a border differ more in ruggedness, elevation, rainfall and irrigation than adjacent districts inside a state. With those differences held fixed, the border still costs 3.4 to 4.1 points of similarity, and the gap sits in rice and wheat, the crops states procure. Pulses and millets show no border gap at all. The gap did not widen between 1998-2002 and 2015-19 on average, though some borders moved a lot: Telangana pulled away from Andhra Pradesh, Karnataka and Maharashtra in cotton, and Jharkhand's rice share fell well below that of the Odisha districts across the border.

When a crop spreads, the districts that take it up are usually next to districts that already grow it, so expansion looks like a wave on a map. Most of that wave is regional. Districts about 100 km away that do not share a border predict take-up nearly as well as true neighbours for cotton, and for most other crops the neighbour and placebo coefficients cannot be separated. Chickpea surges and soybean are the two cases where adjacent districts predict take-up better than the placebo ring, and only same-state neighbours carry that signal. Neither survives a correction across the 14 tests. The data cannot tell copying apart from shared weather, markets and sub-state programmes, and nothing on this page should be read as learning from neighbours.

## Data and definitions

| Item | Definition |
|---|---|
| Unit | 2011 parent district (the crop-group panel unit), 605 units in the 606-district sample, 604 with a polygon |
| Years | 1998-99 to 2019-20 (1997-98 dropped). Unit-years whose annual totals are unreliable are set to missing; season measures also need a reliable season split |
| Windows | "1998-2002" = crop years 1998-99 to 2002-03; "2015-19" = 2015-16 to 2019-20. Shares are area summed over reliable years in the window divided by summed GCA |
| Crop mix | Shares of gross cropped area in nine groups: rice, wheat, maize, millets, pulses, oilseeds, cotton, sugarcane, other |
| Similarity | 1 minus half the L1 distance between two districts' nine-group share vectors, in points (100 = identical mix, 0 = no overlap) |
| D1 | Effective number of crops, exp(Shannon) over all APY crops, mean of annual values over reliable years. Matches the site's index to 1e-15 for single-child units |
| Neighbours | Queen contiguity at 500 m tolerance on 2011 units; 1,638 adjacent pairs, mean centroid distance 79 km |
| State | Present-day state, so Telangana is separate; Ladakh is folded into Jammu and Kashmir. Andhra Pradesh-Telangana pairs (a border only from 2014) are left out of the main border tests and analysed on their own |
| Geography | Long-run (1990-2019) IMD annual rainfall mean and coefficient of variation, SHRUG mean elevation and terrain ruggedness (TRI), census irrigated share of village land (2011, 2001 where 2011 is unusable) |
| Pair SEs | Dyadic-robust: any two pairs that share a district may be correlated. Reported with members = 2011 units (about 550 clusters) and members = states (24 to 29 clusters, conservative version) |
| Panel SEs | Clustered on the 2011 unit (about 600 clusters) and on state (27 to 29) |

## 1. How alike are neighbours?

### Similarity against distance

| Pairs (2015-19) | Pairs | Mean km | Similarity | Gap in D1 |
|---|---|---|---|---|
| Adjacent, same state | 1,217 | 76 | 77.1 | 1.38 |
| Adjacent, across a border | 391 | 88 | 67.4 | 1.93 |
| Not adjacent, 60-100 km, same state | 400 | 85 | 75.2 | 1.20 |
| Not adjacent, 60-100 km, across a border | 211 | 87 | 67.3 | 1.87 |
| 200-300 km, same state | 2,087 | 247 | 62.7 | 1.96 |
| 200-300 km, across a border | 5,205 | 254 | 57.7 | 2.18 |
| 500-600 km, any | 9,643 | 550 | 48.4 | 2.41 |

In 1998-2002 the same rows read 79.1, 69.5, 77.4, 67.4, 65.4, 58.0 and 49.2. Same-state pairs lost about two to three points of similarity over the two decades at every distance, while cross-border pairs lost less, so within states district crop mixes drifted apart a little. The 1998-2002 window covers fewer districts (498 against 620) because more early unit-years are unreliable.

Touching adds little once two districts are close. Same-state districts 60-100 km apart that do not touch score 75 against 77 for touching ones at about the same distance. The state line matters more: in 20 km distance bins below 250 km the cross-border curve sits 5 to 11 points below the same-state curve in 2015-19 (7 to 13 in 1998-2002), and 3 to 5 points below it beyond that. Same-state pairs fall to the similarity of adjacent cross-border pairs (67.4) at about 190 km, against 76 km for adjacent same-state pairs, which is where the figure of about 110 km comes from.

### Spatial autocorrelation

Moran's I with row-standardised queen weights, 999 permutations; every value in the table has a permutation p of 0.001 unless marked. Levels use 600 units; changes from 1998-2002 to 2015-19 use 478 (units with both windows). The cross-border column uses only links that cross a state border (356 and 266 units with such a link).

| Measure | All links | Within-state links | Cross-border links | All links, state means removed |
|---|---|---|---|---|
| D1, level 2015-19 | 0.59 | 0.64 | 0.43 | 0.30 |
| Rice share, level | 0.80 | 0.82 | 0.76 | 0.35 |
| Wheat share, level | 0.91 | 0.92 | 0.85 | 0.32 |
| Pulses share, level | 0.62 | 0.68 | 0.53 | 0.41 |
| Millets share, level | 0.65 | 0.70 | 0.45 | 0.46 |
| Oilseeds share, level | 0.69 | 0.72 | 0.62 | 0.47 |
| D1, change | 0.44 | 0.55 | 0.11 (p = 0.095) | 0.14 |
| Rice share, change | 0.47 | 0.57 | 0.12 (p = 0.066) | 0.17 |
| Wheat share, change | 0.41 | 0.43 | 0.34 | 0.24 |
| Pulses share, change | 0.37 | 0.41 | 0.27 | 0.22 |
| Millets share, change | 0.60 | 0.60 | 0.58 | 0.25 |
| Oilseeds share, change | 0.56 | 0.59 | 0.42 | 0.41 |

Levels are strongly clustered for every crop, and much of that clustering survives across state borders: wheat and rice belts run straight through them. Changes look different. Change in D1 and in the rice share is shared by neighbours inside a state (0.55 and 0.57) and barely at all by neighbours across a border (0.11 and 0.12). Millets fell together on both sides of borders (0.58 across, 0.60 within), which fits a decline driven by agro-climate and prices that pay no attention to states. Removing state means cuts every I by half or more, so a large part of the clustering in levels and most of it in changes is a state-level pattern.

## 2. State borders: geography or policy?

### Are the two kinds of adjacent pair comparable?

They are not. Adjacent pairs across a border differ more on every geographic measure than adjacent pairs inside a state:

| Mean gap between the two districts | Same state (1,235 pairs) | Across a border (395) |
|---|---|---|
| Centroid distance, km | 76 | 88 |
| Annual rainfall, mm | 182 | 254 |
| Rainfall CV | 0.037 | 0.041 |
| Elevation, m | 156 | 249 |
| Ruggedness (TRI) | 2.6 | 5.3 |
| Irrigated share, points | 13.2 | 22.8 |

### Border gap in the cross-section

Coefficient on "pair crosses a state border", adjacent pairs. Outcomes in points (similarity, absolute gap in a share) or effective crops (D1 gap). SEs dyadic-robust by 2011 unit / by state. M1 controls distance; M2 adds the five geography gaps; M3 adds state-membership FE (a cross-border pair is compared with the average internal similarity of its two states).

| Outcome, 2015-19 (n = 1,485 pairs) | Same-state mean | M1 | M2 | M3 |
|---|---|---|---|---|
| Similarity | 77.0 | -7.2 (1.6 / 2.5) | -3.4 (1.4 / 2.2) | -4.1 (1.2 / 1.8) |
| D1 gap | 1.40 | +0.34 (0.15 / 0.15) | +0.14 (0.15 / 0.16) | +0.15 (0.13 / 0.13) |
| Rice share gap | 10.4 | +4.3 (1.2 / 1.8) | +1.9 (1.2 / 1.6) | +1.8 (1.0 / 1.4) |
| Wheat share gap | 4.3 | +1.6 (0.7 / 1.2) | +1.0 (0.6 / 1.0) | +0.9 (0.5 / 0.7) |
| Pulses share gap | 6.1 | +1.0 (0.7 / 1.2) | +0.5 (0.8 / 1.5) | +0.6 (0.8 / 1.5) |
| Millets share gap | 4.5 | +0.3 (0.7 / 1.3) | -0.0 (0.8 / 1.4) | +0.3 (0.8 / 1.5) |

| Outcome, 1998-2002 (n = 1,081 pairs) | Same-state mean | M1 | M2 | M3 |
|---|---|---|---|---|
| Similarity | 79.0 | -8.2 (2.0 / 3.9) | -4.5 (1.4 / 2.5) | -4.8 (1.4 / 2.4) |
| D1 gap | 1.41 | +0.17 (0.14 / 0.18) | -0.04 (0.15 / 0.20) | +0.02 (0.12 / 0.15) |
| Rice share gap | 10.1 | +3.6 (1.4 / 2.2) | +1.6 (1.2 / 1.6) | +1.5 (1.1 / 1.2) |
| Wheat share gap | 4.1 | +1.2 (0.6 / 1.1) | +1.0 (0.6 / 1.0) | +0.5 (0.4 / 0.7) |

p-values for similarity in 2015-19: M1 < 0.001 (unit) and 0.007 (state); M2 0.016 and 0.14; M3 0.001 and 0.034. Geography removes about half the raw border gap. The irrigation gap is the geographic difference that matters most (a 10-point gap in irrigated share lowers similarity by 2.4 points).

### Has the border gap widened?

Pair-year panel of adjacent pairs, 1998-99 to 2019-20 (30,974 pair-years, 1,638 pairs). Pair FE hold every fixed difference between the two districts; year FE hold national swings. Coefficients are cross-border × period, relative to 1998-2002.

| Outcome | × 2003-07 | × 2008-12 | × 2013-19 | Linear, per decade |
|---|---|---|---|---|
| Similarity | +0.61 (0.62 / 1.12) | +0.51 (0.77 / 1.45) | +0.14 (0.92 / 1.60) | -0.01 (0.54 / 0.88) |
| Rice share gap | -0.15 (0.43 / 0.77) | +0.15 (0.56 / 1.02) | +0.68 (0.66 / 1.26) | +0.53 (0.40 / 0.69) |
| Wheat share gap | -0.16 (0.26 / 0.33) | -0.31 (0.30 / 0.44) | +0.15 (0.33 / 0.61) | +0.14 (0.19 / 0.39) |
| D1 gap | +0.14 (0.08 / 0.10) | +0.10 (0.09 / 0.10) | +0.12 (0.10 / 0.17) | +0.06 (0.06 / 0.11) |

On average, cross-border pairs did not drift apart relative to same-state pairs. The yearly means tell the same story: same-state adjacent pairs went from 78.8 similar in 1998-99 to 76.0 in 2019-20, cross-border pairs from 67.6 to 65.3.

The Andhra Pradesh-Telangana border (8 pairs) is the one border created inside the window. After 2014 its pairs did not become less alike (similarity +0.26, SE 1.09). Their rice gap narrowed by 3.6 points (SE 1.4), because Telangana's border districts moved towards the high rice shares of the Krishna and Godavari districts across the line. With two states, only unit-level SEs mean anything here.

### Border by border

Mean within-pair difference, state A minus state B, in points of GCA. This is the border gap with pair fixed effects. SE dyadic by 2011 unit. Borders with fewer than 8 pairs have unreliable SEs.

| Border (pairs) | Crop | 1998-2002 | 2015-19 | Change |
|---|---|---|---|---|
| Punjab - Haryana (12) | Rice | +8.9 (2.7) | +10.7 (3.3) | +2.7 (2.0) |
| | Wheat | +5.9 (0.5) | +4.8 (0.9) | -1.7 (0.5) |
| Haryana - Rajasthan (13) | Wheat | +9.2 (3.6) | +14.4 (2.7) | +3.6 (1.7) |
| | Pulses | -12.2 (1.4) | -13.6 (4.8) | -3.9 (4.3) |
| | Millets | +2.0 (2.7) | -4.0 (1.9) | -4.7 (0.9) |
| Punjab - Rajasthan (3) | Rice | +26.2 | +36.4 | +10.1 |
| | Wheat | +25.0 | +29.6 | +4.6 |
| Chhattisgarh - Odisha (14) | Rice | +9.2 (4.2) | +0.8 (4.3) | -7.9 (1.7) |
| Jharkhand - Odisha (6) | Rice | +7.3 (first five years) | -18.0 (2.4) | trend -16 per decade (1.3) |
| | Pulses | -2.9 (first five years) | +10.2 (1.0) | trend +8.9 per decade (0.8) |
| Telangana - Andhra Pradesh (8) | Rice | -15.8 (4.9) | -2.3 (5.8) | +13.6 (2.4) |
| | Cotton | +5.9 (7.5) | +21.0 (7.0) | +15.1 (1.0) |
| Telangana - Karnataka (7) | Cotton | +2.0 (0.8) | +17.3 (4.2) | +13.3 (0.2) |
| | Pulses | -20.0 (13.5) | -34.7 (7.6) | -7.2 (6.9) |
| | Millets | -14.4 (0.8) | -12.3 (1.2) | +2.6 (2.0) |
| Telangana - Maharashtra (6) | Cotton | +5.3 (10.2) | +15.6 (10.1) | +10.3 (4.0) |
| Andhra Pradesh - Karnataka (8) | Millets | -21.0 (2.3) | -24.1 (7.9) | +7.1 (1.6) |
| Karnataka - Maharashtra (10) | Pulses | +9.6 (7.1) | +26.9 (8.5) | +12.7 (11.5) |
| Madhya Pradesh - Uttar Pradesh (25) | Wheat | -3.6 (2.7) | -3.8 (2.4) | -0.4 (2.6) |
| | Pulses | -7.8 (4.4) | -6.9 (3.5) | +3.5 (3.9) |
| West Bengal - Bihar (5) | Wheat | -8.4 (0.8) | -6.1 (1.6) | +2.3 (1.2) |
| | Rice | +8.5 (6.4) | -2.3 (7.5) | -10.7 (5.7) |

The "change" column is the mean within-pair change, so it uses only pairs with both windows and need not equal the difference of the two level columns. Full results for 27 borders, eight crops and D1 are in `q2_border_gaps.csv`.

How much of a border gap does geography predict? Taking the slope of signed crop-share gaps on signed geography gaps from same-state adjacent pairs, and applying it across each border, the predicted gap averages 1.8 points in absolute size against a raw gap of 5.7 (256 border × crop cells, 2015-19; correlation between predicted and raw gaps 0.33). The Punjab-Haryana rice gap of 10.7 points is predicted at 1.7 from geography; the Haryana-Rajasthan wheat gap of 14.4 at 1.5; the Telangana-Karnataka pulses gap of -34.7 at -1.4. Geography as measured here explains little of the border gaps one by one. That measured geography leaves out soils and groundwater depth, and the pooled regression above, which uses absolute gaps, gives geography a larger role.

The pattern across crops is the strongest piece of evidence for policy. The average border gap sits in rice and wheat, the crops whose procurement is organised by the state. Punjab-Haryana-Rajasthan is the clearest case: across the Punjab-Rajasthan line the rice share drops by 36 points and the wheat share by 30. Haryana-Rajasthan has a wheat gap that grew to 14 points, of which geography predicts 1.5. Chhattisgarh's rice share was 9 points above the Odisha districts across the border in 1998-2002 and level with them by 2015-19, so state rice procurement in Chhattisgarh did not open a gap there. Pulses and millets carry no average border gap.

## 3. Spread episodes

### Episodes and events

Two event types. An entry is the first year a district's share crosses a level L (5 points of GCA; 2 for rabi maize) from below and stays above it. A surge is the first year the share rises L points above the district's own 1998-2000 mean and stays there. Surge suits Bt cotton, which grew mostly inside districts that already grew cotton.

| Episode | Entries | Surges | Where (surges) | Surges by period: 01-04 / 05-09 / 10-14 / 15-19 |
|---|---|---|---|---|
| Cotton | 12 (75 established) | 37 | Maharashtra 11, Telangana 8, Gujarat 8, Haryana 4 | 5 / 15 / 10 / 7 |
| Soybean | 24 (47 established) | 31 | Maharashtra 16, Madhya Pradesh 4, Rajasthan 4 | 13 / 14 / 4 / 0 |
| Maize, all seasons | 51 (135 established) | 65 | Karnataka 13, Tamil Nadu 7, Maharashtra 6, Bihar 5 | 10 / 23 / 17 / 15 |
| Rabi maize | 26 | 24 | Andhra Pradesh 5, Telangana 4, Maharashtra 4, West Bengal 4 | 1 / 8 / 11 / 4 |
| Chickpea | 50 (81 established) | 47 | Maharashtra 13, Madhya Pradesh 10, Karnataka 9 | 5 / 10 / 15 / 17 |
| Moong | 22 | 24 | Rajasthan 11, Odisha 4 | 6 / 4 / 6 / 8 |
| Rice | 7 (380 established) | 107 | Odisha 26, Bihar 11, Madhya Pradesh 10, Punjab 10 | 37 / 33 / 19 / 18 |

Telangana paddy is a statewide push, so state-by-year effects absorb it and it is covered by the border results above. Rice is the least reliable episode: Odisha's rice share was 0.61 of GCA in 1999-00, the super-cyclone year, against 0.76 to 0.85 in the years around it, so its 1998-2000 baseline is depressed and many of its 26 "surges" are a return to normal.

### Hazard: does having neighbours who already grow the crop predict take-up?

Linear probability hazard over at-risk district-years, 2001-2019, with state-by-year FE and controls for the baseline share and the five geography measures. The coefficient is the change in the yearly probability of the event as the share of growers in the set goes from 0 to 1. "Ring" = districts that do not share a border but lie at 0.75 to 1.5 times the district's mean distance to its own neighbours (2.4 districts, 104 km on average, any state). SE clustered by 2011 unit; state-clustered SE for the neighbour term in the last column. q = Benjamini-Hochberg over the 14 neighbour-minus-ring tests.

| Episode | Events | Base rate | Neighbours | Ring | Neighbours minus ring | q | Same-state nbrs | Other-state nbrs | Nbr SE, state |
|---|---|---|---|---|---|---|---|---|---|
| Chickpea surge | 33 | 0.005 | 0.094 (0.032) | 0.011 (0.018) | 0.083 (0.040), p = 0.040 | 0.39 | 0.056 (0.030) | -0.004 (0.019) | 0.034 |
| Soybean entry | 17 | 0.003 | 0.049 (0.021) | 0.008 (0.008) | 0.041 (0.021), p = 0.055 | 0.39 | 0.061 (0.035) | -0.004 (0.009) | 0.027 |
| Soybean surge | 22 | 0.003 | 0.097 (0.038) | 0.031 (0.016) | 0.066 (0.042), p = 0.11 | 0.53 | 0.037 (0.032) | 0.029 (0.022) | 0.045 |
| Maize surge | 49 | 0.007 | 0.047 (0.021) | 0.017 (0.012) | 0.030 (0.026) | 0.56 | 0.025 (0.017) | 0.018 (0.011) | 0.023 |
| Rabi maize surge | 15 | 0.003 | 0.055 (0.044) | -0.004 (0.013) | 0.059 (0.045) | 0.56 | 0.097 (0.065) | -0.016 (0.016) | 0.039 |
| Moong surge | 17 | 0.003 | 0.037 (0.028) | -0.000 (0.009) | 0.037 (0.031) | 0.56 | 0.026 (0.042) | 0.000 (0.002) | 0.012 |
| Cotton surge | 31 | 0.004 | 0.092 (0.033) | 0.085 (0.044) | 0.007 (0.063) | 0.97 | 0.080 (0.071) | 0.022 (0.026) | 0.028 |
| Rice surge | 84 | 0.014 | 0.007 (0.016) | -0.001 (0.007) | 0.008 (0.019) | 0.87 | -0.034 (0.018) | -0.005 (0.008) | 0.024 |
| Chickpea entry | 41 | 0.007 | 0.014 (0.016) | 0.013 (0.010) | 0.001 (0.020) | 0.97 | 0.074 (0.031) | -0.001 (0.012) | 0.012 |
| Cotton entry | 11 | 0.002 | 0.010 (0.007) | 0.003 (0.006) | 0.008 (0.009) | 0.75 | 0.005 (0.012) | 0.018 (0.013) | 0.004 |
| Maize entry | 38 | 0.008 | 0.008 (0.011) | 0.003 (0.005) | 0.005 (0.012) | 0.87 | 0.009 (0.014) | 0.005 (0.008) | 0.012 |
| Rabi maize entry | 18 | 0.003 | -0.004 (0.026) | 0.008 (0.010) | -0.012 (0.029) | 0.87 | 0.016 (0.037) | -0.020 (0.010) | 0.025 |
| Moong entry | 18 | 0.003 | -0.002 (0.011) | 0.004 (0.005) | -0.006 (0.012) | 0.87 | -0.002 (0.014) | -0.007 (0.005) | 0.015 |
| Rice entry | 6 | 0.004 | -0.003 (0.011) | -0.004 (0.005) | 0.001 (0.012) | 0.97 | | | 0.008 |

Events in the hazard sample can be fewer than in the episode table because units without a ring set or a geography measure drop out. Unit-years at risk run from 1,898 (rice entry) to 8,497 (moong surge).

For scale: a district with one of its five neighbours newly surging into chickpea has its yearly chance of a chickpea surge raised by about 0.019 against a base of 0.005. The same-state non-neighbour share, added on top of neighbours and ring, is negative or null in every episode (for example chickpea surge -0.13, SE 0.09), as state-by-year FE require.

Threshold sensitivity (neighbours minus ring, unit SE, at 0.6× / 1× / 2× the level): chickpea surge 0.079 (0.033) / 0.083 (0.040) / 0.088 (0.047); soybean entry -0.009 (0.023) / 0.041 (0.021) / 0.092 (0.046); soybean surge 0.095 (0.037) / 0.066 (0.042) / 0.165 (0.062). Chickpea surge holds across thresholds. Soybean entry vanishes at the low threshold.

### Wave or jump

For each event, is the district adjacent to a district that already grew the crop the year before? The comparison is 999 random reassignments of each year's events among that year's at-risk districts, nationally and within state.

| Episode | Events | Adjacent to a grower | Random, national | Random, within state | p (within state) | Median km to nearest grower | Share more than 150 km away |
|---|---|---|---|---|---|---|---|
| Cotton entry | 12 | 92% | 19% | 64% | 0.026 | 100 | 8% |
| Cotton surge | 32 | 75% | 8% | 49% | 0.002 | 97 | 19% |
| Soybean entry | 24 | 75% | 12% | 47% | 0.001 | 87 | 21% |
| Soybean surge | 29 | 41% | 7% | 25% | 0.027 | 173 | 52% |
| Maize entry | 51 | 82% | 52% | 64% | 0.002 | 86 | 2% |
| Maize surge | 60 | 47% | 18% | 34% | 0.011 | 127 | 40% |
| Rabi maize entry | 26 | 50% | 9% | 37% | 0.056 | 107 | 42% |
| Chickpea entry | 50 | 82% | 21% | 66% | 0.001 | 87 | 12% |
| Chickpea surge | 44 | 64% | 10% | 33% | 0.001 | 121 | 34% |
| Moong surge | 23 | 52% | 7% | 23% | 0.001 | 158 | 52% |

Entries into a new crop move like a wave: three-quarters or more of cotton, soybean, maize and chickpea entries happen next to a district that already grew the crop, well above what random placement inside the same state gives. Surges are patchier. A third to a half of soybean, maize, chickpea and moong surges happen more than 150 km from the nearest earlier surge, so intensification jumps. Contiguity on a map is the expected footprint of both contagion and a moving suitability or market frontier, and the hazard results above, with the ring placebo, are the test that separates them.

## 4. What the evidence can and cannot say

Districts that look alike because they are neighbours could look alike for four different reasons, and the evidence speaks to each differently.

**Shared agro-climate.** This is the largest part of the similarity in levels. Similarity falls smoothly with distance on both sides of state borders, rice and wheat belts cross borders (Moran's I across borders 0.76 and 0.85), and adjacent pairs that differ in irrigation, ruggedness or rainfall are less alike. Measured geography explains half the cross-border gap in the pooled regressions and little of any single border's gap, so unmeasured soils and groundwater probably account for more.

**Shared shocks.** The earlier year-on-year work found that adjacent districts' shares move together strongly in the same year (0.53 to 0.70 for pulses, millets and oilseeds), that the lagged link is about the same size looking backwards as forwards, and that same-state districts 100 km away predict it half as well. Weather and prices that hit a region together fit this.

**State policy and state markets.** The border results are the cleanest evidence on this channel. Crossing a state line costs 3.4 to 4.1 points of similarity after geography, the cost sits in rice and wheat, changes in rice and in diversity are shared within states and hardly across borders, and where a crop does spread from neighbours (chickpea, soybean) only same-state neighbours carry the signal. State procurement, input supply, extension and state-run seed and irrigation programmes are all candidates. These data cannot separate them, and sub-state programmes targeted at clusters of districts (NFSM-pulses, A3P) would produce the same within-state neighbour pattern.

**Learning from neighbours.** This would show up as neighbours predicting take-up better than equally close non-neighbours, at a lag, and also across state borders, since farmers and traders cross them. Two episodes pass the first part (chickpea surges, soybean), neither passes a correction across 14 tests, and in both the effect stops at the state border. Cotton, the textbook diffusion story, gives a neighbour coefficient matched by the ring placebo. The evidence does not show district-to-district learning. It cannot rule it out either, because the confidence intervals on the neighbour-minus-ring differences are about ±0.04 to ±0.12 against base rates of 0.003 to 0.014.

Three limits apply throughout. District shares are coarse, so copying between farmers inside a district or across a short stretch of border is invisible. The reflection problem is not solved here: a neighbour coefficient that survives the ring placebo can still come from a shock that is sharper near the border than 100 km away. The APY records are reported by state departments, so a reporting practice that differs by state looks like a state border effect.

## What holds

1. Adjacent districts in the same state share 77% of their crop mix (1,217 pairs, 2015-19). Similarity falls to 63% at 200-300 km and 48% at 500-600 km.
2. A state border lowers similarity between adjacent districts by 7.2 points (SE 1.6 by unit, 2.5 by state; 1,485 pairs). That is about what an extra 110 km of distance does inside a state.
3. Cross-border neighbours differ more in geography (twice the ruggedness gap, 1.7 times the irrigation gap). Holding the geography gaps fixed halves the border effect to 3.4 points (SE 1.4 / 2.2); with state-membership FE it is 4.1 (1.2 / 1.8).
4. The border gap is a rice and wheat gap. Pulses and millets have none on average. Punjab-Rajasthan (rice 36 points, wheat 30), Haryana-Rajasthan (wheat 14), Punjab-Haryana (rice 11) and Telangana's cotton borders (15 to 21 points) are the large ones, and measured geography predicts a small part of each.
5. Cross-border gaps did not widen on average from 1998-2002 to 2015-19 (similarity × 2013-19 = +0.14, SE 0.92). Some borders moved: Telangana's cotton gap with each neighbour widened by 10 to 15 points, the Jharkhand-Odisha rice gap went from +7 to -13 points in yearly means (first and last five years), and the Chhattisgarh-Odisha rice gap closed.
6. Changes in diversity and rice share cluster within states (Moran's I 0.55, 0.57) and hardly across borders (0.11, 0.12). Millet decline clusters on both sides alike (0.60, 0.58).
7. New crops arrive next to districts that already grow them: 75 to 92% of entries into cotton, soybean, maize and chickpea are adjacent to an existing grower, against 47 to 66% under random placement within the state.
8. In the hazard tests, neighbours beat the 100 km placebo ring for chickpea surges (0.083, SE 0.040) and soybean entry (0.041, SE 0.021), carried only by same-state neighbours, and neither survives multiple-testing correction (q = 0.39). Cotton surges are regional (neighbours 0.092, ring 0.085).

**Does not hold:** any claim that districts learn crops from their neighbours, any claim that state borders have pushed crop mixes further apart over time on average, and any single-border gap resting on fewer than 8 pairs.

## Proposed visuals

1. **Similarity against distance, two lines** (same state, across a border), 2015-19 with 1998-2002 as a faint pair. One picture shows that closeness matters and that a border is worth about 110 km. Data: `chart_data/v1_similarity_vs_distance.csv` (or `.json`, which carries the reference points).
2. **Border dumbbells**, one row per border, two dots for the border districts on each side, small multiples for rice, wheat, pulses and millets, with the 1998-2002 position as a hollow dot. Makes the rice-and-wheat concentration of border gaps visible at a glance. Data: `chart_data/v2_border_dumbbells.csv`.
3. **Neighbour versus placebo bars** per episode: neighbours, the 100 km ring, and the difference with CI, sorted by the difference. The single honest test in one chart. Data: `chart_data/v5_neighbour_vs_placebo.csv`.
4. **Small-multiple maps of a crop's spread**, chickpea surges (or soybean entries) in 2001, 2005, 2009, 2013, 2017, 2019, shading districts by the year they took the crop up. Lets readers see the wave and the jumps for themselves. Data: `chart_data/v4_spread_maps.json`, joined to the site polygons with `chart_data/v4_polygon_to_unit.csv` (geojson `n`, `s`, `k` to unit).
5. **"How alike is your neighbour" explorer**: pick a district, see its nine-group bar next to each neighbour's, with the similarity score and a flag for neighbours across a state line, plus the district's percentile among all districts. Data: `chart_data/v6_neighbour_explorer.json` (602 districts).
6. **Border gap over time**, one line per chosen border and crop (Telangana-Andhra Pradesh cotton and rice, Jharkhand-Odisha rice, Chhattisgarh-Odisha rice, Haryana-Rajasthan wheat), with the average same-state and cross-border similarity lines as context. Answers whether borders are pulling apart. Data: `chart_data/v3_border_gap_over_time.csv`.

Secondary: a two-bar Moran's I chart (within-state links against cross-border links, levels and changes) from `chart_data/v7_moran.csv`, and an observed-versus-random contiguity chart from `chart_data/v8_wave.csv`.

## Files

Scripts (run in order, about 75 seconds in total): `scripts/125_peer_effects_build.py`, `126_peer_effects_similarity.py`, `127_peer_effects_borders.py`, `128_peer_effects_spread.py`, `129_peer_effects_chart_data.py`, helper `scripts/_pe_common.py`.

Outputs in `outputs/revision_nov8/peer_effects/`:

- Base data: `pe_units.csv`, `pe_unit_year.csv`, `pe_unit_period.csv`, `pe_pairs.csv.gz`, `pe_build_log.txt`
- Q1: `q1_similarity_by_distance.csv`, `q1_similarity_summary.csv`, `q1_moran.csv`, `q1_log.txt`
- Q2: `q2_crosssection.csv`, `q2_panel_widening.csv`, `q2_pair_type_year_means.csv`, `q2_border_gaps.csv`, `q2_border_annual.csv`, `q2_border_trends.csv`, `q2_geography_gaps.csv`, `q2_log.txt`
- Q3: `q3_episode_summary.csv`, `q3_entry_years.csv`, `q3_hazard.csv`, `q3_wave.csv`, `q3_entries_detail.csv`, `q3_unit_year_status.csv.gz`, `q3_log.txt`
- Chart data: `chart_data/v1` to `v8`
