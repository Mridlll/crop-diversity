# Q4 Neighbour spread: do districts follow their neighbours into pulses, millets or oilseeds?

## Answer

No crop group shows a change that follows the neighbours' change and not the other way round. Pulses come closest. A district's pulse share rises by about 0.07 points for every point its neighbours' share rose the year before, and the estimate is borderline (p = 0.086). The neighbours' change in the following year predicts the district's change just as well, so the timing gives no sign of spread. Same-state districts at a similar distance that do not share a border also predict it, at about half the size. Millets show nothing, and oilseeds show nothing at the annual level. The pattern fits districts in one region moving together under common weather, prices and programmes. It does not show neighbours copying each other.

## Method

- **Unit and years:** the 606-district analysis sample on 2011 parent boundaries (605 units), crop years 1997-98 to 2019-20.
- **Outcome:** the year-on-year change in a group's share of annual gross cropped area, dS(i,t) = s(i,t) - s(i,t-1). Unit-years with unreliable annual totals are set to missing, so a change is only formed between two reliable years.
- **Spatial lag:** the mean of the neighbours' dS(j,t-1). Neighbours come from queen contiguity at a 500 m tolerance; the mean is about 5.5 neighbours with centroids about 76 km apart. Neighbours outside the sample and across state borders are included. The mean needs at least half the neighbours observed.
- **Main specification (M2):**

  dS(i,t) = b * W dS(t-1) + r * dS(i,t-1) + rain + district FE + state-by-year FE + e

- **Controls:** the district's own lagged change soaks up its own mean reversion. The rain terms are the district's own June-September rainfall z-score and its square, the rabi-season (January-May of t+1) z-score, the lags of both, and the neighbours' mean of the same z-scores, current and lagged. Rain is IMD gridded data, aggregated to the 2011 unit by area. Rows the IMD quality check flags are dropped.
- **Errors:** clustered on the 2011 parent unit (602 clusters in the estimation sample). State clustering (27 clusters) is a robustness check.
- **Fixed effects:** state-by-year FE remove every shock common to a state in a year, including state programmes, state procurement and statewide weather. What is left is co-movement inside a state that is concentrated among adjacent districts.
- **Estimation:** within-transformation by alternating projections, then OLS. Singletons are dropped.

## Sample

| | Value |
|---|---|
| Main estimation sample | 9,866 district-years, 602 units, 1999-00 to 2019-20 as t (two lags needed) |
| Without rain controls (M1) | 10,525 district-years, 604 units |
| Distance-placebo sample | 6,803 district-years, 431 units (432 of 605 units have at least one same-state non-neighbour at a matched distance) |
| SD of the change in share (pulses / millets / oilseeds) | 0.041 / 0.031 / 0.038 |
| SD of the neighbours' lagged change | 0.029 / 0.019 / 0.026 |
| Mean share (pulses / millets / oilseeds) | 0.099 / 0.076 / 0.104 |

## Headline estimates (coefficient on the neighbours' change last year; SE clustered by unit)

| Group | M1, no rain | M2, main | 95% CI (M2) | p (M2) | BH q over 9 groups | n |
|---|---|---|---|---|---|---|
| Pulses | 0.072 (0.036), p = 0.044 | 0.067 (0.039) | -0.009 to 0.144 | 0.086 | 0.26 | 9,866 |
| Millets | -0.042 (0.029), p = 0.15 | -0.037 (0.031) | -0.097 to 0.024 | 0.24 | 0.36 | 9,866 |
| Oilseeds | 0.044 (0.033), p = 0.17 | 0.036 (0.033) | -0.029 to 0.101 | 0.28 | 0.36 | 9,866 |
| Rice | | 0.026 (0.022) | -0.017 to 0.069 | 0.24 | 0.36 | 9,866 |
| Wheat | | 0.041 (0.028) | -0.015 to 0.096 | 0.15 | 0.34 | 9,866 |
| Maize | | 0.060 (0.031) | 0.000 to 0.120 | 0.049 | 0.22 | 9,866 |
| Cotton | | 0.008 (0.049) | -0.088 to 0.104 | 0.87 | 0.87 | 9,866 |
| Sugarcane | | 0.017 (0.035) | -0.053 to 0.086 | 0.63 | 0.71 | 9,866 |
| Other | | 0.108 (0.025) | 0.060 to 0.156 | <0.001 | <0.001 | 9,866 |

**Size:** for pulses, a one-SD rise in the neighbours' change (2.9 points of gross cropped area) goes with a 0.19-point rise in the district's own share the next year. That is about 5% of an SD of the change.

**Own lagged change:** strongly negative for every group (pulses -0.36, SE 0.035), so year-on-year shares mean-revert hard.

**Contemporaneous co-movement (for reference):** the neighbours' change in the same year has a coefficient of 0.57 (SE 0.04) for pulses, 0.53 (0.06) for millets and 0.70 (0.05) for oilseeds. Within a state and year, adjacent districts move together strongly in the same year. Almost none of that carries over to the next year.

## Placebos

### 1. Future neighbour changes

The raw lead placebo is biased. A shock common to i and its neighbours at t raises dS(i,t) and lowers dS(j,t+1) through mean reversion, so the raw t+1 lead is pushed negative by construction: -0.23 (0.03) for pulses, -0.29 (0.04) for millets and -0.22 (0.03) for oilseeds, all p < 0.001. That is not evidence of anything. The fair comparison mirrors the main specification. The lag specification controls the district's own past change, so the lead specification controls its own future change.

| Group | Lag t-1 (same sample) | Lead t+1, mirrored | Joint: lag | Joint: lead | Lag minus lead | n |
|---|---|---|---|---|---|---|
| Pulses | 0.070 (0.040), p = 0.083 | **0.095 (0.044), p = 0.033** | 0.045 (0.031) | 0.027 (0.025) | 0.018 (0.039), p = 0.65 | 9,149 |
| Millets | -0.035 (0.033) | -0.060 (0.027), p = 0.027 | -0.014 (0.029) | -0.040 (0.023) | 0.026 (0.036), p = 0.47 | 9,149 |
| Oilseeds | 0.041 (0.035) | 0.052 (0.036) | 0.037 (0.032) | 0.031 (0.029) | 0.007 (0.049), p = 0.89 | 9,149 |

The "joint" columns put both lag and lead in one regression, with own lag and own lead controlled. For pulses, the future neighbour change predicts as much as the past one. With both in the regression, neither is significant and their difference is 0.018 (SE 0.039).

### 2. Same-state non-neighbours at a similar distance

Each district gets the districts in its own state that do not share a border with it and whose centroid lies between 0.75 and 1.5 times its mean centroid distance to its own neighbours. On average that is 2.2 districts, 104 km away.

| Group | Neighbours, alone | Placebo, alone | Horse race: neighbours | Horse race: placebo | Difference | n |
|---|---|---|---|---|---|---|
| Pulses | 0.101 (0.050), p = 0.044 | **0.047 (0.021), p = 0.023** | 0.092 (0.049), p = 0.062 | 0.038 (0.019), p = 0.049 | 0.054 (0.051), p = 0.29 | 6,803 |
| Millets | -0.071 (0.041) | -0.005 (0.016) | -0.071 (0.041) | 0.000 (0.016) | -0.071 (0.045) | 6,803 |
| Oilseeds | 0.036 (0.040) | -0.031 (0.028) | 0.048 (0.043) | -0.037 (0.029) | 0.085 (0.059), p = 0.15 | 6,803 |

For pulses, the non-adjacent districts predict about half as much as the neighbours, and the two coefficients cannot be told apart.

### 3. Second-order neighbours

These are neighbours of neighbours that do not themselves share a border, about 11.6 per district. Entered alongside the first-order lag: pulses neighbours 0.058 (0.040), second-order 0.031 (0.030). Millets -0.033 (0.031) and -0.009 (0.029). Oilseeds 0.048 (0.036) and -0.024 (0.036).

### 4. Permutation

In 200 draws, each district gets as many random same-state non-adjacent "neighbours" as it really has.

| Group | Observed | Mean of random draws | SD | 5th-95th percentile | p |
|---|---|---|---|---|---|
| Pulses | 0.067 | -0.005 | 0.029 | -0.050 to 0.041 | 0.02 |
| Millets | -0.037 | 0.004 | 0.031 | -0.050 to 0.050 | 0.18 |
| Oilseeds | 0.036 | -0.021 | 0.034 | -0.081 to 0.032 | 0.09 |

Random same-state sets give about zero, as the state-by-year FE require. So whatever links pulse changes is concentrated nearby. The distance placebo shows "nearby" means a radius of about 100 km in the same state, not a shared border.

## Robustness of the main lag coefficient

| Check | Pulses | Millets | Oilseeds |
|---|---|---|---|
| Main (M2) | 0.067 (0.039) | -0.037 (0.031) | 0.036 (0.033) |
| SE clustered by state (27 clusters) | 0.067 (0.042), p = 0.11 | -0.037 (0.035) | 0.036 (0.021), p = 0.083 |
| District + year FE (no state-by-year) | 0.063 (0.049) | 0.005 (0.030) | 0.052 (0.031) |
| State-by-year FE, no district FE | 0.076 (0.041) | -0.030 (0.032) | 0.044 (0.034) |
| Without own lagged change | -0.146 (0.029) | -0.197 (0.044) | -0.213 (0.030) |
| t from 1999-00 only | identical to main (the two lags already exclude earlier years) | | |
| No defect filter (all unit-years) | 0.029 (0.030) | 0.048 (0.032) | -0.022 (0.029) |
| Neighbours' rises only | 0.084 (0.049) | -0.064 (0.053) | 0.108 (0.058), p = 0.063 |
| Neighbours' falls only | 0.049 (0.057) | -0.014 (0.059) | -0.047 (0.057) |
| Lags t-1 and t-2 together: t-1 | 0.087 (0.050) | -0.042 (0.027) | 0.056 (0.038) |
| Lags t-1 and t-2 together: t-2 | -0.009 (0.029) | 0.027 (0.028) | 0.020 (0.032) |
| Change in log area (instead of share) | 0.078 (0.033), p = 0.017 | -0.050 (0.035) | 0.022 (0.021) |
| Neighbours' lagged share level (own level controlled) | 0.043 (0.043) | 0.058 (0.029), p = 0.049 | 0.086 (0.032), p = 0.006 |

**Without own lagged change:** every coefficient turns sharply negative. That is the mean reversion of shocks shared with neighbours, so the own-lag control is needed. It is also a warning that the residual co-movement is mostly transitory.

**3-year changes, non-overlapping windows** (the district's change from t-3 to t on the neighbours' change from t-6 to t-3):

| Group | Lag | Lag, mirror sample | Mirrored lead | n (lag / mirror) |
|---|---|---|---|---|
| Pulses | 0.245 (0.053), p < 0.001 | 0.318 (0.082) | 0.234 (0.099), p = 0.018 | 2,593 / 1,894 |
| Millets | -0.021 (0.051) | -0.006 (0.055) | -0.030 (0.091) | 2,593 / 1,894 |
| Oilseeds | 0.034 (0.052) | 0.059 (0.055) | 0.055 (0.071) | 2,593 / 1,894 |

For pulses, the 3-year lag is the strongest single number, but it is matched by the mirrored lead. So three-year pulse movements are shared across neighbouring districts in both directions of time. That is a regional trend, not a sequence.

## By season (share of the season's own area; only where the season split is reliable)

| Group | Kharif lag | Rabi lag | n (kharif / rabi) |
|---|---|---|---|
| Pulses | -0.001 (0.034) | 0.052 (0.033), p = 0.12 | 8,716 / 8,184 |
| Millets | -0.058 (0.031), p = 0.062 | 0.005 (0.033) | 8,716 / 8,184 |
| Oilseeds | **0.077 (0.039), p = 0.047** | 0.039 (0.026) | 8,716 / 8,184 |
| Maize | 0.117 (0.035), p = 0.001 | 0.047 (0.057) | 8,716 / 8,184 |

Kharif oilseeds is the only headline group with a lag significant at 5%. One borderline hit among six season tests (three groups × two seasons) is about what chance alone produces. The unmirrored season leads are again negative by construction. Sugarcane in rabi has no within-unit variation and is not estimable.

## Other groups

Rice, wheat, cotton and sugarcane: no lag effect.

**Maize** has 0.060 (0.031), p = 0.049, and 0.117 (0.035) in kharif. It survives state clustering (0.060, SE 0.021). Its mirrored lead is 0.016 (0.057), and in the joint regression the lag is 0.055 (0.028) against a lead of 0.022 (0.052). The difference is 0.033 (0.061), so even here the lag cannot be separated from the lead. Its distance placebo is -0.040 (0.036). Maize is the one group where the pattern leans toward neighbour-led spread, but it does not pass the lag-minus-lead test and does not survive BH correction (q = 0.22).

**Other** (vegetables, spices, plantation crops, barley and the rest): 0.108 (0.025), q < 0.001. The joint lag is 0.071 (0.026) against a lead of 0.050 (0.040), a difference of 0.021 (0.057). The difference is again not significant. This group is a residual mix and is most exposed to reporting changes in minor crops.

## Common shocks: what can produce this pattern without any spread

State-by-year FE remove statewide shocks. They do not remove these:

- **Sub-state weather:** rainfall controls absorb only part of it, and the IMD grid has errors in some district-years.
- **Regional markets and prices:** for example a mandi or processor serving a cluster of districts, or regional pulse prices.
- **Programmes targeted at clusters:** NFSM pulses and coarse-cereal districts were chosen in contiguous blocks, A3P, and state seed distribution by agro-climatic zone.
- **Data-reporting patterns that are spatially clustered:** statistical circles, and child districts carved from the same parent.

Any of these would make adjacent districts' changes correlate at every lead and lag, which is what the data show:

- strong same-year co-movement;
- lag and mirrored lead of equal size;
- non-adjacent districts within about 100 km predicting about half as much as neighbours.

A real neighbour-to-neighbour diffusion would show up as lag > lead and neighbours > distance-matched non-neighbours. Neither inequality is detected for any group. These are failures to detect, not proofs of absence: the CIs on the lag-minus-lead differences are about ±0.08, wider than the main effects themselves.

## What holds and what does not

**Holds:**
- Within a state and year, adjacent districts' pulse, millet and oilseed shares move together strongly in the same year (coefficients 0.53 to 0.70).
- Pulse movements are positively linked to neighbours' movements at a one-year horizon (about 0.07) and at a three-year horizon (about 0.25). The link is about the same size looking forward as looking back.

**Does not hold:**
- Any evidence that districts take up pulses, millets or oilseeds after their neighbours do.
- **Pulses:** the lag is borderline (p = 0.086). It loses significance with state clustering, without the defect filter, and against the lead (difference 0.018, SE 0.039). It also does not separate from the same-state distance placebo (difference 0.054, SE 0.051).
- **Millets:** null or slightly negative throughout.
- **Oilseeds:** null at the annual level. The single kharif hit (p = 0.047) does not survive multiple-testing logic.

## Plain findings

1. Neighbouring districts' crop mixes move together within the same year, much more than they move together across years. Most of that co-movement is transitory and reverses.
2. There is no sign that districts take up pulses, millets or oilseeds after their neighbours do. For pulses, a neighbour's future change predicts a district's change as well as its past change. Non-neighbours about 100 km away in the same state predict it about half as well.
3. The pattern fits regional common shocks (weather, markets, programmes targeted at clusters of districts) better than spread from district to district. Neither reading can be ruled out, because the lag-lead and neighbour-versus-placebo differences are measured with SEs of 0.04 to 0.06.
4. Maize is the only group where the lagged neighbour effect leans toward a spread pattern (0.06, robust to state clustering, distance placebo negative). It still fails the direct lag-versus-lead test and does not survive correction across nine groups.

## Files

- Script: `scripts/99_crop_groups_neighbours.py`. It runs in about 2 minutes, including 600 permutation regressions.
- Every estimate, with SE, CI, p and n: `neighbours_estimates.csv`
- Permutation: `neighbours_permutation.csv`
- Descriptives: `neighbours_descriptives.csv`
- Log: `neighbours_log.txt`
- Figure: `neighbours_coefplot.png`
