# Corrected diversity indices, and what changes

Dropped 288 rows on 7 bogus state-district pairs.

## Defect 1: the last year is a stub

| year | rows | million ha | districts |
|---|---|---|---|
| 2017-18 | 18,000 | 186.0 | 682 |
| 2018-19 | 18,294 | 185.1 | 687 |
| 2019-20 | 19,256 | 194.9 | 694 |
| 2020-21 | 319 | 0.9 | 13 |

Dropped 319 rows in 2020-21. Usable range is 1997-98 to 2019-20, 23 years.

## Defect 2: duplicate rows

91 exact duplicates on (district, year, season, crop) remain after the
bogus-pair cleaning. Collapsed by taking the maximum rather than the sum.
Rows after collapsing: 344,572.

## Defects 3 and 4: measure each index per year, then average

District-year observations: 14,089.

Hill numbers added, all in units of effective number of crops:
  D0 richness        mean 21.0
  D1 exp(Shannon)    mean 4.8
  D2 inverse Simpson mean 3.4
  evenness D1/D0     mean 0.252

D0 >= D1 >= D2 must hold for every observation. Violations: 0.

## Why the ABI is kept but demoted

The ABI is the equal-weighted mean of min-max normalised Shannon, Simpson and
richness. Three problems, the first of which is easy to demonstrate.

**1. It is sample-dependent.** Min-max normalisation rescales against whichever
districts happen to be in the file, so a district's ABI changes when other
districts are added or removed, without anything about that district changing.

Recomputing the ABI on 200 random 80 percent subsamples moves a district's own
score by 0.0024 on average, up to 0.0156. The index is not a property of the
district alone.

**2. Equal weighting is arbitrary.** There is no argument for one third each.

**3. It double counts evenness.** Shannon and Simpson are the same family at
different sensitivities and correlate at r = 0.944 here, so two of the three
components measure nearly the same thing, and richness is outvoted two to one.

The Hill numbers have none of these problems. They are absolute, they need no
normalisation, they share one unit, and D0, D1 and D2 are a deliberate ladder
rather than three things to average. Results are reported on all of them; the
ABI is kept only so this rebuild can be compared against the original file.
Districts in the corrected file: 725.

## What changes

| index | original (pooled) | corrected (annual mean) | Pearson r | Spearman rho |
|---|---|---|---|---|
| Shannon | 1.516 | 1.409 | 0.977 | 0.974 |
| Simpson | 0.634 | 0.608 | 0.972 | 0.971 |
| richness | 30.342 | 20.396 | 0.846 | 0.811 |
| ABI | 0.632 | 0.582 | 0.953 | 0.951 |

ABI rank movement between the two: median 30 places, 90th percentile 117, max 310.
Districts moving more than 100 rank places: 95 of 725.

Biggest movers:

| district | years | original ABI | corrected ABI | rank move |
|---|---|---|---|---|
| JHARKHAND|KODERMA | 21 | 0.724 | 0.462 | 310 |
| LADAKH|LEH LADAKH | 21 | 0.612 | 0.302 | 262 |
| TAMIL NADU|THE NILGIRIS | 22 | 0.813 | 0.607 | 262 |
| JHARKHAND|GARHWA | 21 | 0.790 | 0.591 | 261 |
| JHARKHAND|PALAMU | 21 | 0.802 | 0.607 | 251 |
| LADAKH|KARGIL | 15 | 0.564 | 0.264 | 211 |
| PUDUCHERRY|MAHE | 19 | 0.557 | 0.249 | 208 |
| JHARKHAND|CHATRA | 21 | 0.704 | 0.519 | 207 |
| CHHATTISGARH|SURAJPUR | 9 | 0.689 | 0.749 | 178 |
| JHARKHAND|GODDA | 21 | 0.673 | 0.496 | 178 |

The state ranking the README reports (Karnataka most diverse, Punjab least):

| state | n | original | corrected | rank orig | rank corr |
|---|---|---|---|---|---|
| Karnataka | 30 | 0.854 | 0.803 | 1 | 1 |
| Rajasthan | 33 | 0.767 | 0.744 | 4 | 2 |
| Andhra Pradesh | 20 | 0.786 | 0.733 | 2 | 3 |
| Madhya Pradesh | 55 | 0.775 | 0.705 | 3 | 4 |
| Uttarakhand | 13 | 0.726 | 0.700 | 6 | 5 |
| Nagaland | 11 | 0.765 | 0.698 | 5 | 6 |
| Meghalaya | 11 | 0.659 | 0.667 | 11 | 7 |
| Uttar Pradesh | 78 | 0.689 | 0.654 | 7 | 8 |
| Tripura | 8 | 0.343 | 0.322 | 27 | 27 |
| Odisha | 30 | 0.473 | 0.356 | 23 | 26 |
| Jharkhand | 24 | 0.562 | 0.383 | 18 | 25 |