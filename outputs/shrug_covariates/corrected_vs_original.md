# Corrected diversity indices, and what changes

Dropped 288 rows on 7 bogus state-district pairs.

## Defect 1: the last year is a stub

| year | rows | million ha | districts |
|---|---|---|---|
| 2017-18 | 18,000 | 186.0 | 682 |
| 2018-19 | 18,294 | 185.1 | 687 |
| 2019-20 | 19,256 | 194.9 | 694 |
| 2020-21 | 319 | 0.9 | 13 |

Dropped 319 rows in 2020-21 (a stub) and 8549 rows in 1997-98 (broken: see _years.py).
Analysis window is 1998-99 to 2019-20, 22 years.

## Defect 2: duplicate rows

74 exact duplicates on (district, year, season, crop) remain after the
bogus-pair cleaning. Collapsed by taking the maximum rather than the sum.
Rows after collapsing: 336,040.

## Defects 3 and 4: measure each index per year, then average

District-year observations: 13,605.

Hill numbers added, all in units of effective number of crops:
  D0 richness        mean 21.2
  D1 exp(Shannon)    mean 4.9
  D2 inverse Simpson mean 3.4
  evenness D1/D0     mean 0.249

D0 >= D1 >= D2 must hold for every observation. Violations: 0.

## Why the ABI is kept but demoted

The ABI is the equal-weighted mean of min-max normalised Shannon, Simpson and
richness. Three problems, the first of which is easy to demonstrate.

**1. It is sample-dependent.** Min-max normalisation rescales against whichever
districts happen to be in the file, so a district's ABI changes when other
districts are added or removed, without anything about that district changing.

Recomputing the ABI on 200 random 80 percent subsamples moves a district's own
score by 0.0032 on average, up to 0.0190. The index is not a property of the
district alone.

**2. Equal weighting is arbitrary.** There is no argument for one third each.

**3. It double counts evenness.** Shannon and Simpson are the same family at
different sensitivities and correlate at r = 0.943 here, so two of the three
components measure nearly the same thing, and richness is outvoted two to one.

The Hill numbers have none of these problems. They are absolute, they need no
normalisation, they share one unit, and D0, D1 and D2 are a deliberate ladder
rather than three things to average. Results are reported on all of them; the
ABI is kept only so this rebuild can be compared against the original file.
Districts in the corrected file: 724.

## What changes

| index | original (pooled) | corrected (annual mean) | Pearson r | Spearman rho |
|---|---|---|---|---|
| Shannon | 1.518 | 1.414 | 0.978 | 0.976 |
| Simpson | 0.635 | 0.609 | 0.974 | 0.973 |
| richness | 30.383 | 20.602 | 0.845 | 0.812 |
| ABI | 0.633 | 0.580 | 0.953 | 0.953 |

ABI rank movement between the two: median 31 places, 90th percentile 111, max 311.
Districts moving more than 100 rank places: 91 of 724.

Biggest movers:

| district | years | original ABI | corrected ABI | rank move |
|---|---|---|---|---|
| JHARKHAND|KODERMA | 20 | 0.724 | 0.458 | 311 |
| JHARKHAND|GARHWA | 20 | 0.790 | 0.577 | 277 |
| JHARKHAND|PALAMU | 20 | 0.802 | 0.595 | 261 |
| TAMIL NADU|THE NILGIRIS | 22 | 0.813 | 0.604 | 260 |
| LADAKH|LEH LADAKH | 20 | 0.612 | 0.316 | 246 |
| JHARKHAND|CHATRA | 20 | 0.704 | 0.510 | 212 |
| PUDUCHERRY|MAHE | 19 | 0.557 | 0.248 | 206 |
| LADAKH|KARGIL | 14 | 0.564 | 0.282 | 204 |
| JHARKHAND|GODDA | 20 | 0.673 | 0.489 | 183 |
| JHARKHAND|LOHARDAGA | 20 | 0.619 | 0.413 | 177 |

The state ranking the README reports (Karnataka most diverse, Punjab least):

| state | n | original | corrected | rank orig | rank corr |
|---|---|---|---|---|---|
| Karnataka | 30 | 0.854 | 0.803 | 1 | 1 |
| Rajasthan | 33 | 0.767 | 0.745 | 4 | 2 |
| Andhra Pradesh | 20 | 0.786 | 0.731 | 2 | 3 |
| Nagaland | 11 | 0.765 | 0.706 | 5 | 4 |
| Madhya Pradesh | 55 | 0.775 | 0.706 | 3 | 5 |
| Uttarakhand | 13 | 0.726 | 0.697 | 6 | 6 |
| Meghalaya | 11 | 0.659 | 0.662 | 11 | 7 |
| Uttar Pradesh | 78 | 0.689 | 0.647 | 7 | 8 |
| Tripura | 8 | 0.343 | 0.317 | 27 | 27 |
| Odisha | 30 | 0.473 | 0.345 | 23 | 26 |
| Jharkhand | 24 | 0.562 | 0.378 | 18 | 25 |