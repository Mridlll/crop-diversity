# Diversity build on the extended APY file

Rows after script 76 cleaning, 1998-99 to 2022-23: 391,563.
State names in the flag list found: ['HARYANA', 'JAMMU AND KASHMIR', 'MADHYA PRADESH', 'MANIPUR', 'MEGHALAYA', 'NAGALAND', 'THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU'].
Rows in flagged state-years: 4,255.
District-year rows: 15,759.
Full period 1998-99 to 2019-20 rebuilt from the extended file against the site's corrected indices (724 districts): max abs difference {'D0_richness': 0.0, 'D1_exp_shannon': 0.0, 'D2_inv_simpson': 0.0, 'evenness_D1_D0': 0.0}.
Districts differing in D1 by more than 1e-6: 0.

Window x variant district counts:

| window | variant | districts | mean D0 | mean D0 0.5% | mean D1 | mean E |
|---|---|---|---|---|---|---|
| full | all | 624 | 21.15 | 9.63 | 4.892 | 0.250 |
| full | drop6 | 624 | 19.37 | 9.28 | 4.716 | 0.261 |
| w1517 | all | 687 | 21.74 | 9.65 | 4.893 | 0.240 |
| w1517 | drop6 | 687 | 19.74 | 9.27 | 4.698 | 0.252 |
| w1721 | all | 707 | 21.67 | 9.42 | 4.731 | 0.239 |
| w1721 | drop6 | 707 | 19.79 | 9.08 | 4.566 | 0.249 |
| cs0913 | all | 635 | 21.01 | 9.33 | 4.779 | 0.248 |
| cs0913 | drop6 | 635 | 19.32 | 8.99 | 4.613 | 0.257 |
| e9802 | all | 564 | 21.13 | 9.92 | 4.969 | 0.259 |
| e9802 | drop6 | 564 | 19.43 | 9.59 | 4.807 | 0.269 |
| w1721 | flags | 697 | 22.19 | 9.55 | 4.816 | 0.234 |
| w1721 | flags_drop6 | 697 | 20.18 | 9.20 | 4.640 | 0.245 |
| w1721 | flags_stable | 696 | 21.47 | 9.38 | 4.730 | 0.238 |
| w1517 | stable | 687 | 20.20 | 9.12 | 4.690 | 0.258 |

Mean crops per district-year (unweighted), all crops / six crops removed:

| year | D0 all | D0 drop6 | D0 0.5% all | D1 all | D1 drop6 |
|---|---|---|---|---|---|
| 2013-14 | 21.00 | 19.19 | 9.30 | 4.718 | 4.535 |
| 2014-15 | 19.70 | 18.19 | 9.26 | 4.708 | 4.551 |
| 2015-16 | 20.62 | 18.79 | 9.50 | 4.845 | 4.649 |
| 2016-17 | 21.09 | 19.14 | 9.62 | 4.918 | 4.716 |
| 2017-18 | 21.70 | 19.70 | 9.85 | 5.049 | 4.847 |
| 2018-19 | 22.21 | 20.14 | 9.53 | 4.808 | 4.617 |
| 2019-20 | 22.66 | 20.58 | 9.59 | 4.780 | 4.598 |
| 2020-21 | 22.23 | 20.43 | 9.43 | 4.678 | 4.530 |
| 2021-22 | 21.84 | 20.00 | 9.28 | 4.613 | 4.482 |
| 2022-23 | 21.60 | 19.94 | 9.40 | 4.615 | 4.486 |
