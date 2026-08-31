# Robustness of the irrigation-diversity inverted U

Every row is `outcome ~ irrigation + irrigation^2`, standard errors clustered
on the pre-2011 parent district. The claim survives only if the squared term
stays negative and significant with a stable turning point.

| specification | n | linear | squared | p(squared) | turning point | R2 |
|---|---|---|---|---|---|---|
| 1  full set | 697 | +0.602 | -0.502*** | 0.0000 | 0.600 | 0.035 |
| 2  state fixed effects | 697 | +0.324 | -0.416*** | 0.0001 | 0.390 | 0.487 |
| 3  PC11 VD irrigation instead | 656 | +0.189 | -0.321*** | 0.0001 | 0.295 | 0.490 |
| 4  drop shared parents | 542 | +0.195 | -0.304*** | 0.0002 | 0.322 | 0.663 |
| 5  inverse-parent weighted | 697 | +0.264 | -0.359*** | 0.0000 | 0.367 | 0.561 |
| 6  drop source-conflict districts | 605 | +0.306 | -0.409*** | 0.0002 | 0.374 | 0.514 |
| 7  drop MH, JH, AS | 607 | +0.290 | -0.397*** | 0.0002 | 0.364 | 0.506 |
| 8  drop thinnest 10% by GCA | 627 | +0.365 | -0.468*** | 0.0000 | 0.390 | 0.473 |
| 9  Shannon index | 697 | +1.192 | -1.716*** | 0.0000 | 0.347 | 0.457 |
| 10 Simpson index | 697 | +0.327 | -0.442*** | 0.0002 | 0.371 | 0.490 |
| 11 crop richness | 697 | +8.469 | -6.851 | 0.1890 | 0.618 | 0.593 |
| 12 weighted by cropped area | 697 | +0.350 | -0.448*** | 0.0005 | 0.391 | 0.499 |

Of the 9 specifications using the composite index, 9 keep a negative squared
term significant at the 5 percent level.
Turning points across those: min 0.295, median 0.374, max 0.600.

## SECC land ownership (the row script 73 dropped)

Non-missing on SECC landless share: 683 of 697 districts.
raw       coefficient on landless share = -0.0494 (p = 0.1770, n = 683)
adjusted  coefficient on landless share = -0.0723 (p = 0.0674, n = 683)

Also SECC owned-acre measures:
  secc_unirr_acre_per_hh     -0.0010 (p = 0.0057, n = 683)
  secc_twocrop_acre_per_hh   -0.0004 (p = 0.6812, n = 683)

## Irrigation source, with the quadratic in irrigation level included

Script 73 controlled for irrigation linearly, which is inconsistent with a
quadratic relationship. Re-run with the squared term in place.

| outcome | n | canal (vs groundwater) | surface (vs groundwater) | R2 |
|---|---|---|---|---|
| agro_biodiversity_index | 697 | -0.050** | -0.201*** | 0.498 |
| shannon_index | 697 | -0.056 | -0.701*** | 0.471 |
| crop_richness | 697 | -4.579*** | -1.174 | 0.599 |
| share_cereals | 697 | +0.038 | +0.218*** | 0.583 |
| share_pulses | 697 | +0.063*** | -0.109*** | 0.414 |
| share_oilseeds | 697 | -0.100*** | -0.129** | 0.484 |

Significance: *** p<0.01, ** p<0.05, * p<0.10.