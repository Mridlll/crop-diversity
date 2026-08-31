# Final results, on corrected diversity indices

Diversity indices are the annual-mean rebuild from script 76: 2020-21 stub
dropped, duplicate rows collapsed, every index measured per year then averaged
across 1997-98 to 2019-20. Districts with under 10 years of data are excluded.

Analysis set: 606 districts, 30 states. 64 sit on a shared pre-2011 parent.
Median years of data per district: 23.

## 1. The inverted U, on corrected indices

Primary outcome is **D1, the effective number of crops** (exp of Shannon), not
the ABI. The ABI is a min-max composite: sample-dependent, equally weighted for
no stated reason, and two of its three components correlate at 0.94, so it is
really two parts evenness to one part richness. D1 is absolute, needs no
normalisation and reads directly as a count of equally-common crops. The ABI is
kept as one row for comparison.

| specification | n | linear | squared | p(squared) | turning point | R2 |
|---|---|---|---|---|---|---|
| D1: no fixed effects | 606 | +10.444 | -10.185*** | 0.0000 | 0.513 | 0.065 |
| D1: state fixed effects | 606 | +3.874 | -6.877*** | 0.0000 | 0.282 | 0.530 |
| D1: PC11 VD irrigation | 573 | +2.749 | -5.428*** | 0.0000 | 0.253 | 0.534 |
| D1: drop shared parents | 542 | +3.396 | -6.382*** | 0.0000 | 0.266 | 0.538 |
| D1: inverse-parent weighted | 606 | +3.641 | -6.612*** | 0.0000 | 0.275 | 0.531 |
| D1: drop source conflicts | 530 | +3.235 | -6.615*** | 0.0000 | 0.245 | 0.547 |
| D1: full 23 years only | 506 | +3.107 | -6.270*** | 0.0000 | 0.248 | 0.516 |
| D1: area weighted | 606 | +5.778 | -8.083*** | 0.0000 | 0.357 | 0.533 |
| D0 richness | 606 | +10.088 | -10.910*** | 0.0001 | 0.462 | 0.828 |
| D2 inverse Simpson | 606 | +1.760 | -3.925*** | 0.0001 | 0.224 | 0.500 |
| evenness D1/D0 | 606 | +0.066 | -0.188** | 0.0132 | 0.177 | 0.543 |
| Shannon (raw) | 606 | +0.936 | -1.432*** | 0.0000 | 0.327 | 0.578 |
| Simpson (raw) | 606 | +0.268 | -0.367*** | 0.0003 | 0.364 | 0.611 |
| ABI (composite, for comparison) | 606 | +0.310 | -0.417*** | 0.0000 | 0.372 | 0.668 |

**Sample robustness.** 8 of 8 D1 specifications keep a negative squared term
significant at 5 percent. Turning points 0.245 to 0.513, median 0.271.

**Index robustness.** 6 of 6 alternative indices agree, so the hump is not an
artefact of one index choice. It shows in the plain count of crops (D0), in the
effective counts (D1, D2) and in evenness on its own, which says irrigation does
two things at once: it changes how many crops a district grows and how evenly
area is spread across them.

On the pooled construction crop richness showed no hump at all (p = 0.19). It does
now, so that null was an artefact of pooling over years rather than a real result.

### 1b. The effective-number scale

| index | mean | reading |
|---|---|---|
| D0 richness | 21.2 | crops grown in an average year |
| D1 exp(Shannon) | 4.9 | equally-common crops giving the same diversity |
| D2 inverse Simpson | 3.5 | the same, weighted toward the dominant crops |
| evenness D1/D0 | 0.254 | how evenly area is spread |

The average district grows about 21 crops but is effectively growing about 5.
That gap is the whole story of Indian cropping concentration, and a unitless
0-to-1 composite hides it.

## 2. Decile profile

| decile | n | irrigation | D0 | D1 | D2 | evenness | ABI | cereal share |
|---|---|---|---|---|---|---|---|---|
| 1 | 61 | 0.165 | 19.1 | 4.11 | 2.91 | 0.255 | 0.536 | 0.718 |
| 2 | 61 | 0.300 | 21.2 | 4.99 | 3.52 | 0.261 | 0.584 | 0.669 |
| 3 | 60 | 0.385 | 21.9 | 5.29 | 3.68 | 0.267 | 0.608 | 0.617 |
| 4 | 61 | 0.444 | 21.1 | 5.86 | 4.00 | 0.286 | 0.617 | 0.638 |
| 5 | 60 | 0.502 | 20.5 | 5.01 | 3.43 | 0.268 | 0.596 | 0.618 |
| 6 | 61 | 0.563 | 22.1 | 5.63 | 3.90 | 0.278 | 0.633 | 0.587 |
| 7 | 60 | 0.640 | 21.5 | 5.24 | 3.69 | 0.267 | 0.632 | 0.574 |
| 8 | 61 | 0.718 | 24.3 | 4.96 | 3.56 | 0.213 | 0.662 | 0.604 |
| 9 | 60 | 0.826 | 22.0 | 4.54 | 3.27 | 0.216 | 0.624 | 0.772 |
| 10 | 61 | 0.922 | 18.0 | 3.71 | 2.86 | 0.228 | 0.550 | 0.793 |

## 3. Irrigation source, groundwater omitted

| outcome | n | canal | surface | R2 |
|---|---|---|---|---|
| D1_exp_shannon | 606 | -0.310 | -2.030** | 0.537 |
| D0_richness | 606 | -2.593*** | -0.467 | 0.833 |
| D2_inv_simpson | 606 | -0.155 | -1.660** | 0.507 |
| evenness_D1_D0 | 606 | +0.034* | -0.087 | 0.563 |
| share_cereals | 606 | +0.022 | +0.300*** | 0.671 |
| share_pulses | 606 | +0.073*** | -0.104*** | 0.445 |
| share_oilseeds | 606 | -0.097*** | -0.164*** | 0.559 |

## 4. The five dimensions

Outcome is D1, the effective number of crops. `adjusted` adds irrigation and
log mean holding size and state fixed effects.

| dimension | n | raw | adjusted |
|---|---|---|---|
| irrigation share | 606 | -0.753** | +3.901*** |
| canal village share | 606 | -1.265*** | -0.246 |
| mean holding (log ha) | 606 | +0.339*** | +0.019 |
| cultivator share of ag workers | 606 | -1.134*** | -0.060 |
| landless share (SECC) | 592 | -0.204 | -0.939 |
| mandi village share | 606 | -1.931 | -0.142 |
| weekly haat village share | 606 | -0.148 | +0.857 |
| regular market village share | 606 | -2.307*** | -0.716 |
| FPO village share | 606 | +2.188*** | +1.905** |
| cold storage village share | 606 | +6.308*** | +1.335 |
| SC population share | 606 | +1.368 | +1.233 |
| ST population share | 606 | -0.350 | +0.730 |
| cropping intensity | 606 | +0.845** | +2.265*** |

Significance: *** p<0.01, ** p<0.05, * p<0.10.