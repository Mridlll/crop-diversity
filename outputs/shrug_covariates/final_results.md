# Final results, on corrected diversity indices

## Status after referee (Nov 2026)

Written 7 November 2026. Replacement on the site: Irrigation page (commit bcf97f7), `docs/data/site_irrigation_v2.json`; referee `outputs/revision_nov8/irrigation_v2/referee/referee_raw_repro_infer_q1x_q2_q3_q4_mt.md` (scripts 124, 130). Markets: `outputs/revision_nov8/market_pop_weighted_results.md` (scripts 104_market_pop_weighted, 112).

- Section 1, "the inverted U" with a D1 turning point of 0.279: **superseded by flat-then-falling**. The squared term below is right (it survives all eight sample versions and the recent windows, -6.78 for 2015-20 and -6.46 for 2017-22), but the rising arm is not significant, so there is no interior peak to report. Within state, fifths of irrigated share against the least irrigated fifth: **+0.12 (SE 0.16), -0.55 (0.36), -1.05 (wild p 0.072), -2.14 (wild p 0.013)**. Two-piece line with a knot at 0.30: **+2.91 per unit below (p 0.119), -4.10 per unit above (p 0.022)**, about -0.4 crops per 10 points; 85 percent of districts lie above the knot. The decile table in section 2 shows the same shape: none of the first seven deciles differs significantly from the first.
- "6 of 6 alternative indices agree": evenness does not hold up in the recent windows (squared term -0.165, state wild p 0.166).
- Section 3, irrigation source: exploratory. The later source split on census 2011 shares cannot tell canal from groundwater (interaction wild p 0.614); tank and lake irrigation carries the strongest negative link, **-9.28 per unit (state SE 0.93), wild p 0.008**, -0.48 crops per within-state SD.
- Section 4, FPO +1.875 on village shares -> **+1.886 (0.790), p 0.017** on population shares, the site's basis; state-clustered p 0.156 (full window), 0.110 and 0.094 in the recent windows. Cropping intensity +2.315 is partly mechanical (gross over net sown) and has no referee pass.

The original analysis below is kept unchanged for the record, apart from the sentences marked [corrected Nov 2026]. Where it disagrees with this block, this block and `outputs/ANALYSIS_INDEX.md` hold.

---

Diversity indices are the annual-mean rebuild from script 76: 2020-21 stub
dropped, duplicate rows collapsed, every index measured per year then averaged
across 1998-99 to 2019-20. Districts with under 10 years of data are excluded.

Analysis set: 606 districts, 30 states. 64 sit on a shared pre-2011 parent.
Median years of data per district: 22.

## 1. The inverted U, on corrected indices

Primary outcome is **D1, the effective number of crops** (exp of Shannon), not
the ABI. The ABI is a min-max composite: sample-dependent, equally weighted for
no stated reason, and two of its three components correlate at 0.94, so it is
really two parts evenness to one part richness. D1 is absolute, needs no
normalisation and reads directly as a count of equally-common crops. The ABI is
kept as one row for comparison.

| specification | n | linear | squared | p(squared) | turning point | R2 |
|---|---|---|---|---|---|---|
| D1: no fixed effects | 606 | +10.659 | -10.423*** | 0.0000 | 0.511 | 0.067 |
| D1: state fixed effects | 606 | +3.811 | -6.836*** | 0.0000 | 0.279 | 0.534 |
| D1: PC11 VD irrigation | 573 | +2.749 | -5.438*** | 0.0000 | 0.253 | 0.537 |
| D1: drop shared parents | 542 | +3.392 | -6.384*** | 0.0000 | 0.266 | 0.542 |
| D1: inverse-parent weighted | 606 | +3.614 | -6.597*** | 0.0000 | 0.274 | 0.535 |
| D1: drop source conflicts | 530 | +3.166 | -6.571*** | 0.0000 | 0.241 | 0.551 |
| D1: full panel only | 510 | +3.080 | -6.192*** | 0.0000 | 0.249 | 0.514 |
| D1: area weighted | 606 | +5.763 | -8.178*** | 0.0000 | 0.352 | 0.539 |
| D0 richness | 606 | +10.622 | -11.427*** | 0.0001 | 0.465 | 0.828 |
| D2 inverse Simpson | 606 | +1.675 | -3.852*** | 0.0002 | 0.217 | 0.503 |
| evenness D1/D0 | 606 | +0.054 | -0.175** | 0.0230 | 0.155 | 0.540 |
| Shannon (raw) | 606 | +0.926 | -1.423*** | 0.0000 | 0.325 | 0.583 |
| Simpson (raw) | 606 | +0.264 | -0.362*** | 0.0004 | 0.364 | 0.615 |
| ABI (composite, for comparison) | 606 | +0.310 | -0.416*** | 0.0000 | 0.373 | 0.671 |

**Sample robustness.** 8 of 8 D1 specifications keep a negative squared term
significant at 5 percent. Turning points 0.241 to 0.511, median 0.270.

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
| D0 richness | 21.4 | crops grown in an average year |
| D1 exp(Shannon) | 4.9 | equally-common crops giving the same diversity |
| D2 inverse Simpson | 3.5 | the same, weighted toward the dominant crops |
| evenness D1/D0 | 0.250 | how evenly area is spread |

The average district grows about 21 crops but is effectively growing about 5.
That gap is the whole story of Indian cropping concentration, and a unitless
0-to-1 composite hides it.

## 2. Decile profile

| decile | n | irrigation | D0 | D1 | D2 | evenness | ABI | cereal share |
|---|---|---|---|---|---|---|---|---|
| 1 | 61 | 0.165 | 19.3 | 4.13 | 2.92 | 0.252 | 0.534 | 0.718 |
| 2 | 61 | 0.300 | 21.5 | 5.00 | 3.51 | 0.256 | 0.581 | 0.670 |
| 3 | 60 | 0.385 | 22.2 | 5.32 | 3.70 | 0.262 | 0.607 | 0.616 |
| 4 | 61 | 0.444 | 21.3 | 5.89 | 4.02 | 0.282 | 0.615 | 0.638 |
| 5 | 60 | 0.502 | 20.7 | 5.03 | 3.45 | 0.265 | 0.594 | 0.618 |
| 6 | 61 | 0.563 | 22.4 | 5.67 | 3.91 | 0.273 | 0.631 | 0.594 |
| 7 | 60 | 0.640 | 21.7 | 5.26 | 3.70 | 0.263 | 0.630 | 0.583 |
| 8 | 61 | 0.718 | 24.5 | 4.99 | 3.58 | 0.212 | 0.659 | 0.612 |
| 9 | 60 | 0.826 | 22.2 | 4.53 | 3.26 | 0.214 | 0.620 | 0.777 |
| 10 | 61 | 0.922 | 18.1 | 3.69 | 2.84 | 0.224 | 0.545 | 0.794 |

## 3. Irrigation source, groundwater omitted

| outcome | n | canal | surface | R2 |
|---|---|---|---|---|
| D1_exp_shannon | 606 | -0.315 | -2.030** | 0.540 |
| D0_richness | 606 | -2.612*** | -0.422 | 0.833 |
| D2_inv_simpson | 606 | -0.160 | -1.670** | 0.510 |
| evenness_D1_D0 | 606 | +0.034* | -0.086 | 0.560 |
| share_cereals | 606 | +0.017 | +0.297*** | 0.675 |
| share_pulses | 606 | +0.071*** | -0.105*** | 0.445 |
| share_oilseeds | 606 | -0.085*** | -0.152** | 0.551 |

## 4. The five dimensions

Outcome is D1, the effective number of crops. `adjusted` adds irrigation and
log mean holding size and state fixed effects.

| dimension | n | raw | adjusted |
|---|---|---|---|
| irrigation share | 606 | -0.799** | +3.844*** |
| canal village share | 606 | -1.268*** | -0.251 |
| mean holding (log ha) | 606 | +0.354*** | +0.024 |
| cultivator share of ag workers | 606 | -1.086** | -0.004 |
| landless share (SECC) | 592 | -0.213 | -0.955 |
| mandi village share | 606 | -1.987 | -0.130 |
| weekly haat village share | 606 | -0.215 | +0.820 |
| regular market village share | 606 | -2.310*** | -0.710 |
| FPO village share | 606 | +2.197*** | +1.875** |
| cold storage village share | 606 | +6.325*** | +1.269 |
| SC population share | 606 | +1.269 | +1.235 |
| ST population share | 606 | -0.331 | +0.704 |
| cropping intensity | 606 | +0.848** | +2.315*** |

Significance: *** p<0.01, ** p<0.05, * p<0.10.