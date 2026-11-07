# Referee report, 9 Nov 2026 (on commit a4cde1d)

Script: `scripts/142_referee_nov9.py` (parts H, S1 to S5). Raw logs `raw_H.md`, `raw_S1.md` to `raw_S5.md`; tables `H_census_haat.csv`, `H2_census_haat_loso.csv`, `H8_bh.csv`, `S1_knot.csv`, `S2_attenuation.csv`, `S3_haat_floors.csv`, `S4_soybean_edge.csv`, `S5_border_gap.csv`, `S5_border_gap_loso.csv`. Inference unless stated: SE on the 2011 parent district; state clustering with t(G-1); restricted wild cluster bootstrap by state, Webb weights, 9,999 draws (S5 uses script 139's 999-draw dyadic version).

## Verdicts

| claim | verdict | key numbers |
|---|---|---|
| C11 census 2011 haat reach, D1 | SURVIVES | +1.857 (wild p 0.027); every added control +1.56 to +2.36, wild p 0.004 to 0.041; LOSO max state p 0.085; every window wild p 0.007 to 0.040 |
| C11 census 2011 haat reach, D0 at 0.5% | WEAKENED | +2.108 (state t p 0.054, wild p 0.031); wild p 0.10 to 0.12 in three of four sub-windows; unweighted village share wild p 0.11; LOSO without Karnataka state p 0.10 |
| C11 reading "the thin-crop pattern belongs to haat-only villages" | REFUTED | census haat-only and haat-and-mandi villages give the same slopes: difference on D1 -0.07 (SE 1.05), on D0 at 0.5% +0.22 (SE 1.35) |
| C1 haat and the count of crops (found here) | WEAKENED | log cropped area as a control: +6.276 to +3.762 (state p 0.019, wild p 0.050) |
| C10 haat collapse at 0.5% | SURVIVES | share floors 0.2/0.3/0.4/0.5%: +1.19, +0.53, +0.10, -0.06 (wild p all > 0.48); absolute floors are positive only because haat districts are larger |
| B18 knot near 0.43 to 0.50, fall above | SURVIVES | with the markets-page controls: knot 0.43 (bootstrap 0.28 to 0.72), slope above 0.50 -4.98 (wild p 0.043); sup-Wald p 0.072 (the kink is still weakly separated from a straight line) |
| B19 LUS mostly noise, so the within null says little | WEAKENED (implication) | reproduce corr 0.130, LUS signal 0.116; drop top 5% of abs(dLUS): signal 0.23; winsorised 0.14 to 0.18; rank corr 0.27 |
| E11 soybean within-minus-across edge 0.094 | WEAKENED (not significant at state level) | state CR1 t(G-1) p 0.064, restricted wild by state p 0.133; leave one event out: 4 of 7 give p > 0.05 (max 0.137) |
| E13 common-crop border gap | SURVIVES | -6.12 / -3.22 / -4.35 reproduced; without cotton -6.29 / -3.41 / -4.56 (wild p 0.003 / 0.031 / 0.006); M2 fragile under LOSO |

## H. Census 2011 haat reach (primary)

**The variable.** `pc11_vd_wkl_haat` in `shrug-vd11-dta/pc11_vd_clean_shrid.dta`, label "Weekly Haat", values 0 (495,234 shrids) and 1 (93,738), one missing; `pc11_vd_mrkt` "Mandis/Regular Market" is a separate 0/1 item. 99.0 percent of rural shrids are one census village; in the 1.0 percent that combine villages, the value stays 0/1 (any village with a haat). Reach rebuilt from villages reproduces script 136's `wk11_all` to 1e-16. It is the weekly haat item, coded correctly. Mean 0.240 (SD 0.197) against MA haat 0.190; correlation 0.43. The census split: haat-only 0.109, haat and mandi/regular 0.131, mandi/regular only 0.095.

**Reproduce (site controls, state FE, 606 districts).**

| outcome | b | p parent | p state t(G-1) | p wild | BH q (7) | Holm |
|---|---|---|---|---|---|---|
| D0 | +3.258 | 0.0005 | 0.024 | 0.023 | 0.037 | 0.116 |
| D0 at 0.1% | +3.081 | 0.0007 | 0.022 | 0.015 | 0.037 | 0.104 |
| D0 at 0.5% | +2.108 | 0.0029 | 0.054 | 0.031 | 0.037 | 0.116 |
| D0 at 1% | +2.073 | 0.0002 | 0.030 | 0.019 | 0.037 | 0.116 |
| D1 | +1.857 | 0.0002 | 0.044 | 0.027 | 0.037 | 0.116 |
| D2 | +1.360 | 0.0002 | 0.040 | 0.022 | 0.037 | 0.116 |
| evenness | +0.070 | 0.012 | 0.140 | 0.125 | 0.125 | 0.125 |

BH holds (q 0.037 for six of seven); Holm does not (0.10 to 0.12). Every outcome moves the same way, so the BH reading fits better.

**Leave one state out.** D1 +1.38 (without Karnataka, state p 0.085) to +2.19; no state pushes state p above 0.10. D0 at 0.5%: +1.38 (without Maharashtra, state p 0.082) to +2.40; without Karnataka state p 0.100.

**Controls** (term = census haat):

| added | D0 at 0.5% b (wild p) | D1 b (wild p) |
|---|---|---|
| census mandi/regular | +2.63 (0.023) | +2.13 (0.030) |
| log rural density (n 573) | +2.02 (0.056) | +1.89 (0.041) |
| village size (pop-weighted mean log pop, share in 5k+ villages) | +2.00 (0.017) | +1.56 (0.020) |
| MA mandi + regular market | +2.11 (0.029) | +1.84 (0.030) |
| MA haat | +2.40 (0.011) | +1.94 (0.015) |
| log cropped area | +1.94 (0.029) | +1.71 (0.024) |
| all of the above (n 573) | +3.24 (0.005) | +2.36 (0.029) |

The site controls already cover irrigation and its square, holding size, ST share, night lights, non-farm establishments and connectivity.

**Coverage.** The 2011 directory covers 100 percent of the rural population in every one of the 606 districts (median 1.000, none below 0.90). That leaves nothing to drop.

**Drop6.** D0 at 0.5% +2.109 (wild 0.039); D1 +1.744 (wild 0.032).

**Definitions.** Unweighted village share: D0 at 0.5% +1.33 (wild 0.114), D1 +1.28 (wild 0.032). Haat-only villages (haat, no mandi/regular): D0 +5.57 (wild 0.005), D0 at 0.5% +2.35 (wild 0.165), D1 +1.89 (wild 0.192). Haat-and-mandi villages: D0 +1.80 (wild 0.41), D0 at 0.5% +1.99 (wild 0.085), D1 +1.87 (wild 0.006). Jointly with mandi-only reach the two haat terms are equal on the fuller-mix outcomes (differences +0.22 and -0.07, SE 1.35 and 1.05). In the census, haat-only villages carry the count of crops, and both kinds go with a higher D1 by the same amount. The divergence from Mission Antyodaya is therefore between the two sources, which correlate only 0.52 even for haat-only villages. It does not come from the single-answer coding. The page's sentence saying the reverse is refuted and has been replaced.

**Windows** (b, wild p): D1 1998-2002 +1.94 (0.024), 2009-13 +1.83 (0.025), 2015-17 +2.23 (0.007), 2017-21 drop6 +2.04 (0.040). D0 at 0.5% +2.73 (0.012), +1.57 (0.111), +2.11 (0.104), +2.33 (0.121).

**Reading.** Census haat reach and D1 hold up under state-level inference, every control, every window and leaving out any state. Size is about +0.37 per SD. It is one cross-section with roughly 20 state clusters, and the BH family is significant while Holm is not. The 0.5% result is the weaker partner: significant in the full period and with controls, but not in three of four sub-windows or on village shares.

**Mission Antyodaya haat with district size (new, affects C1).** Haat-rich districts are larger (within-state correlation of MA haat reach with log cropped area 0.18). With log cropped area added, the MA haat coefficient on the count of crops falls from +6.276 to +3.762 (state p 0.019, wild p 0.050). Adding census haat as well gives +3.16 (wild p 0.078). D1 stays null (+0.17). Larger districts record more crop names, so a count needs a size control. Cropped area could partly respond to markets, so the +3.76 is a lower-side estimate, not the answer. The page now gives it in section 10.

## S1. B18 irrigation knot

Reproduced: knot 0.43, LR set 0.36 to 0.65; at 0.50 slope below -0.54 (wild p 0.62), above -5.69 (state SE 1.68, wild p 0.036). Break attempt with the markets-page controls (log holding, ST share, night lights, non-farm, connectivity): knot 0.43, LR set 0.36 to 0.70, Hansen 0.29 to 0.80, bootstrap 0.28 to 0.72, sup-Wald p 0.072; at 0.50 below -0.20 (wild 0.85), above -4.98 (wild p 0.043). SURVIVES. The fall above holds and the location stays imprecise, as the page says.

## S2. B19 attenuation

Reproduced: 388 districts, within-state correlation 0.130, LUS signal share 0.116, census 0.147. Break attempt: winsorising both changes at 1/2.5/5 percent gives correlation 0.22/0.23/0.28 and LUS signal 0.14/0.15/0.18 (census 0.33 to 0.42). Dropping the top 5 or 10 percent of abs(dLUS) gives signal 0.23 and 0.25. The within-state rank correlation is 0.27. With no territory break: 0.13 and 0.11. "Mostly noise" survives (at most a quarter signal). The 0.12 itself leans on a few large changes, though. At 0.23 the scaled within range is about -0.26 to +0.07 per 10 points, which clearly excludes the cross-sectional -0.57, so the step "the within null cannot rule out a fall as steep as the cross-section" is WEAKENED. The signal share is also a lower bound if the two records measure slightly different concepts. Page: the main-text sentence "With a slightly noisier record it would not" is replaced by one saying a few large changes carry much of the noise.

## S3. C10 haat collapse at 0.5%

Reproduced -0.058 at 0.5%. Share floors: 0.1% +2.46 (wild 0.16), 0.2% +1.19 (0.49), 0.3% +0.53 (0.73), 0.4% +0.10 (0.93). Absolute floors, site controls: 100 ha +5.76 (wild 0.020), 500 ha +4.07 (0.030), 1000 ha +3.27 (0.037; mean 9.5 crops, about the same as the 0.5% floor), 2000 ha +2.35 (0.074). With log cropped area: 100 ha +2.85 (0.13), 500 ha +1.10 (0.43), 1000 ha +0.38 (0.72), 2000 ha -0.29 (0.77); 0.5% share -0.53 (0.67). SURVIVES. The extra crops are small relative to the district. In hectares they are not tiny in big districts (a 0.3% crop in the mean district is about 860 ha), so "grown on a few plots" was dropped from the page.

## S4. E11 soybean edge within minus across

Reproduced: +0.0941 (unit SE 0.0473, p 0.047; script 140's state p 0.048 uses a normal reference), 1,585 unit-years, 7 events. Terms: same-state neighbours +0.127 (0.059), other-state +0.022, rings about 0. State CR1 with t(17) p 0.064; restricted wild bootstrap by state p 0.133 (by unit 0.035). Leave one event unit out: +0.070 (without Nizamabad, p 0.085), +0.077 (Rajnandgaon, 0.088), +0.074 (Panna, 0.137), +0.076 (Nanded, 0.070), +0.089 (Osmanabad, 0.045), +0.096 (Adilabad, 0.043), +0.108 (Nainital, 0.026). Chickpea surge: +0.080, wild by state p 0.50. WEAKENED. Every point estimate fits a within-state edge, but no estimate in E11 is significant under state-level inference. Page heading and prose now say it rests on a handful of take-ups and is not statistically significant for either crop.

## S5. E13 common-crop border gap

Reproduced 2015-19 sim8: M1 -6.12 (wild p 0.002), M2 -3.22 (0.033), M3 -4.35 (0.004). Without cotton (seven crops): -6.29 / -3.41 / -4.56 (wild 0.003 / 0.031 / 0.006). 1998-2002: -4.29 / -2.57 / -2.55 (wild 0.006 / 0.029 / 0.022; analytic state-dyadic p 0.054 / 0.176 / 0.128). Leave one state out: M1 -7.04 (no Odisha) to -5.06 (no Kerala), max state p 0.042; M3 -5.14 to -3.18 (no Karnataka), max state p 0.064; M2 -4.27 to -2.16 (no Karnataka, state p 0.34). SURVIVES. M2 is the fragile specification; the page's 6-point figure is M1.

## Page and index changes

- `docs/markets.html`: section 10 heading and one clause (cropped-area control, +3.76, p = 0.05) in place of "does not hang on any one control"; section 11 "thin ones, grown on a few plots and sold in small lots" shortened to "a sliver of the district's land"; the census paragraph's last sentence replaced by "Census villages whose only market is a haat show it too."
- `docs/neighbours.html`: section 06 heading; last sentence of its first paragraph (not significant for either crop).
- `docs/irrigation.html`: one sentence in section 05 main text.
- `outputs/ANALYSIS_INDEX.md`: status cells of B18, B19, C1, C10, C11, E11, E13.
- No JS or JSON changed, so no ?v= stamps were bumped.
