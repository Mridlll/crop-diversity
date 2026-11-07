# Seasons and double counting: season-separated crop diversity

Scripts: `scripts/104_season_separated_diversity.py` (construction, estimates), `scripts/107_seasons_supplement.py` (extra checks). Outputs in `outputs/revision_nov8/seasons_*`. Logs: `seasons_log.txt`, `seasons_supplement_log.txt`.

## Data and method

- **Crop data.** Raw APY 1997-2019 with script 76's cleaning copied verbatim. Annual D1 reproduces script 76 to 1.8e-15 on all 14,089 district-years.
- **Season pools.** Kharif = Kharif + Autumn + Winter rows. Rabi = Rabi. Summer = Summer. These are the same slots as `site_monoculture.json`. Whole Year crops (sugarcane, coconut, banana and similar) cannot be assigned to one season. I handle them three ways:
  1. **D1_ws (literal spec):** Whole Year area is added to every season pool. D1_ws is then the season-area-weighted mean of the pool D1s.
  2. **D1_ws_noW:** Whole Year rows are left out of the season pools.
  3. **A_alpha (preferred):** an exact Shannon decomposition on the exclusive partition {kharif, rabi, summer, whole year}. H_annual = H_alpha + H_beta, where H_alpha = sum_s w_s H_s and w_s is the block's share of GCA. Then D1_annual = A_alpha x B_beta exactly. A_alpha = exp(H_alpha) is within-season variety. B_beta >= 1 is the between-season (rotation) multiplier. beta_share = H_beta / H_annual is the share of annual entropy that comes from between seasons.
  - Option 1 inflates small season pools. In Assam the rabi D1 is 8.29 with Whole Year crops added and 3.72 without (rabi is 24% of GCA). Because of this, D1_ws exceeds annual D1 in 5.7% of districts. I report D1_ws because it was asked for, but A_alpha and D1_ws_noW are the measures to quote.
- **Season reliability.** Unit-year `flag_season_unreliable` from `outputs/crop_groups/crop_group_flags.csv` (district inherits its 2011 unit's flag), plus a district-level Whole Year share below 0.5.
  - 12,684 of 14,089 district-years are season-reliable.
  - Districts need at least 5 reliable years. **561 of the 606 sample districts qualify.**
  - 45 drop out: Tamil Nadu 31 (rice season flag on 85% of district-years), Kerala 13 (Whole Year at least half of GCA in 96% of district-years), Andaman 1. These results therefore say nothing about Tamil Nadu or Kerala.
  - Robustness sample "flags ignored" keeps all 606.
- **Cropping intensity.** Four versions:
  - `ci_lus`: DES Land Use Statistics GCA / NSA, district-year, from the India Data Portal (`raw_lus/classification-of-area.csv`, matched for the irrigation-panel task). This is the preferred version. Values are kept in 0.9-3.0 and cover 12,439 district-years and 557 of the 561 districts.
  - `ci_apy`: APY GCA / (Whole Year + largest season). This is an upper bound.
  - `ci_ay`: Mission Antyodaya 2019-20.
  - `secc_2crop`: SECC 2011, share of land that is irrigated and double-cropped.
  - The versions agree moderately. Spearman of LUS with APY is 0.72, with Antyodaya 0.66, and with SECC 0.59. APY GCA averages 0.895 of LUS GCA (median 0.92), because APY misses fodder and much horticulture. This affects every index here.
- **Irrigation spec.** This matches the site: y ~ irr + irr^2 + C(state), SEs clustered on the 2011 parent (`shrug_key`), and irrigation = Antyodaya irr / (irr + unirr). Turning-point CIs use the delta method.
- **Systems** come from `docs/data/site_monoculture.json` (modal district-year class): rice-wheat n = 117 and rice-rice n = 39 within the 561.

## Headline numbers (561 districts; mean, SE in parentheses)

| measure | mean (SE) | median |
|---|---|---|
| Annual D0 / D1 / D2 | 21.06 (0.29) / 4.83 (0.09) / 3.42 (0.07) | 22.4 / 4.29 / 3.01 |
| Kharif D0 / D1 / D2 | 15.02 (0.25) / 3.41 (0.08) / 2.49 (0.06) | |
| Rabi D0 / D1 / D2 (n 560) | 12.24 (0.21) / 3.84 (0.08) / 2.88 (0.06) | |
| D1_ws (Whole Year in each pool) | 3.29 (0.06) | 2.98 |
| D1_ws_noW | 2.77 (0.06) | 2.41 |
| A_alpha (within-season, exact) | 2.63 (0.05) | 2.34 |
| B_beta (between-season multiplier) | 1.85 (0.016) | 1.94 |
| beta_share of annual entropy | 0.420 (0.007) | 0.418 |
| Annual D1 minus D1_ws | 1.54 (0.046) | |
| Annual D1 / D1_ws | 1.50 (0.013) | |

GCA-weighted national figures: annual D1 5.52, D1_ws 3.64, B_beta 1.95, beta_share 0.42.

**Reading.** About two fifths of annual Shannon entropy comes from different crops in different seasons, not from variety within a season. A typical district grows about 2.6 effective crops in a given season. Rotation then multiplies that by about 1.85 to give the annual figure of about 4.8.

### Rice-wheat and rice-rice

Mean (SE) by system:

| system | n | annual D1 | A_alpha | B_beta | beta_share | kharif D1 (no W) | rabi D1 (no W) | LUS CI | irrigation |
|---|---|---|---|---|---|---|---|---|---|
| rice-wheat | 117 | 4.25 (0.14) | 1.99 (0.07) | 2.16 (0.02) | 0.58 (0.01) | 2.19 (0.10) | 1.94 (0.07) | 1.61 (0.02) | 0.76 (0.02) |
| rice-rice | 39 | 3.69 (0.25) | 2.17 (0.15) | 1.70 (0.06) | 0.43 (0.03) | 2.07 (0.18) | 3.77 (0.16) | 1.62 (0.05) | 0.56 (0.02) |
| rice-only | 143 | 3.66 (0.15) | 2.41 (0.08) | 1.48 (0.03) | 0.32 (0.01) | 2.35 (0.10) | 3.75 (0.12) | 1.23 (0.01) | 0.40 (0.01) |
| cotton-only | 16 | 6.64 (0.45) | 4.20 (0.25) | 1.60 (0.06) | 0.24 (0.02) | | | 1.18 (0.02) | 0.53 (0.02) |
| millet-based | 19 | 7.38 (0.60) | 3.79 (0.27) | 1.94 (0.06) | 0.34 (0.02) | | | 1.27 (0.04) | 0.39 (0.04) |

- **Rice-wheat.** beta_share is 0.21 above all other districts (Welch p = 1.7e-29, n 117 vs 405). More than half of rice-wheat's annual entropy is the rotation itself. Within a season these districts have about 2 effective crops. That is the lowest of any major system and about the level of the rice-oilseed districts (1.72). Rice-wheat looks average on the annual index (4.25 against a national 4.83) and sits near the bottom within season.
- **Rice-rice.** Same-crop double cropping does not create between-season diversity. beta_share is only 0.055 above the rest (p = 0.050), and B_beta is 1.70 against 2.16 for rice-wheat. The second-season pool in these districts (Assam, coastal Andhra, Odisha, West Bengal) is the non-rice rabi pool, which is fairly diverse (3.77). The second rice crop is mostly in the summer slot. Annual D1 does not inflate rice-rice the way it inflates rice-wheat.

## Rankings: annual against within-season

- Spearman of annual D1 with each within-season measure: D1_ws 0.868, D1_ws_noW 0.890, A_alpha 0.882, kharif D1 0.895. With rabi D1 it is only 0.158.
- Median absolute rank move (out of 561): 58 on D1_ws and 52 on A_alpha. Moves of more than 100 places: 135 districts on D1_ws and 109 on A_alpha.
- **Fallers** (annual rank flatters them) on A_alpha are the Indo-Gangetic rice-wheat belt:
  - Yamunanagar fell 192 places (D1 3.75 annual, 1.32 within season, B 2.84).
  - Pashchim Champaran fell 184.
  - Also Sitamarhi, Gopalganj, Muzaffarpur, Haridwar, Udham Singh Nagar, Ambala, Baran and Jhunjhunu.
  - The 50 largest fallers are 23 rice-wheat and 5 soybean-wheat. By state: Bihar 10, Rajasthan 10, UP 8, Haryana 7.
- **Risers** on A_alpha are single-season rainfed rice districts, where annual and within-season diversity are nearly the same thing (B about 1.1):
  - Narayanpur rose 239 places.
  - Nabarangpur rose 229.
  - Also Dantewada, Koraput, Balangir, Bastar, Anjaw, Nayagarh, Anugul, Baramulla, Tirap and Mamit.
  - Of the 50 largest risers, 37 are rice-only. By state: Odisha 15, Chhattisgarh 7, Arunachal 6, Mizoram 6.
  - These districts are not diverse. They only lose less to the correction.
- On D1_ws the risers list is dominated by Assam (26 of the top 50), and this is the Whole Year inflation described above. Do not use that list. Use `seasons_movers_alpha.csv`.

## Gross against net: cropping intensity

LUS cropping intensity in the sample: mean 1.41 (SE 0.011), median 1.37, n 557.

Cross-section with state baseline, standardised coefficient (SE), clustered:

| y on CI | LUS | APY bound | Antyodaya | SECC 2-crop |
|---|---|---|---|---|
| annual D1 | -0.028 (0.049), p 0.57 | +0.171 (0.057) | +0.106 (0.050) | -0.064 (0.050) |
| A_alpha | **-0.175 (0.048), p 0.0003** | -0.046 (0.057) | -0.024 (0.055) | -0.172 (0.048) |
| B_beta | **+0.381 (0.034), p < 1e-15** | +0.490 (0.046) | +0.316 (0.042) | +0.263 (0.039) |
| beta_share | +0.359 (0.046) | +0.317 (0.049) | +0.188 (0.043) | +0.257 (0.042) |

LUS CI tertiles:

| tertile | mean CI | annual D1 | A_alpha | B_beta | beta_share |
|---|---|---|---|---|---|
| low | 1.15 | 4.98 | 3.02 | 1.61 | 0.31 |
| mid | 1.38 | 5.27 | 2.80 | 1.87 | 0.40 |
| high | 1.72 | 4.25 | 2.04 | 2.09 | 0.55 |

- **Annual D1 is roughly flat in cropping intensity**, but only because two opposite movements cancel. More intensive districts carry a larger between-season multiplier and less variety within each season.
- **The sign on annual D1 depends on which CI is used.** It is null on LUS and SECC and positive on the APY bound and Antyodaya. Do not quote an annual-D1-on-intensity coefficient as a finding. The B and A_alpha signs are stable across all four measures. The A_alpha magnitude is not: it is significant on LUS and SECC and null on APY and Antyodaya.
- **Within district over time** (district and year FE, 10,398 district-years, 557 districts, LUS CI):
  - A one-unit rise in CI raises H_beta by 0.113 (SE 0.016, p 1e-12).
  - It raises annual H by 0.088 (SE 0.029, p 0.002).
  - It leaves H_alpha unchanged (-0.025, SE 0.025, p 0.32).
  - So when a district adds a second season, its annual index rises mechanically through rotation, with no measurable change in within-season variety.
- Irrigation raises LUS CI: +0.42 per unit of irrigation share (SE 0.065, state FE, n 557).

## Does the irrigation inverted U hold within season?

Main sample n = 561. b2 is the coefficient on irr^2 (SE). tp is the turning point [95% CI]. The upslope and downslope columns give the slope at the 2nd and 98th percentiles of irrigation (0.95).

| y | b2 (SE) | p | tp [95% CI] | upslope at p2, p | downslope at p98, p |
|---|---|---|---|---|---|
| annual D1 | -7.12 (1.40) | <1e-6 | 0.322 [0.229, 0.414] | +2.62, 0.017 | -8.94, <1e-6 |
| D1_ws | -4.33 (0.86) | <1e-6 | 0.304 [0.204, 0.405] | +1.44, 0.034 | -5.59, <1e-6 |
| D1_ws_noW | -3.69 (0.75) | <1e-6 | 0.219 [0.089, 0.349] | +0.60, 0.31 | -5.39, <1e-6 |
| A_alpha | -3.54 (0.69) | <1e-6 | 0.221 [0.096, 0.346] | +0.58, 0.29 | -5.15, <1e-6 |
| kharif D1 | -4.41 (1.22) | 0.0003 | 0.281 [0.137, 0.424] | +1.26, 0.18 | -5.90, <1e-6 |
| rabi D1 (n 514, dry share >= 5%) | -3.23 (1.12) | 0.004 | 0.296 [0.081, 0.510] | +0.95, 0.35 | -4.23, <1e-6 |
| D0_ws | -5.90 (2.09) | 0.005 | 0.341 [0.152, 0.531] | | |
| D2_ws | -2.88 (0.64) | <1e-5 | 0.285 [0.168, 0.402] | | |
| B_beta | +0.10 (0.30) | 0.73 | none | | |

**Decomposition of the curvature.** Because H_ann = H_alpha + H_beta exactly, the coefficients add:

| sample | b2 on H_ann | b2 on H_alpha | b2 on H_beta |
|---|---|---|---|
| main | -1.49 (0.31) | -1.45 (0.27) | -0.04 (0.16), p 0.81 |
| flags ignored (606) | -1.43 | -1.33 | -0.11, p 0.49 |
| drop shared parents (501) | -1.29 | -1.36 | +0.07, p 0.67 |

Essentially all of the hump is within season. The between-season part rises linearly with irrigation and has no curvature: H_beta +0.26 per unit of irrigation (SE 0.043, p 2e-9), beta_share +0.31 (SE 0.037).

Predicted fall from the turning point to the 98th percentile of irrigation:

| index | fall | share of its mean |
|---|---|---|
| annual D1 | -2.81 | 58% |
| D1_ws | -1.81 | 55% |
| D1_ws_noW | -1.97 | 71% |
| H_alpha (entropy units) | -0.67 | 78% |

Proportionally, the irrigated downslope is steeper within season than annually, because rotation props up the annual index at high irrigation.

**What holds, and what does not:**
- **The downslope holds** on every within-season index, in all three samples. The p98 slope is significant at p < 0.001 everywhere (largest p 0.0002, D0_ws in the drop-shared-parents sample). It also survives a LUS CI control: b2 on D1_ws is -4.28 (SE 0.88) and on A_alpha -3.38 (SE 0.69). It survives GCA weighting: D1_ws -4.59 (SE 1.33) and A_alpha -4.06 (SE 1.05).
- **The upslope does not hold as a separate finding within season.** On the clean within-season indices (A_alpha, D1_ws_noW, kharif D1, rabi D1), the slope at low irrigation is positive but not significant (p 0.18-0.35). The turning point moves left to about 0.22, with a CI reaching down to 0.09. On these indices the shape reads better as flat, then falling past about a fifth to a third irrigated, than as a clean inverted U. On annual D1 and on D1_ws the upslope is significant (p 0.017 and 0.034). Part of the annual rise from low to moderate irrigation is therefore the second season appearing.
- Turning points on annual D1 here (0.32 on 561; 0.28 on 606) sit inside the site's quoted range of 0.245-0.513. The within-season ones (0.22-0.30) sit at or below its lower edge.

## Robustness summary

- **Season flags ignored (606 districts):** every within-season b2 is still negative at p < 0.005. H_beta curvature is still null.
- **Drop shared 2011 parents (501):** same pattern.
- **Whole Year treatment:** three versions are reported. Conclusions are identical, except for the Assam-driven movers list on D1_ws.
- **Cropping intensity:** the conclusions about the intensity components (B up, A_alpha down) hold on all four CI measures for B. For A_alpha they hold on two of the four.

## Data limits and what I could not get

- **Season labels follow state reporting conventions.** Assam's autumn (ahu) and winter (sali) rice both fall in the kharif pool, so part of Assam's rice double-cropping is invisible as rotation.
- **APY covers about 90% of LUS GCA.** Vegetables, fodder and most horticulture are missing or patchy, and they are disproportionately rabi and summer crops. Within-season rabi diversity is probably understated.
- **The irrigation cross-section has a timing mismatch.** It uses Antyodaya 2019-20 irrigation against cropping averaged over 1997-2019, as on the site.
- **LUS CI is quality-filtered to 0.9-3.0.** It is unavailable for 4 sample districts.
- **No plot-level data.** I cannot see true sequential cropping on the same plot. "Between-season" means different crop mixes in different seasons at district level, which includes different land being sown in different seasons, not only the same field rotating.

## Findings a reader should take away

1. Roughly two fifths of the annual crop diversity index comes from different crops in different seasons. A typical district grows about 2.6 effective crops at any one time, not the 4.8 the annual figure suggests.
2. Rice-wheat districts look middling on the annual index only because of the rotation. Within a season they grow about two effective crops, among the lowest in the country. Their place in annual rankings flatters them by 100 to 190 places in the Bihar, Haryana and Uttarakhand rice-wheat belt. Rice-rice districts are not flattered this way.
3. Higher cropping intensity buys rotation, not variety. Across districts and within districts over time, a second season adds to the between-season multiplier and does not add within-season variety. Across districts, it goes with less within-season variety.
4. The fall in diversity at high irrigation is entirely a within-season phenomenon. The downslope is robust in every check. The rising arm at low irrigation is weak once seasons are separated, so the within-season shape is closer to flat, then falling.
