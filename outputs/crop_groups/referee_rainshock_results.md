# Referee check of Q3 (monsoon shocks and the crop-group mix)

Scripts: `scripts/100_referee_rainshock_checks.py` (the script 99 build and estimator, copied unchanged, plus checks) and `scripts/101_referee_rainshock_followups.py`. Logs: `referee_rainshock_log.txt`, `referee_rainshock_followups_log.txt`. Coefficient table: `referee_rainshock_checks.csv`. Script 99 and its outputs were not modified.

## What was confirmed
- The baseline numbers reproduce exactly.
- The crop mapping matches the specified groups.
- The year timing is right. APY "t-(t+1)" kharif rice collapses in 2002 and 2009, which were the all-India drought monsoons. Change in kharif yield correlates +0.43 with change in the same-year monsoon departure and 0.00 with the following year's.
- Spatial correlation does not account for the rabi results. Permuting whole rainfall years 300 times gives randomization p-values of 0.007 for the millet share, 0.010 for the rice share and 0.003 for log total rabi area. Clustering on state gives p of 0.032, 0.098 and 0.012.
- The rabi results survive unit-specific linear trends.
- The lag placebo is clean. The lead placebo is clean except for the wheat share result the analyst already reported.

## Problems found
1. **Unreliable annual totals stay in the sample.** Script 99 drops only `flag_season_unreliable`. It keeps `flag_annual_unreliable` unit-years (severe GCA jumps, pre-split donors, child gaps, years before a unit enters), and it keeps 1997-98, which the panel note calls weak. Dropping those years, the mild-jump years and 1997 shrinks several estimates.
2. **log(1+ha) over every unit hides real area changes.** In units that actually grow the crop that season:
   - Rabi rice area falls about 25-30% after a drought. The log coefficient is -0.30 (SE 0.07) and -0.255 (0.063) after the data-flag drops. Total rabi area in these units does not fall (-0.01).
   - Next-kharif millet area falls about 5% after an excess monsoon: -0.054 (0.017), and -0.047 to -0.055 in every cleaning variant. The lead placebo on this measure is -0.025 (0.021).
3. **The next-kharif millet share result depends heavily on one year.** About half of it comes from kharif 1997-98, the first and weakest panel year:
   - Dropping that year alone: -0.31 (0.15).
   - Dropping it with the unreliable annual rows: -0.24 (0.14), p=0.086. Clustered on state, p=0.098.
   - On that cleaned sample the bins are not monotone: -0.06 at +10 to +20, -0.30 at +20 to +40, -0.06 at +40 or more.
4. **The low-irrigation tercile mixes in hill and north-eastern states.** 79 of its 188 units are in Jammu and Kashmir, Himachal, Uttarakhand, Arunachal, Assam and the other north-eastern states, where rabi wheat is mostly rainfed.
   - Wheat area does not rise there (+0.02, SE 0.04). The +1.6 pp wheat share is a denominator effect.
   - Without those states, the wheat share is +1.04 (0.57) and the pulses share -1.12 (0.85, p=0.19).
   - Pulse area still falls by -0.21 (0.08).

## Corrected numbers (unit and state-by-year FE; unit-clustered SE; cleaned sample drops annual-unreliable rows, mild GCA jumps and 1997)
| Result | Analyst | Cleaned |
|---|---|---|
| Log total rabi area, deficient | -0.073 (0.014) | -0.063 (0.013) |
| Rabi millet share, deficient (pp) | +0.39 (0.11) | +0.21 (0.10), p=0.04 |
| Rabi rice share, deficient (pp) | -0.31 (0.09) | -0.21 (0.09), p=0.03 |
| Log rabi rice area, rabi-rice units | -0.005 (log1p, all units) | -0.255 (0.063) |
| Next-kharif millet share, excess (pp) | -0.59 (0.17) | -0.22 to -0.24 (0.13-0.14), p≈0.09-0.10 |
| Log next-kharif millet area, millet units | not reported | -0.053 (0.019) |
| Log rabi pulses area, low irrigation | -0.208 (0.058) | -0.221 (0.055) |
| Log rabi total area, low / mid / high irrigation | -0.099 / -0.122 / -0.019 | -0.095 / -0.106 / -0.026 |
