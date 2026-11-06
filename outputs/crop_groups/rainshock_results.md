# Q3. Monsoon shocks and the crop-group mix (script 99_crop_groups_rainfall_shocks.py)

Note on the filename: `rainfall_results.md` already holds the rainfall data note, so this
results note is `rainshock_results.md` to avoid overwriting it.

## Question
Does a deficient or excess June-September monsoon, measured against the district's own
history, move land toward millets and pulses, or away from rice, in the rabi of the same
crop year and in the kharif of the next crop year?

## Method
- Unit: 2011-parent district (the crop-group panel unit), 605 units in the 606-district sample.
- Shock: June-September rainfall of calendar year t, as percent departure from the unit's
  own 1990-2020 mean (QC-flagged years left out of that mean). The unit series is the
  area-weighted mean of its child districts' IMD gridded rainfall. Deficient means a
  departure of -20% or worse; excess means +20% or more. Reference is everything in
  between.
- Timing: crop year t-(t+1). The monsoon of t is matched to rabi(t) and to kharif(t+1).
  Kharif(t) is shown as the same-season reference. The kharif(t+1) regressions also
  control for the deficient and excess dummies of the t+1 monsoon (dropping that control
  changes nothing).
- Outcomes: crop-group share of that season's area, in percentage points; log(1 + ha) of
  each group's season area; and log total season area as the mechanical check.
- Specification: OLS with unit FE and state-by-year FE, so identification is from
  districts that were drier or wetter than the rest of their state in the same year.
  Standard errors clustered on the 2011 parent (605 groups; p-values on G-1 df).
- Dose-response: seven bins of the percent departure (<= -40, -40 to -20, -20 to -10,
  reference -10 to +10, +10 to +20, +20 to +40, >= +40).
- Heterogeneity: deficient and excess dummies interacted with irrigation terciles. The
  irrigation share is irr / (irr + unirr) from the 2019-20 Mission Antyodaya layer
  already used on the site (unit value = cropped-area-weighted mean over child
  districts). Tercile ranges: low 0.04-0.43 (mean 0.30), mid 0.43-0.65 (0.54),
  high 0.66-0.99 (0.81).
- Sample rules: unit-years with `flag_season_unreliable` = 0 (drops Odisha 2001-04 and
  2012-13, Bihar 1998-2001, Tamil Nadu rice years, Whole-Year-heavy rows, severe area
  jumps, pre-split donors and late-entry stubs) and rainfall `qc_jjas` = 0; season area > 0.

## Sample
| Season | Unit-years | Units | Deficient years | Excess years | Mean total season area |
|---|---|---|---|---|---|
| Rabi(t) | 11,086 | 586 | 2,915 | 2,145 | exp(10.68) ha |
| Kharif(t+1) | 11,465 | 587 | 2,953 | 2,157 | exp(11.64) ha |
| Kharif(t), reference | 11,638 | 587 | 2,972 | 2,270 | |

Bin counts in the rabi sample: <= -40: 739; -40 to -20: 2,176; -20 to -10: 1,586;
reference: 3,270; +10 to +20: 1,174; +20 to +40: 1,308; >= +40: 837.

Mean rabi shares (pp): wheat 46.3, pulses 22.4, oilseeds 17.5, other 5.2, millets 4.3,
rice 2.3, maize 1.8. Mean kharif shares: rice 50.1, millets 12.0, oilseeds 10.8,
maize 10.6, pulses 7.9, cotton 4.9.

## Headline results (coef, clustered SE, p; n as in the sample table)

### Rabi of the same crop year
| Outcome | Deficient | Excess |
|---|---|---|
| Log total rabi area | -0.073 (0.014), p<0.001 | +0.009 (0.011), p=0.43 |
| Millets share (pp) | +0.393 (0.108), p<0.001, BH q=0.005 | -0.177 (0.115), p=0.12 |
| Pulses share (pp) | +0.131 (0.270), p=0.63 | -0.303 (0.240), p=0.21 |
| Rice share (pp) | -0.307 (0.091), p=0.001, BH q=0.007 | +0.031 (0.076), p=0.68 |
| Wheat share (pp) | +0.382 (0.244), p=0.12 | +0.119 (0.202), p=0.56 |
| Oilseeds share (pp) | -0.384 (0.212), p=0.07 | +0.116 (0.230), p=0.61 |
| Log millets area | +0.002 (0.028), p=0.93 | -0.070 (0.031), p=0.02 |
| Log pulses area | -0.081 (0.026), p=0.002 | +0.026 (0.026), p=0.33 |
| Log wheat area | -0.052 (0.017), p=0.003 | +0.021 (0.019), p=0.27 |
| Log oilseeds area | -0.076 (0.027), p=0.006 | +0.041 (0.031), p=0.19 |
| Log rice area | -0.005 (0.012), p=0.68 | +0.037 (0.024), p=0.12 |

BH q is over the 18 share tests (9 groups x 2 shocks) within the season. Only the rabi
millets and rice deficient coefficients survive.

Dose-response (rabi): log total area -0.146 (0.027) at <= -40, -0.068 (0.015) at -40 to
-20, -0.013 (0.012) at -20 to -10, and +0.027 (0.014) at >= +40. Millets share +0.77
(0.27), +0.37 (0.11), +0.02 (0.13) on the dry side. Log millets area is flat on the dry
side (-0.007 (0.058) at <= -40) while log pulses falls -0.213 (0.058) and log wheat
-0.142 (0.038) at <= -40. Figure: `rainshock_bins_rabi_t.png`.

### Kharif of the next crop year
| Outcome | Deficient (t) | Excess (t) |
|---|---|---|
| Log total kharif area | +0.017 (0.006), p=0.007 | +0.003 (0.008), p=0.76 |
| Millets share (pp) | -0.089 (0.137), p=0.52 | -0.587 (0.172), p=0.001, BH q=0.012 |
| Pulses share (pp) | +0.268 (0.145), p=0.064, q=0.37 | +0.186 (0.140), p=0.18 |
| Rice share (pp) | -0.239 (0.180), p=0.19 | +0.255 (0.199), p=0.20 |
| Log pulses area | +0.060 (0.024), p=0.012 | +0.047 (0.025), p=0.06 |
| Log oilseeds area | +0.066 (0.034), p=0.053 | +0.042 (0.031), p=0.18 |
| Log cotton area | +0.061 (0.031), p=0.048 | -0.010 (0.033), p=0.76 |
| Log millets area | -0.009 (0.032), p=0.78 | -0.011 (0.034), p=0.75 |
| Log rice area | +0.023 (0.023), p=0.31 | +0.038 (0.024), p=0.11 |

Dose-response (kharif t+1): the millets-share fall after a wet year is graded, -0.56
(0.19) at +20 to +40 and -0.61 (0.30) at >= +40. The pulses response after a dry year
sits in the -40 to -20 bin (share +0.28 (0.15), log area +0.055 (0.026)); the <= -40 bin
is +0.41 (0.31) on the share and not distinguishable from zero. Figure:
`rainshock_bins_kharif_t1.png`.

### Kharif of the same crop year (reference)
Deficient: log total kharif area -0.005 (0.007), rice share -0.51 (0.18), pulses share
+0.36 (0.14). Excess: rice share -0.45 (0.20), pulses share +0.34 (0.14). Only the
<= -40 bin cuts kharif area (-0.038 (0.015)). Recorded kharif area barely responds to the
same-year monsoon; that is a property of area-sown statistics, which are fixed before
most of the season's deficit is known.

### Irrigation terciles (deficient monsoon, rabi of the same year)
| Outcome | Low irrigation | Mid | High | Low minus high |
|---|---|---|---|---|
| Log total rabi area | -0.099 (0.035) | -0.122 (0.023) | -0.019 (0.013) | -0.081 (0.036), p=0.026 |
| Millets share (pp) | +0.612 (0.237) | +0.821 (0.244) | -0.064 (0.077) | +0.676 (0.243), p=0.006 |
| Pulses share (pp) | -1.915 (0.676) | +1.021 (0.459) | +0.682 (0.223) | -2.598 (0.699), p<0.001 |
| Wheat share (pp) | +1.612 (0.499) | +0.371 (0.423) | -0.348 (0.278) | +1.960 (0.553), p<0.001 |
| Rice share (pp) | -0.354 (0.174) | -0.565 (0.206) | -0.081 (0.071) | -0.272 (0.192), p=0.16 |
| Log pulses area | -0.208 (0.058) | -0.127 (0.042) | +0.033 (0.037) | -0.240 (0.069), p<0.001 |
| Log millets area | -0.012 (0.042) | +0.028 (0.058) | -0.009 (0.032) | -0.003 (0.051), p=0.95 |
| Log oilseeds area | -0.151 (0.058) | -0.117 (0.046) | +0.002 (0.034) | -0.153 (0.066), p=0.02 |

Kharif(t+1) after a deficient year, by tercile: pulses share -0.12 (0.26), +0.77 (0.26),
+0.14 (0.20); log pulses area -0.005 (0.040), +0.075 (0.034), +0.096 (0.039); log total
area +0.030 (0.013), +0.004 (0.011), +0.019 (0.008). Figure:
`rainshock_irrigation_terciles.png`.

## Robustness (deficient coefficient unless stated)
Rabi, millets share / log total area / rice share:
- baseline +0.393 (0.108) / -0.073 (0.014) / -0.307 (0.091)
- cluster on state (about 30 clusters) +0.393 (0.173) / -0.073 (0.027) / -0.307 (0.179)
- unit + year FE in place of state-by-year +0.500 (0.116) / -0.070 (0.016) / -0.195 (0.113)
- weighted by mean GCA +0.555 (0.145) / -0.092 (0.013) / -0.266 (0.095)
- piecewise z, per SD dry +0.315 (0.096) / -0.064 (0.013) / -0.246 (0.091)
- keep rainfall-QC rows +0.375 / -0.073 / -0.310, all p<0.01
- drop Whole-Year >= 20% rows +0.421 / -0.077 / -0.259, all p<0.01
- drop 1997-98 +0.253 (0.106) / -0.077 (0.014) / -0.206 (0.084)
- units seen >= 20 years +0.351 / -0.073 / -0.321, all p<0.01
- control for Oct-Dec and Jan-May rain (z) +0.393 / -0.074 / -0.264, all p<0.01
- Placebo, next year's monsoon on this rabi (with this year's in the model): deficient
  lead on millets share -0.070 (0.097), on log total area -0.015 (0.012). Passes.
  The excess lead fails for wheat share, -0.696 (0.215), and is marginal for pulses share,
  +0.486 (0.249). Read every excess-monsoon wheat or pulses coefficient with that in mind.

Rabi wheat share is not stable: +0.38 (0.24) with state-by-year FE, -0.79 (0.30) with
year FE only, -0.25 (0.28) GCA-weighted. No wheat claim is made.

Kharif(t+1), excess on millets share: -0.587 (0.172) baseline; -0.587 (0.156) state
clusters; -0.474 (0.157) year FE; -0.591 (0.248) GCA-weighted; -0.343 (0.122) per SD wet;
-0.312 (0.154) dropping 1997-98; -0.633 (0.179) dropping Whole-Year rows. Holds.
Kharif(t+1), deficient on pulses share: +0.268 (0.145) baseline, +0.268 (0.262) state
clusters, +0.618 (0.185) year FE, +0.208 (0.219) GCA-weighted, +0.342 (0.131) per SD dry,
+0.328 (0.146) dropping Whole-Year rows. Positive throughout, significant in about half.
Kharif(t+1) log total area after a deficient year: +0.017 (0.006) baseline, -0.007
(0.008) with year FE only. Not robust.

## What holds
1. A deficient monsoon shrinks the following rabi. Total rabi area falls about 7% after a
   year at -20% or worse and about 14% after -40% or worse, almost entirely in the low and
   mid irrigation terciles (-10% and -12%) and not in the most irrigated third (-2%, n.s.).
2. The rise in the rabi millets share (+0.39 pp on a 4.3 pp mean) is a denominator
   effect. Rabi millet area does not move in any bin or tercile (log change +0.002,
   SE 0.028). Millets hold their ground while pulses (-8%), oilseeds (-8%) and wheat (-5%)
   contract around them.
3. Rabi pulses are among the crops hit hardest in rainfed districts. In the least
   irrigated third, a deficient monsoon cuts rabi pulse area by about 19% (log -0.208) and
   the pulses share by 1.9 pp, while wheat's share rises 1.6 pp because irrigated wheat is
   protected. In the mid and high terciles the pulses share rises (+1.0 and +0.7 pp).
4. After a wet monsoon the next kharif carries less millet: -0.59 pp of kharif area on a
   12 pp mean, graded with the size of the surplus and robust to every check. Millet area
   itself does not fall significantly; the share loss comes from other crops expanding.

## What does not hold
- No reallocation away from rice. The rabi rice share falls 0.31 pp after a drought, but
  rabi rice area does not (log -0.005, SE 0.012), and the next kharif's rice share and area
  show nothing (share -0.24, SE 0.18).
- No drought-driven move into pulses in the rabi on average (share +0.13, SE 0.27). In
  the next kharif there is a modest pulses gain (+0.27 pp, p=0.064; area +6%, p=0.012),
  concentrated in mid and high irrigation districts and not surviving the multiple-testing
  correction or state-level clustering.
- No millet expansion anywhere: not in the rabi after a drought, not in the next kharif.
- The irrigation gradient runs as expected for area loss, but the share responses are not
  monotone in irrigation (mid-tercile pulses and millets often move most).
- Excess-monsoon effects on the rabi are small and mostly null; the one placebo failure
  (wheat on next year's wet monsoon) means wet-year rabi coefficients carry extra noise.

## Plain findings
1. A bad monsoon shrinks the rabi that follows it, by about 7% at -20% or worse and 14% at -40% or worse,
   and the loss sits in rainfed districts.
2. Millet shares rise after a drought only because everything else shrinks; farmers do
   not plant more millet in response to a dry year, and rice area does not give way.
3. In rainfed districts a drought cuts rabi pulse area by about 19% and lifts the wheat
   share, since irrigated wheat is protected.
4. A wet monsoon is followed by a smaller millet share in the next kharif (about -0.6 pp),
   the one share response that is large relative to its noise and robust.

## Caveats
- Recorded area is area sown, not area harvested, so crop failure after sowing is
  invisible here; the kharif(t) reference result should be read that way.
- Irrigation is a single 2019-20 cross-section and is used as a fixed district trait.
- log(1 + ha) includes zeros; rabi rice and rabi maize are zero in most districts, so
  their log-area coefficients mostly reflect a few districts.
- Units with several child polygons (mostly Telangana parents) use an area-weighted
  mean whose child polygons can overlap; weights are approximate there.
- The Andaman districts have no rainfall and fall out.

## Files
- scripts/99_crop_groups_rainfall_shocks.py
- outputs/crop_groups/rainshock_main.csv (all 9 groups, shares and log areas, 3 samples)
- outputs/crop_groups/rainshock_bins.csv
- outputs/crop_groups/rainshock_irrigation.csv
- outputs/crop_groups/rainshock_robustness.csv
- outputs/crop_groups/rainshock_bins_rabi_t.png, rainshock_bins_kharif_t1.png,
  rainshock_bins_kharif_t.png, rainshock_irrigation_terciles.png
- outputs/crop_groups/rainshock_log.txt
