# Irrigation v3: where the fall begins, and how much the within-district null can say

Scripts: `scripts/134_irrigation_threshold.py` (knot), `scripts/135_irrigation_attenuation.py` (measurement error). Site data: `scripts/130_irrigation_v2_site_data.py` now takes the spline from 134 and an `attenuation` block from 135 (`docs/data/site_irrigation_v2.json`). Raw logs: `threshold_raw.md`, `attenuation_raw.md`. Window rule: `scripts/_years.py`, 1998-99 to 2019-20.

## 1. The knot of the two-piece line

Model: D1 = state FE + b_lo min(irr, g) + b_hi max(irr - g, 0), with the Antyodaya 2019-20 share, 606 districts and 30 states. This is the same specification as the referee spline in 124 (q1x). At g = 0.30 it reproduces +2.91 and -4.10.

| | knot | LR 95% set (chi2(1)) | Hansen 7.35 set | wild cluster bootstrap 95% (state, Webb, 1,999) | sup-Wald p (kink against a line) |
|---|---|---|---|---|---|
| D1 full period | **0.43** | 0.36 to 0.65 | 0.29 to 0.75 | **0.29 to 0.72** | 0.077 |
| D0 | 0.38 | 0.31 to 0.67 | 0.20 to 0.80 | 0.25 to 0.77 | 0.156 |
| D0 at 0.5% | 0.57 | 0.43 to 0.63 | 0.39 to 0.67 | 0.43 to 0.72 | 0.042 |
| D2 | 0.42 | 0.30 to 0.73 | 0.21 to 0.80 | 0.26 to 0.73 | 0.151 |
| Evenness | 0.59 | 0.28 to 0.79 | 0.20 to 0.80 | 0.25 to 0.80 | 0.297 |

SSR profile for D1, with LR = n (SSR - min) / min: 0.20 12.2; 0.30 7.0; 0.35 3.9; 0.40 1.1; **0.43 0**; 0.50 3.5; 0.55 1.8; 0.60 1.05 (a second local minimum); 0.65 3.6; 0.70 5.7; 0.80 9.2. The straight-line SSR is 1588.0 and the best kink's is 1548.4. The full profile is in `threshold_ssr_profile.csv`.

**Decision.** The bootstrap interval is 0.43 wide, so the estimate is imprecise. Because 0.50 lies inside both sets, the site now draws the knot at **0.50** instead of 0.30. 56 percent of districts lie above it, against 85 percent above 0.30.

Slopes at 0.50, with wild restricted p from 9,999 draws:
- Full period: below -0.54 (95% -2.70 to +1.63, p 0.607); above -5.69 (-9.13 to -2.26, p 0.037), which is -0.57 per 10 points.
- 2017-22: below -0.24 (p 0.791); above -5.17 (p 0.032).
- D0 at 0.5 percent, full period: above -8.97 (p 0.028).

### Adversarial checks on the knot
- **Recent windows.** (a) 2015-20: knot 0.58, bootstrap 0.42 to 0.73, sup-Wald p 0.047. (b) 2017-22: 0.58, 0.42 to 0.77, p 0.021. Extension only: 0.58, p 0.015. Recent years move the bend up and make the kink clearer.
- **Leave one state out.** The knot stays between 0.42 and 0.60 (median 0.43), and all 30 drops stay inside the main bootstrap interval. The drops that move it most are Madhya Pradesh, Andhra Pradesh, Bihar and Tamil Nadu (0.59 to 0.60). They move it to the second local minimum.
- **Terrain controls.** Knot 0.43, with the same intervals.
- **Dropping districts that share a 2011 parent** (n 542): knot 0.59, bootstrap 0.43 to 0.74, p 0.046.
- **Other irrigation measures break the location.**
  - LUS 1998-2019 mean: knot at the 0.20 edge. On a grid widened down to 0.05 it is 0.16, bootstrap 0.12 to 0.40, sup-Wald p 0.003.
  - Census 2011: 0.19, bootstrap 0.11 to 0.57.
  - Census 1991: no clear kink (p 0.25 to 0.48).
  - These measures run lower (means: Antyodaya 0.55, census 2011 0.47, LUS 0.45, 1991 0.34). Even on a percentile-rank scale, though, the knot sits at the 60th (Antyodaya), 64th (census 2011), 33rd (LUS) and 20th (1991) percentile.
  - **Conclusion:** the fall in the upper part of the range is robust. Where it begins is not pinned down and depends on the irrigation record.
- **The kink itself.** A bend fits only modestly better than a straight declining line over the full period (sup-Wald p 0.077). The fall is clear. "Flat, then falling" is weaker than "falling".

## 2. Attenuation of the within-district slope

The within slope, with district FE plus state x year FE, is D1 -0.225 (0.192), n 11,074, G 581. For D0 at 0.5 percent it is -0.815 (0.290).

**Yearly noise.** Year-on-year changes in LUS irrigation, net of state-year, have a first-order autocorrelation of -0.30. That implies a transitory noise variance of 0.00102 (district bootstrap 0.00077 to 0.00131), which is 21 percent of the FE-residual variance (residual SD 0.068). Correcting for it alone gives -0.28 (95% -0.76 to +0.19). Transitory noise therefore does not explain the null.

**Decadal signal share.** The comparison covers 388 districts in 18 states with both a usable 2001 and a usable 2011 census and LUS coverage. It sets the LUS change from 2000-02 to 2010-12 against the census change from 2001 to 2011, within state:
- Correlation is **0.13** (state cluster bootstrap 0.03 to 0.26).
- Under independent errors, the signal share of the LUS change is **0.12 (0.02 to 0.27)**. The census change's share is 0.15 (0.03 to 0.37).
- Mean changes: LUS +0.066, census +0.050.
- Scaling the within slope by 0.12 gives -1.94, 95% -5.18 to +1.30 per unit (-0.52 to +0.13 per 10 points). That only just excludes the cross-sectional slope above the knot (-5.69).
- The reliability at which the within interval would reach -5.69 is **0.11**. That is inside the bootstrap interval for the signal share.

**Long-difference IV.** dD1 (2000-02 to 2010-12) is regressed on dLUS, with dCensus as the instrument, state FE and parent clusters:
- OLS -0.39 (0.46).
- Reduced form +0.99 (0.45).
- First stage +0.147, **F 4.0** (3.4 with state clusters).
- **IV +6.76 (4.57), 95% -2.19 to +15.72**, p 0.14.
- The Anderson-Rubin set runs from +0.75 with no upper bound.
- With state clusters, the IV interval is -4.21 to +17.74 and the AR set is unbounded both ways.
- For D0 at 0.5 percent, the IV is +1.61 (-10.5 to +13.7).

Adversarial runs on the IV all have first-stage F between 1.8 and 7.1, and every IV interval includes zero:
- 1999-2003 to 2009-13: F 5.9, IV +4.58 (-1.54 to +10.71).
- Drop shared parents: F 3.2.
- Drop LUS territory breaks: F 3.2.
- Spike years set to missing: F 7.1, IV +6.24 (-0.75 to +13.22), AR +1.30 to +27.55.
- Low 2001 unattributed share: F 1.8.
- Dropping West Bengal, Maharashtra and Manipur changes nothing, because they are already outside the overlap.

**Reading.** The within-district null rules out a large fall in D1 that the yearly LUS figures record cleanly. It cannot rule out a fall as steep as the cross-section, because LUS changes over a decade agree only weakly with the census changes. The instrument is too weak to settle the question. Its point estimate is positive, and that sign should not be read as a finding. The same weakness applies to B9: the 1991-2011 census change has a signal share of about 0.15 against LUS.

## 3. Copy on the site
- Irrigation lede, meta and title: "Districts irrigated early grow fewer crops today". The 1991 top-quintile gap (-1.95, p 0.003; -1.78, p 0.002) is the firm part. Section 03 now says that the smooth 1991 curve is clear only in 2017-22 (wild p 0.010), has p 0.11 over the full period, and weakens to 0.19 with reported totals.
- Sections 01, 02, 05 and the method drawer now use the 0.50 knot and report the estimate and its intervals. Chart v9 gains a third row with the noise-scaled within slope.
- Section 04 adds a plain link to the Neighbours page.
- Overview card (index.html, scene 09) rewritten to match.
- No docs page says the peak is at about 28 percent or 24 to 36 percent. Neither scatterQuad nor its "peak" label is called anywhere, and data.js overrides the stale models note.
