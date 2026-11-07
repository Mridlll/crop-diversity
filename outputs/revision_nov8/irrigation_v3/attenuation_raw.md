# Irrigation within districts: attenuation check

Window rule 1998-99 to 2019-20 (scripts/_years.py). Panel: 11,099 district-years with LUS irrigation, 600 districts.

## A. Within slope (district FE + state x year FE, clusters = 2011 parents)
- D1: -0.225 (0.192), n 11,074, G 581
- D0_05: -0.815 (0.290), n 11,074, G 581
Residual SD of irrigation after district and state x year FE: 0.0681 (variance 0.00464); total SD 0.311.

## B. Yearly noise in the LUS share (first-difference autocovariance)
- all years: n 9,253 pairs of changes; Var(change) 0.00345; implied noise variance 0.00102 (SD 0.032; district bootstrap 95% 0.00077 to 0.00131); first-order autocorrelation of changes -0.30. Mean years per district 18.5. Implied reliability of the FE-residual variation 0.79 (noise share 0.21)
- spike-flagged years dropped: n 9,187 pairs of changes; Var(change) 0.00273; implied noise variance 0.00106 (SD 0.033; district bootstrap 95% 0.00083 to 0.00137); first-order autocorrelation of changes -0.39. Mean years per district 18.5. Implied reliability of the FE-residual variation 0.78 (noise share 0.22)
Within D1 slope corrected for this reliability: -0.284 (SE 0.242), 95% -0.76 to +0.19 per unit share.

## C-D. Long differences, census change as the instrument for the LUS change

### main: 2000-02 to 2010-12 (at least 2 of 3 years each end)

n 388 districts (388 parents, 18 states). Mean change: LUS +0.066, census +0.050. Within state: Var(dLUS) 0.0091, Var(dCensus) 0.0072, Cov 0.0011, correlation 0.13; signal share of the LUS change 0.12, of the census change 0.15.
State cluster bootstrap 95%: LUS signal share 0.02 to 0.27; census 0.03 to 0.37; correlation 0.03 to 0.26.
- dD1 (clusters 2011 parent): OLS -0.387 (0.455); reduced form on census change +0.991 (0.449); first stage +0.147, F 4.0; IV +6.764 (4.567), 95% -2.19 to +15.72, p 0.139; AR 95% +0.75 to +40.00 (unbounded)
- dD1 (clusters state): OLS -0.387 (0.333); reduced form on census change +0.991 (0.566); first stage +0.147, F 3.4; IV +6.764 (5.202), 95% -4.21 to +17.74, p 0.211; AR 95% -40.00 to +40.00 (unbounded)
- dD0_05 (clusters 2011 parent): OLS -0.747 (0.798); reduced form on census change +0.236 (0.874); first stage +0.147, F 4.0; IV +1.611 (6.171), 95% -10.48 to +13.71, p 0.794; AR 95% -16.05 to +40.00 (unbounded)
- dD0_05 (clusters state): OLS -0.747 (0.613); reduced form on census change +0.236 (0.966); first stage +0.147, F 3.4; IV +1.611 (6.870), 95% -12.88 to +16.11, p 0.817; AR 95% -40.00 to +40.00 (unbounded)

## E. Adversarial

### windows 1999-2003 to 2009-2013 (at least 3 of 5 years)

n 405 districts (405 parents, 19 states). Mean change: LUS +0.061, census +0.047. Within state: Var(dLUS) 0.0066, Var(dCensus) 0.0073, Cov 0.0012, correlation 0.17; signal share of the LUS change 0.18, of the census change 0.16.
- dD1 (clusters 2011 parent): OLS -1.147 (0.469); reduced form on census change +0.751 (0.367); first stage +0.164, F 5.9; IV +4.584 (3.125), 95% -1.54 to +10.71, p 0.142; AR 95% +0.20 to +28.60 (connected)
- dD0_05 (clusters 2011 parent): OLS -1.842 (0.861); reduced form on census change -0.296 (0.682); first stage +0.164, F 5.9; IV -1.805 (4.083), 95% -9.81 to +6.20, p 0.658; AR 95% na to na ()

### drop districts sharing a 2011 parent

n 347 districts (347 parents, 17 states). Mean change: LUS +0.063, census +0.050. Within state: Var(dLUS) 0.0092, Var(dCensus) 0.0075, Cov 0.0010, correlation 0.12; signal share of the LUS change 0.11, of the census change 0.13.
- dD1 (clusters 2011 parent): OLS -0.475 (0.491); reduced form on census change +0.945 (0.453); first stage +0.133, F 3.2; IV +7.117 (5.335), 95% -3.34 to +17.57, p 0.182; AR 95% +0.40 to +40.00 (unbounded)
- dD0_05 (clusters 2011 parent): OLS -0.577 (0.861); reduced form on census change +0.085 (0.922); first stage +0.133, F 3.2; IV +0.643 (7.036), 95% -13.15 to +14.43, p 0.927; AR 95% na to na ()

### drop districts with an LUS territory break

n 327 districts (327 parents, 18 states). Mean change: LUS +0.066, census +0.050. Within state: Var(dLUS) 0.0083, Var(dCensus) 0.0055, Cov 0.0009, correlation 0.13; signal share of the LUS change 0.11, of the census change 0.16.
- dD1 (clusters 2011 parent): OLS -0.440 (0.508); reduced form on census change +0.945 (0.641); first stage +0.163, F 3.2; IV +5.798 (5.696), 95% -5.37 to +16.96, p 0.309; AR 95% -1.65 to +40.00 (unbounded)
- dD0_05 (clusters 2011 parent): OLS -0.800 (0.911); reduced form on census change -0.063 (1.286); first stage +0.163, F 3.2; IV -0.386 (7.835), 95% -15.74 to +14.97, p 0.961; AR 95% na to na ()

### spike-flagged LUS years set to missing

n 385 districts (385 parents, 18 states). Mean change: LUS +0.066, census +0.050. Within state: Var(dLUS) 0.0082, Var(dCensus) 0.0071, Cov 0.0013, correlation 0.17; signal share of the LUS change 0.16, of the census change 0.18.
- dD1 (clusters 2011 parent): OLS -0.665 (0.471); reduced form on census change +1.128 (0.441); first stage +0.181, F 7.1; IV +6.237 (3.565), 95% -0.75 to +13.22, p 0.080; AR 95% +1.30 to +27.55 (connected)
- dD0_05 (clusters 2011 parent): OLS -1.151 (0.806); reduced form on census change +0.572 (0.865); first stage +0.181, F 7.1; IV +3.162 (5.334), 95% -7.29 to +13.62, p 0.553; AR 95% na to na ()

### drop West Bengal, Maharashtra, Manipur (thin LUS coverage)

n 388 districts (388 parents, 18 states). Mean change: LUS +0.066, census +0.050. Within state: Var(dLUS) 0.0091, Var(dCensus) 0.0072, Cov 0.0011, correlation 0.13; signal share of the LUS change 0.12, of the census change 0.15.
- dD1 (clusters 2011 parent): OLS -0.387 (0.455); reduced form on census change +0.991 (0.449); first stage +0.147, F 4.0; IV +6.764 (4.567), 95% -2.19 to +15.72, p 0.139; AR 95% +0.75 to +40.00 (unbounded)
- dD0_05 (clusters 2011 parent): OLS -0.747 (0.798); reduced form on census change +0.236 (0.874); first stage +0.147, F 4.0; IV +1.611 (6.171), 95% -10.48 to +13.71, p 0.794; AR 95% na to na ()

### 2001 census unattributed share at or below its median

n 194 districts (194 parents, 18 states). Mean change: LUS +0.037, census +0.045. Within state: Var(dLUS) 0.0066, Var(dCensus) 0.0096, Cov 0.0010, correlation 0.12; signal share of the LUS change 0.15, of the census change 0.10.
- dD1 (clusters 2011 parent): OLS -0.128 (0.614); reduced form on census change +1.199 (0.591); first stage +0.099, F 1.8; IV +12.061 (10.731), 95% -8.97 to +33.09, p 0.261; AR 95% -40.00 to +40.00 (unbounded)
- dD0_05 (clusters 2011 parent): OLS -0.420 (1.449); reduced form on census change +0.930 (0.923); first stage +0.099, F 1.8; IV +9.352 (12.306), 95% -14.77 to +33.47, p 0.447; AR 95% na to na ()

## F. What the within slope can and cannot rule out

Cross-sectional slope above the knot (0.50): -5.69 per unit (95% -9.13 to -2.26), i.e. -0.57 per 10 points.
Within slope as estimated: -0.225 per unit, 95% -0.60 to +0.15.
- corrected by the yearly-noise reliability 0.79: -0.28 per unit, 95% -0.76 to +0.19; the cross-sectional slope -5.69 is outside this interval.
- corrected by the decadal signal share (census comparison) 0.12: -1.94 per unit, 95% -5.18 to +1.30; the cross-sectional slope -5.69 is outside this interval.
Reliability at which the lower end of the within interval would reach the cross-sectional slope: 0.11.
Long-difference IV (main, parent clusters): +6.76, 95% -2.19 to +15.72; AR 0.75 to 40.0 (unbounded).
