# IV follow-up: recentred shift-share and CGWB depth

CGWB pre-monsoon readings 2013-15: 18,684 stations, 18,675 placed in a 2011 district; 521 districts with a depth
District pre-monsoon depth (m), analysis districts with >= 3 wells: median 7.4, p10 4.1, p90 12.7

Correlation, across districts, of baseline tubewell fraction of irrigated area with baseline irrigation: 0.576

## shift-share, just identified, NO baseline control (106 replication)
First stage (n=8,073, districts 463, clusters 463): z_ss -0.2126 (0.0562); cluster-robust F = 14.30
- D1: reduced form z_ss -0.632 (0.499); 2SLS +2.972 (2.439, p=0.224, 95% CI -1.81 to +7.75); OLS -0.140 (0.239); n=8,073
- D0_05: reduced form z_ss -1.706 (0.825); 2SLS +8.025 (4.386, p=0.068, 95% CI -0.57 to +16.62); OLS -0.759 (0.333); n=8,073
- D0: reduced form z_ss -2.898 (1.198); 2SLS +13.632 (6.758, p=0.044, 95% CI +0.39 to +26.88); OLS -0.737 (0.548); n=8,073
- D2: reduced form z_ss -0.172 (0.404); 2SLS +0.808 (1.906, p=0.672, 95% CI -2.93 to +4.54); OLS +0.107 (0.206); n=8,073
- even: reduced form z_ss +0.004 (0.023); 2SLS -0.020 (0.110, p=0.853, 95% CI -0.24 to +0.20); OLS +0.011 (0.011); n=8,073
- cereal_share: reduced form z_ss +0.046 (0.035); 2SLS -0.218 (0.175, p=0.214, 95% CI -0.56 to +0.13); OLS +0.029 (0.018); n=8,073

## shift-share, recentred: baseline irrigation x year controls
First stage (n=8,073, districts 463, clusters 463): z_ss -0.1389 (0.0740); cluster-robust F = 3.53
- D1: reduced form z_ss +0.163 (0.691); 2SLS -1.173 (5.044, p=0.816, 95% CI -11.06 to +8.71); OLS -0.190 (0.245); n=8,073
- D0_05: reduced form z_ss +0.159 (0.951); 2SLS -1.144 (6.845, p=0.867, 95% CI -14.56 to +12.27); OLS -0.911 (0.339); n=8,073
- D0: reduced form z_ss -0.512 (1.475); 2SLS +3.689 (10.734, p=0.731, 95% CI -17.35 to +24.73); OLS -0.946 (0.546); n=8,073
- D2: reduced form z_ss +0.174 (0.585); 2SLS -1.252 (4.307, p=0.771, 95% CI -9.69 to +7.19); OLS +0.096 (0.212); n=8,073
- even: reduced form z_ss +0.026 (0.032); 2SLS -0.184 (0.247, p=0.457, 95% CI -0.67 to +0.30); OLS +0.011 (0.011); n=8,073
- cereal_share: reduced form z_ss -0.109 (0.042); 2SLS +0.788 (0.474, p=0.097, 95% CI -0.14 to +1.72); OLS +0.040 (0.017); n=8,073

## shift-share, recentred, with square
First stage (n=8,073, districts 463, clusters 463): z_ss +0.4758 (0.1776); z_ss2 -0.2529 (0.0594); cluster-robust F = 13.86
- D1: reduced form z_ss +1.088 (1.667); 2SLS +0.942 (1.681, p=0.575, 95% CI -2.35 to +4.24); OLS -0.190 (0.245); n=8,073
  quadratic: irr -10.247 (27.596), irr2 +6.501 (16.020), p(irr2)=0.685, peak 0.788 (0.224); first-stage F irr 13.86, irr2 22.73
- D0_05: reduced form z_ss -0.824 (2.269); 2SLS -1.504 (2.903, p=0.605, 95% CI -7.19 to +4.19); OLS -0.911 (0.339); n=8,073
  quadratic: irr +0.397 (32.178), irr2 -1.105 (18.557), p(irr2)=0.953, peak 0.180 (11.559); first-stage F irr 13.86, irr2 22.73
- D0: reduced form z_ss -4.805 (3.777); 2SLS -4.744 (4.532, p=0.296, 95% CI -13.63 to +4.14); OLS -0.946 (0.546); n=8,073
  quadratic: irr +39.854 (65.748), irr2 -25.914 (38.529), p(irr2)=0.502, peak 0.769 (0.160); first-stage F irr 13.86, irr2 22.73
- D2: reduced form z_ss +0.821 (1.411); 2SLS +0.569 (1.470, p=0.699, 95% CI -2.31 to +3.45); OLS +0.096 (0.212); n=8,073
  quadratic: irr -9.065 (23.605), irr2 +5.598 (13.717), p(irr2)=0.683, peak 0.810 (0.187); first-stage F irr 13.86, irr2 22.73
- even: reduced form z_ss +0.117 (0.075); 2SLS +0.079 (0.087, p=0.360, 95% CI -0.09 to +0.25); OLS +0.011 (0.011); n=8,073
  quadratic: irr -1.314 (1.760), irr2 +0.809 (1.026), p(irr2)=0.431, peak 0.811 (0.096); first-stage F irr 13.86, irr2 22.73
- cereal_share: reduced form z_ss +0.116 (0.122); 2SLS +0.455 (0.164, p=0.006, 95% CI +0.13 to +0.78); OLS +0.040 (0.017); n=8,073
  quadratic: irr +2.215 (3.038), irr2 -1.023 (1.774), p(irr2)=0.565, peak 1.083 (0.413); first-stage F irr 13.86, irr2 22.73

## source composition (shares of irrigated area) x leave-state-out growth, baseline x year controls
First stage (n=8,073, districts 463, clusters 463): z_comp -0.0108 (0.0497); cluster-robust F = 0.05
- D1: reduced form z_comp +0.731 (0.523); 2SLS -67.767 (319.256, p=0.832, 95% CI -693.51 to +557.98); OLS -0.190 (0.245); n=8,073
- D0_05: reduced form z_comp +0.403 (0.695); 2SLS -37.343 (179.362, p=0.835, 95% CI -388.89 to +314.21); OLS -0.911 (0.339); n=8,073
- D0: reduced form z_comp -1.519 (1.087); 2SLS +140.876 (655.074, p=0.830, 95% CI -1143.07 to +1424.82); OLS -0.946 (0.546); n=8,073
- D2: reduced form z_comp +0.765 (0.462); 2SLS -70.961 (333.436, p=0.832, 95% CI -724.49 to +582.57); OLS +0.096 (0.212); n=8,073
- even: reduced form z_comp +0.051 (0.023); 2SLS -4.736 (22.035, p=0.830, 95% CI -47.93 to +38.45); OLS +0.011 (0.011); n=8,073
- cereal_share: reduced form z_comp -0.012 (0.032); 2SLS +1.084 (5.746, p=0.850, 95% CI -10.18 to +12.35); OLS +0.040 (0.017); n=8,073

## CGWB depth x trend (2013-15 depth; NOT a valid exclusion restriction)
First stage (n=8,100, districts 473, clusters 472): z_dep_t -0.0072 (0.0042); cluster-robust F = 2.87
- D1: reduced form z_dep_t +0.077 (0.036); 2SLS -10.771 (8.151, p=0.187, 95% CI -26.75 to +5.21); OLS -0.078 (0.227); n=8,100
- D0_05: reduced form z_dep_t +0.199 (0.063); 2SLS -27.651 (18.788, p=0.142, 95% CI -64.47 to +9.17); OLS -0.709 (0.331); n=8,100
- D0: reduced form z_dep_t +0.119 (0.088); 2SLS -16.584 (15.950, p=0.299, 95% CI -47.85 to +14.68); OLS -0.504 (0.533); n=8,100
- D2: reduced form z_dep_t +0.048 (0.030); 2SLS -6.671 (5.802, p=0.251, 95% CI -18.04 to +4.70); OLS +0.176 (0.197); n=8,100
- even: reduced form z_dep_t +0.001 (0.002); 2SLS -0.116 (0.234, p=0.621, 95% CI -0.58 to +0.34); OLS +0.007 (0.010); n=8,100
- cereal_share: reduced form z_dep_t -0.006 (0.003); 2SLS +0.789 (0.703, p=0.262, 95% CI -0.59 to +2.17); OLS +0.018 (0.017); n=8,100

## CGWB depth x trend and its square
First stage (n=8,100, districts 473, clusters 472): z_dep_t -0.0017 (0.0045); z_dep_t2 -0.0020 (0.0007); cluster-robust F = 6.47
- D1: reduced form z_dep_t +0.078 (0.043); 2SLS -3.114 (2.209, p=0.159, 95% CI -7.44 to +1.22); OLS -0.078 (0.227); n=8,100
  quadratic: irr +32.194 (32.850), irr2 -28.217 (26.543), p(irr2)=0.288, peak 0.570 (0.091); first-stage F irr 6.47, irr2 4.70
- D0_05: reduced form z_dep_t +0.196 (0.067); 2SLS -8.427 (5.118, p=0.100, 95% CI -18.46 to +1.60); OLS -0.709 (0.331); n=8,100
  quadratic: irr +80.223 (72.726), irr2 -70.846 (59.306), p(irr2)=0.233, peak 0.566 (0.085); first-stage F irr 6.47, irr2 4.70
- D0: reduced form z_dep_t +0.197 (0.107); 2SLS +5.130 (4.972, p=0.303, 95% CI -4.61 to +14.88); OLS -0.504 (0.533); n=8,100
  quadratic: irr +105.265 (93.368), irr2 -80.024 (75.496), p(irr2)=0.290, peak 0.658 (0.109); first-stage F irr 6.47, irr2 4.70
- D2: reduced form z_dep_t +0.047 (0.034); 2SLS -2.016 (1.981, p=0.309, 95% CI -5.90 to +1.87); OLS +0.176 (0.197); n=8,100
  quadratic: irr +19.449 (22.624), irr2 -17.154 (18.261), p(irr2)=0.348, peak 0.567 (0.105); first-stage F irr 6.47, irr2 4.70
- even: reduced form z_dep_t -0.000 (0.002); 2SLS -0.193 (0.101, p=0.058, 95% CI -0.39 to +0.01); OLS +0.007 (0.010); n=8,100
  quadratic: irr -0.546 (1.202), irr2 +0.282 (0.950), p(irr2)=0.766, peak 0.967 (1.161); first-stage F irr 6.47, irr2 4.70
- cereal_share: reduced form z_dep_t -0.004 (0.004); 2SLS +0.395 (0.227, p=0.083, 95% CI -0.05 to +0.84); OLS +0.018 (0.017); n=8,100
  quadratic: irr -1.418 (2.535), irr2 +1.449 (2.012), p(irr2)=0.472, peak 0.489 (0.216); first-stage F irr 6.47, irr2 4.70

## CGWB depth x trend, baseline irrigation x year controls
First stage (n=7,032, districts 406, clusters 406): z_dep_t -0.0075 (0.0043); cluster-robust F = 3.02
- D1: reduced form z_dep_t +0.079 (0.036); 2SLS -10.446 (7.390, p=0.158, 95% CI -24.93 to +4.04); OLS -0.113 (0.259); n=7,032
- D0_05: reduced form z_dep_t +0.227 (0.061); 2SLS -30.219 (18.961, p=0.112, 95% CI -67.38 to +6.94); OLS -0.894 (0.362); n=7,032
- D0: reduced form z_dep_t +0.162 (0.088); 2SLS -21.575 (16.894, p=0.202, 95% CI -54.69 to +11.54); OLS -0.978 (0.603); n=7,032
- D2: reduced form z_dep_t +0.041 (0.031); 2SLS -5.460 (5.026, p=0.278, 95% CI -15.31 to +4.39); OLS +0.152 (0.232); n=7,032
- even: reduced form z_dep_t +0.001 (0.002); 2SLS -0.106 (0.219, p=0.627, 95% CI -0.54 to +0.32); OLS +0.009 (0.012); n=7,032
- cereal_share: reduced form z_dep_t -0.007 (0.003); 2SLS +0.867 (0.664, p=0.192, 95% CI -0.43 to +2.17); OLS +0.038 (0.019); n=7,032