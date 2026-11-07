
## R2. Q1 extras

Extension-only window: 2020-21 and 2021-22, flagged state-years dropped, six crops removed, district needs both years.
- D1: n 506 (G 25): b1 3.88 (p 0.016/0.019), b2 -6.71 (SE 1.57; 1.86, p 0.000/0.000), peak 0.29
- D0_05: n 506 (G 25): b1 8.05 (p 0.006/0.021), b2 -13.35 (SE 2.78; 3.34, p 0.000/0.000), peak 0.30
- D0: n 506 (G 25): b1 12.86 (p 0.000/0.000), b2 -13.49 (SE 3.29; 2.70, p 0.000/0.000), peak 0.48
- E: n 506 (G 25): b1 0.02 (p 0.868/0.863), b2 -0.15 (SE 0.09; 0.09, p 0.105/0.104), peak 0.06
- single year 2019-20, D1 six removed: n 591 (G 29): b1 3.27 (p 0.028/0.080), b2 -6.05 (SE 1.45; 2.12, p 0.000/0.004), peak 0.27
- single year 2020-21, D1 six removed: n 524 (G 27): b1 4.18 (p 0.010/0.013), b2 -6.86 (SE 1.60; 1.92, p 0.000/0.000), peak 0.30
- single year 2021-22, D1 six removed: n 506 (G 25): b1 3.89 (p 0.015/0.019), b2 -6.72 (SE 1.56; 1.83, p 0.000/0.000), peak 0.29

Leave-one-state-out on the main (b) D1 bend (flags dropped, six removed):
b2 range -7.27 to -4.61; max state p 0.023 (dropping UTTAR PRADESH); peak range 0.26 to 0.32
Most influential: UTTAR PRADESH -4.61; GUJARAT -7.27; MADHYA PRADESH -7.12; ASSAM -5.87
Share of districts above the D1 peak (0.30): 85%
Share of districts above the D0_05 peak (0.30): 85%

## R3. Q2: canal vs groundwater, tank

Spread of the source shares (share of sown area) in the spec D sample, n 534:
- sh_can: mean 0.148, sd 0.180, within-state sd 0.152, p90 0.380, max 0.956
- sh_gw: mean 0.233, sd 0.232, within-state sd 0.148, p90 0.557, max 0.982
- sh_tk: mean 0.027, sd 0.061, within-state sd 0.051, p90 0.060, max 0.679
- sh_ro: mean 0.061, sd 0.098, within-state sd 0.071, p90 0.152, max 0.791
- D1 effect of +1 within-state SD of sh_can at its mean: -0.272
- D1 effect of +1 within-state SD of sh_gw at its mean: -0.248
- D1 effect of +1 within-state SD of sh_tk: -0.476; of sh_ro: -0.223

Tank share by state (spec D sample): n with tank share > 0.05, state mean tank share
                 n  mean_tank  n_gt_05
state_name                            
TAMIL NADU      31      0.134       14
ANDHRA PRADESH  20      0.067        9
WEST BENGAL     18      0.060       11
JHARKHAND       24      0.059       13
TELANGANA        2      0.036        0
BIHAR           38      0.035        5
GOA              2      0.028        0
KARNATAKA       30      0.028        4

Leave-one-state-out, spec D tank coefficient:
- D1: range -10.58 to -8.78; max state p 0.002; biggest moves: TAMIL NADU -10.58; MADHYA PRADESH -8.78; KARNATAKA -8.87; UTTAR PRADESH -8.90
  drop Tamil Nadu: b -10.584, state SE 3.481, t -3.04, G 24, n 503: p normal 0.002, p t(G-1) 0.006, p wild-cluster (Webb, 9999 draws) 0.003
  drop Andhra Pradesh+Telangana: b -9.626, state SE 1.082, t -8.90, G 23, n 512: p normal 0.000, p t(G-1) 0.000, p wild-cluster (Webb, 9999 draws) 0.059
  drop Karnataka: b -8.873, state SE 0.838, t -10.59, G 24, n 504: p normal 0.000, p t(G-1) 0.000, p wild-cluster (Webb, 9999 draws) 0.005
  drop Tamil Nadu+Andhra Pradesh+Telangana+Karnataka: b -11.626, state SE 3.289, t -3.53, G 21, n 451: p normal 0.000, p t(G-1) 0.002, p wild-cluster (Webb, 9999 draws) 0.008
- D0_05: range -11.68 to -8.53; max state p 0.095; biggest moves: TAMIL NADU -8.53; ANDHRA PRADESH -11.68; MADHYA PRADESH -10.05; UTTAR PRADESH -10.18
  drop Tamil Nadu: b -8.535, state SE 5.118, t -1.67, G 24, n 503: p normal 0.095, p t(G-1) 0.109, p wild-cluster (Webb, 9999 draws) 0.117
  drop Andhra Pradesh+Telangana: b -11.673, state SE 1.212, t -9.63, G 23, n 512: p normal 0.000, p t(G-1) 0.000, p wild-cluster (Webb, 9999 draws) 0.036
  drop Karnataka: b -10.840, state SE 1.448, t -7.48, G 24, n 504: p normal 0.000, p t(G-1) 0.000, p wild-cluster (Webb, 9999 draws) 0.064
  drop Tamil Nadu+Andhra Pradesh+Telangana+Karnataka: b -14.995, state SE 4.880, t -3.07, G 21, n 451: p normal 0.002, p t(G-1) 0.006, p wild-cluster (Webb, 9999 draws) 0.024
- tank share alone + terrain + state FE, D1: b -8.215, state SE 0.932, t -8.81, G 25, n 534: p normal 0.000, p t(G-1) 0.000, p wild-cluster (Webb, 9999 draws) 0.046
- tank share + total irrigated quadratic, D1: b -7.080, state SE 0.936, t -7.56, G 25, n 534: p normal 0.000, p t(G-1) 0.000, p wild-cluster (Webb, 9999 draws) 0.095

Canal-majority vs groundwater-majority districts: support of the irrigated share
- gw-majority: n 231, states 17, irr share p10/p50/p90 0.19/0.51/0.98, within-state sd 0.144; share above 0.5 53%
- canal-majority: n 136, states 21, irr share p10/p50/p90 0.09/0.36/0.88, within-state sd 0.171; share above 0.5 40%
- neither: n 164, states 18, irr share p10/p50/p90 0.05/0.29/0.85, within-state sd 0.150; share above 0.5 35%
- D1: canal-majority minus rest, squared term: b 1.640, state SE 1.683, t 0.97, G 25, n 534: p normal 0.330, p t(G-1) 0.340, p wild-cluster (Webb, 9999 draws) 0.317; minimum detectable difference (80% power, 5%) about 4.7
- D0_05: canal-majority minus rest, squared term: b 1.854, state SE 3.649, t 0.51, G 25, n 534: p normal 0.611, p t(G-1) 0.616, p wild-cluster (Webb, 9999 draws) 0.621; minimum detectable difference (80% power, 5%) about 10.2
