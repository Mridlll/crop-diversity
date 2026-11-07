# Referee checks on the D0 analysis

Balanced panel 429 districts, 22 states, 429 parent clusters.

## 1. State decomposition of the +3.71 endpoint rise

within = average weight x change in the state's area-weighted D0; between = weight shift x average D0. Sum within +3.43, sum between +0.28 (between terms net to the mean-level effect of reweighting).

| state | n | weight 98 | weight 19 | aw D0 98 | aw D0 19 | 108 contribution | within | between |
|---|---|---|---|---|---|---|---|---|
| Madhya Pradesh | 38 | 0.108 | 0.161 | 14.0 | 27.6 | +2.92 | +1.83 | +1.09 |
| Karnataka | 20 | 0.057 | 0.057 | 25.9 | 37.8 | +0.68 | +0.68 | +0.00 |
| Andhra Pradesh | 13 | 0.048 | 0.047 | 17.1 | 30.6 | +0.60 | +0.64 | -0.04 |
| Odisha | 30 | 0.035 | 0.034 | 12.3 | 28.9 | +0.56 | +0.58 | -0.02 |
| Chhattisgarh | 6 | 0.018 | 0.009 | 13.3 | 33.7 | +0.08 | +0.28 | -0.20 |
| Uttar Pradesh | 69 | 0.155 | 0.147 | 24.0 | 25.7 | +0.06 | +0.25 | -0.18 |
| Assam | 23 | 0.023 | 0.019 | 25.6 | 30.5 | -0.01 | +0.10 | -0.11 |
| Nagaland | 7 | 0.001 | 0.002 | 3.0 | 32.2 | +0.05 | +0.05 | +0.01 |
| Telangana | 2 | 0.009 | 0.003 | 20.6 | 24.1 | -0.13 | +0.02 | -0.15 |
| Punjab | 17 | 0.046 | 0.038 | 10.3 | 8.9 | -0.14 | -0.06 | -0.08 |
| Maharashtra | 29 | 0.132 | 0.124 | 21.9 | 21.4 | -0.23 | -0.06 | -0.17 |
| Gujarat | 19 | 0.080 | 0.046 | 21.0 | 19.0 | -0.80 | -0.13 | -0.67 |
| Bihar | 37 | 0.049 | 0.045 | 26.7 | 21.4 | -0.36 | -0.25 | -0.11 |
| West Bengal | 18 | 0.055 | 0.052 | 25.6 | 20.8 | -0.33 | -0.26 | -0.07 |
| Rajasthan | 32 | 0.122 | 0.164 | 27.7 | 25.9 | +0.86 | -0.26 | +1.13 |

Madhya Pradesh area weight by year: 1997 0.075, 1998 0.108, 1999 0.121, 2019 0.161.

## 2. Endpoint with collapsed state-years removed from BOTH endpoints

108 sets collapsed district-years missing and then compares all 1998 districts present against all 2019 districts present, so Madhya Pradesh, Andhra Pradesh, Chhattisgarh and Nagaland enter 2019 but not 1998. Here both endpoints use the same districts.

| version | D0_raw | D0_0.1% | D0_0.5% | D0_1% | D1 | districts |
|---|---|---|---|---|---|---|
| as published (common by construction) | +3.71 (0.44) [+2.84, +4.57] | -0.07 (0.24) [-0.56, +0.39] | -0.47 (0.17) [-0.81, -0.14] | -0.34 (0.15) [-0.64, -0.05] | -0.04 (0.13) [-0.31, +0.22] | 429 |
| drop collapsed, common set | +1.31 (0.37) [+0.61, +2.07] | -0.79 (0.25) [-1.29, -0.30] | -0.74 (0.18) [-1.11, -0.39] | -0.46 (0.18) [-0.81, -0.10] | -0.12 (0.15) [-0.41, +0.17] | 365 |
| 108 construction (different sets) | +2.32 | | | | | 365 vs 429 |
| drop the four 1998-collapsed states in every year | +1.31 (0.36) [+0.60, +2.01] | -0.79 (0.25) [-1.28, -0.28] | -0.74 (0.19) [-1.11, -0.37] | -0.46 (0.17) [-0.81, -0.13] | -0.12 (0.15) [-0.41, +0.17] | 365 |
| also drop Odisha and Karnataka | -0.32 (0.24) [-0.78, +0.16] | -1.48 (0.24) [-1.96, -1.00] | -0.95 (0.20) [-1.34, -0.57] | -0.52 (0.19) [-0.90, -0.14] | -0.18 (0.15) [-0.49, +0.13] | 315 |

## 3. Clustering on state instead of district/parent (22 states)

| statistic | D0_raw | D0_0.1% | D0_0.5% | D0_1% | D1 |
|---|---|---|---|---|---|
| aw 1998->2019, parent | +3.71 (0.43) [+2.89, +4.56] | -0.07 (0.24) [-0.54, +0.40] | -0.47 (0.17) [-0.79, -0.16] | -0.34 (0.16) [-0.64, -0.03] | -0.04 (0.13) [-0.29, +0.23] |
| aw window, parent | -0.78 (0.22) [-1.22, -0.35] | -1.22 (0.17) [-1.56, -0.89] | -0.89 (0.12) [-1.12, -0.65] | -0.61 (0.10) [-0.81, -0.41] | -0.22 (0.08) [-0.38, -0.08] |
| mean window, parent | -0.65 (0.25) [-1.12, -0.15] | -0.87 (0.18) [-1.24, -0.53] | -0.65 (0.11) [-0.88, -0.44] | -0.45 (0.08) [-0.61, -0.28] | -0.14 (0.06) [-0.25, -0.03] |
| aw 1998->2019, state | +3.71 (1.91) [+0.43, +7.77] | -0.07 (0.83) [-1.49, +1.71] | -0.47 (0.52) [-1.40, +0.64] | -0.34 (0.41) [-1.07, +0.48] | -0.04 (0.31) [-0.59, +0.57] |
| aw window, state | -0.78 (1.01) [-2.95, +1.05] | -1.22 (0.67) [-2.73, -0.10] | -0.89 (0.40) [-1.72, -0.14] | -0.61 (0.26) [-1.15, -0.14] | -0.22 (0.14) [-0.51, +0.04] |
| mean window, state | -0.65 (1.25) [-3.24, +1.55] | -0.87 (0.79) [-2.53, +0.53] | -0.65 (0.44) [-1.55, +0.20] | -0.45 (0.26) [-0.95, +0.06] | -0.14 (0.14) [-0.42, +0.14] |

FE trends per decade, balanced 1998-2019, clustered on state:
  D0_raw: -0.113 (SE 0.769, p 0.883)
  D0_0.1%: -0.454 (SE 0.476, p 0.340)
  D0_0.5%: -0.381 (SE 0.267, p 0.154)
  D0_1%: -0.272 (SE 0.162, p 0.094)
  D1: -0.085 (SE 0.093, p 0.360)

## 5. Balanced panel over 1998-2019 only

500 districts (against 429). Added by state: Tamil Nadu 28; Himachal Pradesh 12; Karnataka 7; Madhya Pradesh 7; Chhattisgarh 6; Tripura 4; Puducherry 3; Uttarakhand 2; Chandigarh 1; The Dadra And Nagar Haveli And Daman And Diu 1.
States with no district in the 1997-2019 balanced panel but present 1998-2019: Chandigarh, Himachal Pradesh, Puducherry, Tamil Nadu, The Dadra And Nagar Haveli And Daman And Diu, Tripura.

| statistic | D0_raw | D0_0.1% | D0_0.5% | D0_1% | D1 |
|---|---|---|---|---|---|
| levels 1998 / 2019 (aw) | 21.24 / 25.35 | 14.41 / 14.36 | 10.39 / 9.91 | 8.42 / 8.04 | 5.58 / 5.48 |
| aw 1998->2019, parent | +4.11 (0.41) [+3.32, +4.91] | -0.04 (0.23) [-0.49, +0.42] | -0.48 (0.16) [-0.80, -0.19] | -0.38 (0.15) [-0.68, -0.09] | -0.10 (0.12) [-0.33, +0.14] |
| aw window, parent | -0.47 (0.20) [-0.88, -0.08] | -1.10 (0.16) [-1.40, -0.79] | -0.83 (0.11) [-1.07, -0.61] | -0.59 (0.09) [-0.78, -0.41] | -0.22 (0.07) [-0.37, -0.08] |
| aw window, state | -0.47 (0.95) [-2.38, +1.28] | -1.10 (0.62) [-2.40, -0.00] | -0.83 (0.39) [-1.68, -0.11] | -0.59 (0.25) [-1.10, -0.12] | -0.22 (0.14) [-0.50, +0.05] |

## 6. Share of year-on-year district changes explained by state-year means (R2)

  D0_raw: state-year R2 0.77; year-only R2 0.05
  D0_0.5%: state-year R2 0.48; year-only R2 0.04
  D1: state-year R2 0.45; year-only R2 0.04

## 4. Raw-row checks

Rajasthan area (Mha), Moth / Guar Seed / Other Kharif Pulses / Other Oilseeds:
  1998: Guar Seed 0.00, Moth 0.88, Other Kharif Pulses 0.11, Other Oilseeds 0.00
  1999: Guar Seed 2.65, Moth 0.77, Other Kharif Pulses 0.11, Other Oilseeds 0.00
  2000: Guar Seed 3.06, Moth 0.96, Other Kharif Pulses 0.11, Other Oilseeds 0.00
  2001: Guar Seed 2.41, Moth 1.30, Other Kharif Pulses 0.14, Other Oilseeds 0.00
  2010: Guar Seed 2.98, Moth 1.59, Other Kharif Pulses 0.12, Other Oilseeds 1.20
  2019: Guar Seed 2.84, Moth 0.95, Other Kharif Pulses 0.05, Other Oilseeds 0.77
National Moth+Guar (Mha): 1997 0.98, 1998 3.82, 1999 3.78, 2000 4.34, 2001 4.02, 2002 1.26
Other Oilseeds national (kha, states): 1998 7/1.0, 2000 1/4.0, 2005 2/4.0, 2010 1301/4.0, 2015 34/7.0, 2016 269/9.0, 2017 124/11.0, 2018 156/11.0, 2019 927/12.0
Other Oilseeds 2019 by state (kha): Rajasthan 768, Andhra Pradesh 98, Mizoram 27, Maharashtra 22, Haryana 5
Odisha area by season (Mha):
  1997: Autumn 0.29, Kharif 0.00, Rabi 0.01, Summer 0.15, Whole Year 0.01, Winter 0.46
  1998: Autumn 1.01, Kharif 0.00, Rabi 0.00, Summer 0.42, Whole Year 0.00, Winter 3.80
  1999: Autumn 0.80, Kharif 1.55, Rabi 1.63, Summer 0.38, Whole Year 0.07, Winter 3.45
  2000: Autumn 0.85, Kharif 1.05, Rabi 0.11, Summer 0.21, Whole Year 0.22, Winter 3.37
Madhya Pradesh 1998-99 names (17): Arhar/Tur; Bajra; Barley; Cotton(Lint); Gram; Groundnut; Jowar; Linseed; Maize; Rapeseed &Mustard; Rice; Sesamum; Small Millets; Soyabean; Sugarcane; Sunflower; Wheat.
Madhya Pradesh: in 1999 not 1998: Castor Seed; Coriander; Dry Chillies; Garlic; Ginger; Niger Seed; Onion; Other  Rabi Pulses; Other Kharif Pulses; Potato; Ragi; Safflower; Sannhamp; Sweet Potato; Turmeric; Urad.
  MP Moong(Green Gram): in 1998 list False
  MP Urad: in 1998 list False
  MP Masoor: in 1998 list False
  MP Peas & Beans (Pulses): in 1998 list False
  MP Potato: in 1998 list False

## 7. Actual top 10 endpoint gainers

  Coriander: +0.364, median share 0.034%, 2019 presences <0.5% 93%, area 98/19 218/315 kha
  Guar Seed: +0.353, median share 0.030%, 2019 presences <0.5% 69%, area 98/19 nan/3,099 kha
  Garlic: +0.350, median share 0.034%, 2019 presences <0.5% 88%, area 98/19 44/277 kha
  Other Oilseeds: +0.348, median share 0.158%, 2019 presences <0.5% 67%, area 98/19 7/906 kha
  Sweet Potato: +0.314, median share 0.028%, 2019 presences <0.5% 90%, area 98/19 27/74 kha
  Peas & Beans (Pulses): +0.263, median share 0.287%, 2019 presences <0.5% 63%, area 98/19 516/524 kha
  Onion: +0.234, median share 0.142%, 2019 presences <0.5% 78%, area 98/19 186/500 kha
  Moong(Green Gram): +0.214, median share 0.318%, 2019 presences <0.5% 59%, area 98/19 2,283/4,111 kha
  Masoor: +0.214, median share 0.442%, 2019 presences <0.5% 54%, area 98/19 661/1,115 kha
  Ginger: +0.211, median share 0.151%, 2019 presences <0.5% 76%, area 98/19 37/97 kha
Write-up list sum +2.72; actual top-10 sum +2.86.
Pulses + potato with median 2019 share >0.25%: +1.07 of the rise.
