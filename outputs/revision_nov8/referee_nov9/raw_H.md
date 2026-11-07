# H. Census 2011 weekly haat reach: referee log

## H0. The variable
shrug-vd11 labels: pc11_vd_wkl_haat = 'Weekly Haat'; pc11_vd_mrkt = 'Mandis/Regular Market'; pc11_vd_area = 'Total Geographical Area (in Hectares)'.
Values of pc11_vd_wkl_haat over 588,973 shrids: 0.0: 495,234, 1.0: 93,738, nan: 1.
Distinct values of pc11_vd_mrkt: [0.0, 1.0].
Rural key columns: ['pc11_state_id', 'pc11_district_id', 'pc11_subdistrict_id', 'pc11_village_id', 'pc11_land_area', 'pc11_pca_tot_p']. Census villages per rural shrid: 1 in 99.0% of shrids; more than one in 5,720 shrids (1.0%).
Haat share among multi-village shrids 18.6% against 15.9% overall (values stay 0/1, so a shrid counts as a haat shrid if any of its villages has one).
Rebuilt census haat reach against script 136 wk11_all: max abs diff 1.11e-16; mandi/regular 1.11e-16; missing over the 606: 0.
Means over the 606: census haat 0.240 (SD 0.197); haat-only 0.109; haat and mandi/regular 0.131; mandi/regular only 0.095; village share 0.158; MA haat 0.190.
Correlations: census haat with MA haat 0.43; census haat-only with MA haat 0.52; census haat-and-mandi with MA haat 0.17; census haat with log density 0.21, village size 0.19, irrigation 0.04, ST share -0.11, census mandi/regular 0.42.
2011 village-directory coverage of rural population: median 1.000; districts below 0.95: 0; below 0.90: 0.

## H1. Reproduce, all outcomes (census haat reach, all site controls, state FE)
| block | spec | outcome | n | b | SE parent | p parent | p state t(G-1) | p wild |
|---|---|---|---|---|---|---|---|---|
| H1 | base | D0 | 606 | +3.258 | 0.940 | 0.0005 | 0.0237 | 0.0226 |
| H1 | base | D0_01 | 606 | +3.081 | 0.908 | 0.0007 | 0.0220 | 0.0148 |
| H1 | base | D0_05 | 606 | +2.108 | 0.708 | 0.0029 | 0.0536 | 0.0314 |
| H1 | base | D0_1 | 606 | +2.073 | 0.556 | 0.0002 | 0.0299 | 0.0193 |
| H1 | base | D1 | 606 | +1.857 | 0.493 | 0.0002 | 0.0443 | 0.0270 |
| H1 | base | D2 | 606 | +1.360 | 0.365 | 0.0002 | 0.0401 | 0.0222 |
| H1 | base | E | 606 | +0.070 | 0.028 | 0.0120 | 0.1404 | 0.1249 |
Check: D1 is the site's D1_exp_shannon (max diff 5.3e-15).

## H8. Multiplicity over the seven outcomes
- D0: wild p 0.0226, BH q 0.0366, Holm 0.1158; parent p 0.0005, BH q 0.0009
- D0_01: wild p 0.0148, BH q 0.0366, Holm 0.1036; parent p 0.0007, BH q 0.0010
- D0_05: wild p 0.0314, BH q 0.0366, Holm 0.1158; parent p 0.0029, BH q 0.0034
- D0_1: wild p 0.0193, BH q 0.0366, Holm 0.1158; parent p 0.0002, BH q 0.0005
- D1: wild p 0.0270, BH q 0.0366, Holm 0.1158; parent p 0.0002, BH q 0.0005
- D2: wild p 0.0222, BH q 0.0366, Holm 0.1158; parent p 0.0002, BH q 0.0005
- E: wild p 0.1249, BH q 0.1249, Holm 0.1249; parent p 0.0120, BH q 0.0120

## H2. Leave one state out
D0_05: b from +1.379 (without MAHARASHTRA, p 0.0464, state p 0.0821) to +2.399 (without BIHAR); max parent p 0.0464 (without MAHARASHTRA); max state p 0.1003 (without KARNATAKA); states where state p > 0.10: KARNATAKA.
D1: b from +1.381 (without KARNATAKA, p 0.0019, state p 0.0853) to +2.188 (without ANDHRA PRADESH); max parent p 0.0023 (without MAHARASHTRA); max state p 0.0853 (without KARNATAKA); states where state p > 0.10: none.

## H3-H6. Controls, coverage, drop6, definitions
| block | spec | outcome | n | b | SE parent | p parent | p state t(G-1) | p wild |
|---|---|---|---|---|---|---|---|---|
| H3 | + census mandi/regular | D0 | 606 | +4.575 | 1.060 | 0.0000 | 0.0034 | 0.0033 |
| H3 | + log rural density | D0 | 573 | +3.238 | 0.991 | 0.0011 | 0.0314 | 0.0362 |
| H3 | + village size (log pop, share 5k+) | D0 | 606 | +3.297 | 0.950 | 0.0005 | 0.0143 | 0.0146 |
| H3 | + MA mandi + regular market | D0 | 606 | +3.206 | 0.936 | 0.0006 | 0.0238 | 0.0232 |
| H3 | + MA haat | D0 | 606 | +1.803 | 0.901 | 0.0454 | 0.1210 | 0.1197 |
| H3 | + log cropped area | D0 | 606 | +2.190 | 0.839 | 0.0090 | 0.0279 | 0.0308 |
| H3 | + all of the above | D0 | 573 | +3.509 | 0.955 | 0.0002 | 0.0078 | 0.0040 |
| H3 | MA haat, + log cropped area (term is census haat; MA haat as control) | D0 | 606 | +1.392 | 0.855 | 0.1036 | 0.1171 | 0.1192 |
| H4 | VD coverage >= 0.95 | D0 | 606 | +3.258 | 0.940 | 0.0005 | 0.0237 | 0.0234 |
| H4 | VD coverage >= 0.90 | D0 | 606 | +3.258 | 0.940 | 0.0005 | 0.0237 | 0.0260 |
| H5 | drop6 | D0_d6 | 606 | +2.864 | 0.819 | 0.0005 | 0.0242 | 0.0238 |
| H6 | village share (unweighted) | D0 | 606 | +2.369 | 0.988 | 0.0165 | 0.0982 | 0.0937 |
| H6 | haat-only villages (no mandi/regular) | D0 | 606 | +5.574 | 1.336 | 0.0000 | 0.0070 | 0.0046 |
| H6 | haat-and-mandi villages | D0 | 606 | +1.796 | 1.211 | 0.1380 | 0.2983 | 0.4095 |
| H6 | joint: haat-only term | D0 | 606 | +5.289 | 1.344 | 0.0001 | 0.0094 | 0.0069 |
| H6 | joint: haat-and-mandi term | D0 | 606 | +1.618 | 1.205 | 0.1794 | 0.3631 | 0.4740 |
| H6 | haat-only, all H3 controls | D0 | 573 | +3.387 | 1.194 | 0.0045 | 0.0357 | 0.0504 |
| H3x | MA haat term, + log cropped area | D0 | 606 | +3.762 | 1.012 | 0.0002 | 0.0190 | 0.0499 |
| H3x | MA haat term, + log cropped area + census haat | D0 | 606 | +3.162 | 1.049 | 0.0026 | 0.0525 | 0.0784 |
| H3 | + census mandi/regular | D0_05 | 606 | +2.627 | 0.775 | 0.0007 | 0.0396 | 0.0226 |
| H3 | + log rural density | D0_05 | 573 | +2.019 | 0.747 | 0.0069 | 0.0804 | 0.0558 |
| H3 | + village size (log pop, share 5k+) | D0_05 | 606 | +2.003 | 0.723 | 0.0056 | 0.0360 | 0.0173 |
| H3 | + MA mandi + regular market | D0_05 | 606 | +2.111 | 0.713 | 0.0031 | 0.0530 | 0.0291 |
| H3 | + MA haat | D0_05 | 606 | +2.404 | 0.767 | 0.0017 | 0.0407 | 0.0111 |
| H3 | + log cropped area | D0_05 | 606 | +1.942 | 0.704 | 0.0058 | 0.0434 | 0.0288 |
| H3 | + all of the above | D0_05 | 573 | +3.244 | 0.901 | 0.0003 | 0.0235 | 0.0045 |
| H3 | MA haat, + log cropped area (term is census haat; MA haat as control) | D0_05 | 606 | +2.329 | 0.764 | 0.0023 | 0.0344 | 0.0100 |
| H4 | VD coverage >= 0.95 | D0_05 | 606 | +2.108 | 0.708 | 0.0029 | 0.0536 | 0.0270 |
| H4 | VD coverage >= 0.90 | D0_05 | 606 | +2.108 | 0.708 | 0.0029 | 0.0536 | 0.0319 |
| H5 | drop6 | D0_05_d6 | 606 | +2.109 | 0.687 | 0.0021 | 0.0568 | 0.0386 |
| H6 | village share (unweighted) | D0_05 | 606 | +1.334 | 0.740 | 0.0713 | 0.1535 | 0.1141 |
| H6 | haat-only villages (no mandi/regular) | D0_05 | 606 | +2.346 | 1.061 | 0.0270 | 0.1664 | 0.1646 |
| H6 | haat-and-mandi villages | D0_05 | 606 | +1.992 | 0.935 | 0.0332 | 0.1055 | 0.0846 |
| H6 | joint: haat-only term | D0_05 | 606 | +2.077 | 1.050 | 0.0479 | 0.1995 | 0.1999 |
| H6 | joint: haat-and-mandi term | D0_05 | 606 | +1.854 | 0.939 | 0.0483 | 0.1160 | 0.0872 |
| H6 | haat-only, all H3 controls | D0_05 | 573 | +2.964 | 1.225 | 0.0155 | 0.0913 | 0.0595 |
| H3x | MA haat term, + log cropped area | D0_05 | 606 | -0.532 | 0.994 | 0.5925 | 0.6732 | 0.6669 |
| H3x | MA haat term, + log cropped area + census haat | D0_05 | 606 | -1.535 | 1.045 | 0.1418 | 0.2673 | 0.2678 |
| H3 | + census mandi/regular | D1 | 606 | +2.133 | 0.614 | 0.0005 | 0.0531 | 0.0295 |
| H3 | + log rural density | D1 | 573 | +1.891 | 0.521 | 0.0003 | 0.0558 | 0.0411 |
| H3 | + village size (log pop, share 5k+) | D1 | 606 | +1.559 | 0.482 | 0.0012 | 0.0384 | 0.0195 |
| H3 | + MA mandi + regular market | D1 | 606 | +1.844 | 0.495 | 0.0002 | 0.0457 | 0.0301 |
| H3 | + MA haat | D1 | 606 | +1.937 | 0.530 | 0.0003 | 0.0395 | 0.0149 |
| H3 | + log cropped area | D1 | 606 | +1.714 | 0.485 | 0.0004 | 0.0381 | 0.0242 |
| H3 | + all of the above | D1 | 573 | +2.357 | 0.699 | 0.0007 | 0.0458 | 0.0285 |
| H3 | MA haat, + log cropped area (term is census haat; MA haat as control) | D1 | 606 | +1.876 | 0.524 | 0.0003 | 0.0354 | 0.0145 |
| H4 | VD coverage >= 0.95 | D1 | 606 | +1.857 | 0.493 | 0.0002 | 0.0443 | 0.0294 |
| H4 | VD coverage >= 0.90 | D1 | 606 | +1.857 | 0.493 | 0.0002 | 0.0443 | 0.0246 |
| H5 | drop6 | D1_d6 | 606 | +1.744 | 0.475 | 0.0002 | 0.0491 | 0.0321 |
| H6 | village share (unweighted) | D1 | 606 | +1.278 | 0.471 | 0.0067 | 0.0694 | 0.0322 |
| H6 | haat-only villages (no mandi/regular) | D1 | 606 | +1.891 | 0.908 | 0.0373 | 0.1882 | 0.1919 |
| H6 | haat-and-mandi villages | D1 | 606 | +1.870 | 0.581 | 0.0013 | 0.0372 | 0.0055 |
| H6 | joint: haat-only term | D1 | 606 | +1.716 | 0.885 | 0.0525 | 0.2083 | 0.2118 |
| H6 | joint: haat-and-mandi term | D1 | 606 | +1.781 | 0.584 | 0.0023 | 0.0391 | 0.0064 |
| H6 | haat-only, all H3 controls | D1 | 573 | +2.189 | 1.037 | 0.0348 | 0.1392 | 0.1533 |
| H3x | MA haat term, + log cropped area | D1 | 606 | +0.167 | 0.720 | 0.8163 | 0.8315 | 0.8334 |
| H3x | MA haat term, + log cropped area + census haat | D1 | 606 | -0.641 | 0.764 | 0.4011 | 0.4302 | 0.4329 |
Joint model D0_05: haat-only minus haat-and-mandi +0.223 (SE 1.354, p 0.8693).
Joint model D1: haat-only minus haat-and-mandi -0.065 (SE 1.049, p 0.9504).

## H7. Windows (div_windows.csv)
| block | spec | outcome | n | b | SE parent | p parent | p state t(G-1) | p wild |
|---|---|---|---|---|---|---|---|---|
| H7 | full all | wD0 | 606 | +3.258 | 0.940 | 0.0005 | 0.0237 | 0.0233 |
| H7 | full all | wD0_05 | 606 | +2.108 | 0.708 | 0.0029 | 0.0536 | 0.0305 |
| H7 | full all | wD1 | 606 | +1.857 | 0.493 | 0.0002 | 0.0443 | 0.0250 |
| H7 | e9802 all | wD0 | 547 | +3.156 | 1.160 | 0.0065 | 0.0813 | 0.0991 |
| H7 | e9802 all | wD0_05 | 547 | +2.730 | 0.803 | 0.0007 | 0.0212 | 0.0120 |
| H7 | e9802 all | wD1 | 547 | +1.937 | 0.550 | 0.0004 | 0.0351 | 0.0243 |
| H7 | cs0913 all | wD0 | 606 | +2.935 | 0.985 | 0.0029 | 0.0339 | 0.0339 |
| H7 | cs0913 all | wD0_05 | 606 | +1.572 | 0.738 | 0.0332 | 0.1282 | 0.1106 |
| H7 | cs0913 all | wD1 | 606 | +1.832 | 0.514 | 0.0004 | 0.0456 | 0.0249 |
| H7 | w1517 all | wD0 | 598 | +3.370 | 1.021 | 0.0010 | 0.0158 | 0.0160 |
| H7 | w1517 all | wD0_05 | 598 | +2.111 | 0.776 | 0.0065 | 0.1045 | 0.1037 |
| H7 | w1517 all | wD1 | 598 | +2.231 | 0.531 | 0.0000 | 0.0251 | 0.0067 |
| H7 | w1721 flags_drop6 | wD0 | 591 | +3.018 | 1.020 | 0.0031 | 0.0186 | 0.0165 |
| H7 | w1721 flags_drop6 | wD0_05 | 591 | +2.331 | 0.810 | 0.0040 | 0.1156 | 0.1206 |
| H7 | w1721 flags_drop6 | wD1 | 591 | +2.041 | 0.526 | 0.0001 | 0.0540 | 0.0396 |
