# Referee checks on irrigation v2

## R0. Reproduction

- Q1 D1 (b) flags drop6: n 591 (G 29): b1 3.85 (p 0.007/0.022), b2 -6.46 (SE 1.40; 1.89, p 0.000/0.001), peak 0.30 [note says -6.46]
- Q1 D1 (a) all: n 598 (G 30): b1 3.80 (p 0.011/0.021), b2 -6.78 (SE 1.46; 1.80, p 0.000/0.000), peak 0.28 [note says -6.78]
- Q1 D0_05 (b): n 591 (G 29): b1 7.64 (p 0.003/0.027), b2 -12.81 (SE 2.47; 3.37, p 0.000/0.000), peak 0.30 [note says -12.81]
- Q2 spec D tank, D1: -9.28 (state SE 0.93) n 534 [note -9.28 (0.93)]
- Q2 E gw-majority D1: n 231 (G 17): b1 3.95 (p 0.176/0.334), b2 -5.68 (SE 2.33; 3.33, p 0.015/0.089), peak 0.35
- Q2 E canal-majority D1: n 136 (G 21): b1 0.97 (p 0.552/0.647), b2 -2.42 (SE 1.63; 1.76, p 0.139/0.169), peak 0.20
- Q3 D1 1991 full : n 539 (G 24): b1 -0.78 (p 0.508/0.646), b2 -2.58 (SE 1.09; 1.56, p 0.018/0.097), peak -0.15
- Q3 D1 1991 w1517 : n 532 (G 24): b1 -0.35 (p 0.772/0.829), b2 -3.17 (SE 1.11; 1.57, p 0.004/0.044), peak -0.05
- Q3 D1 1991 w1721 : n 532 (G 24): b1 0.10 (p 0.935/0.945), b2 -3.51 (SE 1.09; 1.35, p 0.001/0.009), peak 0.01
- Q4 dD0_05 to cs0913: -1.75 (state SE 0.64) n 454
- Q4 dD0_05 to w1517: -1.47 (state SE 0.86) n 447

## R6. Multiple testing across source-split tests
Source-split tests collected: 216 (Q2 32, Q2ay 8, Q3 48, Q4 128).
Raw p < 0.05: parent 36 , state 18. Expected under the global null: 10.8.
BH q < 0.05: parent 10, state 6; BH q < 0.10: parent 11, state 6.

Tests surviving BH q < 0.10 on either clustering:
- Q2 | D sh_tk | D1: p 0.0000/0.0000, q 0.000/0.000
- Q2 | D sh_tk | E: p 0.0000/0.0000, q 0.000/0.000
- Q2 | D sh_tk | D0_05: p 0.0000/0.0000, q 0.000/0.000
- Q2 | D sh_tk | D0: p 0.0064/0.0000, q 0.115/0.000
- Q4 | by source dgw_2001_2011 | d_D0_05_cs0913 | 2001_2011 gross area within 2/3 to 1.5x: p 0.0018/0.0004, q 0.039/0.015
- Q4 | canal - gw diff | d_D0_05_cs0913 | 2001_2011 gross area within 2/3 to 1.5x: p 0.0118/0.0013, q 0.149/0.046
- Q2 | D sh_ro | D1: p 0.0001/0.0057, q 0.004/0.177
- Q2 | D sh_ro | D0_05: p 0.0002/0.0288, q 0.005/0.478
- Q3 | sources I(gw91s ** 2) | D0 | w1721: p 0.0000/0.0451, q 0.002/0.585
- Q3 | sources I(gw91s ** 2) | D0 | full: p 0.0000/0.0565, q 0.002/0.585
- Q3 | sources I(gw91s ** 2) | D0 | w1517: p 0.0001/0.0595, q 0.003/0.585
- Q2 | D I(sh_gw ** 2) | D0: p 0.0014/0.0690, q 0.033/0.593
- Q3 | sources I(gw91s ** 2) | D0_05 | full: p 0.0034/0.1736, q 0.067/0.736
Within Q4 alone (128 tests): BH q<0.10 parent 0, state 2.
