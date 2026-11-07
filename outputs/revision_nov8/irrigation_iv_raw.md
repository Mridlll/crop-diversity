Source cells blank but total reported are read as zero.
# IV feasibility: raw

Balanced source panel: 421 districts reporting NIA in >= 20 of 22 years (others interpolated out of the shift).
National NIA by source, 2017-19 over 1998-2000 (balanced districts): canal 1.01, tank 0.80, tubewell 1.51, otherwell 0.96, other 1.32
Districts with SHRUG terrain: 722

Estimation years 2001-02 to 2019-20 (baseline years excluded).

## shift-share on source
First stage for irrigation (n=8,073, districts 463, clusters 463): z_ss +0.4623 (0.1728); z_ss2 -0.2709 (0.0595); cluster-robust F = 24.64
- D1 linear: 2SLS +2.131 (1.025, p=0.038) vs OLS -0.140 (0.239); first-stage F 24.64
  D1 quadratic: irr +4.160 (5.271), irr2 -1.629 (4.175), peak 1.277 (1.710); first-stage F irr 24.64, irr2 23.47
- D0_05 linear: 2SLS +3.230 (2.323, p=0.165) vs OLS -0.759 (0.333); first-stage F 24.64
  D0_05 quadratic: irr +14.803 (7.666), irr2 -9.292 (5.852), peak 0.797 (0.160); first-stage F irr 24.64, irr2 23.47
- D0 linear: 2SLS +3.186 (3.883, p=0.412) vs OLS -0.737 (0.548); first-stage F 24.64
  D0 quadratic: irr +28.401 (12.481), irr2 -20.245 (9.593), peak 0.701 (0.106); first-stage F irr 24.64, irr2 23.47
- even linear: 2SLS +0.071 (0.052, p=0.166) vs OLS +0.011 (0.011); first-stage F 24.64
  even quadratic: irr -0.150 (0.251), irr2 +0.178 (0.195), peak 0.422 (0.272); first-stage F irr 24.64, irr2 23.47

## terrain x trend
First stage for irrigation (n=9,736, districts 575, clusters 574): z_tri_t -0.0289 (0.0169); z_elev_t +0.0165 (0.0151); z_tri_t2 +0.0024 (0.0018); z_elev_t2 -0.0014 (0.0009); cluster-robust F = 1.57
- D1 linear: 2SLS -3.792 (5.551, p=0.495) vs OLS -0.150 (0.204); first-stage F 1.57
  D1 quadratic: irr +13.135 (14.454), irr2 -10.685 (9.776), peak 0.615 (0.297); first-stage F irr 1.57, irr2 2.80
- D0_05 linear: 2SLS -8.184 (8.888, p=0.358) vs OLS -0.626 (0.296); first-stage F 1.57
  D0_05 quadratic: irr +31.954 (20.939), irr2 -25.338 (15.492), peak 0.631 (0.199); first-stage F irr 1.57, irr2 2.80
- D0 linear: 2SLS -14.296 (13.443, p=0.288) vs OLS -0.495 (0.463); first-stage F 1.57
  D0 quadratic: irr +92.283 (49.879), irr2 -67.279 (32.751), peak 0.686 (0.184); first-stage F irr 1.57, irr2 2.80
- even linear: 2SLS +0.182 (0.322, p=0.572) vs OLS +0.004 (0.010); first-stage F 1.57
  even quadratic: irr -0.193 (0.881), irr2 +0.237 (0.520), peak 0.408 (1.061); first-stage F irr 1.57, irr2 2.80

## shift-share, just identified (z_ss only), and the reduced form
First stage: z_ss -0.2126 (0.0562), F = 14.30, n=8,073
Corr of baseline tubewell share with the district's z_ss change 2001->2019: 0.922
- D1: reduced form -0.632 (0.499); 2SLS +2.972 (2.439, p=0.224), n=8,073
- D0_05: reduced form -1.706 (0.825); 2SLS +8.025 (4.386, p=0.068), n=8,073
- D0: reduced form -2.898 (1.198); 2SLS +13.632 (6.758, p=0.044), n=8,073
- D2: reduced form -0.172 (0.404); 2SLS +0.808 (1.906, p=0.672), n=8,073
- even: reduced form +0.004 (0.023); 2SLS -0.020 (0.110, p=0.853), n=8,073
- cereal_share: reduced form +0.046 (0.035); 2SLS -0.218 (0.175, p=0.214), n=8,073