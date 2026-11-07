# Peer effects, reviewer round (7 Nov 2026): results

Scripts 138, 139 and 140 produce the tables in this folder. Script 141 merges them into `docs/data/site_neighbours.json` under the keys `leadlag`, `common_border` and `within_cross`; it must run again after any rerun of script 131. The standard errors in brackets are clustered on the 2011 parent unit unless the text says otherwise.

## 1. Same-state against other-state neighbours in take-up (script 140)

All hazard specifications come from script 128: linear probability, state-by-year FE, baseline share and five geography controls. Script 140 reproduces script 128 exactly (max |diff| 1e-16 for nbr + ring and for the split).

| Episode | Same-state nbrs | Other-state nbrs | Same minus other (unit p; state p) | Edge within minus edge across, 4 sets (unit p; state p) | Events in border units |
|---|---|---|---|---|---|
| Chickpea surge | 0.056 (0.030) | -0.005 (0.019) | 0.060 (0.034); p 0.074; 0.265 | 0.080 (0.070); 0.25; 0.43 | 17 (10 in 4-set) |
| Soybean entry | 0.061 (0.035) | -0.004 (0.009) | 0.065 (0.039); 0.096; 0.001 | 0.094 (0.047); 0.047; 0.048 | 10 (7) |
| Soybean surge | 0.037 (0.032) | 0.029 (0.022) | 0.008; 0.79; 0.78 | -0.035; 0.47; 0.49 | 11 (8) |
| Cotton surge | 0.080 (0.071) | 0.022 (0.026) | 0.058; 0.47; 0.56 | -0.105; 0.40; 0.65 | 20 (10) |
| Chickpea entry | 0.074 (0.031) | -0.001 (0.012) | 0.075 (0.036); 0.035; 0.107 | 0.012; 0.86; 0.84 | 27 (15) |

- In full-sample same-state comparisons (S4, same-state nbrs against the same-state ring), the chickpea surge edge is 0.097 (0.041), p 0.019 (state p 0.13), and the soybean entry edge is 0.064 (0.026), p 0.015 (state p 0.005).
- Robustness: S6 counts Telangana inside Andhra Pradesh before 2014, both for "same state" and for the FE. The soybean entry edge-within-minus-across is then 0.105 (0.049), p 0.031. S5 drops AP and TS: 0.070 (0.043), p 0.10, on 5 events.
- NFSM pulses district lists. From 2010-11 the pulses component covers whole states, so coverage varies within states only in 2007-09 (15 states) and in one state afterwards. Holding own, neighbour and ring coverage fixed leaves the chickpea surge nbr-minus-ring edge at 0.083 (0.040). The coverage terms are all near zero. 34 of 47 chickpea surge events fall in a covered unit-year, against 51 percent of at-risk unit-years.
- Event timing. Among adjacent same-state pairs where both units surge into chickpea, 23 percent surge in the same year (52 pairs). For adjacent other-state pairs the figure is 0 percent (9 pairs).
- Verdict. Every point estimate says the chickpea and soybean take-up edge comes from same-state neighbours, and other-state neighbours sit at zero. The within-versus-across difference is statistically significant only for soybean entry, at p about 0.05 on 7 events. Soybean surges lean the other way. So the data support "the edge stops at the state line" as a description, but not as a precise test. They also do not tie the edge to the NFSM district lists. Wording used on the page: the pattern "points to state programmes more than to farmers copying farmers", with the significance caveat in the body text.

## 2. Leads and lags, t-3 to t+3 (script 138)

The model regresses dS_it on the neighbours' and same-state ring's dS at t-3..t+3, estimated jointly with rain, unit FE and state-year FE. The page figure uses spec `no_own`. Spec `main` also holds own dS at t-1 and t+1, and spec `no_k0` drops the same-year terms. Script 138 reproduces script 99's pulses P2 lag (0.0917).

Neighbour coefficients in `no_own` (k = +1 means the neighbours' change one year earlier):

| Crop | same year | +1 | -1 | sum lags 1-3 minus sum leads 1-3 (p) |
|---|---|---|---|---|
| Chickpea | 0.88 | 0.005 | 0.009 | 0.10 (0.13), 0.44 |
| Soybean | 0.75 | 0.064 | 0.059 | -0.14 (0.17), 0.43 |
| Cotton | 0.79 | 0.064 | 0.058 | -0.11 (0.20), 0.57 |
| Pulses | 0.57 | 0.075 | 0.017 | 0.25 (0.12), 0.033 |

The other crops are maize at -0.32 (p 0.076), millets at 0.02 (0.84) and oilseeds at -0.11 (0.30). Adversarial note: pulses show lags above leads in all three specs (p 0.036 main, 0.002 no_k0). The asymmetry comes from t-3 (+0.145) and t+2 (-0.122), not from t-1 against t+1 (0.058, p 0.23). Across the seven outcomes the BH q for pulses is 0.23. Chickpea, the pulse that moved most, shows none. The `main` and `no_k0` specs inflate the ±1 terms mechanically, because they condition on own t±1 or drop the same-year link. Only `no_own` is plotted.

## 3. Border gap on crops every state records (script 139)

The crops are rice, wheat, maize, gram, tur, groundnut, rapeseed and mustard, and cotton. The reporting table is `139_reporting_by_state.csv`. Where one of the eight is missing from a state's record, the state across the line grows it on under 1 percent of GCA, except for Goa against Karnataka and Maharashtra. Window shares reproduce `pe_unit_period` (rice max |diff| 2e-7). The eight crops hold 56 percent of GCA in 1998-2002 and 70 percent in 2015-19.

Adjacent pairs, 2015-19, n 1,485, with the wild state bootstrap at 999 draws:

| Likeness | M1 | M2 | M3 | km equivalent (interp [95%]; linear; regression) |
|---|---|---|---|---|
| Nine groups (page) | -7.23 (1.57; 2.46), wild p 0.002 | -3.42 (1.42; 2.24), 0.045 | -4.07 (1.20; 1.81), 0.003 | 109 [37, 176]; 111; 101 |
| Eight common crops, renormalised | -6.12 (1.77; 2.33), 0.003 | -3.22 (1.73; 2.09), 0.037 | -4.35 (1.34; 1.92), 0.011 | 118 [31, 188]; 119; 73 |
| Eight crops, overlap in GCA shares | -5.20 (2.13; 4.00), 0.031 | -2.01 (2.08; 3.44), 0.37 | -3.39 (1.24; 1.39), 0.010 | 147 [55, 244]; 147; 45 |

- Dropping Goa gives M1 -6.14, M2 -3.38 and M3 -4.47 on the eight crops.
- Adjacent likeness on the eight crops is 79.3 within a state and 70.9 across a line.
- Adversarial note: the GCA-overlap version loses M2. It also carries how much land the eight crops hold at all, which itself differs at borders. The renormalised version is the clean one.
- Verdict: the border gap is not a crop-list artefact.

## 4. Citations (verified by web search, 7 Nov 2026)
- Dube, A., T. W. Lester and M. Reich (2010), "Minimum Wage Effects Across State Borders: Estimates Using Contiguous Counties," Review of Economics and Statistics 92(4): 945-964.
- Stone, G. D. (2007), "Agricultural Deskilling and the Spread of Genetically Modified Cotton in Warangal," Current Anthropology 48(1): 67-103.
- Maertens, A. (2017), "Who Cares What Others Think (or Do)? Social Learning and Social Pressures in Cotton Farming in India," American Journal of Agricultural Economics 99(4): 988-1007.
- Foster, A. D. and M. R. Rosenzweig (1995), "Learning by Doing and Learning from Others: Human Capital and Technical Change in Agriculture," Journal of Political Economy 103(6): 1176-1209.
