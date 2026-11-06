# Referee checks on the Q2 policy event studies

Script: `scripts/102_referee_policy_event_studies.py`. Log: `referee_policy_es_log.txt`. Estimates: `referee_policy_es_checks.csv`.
It reuses script 99's estimator and sample unchanged. None of script 99's outputs were overwritten. No APCNF, PMDS or RySS data are used.

## Defects found

1. **The treated-trend specification is not identified.** In script 99, `event_study(..., trend=True)` adds a treated linear trend next to a full set of event dummies and unit fixed effects. Those already span the trend: the demeaned trend regressed on the demeaned dummies gives an R squared of 1.0000, and the design has 13 columns but rank 12. `np.linalg.pinv` then returns an arbitrary minimum-norm split. The reported −0.0088 (0.0032) on pulse share and −0.057 (0.029) on log pulse area are therefore arbitrary numbers.
2. **Odisha's crop coverage breaks in 2019-20.** APY reports 13 or 14 crops for Odisha in every year from 2005-06 to 2018-19, and 39 in 2019-20. Over 2005-18 the millet group is ragi alone. Jowar, bajra and small millets only appear in 2019-20 (40.6 kha together), and cotton, gram, tur and 22 other crops appear in the same year. None of the defect flags catch this. It falls on the only post year for the 2019 cohort, so the OMM millet series and its denominator both jump for reasons that have nothing to do with the mission.

## NFSM-Pulses 2007: WEAKENED

- The simple DiD reproduces at +0.0103 (0.0038), and it does come from the early years. Within a 2002-09 window it falls to +0.0036 (0.0035), and log area falls to +0.009 (0.041).
- The trend-adjusted numbers have to be withdrawn (defect 1). The identified alternatives point in different directions depending on the window:

| Identified trend adjustment | Pulse share | Log pulse area |
|---|---|---|
| post + treated trend, 1997-2009 | −0.0050 (0.0044) | −0.201 (0.063) |
| post + treated trend, 2002-2009 | +0.0092 (0.0034) | −0.073 (0.059) |
| pre-period trend extrapolated | −0.0072 (0.0044) | −0.222 (0.070) |

- The answer depends on which pre-period years serve as the baseline. Mean post minus e = −1 gives −0.0030 (0.0027). Minus e = −5..−1 gives +0.0037 (0.0035), and minus e = −3..−2 gives +0.0113 (0.0039). 2006-07, the reference year, sits high.
- Leaving out one state at a time keeps the short-window contrast between +0.0013 and +0.0056, and none of those values is significant. Dropping the 22 units that contain both listed and unlisted child districts gives a DiD of +0.0123 (0.0041).
- **What can be stated:** there is no detectable rise in pulse share once the convergence before 1997-2002 is excluded. The short-window contrast is +0.004 (0.003), which rules out increases above about 0.010, roughly 6 percent of the 0.18 mean. A negative effect cannot be claimed.

## NFSM-Rice and NFSM-Wheat 2007: SURVIVES as "no increase"

- Rice share DiD −0.0015 (0.0048), and −0.0013 (0.0049) without Tamil Nadu. Wheat share DiD −0.0047 (0.0043).
- With a treated trend (identified version), rice share is −0.0127 (0.0043) and wheat share −0.0034 (0.0042). The positive log rice area DiD of +0.082 (0.034) is a pre-trend: the leads sit at −0.11 to −0.21 up to e = −6 and then flatten.
- **What can be stated:** neither list raised its own crop's share. "Unchanged" goes too far for rice, where the trend-adjusted and short-window estimates (−0.007) lean negative.

## Odisha Millets Mission: REFUTED as reported, WEAKENED in substance

- The headline ATT of −0.0011 (CI −0.008 to +0.006) mixes ragi-only years with a 2019-20 series that adds three millets (defect 2). The statement that "millet share rose in 2019-20 in untreated districts too" is that same artefact. Ragi itself did not rise.
- The controls are not usable. The 16 untreated districts hold 28 to 462 ha of millets in total, and in 2018-19 only 2 of them report any. Each ATT is therefore close to a before-and-after comparison within the treated districts.
- Holding the crop basket fixed (ragi, and ragi's share of the 13 crops reported in every year): ragi share ATT −0.0052 (0.0030), CI −0.012 to −0.001, reshuffle p 0.04. log(1 + ragi area) ATT +0.28 (0.40). Dropping 2019-20 from the original millet share gives −0.0054 (0.0025).
- Ragi area in the seven 2017 districts was 43.2 kha in 2016-17, 41.6 in 2017-18, 34.1 in 2018-19 and 33.8 in 2019-20. It had already been falling since 2010 (66 kha).
- **What can be stated:** district APY data show no rise in ragi area or share in the OMM districts through 2019-20. The decline continues the trend in place before entry. No valid within-Odisha comparison group exists, so the data identify no effect in either direction. The ln(1 + millet area) ATT of +0.64 should not be quoted.

## 2018-19 MSP: SURVIVES

- C1: no group is significant. The smallest raw p is 0.06 (other), Holm-adjusted 0.56, BH q 0.42. Dropping Odisha changes nothing; every per-SD estimate stays within 0.0003 of the original.
- Taking 2019-20 alone (the first season fully sown after the July 2018 announcement), pulses are +0.0050 (0.0022) and the rest are not significant. That is one hit in nine.
- With exposure times a linear trend, cotton is −0.0034 (0.0015) and millets −0.0046 (0.0025). Those are the two most shocked groups, and both move the wrong way. Sugarcane, the least shocked, is +0.0014 (0.0004). Nothing points toward a shift to the crops whose MSP rose most.
- Whole Year rows duplicating a seasonal row for the same crop amount to 0.11 Mha of 724 Mha in 2013-16, so they do not distort the exposure.
- C2: the permutation p of 0.60 is the relevant number, and it holds.

## Findings fit to state publicly

1. NFSM-Pulses 2007-08 districts were catching up on pulse share for about five years before entry, from 1997-98 to 2002-03. Measured against 2002-07, entry shows no detectable rise in pulse share (+0.004, SE 0.003) or pulse area (+0.009, SE 0.041).
2. NFSM-Rice and NFSM-Wheat 2007-08 did not raise their own crop's share: rice −0.0015 (SE 0.0048), wheat −0.0047 (SE 0.0043).
3. APY shows no rise in ragi in the Odisha Millets Mission districts through 2019-20. Ragi area in the first seven districts fell from 43 kha to 34 kha, continuing a decline that began before the mission, and the data allow no causal estimate.
4. After the 2018-19 MSP increase, districts more exposed to it did not shift area toward the crops whose MSP rose most (no group significant after multiple-testing correction; permutation p 0.60 at the group level).
