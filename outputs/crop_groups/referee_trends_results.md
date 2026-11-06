# Referee check of Q1 (crop-group trends, AP/Telangana rice)

Corrected script: `scripts/99_crop_groups_trends_referee.py`. It is a copy of script 99 with two raw-data artefacts removed. `--w1end` moves the end window. Outputs are `referee_fix_*` (end window 2017-19) and `referee_fix_end18_*` (end window 2016-18). The original outputs are untouched.

## Two APY artefacts the original kept

| artefact | evidence | effect |
|---|---|---|
| Andhra Pradesh "Niger Seed", 2015-16 to 2019-20 | Absent before 2015. Then 0.86 to 1.34 Mha a year, mostly in Anantapur (0.39-0.62 Mha), Chittoor, Kurnool and the delta. Yields run from 0.23 to 15.8 t/ha. Anantapur GCA goes from about 1.05 to 1.2-1.5 Mha in the same years. | Adds about 1 Mha of phantom oilseed area to AP's end window, which inflates AP GCA and deflates the AP rice share |
| Gujarat 1998-99 "Moth", Whole Year | 2.90 Mha across Saurashtra (Rajkot 0.47, Junagadh 0.45, Jamnagar 0.44 Mha). Every other year is 0.01-0.04 Mha. | Inflates Gujarat and national pulses at the start window |

Both are dropped at unit-year level and shares are recomputed (146 rows, all matched to units). One other item looks odd but is kept: MP wheat at 7.7 and 10.2 Mha in 2018-19 and 2019-20. MP's own reported production for 2019-20 (37.2 Mt) matches APY, so it is a state-reported level. The 2016-18 end window shows how much it matters.

## Corrected national endpoint (467 units; pp, SE)

| group | area-weighted | district mean | area change, Mha | end 2016-18: area-weighted |
|---|---|---|---|---|
| rice | -2.00 (0.45) | -0.39 (0.42) | +0.75 (0.70) | -2.24 (0.40) |
| wheat | +1.35 (0.34) | +0.58 (0.21) | +4.71 (0.67) | +0.95 (0.29) |
| maize | +1.28 (0.33) | +0.75 (0.27) | +2.62 (0.55) | +1.25 (0.30) |
| millets | -5.58 (0.59) | -3.19 (0.34) | -7.05 (0.93) | -5.26 (0.52) |
| pulses | +2.86 (0.60) | +0.93 (0.38) | +6.59 (1.22) | +3.25 (0.61) |
| oilseeds | -0.72 (0.70) | -1.45 (0.40) | +1.07 (1.14) | -0.65 (0.65) |
| cotton | +1.73 (0.53) | +0.77 (0.27) | +3.70 (0.93) | +1.44 (0.52) |

Rice decomposition, within districts versus between them: the -2.00 is +0.34 within districts and -2.33 between them. GCA grew in districts that grow little rice, so the national rice share fell even though the typical district's rice share did not.

## Andhra Pradesh and Telangana, corrected

| | AP (13) | TS, end 2017-19 | TS, end 2016-18 |
|---|---|---|---|
| rice share, pp | -2.7 (2.0) | +7.8 (1.8) | +3.1 (1.5) |
| rice area, kha | -442 (143) | +882 (226) | +420 (146) |
| of which rabi | -97 | +613 | +398 (kharif +21) |
| GCA, kha | -709 (278) | +1,301 (260) | +794 (204) |
| gap vs rest of India, district share | -1.7, p = 0.46 | +6.8, p = 0.001 | +2.3, p = 0.17 |
| gap, log rice area | -0.09, p = 0.34 | +0.55, p < 0.001 | +0.35, p < 0.001 |
| FE slope, pp/decade | -0.66 (1.12) | +3.24 (0.80) | +1.70 (0.71); t(8) p = 0.044 |

AP's corrected oilseed share falls by 12.9 pp (-1.01 Mha), against the -1.8 pp reported. In Gujarat, the pulses share falls by 0.3 pp, against the -12.0 pp reported.

## Baseline weather
June-September rainfall departures in 1999 and 2000 were -34 and -31 percent in Rajasthan and -48 and -42 percent in Gujarat. In 2019 they were +31 percent in Rajasthan, +35 in Gujarat and +39 in Madhya Pradesh. All-India rainfall hides this. Rajasthan's pulses gain, and the cotton and millet changes in Gujarat, start from a drought baseline.
