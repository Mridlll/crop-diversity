# D0 noise and the 1997-98 break

Sample: 725 crop-record districts; balanced panel 429 districts x 23 years (1997-98 to 2019-20); 429 cluster groups in the balanced panel.
Raw D0 and D1 reproduce script 76's corrected panel exactly (14,089 district-years checked).

## 1. National series, balanced panel

Area-weighted mean (the site's weighting), unweighted mean, and district median.

| measure | aw 1998 | aw 2019 | aw change | mean 1998 | mean 2019 | median 1998 | median 2019 |
|---|---|---|---|---|---|---|---|
| D0_raw | 21.23 | 24.94 | +3.71 | 19.84 | 23.79 | 21.0 | 25.0 |
| D0_0.1% | 14.28 | 14.21 | -0.07 | 13.58 | 14.67 | 13.0 | 14.0 |
| D0_0.5% | 10.28 | 9.81 | -0.47 | 9.59 | 9.80 | 9.0 | 9.0 |
| D0_1% | 8.29 | 7.95 | -0.34 | 7.70 | 7.67 | 7.0 | 7.0 |
| D1 | 5.46 | 5.42 | -0.04 | 4.85 | 4.88 | 4.6 | 4.5 |

### Changes with cluster-bootstrap SEs and 95% CIs (1000 draws, 429 clusters)

aw98_19: area-weighted, 1998-99 vs 2019-20 single years (site's 21.2 -> 24.9).
mean_win / median_win / aw_win: district change between 1998-2004 and 2013-2019 seven-year means; unweighted mean, median, and weighted by district mean area.

| measure | statistic | change | SE | 95% CI | share of raw D0 change |
|---|---|---|---|---|---|
| D0_raw | aw98_19 | +3.71 | 0.44 | [+2.87, +4.57] | 100% |
| D0_0.1% | aw98_19 | -0.07 | 0.25 | [-0.55, +0.40] | -2% |
| D0_0.5% | aw98_19 | -0.47 | 0.17 | [-0.80, -0.13] | -13% |
| D0_1% | aw98_19 | -0.34 | 0.16 | [-0.64, -0.01] | -9% |
| D1 | aw98_19 | -0.04 | 0.13 | [-0.28, +0.21] |  |
| D0_raw | aw_win | -0.80 | 0.22 | [-1.23, -0.41] | 100% |
| D0_0.1% | aw_win | -1.16 | 0.18 | [-1.49, -0.82] | 145% |
| D0_0.5% | aw_win | -0.84 | 0.12 | [-1.07, -0.59] | 105% |
| D0_1% | aw_win | -0.56 | 0.10 | [-0.76, -0.35] | 70% |
| D1 | aw_win | -0.19 | 0.08 | [-0.33, -0.03] |  |
| D0_raw | mean_win | -0.65 | 0.25 | [-1.13, -0.18] | 100% |
| D0_0.1% | mean_win | -0.87 | 0.19 | [-1.25, -0.52] | 134% |
| D0_0.5% | mean_win | -0.65 | 0.12 | [-0.88, -0.42] | 101% |
| D0_1% | mean_win | -0.45 | 0.08 | [-0.61, -0.28] | 69% |
| D1 | mean_win | -0.14 | 0.05 | [-0.25, -0.04] |  |
| D0_raw | median_win | -0.29 | 0.22 | [-0.71, +0.14] | 100% |
| D0_0.1% | median_win | -0.57 | 0.19 | [-1.00, -0.14] | 200% |
| D0_0.5% | median_win | -0.43 | 0.13 | [-0.71, -0.29] | 150% |
| D0_1% | median_win | -0.43 | 0.09 | [-0.57, -0.29] | 150% |
| D1 | median_win | -0.19 | 0.03 | [-0.26, -0.13] |  |

### Within-district linear trend, per decade (district FE, 1998-2019)

OLS of the measure on year with district fixed effects, clustered on 2011 parent. Weighted version uses district mean cropped area.

| measure | sample | weighting | slope per decade | SE | p | n district-years |
|---|---|---|---|---|---|---|
| D0_raw | balanced 429 | unweighted | -0.113 | 0.154 | 0.4627 | 9,438 |
| D0_raw | balanced 429 | area | -0.218 | 0.135 | 0.1058 | 9,438 |
| D0_0.1% | balanced 429 | unweighted | -0.454 | 0.115 | 0.0001 | 9,438 |
| D0_0.1% | balanced 429 | area | -0.689 | 0.108 | 0.0000 | 9,438 |
| D0_0.5% | balanced 429 | unweighted | -0.381 | 0.073 | 0.0000 | 9,438 |
| D0_0.5% | balanced 429 | area | -0.531 | 0.080 | 0.0000 | 9,438 |
| D0_1% | balanced 429 | unweighted | -0.272 | 0.055 | 0.0000 | 9,438 |
| D0_1% | balanced 429 | area | -0.368 | 0.068 | 0.0000 | 9,438 |
| D1 | balanced 429 | unweighted | -0.085 | 0.036 | 0.0174 | 9,438 |
| D1 | balanced 429 | area | -0.124 | 0.054 | 0.0211 | 9,438 |
| D0_raw | all 725 | unweighted | +0.537 | 0.135 | 0.0001 | 13,605 |
| D0_raw | all 725 | area | +0.008 | 0.123 | 0.9498 | 13,605 |
| D0_0.1% | all 725 | unweighted | +0.001 | 0.106 | 0.9949 | 13,605 |
| D0_0.1% | all 725 | area | -0.580 | 0.099 | 0.0000 | 13,605 |
| D0_0.5% | all 725 | unweighted | -0.088 | 0.070 | 0.2070 | 13,605 |
| D0_0.5% | all 725 | area | -0.478 | 0.073 | 0.0000 | 13,605 |
| D0_1% | all 725 | unweighted | -0.074 | 0.054 | 0.1691 | 13,605 |
| D0_1% | all 725 | area | -0.339 | 0.062 | 0.0000 | 13,605 |
| D1 | all 725 | unweighted | +0.008 | 0.035 | 0.8084 | 13,605 |
| D1 | all 725 | area | -0.130 | 0.050 | 0.0090 | 13,605 |

### Year-on-year steps in raw D0 (balanced panel, unweighted mean)

Total 1998->2019 change in unweighted mean raw D0: +3.95. Largest single-year steps: 1998-99 +2.28; 2018-19 +1.17; 2015-16 +0.90; 2014-15 +0.85.
Area-weighted: total +3.71; largest steps: 1998-99 +2.45; 2014-15 +1.47; 2005-06 +0.96; 2012-13 +0.95.

Full series (balanced, area-weighted):

| year | D0 raw | D0 0.1% | D0 0.5% | D0 1% | D1 | D0 raw mean | D0 0.5% mean |
|---|---|---|---|---|---|---|---|
| 1997 | 18.67 | 10.88 | 7.54 | 6.00 | 3.81 | 15.42 | 8.53 |
| 1998 | 21.23 | 14.28 | 10.28 | 8.29 | 5.46 | 19.84 | 9.59 |
| 1999 | 23.68 | 15.65 | 10.84 | 8.65 | 5.69 | 22.11 | 10.42 |
| 2000 | 24.16 | 15.73 | 10.91 | 8.73 | 5.60 | 22.84 | 10.48 |
| 2001 | 24.60 | 15.51 | 10.68 | 8.41 | 5.52 | 23.16 | 10.27 |
| 2002 | 24.66 | 15.60 | 10.67 | 8.53 | 5.53 | 23.36 | 10.32 |
| 2003 | 25.43 | 15.85 | 10.85 | 8.60 | 5.70 | 23.92 | 10.51 |
| 2004 | 24.32 | 14.87 | 10.25 | 8.18 | 5.43 | 22.80 | 9.76 |
| 2005 | 23.51 | 14.52 | 10.10 | 8.12 | 5.42 | 21.67 | 9.57 |
| 2006 | 24.46 | 14.95 | 10.38 | 8.42 | 5.70 | 22.21 | 9.75 |
| 2007 | 23.63 | 14.78 | 10.20 | 8.32 | 5.64 | 21.97 | 9.59 |
| 2008 | 24.14 | 14.59 | 10.05 | 7.98 | 5.38 | 22.33 | 9.61 |
| 2009 | 23.87 | 14.60 | 9.97 | 7.99 | 5.43 | 21.91 | 9.44 |
| 2010 | 23.70 | 14.85 | 10.36 | 8.35 | 5.86 | 21.79 | 9.69 |
| 2011 | 23.87 | 14.52 | 10.08 | 8.17 | 5.54 | 22.00 | 9.58 |
| 2012 | 22.73 | 13.64 | 9.37 | 7.58 | 5.16 | 21.69 | 9.26 |
| 2013 | 23.68 | 14.17 | 9.87 | 7.97 | 5.45 | 21.71 | 9.37 |
| 2014 | 21.37 | 13.86 | 9.75 | 7.89 | 5.38 | 20.24 | 9.34 |
| 2015 | 22.84 | 14.39 | 9.91 | 7.99 | 5.52 | 21.10 | 9.44 |
| 2016 | 23.44 | 14.36 | 10.11 | 8.25 | 5.69 | 22.00 | 9.78 |
| 2017 | 23.50 | 14.35 | 10.10 | 8.29 | 5.69 | 22.03 | 9.74 |
| 2018 | 24.41 | 13.75 | 9.44 | 7.72 | 5.29 | 22.62 | 9.30 |
| 2019 | 24.94 | 14.21 | 9.81 | 7.95 | 5.42 | 23.79 | 9.80 |

## 2. Where the raw D0 change sits, by the crop's share of district area

Mean number of crops per district-year in each share bin, early (1998-2004) and late (2013-2019) windows, balanced panel. Bins sum to raw D0.

| share bin | early mean | late mean | change (unweighted) | SE | early aw | late aw | change (aw) |
|---|---|---|---|---|---|---|---|
| <0.1% | 7.65 | 7.88 | +0.22 | 0.16 | 8.61 | 9.05 | +0.44 |
| 0.1-0.5% | 4.73 | 4.51 | -0.22 | 0.12 | 4.70 | 4.36 | -0.34 |
| 0.5-1% | 2.18 | 1.98 | -0.20 | 0.07 | 2.14 | 1.87 | -0.28 |
| 1-5% | 4.36 | 3.92 | -0.45 | 0.09 | 4.51 | 3.87 | -0.64 |
| >=5% | 3.65 | 3.65 | -0.00 | 0.04 | 3.97 | 4.00 | +0.03 |

## 3. Which crops drive the raw D0 change

Per crop, the share of balanced-panel district-years in which it is recorded at any area, early vs late window. The change in mean raw D0 equals the sum of these changes. Also shown: median share of district area when present (late), and the share of late presences below 0.5% of district area.

Sum of presence changes: unweighted -0.65 crops, area-weighted -0.53.
Gross gains (crops with rising presence) +1.62 aw; gross losses -2.15 aw.

| crop | presence early | late | change | aw change | median share late | late presences <0.5% | area early kha | area late kha |
|---|---|---|---|---|---|---|---|---|
| Peas & Beans (Pulses) | 0.42 | 0.63 | +0.206 | +0.217 | 0.291% | 62% | 398 | 677 |
| Guar Seed | 0.10 | 0.26 | +0.159 | +0.200 | 0.032% | 64% | 2,024 | 4,222 |
| Other Oilseeds | 0.04 | 0.15 | +0.107 | +0.187 | 0.111% | 74% | 52 | 283 |
| Coriander | 0.33 | 0.43 | +0.093 | +0.145 | 0.022% | 94% | 316 | 339 |
| Moong(Green Gram) | 0.71 | 0.85 | +0.137 | +0.135 | 0.413% | 54% | 2,451 | 3,485 |
| Masoor | 0.41 | 0.55 | +0.141 | +0.131 | 0.484% | 51% | 816 | 1,182 |
| Garlic | 0.37 | 0.43 | +0.053 | +0.108 | 0.031% | 89% | 76 | 217 |
| Cowpea(Lobia) | 0.00 | 0.12 | +0.120 | +0.106 | 0.246% | 66% | 0 | 88 |
| Urad | 0.71 | 0.81 | +0.102 | +0.087 | 0.719% | 42% | 2,259 | 3,214 |
| Other Cereals | 0.04 | 0.08 | +0.040 | +0.053 | 0.258% | 68% | 61 | 59 |
| Potato | 0.72 | 0.73 | +0.002 | +0.052 | 0.643% | 46% | 1,130 | 1,472 |
| Cashewnut | 0.06 | 0.10 | +0.047 | +0.040 | 0.854% | 43% | 189 | 203 |
| Other Summer Pulses | 0.00 | 0.02 | +0.016 | +0.035 | 0.009% | 100% | 0 | 2 |
| Soyabean | 0.36 | 0.38 | +0.010 | +0.027 | 2.423% | 32% | 5,955 | 9,646 |
| Sweet Potato | 0.44 | 0.43 | -0.006 | +0.026 | 0.021% | 94% | 70 | 35 |
| Tobacco | 0.33 | 0.24 | -0.096 | -0.092 | 0.022% | 80% | 308 | 321 |
| Linseed | 0.56 | 0.43 | -0.129 | -0.110 | 0.038% | 85% | 433 | 149 |
| Other  Rabi Pulses | 0.43 | 0.31 | -0.117 | -0.110 | 0.085% | 76% | 897 | 186 |
| Ragi | 0.32 | 0.23 | -0.092 | -0.111 | 0.194% | 61% | 1,372 | 811 |
| Other Kharif Pulses | 0.41 | 0.37 | -0.046 | -0.113 | 0.107% | 69% | 929 | 243 |
| Sannhamp | 0.29 | 0.19 | -0.092 | -0.136 | 0.009% | 100% | 18 | 7 |
| Sunflower | 0.49 | 0.34 | -0.148 | -0.172 | 0.029% | 83% | 1,290 | 245 |
| Small Millets | 0.56 | 0.42 | -0.144 | -0.184 | 0.146% | 64% | 767 | 285 |

Crop names first appearing anywhere in the records after 1998-99: Guar Seed (1999); Cowpea(Lobia) (2006); Other Summer Pulses (2016).
Crop names last appearing before 2019-20: none.

Distinct crop names nationally per year: 1997 49, 1998 51, 1999 51, 2000 52, 2001 52, 2002 52, 2003 52, 2004 52, 2005 52, 2006 53, 2007 52, 2008 53, 2009 53, 2010 53, 2011 53, 2012 53, 2013 53, 2014 53, 2015 53, 2016 54, 2017 54, 2018 54, 2019 54.

## 4. Reporting coverage: crop names per state per year

State-years where the number of distinct crop names rises by 4 or more over the previous reported year (1999 onward): 68 state-years in 29 states.

| state | year | names before -> after | district change | new crop names | their share of state area | mean later years they persist / years after |
|---|---|---|---|---|---|---|
| Odisha | 2019 | 14 -> 39 | 0 | Arhar/Tur; Bajra; Castor Seed; Coriander; Cotton(Lint); Cowpea(Lobia); Dry Chillies; Garlic; Ginger; Gram; Jowar; Linseed; Masoor; Mesta; Onion; Other  Rabi Pulses; Other Kharif Pulses; Peas & Beans (Pulses); Safflower; Sannhamp; Small Millets; Sunflower; Sweet Potato; Tobacco; Turmeric | 17.06% | 0.0 / 0 |
| Himachal Pradesh | 2005 | 7 -> 31 | 0 | Arhar/Tur; Coriander; Cotton(Lint); Dry Chillies; Garlic; Gram; Groundnut; Horse-Gram; Linseed; Masoor; Moong(Green Gram); Moth; Onion; Other Kharif Pulses; Peas & Beans (Pulses); Ragi; Rapeseed &Mustard; Sannhamp; Sesamum; Small Millets; Soyabean; Sugarcane; Turmeric; Urad | 7.39% | 13.3 / 14 |
| Odisha | 1999 | 13 -> 36 | 0 | Arhar/Tur; Bajra; Castor Seed; Coriander; Cotton(Lint); Dry Chillies; Garlic; Ginger; Gram; Jowar; Linseed; Masoor; Mesta; Niger Seed; Onion; Safflower; Sannhamp; Small Millets; Soyabean; Sunflower; Sweet Potato; Tobacco; Turmeric | 10.23% | 5.3 / 20 |
| Tamil Nadu | 2011 | 14 -> 36 | 0 | Arecanut; Bajra; Banana; Black Pepper; Cardamom; Cashewnut; Coconut; Coriander; Dry Chillies; Garlic; Ginger; Jowar; Jute; Onion; Potato; Rapeseed &Mustard; Small Millets; Sweet Potato; Tapioca; Tobacco; Turmeric; Wheat | 23.97% | 6.4 / 8 |
| Tamil Nadu | 2002 | 12 -> 34 | 1 | Arecanut; Banana; Cashewnut; Castor Seed; Coconut; Coriander; Dry Chillies; Garlic; Ginger; Gram; Horse-Gram; Maize; Onion; Other Kharif Pulses; Potato; Rapeseed &Mustard; Small Millets; Sweet Potato; Tapioca; Tobacco; Turmeric; Wheat | 24.89% | 14.4 / 17 |
| Tamil Nadu | 2013 | 14 -> 34 | 0 | Arecanut; Bajra; Banana; Black Pepper; Cardamom; Cashewnut; Coconut; Coriander; Dry Chillies; Garlic; Ginger; Jowar; Onion; Potato; Rapeseed &Mustard; Small Millets; Sweet Potato; Tapioca; Tobacco; Turmeric | 24.87% | 6.0 / 6 |
| Chhattisgarh | 2000 | 22 -> 41 | 4 | Bajra; Banana; Barley; Coriander; Dry Chillies; Groundnut; Horse-Gram; Jowar; Khesari; Masoor; Mesta; Moong(Green Gram); Peas & Beans (Pulses); Sannhamp; Small Millets; Tobacco; Turmeric; Urad; Wheat | 17.22% | 18.9 / 19 |
| Telangana | 2013 | 21 -> 39 | 7 | Banana; Cashewnut; Coconut; Coriander; Cowpea(Lobia); Ginger; Linseed; Masoor; Mesta; Other  Rabi Pulses; Other Kharif Pulses; Other Oilseeds; Potato; Rapeseed &Mustard; Safflower; Soyabean; Sweet Potato; Turmeric | 5.56% | 4.1 / 6 |
| Bihar | 2007 | 21 -> 38 | 1 | Banana; Castor Seed; Coriander; Dry Chillies; Garlic; Ginger; Groundnut; Onion; Other  Rabi Pulses; Other Kharif Pulses; Potato; Safflower; Small Millets; Sugarcane; Sweet Potato; Tobacco; Turmeric | 4.47% | 7.2 / 12 |
| Madhya Pradesh | 1999 | 17 -> 32 | -1 | Castor Seed; Coriander; Dry Chillies; Garlic; Ginger; Niger Seed; Onion; Other  Rabi Pulses; Other Kharif Pulses; Potato; Ragi; Safflower; Sannhamp; Sweet Potato; Turmeric; Urad | 11.75% | 16.2 / 20 |
| Andhra Pradesh | 2008 | 24 -> 39 | 0 | Arecanut; Banana; Cashewnut; Coconut; Coriander; Ginger; Linseed; Niger Seed; Potato; Rapeseed &Mustard; Safflower; Soyabean; Sweet Potato; Tapioca; Turmeric | 4.54% | 10.9 / 11 |
| Kerala | 2012 | 14 -> 29 | 0 | Arhar/Tur; Cowpea(Lobia); Garlic; Groundnut; Horse-Gram; Jowar; Maize; Moong(Green Gram); Onion; Ragi; Small Millets; Soyabean; Sweet Potato; Urad; Wheat | 0.31% | 5.3 / 7 |
| Jammu And Kashmir | 2009 | 14 -> 29 | 1 | Bajra; Cotton(Lint); Dry Chillies; Ginger; Groundnut; Horse-Gram; Maize; Moong(Green Gram); Moth; Other Cereals; Other Kharif Pulses; Rice; Sannhamp; Sesamum; Sugarcane; Tobacco; Turmeric | 61.66% | 8.0 / 10 |
| Andhra Pradesh | 2000 | 24 -> 38 | 0 | Arecanut; Arhar/Tur; Banana; Cashewnut; Coconut; Coriander; Ginger; Mesta; Other  Rabi Pulses; Other Kharif Pulses; Potato; Small Millets; Sweet Potato; Tapioca; Turmeric | 9.36% | 18.0 / 19 |
| Madhya Pradesh | 2015 | 23 -> 37 | 0 | Banana; Castor Seed; Coriander; Dry Chillies; Garlic; Ginger; Jute; Onion; Other  Rabi Pulses; Other Cereals; Potato; Sunflower; Sweet Potato; Turmeric | 3.73% | 3.5 / 4 |
| Nagaland | 1999 | 5 -> 19 | 0 | Arhar/Tur; Bajra; Castor Seed; Cotton(Lint); Gram; Groundnut; Jowar; Linseed; Moong(Green Gram); Peas & Beans (Pulses); Potato; Rapeseed &Mustard; Sesamum; Soyabean; Sugarcane; Sunflower | 20.59% | 18.8 / 20 |
| Jharkhand | 2017 | 10 -> 24 | 0 | Bajra; Castor Seed; Groundnut; Horse-Gram; Jowar; Linseed; Masoor; Moong(Green Gram); Niger Seed; Other  Rabi Pulses; Other Kharif Pulses; Peas & Beans (Pulses); Safflower; Sesamum; Soyabean; Sunflower | 9.48% | 2.0 / 2 |
| Telangana | 2000 | 21 -> 35 | 0 | Arhar/Tur; Banana; Cashewnut; Coconut; Coriander; Mesta; Other  Rabi Pulses; Other Kharif Pulses; Potato; Small Millets; Sunflower; Sweet Potato; Tapioca; Tobacco; Turmeric | 9.70% | 13.8 / 19 |
| Jammu And Kashmir | 2005 | 12 -> 25 | 0 | Cotton(Lint); Dry Chillies; Ginger; Groundnut; Onion; Other Cereals; Other Oilseeds; Potato; Rapeseed &Mustard; Sannhamp; Sesamum; Sugarcane; Tobacco; Turmeric | 1.37% | 11.3 / 14 |
| Arunachal Pradesh | 2016 | 9 -> 22 | 0 | Arhar/Tur; Groundnut; Masoor; Moong(Green Gram); Other  Rabi Pulses; Other Kharif Pulses; Other Oilseeds; Peas & Beans (Pulses); Rapeseed &Mustard; Sesamum; Soyabean; Sunflower; Urad | 17.21% | 3.0 / 3 |
| Tripura | 2004 | 6 -> 18 | 0 | Arhar/Tur; Gram; Groundnut; Jute; Masoor; Mesta; Moong(Green Gram); Other  Rabi Pulses; Other Kharif Pulses; Peas & Beans (Pulses); Rapeseed &Mustard; Sesamum; Urad | 5.16% | 15.0 / 15 |
| Puducherry | 2007 | 12 -> 23 | 0 | Bajra; Coconut; Coriander; Moong(Green Gram); Other  Rabi Pulses; Ragi; Small Millets; Sunflower; Sweet Potato; Turmeric; Urad | 21.73% | 9.9 / 12 |
| Madhya Pradesh | 2006 | 28 -> 39 | 0 | Banana; Coriander; Dry Chillies; Garlic; Ginger; Jute; Khesari; Mesta; Other  Rabi Pulses; Other Kharif Pulses; Safflower; Tobacco; Turmeric | 1.51% | 9.5 / 13 |
| Himachal Pradesh | 2001 | 21 -> 32 | 0 | Coriander; Cotton(Lint); Dry Chillies; Garlic; Ginger; Onion; Other  Rabi Pulses; Potato; Sannhamp; Sweet Potato; Tobacco; Turmeric | 1.81% | 11.3 / 18 |
| Meghalaya | 2008 | 16 -> 26 | 0 | Arecanut; Banana; Black Pepper; Dry Chillies; Ginger; Potato; Soyabean; Sweet Potato; Tapioca; Turmeric | 28.04% | 10.7 / 11 |
| Karnataka | 1999 | 32 -> 41 | 0 | Banana; Mesta; Other  Rabi Pulses; Other Kharif Pulses; Sannhamp; Small Millets; Sugarcane; Sweet Potato; Tapioca; Tobacco; Urad | 7.79% | 18.6 / 20 |
| Ladakh | 2015 | 2 -> 11 | 0 | Barley; Linseed; Masoor; Other  Rabi Pulses; Peas & Beans (Pulses); Rapeseed &Mustard; Sesamum; Small Millets; Wheat | 42.16% | 2.2 / 4 |
| Ladakh | 2006 | 1 -> 9 | 1 | Barley; Masoor; Onion; Other  Rabi Pulses; Peas & Beans (Pulses); Potato; Rapeseed &Mustard; Wheat | 38.81% | 10.1 / 13 |
| Uttarakhand | 2000 | 24 -> 32 | 3 | Garlic; Ginger; Horse-Gram; Linseed; Masoor; Other  Rabi Pulses; Other Cereals; Other Kharif Pulses; Other Oilseeds; Sesamum | 4.19% | 17.7 / 19 |
| Jammu And Kashmir | 2000 | 11 -> 19 | 0 | Dry Chillies; Garlic; Ginger; Onion; Other Oilseeds; Sannhamp; Tobacco; Turmeric; Wheat | 29.44% | 14.4 / 19 |
| Jammu And Kashmir | 2015 | 20 -> 28 | 0 | Cotton(Lint); Cowpea(Lobia); Garlic; Horse-Gram; Linseed; Moth; Other  Rabi Pulses; Other Oilseeds; Small Millets; Wheat | 29.16% | 3.5 / 4 |
| Jammu And Kashmir | 1999 | 3 -> 11 | 0 | Bajra; Barley; Gram; Groundnut; Jowar; Linseed; Rapeseed &Mustard; Sesamum; Sugarcane | 15.08% | 15.0 / 20 |
| Haryana | 2010 | 26 -> 33 | 0 | Ginger; Other  Rabi Pulses; Other Kharif Pulses; Rapeseed &Mustard; Sannhamp; Soyabean; Turmeric | 8.80% | 4.6 / 9 |
| Puducherry | 1999 | 14 -> 21 | 0 | Banana; Cashewnut; Coconut; Ginger; Jowar; Moong(Green Gram); Other  Rabi Pulses; Rapeseed &Mustard; Urad | 22.09% | 14.1 / 20 |
| Jammu And Kashmir | 2014 | 13 -> 20 | 1 | Bajra; Dry Chillies; Maize; Masoor; Other Cereals; Other Kharif Pulses; Rice; Sannhamp; Sesamum; Sugarcane; Tobacco; Urad | 99.41% | 4.2 / 5 |
| The Dadra And Nagar Haveli And Daman And Diu | 2015 | 7 -> 14 | 0 | Arhar/Tur; Groundnut; Niger Seed; Other Kharif Pulses; Rapeseed &Mustard; Sugarcane; Urad | 18.02% | 3.6 / 4 |
| Chhattisgarh | 1999 | 15 -> 22 | 0 | Castor Seed; Cotton(Lint); Garlic; Ginger; Niger Seed; Onion; Other  Rabi Pulses; Other Kharif Pulses; Potato; Ragi; Safflower; Sweet Potato | 16.20% | 19.9 / 20 |
| Haryana | 2000 | 22 -> 28 | 0 | Garlic; Ginger; Other Kharif Pulses; Sannhamp; Sweet Potato; Turmeric | 0.08% | 10.7 / 19 |
| Nagaland | 2000 | 19 -> 25 | 1 | Barley; Niger Seed; Other  Rabi Pulses; Other Kharif Pulses; Small Millets; Wheat | 9.71% | 13.7 / 19 |
| Bihar | 2016 | 30 -> 36 | 0 | Coriander; Garlic; Ginger; Onion; Potato; Turmeric | 2.54% | 3.0 / 3 |
| Telangana | 2008 | 21 -> 27 | 0 | Cashewnut; Coconut; Coriander; Safflower; Soyabean; Sugarcane; Turmeric | 7.20% | 9.3 / 11 |
| The Dadra And Nagar Haveli And Daman And Diu | 2010 | 9 -> 15 | 0 | Groundnut; Maize; Niger Seed; Sannhamp; Sesamum; Sugarcane | 4.48% | 6.2 / 9 |
| Uttar Pradesh | 2006 | 26 -> 32 | 0 | Arhar/Tur; Ginger; Groundnut; Masoor; Other  Rabi Pulses; Other Kharif Pulses | 9.04% | 8.7 / 13 |
| Mizoram | 2010 | 10 -> 15 | 0 | Arhar/Tur; Cowpea(Lobia); Onion; Rapeseed &Mustard; Sesamum; Soyabean | 8.40% | 8.5 / 9 |
| Goa | 2013 | 6 -> 11 | 0 | Arecanut; Banana; Black Pepper; Cashewnut; Coconut | 61.39% | 6.0 / 6 |
| Assam | 2005 | 25 -> 30 | 4 | Arecanut; Banana; Coconut; Maize; Tapioca | 4.73% | 14.0 / 14 |
| Andaman And Nicobar Islands | 2010 | 13 -> 18 | 0 | Arhar/Tur; Groundnut; Maize; Moong(Green Gram); Sunflower; Urad | 6.71% | 7.5 / 9 |
| Manipur | 2002 | 12 -> 17 | 0 | Cashewnut; Potato; Soyabean; Sugarcane; Tapioca; Turmeric | 3.17% | 11.5 / 16 |
| Jammu And Kashmir | 2010 | 29 -> 34 | 0 | Arhar/Tur; Cowpea(Lobia); Gram; Jowar; Small Millets | 0.45% | 3.2 / 9 |
| Jammu And Kashmir | 2006 | 25 -> 30 | 0 | Cowpea(Lobia); Garlic; Horse-Gram; Linseed; Masoor; Moth; Other  Rabi Pulses | 0.83% | 9.7 / 13 |
| Manipur | 2008 | 16 -> 21 | 0 | Arhar/Tur; Cowpea(Lobia); Gram; Groundnut; Masoor; Moong(Green Gram); Other  Rabi Pulses; Urad; Wheat | 4.42% | 10.0 / 10 |
| Madhya Pradesh | 2009 | 35 -> 40 | 0 | Cowpea(Lobia); Dry Chillies; Ginger; Mesta; Moong(Green Gram) | 0.69% | 7.2 / 10 |
| Himachal Pradesh | 2018 | 28 -> 32 | 0 | Cotton(Lint); Dry Chillies; Ginger; Sannhamp | 0.37% | 0.8 / 1 |
| Ladakh | 2010 | 8 -> 12 | 0 | Cowpea(Lobia); Horse-Gram; Moth; Onion; Small Millets | 5.88% | 2.2 / 9 |
| Andhra Pradesh | 2016 | 38 -> 42 | 0 | Guar Seed; Other  Rabi Pulses; Other Kharif Pulses; Other Oilseeds | 0.14% | 2.5 / 3 |
| Madhya Pradesh | 2001 | 30 -> 34 | 0 | Banana; Potato; Sweet Potato; Tobacco | 0.33% | 14.5 / 18 |
| Assam | 2001 | 26 -> 30 | 0 | Masoor; Moong(Green Gram); Peas & Beans (Pulses); Urad | 2.71% | 18.0 / 18 |
| Madhya Pradesh | 2003 | 35 -> 39 | 2 | Horse-Gram; Masoor; Mesta; Moong(Green Gram); Peas & Beans (Pulses); Urad | 3.33% | 14.2 / 16 |
| Tamil Nadu | 2005 | 35 -> 39 | 0 | Black Pepper; Mesta; Sannhamp; Soyabean; Tobacco | 0.18% | 4.8 / 14 |
| Jharkhand | 2011 | 7 -> 11 | 0 | Maize; Ragi; Rice; Sugarcane | 88.90% | 6.8 / 8 |
| Sikkim | 2000 | 7 -> 11 | 0 | Other Cereals; Other Kharif Pulses; Other Oilseeds; Small Millets | 10.32% | 13.0 / 19 |
| Rajasthan | 1999 | 35 -> 39 | 1 | Banana; Guar Seed; Small Millets; Tapioca | 14.74% | 14.8 / 20 |
| Meghalaya | 2000 | 21 -> 25 | 0 | Linseed; Other  Rabi Pulses; Small Millets; Wheat | 4.23% | 16.0 / 19 |
| Meghalaya | 2010 | 25 -> 29 | 0 | Cashewnut; Dry Chillies; Soyabean; Turmeric | 5.94% | 9.0 / 9 |
| Mizoram | 2005 | 14 -> 18 | 0 | Arhar/Tur; Other  Rabi Pulses; Other Kharif Pulses; Peas & Beans (Pulses); Potato; Sugarcane | 10.75% | 11.0 / 14 |
| Himachal Pradesh | 2011 | 26 -> 30 | 0 | Dry Chillies; Ginger; Onion; Turmeric | 0.46% | 6.8 / 8 |
| Nagaland | 2014 | 28 -> 32 | 0 | Horse-Gram; Linseed; Other  Rabi Pulses; Other Kharif Pulses; Sugarcane | 6.44% | 5.0 / 5 |
| Karnataka | 2002 | 37 -> 41 | 0 | Banana; Cardamom; Garlic; Sweet Potato; Tapioca; Tobacco | 1.42% | 15.5 / 17 |

Balanced panel, sum of district year-on-year raw D0 changes 1998->2019: +1695 crop-district units; of which in jump state-years: +4861 (287%), which are 8.1% of district-years. Same for D0 at 0.5%: +91 total, +1440 in jump state-years. D1: +12.4 total, +610.9 in jump state-years.

## 5. Within district: does D0 rise when D1 does not?

D0_raw: rose in 199 of 429 districts; rose while D1 did not (D1 change <= 0) in 94 (22%); rose while D1 rose in 105; fell while D1 rose in 60.
D0_0.5%: rose in 147 of 429 districts; rose while D1 did not (D1 change <= 0) in 40 (9%); rose while D1 rose in 107; fell while D1 rose in 58.

Districts whose D1 moved by less than 0.25 between windows (n = 103): mean raw D0 change +0.65 (SE 0.41, p 0.111); D0 at 0.5% -0.23 (SE 0.11, p 0.030).
Window change D0_raw on window change D1: slope 2.17 (SE 0.24), intercept -0.34 (SE 0.22, p 0.1162) = the D0 change at zero D1 change; R2 0.22; n 429.
Window change D0_0.5% on window change D1: slope 1.46 (SE 0.10), intercept -0.44 (SE 0.08, p 0.0000) = the D0 change at zero D1 change; R2 0.48; n 429.
Correlation of window changes: raw D0 with D1 0.47; D0 0.5% with D1 0.70; raw D0 with D0 0.5% 0.60.

Per-district OLS trends 1998-2019 (|t| > 1.96, naive iid SEs, so read as descriptive): raw D0 significantly up in 136 of 429; of those, D1 not significantly up in 90 and significantly down in 20. D0 at 0.5% significantly up in 83; of those D1 not up in 26. D1 significantly up in 98, down in 162.

Two-way FE (district + year), D0_raw on D1: 1.89 (SE 0.14), n 9438.
Two-way FE (district + year), D0_0.5% on D1: 1.44 (SE 0.06), n 9438.

# Part 2. The 1997-98 break

## Year profile, raw APY after cleaning (all districts)

| year | rows | states | districts | crop names | seasons | area Mha | crops/district | rows/district | area/district kha |
|---|---|---|---|---|---|---|---|---|---|
| 1997 | 8,532 | 27 | 484 | 49 | 6 | 182.2 | 15.0 | 17.6 | 376 |
| 1998 | 12,261 | 30 | 543 | 51 | 6 | 168.0 | 19.5 | 22.6 | 309 |
| 1999 | 12,987 | 32 | 562 | 51 | 6 | 164.9 | 20.4 | 23.1 | 294 |
| 2000 | 13,567 | 32 | 546 | 52 | 6 | 165.2 | 22.2 | 24.8 | 303 |
| 2001 | 13,280 | 32 | 546 | 52 | 6 | 164.6 | 21.9 | 24.3 | 302 |
| 2002 | 14,155 | 33 | 568 | 52 | 6 | 154.9 | 22.2 | 24.9 | 273 |
| 2003 | 14,635 | 33 | 583 | 52 | 6 | 170.0 | 22.4 | 25.1 | 292 |
| 2004 | 14,139 | 34 | 588 | 52 | 6 | 167.9 | 21.3 | 24.0 | 286 |
| 2005 | 14,050 | 34 | 594 | 52 | 6 | 168.9 | 20.9 | 23.7 | 284 |
| 2006 | 14,665 | 34 | 597 | 53 | 6 | 172.9 | 21.6 | 24.6 | 290 |
| 2007 | 14,645 | 33 | 606 | 52 | 6 | 173.8 | 21.0 | 24.2 | 287 |
| 2008 | 15,116 | 33 | 617 | 53 | 6 | 172.8 | 21.3 | 24.5 | 280 |
| 2009 | 15,305 | 33 | 619 | 53 | 6 | 169.4 | 21.3 | 24.7 | 274 |
| 2010 | 14,878 | 34 | 624 | 53 | 6 | 183.0 | 20.4 | 23.8 | 293 |
| 2011 | 16,121 | 34 | 636 | 53 | 6 | 177.6 | 21.5 | 25.3 | 279 |
| 2012 | 15,268 | 34 | 641 | 53 | 6 | 181.3 | 20.5 | 23.8 | 283 |
| 2013 | 16,169 | 34 | 648 | 53 | 6 | 182.5 | 21.0 | 25.0 | 282 |
| 2014 | 15,570 | 34 | 660 | 53 | 6 | 177.3 | 19.7 | 23.6 | 269 |
| 2015 | 16,305 | 34 | 662 | 53 | 6 | 179.4 | 20.6 | 24.6 | 271 |
| 2016 | 17,398 | 34 | 687 | 54 | 6 | 184.9 | 21.1 | 25.3 | 269 |
| 2017 | 17,989 | 34 | 687 | 54 | 6 | 186.0 | 21.7 | 26.2 | 271 |
| 2018 | 18,282 | 34 | 692 | 54 | 6 | 185.1 | 22.2 | 26.4 | 268 |
| 2019 | 19,255 | 33 | 699 | 54 | 6 | 194.9 | 22.7 | 27.5 | 279 |
| 2020 | 319 | 1 | 13 | 30 | 4 | 0.9 | 22.8 | 24.5 | 71 |

States in 1998-99 absent in 1997-98: Chandigarh (0.00 Mha in 1998); Himachal Pradesh (0.87 Mha in 1998); Puducherry (0.03 Mha in 1998); Tamil Nadu (6.78 Mha in 1998); The Dadra And Nagar Haveli And Daman And Diu (0.03 Mha in 1998); Tripura (0.28 Mha in 1998).
States in 1997-98 absent in 1998-99: Goa; Jammu And Kashmir; Ladakh.

### By state: area (Mha), districts and crop names, 1997-98 against 1998-99

| state | area 97 | area 98 | area 99 | ratio 97/98 | districts 97/98 | crop names 97/98 |
|---|---|---|---|---|---|---|
| Uttar Pradesh | 24.19 | 23.49 | 22.85 | 1.03 | 72/73 | 24/30 |
| Maharashtra | 12.74 | 19.54 | 20.04 | 0.65 | 30/29 | 13/24 |
| Madhya Pradesh | 13.14 | 18.18 | 19.60 | 0.72 | 38/48 | 14/17 |
| Rajasthan | 16.12 | 18.03 | 18.06 | 0.89 | 32/32 | 18/35 |
| Gujarat | 8.89 | 11.86 | 8.17 | 0.75 | 19/19 | 26/27 |
| Karnataka | 7.76 | 10.83 | 11.37 | 0.72 | 20/27 | 18/32 |
| Andhra Pradesh | 9.34 | 10.56 | 9.74 | 0.88 | 20/20 | 23/23 |
| West Bengal | 58.97 | 8.12 | 8.39 | 7.26 | 18/18 | 36/38 |
| Bihar | 7.10 | 7.28 | 4.79 | 0.98 | 37/37 | 35/36 |
| Punjab | 6.82 | 6.88 | 6.91 | 0.99 | 17/17 | 15/17 |
| Tamil Nadu | - | 6.78 | 6.37 | - | 0/28 | 0/35 |
| Haryana | 5.05 | 5.74 | 5.45 | 0.88 | 19/19 | 11/24 |
| Odisha | 0.91 | 5.23 | 7.88 | 0.17 | 30/30 | 11/13 |
| Chhattisgarh | 0.75 | 3.89 | 4.42 | 0.19 | 6/12 | 12/15 |
| Assam | 3.42 | 3.40 | 3.56 | 1.00 | 23/23 | 25/26 |
| Kerala | 1.36 | 1.81 | 1.88 | 0.75 | 14/14 | 5/20 |
| Jharkhand | 1.91 | 1.78 | 0.35 | 1.07 | 18/18 | 34/34 |
| Telangana | 1.18 | 1.38 | 1.36 | 0.86 | 3/3 | 22/23 |
| Uttarakhand | 0.87 | 0.96 | 0.95 | 0.91 | 8/10 | 23/27 |
| Himachal Pradesh | - | 0.87 | 0.84 | - | 0/12 | 0/26 |
| Tripura | - | 0.28 | 0.25 | - | 0/4 | 0/18 |
| Arunachal Pradesh | 0.23 | 0.24 | 0.22 | 0.97 | 13/13 | 10/14 |
| Meghalaya | 0.22 | 0.22 | 0.20 | 1.00 | 7/7 | 25/24 |
| Nagaland | 0.18 | 0.21 | 0.23 | 0.84 | 7/7 | 4/5 |
| Manipur | 0.00 | 0.19 | 0.18 | 0.00 | 9/9 | 10/17 |
| Mizoram | 0.09 | 0.09 | 0.06 | 0.94 | 4/4 | 14/11 |
| Sikkim | 0.10 | 0.09 | 0.08 | 1.11 | 4/4 | 9/9 |
| Puducherry | - | 0.03 | 0.04 | - | 0/4 | 0/14 |
| The Dadra And Nagar Haveli And Daman And Diu | - | 0.03 | 0.03 | - | 0/1 | 0/20 |
| Chandigarh | - | 0.00 | 0.00 | - | 0/1 | 0/11 |
| Goa | 0.04 | 0.00 | - | - | 2/0 | 5/0 |
| Jammu And Kashmir | 0.83 | 0.00 | 0.67 | - | 12/0 | 3/0 |
| Ladakh | 0.01 | 0.00 | 0.01 | - | 2/0 | 1/0 |

### Area by season (Mha)

| year | Autumn | Kharif | Rabi | Summer | Whole Year | Winter |
|---|---|---|---|---|---|---|
| 1997 | 2.1 | 58.1 | 97.4 | 2.6 | 11.3 | 10.7 |
| 1998 | 2.3 | 83.2 | 53.9 | 3.5 | 11.1 | 14.0 |
| 1999 | 1.9 | 83.1 | 53.6 | 3.6 | 12.8 | 10.0 |
| 2000 | 4.9 | 86.7 | 48.2 | 3.0 | 13.5 | 9.0 |
| 2010 | 2.4 | 89.4 | 64.8 | 4.7 | 10.6 | 11.2 |
| 2019 | 1.9 | 95.9 | 67.4 | 5.6 | 10.5 | 13.6 |

### National area by crop (thousand ha), and districts reporting it

| crop | 1997 | 1998 | 1999 | 2000 | ratio 97/98 | districts 97 | districts 98 |
|---|---|---|---|---|---|---|---|
| Rice | 30,244 | 43,843 | 41,138 | 43,191 | 0.69 | 370 | 522 |
| Wheat | 26,186 | 27,169 | 27,122 | 25,626 | 0.96 | 403 | 448 |
| Jowar | 10,332 | 9,894 | 10,334 | 9,990 | 1.04 | 258 | 308 |
| Bajra | 9,363 | 9,334 | 8,893 | 9,912 | 1.00 | 246 | 288 |
| Cotton(Lint) | 5,002 | 9,332 | 8,718 | 8,551 | 0.54 | 138 | 252 |
| Gram | 4,601 | 8,430 | 6,159 | 5,186 | 0.55 | 287 | 408 |
| Groundnut | 5,643 | 7,640 | 7,071 | 6,677 | 0.74 | 283 | 368 |
| Soyabean | 5,031 | 6,452 | 5,979 | 6,380 | 0.78 | 127 | 201 |
| Maize | 5,878 | 5,869 | 6,511 | 6,590 | 1.00 | 414 | 483 |
| Rapeseed &Mustard | 5,433 | 5,372 | 5,686 | 3,982 | 1.01 | 309 | 434 |
| Moth | 982 | 3,825 | 798 | 1,006 | 0.26 | 75 | 84 |
| Sugarcane | 3,316 | 3,790 | 4,245 | 4,329 | 0.88 | 321 | 437 |
| Arhar/Tur | 3,314 | 3,257 | 3,004 | 3,626 | 1.02 | 370 | 414 |
| Moong(Green Gram) | 1,424 | 2,775 | 3,102 | 2,975 | 0.51 | 169 | 309 |
| Urad | 1,803 | 2,327 | 3,193 | 2,732 | 0.77 | 203 | 314 |
| Ragi | 1,381 | 1,775 | 1,790 | 1,725 | 0.78 | 126 | 192 |
| Sunflower | 435 | 1,714 | 1,055 | 1,053 | 0.25 | 160 | 255 |
| Sesamum | 1,228 | 1,594 | 1,697 | 1,683 | 0.77 | 327 | 473 |
| Small Millets | 1,077 | 1,281 | 531 | 1,324 | 0.84 | 199 | 246 |
| Coconut | 928 | 1,238 | 1,293 | 1,437 | 0.75 | 57 | 83 |
| Potato | 988 | 1,139 | 1,171 | 1,074 | 0.87 | 233 | 322 |
| Jute | 980 | 883 | 855 | 836 | 1.11 | 78 | 83 |
| Other  Rabi Pulses | 250 | 849 | 1,600 | 1,001 | 0.29 | 133 | 159 |
| Barley | 820 | 781 | 686 | 764 | 1.05 | 230 | 255 |
| Castor Seed | 557 | 675 | 789 | 1,090 | 0.83 | 65 | 161 |
| Masoor | 562 | 664 | 732 | 962 | 0.85 | 105 | 155 |
| Horse-Gram | 574 | 663 | 899 | 694 | 0.86 | 76 | 121 |
| Linseed | 567 | 649 | 628 | 488 | 0.87 | 195 | 276 |
| Dry Chillies | 467 | 611 | 800 | 735 | 0.76 | 153 | 278 |
| Other Kharif Pulses | 104 | 572 | 1,307 | 1,496 | 0.18 | 102 | 151 |

Crops reported in 1998-99 but absent everywhere in 1997-98: Other Cereals (155 kha in 1998); Other Oilseeds (7 kha in 1998).

### Same districts, 1997-98 against 1998-99 (n = 467, of which 429 in balanced panel)

D0_97 vs D0_98: means 15.48 vs 19.68, aw 18.72 vs 21.18; mean paired difference -4.20 (SE 0.23); lower in 1997 in 373 of 467 districts.
D1_97 vs D1_98: means 4.36 vs 4.79, aw 3.87 vs 5.48; mean paired difference -0.43 (SE 0.07); lower in 1997 in 309 of 467 districts.
Cropped area ratio 1997/1998 per district: median 0.95; below 0.8 in 139 of 467; below 0.5 in 55; total 181.3 vs 154.0 Mha.

Districts in both years: seasons present in 1998-99 but missing in 1997-98: Whole Year 89, Winter 23, Autumn 22, Summer 21, Kharif 12, Rabi 9 (of 467 districts).
Crops present in 1998-99 but missing in 1997-98 within the same district, most frequent: Sannhamp 98, Moong(Green Gram) 97, Rice 93, Sesamum 91, Rapeseed &Mustard 91, Sugarcane 90, Peas & Beans (Pulses) 89, Tobacco 88, Dry Chillies 83, Castor Seed 82, Onion 78, Sunflower 75, Gram 72, Other Kharif Pulses 72, Turmeric 72.

Largest crop's share of district area, same districts: mean 0.519 in 1997 vs 0.512 in 1998 (n 467).
Most common largest crop 1997: Rice 163, Wheat 113, Jowar 31, Maize 26, Niger Seed 18, Bajra 16
Most common largest crop 1998: Rice 261, Wheat 120, Jowar 24, Maize 19, Soyabean 19, Cotton(Lint) 17
