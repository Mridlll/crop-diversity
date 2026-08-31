# Audit: how the crop diversity indices are constructed

Raw source: `all_crops_apy_1997_2021_india_data_portal.csv`
Compared against: `district_diversity_indices.csv` (725 districts)

Raw file: 345,273 rows, 16 columns.
Columns: id, year, state_name, state_code, district_name, district_code, season, crop_code, crop_name, crop_type, area, area_unit, production, production_unit, yield, yield_unit

After dropping missing and non-positive area: 345,270 rows, 755 districts, 54 crops, 24 years.

## B1. Seasons, and whether Whole Year double counts

| season | rows | share of area |
|---|---|---|
| Kharif | 138,328 | 0.485 |
| Rabi | 100,941 | 0.331 |
| Whole Year | 68,664 | 0.075 |
| Summer | 22,099 | 0.025 |
| Winter | 8,249 | 0.070 |
| Autumn | 6,989 | 0.015 |

Crop-district-years appearing under BOTH Whole Year and a named season: 441
As a share of all Whole Year crop-district-years: 0.0064

Example overlaps:
  ARUNACHAL PRADESH|CHANGLANG 1998 Maize: Kharif=1575; Whole Year=5625
  ARUNACHAL PRADESH|DIBANG VALLEY 1998 Maize: Kharif=5225; Whole Year=3555
  ARUNACHAL PRADESH|EAST KAMENG 1998 Maize: Kharif=1601; Whole Year=686
  ARUNACHAL PRADESH|EAST SIANG 1998 Maize: Kharif=2478; Whole Year=3855
  ARUNACHAL PRADESH|LOHIT 1998 Maize: Kharif=7784; Whole Year=7365
  ARUNACHAL PRADESH|LOWER SUBANSIRI 1998 Maize: Kharif=4178; Whole Year=1116
  ARUNACHAL PRADESH|PAPUM PARE 1998 Maize: Kharif=1607; Whole Year=1120
  ARUNACHAL PRADESH|TAWANG 1998 Maize: Kharif=872; Whole Year=285

If a Whole Year row equals the sum of that crop's seasonal rows, summing all
of them double counts. Checking whether Whole Year equals the seasonal sum:
  of 441 overlapping cases, Whole Year is within 2% of the seasonal sum in 34 (7.7%)

## B2. Duplicate rows

Exact duplicates on (district, year, season, crop): 307 of 345,270 rows.

These are summed by the groupby, which inflates area for the affected cells.
Example:
```
                       district_key    year season crop_name    area
ANDAMAN AND NICOBAR ISLANDS|PURULIA 2007-08 Kharif     Wheat 2439.60
ANDAMAN AND NICOBAR ISLANDS|PURULIA 2007-08 Kharif     Wheat  837.95
ANDAMAN AND NICOBAR ISLANDS|PURULIA 2007-08 Kharif     Wheat  366.16
ANDAMAN AND NICOBAR ISLANDS|PURULIA 2007-08 Kharif     Wheat  218.20
ANDAMAN AND NICOBAR ISLANDS|PURULIA 2007-08 Kharif     Wheat 7333.75
ANDAMAN AND NICOBAR ISLANDS|PURULIA 2007-08 Kharif     Wheat  106.50
```

## B3. Does reconstructed area look like real gross cropped area?

India gross cropped area implied by this file, million hectares:

| year | implied GCA (m ha) |
|---|---|
| 1997 | 182.2 |
| 2000 | 165.4 |
| 2005 | 168.9 |
| 2010 | 183.0 |
| 2015 | 179.5 |
| 2019 | 194.9 |
| 2020 | 0.9 |

> Published Indian gross cropped area is about 195-200 million hectares.
> A figure far below that means the file is a partial crop or district set;
> far above means double counting.

## A1. Year coverage per district, and its effect on richness

Years of data per district: min 1, p25 17, median 23, p75 23, max 24.
Districts with fewer than 20 years: 205 of 755 (27.2%).

Correlation between years of coverage and POOLED richness: r = 0.545 (p = 1.3e-59).
Correlation between years of coverage and MEAN ANNUAL richness: r = 0.309 (p = 3.4e-18).

If the first is much larger than the second, pooled richness is partly measuring
how long a district was observed rather than how diverse it is.

## A2. Pooled richness vs mean annual richness

| statistic | pooled over 24 years | mean annual |
|---|---|---|
| mean | 29.8 | 20.1 |
| median | 32.0 | 21.1 |
| p10 | 17.0 | 9.6 |
| p90 | 40.0 | 29.2 |
| max | 45.0 | 37.4 |

Ratio of pooled to annual, median: 1.49x.
Rank correlation between the two: rho = 0.825.

## C1. Pooled Shannon vs mean annual Shannon

| index | pooled mean | annual mean | difference | rank correlation |
|---|---|---|---|---|
| Shannon | 1.496 | 1.391 | +0.105 | 0.975 |
| Simpson | 0.626 | 0.601 | +0.025 | 0.972 |

Reproducing the published `shannon_index` from raw with the pooled method: r = 0.9999 on 725 districts.
Median absolute difference: 0.0000.

A near-perfect match confirms the published indices ARE the 24-year pooled
version, and that this audit is reading the pipeline correctly.

## D1. Does the inverted U survive a corrected diversity measure?

Districts with a corrected measure: 697 of 697.
Correlation between published ABI and the annual-mean ABI: r = 0.948, rho = 0.948.

| outcome | n | linear | squared | p(squared) | turning point |
|---|---|---|---|---|---|
| published ABI (24-yr pooled) | 697 | +0.324 | -0.416*** | 0.0001 | 0.390 |
| ABI on annual means | 697 | +0.383 | -0.474*** | 0.0000 | 0.404 |
| Shannon, annual mean | 697 | +1.118 | -1.579*** | 0.0000 | 0.354 |
| Simpson, annual mean | 697 | +0.314 | -0.419*** | 0.0003 | 0.375 |
| richness, annual mean | 697 | +13.904 | -13.311*** | 0.0013 | 0.522 |