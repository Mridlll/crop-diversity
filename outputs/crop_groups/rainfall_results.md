# District rainfall 1990-2020 (script 97_imd_district_rainfall.py)

## Source
IMD Pune 0.25-degree gridded daily rainfall (Pai et al. 2014), binary files for 1990-2021,
fetched by POST to https://imdpune.gov.in/cmpg/Griddata/rainfall.php (form field
`rain=<year>`, the endpoint imdlib 0.1.21 uses; landing page
https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html). Downloads were not blocked.
All 32 files have the exact expected size (129 x 135 x days x 4 bytes). Raw files sit in
`data_raw/imd_rain_025/` (about 810 MB). `data_raw/` was added to `.gitignore`.

Validation source: IMD's official subdivision monthly rainfall 1901-2015 (the data.gov.in
"Sub_Division_IMD" table), from the GitHub mirror
github.com/SAMUR274/rainfall-in-india-1901-2015-1-, saved to `data_raw/imd_subdivision/`.

## Method
1. Daily grids summed to monthly totals per cell. 4,964 cells are valid on every day.
2. Cell QC. The grid writes 0.0, not the missing code, where no station fed a cell, so a
   cell can show a season of zero rain inside a wet region. A cell-season that totals
   exactly 0 is set missing when the cell averages above 50 mm in its non-zero years and
   a cell within 3 cells got above 200 mm (JJAS) or 100 mm (OND, Jan-May). This masked
   519 JJAS, 559 OND and 642 Jan-May cell-seasons. Saiha and Lawngtlai (Mizoram) lose
   most JJAS years before 2007 to this.
3. Bull's-eye test (flagged but kept): a cell whose seasonal total is above 2.5 times its
   own 1990-2020 mean and above 2 times the median of that ratio over the 7 x 7 cells
   around it.
4. Cells were intersected with the district polygons of `docs/data/districts.geojson`
   in an equal-area projection. Each polygon's rainfall is the area-weighted mean of
   its cells, renormalised over cells with data. It is left missing if under half the
   weight has data. Polygons sharing a key are pooled by area.
5. Seasons: `jjas` is Jun-Sep of year t, `ond` is Oct-Dec of t, `jfmam` is Jan-May of
   t, `jfmam_next` is Jan-May of t+1 (the rabi months of crop year "t-(t+1)") and
   `annual` is Jan-Dec of t.
6. Anomalies are taken against each key's own 1990-2020 mean and SD, with QC-flagged
   years left out of the baseline: `pct_` (percent departure), `z_`, `dry20_`
   (pct <= -20), `wet20_` (pct >= +20) and `imdcat_`. `imdcat_` uses IMD's district
   bands (large excess >= 60, excess 20 to 59, normal -19 to 19, deficient -20 to -59,
   large deficient -60 to -99, no rain -100) on the own-period baseline, not on IMD's
   LPA. Percent departure is blanked where a key's season mean is under 10 mm. That
   affects 5 keys for OND and 39 for Jan-May.
7. `qc_<season>` = 1 if at least 10% of the key's area lies in bull's-eye cells. For JJAS
   it is also 1 if the key lies in a subdivision-year where the grid departure and IMD's
   official departure differ by more than 30 points. `qc_subdiv_jjas` and
   `spike_share_*` hold the parts.

## Keys and match rate
Keys are the `k` values of `docs/data/districts.geojson`. All 679 geojson keys have
rainfall. The geojson leaves 44 crop-record keys without a polygon. An explicit alias
table in the script adds them:
- renames: Gurugram, Nuh, Prayagraj, Bhadohi, Y.S.R., Leh, Upper Dibang Valley, the 24
  Parganas, S.A.S. Nagar, Muktsar and others
- 2011-parent unions: the seven Telangana parents filed under Andhra Pradesh, plus
  Adilabad and Mahabubnagar (whose 23-year series spans the 2016 split), Firozpur with
  Fazilka, Kanchipuram with Chengalpattu, Villupuram with Kallakurichi, and Pashchim
  Medinipur with Jhargram

Total keys with rainfall: 721.
- 606-district analysis sample: 604 of 606 matched. The two misses are the Andaman
  districts, which have no IMD land cells.
- 725 crop-record districts: 721 matched. The misses are the 3 Andaman and Nicobar
  districts and Gaurella-Pendra-Marwahi (one crop year, no polygon).

Output columns `district_key` and `in606` link to the analysis key and the sample.

## Sanity checks
All-India JJAS, area-weighted, percent departure from the 1990-2020 mean:

| year | grid | districts | share of districts at -20% or worse |
|---|---|---|---|
| 2002 | -19.1 | -19.8 | 0.51 |
| 2009 | -15.2 | -16.2 | 0.51 |
| 2014 | -10.2 | -10.7 | 0.37 |
| 2015 | -10.9 | -11.0 | 0.39 |
| 2004 | -10.5 | -10.6 | 0.33 |
| 2019 | +12.5 | +12.5 | 0.13 |
| 2020 | +12.2 | +10.8 | 0.16 |

2002 and 2009 are the two driest years, and 2014, 2015 and 2004 come next. In other
years the share of districts at -20% or worse runs from 0.12 to 0.31. IMD's own LPA
departures are -19, -22, -12, -14 and -14 for those five years. Our 2009 figure is
milder because it is taken against a 1990-2020 mean on an area-weighted grid rather
than IMD's subdivision-weighted LPA. The rank order is the same. The all-India mean
JJAS is 860 mm on the grid, against IMD's LPA of 870 to 880 mm.

Regional checks: in 2002, Rajasthan's median district JJAS is -67%. In 2014, Haryana's
median district is -47%. Kerala in 2018 is +36%.

Against IMD's official subdivision JJAS series (16 subdivisions, 1990-2015, 416
subdivision-years):
- pooled anomaly correlation: 0.825, or 0.935 once the 13 flagged subdivision-years are
  left out
- at or above 0.94: Bihar, Odisha, Jharkhand, Chhattisgarh, Haryana, Telangana, Kerala,
  Saurashtra
- weak: Arunachal Pradesh 0.57, Assam and Meghalaya 0.73, NMMT 0.79

## Problems
- **The IMD grid has real artefacts.**
  - Tamil Nadu: JJAS 2011 and 2012 are 2.7 times the official subdivision value
    (grid +149% and +104%, official -0.4% and -16%). Tirunelveli shows 100 to 300 mm in
    every month of 2011.
  - Bull's-eyes: Pune 2005 is 4,224 mm against a 987 mm mean, Amreli 2011 is +316%, and
    there are others in Saurashtra 2011, Punjab 2003 and 2012, and Uttarakhand 2010.
  - Arunachal and Assam and Meghalaya diverge from the official series in 1998, 2008
    and 2012.

  All of these carry `qc_jjas` = 1. In total, 393 key-years are flagged for JJAS, 159
  for OND and 175 for Jan-May; 138 of the JJAS flags fall in 2012. Drop or
  sensitivity-test flagged rows.
- The subdivision check covers only 16 subdivisions: those that coincide with states,
  plus Saurashtra and Madhya Maharashtra. Karnataka, Madhya Pradesh, Uttar Pradesh,
  Rajasthan, coastal Andhra, Rayalaseema, West Bengal and the rest of Maharashtra are
  checked only by the bull's-eye test, and 2016-2020 only by that test as well. A
  regional artefact like Tamil Nadu 2011 would go uncaught there.
- The Northeast is weakest throughout because the station network is sparse.
- Some Telangana parent unions are approximate. Jayashankar took mandals from both
  Warangal and Karimnagar; it is assigned to Warangal.
- `districts.geojson` gives four polygons a neighbour's key: Firozpur to Fazilka,
  Bengaluru Rural to Bengaluru Urban, Viluppuram to Kallakurichi, and Kancheepuram to
  Chengalpattu. The script corrects this for rainfall. The site maps still carry it.
- Keys whose district split after the crop series began are measured on the current
  map polygon unless they are listed in the alias table. For a within-district anomaly
  this matters little.
- Small or coastal keys have thin grid cover: Chennai 0.27, Chengalpattu 0.69 and four
  others under 0.80. Diu is a single cell.
- OND and Jan-May percent departures are volatile where the season mean is small. Use
  `z_` or a millimetre threshold there.

## Files
- `outputs/crop_groups/rainfall_district_year.csv`: 721 keys x 31 years, 64 columns
- `outputs/crop_groups/rainfall_polygon_year.csv`: seasonal totals per map polygon,
  including polygons without a key
- `outputs/crop_groups/rainfall_allindia_check.csv`
- `outputs/crop_groups/rainfall_subdivision_validation.csv`
