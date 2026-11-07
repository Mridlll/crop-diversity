# APY extension past 2019-20

Old file: 345,273 rows. New file (DES via India Data Portal): 455,359 rows, years 1997-2022.

## Units

Old area/production units: ['Hectare'] / ['Tonnes']. New: ['Hectare'] / ['Tonnes']. Both hectares and tonnes; no
conversion needed. Yield unit in both: ['Tonnes/Hectare'] / ['Tonnes/Hectare'].

## Seasons

Old labels: ['Autumn', 'Kharif', 'Rabi', 'Summer', 'Whole Year', 'Winter']. New labels: ['Autumn', 'Kharif', 'Rabi', 'Summer', 'Total', 'Whole Year', 'Winter'].
The new file adds a season label 'Total' (48,044 rows, 1292.7 Mha) that is the sum of a
crop's seasonal rows in the same district-year; the old file has no such rows.
Dropped to avoid double counting. Total-only cells with no seasonal row to stand in for them: 0.
Season labels and year strings are written in the old file's padded format (e.g. 'Kharif     ', ' 1998-99').

Bogus state-district pairs (script 76 list) found in new file: 0 rows.

## Crop names

Old file has 54 crop names; new file 115. 53 are identical, 1 differs by whitespace
('Other Rabi Pulses' -> 'Other  Rabi Pulses'). 61 new-file crop names have no old
counterpart and are excluded. Excluded area, all years: 9.5 Mha of 4686.1 Mha (0.20 percent).

Excluded crops with area in 2020-21..2022-23 (hectares; full list in apy_extend_unmapped.csv):

| crop | 2020-21 | 2021-22 | 2022-23 |
|---|---|---|---|
| Arcanut (Processed) | 549,661 | 603,122 | 674,811 |
| Dry Ginger | 45,504 | 45,158 | 25,100 |

Arecanut: from 2020-21 the new file carries BOTH 'Arecanut' and 'Arcanut (Processed)'
(2022-23: 891k and 675k ha). Only 'Arecanut' is kept, so arecanut is not double counted.

## Districts

Matching rule is script 104's: exact state|district key, else normalised name within
state, else alias list, else difflib >= 0.85 within state, else same name in a state the
old file may have filed it under. New-file (state, district) pairs by method: {'exact': 643, 'alias': 36, 'none': 31, 'fuzzy': 29, 'normalised': 1}.

Unmatched districts (no old key; kept in the output under their own name so they are
not lost, but they will not join to the existing district keys). Mean annual area
2020-23 in hectares; share of all new-year area 0.73 percent:

| state | district | mean annual ha | years present |
|---|---|---|---|
| Tamil Nadu | Mayiladuthurai | 162,062 | 2020,2021,2022 |
| Karnataka | Vijayanagara | 139,260 | 2022 |
| Andhra Pradesh | Nandyal | 134,224 | 2022 |
| Andhra Pradesh | Sri Sathya Sai | 103,070 | 2022 |
| Andhra Pradesh | Palnadu | 90,952 | 2022 |
| Andhra Pradesh | Bapatla | 87,052 | 2022 |
| Andhra Pradesh | Eluru | 83,153 | 2022 |
| Punjab | Malerkotla | 71,333 | 2021,2022 |
| Andhra Pradesh | Kakinada | 61,793 | 2022 |
| Chhattisgarh | Khairagarh-Chhuikhadan-Gandai | 60,320 | 2022 |
| Andhra Pradesh | Dr. B.R. Ambedkar Konaseema | 56,284 | 2022 |
| Andhra Pradesh | Ntr | 48,578 | 2022 |
| Chhattisgarh | Sakti | 42,188 | 2022 |
| Chhattisgarh | Sarangarh-Bilaigarh | 39,492 | 2022 |
| Andhra Pradesh | Tirupati | 39,168 | 2022 |
| Andhra Pradesh | Parvathipuram Manyam | 38,851 | 2022 |
| Andhra Pradesh | Alluri Sitharama Raju | 36,074 | 2022 |
| Andhra Pradesh | Anakapalli | 34,114 | 2022 |
| Chhattisgarh | Mohla-Manpur-Ambagarh Chouki | 32,237 | 2022 |
| Andhra Pradesh | Annamayya | 23,806 | 2021,2022 |
| Chhattisgarh | Manendragarh-Chirmiri-Bharatpur(M C B) | 16,019 | 2022 |
| Nagaland | Niuland | 9,248 | 2022 |
| Nagaland | Chumoukedima | 8,705 | 2022 |
| Nagaland | Noklak | 7,981 | 2021,2022 |
| Nagaland | Shamator | 4,262 | 2022 |
| Sikkim | Pakyong | 3,951 | 2021,2022 |
| Nagaland | Tseminyu | 3,049 | 2022 |
| Mizoram | Saitual | 3,020 | 2020,2021,2022 |
| Sikkim | Soreng | 2,677 | 2022 |
| Mizoram | Khawzawl | 2,296 | 2020,2021,2022 |
| Mizoram | Hnahthial | 1,148 | 2020,2021,2022 |

Non-exact matches (new -> old key):

- Andhra Pradesh | Ananthapuramu -> ANDHRA PRADESH|ANANTAPUR [alias]
- Andhra Pradesh | Visakhapatnam -> ANDHRA PRADESH|VISAKHAPATANAM [fuzzy]
- Andhra Pradesh | Y.S.R. Kadapa -> ANDHRA PRADESH|KADAPA [alias]
- Assam | Sribhumi -> ASSAM|KARIMGANJ [alias]
- Chhattisgarh | Balodabazar-Bhatapara -> CHHATTISGARH|BALODA BAZAR [alias]
- Chhattisgarh | Balrampur-Ramanujganj -> CHHATTISGARH|BALRAMPUR [alias]
- Chhattisgarh | Gaurela-Pendra-Marwahi -> CHHATTISGARH|GAURELLA-PENDRA-MARWAHI [alias]
- Chhattisgarh | Kabeerdham -> CHHATTISGARH|KABIRDHAM [alias]
- Chhattisgarh | Uttar Bastar Kanker -> CHHATTISGARH|KANKER [alias]
- Gujarat | Ahmedabad -> GUJARAT|AHMADABAD [fuzzy]
- Gujarat | Arvalli -> GUJARAT|ARAVALLI [fuzzy]
- Gujarat | Dahod -> GUJARAT|DOHAD [alias]
- Gujarat | Dangs -> GUJARAT|DANG [fuzzy]
- Haryana | Charkhi Dadri -> HARYANA|CHARKI DADRI [fuzzy]
- Haryana | Gurugram -> HARYANA|GURGAON [alias]
- Haryana | Nuh -> HARYANA|MEWAT [alias]
- Himachal Pradesh | Lahaul And Spiti -> HIMACHAL PRADESH|LAHUL AND SPITI [fuzzy]
- Jammu And Kashmir | Budgam -> JAMMU AND KASHMIR|BADGAM [alias]
- Jammu And Kashmir | Rajouri -> JAMMU AND KASHMIR|RAJAURI [fuzzy]
- Karnataka | Bengaluru Rural -> KARNATAKA|BANGALORE RURAL [alias]
- Karnataka | Bengaluru South -> KARNATAKA|RAMANAGARA [alias]
- Karnataka | Chamarajanagar -> KARNATAKA|CHAMARAJANAGARA [fuzzy]
- Karnataka | Davanagere -> KARNATAKA|DAVANGERE [fuzzy]
- Karnataka | Yadgir -> KARNATAKA|YADAGIRI [fuzzy]
- Madhya Pradesh | Agar-Malwa -> MADHYA PRADESH|AGAR MALWA [normalised]
- Madhya Pradesh | Khandwa (East Nimar) -> MADHYA PRADESH|KHANDWA [alias]
- Madhya Pradesh | Khargone (West Nimar) -> MADHYA PRADESH|KHARGONE [alias]
- Madhya Pradesh | Narmadapuram -> MADHYA PRADESH|HOSHANGABAD [alias]
- Madhya Pradesh | Narsimhapur -> MADHYA PRADESH|NARSINGHPUR [alias]
- Maharashtra | Ahilyanagar -> MAHARASHTRA|AHMEDNAGAR [alias]
- Maharashtra | Chhatrapati Sambhajinagar -> MAHARASHTRA|AURANGABAD [alias]
- Maharashtra | Dharashiv -> MAHARASHTRA|OSMANABAD [alias]
- Mizoram | Siaha -> MIZORAM|SAIHA [alias]
- Odisha | Angul -> ODISHA|ANUGUL [fuzzy]
- Odisha | Balasore -> ODISHA|BALESHWAR [alias]
- Odisha | Jajpur -> ODISHA|JAJAPUR [fuzzy]
- Odisha | Keonjhar -> ODISHA|KENDUJHAR [alias]
- Puducherry | Puducherry -> PUDUCHERRY|PONDICHERRY [alias]
- Punjab | Ferozepur -> PUNJAB|FIROZEPUR [fuzzy]
- Punjab | Sri Muktsar Sahib -> PUNJAB|MUKTSAR [alias]
- Tamil Nadu | Kancheepuram -> TAMIL NADU|KANCHIPURAM [fuzzy]
- Tamil Nadu | Thoothukkudi -> TAMIL NADU|TUTICORIN [alias]
- Tamil Nadu | Viluppuram -> TAMIL NADU|VILLUPURAM [fuzzy]
- Telangana | Bhadradri Kothagudem -> TELANGANA|BHADRADRI [alias]
- Telangana | Jayashankar Bhupalapally -> TELANGANA|JAYASHANKAR [alias]
- Telangana | Jogulamba Gadwal -> TELANGANA|JOGULAMBA [alias]
- Telangana | Kumuram Bheem Asifabad -> TELANGANA|KOMARAM BHEEM ASIFABAD [fuzzy]
- Telangana | Narayanpet -> TELANGANA|NARAYANAPET [fuzzy]
- Telangana | Rajanna Sircilla -> TELANGANA|RAJANNA [alias]
- Telangana | Ranga Reddy -> TELANGANA|RANGAREDDI [alias]
- Telangana | Yadadri Bhuvanagiri -> TELANGANA|YADADRI [alias]
- Uttar Pradesh | Ayodhya -> UTTAR PRADESH|FAIZABAD [alias]
- Uttar Pradesh | Bara Banki -> UTTAR PRADESH|BARABANKI [fuzzy]
- Uttar Pradesh | Bhadohi -> UTTAR PRADESH|SANT RAVIDAS NAGAR [alias]
- Uttar Pradesh | Kushinagar -> UTTAR PRADESH|KUSHI NAGAR [fuzzy]
- Uttar Pradesh | Mahrajganj -> UTTAR PRADESH|MAHARAJGANJ [fuzzy]
- Uttar Pradesh | Prayagraj -> UTTAR PRADESH|ALLAHABAD [alias]
- Uttar Pradesh | Sant Kabir Nagar -> UTTAR PRADESH|SANT KABEER NAGAR [fuzzy]
- Uttar Pradesh | Shrawasti -> UTTAR PRADESH|SHRAVASTI [fuzzy]
- Uttar Pradesh | Siddharthnagar -> UTTAR PRADESH|SIDDHARTH NAGAR [fuzzy]
- Uttarakhand | Rudraprayag -> UTTARAKHAND|RUDRA PRAYAG [fuzzy]
- Uttarakhand | Udham Singh Nagar -> UTTARAKHAND|UDAM SINGH NAGAR [fuzzy]
- Uttarakhand | Uttarkashi -> UTTARAKHAND|UTTAR KASHI [fuzzy]
- West Bengal | Cooch Behar -> WEST BENGAL|COOCHBEHAR [fuzzy]
- West Bengal | Malda -> WEST BENGAL|MALDAH [fuzzy]
- West Bengal | Paschim Medinipur -> WEST BENGAL|PASHCHIM MEDINIPUR [fuzzy]

Old-file districts present in 2019-20 but absent from 2020-23: 3 of 699.
Largest by 2019-20 area: MAHARASHTRA|MUMBAI SUBURBAN (1,046 ha); PUDUCHERRY|YANAM (714 ha); PUDUCHERRY|MAHE (550 ha)

Duplicate (district, year, season, crop) rows in the new file after mapping: 0.
Left in the CSV as delivered; script 76 collapses them by maximum downstream.

## Output

`data_raw/apy_extended_1997_2024.csv`: 344,954 old rows (1997-98..2019-20, byte-for-byte values,
the 319 stub rows of 2020-21 removed) + 55,523 new rows for 2020-21..2022-23. File name
follows the request; no DES data past 2022-23 exists in this source.

| year | rows | Mha | districts | crops |
|---|---|---|---|---|
| 2020-21 | 18,599 | 195.9 | 701 | 54 |
| 2021-22 | 18,278 | 197.7 | 707 | 54 |
| 2022-23 | 18,646 | 199.8 | 731 | 54 |

## Overlap check, 1997-98 to 2019-20 (old file vs new file, same crops)

Cells (district, year, season, crop): old 344,647, new 345,453, in both 339,274; only old 5,373; only new 6,179.
Cell level area: Pearson r = 0.96680; identical in 99.8 percent of shared cells;
median abs percent difference 0.000; 99th percentile 0.00 percent; cells differing by over 10 percent: 519.
District-year gross area totals (13,855 pairs): r = 0.98262; median abs percent difference 0.000;
pairs differing by more than 5 percent: 193.

Worst ten district-years by absolute percent difference (full list in apy_extend_overlap_districts.csv):

| district | year | old ha | new ha | pct |
|---|---|---|---|---|
| UTTARAKHAND|TEHRI GARHWAL | 2014-15 | 80,710 | 258,480 | +220.3 |
| UTTARAKHAND|PITHORAGARH | 2014-15 | 69,448 | 219,336 | +215.8 |
| UTTARAKHAND|BAGESHWAR | 2014-15 | 39,314 | 122,850 | +212.5 |
| UTTARAKHAND|PAURI GARHWAL | 2014-15 | 91,498 | 285,158 | +211.7 |
| UTTARAKHAND|UTTAR KASHI | 2014-15 | 39,376 | 120,902 | +207.0 |
| UTTARAKHAND|ALMORA | 2014-15 | 113,254 | 345,298 | +204.9 |
| UTTARAKHAND|CHAMPAWAT | 2014-15 | 26,002 | 78,176 | +200.7 |
| UTTARAKHAND|CHAMOLI | 2014-15 | 46,420 | 139,086 | +199.6 |
| UTTARAKHAND|RUDRA PRAYAG | 2014-15 | 31,264 | 93,306 | +198.4 |
| UTTARAKHAND|DEHRADUN | 2014-15 | 50,728 | 146,408 | +188.6 |

National gross area by year, Mha (cells in both sources only would differ less; this is all cells):

| year | old | new | pct |
|---|---|---|---|
| 1997-98 | 182.2 | 228.3 | +25.3 |
| 1998-99 | 168.1 | 175.3 | +4.2 |
| 1999-00 | 165.1 | 165.0 | -0.1 |
| 2000-01 | 165.4 | 165.2 | -0.1 |
| 2001-02 | 164.8 | 164.7 | -0.1 |
| 2002-03 | 155.0 | 154.9 | -0.1 |
| 2003-04 | 170.1 | 170.0 | -0.1 |
| 2004-05 | 168.0 | 167.9 | -0.0 |
| 2005-06 | 168.9 | 168.9 | -0.0 |
| 2006-07 | 173.0 | 172.9 | -0.0 |
| 2007-08 | 173.8 | 173.8 | -0.0 |
| 2008-09 | 172.9 | 172.8 | -0.0 |
| 2009-10 | 169.5 | 169.4 | -0.0 |
| 2010-11 | 183.0 | 183.0 | -0.0 |
| 2011-12 | 177.6 | 177.6 | -0.0 |
| 2012-13 | 181.3 | 181.3 | -0.0 |
| 2013-14 | 182.5 | 182.5 | +0.0 |
| 2014-15 | 177.3 | 179.1 | +1.1 |
| 2015-16 | 179.4 | 179.5 | +0.1 |
| 2016-17 | 184.9 | 184.9 | +0.0 |
| 2017-18 | 186.0 | 186.0 | +0.0 |
| 2018-19 | 185.1 | 185.1 | +0.0 |
| 2019-20 | 194.9 | 195.1 | +0.1 |

Per crop, shared cells: lowest correlation / highest median difference (all 1997-2019):

| crop | total old ha | total new ha | pct | r | median abs pct |
|---|---|---|---|---|---|
| Potato | 31,575,340 | 54,301,933 | +71.98 | 0.1392 | 0.000 |
| Tobacco | 8,730,076 | 18,698,994 | +114.19 | 0.9944 | 0.000 |
| Cotton(Lint) | 232,223,123 | 232,450,901 | +0.10 | 0.9960 | 0.000 |
| Groundnut | 132,604,032 | 133,471,665 | +0.65 | 0.9974 | 0.000 |
| Garlic | 3,700,747 | 3,701,728 | +0.03 | 0.9991 | 0.000 |
| Onion | 9,902,781 | 9,956,010 | +0.54 | 0.9992 | 0.000 |
| Castor Seed | 19,666,425 | 19,692,236 | +0.13 | 0.9992 | 0.000 |
| Sesamum | 38,828,174 | 38,916,158 | +0.23 | 0.9993 | 0.000 |

Largest single-cell disagreements (old vs new area, ha):

- WEST BENGAL|MURSHIDABAD 1997-98 Whole Year Potato: old 6,964, new 4,314,164
- WEST BENGAL|NADIA 1997-98 Whole Year Potato: old 5,337, new 4,100,237
- WEST BENGAL|MALDAH 1997-98 Whole Year Potato: old 2,254, new 2,880,154
- WEST BENGAL|BIRBHUM 1997-98 Whole Year Potato: old 10,034, new 1,174,834
- WEST BENGAL|NORTH 24 PARGANAS 1997-98 Whole Year Potato: old 6,199, new 981,399
- WEST BENGAL|PASHCHIM MEDINIPUR 1997-98 Whole Year Potato: old 44,109, new 850,409
- WEST BENGAL|COOCHBEHAR 1997-98 Whole Year Potato: old 14,158, new 799,858
- WEST BENGAL|PURBA BARDHAMAN 1997-98 Whole Year Potato: old 40,940, new 795,640
- WEST BENGAL|PURBA MEDINIPUR 1997-98 Whole Year Potato: old 18,011, new 743,411
- WEST BENGAL|JALPAIGURI 1997-98 Whole Year Potato: old 12,558, new 681,958
- WEST BENGAL|DAKSHIN DINAJPUR 1997-98 Whole Year Potato: old 2,897, new 628,997
- WEST BENGAL|SOUTH 24 PARGANAS 1997-98 Whole Year Potato: old 1,446, new 511,046

Shared cells differing by more than 1 percent, by year: 1997:40; 1998:12; 1999:21; 2000:23; 2001:21; 2002:21; 2003:24; 2004:9; 2005:8; 2006:4; 2007:8; 2008:9; 2009:11; 2010:10; 2011:9; 2012:6; 2013:4; 2014:82; 2015:74; 2016:71; 2017:69; 2018:70; 2019:66
Cells present in one source only, by year (old only / new only): 1997:483/1359; 1998:645/754; 1999:682/662; 2000:306/283; 2001:290/272; 2002:289/272; 2003:295/276; 2004:291/277; 2005:278/265; 2006:268/260; 2007:160/144; 2008:258/244; 2009:263/247; 2010:264/252; 2011:270/259; 2012:144/137

Years 2018-19 and 2019-20 only (the two years closest to the extension):
r = 0.99849; identical cells 99.6 percent; median abs percent difference 0.000.

## Collapse flags, state by state

Computed on the extended file after script 76's collapse rule (duplicates -> maximum),
seasons summed. Baseline is each state's 2015-16..2019-20 median. Thresholds:
crop count < 60 percent of baseline; gross area outside 75-125 percent of baseline;
districts reporting < 80 percent of baseline; any single crop in a district-year above
1.5 times that district's baseline gross area (district area is not in the file, so the
district's own 2015-20 gross cropped area stands in for it); any crop-level yield
above 100 t/ha for a non-sugarcane, non-coconut, non-banana, non-tapioca crop.

Calibration: the same tests flag 14 of 169 state-years in 2015-16..2019-20 (by construction
mostly the single-crop and yield tests, since the baseline is those years).

New years: 35 of 101 state-years flagged.

| state | year | crops | usual | area ratio | districts ratio | flags |
|---|---|---|---|---|---|---|
| Andaman And Nicobar Islands | 2021-22 | 16 | 19 | 0.75 | 1.00 | area 75pct of 2015-19 median |
| Assam | 2020-21 | 32 | 32 | 0.97 | 1.22 | 26 crop-cells > 1.5x district baseline area |
| Assam | 2021-22 | 31 | 32 | 0.96 | 1.22 | 26 crop-cells > 1.5x district baseline area |
| Assam | 2022-23 | 31 | 32 | 0.98 | 1.22 | 25 crop-cells > 1.5x district baseline area |
| Haryana | 2022-23 | 10 | 29 | 0.97 | 1.00 | crops 34pct of usual |
| Jammu And Kashmir | 2020-21 | 26 | 28 | 0.63 | 0.95 | area 63pct of 2015-19 median |
| Jammu And Kashmir | 2021-22 | 19 | 28 | 0.34 | 0.90 | area 34pct of 2015-19 median |
| Jammu And Kashmir | 2022-23 | 19 | 28 | 0.34 | 0.90 | area 34pct of 2015-19 median |
| Karnataka | 2020-21 | 44 | 43 | 1.16 | 1.00 | 1 crop-cells > 1.5x district baseline area |
| Karnataka | 2021-22 | 43 | 43 | 1.16 | 1.00 | 1 crop-cells > 1.5x district baseline area |
| Karnataka | 2022-23 | 45 | 43 | 1.16 | 1.03 | 1 crop-cells > 1.5x district baseline area |
| Kerala | 2022-23 | 29 | 27 | 0.96 | 1.00 | 1 yields > 100 t/ha |
| Ladakh | 2020-21 | 9 | 11 | 2.19 | 1.00 | area 219pct of 2015-19 median |
| Ladakh | 2021-22 | 8 | 11 | 2.03 | 1.00 | area 203pct of 2015-19 median |
| Ladakh | 2022-23 | 8 | 11 | 2.07 | 2.00 | area 207pct of 2015-19 median; 3 crop-cells > 1.5x district baseline area |
| Madhya Pradesh | 2020-21 | 20 | 37 | 1.14 | 1.02 | crops 54pct of usual |
| Madhya Pradesh | 2021-22 | 21 | 37 | 1.09 | 1.02 | crops 57pct of usual; 1 yields > 100 t/ha |
| Madhya Pradesh | 2022-23 | 22 | 37 | 1.16 | 1.02 | crops 59pct of usual |
| Manipur | 2020-21 | 2 | 22 | 0.49 | 1.00 | crops 9pct of usual; area 49pct of 2015-19 median |
| Manipur | 2021-22 | 8 | 22 | 0.85 | 1.00 | crops 36pct of usual |
| Manipur | 2022-23 | 8 | 22 | 0.83 | 1.00 | crops 36pct of usual |
| Meghalaya | 2021-22 | 12 | 31 | 0.66 | 1.00 | crops 39pct of usual; area 66pct of 2015-19 median |
| Nagaland | 2021-22 | 32 | 35 | 0.47 | 1.09 | area 47pct of 2015-19 median |
| Nagaland | 2022-23 | 33 | 35 | 0.74 | 1.45 | area 74pct of 2015-19 median |
| Puducherry | 2020-21 | 23 | 23 | 1.01 | 0.50 | districts 50pct of usual |
| Puducherry | 2021-22 | 22 | 23 | 1.06 | 0.50 | districts 50pct of usual |
| Puducherry | 2022-23 | 24 | 23 | 1.13 | 0.50 | districts 50pct of usual |
| Tamil Nadu | 2020-21 | 38 | 38 | 1.07 | 1.23 | 20 crop-cells > 1.5x district baseline area |
| Tamil Nadu | 2021-22 | 37 | 38 | 1.11 | 1.23 | 20 crop-cells > 1.5x district baseline area |
| Tamil Nadu | 2022-23 | 38 | 38 | 1.09 | 1.23 | 19 crop-cells > 1.5x district baseline area |
| Telangana | 2020-21 | 32 | 32 | 1.45 | 1.07 | area 145pct of 2015-19 median |
| Telangana | 2021-22 | 32 | 32 | 1.37 | 1.07 | area 137pct of 2015-19 median |
| Telangana | 2022-23 | 32 | 32 | 1.52 | 1.07 | area 152pct of 2015-19 median; 2 crop-cells > 1.5x district baseline area |
| The Dadra And Nagar Haveli And Daman And Diu | 2021-22 | 3 | 15 | 0.83 | 1.00 | crops 20pct of usual |
| The Dadra And Nagar Haveli And Daman And Diu | 2022-23 | 3 | 15 | 0.83 | 1.00 | crops 20pct of usual |

Unflagged new state-years: Andaman And Nicobar Islands 2020, Andaman And Nicobar Islands 2022, Andhra Pradesh 2020, Andhra Pradesh 2021, Andhra Pradesh 2022, Arunachal Pradesh 2020, Arunachal Pradesh 2021, Arunachal Pradesh 2022, Bihar 2020, Bihar 2021, Bihar 2022, Chandigarh 2020, Chandigarh 2021, Chandigarh 2022, Chhattisgarh 2020, Chhattisgarh 2021, Chhattisgarh 2022, Goa 2020, Goa 2021, Goa 2022, Gujarat 2020, Gujarat 2021, Gujarat 2022, Haryana 2020, Haryana 2021, Himachal Pradesh 2020, Himachal Pradesh 2021, Himachal Pradesh 2022, Jharkhand 2020, Jharkhand 2021, Jharkhand 2022, Kerala 2020, Kerala 2021, Maharashtra 2020, Maharashtra 2021, Maharashtra 2022, Meghalaya 2020, Meghalaya 2022, Mizoram 2020, Mizoram 2021, Mizoram 2022, Nagaland 2020, Odisha 2020, Odisha 2021, Odisha 2022, Punjab 2020, Punjab 2021, Punjab 2022, Rajasthan 2020, Rajasthan 2021, Rajasthan 2022, Sikkim 2020, Sikkim 2021, Sikkim 2022, Tripura 2020, Tripura 2021, Tripura 2022, Uttar Pradesh 2020, Uttar Pradesh 2021, Uttar Pradesh 2022, Uttarakhand 2020, Uttarakhand 2021, Uttarakhand 2022, West Bengal 2020, West Bengal 2021, West Bengal 2022

National gross area, Mha: 2015-16 179.4; 2016-17 184.9; 2017-18 186.0; 2018-19 185.1; 2019-20 194.9; 2020-21 195.9; 2021-22 197.7; 2022-23 199.8
National mean crops per district-year: 2015-16 20.6; 2016-17 21.1; 2017-18 21.7; 2018-19 22.2; 2019-20 22.7; 2020-21 21.3; 2021-22 20.7; 2022-23 20.2

Crops whose national area in a new year is under 60 percent of their 2015-19 median
(baseline over 50,000 ha); these are the crop-list drops that move diversity indices:

| crop | baseline ha | 2020-21 | 2021-22 | 2022-23 |
|---|---|---|---|---|
| Cashewnut | 373,567 | 241,664 (65pct) | 221,302 (59pct) | 216,207 (58pct) |
| Coriander | 403,874 | 131,855 (33pct) | 65,755 (16pct) | 121,890 (30pct) |
| Garlic | 329,936 | 158,398 (48pct) | 180,045 (55pct) | 127,492 (39pct) |
| Jowar | 6,067,677 | 4,606,986 (76pct) | 4,461,832 (74pct) | 3,553,537 (59pct) |
| Niger Seed | 1,082,348 | 81,813 (8pct) | 71,448 (7pct) | 57,326 (5pct) |
| Small Millets | 440,099 | 249,194 (57pct) | 297,072 (68pct) | 388,606 (88pct) |
| Turmeric | 169,991 | 112,345 (66pct) | 105,257 (62pct) | 96,755 (57pct) |

State-crop pairs with baseline over 50,000 ha that fall under 25 percent of baseline in a new year:

- Andhra Pradesh / Banana 2020-21: 0 ha vs baseline 59,098
- Andhra Pradesh / Banana 2021-22: 0 ha vs baseline 59,098
- Andhra Pradesh / Banana 2022-23: 0 ha vs baseline 59,098
- Andhra Pradesh / Cashewnut 2020-21: 0 ha vs baseline 112,612
- Andhra Pradesh / Cashewnut 2021-22: 0 ha vs baseline 112,612
- Andhra Pradesh / Cashewnut 2022-23: 0 ha vs baseline 112,612
- Andhra Pradesh / Niger Seed 2020-21: 3,715 ha vs baseline 914,698
- Andhra Pradesh / Niger Seed 2021-22: 3,729 ha vs baseline 914,698
- Andhra Pradesh / Niger Seed 2022-23: 2,498 ha vs baseline 914,698
- Haryana / Guar Seed 2022-23: 0 ha vs baseline 135,806
- Haryana / Sugarcane 2022-23: 0 ha vs baseline 101,500
- Jammu And Kashmir / Maize 2020-21: 0 ha vs baseline 293,858
- Jammu And Kashmir / Maize 2021-22: 0 ha vs baseline 293,858
- Jammu And Kashmir / Maize 2022-23: 0 ha vs baseline 293,858
- Jammu And Kashmir / Rice 2021-22: 0 ha vs baseline 280,513
- Jammu And Kashmir / Rice 2022-23: 0 ha vs baseline 280,513
- Madhya Pradesh / Barley 2021-22: 16,208 ha vs baseline 94,000
- Madhya Pradesh / Coriander 2020-21: 0 ha vs baseline 279,837
- Madhya Pradesh / Coriander 2021-22: 0 ha vs baseline 279,837
- Madhya Pradesh / Coriander 2022-23: 0 ha vs baseline 279,837
- Madhya Pradesh / Dry Chillies 2020-21: 0 ha vs baseline 94,415
- Madhya Pradesh / Dry Chillies 2021-22: 0 ha vs baseline 94,415
- Madhya Pradesh / Dry Chillies 2022-23: 0 ha vs baseline 94,415
- Madhya Pradesh / Garlic 2020-21: 0 ha vs baseline 178,157
- Madhya Pradesh / Garlic 2021-22: 0 ha vs baseline 178,157
- Madhya Pradesh / Garlic 2022-23: 0 ha vs baseline 178,157
- Madhya Pradesh / Niger Seed 2021-22: 7,616 ha vs baseline 61,300
- Madhya Pradesh / Niger Seed 2022-23: 10,080 ha vs baseline 61,300
- Madhya Pradesh / Onion 2020-21: 0 ha vs baseline 150,839
- Madhya Pradesh / Onion 2021-22: 0 ha vs baseline 150,839
- Madhya Pradesh / Onion 2022-23: 0 ha vs baseline 150,839
- Madhya Pradesh / Potato 2020-21: 0 ha vs baseline 145,736
- Madhya Pradesh / Potato 2021-22: 0 ha vs baseline 145,736
- Madhya Pradesh / Potato 2022-23: 0 ha vs baseline 145,736
- Madhya Pradesh / Small Millets 2020-21: 0 ha vs baseline 130,000
- Nagaland / Moong(Green Gram) 2020-21: 480 ha vs baseline 68,080
- Nagaland / Moong(Green Gram) 2021-22: 416 ha vs baseline 68,080
- Nagaland / Moong(Green Gram) 2022-23: 656 ha vs baseline 68,080
- Uttar Pradesh / Banana 2020-21: 3,331 ha vs baseline 69,378
- Uttar Pradesh / Banana 2021-22: 3,279 ha vs baseline 69,378
- Uttar Pradesh / Banana 2022-23: 3,283 ha vs baseline 69,378

## Cross-check against the UPAG resource (same package)

UPAG: 30 crop names, seasons ['Kharif', 'Rabi', 'Summer'], years 2013-14 to 2024-25, 156,189 rows.
Area column is described as hectares; national totals below are over 30 crops only.

| year | shared cells | r | median abs pct diff | UPAG Mha | DES Mha (same cells) |
|---|---|---|---|---|---|
| 2013-14 | 7,756 | 0.9963 | 0.00 | 152.4 | 151.4 |
| 2014-15 | 7,696 | 0.9948 | 0.00 | 149.8 | 149.0 |
| 2015-16 | 7,776 | 0.9915 | 0.00 | 149.5 | 148.9 |
| 2016-17 | 8,184 | 0.9878 | 0.00 | 153.5 | 152.1 |
| 2017-18 | 8,437 | 0.9919 | 0.00 | 154.6 | 153.6 |
| 2018-19 | 8,717 | 0.9931 | 0.00 | 154.3 | 153.4 |
| 2019-20 | 8,820 | 0.9938 | 0.00 | 161.9 | 160.9 |
| 2020-21 | 8,723 | 0.9951 | 0.00 | 165.1 | 164.2 |
| 2021-22 | 8,550 | 0.9943 | 0.00 | 166.7 | 165.6 |
| 2022-23 | 8,840 | 0.9962 | 0.00 | 169.0 | 168.2 |

UPAG covers 28 of the 54 old crop names (52 percent), so its 2023-24 and 2024-25 years
cannot extend the 54-crop series without a crop-list break; they are not merged.
