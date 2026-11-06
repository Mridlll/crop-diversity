# Crop-group panel and district adjacency

Built by `scripts/97_crop_group_adjacency.py` (run it first) and `scripts/98_crop_group_panel.py`. Both use only APY crop records, the script 72 crosswalk and `docs/data/districts.geojson`.

## Unit and period

- The unit is the 2011 (PC11) parent district, `unit_key`, which is the same as the SHRUG `shrug_key`. All 725 crop-record districts are summed to their 2011 parent through `outputs/shrug_covariates/district_crosswalk.csv`. Mumbai, Daman and Diu have no 2011 parent, so they stay as their own units and carry `unit_no_2011_parent = 1`.
- That gives 629 units, 13,686 unit-years, 1997-98 to 2019-20 (23 years; 2020-21 is dropped).
- The 606-district analysis sample sits in 605 units (`in_606 = 1`, with Sultanpur and Amethi sharing one), which matches the 605 clustering groups. Since the unit is the 2011 parent, cluster on `unit_key`. Those 605 units have 13,291 unit-years. 436 of them are observed in all 23 years and 547 in at least 20.
- Number of child districts per unit: one child for 563 units, two for 49, three for 9, four for 5, five for 2, seven for 1.
- Unit-years by year (all units): 484 in 1997-98, 543 in 1998-99, then rising to 606 by 2007-08 and 622-626 from 2010-11. 2019-20 has 617.

## Cleaning

The cleaning is script 76's, unchanged. Bogus state-district pairs are dropped. The Uttarakhand, Chhattisgarh and Jharkhand districts are remapped. 2020-21 is dropped. Duplicate (district, year, season, crop) cells (91 of them) are collapsed by taking the maximum. Area is in hectares.

## Seasons

| panel season | APY seasons |
|---|---|
| kharif | Kharif, Autumn, Winter (Autumn and Winter are the eastern aus and aman rice crops) |
| rabi | Rabi |
| summer | Summer (boro rice and other dry-season crops) |
| whole_year | Whole Year |
| annual | sum of the four; `season_area_ha` equals `gca_ha` |

Unit-years with any area in each season: kharif 13,582, rabi 12,355, summer 6,665, whole_year 10,657.

## Crop-name mapping (all 54 APY names)

| group | APY crop names |
|---|---|
| rice | Rice |
| wheat | Wheat |
| maize | Maize |
| millets | Jowar, Bajra, Ragi, Small Millets |
| pulses | Gram, Arhar/Tur, Moong(Green Gram), Urad, Masoor, Moth, Horse-Gram, Khesari, Cowpea(Lobia), Peas & Beans (Pulses), Other Kharif Pulses, Other  Rabi Pulses, Other Summer Pulses |
| oilseeds | Groundnut, Rapeseed &Mustard, Soyabean, Sunflower, Sesamum, Castor Seed, Linseed, Niger Seed, Safflower, Other Oilseeds |
| cotton | Cotton(Lint) |
| sugarcane | Sugarcane |
| other | Coconut, Guar Seed, Barley, Other Cereals, Potato, Onion, Tapioca, Sweet Potato, Jute, Mesta, Sannhamp, Tobacco, Dry Chillies, Coriander, Garlic, Ginger, Turmeric, Black Pepper, Cardamom, Arecanut, Cashewnut, Banana |

Three judgement calls:
- Coconut is typed as Oilseeds in APY but is not on the specified oilseed list, so it goes to other.
- Guar is a fodder legume, not a pulse, so it goes to other.
- Barley and Other Cereals go to other.

Area by crop is in `crop_group_crop_mapping.csv`.

National share of gross cropped area, using only unit-years whose annual totals are reliable:

| year | rice | wheat | maize | millets | pulses | oilseeds | cotton | sugarcane | other |
|---|---|---|---|---|---|---|---|---|---|
| 1998-99 | 0.263 | 0.167 | 0.033 | 0.130 | 0.138 | 0.148 | 0.054 | 0.023 | 0.044 |
| 2005-06 | 0.258 | 0.153 | 0.040 | 0.122 | 0.116 | 0.166 | 0.054 | 0.025 | 0.066 |
| 2012-13 | 0.228 | 0.180 | 0.050 | 0.090 | 0.125 | 0.153 | 0.071 | 0.029 | 0.074 |
| 2019-20 | 0.241 | 0.184 | 0.050 | 0.073 | 0.145 | 0.146 | 0.070 | 0.024 | 0.067 |

## Defect flags (unit-year; 1 = affected)

| flag | meaning | unit-years (all / 605 units) |
|---|---|---|
| f_odisha_season_collapsed | Odisha 2001-02 to 2003-04: all crops filed as Kharif, so the season split is lost | 90 / 90 |
| f_odisha_summer_rice_2012 | Odisha 2012-13: summer rice is 8.6 Mha against a usual 0.3 Mha, in every district, so annual rice and GCA are inflated | 30 / 30 |
| f_bihar_rice_season_swap | Bihar 1998-99 to 2001-02: rice moves between Autumn, Winter, Kharif and Summer | 144 / 144 |
| f_bihar_1999_rice_missing | Bihar 1999-00: winter rice is absent, about 3 Mha, so annual GCA is too low | 37 / 37 |
| f_tn_rice_single_season | Tamil Nadu up to 2016-17: rice is reported only as Kharif | 566 / 566 |
| f_whole_year_ge50, f_whole_year_ge20 | Whole Year rows hold at least 50 or 20 percent of GCA. This catches Kerala and MP 2009-10, where 19 Mha was filed as Whole Year | 685 / 653; 1,594 / 1,521 |
| f_gca_jump_severe, f_gca_jump_mild | GCA divided by the unit's own median is below 0.5 or above 2, or below 2/3 or above 1.5. This catches all of West Bengal 1997-98 (rabi of 51 Mha), Odisha 1997-98 (partial), Manipur 1997-98 and Chhattisgarh before it was formed | 457 / 418; 1,099 / 1,035 |
| f_child_gap | a child district is missing in a year when it reports both before and after | 15 / 15 |
| f_late_entrant | the unit enters after 1998-99 (entry is the first year it holds at least a fifth of its median GCA) | 1,040 / 960 |
| f_before_entry | the stub years before a unit's entry | 41 / 37 |
| f_pre_split_donor | the years before a neighbouring late entrant appears, for a unit that held that entrant before a pre-2011 split. 89 donor links are in `crop_group_presplit_donors.csv` | 557 / 526 |
| flag_season_unreliable | Odisha collapsed seasons, Odisha 2012 summer rice, the Bihar swap, Tamil Nadu rice, or whole_year_ge50 | 1,395 / 1,363 |
| flag_annual_unreliable | Odisha 2012, Bihar 1999, severe GCA jump, pre-split donor, child gap, or before entry | 978 / 918 |
| flag_any | any of the above | 4,080 / 3,898 |

For the 605 sample units:

| | unit-years |
|---|---|
| Annual totals reliable | 12,373 of 13,291 |
| Season split reliable | 11,928 |
| Both reliable | 11,170 |

## Adjacency

- The geojson has 735 present-day polygons. 682 are matched through the geojson's own key and the crosswalk. 37 are post-2011 districts or renames matched to their 2011 parent by hand. One is a name override: the key sent Bengaluru Rural to Bengaluru Urban.
- The polygons are dissolved to units, and contiguity is queen with a 500 m tolerance. The tolerance is needed because the web geojson is simplified at 1.2 km.
- The number of links depends little on the tolerance: 1,696 at 0 m, 1,712 at 500 m and 1,728 at 1,000 m. The 16 links that 500 m adds are listed in `crop_group_adjacency_log.txt`.
- Mean degree is 5.45, ranging from 1 to 11.
- Every one of the 629 panel units has a polygon.
- The only isolated unit is Nicobars.
- 15 polygons have no unit because they have no crop records: Delhi's 11 districts, Kolkata, Lakshadweep, Muzaffarabad and Mirpur.

## Problems

- The pre-split donor flag is a heuristic and errs toward flagging too much. It flags every neighbour of a late entrant whose GCA falls at least 10 percent when the entrant appears. Where the whole state dips in that year it flags several donors: Arwal has 5, Baksa 4, and the 2007-08 Jammu and Kashmir entrants several each. Most real donors are found, for example Gondiya from Bhandara, Washim from Akola, Tarn Taran from Amritsar and Yadgir from Gulbarga.
- Constant 2011 boundaries do not fix splits from before 2011. The flag marks those years but the panel does not repair them.
- 14 late entrants have no donor found. They are listed in the log, and include Porbandar, North Goa, Saiha, Simdega and Saraikela.
- 1997-98 is weak across several states, so treat it with caution even where it is not flagged.
- Odisha 2012-13 and West Bengal 1997-98 are flagged but not corrected. Drop them for any annual-share work.
- The season shares for Assam, West Bengal, Odisha and Bihar depend on putting Autumn and Winter into kharif. Summer is the only dry-season rice.
