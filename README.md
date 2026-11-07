# Crop diversity in Indian districts, 1998-99 to 2019-20

District-level study of what Indian farmers grow and how varied it is: 724 districts, 54 crops, 22 agricultural years (1998-99 to 2019-20), set against irrigation, markets, monsoon shocks and neighbouring districts. Diversity is measured with Hill numbers computed for each district-year and then averaged.

**Live site: https://mridlll.github.io/crop-diversity/**

| Page | What it covers |
|---|---|
| [Overview](https://mridlll.github.io/crop-diversity/index.html) | The main results in one scroll |
| [Measurement](https://mridlll.github.io/crop-diversity/measurement.html) | How diversity is counted, the count-of-crops artefact, seasons and rotation |
| [Irrigation](https://mridlll.github.io/crop-diversity/irrigation.html) | Diversity across the irrigation gradient, over time and by water source |
| [Markets](https://mridlll.github.io/crop-diversity/markets.html) | Haats, mandis, input shops and producer organisations |
| [Crop groups](https://mridlll.github.io/crop-diversity/crop-groups.html) | Which crops gained and lost land, monsoon shocks, policy events |
| [Neighbours](https://mridlll.github.io/crop-diversity/neighbours.html) | How alike neighbouring districts are, state borders, spread of new crops |
| [Data and methods](https://mridlll.github.io/crop-diversity/data.html) | Sources, sample, checks and models |
| Maps: [diversity](https://mridlll.github.io/crop-diversity/diversity.html), [year by year](https://mridlll.github.io/crop-diversity/timeline.html), [calorie diversity](https://mridlll.github.io/crop-diversity/calorie-diversity.html), [food and non-food](https://mridlll.github.io/crop-diversity/food-nonfood.html) | District maps |

Every analysis in the repository, where its numbers live and how far it survived the referee passes is listed in **[outputs/ANALYSIS_INDEX.md](outputs/ANALYSIS_INDEX.md)**. Where this README and a results note disagree, the index holds.

## Key findings

All results are within-state comparisons (state fixed effects) unless stated, with standard errors clustered on the 2011 parent district and checked by a wild cluster bootstrap by state.

1. **Districts grow many crops but rely on a few.** The average district records 20.6 crops in a year, yet its effective number of crops (exp of Shannon entropy over area shares) is 4.73 and its number of main crops is 3.34.

2. **The rise in the count of crops is a reporting effect.** In the 500 districts observed every year, the raw count rises from 21.2 to 25.3 between 1998-99 and 2019-20. The whole rise sits in crops holding under 0.1 percent of a district's area, and once those are left out the count is flat or falling (at 0.5 percent of area it falls slightly). State-year reporting shocks explain 77 percent of the year-on-year change in the raw count. The effective number of crops is flat nationally (area-weighted 5.58 in 1998-99, 5.48 in 2019-20), and in the median district it changed by -0.17 between 1998-2004 and 2013-2019; the area-weighted change of -0.20 is not statistically significant once state-wide swings are allowed for.

3. **1997-98 is dropped because its records are broken.** That year West Bengal files 50.8 million ha of rabi niger seed (28 percent of all area recorded nationally), Odisha has no kharif returns, Tamil Nadu, Himachal Pradesh and Tripura are absent, and twelve states file crop lists under 60 percent of their usual length. Every average and change starts in 1998-99.

4. **Diversity is flat across the less irrigated districts and falls once irrigation passes about half of sown area.** Against the least irrigated fifth of districts in the same state, the effective number of crops differs by +0.12, -0.55, -1.05 and -2.14 crops across the higher fifths (wild bootstrap p 0.013 for the top fifth). Above an irrigated share of 0.30 it falls by about 0.4 crops for every 10 percentage points. The fall holds in 2015-16 to 2019-20 and in 2017-18 to 2021-22. Irrigation measured in the 1991 census predicts lower diversity today, although only near significance over the full period. Inside a district over time, more irrigation barely moves the effective number (a 10-point rise moves it by between -0.06 and +0.02 crops). The data cannot tell canal water from groundwater; tank and lake irrigation carries the strongest negative link (-0.48 crops per within-state standard deviation, p 0.008).

5. **A third to two fifths of annual diversity is rotation between seasons.** Splitting the year into seasons, between-season variety accounts for 0.34 to 0.42 of annual Shannon entropy across 560 districts. Rice-wheat districts drop 100 to 190 rank places once seasons are separated, because their annual diversity is mostly two crops in turn.

6. **Weekly haats go with more crops, while producer organisations go with more even cropping.** Measured as the share of a district's rural population living in a village with the facility, a full haat share goes with 6.3 more crops counted (about 0.9 crops per standard deviation) but no change in the effective number. Producer organisations go with +1.9 effective crops, which is only near significance when errors are clustered by state (p 0.09 to 0.16). Fertiliser shops go with a lower pulse share, and mandis do not pull districts toward cereals.

7. **Millets are the one large, uniform retreat.** Between 1998-2000 and 2017-2019 millets lost 5.6 percentage points of gross cropped area (7.05 million ha), falling in 358 districts and rising in 72; Maharashtra alone lost 3.45 million ha. Wheat (+1.35), maize (+1.28) and cotton (+1.73) gained share. Rice has no robust trend, and pulses rose through a step after 2015 (+2.86 points area-weighted, +0.93 in the typical district). A monsoon 20 percent or more short shrinks the next rabi area by about 6 percent, almost entirely in the less irrigated districts.

8. **Neighbouring districts grow much the same crops, and a state border costs about 100 km of likeness.** Adjacent districts in the same state share 77 percent of their crop mix, falling to 63 percent at 200 to 300 km and 48 percent at 500 to 600 km. Adjacent districts across a state line share 67 percent, which same-state pairs reach at about 190 km (border equivalent 109 km, interval 44 to 169). Borders did not pull apart between 1998-2002 and 2015-19. The data show no evidence of districts learning crops from their neighbours, and they cannot rule it out either.

## Superseded earlier results

The first version of this repository (scripts 57 to 65) reported findings that later checks overturned. They are kept in the history and in a few exploratory pages, marked as superseded:

- "Semi-irrigated districts are the most diverse" (ABI 0.69 against 0.62 rainfed and 0.67 irrigated) and the related claims about irrigation regimes rest on a hand-scraped irrigation list of unknown vintage. It agrees with the Mission Antyodaya village survey at a rank correlation of only 0.656 and records zero irrigation for heavily irrigated districts such as Harda. Rebuilt on the survey, the regime pattern disappears.
- The Agro-Biodiversity Index (ABI) and the indices behind it pooled all years, so crop richness counted every crop ever recorded and tracked how long a district was observed (pooled richness averaged 30.4 against 20.6 in a typical year). The ABI is also sample-dependent. It survives only as a comparison row on the Measurement page.
- National and state statistics from that version (357 districts losing diversity, Karnataka ABI 0.85, Punjab 0.44, the ABI-calorie quadrant shares and the ABI-calorie correlation of -0.18) were computed on the pooled index over 1997-98 to 2020-21 and have not been re-estimated.
- An inverted U in irrigation with diversity peaking near a quarter of sown area irrigated was the irrigation headline until November 2026. The squared term survives every check, but the rising side does not, so it was replaced by finding 4.

The static gallery and the ABI-calorie scatter pages on the site carry a banner saying they belong to this earlier version.

## Measures

| Measure | Definition | Site name |
|---|---|---|
| D0 | Crops with positive area in the district-year | Count of crops |
| D1 | exp(Shannon entropy of area shares) | Effective number of crops (main outcome) |
| D2 | 1 / sum of squared area shares (inverse Simpson) | Number of main crops |
| D1 / D0 | Evenness | Evenness |
| D0 at 0.5 percent | Crops holding at least 0.5 percent of the district's gross cropped area | Count of main crops |

Each measure is computed per district-year and then averaged over a district's years. Seasons are summed into annual gross cropped area.

## Data sources

| Source | Use |
|---|---|
| District crop area, production and yield (APY), ISB India Data Portal, from Ministry of Agriculture and Farmers Welfare publications | Crop mix, 1998-99 to 2019-20 (extension to 2022-23 from the DES district file for recent windows) |
| SHRUG 2.1 (Development Data Lab): Mission Antyodaya 2019-20, Census village directories 1991, 2001 and 2011, Census 2011 population abstract, SECC 2012, night lights, Economic Census 2013, terrain | Irrigation, markets, farm structure and controls |
| DES Land Use Statistics, district tables (India Data Portal) | Irrigation and cropping intensity by year, 1998-99 to 2019-20 |
| IMD 0.25-degree gridded daily rainfall | Monsoon shocks |
| NFSM district lists, Odisha Millets Mission documents, MSP series | Policy event studies |
| CGWB well readings | Groundwater depth (instrument checks) |
| IFCT 2017 (NIN Hyderabad) | Calorie conversion factors |
| Census 2011 district boundaries | Maps (735 polygons) |

Full provenance, local paths and rebuild scripts are in section 1 of `outputs/ANALYSIS_INDEX.md`.

## Repository structure

```
scripts/        numbered analysis scripts; run_all.py runs the main pipeline in order
  _years.py     the one place that sets the window (1998-99 to 2019-20)
  57-65         first version (pooled ABI, scraped irrigation), superseded
  70-96         SHRUG covariates, corrected indices, site data
  97-103        crop groups: panel, trends, policy, rainfall shocks, neighbours, referees
  104-112       November 2026 revision: count-of-crops artefact, seasons, irrigation panel, markets by population
  120-133       irrigation v2 (recent windows, 1991 census), peer effects, state validation
outputs/        results notes (*.md), tables and logs; ANALYSIS_INDEX.md is the map
docs/           the live site (GitHub Pages)
deck/, notebooks/, outputs/deliverables/   slides, notebooks, workbook and Word results
```

## Reproducing the analysis

The interpreter on the author's machine is `C:\Users\mridu\anaconda3\envs\geo_env\python.exe` (pandas, numpy 1.26.4, scipy, statsmodels, geopandas). Bare `python` is broken there, so call it by full path.

```bash
C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/run_all.py --check   # verify inputs
C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/run_all.py           # full pipeline
C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/run_all.py --from 98 # resume
```

Three inputs sit outside the repository (the raw APY file, SHRUG 2.1 and the district shapefile), and the rainfall grids, land-use tables, policy PDFs and groundwater readings were fetched by hand. The irrigation v2 and peer-effects chains, and a few one-off referee scripts, run by hand around `run_all.py`. The full run order, ordering traps and runtimes are in section 4 of `outputs/ANALYSIS_INDEX.md`.

## References

1. Hill, M.O. (1973). "Diversity and Evenness: A Unifying Notation and Its Consequences." *Ecology*, 54(2), 427-432.
2. Jost, L. (2006). "Entropy and Diversity." *Oikos*, 113(2), 363-375.
3. Shannon, C.E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27(3), 379-423.
4. Di Falco, S. and Chavas, J.P. (2009). "On Crop Biodiversity, Risk Exposure, and Food Security in the Highlands of Ethiopia." *American Journal of Agricultural Economics*, 91(3), 599-611.
5. Birthal, P.S. et al. (2015). "Crop Diversification and Resilience of Agriculture to Climatic Shocks." *Indian Journal of Agricultural Economics*, 70(1).

## License

This work is independent research by Mridul.
