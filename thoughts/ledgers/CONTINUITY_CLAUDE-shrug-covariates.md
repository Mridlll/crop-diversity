# CONTINUITY: crop diversity, SHRUG covariates, site and deck

Filename kept for continuity. Scope grew well past the original title: it now covers
the diversity indices themselves, the public site, and the CEEW deck.

## Goal
Explain what Indian district crop diversity is arranged around, using SHRUG 2.1 for the
right-hand side, and publish it as a site and a deck.

Status: COMPLETE and pushed. Site live at https://mridlll.github.io/crop-diversity/
Deck at `deck/crop_diversity.pptx`, 15 slides. Everything regenerates from
`python scripts/run_all.py` (16 steps, about 155 seconds).

## Environment (CHANGED 2026-09-01, older memory is out of date)
`C:\Users\Mridul\anaconda3\envs\geo_env\python.exe` now runs EVERYTHING.
statsmodels 0.14.6 and nbformat added; numpy 1.26.4, scipy 1.13.1 and geopandas 1.0.1
deliberately left untouched and verified afterwards.
A `sitecustomize.py` in that env's site-packages sets GDAL_DATA and PROJ_LIB, because
conda only sets them on `conda activate` and we call the interpreter by full path.
The old split (geo_env for geometry, "CEEW Assignment" for statsmodels) is GONE.

## Key decisions, and why
- **Irrigation share is irr / (irr + unirr)**, not irr / gross cropped area. The first
  attempt used GCA and was wrong: `area_irrigated_in_hac` is NET irrigated area.
  Evidence: irr+unirr reconstructs net sown (1.15) not gross (0.66); against published
  state figures MAE 0.086 and bias +0.051, versus 0.181 and -0.129 for the GCA version.
  Wrong columns kept as `DEPRECATED_*`.
- **SHRUG levels are unusable here, only ratios.** `antyodaya_shrid.dta` covers 857m
  people against DDL's own district file at 1,007m. Ratios reproduce at r = 0.992-0.996.
- **Diversity is measured within a year then averaged**, never pooled. Pooling made
  richness scale with how long a district was observed (r = 0.545 against 0.309).
- **Hill numbers, not a composite.** D0 count, D1 exp(Shannon), D2 inverse Simpson,
  evenness D1/D0. All in units of crops. D0 >= D1 >= D2 is a free correctness check.
  The ABI is sample-dependent, arbitrarily weighted, and two of its three components
  correlate at 0.94. It survives only as one comparison row.
- **Telangana does not exist in SHRUG.** All 33 districts trace to 10 pre-2014 parents
  filed under Andhra Pradesh. 162 diversity districts share a parent, so every estimate
  clusters on the parent and is re-run dropping shared parents.
- Coconut production is a COUNT OF NUTS mislabelled as tonnes. Convert at 0.00015 t/nut
  or kcal per hectare comes out two orders of magnitude too high. Also drop non-coconut
  rows above 200 t/ha.
- Food crop is defined by crop TYPE, not by carrying an energy value. All 54 crops have
  an energy value, so the latter makes every district 100 percent food.

## State: all phases done
- [x] SHRUG covariate table: 631 districts, 125 cols, 93 documented variables
- [x] Crosswalk: 722 of 725 matched (99.6%); misses are Mumbai, Daman, Diu
- [x] Validation battery: 6 PASS, 3 WARN, 0 FAIL
- [x] Audit of the diversity indices against raw APY, then corrected rebuild
- [x] Irrigation layer, market layer, robustness
- [x] Site: overview, irrigation, markets, data and methods, four rebuilt maps
- [x] Deck: 15 slides, CEEW house style, every slide exported and read
- [x] Environment consolidated

## FINDINGS (quote these, from final_results.md and market_analysis.md)
- National: D0 21.2 crops grown, D1 4.9 effective, D2 3.5, evenness 0.254.
- **Inverted U in irrigation holds in 8 of 8 sample specs at p < 0.0001**, and in all 6
  alternative indices. Turning point 0.245 to 0.513 on D1, median 0.271. It moves with
  the index, so quote a RANGE (about a third to a half), never a point.
- Crop richness showed NO hump on the pooled construction (p = 0.19) and does show one
  corrected (p = 0.0001). That null was a construction artefact.
- **Irrigation source**: surface water -2.03 effective crops (p = 0.026), cereal share
  +0.300 (p = 0.0001); canal -2.59 crops grown (p = 0.003) but leaves the effective
  number alone and raises pulse share +0.073.
- **Nine of ten rural facilities correlate 0.24 to 0.83 with each other. The weekly haat
  does not** (-0.21 regular market, -0.15 mandi, +0.04 nightlights, -0.16 non-farm
  density). This is the structural finding the market layer rests on, and it is now its
  own deck page.
- **Haat raises D0 (+7.33, p<0.001) but NOT D1 (+0.49, p=0.56)**, evenness slightly
  negative. Crops added at the margin around a staple that stays dominant. One SD of
  haat share is 0.117, so about +0.9 crops on a mean of 21. Always quote the
  standardised magnitude and say which index it is on.
- **Producer organisations are the mirror**: D1 up without D0 up, cereal share -0.222.
- Fertiliser shops, cold storage and regular markets each cut pulse share at p<0.01;
  fertiliser shop survives BH across all ten crop categories.
- Trend: national area-weighted series FLAT across two decades. District level, 264 lost
  effective diversity against 165 gained (balanced panel, 429 districts, 1998-2004
  against 2013-2019).
- **REJECTED, and on the deck as such**: mandi density does NOT raise cereal share (it
  is -0.240, the wrong sign); and market or input density does NOT steepen the irrigation
  downslope (all five interactions POSITIVE, only mandi near significance at p = 0.033).
- Mean holding size is null on everything. Read as much a comment on the proxy as a
  result.

## Site conventions (agreed with the user, do not drift)
- One continuous dark ground #0B0B0C. Depth from rgba-white translucency. NO alternating
  light and dark section blocks.
- NO coral, salmon or orange anywhere. Palette cyan #0E9BB5, violet #7C5CE0,
  green #1FA35A, magenta #C63FD8, validated on the dark surface.
- NO headline-number callouts. Academic register, reasoning in prose, numbers in figures.
- NO documenting our own process errors on the site. The ledger keeps them.
- NO .md, script or file names, no "this page", no "the pages after". Self-contained.
- Maps are embedded IN the narrative, never a footer list.
- Wide reading column with prose in two columns (`prose--flow`), not a narrow pillar.
- Assets are content-hash versioned. Script 86 stamps them; do not regenerate pages
  without that or the cache busting silently drops.

## Deck conventions
- Built on the CEEW furniture at `D:\Alternative Proteins\demand_pathways\deck`.
  IMPORT it, never modify it. The table helper lives in our own builder.
- `check_fits` and the panel helpers RAISE on overflow. Shorten copy, do not grow boxes.
- Writing rules from `D:\Alternative Proteins\.claude\skills\writing-style`: titles are
  labels, headlines state one finding flatly in one clause, no antithesis, no corrective
  negation, no contrasting pairs, no em dashes, no operational vocabulary.
- ONE name per concept: "the effective number of crops" everywhere.
- The house palette keeps orange and blue apart as adjacent categories. Use blue+green.
- NEVER a dual y axis. Split into stacked panels.
- Counts and shares never share an x axis. Separate panels.
- QA by LOOKING: run `deck/export_qa.ps1` then read every PNG. A build that succeeds is
  not a deck that reads; the adversarial read caught a slide whose headline asserted the
  finding its own verdict badge refuted.

## Open questions
- UNCONFIRMED: the published state benchmarks in script 71 (REF_CI, REF_IRR) are entered
  from memory of DES Land Use Statistics. Rank agreement only. Re-check before any of it
  is published.
- Mission Antyodaya over-reports irrigation in Maharashtra (0.54 against 0.21 published),
  Jharkhand (0.45 against 0.12) and Assam (0.35 against 0.11). Dropping them does not
  change the result; their levels are not quotable.
- Antyodaya is a single 2019-20 cross-section against cropping averaged over two decades.
  Fine for slow-moving infrastructure, not for anything called a change.

## Not done
- 24 should-fix and cosmetic QA items on the deck: slides 8, 9 and 10 are three
  consecutive full-width tables; number alignment on starred values; two date formats.
- `docs/measurement.html` is ORPHANED (content folded into data.html), unlinked, still on
  disk. Delete pending a decision.
- District MSP procurement volumes are not published at district level anywhere. The
  natural companion to the market results, and absent.
- Operational holdings from Ag Census 2015-16 would replace the inferred holding proxy.

## SEEDNET reconnaissance (site down, checked 2026-08-31)
- `https://seednet.gov.in/` serves an "under maintenance" page; every real path 404s.
  Plain `http://` fails at the connection level, which looks like a different problem.
- Wayback has 1,433 URLs at HTTP 200: 848 PDF, 129 Excel, 52 Word, 210 HTML. Dense to
  2022, thin after (55 snapshots in 2024, 8 in 2025).
- The Excel files are BREEDER SEED INDENT AND ALLOCATION by crop, variety and season,
  roughly 2010 to 2017. `Material/Agriculture_Variety.htm` holds notified varieties by
  crop since 1994 (paddy 197, wheat 66, barley 24).
- data.gov.in has "List of field crops varieties and hybrids released and notified",
  last updated September 2014.
- Nobody has scraped it. Not on GitHub, not on Kaggle.
- Why it matters here: variety counts per crop sit upstream of the market layer. If
  paddy has 197 notified varieties and a millet has three, the choice set is fixed long
  before a mandi enters the picture.

## Working set
Repo: `D:\crop-diversity` (branch main, all pushed).
Raw APY: `E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv`
Shapefile: `E:/CEEW Project/Package_Maps_Share_20251120_FINAL/shapefiles/in_district.shp`
SHRUG: `D:\SHRUG_2.1_Data\extracted` (join key is `shrid2`, never `shrid`)

Run order (all in `run_all.py`; `--check` verifies inputs, `--from NN` resumes):
    70 covariates -> 72 corrections and crosswalk -> 71 validation (MUST follow 72)
    75 index audit -> 76 corrected indices
    73 results on original -> 74 robustness -> 77 FINAL results (quote these)
    79 market covariates -> 80 market layer
    78 notebook -> 81 site JSON -> 85 map layers -> 86 map pages
    83 deck figures -> 84 deck

Reports, in reading order:
    diversity_construction_audit.md -> corrected_vs_original.md
    -> validation_report.md -> final_results.md -> market_analysis.md

Notebook: `notebooks/shrug_covariates_analysis.ipynb`, executed with outputs.
Deck QA: `deck/export_qa.ps1` writes `deck/qa_png/` (gitignored).
