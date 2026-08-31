# CONTINUITY: SHRUG covariates for crop diversity

## Goal
Replace the scraped 503-district irrigation variable in `district_diversity_indices.csv`
with a SHRUG-built district covariate table, then re-test the repo's headline finding
(semi-irrigated districts most diverse) and decompose irrigation by source.

Status: phases 0-7 DONE. Pipeline reproducible via `scripts/run_all.py` (89s).
Scope grew: the diversity indices themselves were audited and rebuilt, because
asking whether the covariates were right made it necessary to ask the same of
the outcome. Remaining: README corrections and commit.

## Constraints
- SHRUG 2.1 at `D:\SHRUG_2.1_Data\extracted`. Join key inside SHRUG is `shrid2`, never `shrid`.
- Python: scripts 70 and 72 run anywhere; 71, 73, 74 need statsmodels/scipy, so use
  `C:\Users\Mridul\anaconda3\envs\CEEW Assignment\python.exe`. `geo_env` has NO statsmodels.
- SHRUG has ZERO crop-area data. Diversity outcomes stay district-level from the APY panel.
- No variable enters the table without a provenance row (93 documented in provenance.csv).
- Every merge asserts row counts and reports both-side non-matches.

## Key Decisions
- **Irrigation share is irr / (irr + unirr), NOT irr / gross cropped area.**
  The first attempt used GCA and was wrong. `area_irrigated_in_hac` is NET irrigated
  area. Evidence: irr+unirr reconstructs net sown (ratio 1.15) not gross (0.66); against
  published state figures irr/(irr+unirr) gives MAE 0.086 and bias +0.051 while
  irr/GCA gives MAE 0.181 and bias -0.129. The wrong columns are kept as
  `DEPRECATED_*` for audit.
- Levels from `antyodaya_shrid.dta` are NOT comparable to DDL's own district file
  (857m vs 1,007m population, a 22% village-coverage gap). Only RATIOS are, and those
  reproduce at r = 0.992 to 0.996. Never use SHRUG-derived levels, only ratios.
- Telangana is absent from SHRUG: all 33 of its current districts map to 10 pre-2014
  parents registered under Andhra Pradesh. 162 diversity districts share a parent,
  handled by inverse-parent weighting and SEs clustered on the parent.
- Kerala, Tripura and 24 other districts are too thin in Mission Antyodaya and are
  excluded via `flag_low_ay_coverage`, not silently averaged in.
- Fuzzy name matching auto-accepts at 0.86 and logs everything above 0.75 for review.
  Four false positives were caught by eye and are hard-blocked in `REJECT_FUZZY`
  (Pauri Garhwal, Tirupathur, North Garo Hills, South West Garo Hills).

## State
- Done:
  - [x] Phase 0: Scope, source verification, repo cloned to D:\crop-diversity
  - [x] Phase 1: District covariate table, 631 districts, 123 cols, 93 documented vars
  - [x] Phase 2: Crosswalk, 722 of 725 matched (99.6%), 30 states
  - [x] Phase 3: Validation battery, 6 PASS / 3 WARN / 0 FAIL
  - [x] Phase 4: Headline finding re-tested, survives
  - [x] Phase 5: Irrigation-source decomposition and the other four dimensions
  - [x] Phase 6: AUDIT of the diversity indices themselves (script 75). Four
        construction defects found, three real. Corrected rebuild in script 76,
        final results on corrected indices in script 77.
  - [x] Phase 7: Reproducibility. `scripts/run_all.py` runs all 8 steps in 89s
        with an input/package precheck. Notebook at
        `notebooks/shrug_covariates_analysis.ipynb`, 48 cells, 5 figures,
        executed with outputs, zero errors.
  - [x] Phase 8: MARKET LAYER (scripts 79, 80). Mandi / haat / fertiliser shop /
        input supply / post-harvest, with nightlights + Economic Census + connectivity
        as development controls. Six sections including a collinearity and
        one-at-a-time robustness block.
  - [x] Phase 9: SITE. docs/ rebuilt as a narrative: index (what agrobiodiversity
        looks like) -> irrigation -> markets -> data & methods. Dark ground with
        translucent layers, validated 4-colour palette (cyan/violet/green/magenta,
        no warm tones). Native SVG charts + district choropleths from
        docs/data/districts.geojson. Scripts 81 (JSON export) and 82 (geojson).
- Now: [->] Phase 10: README update and commit (use the /commit skill)
- Remaining:
  - [ ] Fix the repo's own "24 years / 1997-2021" claims to "23 years / 1997-2020"
  - [ ] Decide whether the docs/ maps and hover pages get regenerated on corrected
        indices, or stay as-is with a note

## Diversity construction defects found (script 75)
Checked against the raw APY file at `E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv`.

- CLEARED, not a defect: summing seasons does NOT double count. Only 441
  crop-district-years (0.64% of Whole Year cells) appear under both Whole Year and a
  named season, and only 7.7% of those look like a genuine total. Reconstructed
  national gross cropped area is 182-195m ha against a published 195-200m.
- DEFECT 1: 2020-21 is a stub. 319 rows, 13 districts, 0.9m ha, against 19,256 rows
  and 194.9m ha in 2019-20. The repo claims "24 years / 1997-2021" everywhere. It is
  23 usable years, 1997-98 to 2019-20.
- DEFECT 2: 91 exact duplicates on (district, year, season, crop) survive script 57's
  bogus-pair cleaning, mostly Niger Seed in Andhra Pradesh, and get summed.
- DEFECT 3 (the serious one): script 57 groups by district ONLY, so `crop_richness`
  counts crops grown at any point in the period. That is 1.49x mean annual richness and
  correlates with a district's years of coverage at r = 0.545 against r = 0.309 for
  annual richness. 27% of districts have under 20 years. Richness is 1/3 of the ABI.
- DEFECT 4: Shannon and Simpson also pooled. Much less serious, rank correlation 0.97
  against annual means.
- Effect of fixing: ABI correlates r = 0.95 with the original, so the map is not
  overturned, but 95 of 725 districts move more than 100 rank places and Jharkhand
  falls from 18th among states to 25th. Karnataka stays 1st.
- IMPORTANT: the corrected measure STRENGTHENS the headline result. Crop richness
  showed no hump on pooled indices (p = 0.19) and does show one on corrected
  (p = 0.0001). The defect was hiding a real result, not manufacturing one.

## Findings
Quote the CORRECTED numbers (script 77 / final_results.md), not the originals below.

- FINAL, on corrected indices: inverted U holds in 8 of 8 composite specifications, all
  at p < 0.0001, turning points 0.345 to 0.598, median 0.367. Crop richness shows the
  hump too (p = 0.0001).
- SUPERSEDED, on the original pooled indices: 9 of 9 at p < 0.05, turning points 0.295
  to 0.600, and richness showed NO hump (p = 0.19). That null was a pooling artefact.
- The scraped `irrigation_pct` agrees with the SHRUG measure at only rho = 0.656.
- Surface-water dependence is strongly negative for diversity (corrected: ABI -0.168,
  Shannon -0.543, both p < 0.01) and raises cereal share (+0.300). Canal vs groundwater
  cuts crop richness (-2.59 crops, p < 0.01) and ABI slightly, but not Shannon.
- Regular-market access is negative for diversity (-0.112, p < 0.05); FPO presence is
  positive (+0.122, p < 0.05). Mandi presence is null once state and irrigation are in.
- SC population share is positive and holds up (+0.207, p < 0.05). ST share is null
  once state and irrigation are controlled.
- Mean holding size is null. The inverse farm-size-diversity relationship does not
  appear here, which is as much a warning about the proxy as a result.

## Market layer findings (script 80 / market_analysis.md)
Robust:
- The weekly haat is the ONLY facility that does not load on the common
  agri-commercial infrastructure factor. It correlates NEGATIVELY with mandi (-0.15),
  regular market (-0.21), custom hiring (-0.09) and non-farm establishment density
  (-0.16), and ~0 with nightlights. Everything else correlates 0.4-0.83 with everything.
- Haat -> crop richness +7.3 (p<0.01), stable +6.9 to +8.0 across every spec including
  one carrying all nine other facilities. BUT the honest magnitude is small: 1 SD of
  haat share (0.117) buys ~0.9 crops on a mean of 21. Always quote the standardised
  version, never the raw coefficient alone.
- Commercial/input infrastructure displaces PULSES: fert shop -0.174, cold storage
  -0.172, regular market -0.094, all p<0.01; fert shop survives BH across 10 categories.
- FPO is the one facility positive on ABI (+0.105 solo, +0.123 joint) and negative on
  cereal share (-0.222).

Rejected, and written into the report as such:
- Mandi does NOT push cereals. Coefficient is -0.240, the opposite sign to the hypothesis.
- Infrastructure does NOT steepen the irrigation downslope. All five interactions are
  POSITIVE (flatter where dense), only mandi approaches significance (p=0.059).
  The Punjab procurement story does not appear in this cross-section.
- The 2x2 market typology has large raw gaps (ABI 0.53 to 0.68) that all vanish once
  controls go in. Three null adjusted contrasts.

Caveats: single 2019-20 cross-section; infrastructure is not randomly placed and
causality plausibly runs both ways; input-supply block VIFs are 3.4-4.1 so read the
joint coefficients alongside the one-at-a-time table (output block is clean, VIF < 2).
No Economic Census industry split exists at district level in SHRUG 2.1, so there is
no way to count actual fertiliser dealers or agri-wholesalers; the village dummies
are what we have.

## Site conventions (agreed with the user, do not drift)
- NO alternating light/dark section blocks. One continuous dark ground (#0B0B0C),
  depth from rgba-white translucency only.
- NO coral/salmon/orange anywhere. Palette is cyan #0E9BB5, violet #7C5CE0,
  green #1FA35A, magenta #C63FD8, validated on the dark surface.
- NO headline-number callouts. Academic register, reasoning worked through in prose,
  numbers live in the figures.
- NO documenting our own process errors on the site. Normal research iteration is not
  the story; the findings are. (The ledger keeps them, the site does not.)
- NO references to .md/audit/script filenames, and no "this page"/"the pages after".
  The narrative is self-contained.
- Maps are embedded IN the narrative sections, not parked in a footer list.
- docs/measurement.html is now ORPHANED (folded into data.html). Left on disk
  pending a decision, not linked from anywhere.

## Open Questions
- UNCONFIRMED: the published state benchmarks in script 71 (REF_CI, REF_IRR) are
  entered from memory of DES Land Use Statistics. Re-check against the published
  tables before any of it goes in a report. Used for rank agreement only.
- UNCONFIRMED: Mission Antyodaya over-reports irrigation in Maharashtra (0.54 vs
  published 0.21), Jharkhand (0.45 vs 0.12) and Assam (0.35 vs 0.11). Dropping those
  three states does not change the result (spec 7), but the levels there are not
  trustworthy.
- C5 cropping intensity only reaches rho = 0.653 against published state figures.
  Good enough to use as a control, not good enough to report as a finding.

## Working Set
- Repo: D:\crop-diversity (branch main, nothing committed yet)
- Scripts, run in this order:
    70_shrug_district_covariates.py    build the table from SHRUG
    72_fix_and_crosswalk.py            corrections + crosswalk + analysis_panel.csv
    71_validate_shrug_covariates.py    the 9 checks (must run AFTER 72)
    75_audit_diversity_construction.py audit the indices against raw APY
    76_rebuild_corrected_indices.py    corrected indices
    73_irrigation_diversity_rebuilt.py results on ORIGINAL indices
    74_robustness.py                   12 robustness specs
    77_final_results_corrected.py      FINAL results, quote these
    79_market_covariates.py            market + development controls
    80_market_analysis.py              the market layer
    78_generate_shrug_notebook.py      builds the notebook
  Or just: python scripts/run_all.py   (add --check to verify inputs only)
- Notebook: notebooks/shrug_covariates_analysis.ipynb (48 cells, 5 figures, executed)
- Reports, read in this order:
    diversity_construction_audit.md -> corrected_vs_original.md
    -> validation_report.md -> final_results.md
- Data: district_diversity_indices_corrected.csv (725), final_panel.csv,
  shrug_district_covariates.csv (631 x 125), provenance.csv, district_crosswalk.csv
