"""
run_all.py

Runs the SHRUG covariate and diversity-audit pipeline end to end, in order,
and stops at the first failure.

    python scripts/run_all.py            # full pipeline
    python scripts/run_all.py --from 73  # resume from a step
    python scripts/run_all.py --check    # verify inputs exist, run nothing

Order matters. In particular script 71 validates a column that script 72
creates, so 72 MUST run before 71.

The analysis window, 1998-99 to 2019-20, is set once in _years.py. 1997-98 is
broken in the raw file and every multi-year district average leaves it out.

Requirements: pandas, numpy, scipy, statsmodels, and geopandas for the map steps.
On this machine `geo_env` carries all of them:
  C:/Users/Mridul/anaconda3/envs/geo_env/python.exe
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

STEPS = [
    ("70", "70_shrug_district_covariates.py",
     "Build the district covariate table from SHRUG 2.1"),
    ("72", "72_fix_and_crosswalk.py",
     "Correct the irrigation denominator, build the district crosswalk"),
    ("71", "71_validate_shrug_covariates.py",
     "Nine validation checks (must run AFTER 72)"),
    ("75", "75_audit_diversity_construction.py",
     "Audit how the diversity indices were built, against the raw APY file"),
    ("76", "76_rebuild_corrected_indices.py",
     "Rebuild the indices with the four construction defects fixed"),
    ("73", "73_irrigation_diversity_rebuilt.py",
     "Headline results on the original indices"),
    ("74", "74_robustness.py",
     "Twelve robustness specifications"),
    ("77", "77_final_results_corrected.py",
     "Final results on the corrected indices (the version to quote)"),
    ("79", "79_market_covariates.py",
     "Market, input-supply and connectivity covariates plus development controls"),
    ("80", "80_market_analysis.py",
     "The market layer: mandis, haats, fertiliser shops"),
    ("78", "78_generate_shrug_notebook.py",
     "Regenerate the notebook (does not execute it)"),
    ("81", "81_export_site_data.py",
     "Export every figure's data to docs/data as JSON"),
    ("85", "85_rebuild_map_layers.py",
     "Rebuild every map layer on the corrected basis"),
    ("86", "86_build_map_pages.py",
     "Write the four interactive map pages"),
    ("89", "89_market_typology.py",
     "Market typology on village shares (112 rewrites the site file on population shares)"),
    ("90", "90_market_map_layers.py",
     "Market map layers on village shares (112 rewrites the site file on population shares)"),
    ("91", "91_overview_site_data.py",
     "Overview page: example districts, name fixes, district change"),
    ("92", "92_irrigation_site_data.py",
     "Irrigation page: curves, groups, robustness"),
    ("95a", "95_data_page_export.py",
     "Data and methods page tables"),
    ("95b", "95_measurement_export.py",
     "Measurement page data"),
    ("96", "96_monoculture_baseline.py",
     "Reference scale, monoculture and cropping systems"),
    # Crop groups. Two fetch scripts run once by hand and are not repeated here
    # because they download from the web: 97_imd_district_rainfall.py (IMD grids,
    # about 810 MB into data_raw/) and 97_policy_nfsm_omm_msp.py (NFSM, OMM, MSP).
    ("97", "97_crop_group_adjacency.py",
     "Crop groups: district adjacency on 2011 parents"),
    ("98", "98_crop_group_panel.py",
     "Crop groups: district x year x season panel with defect flags"),
    ("99a", "99_crop_groups_trends.py",
     "Crop groups: trends and the Andhra Pradesh / Telangana rice check"),
    ("99b", "99_crop_groups_trends_referee.py",
     "Crop groups: referee re-check of the trends"),
    ("99c", "99_crop_groups_rainfall_shocks.py",
     "Crop groups: monsoon shocks"),
    ("99d", "99_crop_groups_neighbours.py",
     "Crop groups: neighbour spread"),
    ("99e", "99_crop_groups_policy_event_studies.py",
     "Crop groups: NFSM, Odisha Millets Mission and MSP"),
    ("100a", "100_referee_neighbours_checks.py",
     "Crop groups: referee checks on neighbour spread"),
    ("100b", "100_referee_neighbours_filters.py",
     "Crop groups: referee filters on neighbour spread"),
    ("100c", "100_referee_rainshock_checks.py",
     "Crop groups: referee checks on monsoon shocks"),
    ("101", "101_referee_rainshock_followups.py",
     "Crop groups: referee follow-ups on monsoon shocks"),
    ("102", "102_referee_policy_event_studies.py",
     "Crop groups: referee checks on the policy event studies"),
    ("103", "103_crop_groups_site_export.py",
     "Crop groups: export the page data to docs/data/site_crop_groups.json"),
    # Revision of November 2026: within-district irrigation panel, seasons, the
    # count-of-crops threshold panel and population-weighted market facilities.
    # 104i needs the DES Land Use Statistics extract in outputs/revision_nov8/raw_lus/
    # and 105 reads the rainfall panel from step 99c. One-off referee checks are run
    # by hand and not repeated here: 106 and 107 (irrigation IV feasibility),
    # 108_d0_1997_followups.py and 109_d0_referee_checks.py (the 1997-98 break).
    # Deliverables outside the site, also by hand: 87 (workbook), 88 (results doc).
    ("104i", "104_irrigation_panel_build.py",
     "Irrigation panel: district-year irrigation from Land Use Statistics"),
    ("104s", "104_season_separated_diversity.py",
     "Seasons: diversity within and between seasons (needs 104i and 98)"),
    ("107s", "107_seasons_supplement.py",
     "Seasons: supplementary checks"),
    ("105", "105_irrigation_panel_estimates.py",
     "Irrigation panel: within-district estimates"),
    ("108i", "108_irrigation_referee.py",
     "Irrigation panel: referee checks"),
    ("111", "111_irrigation_panel_site_data.py",
     "Irrigation page: within-district panel data"),
    ("104d", "104_d0_threshold_and_1997.py",
     "Count of crops above share thresholds; the 1997-98 break"),
    ("110", "110_measurement_revision_site_data.py",
     "Measurement page: threshold series and seasons (needs 104d, 104s, 96)"),
    ("104m", "104_market_pop_weighted.py",
     "Markets: facilities as shares of rural population"),
    ("112", "112_market_pop_site_data.py",
     "Markets page on population shares (reruns 89 and 90 on them)"),
    ("83", "83_deck_figures.py",
     "Deck figures in the CEEW palette"),
    ("84", "84_build_deck.py",
     "Build the deck"),
]

# The map and deck steps need geopandas. A run under an interpreter without it
# skips them rather than failing the whole pipeline.
OPTIONAL = {"85": "geopandas", "86": "geopandas", "83": "geopandas", "89": "sklearn",
            "90": "geopandas", "112": "sklearn"}

INPUTS = {
    "SHRUG 2.1 extract": r"D:/SHRUG_2.1_Data/extracted/shrug-antyodaya-dta/antyodaya_shrid.dta",
    "SHRUG location names": r"D:/SHRUG_2.1_Data/extracted/shrug-shrid-keys-dta/shrid_loc_names.dta",
    "raw APY crop file": r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv",
    "published diversity indices": REPO + "/outputs/crop_diversity_analysis/district_diversity_indices.csv",
}


def check_inputs():
    ok = True
    print("Checking inputs\n" + "-" * 60)
    for name, path in INPUTS.items():
        good = os.path.exists(path)
        ok &= good
        size = "{:.1f} MB".format(os.path.getsize(path) / 1e6) if good else "MISSING"
        print("  [{}] {:32s} {}".format("ok" if good else "!!", name, size))
    print()
    for mod in ["pandas", "numpy", "scipy", "statsmodels"]:
        try:
            __import__(mod)
            print("  [ok] {}".format(mod))
        except ImportError:
            print("  [!!] {} NOT INSTALLED".format(mod))
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None,
                    help="step number to resume from, e.g. 73")
    ap.add_argument("--check", action="store_true", help="verify inputs, run nothing")
    a = ap.parse_args()

    if not check_inputs():
        print("\nInputs or packages missing. Fix those before running.")
        return 1
    if a.check:
        print("\nCheck only, nothing run.")
        return 0

    steps = STEPS
    if a.start:
        idx = [i for i, s in enumerate(STEPS) if s[0] == a.start]
        if not idx:
            print("No step " + a.start)
            return 1
        steps = STEPS[idx[0]:]

    print("\nRunning {} steps\n".format(len(steps)) + "=" * 60)
    t0 = time.time()
    for num, script, desc in steps:
        if num in OPTIONAL:
            try:
                __import__(OPTIONAL[num])
            except ImportError:
                print("\n[{}] {}\n     skipped: {} is not installed here. Run this one "
                      "under an environment that has it.".format(
                          num, script, OPTIONAL[num]))
                continue
        print("\n[{}] {}\n     {}".format(num, script, desc))
        t = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, script)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("     FAILED after {:.0f}s".format(time.time() - t))
            print("\n".join(r.stdout.splitlines()[-10:]))
            print("\n".join(r.stderr.splitlines()[-20:]))
            return 1
        print("     ok, {:.0f}s".format(time.time() - t))

    print("\n" + "=" * 60)
    print("All steps completed in {:.0f}s.".format(time.time() - t0))
    print("\nRead, in this order:")
    for f in ["outputs/shrug_covariates/diversity_construction_audit.md",
              "outputs/shrug_covariates/corrected_vs_original.md",
              "outputs/shrug_covariates/validation_report.md",
              "outputs/shrug_covariates/final_results.md",
              "outputs/shrug_covariates/market_analysis.md"]:
        print("  " + f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
