"""
run_all.py

Runs the SHRUG covariate and diversity-audit pipeline end to end, in order,
and stops at the first failure.

    python scripts/run_all.py            # full pipeline
    python scripts/run_all.py --from 73  # resume from a step
    python scripts/run_all.py --check    # verify inputs exist, run nothing

Order matters. In particular script 71 validates a column that script 72
creates, so 72 MUST run before 71.

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
    ("83", "83_deck_figures.py",
     "Deck figures in the CEEW palette"),
    ("84", "84_build_deck.py",
     "Build the deck"),
]

# The map and deck steps need geopandas. A run under an interpreter without it
# skips them rather than failing the whole pipeline.
OPTIONAL = {"85": "geopandas", "86": "geopandas", "83": "geopandas"}

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
