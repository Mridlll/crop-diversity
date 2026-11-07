"""
133_validate_state_irrigation.py

Sourced state reference values for checks C5 and C6 of the SHRUG validation
report (script 71). Until November 2026 those checks compared the Mission
Antyodaya 2019-20 state figures with a reference column typed in from memory.
This script replaces that column with published DES figures.

Source: DES Land Use Statistics, district-wise tables "Sources of Irrigation"
and "Classification of Area" (India Data Portal CKAN package
`land-use-statistics`), the same files script 104_irrigation_panel_build.py
reads. State totals are the sum of the state's districts, which is how DES
builds its state tables.

  net irrigated area  = sum of the six "Net Irrigated Area" source rows
  irrigation share    = net irrigated area / net area sown   (NIA / NSA)
  cropping intensity  = cropped area / net area sown         (GCA / NSA)

Window: 2017-18 to 2019-20, pooled (sum over district-years with both values),
which brackets the 2019-20 Antyodaya survey. Andhra Pradesh and Telangana are
also given combined, because the SHRUG table uses undivided 2011 Andhra Pradesh.
Coverage is reported as the share of the state's net sown area in district-years
that also report net irrigated area; under 50 percent is flagged `thin`.
Maharashtra and West Bengal report no district net irrigated area in any year of
the window, so they have no irrigation reference. Shares above 1 are set missing.

Input  (ignored by git, fetched by hand):
  outputs/revision_nov8/raw_lus/sources-of-irrigation.csv
  outputs/revision_nov8/raw_lus/classification-of-area.csv
Fallback when the raw files are absent: outputs/revision_nov8/irrigation_panel_district_year.csv
  (crop-record districts only, so coverage is lower).
Output (committed): outputs/shrug_covariates/state_irrigation_reference_lus.csv

Run:  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/133_validate_state_irrigation.py
Then rerun scripts/71_validate_shrug_covariates.py, which reads the output.
"""
import os
import pandas as pd

REPO = r"D:/crop-diversity"
RAW = REPO + "/outputs/revision_nov8/raw_lus"
PANEL = REPO + "/outputs/revision_nov8/irrigation_panel_district_year.csv"
OUT = REPO + "/outputs/shrug_covariates/state_irrigation_reference_lus.csv"
YEARS = ["2017-2018", "2018-2019", "2019-2020"]


def from_raw():
    s = pd.read_csv(RAW + "/sources-of-irrigation.csv")
    c = pd.read_csv(RAW + "/classification-of-area.csv")
    s = s[s["year"].isin(YEARS) & (s["irrigated_area_type"] == "Net Irrigated Area")]
    c = c[c["year"].isin(YEARS)]
    idx = ["year", "state_name", "district_name", "district_code"]
    nia = s.groupby(idx, dropna=False)["irrigated_area"].sum(min_count=1).rename("nia")
    m = c[c["land_use_metrics"].isin(["Net Area Sown", "Cropped Area"])]
    m = m.pivot_table(index=idx, columns="land_use_metrics", values="area",
                      aggfunc="sum").rename(columns={"Net Area Sown": "nsa",
                                                     "Cropped Area": "gca"})
    d = m.join(nia, how="left").reset_index()
    d["state"] = d["state_name"].str.lower().str.strip()
    return d, "DES Land Use Statistics, raw district tables"


def from_panel():
    p = pd.read_csv(PANEL)
    p = p[p["year_start"].between(2017, 2019)]
    d = p[["state_name", "district_key", "nia", "nsa", "gca"]].copy()
    d["state"] = d["state_name"].str.lower().str.strip()
    return d, "irrigation_panel_district_year.csv (crop-record districts)"


def summarise(d):
    d = d[d["nsa"] > 0].copy()
    both = d["nia"].notna()
    g = d.groupby("state").agg(nsa_all=("nsa", "sum"))
    g["nia"] = d[both].groupby("state")["nia"].sum()
    g["nsa_irr"] = d[both].groupby("state")["nsa"].sum()
    gc = d[d["gca"] > 0]
    g["gca"] = gc.groupby("state")["gca"].sum()
    g["nsa_ci"] = gc.groupby("state")["nsa"].sum()
    g["district_years_irr"] = d[both].groupby("state").size()
    g = g.fillna({"district_years_irr": 0})
    g["lus_irr_share"] = g["nia"] / g["nsa_irr"]
    g["lus_cropping_intensity"] = g["gca"] / g["nsa_ci"]
    g["irr_coverage_of_nsa"] = (g["nsa_irr"] / g["nsa_all"]).fillna(0.0)
    g["thin"] = g["irr_coverage_of_nsa"] < 0.5
    # a share above 1 means the district rows do not line up (Puducherry): unusable
    g.loc[g["lus_irr_share"] > 1.0, "lus_irr_share"] = float("nan")
    return g


d, src = from_raw() if os.path.exists(RAW + "/sources-of-irrigation.csv") else from_panel()
g = summarise(d)
ap = d[d["state"].isin(["andhra pradesh", "telangana"])].copy()
ap["state"] = "andhra pradesh (undivided)"
g = pd.concat([g, summarise(ap)])
tot = d.copy()
tot["state"] = "india"
g = pd.concat([g, summarise(tot)])
g["source"] = src
g["window"] = "2017-18 to 2019-20"
g.index.name = "state"
g = g.round({"lus_irr_share": 4, "lus_cropping_intensity": 4, "irr_coverage_of_nsa": 3})
g.to_csv(OUT)

print("Source:", src)
print(g[["lus_irr_share", "lus_cropping_intensity", "irr_coverage_of_nsa",
         "district_years_irr", "thin"]].to_string())
print("Wrote", OUT)
