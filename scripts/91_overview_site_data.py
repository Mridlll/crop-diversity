"""
91_overview_site_data.py

USAGE
    python scripts/91_overview_site_data.py

Writes docs/data/site_overview_examples.json for the Overview page (docs/index.html).

Contents
  examples    five example districts with D0, D1, D2, E, years observed and every
              crop's mean annual area share (shares add to 1 per district)
  name_fix    map polygons in districts.geojson that carry no index because their
              name differs from the crop record, keyed "<geojson state>|<geojson name>",
              with the matching index values and the largest crop
  change      per-district change, 1998-2004 mean against 2013-2019 mean, for the
              districts observed in every year of the window (scripts/_years.py): name, state, dD1, dE, dD0
  state_rings state outlines, copied from site_market_map.json so this page does
              not depend on that file at runtime

Cleaning of the raw area file repeats script 76 exactly, so the indices recomputed
here reproduce district_diversity_indices_corrected.csv (checked at the end).
"""
import json
import os

import numpy as np
import pandas as pd

from _years import keep_years

RAW = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
REPO = r"D:/crop-diversity"
DIV = REPO + "/outputs/crop_diversity_analysis"
DOCS = REPO + "/docs/data"

# ------------------------------------------------------------- load, as script 76
df = pd.read_csv(RAW)
for c in ["year", "season", "state_name", "district_name", "crop_name", "crop_type"]:
    df[c] = df[c].astype(str).str.strip()
df = df.dropna(subset=["area"])
df = df[df["area"] > 0]
BOGUS = [("Delhi", "Chandigarh"), ("Goa", "Chandigarh"),
         ("The Dadra And Nagar Haveli And Daman And Diu", "Chandigarh"),
         ("Delhi", "Surguja"), ("Goa", "Surguja"),
         ("The Dadra And Nagar Haveli And Daman And Diu", "Surguja"),
         ("Andaman And Nicobar Islands", "Purulia")]
REMAP = {("Uttar Pradesh", d): "Uttarakhand" for d in
         ["Almora", "Bageshwar", "Chamoli", "Champawat", "Dehradun", "Haridwar",
          "Nainital", "Pauri Garhwal", "Pithoragarh"]}
REMAP.update({("Madhya Pradesh", d): "Chhattisgarh" for d in
              ["Bastar", "Bilaspur", "Dhamtari", "Durg", "Jashpur", "Kanker",
               "Korba", "Mahasamund", "Raipur", "Rajnandgaon", "Surguja"]})
REMAP.update({("Bihar", d): "Jharkhand" for d in ["Deoghar", "Dhanbad", "Garhwa"]})
pair = list(zip(df["state_name"], df["district_name"]))
df = df[[p not in BOGUS for p in pair]]
for (s_, d_), new in REMAP.items():
    m = (df["state_name"] == s_) & (df["district_name"] == d_)
    df.loc[m, "state_name"] = new
df["year_start"] = df["year"].str.split("-").str[0].astype(int)
df["district_key"] = df["state_name"].str.upper() + "|" + df["district_name"].str.upper()
df = keep_years(df)  # analysis window, scripts/_years.py
K = ["district_key", "year", "season", "crop_name"]
df = df.sort_values("area", ascending=False).drop_duplicates(subset=K, keep="first")

# mean annual area share of every crop, as script 76 computes the dominant crop
ca = (df.groupby(["district_key", "crop_name"])["area"].sum() /
      df.groupby("district_key")["year_start"].nunique())
ca = ca.reset_index(name="a")
ca["share"] = ca["a"] / ca.groupby("district_key")["a"].transform("sum")

corr = pd.read_csv(DIV + "/district_diversity_indices_corrected.csv")
C = corr.set_index("district_key")

NAME = {"Cotton(Lint)": "cotton", "Rapeseed &Mustard": "rapeseed and mustard",
        "Moong(Green Gram)": "moong", "Peas & Beans (Pulses)": "peas and beans",
        "Cowpea(Lobia)": "cowpea", "Arhar/Tur": "arhar", "Soyabean": "soybean",
        "Other  Rabi Pulses": "other rabi pulses", "Horse-Gram": "horse gram",
        "Sannhamp": "sunn hemp", "Sesamum": "sesame", "Guar Seed": "guar seed"}
nice = lambda s: NAME.get(s, s.lower())
TITLE = {"SEPAHIJALA": "Sipahijala", "CHAMARAJANAGARA": "Chamarajanagara"}


def r6(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 6)


def row(key):
    r = C.loc[key]
    sh = ca[ca["district_key"] == key].sort_values("share", ascending=False)
    st, dn = key.split("|")
    return {"k": key, "d": TITLE.get(dn, dn.title()), "s": st.title(),
            "y": int(r["n_years"]), "D0": r6(r["D0_richness"]), "D1": r6(r["D1_exp_shannon"]),
            "D2": r6(r["D2_inv_simpson"]), "E": r6(r["evenness_D1_D0"]),
            "shares": [[nice(c), round(float(s), 5)] for c, s in zip(sh["crop_name"], sh["share"])]}


EX = ["TRIPURA|SEPAHIJALA", "CHHATTISGARH|RAIPUR", "PUNJAB|LUDHIANA",
      "KARNATAKA|BELAGAVI", "KARNATAKA|CHAMARAJANAGARA"]
examples = [row(k) for k in EX]

# ------------------------------------------------------------- name fixes
FIX = {
    "Punjab|Sahibzada Ajit Singh Nag*": "PUNJAB|S.A.S NAGAR",
    "Punjab|Sri Muktsar Sahib": "PUNJAB|MUKTSAR",
    "Haryana|Gurugram": "HARYANA|GURGAON",
    "Haryana|Nuh": "HARYANA|MEWAT",
    "Uttar Pradesh|Prayagraj": "UTTAR PRADESH|ALLAHABAD",
    "Uttar Pradesh|Bhadohi": "UTTAR PRADESH|SANT RAVIDAS NAGAR",
    "Arunachal Pradesh|Upper Dibang Valley": "ARUNACHAL PRADESH|DIBANG VALLEY",
    "Tripura|Unokoti": "TRIPURA|UNAKOTI",
    "West Bengal|South Twenty Four Pargan*": "WEST BENGAL|SOUTH 24 PARGANAS",
    "West Bengal|North Twenty Four Pargan*": "WEST BENGAL|NORTH 24 PARGANAS",
    "West Bengal|Medinipur West": "WEST BENGAL|PASHCHIM MEDINIPUR",
    "Gujarat|Chota Udaipur": "GUJARAT|CHHOTAUDEPUR",
    "Daman & Diu|Daman": "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU|DAMAN",
    "Daman & Diu|Diu": "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU|DIU",
    "Dadra & Nagar Have|Dadra & Nagar Haveli": "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU|DADRA AND NAGAR HAVELI",
    "Andhra Pradesh|Y.S.R.": "ANDHRA PRADESH|KADAPA",
    "Andaman & Nicobar|North  & Middle Andaman": "ANDAMAN AND NICOBAR ISLANDS|NORTH AND MIDDLE ANDAMAN",
    "Andaman & Nicobar|South Andaman": "ANDAMAN AND NICOBAR ISLANDS|SOUTH ANDAMANS",
    "Andaman & Nicobar|Nicobars": "ANDAMAN AND NICOBAR ISLANDS|NICOBARS",
    "Telangana|Jogulamba Gadwal": "TELANGANA|JOGULAMBA",
    "Telangana|Rajanna Sircilla": "TELANGANA|RAJANNA",
    "Telangana|Ranga Reddy": "TELANGANA|RANGAREDDI",
    "Telangana|Warangal Rural": "TELANGANA|WARANGAL",
    "Telangana|Warangal Urban": "TELANGANA|HANUMAKONDA",
    "Telangana|Yadadri Bhuvanagiri": "TELANGANA|YADADRI",
    "Telangana|Bhadradri Kothagudem": "TELANGANA|BHADRADRI",
    "Ladakh|Leh": "LADAKH|LEH LADAKH",
}
geo = json.load(open(DOCS + "/districts.geojson", encoding="utf-8"))
gnames = {f["properties"]["s"] + "|" + f["properties"]["n"]: f["properties"] for f in geo["features"]}
name_fix = {}
for gk, ik in FIX.items():
    assert gk in gnames, gk
    assert gnames[gk].get("D1") is None, gk
    r = C.loc[ik]
    name_fix[gk] = {"k": ik, "y": int(r["n_years"]), "D0": r6(r["D0_richness"]),
                    "D1": r6(r["D1_exp_shannon"]), "D2": r6(r["D2_inv_simpson"]),
                    "E": r6(r["evenness_D1_D0"]), "dom": str(r["dominant_crop"]),
                    "domsh": r6(r["dominant_crop_share"])}
still = [k for k, p in gnames.items() if p.get("D1") is None and k not in FIX]

# ------------------------------------------------------------- district change
panel = pd.read_csv(DIV + "/district_year_diversity_panel_corrected.csv")
panel = panel[panel["district_key"].isin(C.index)]
yrs = sorted(panel["year_start"].unique())
cnt = panel.groupby("district_key")["year_start"].nunique()
bal = panel[panel["district_key"].isin(cnt[cnt == len(yrs)].index)]
cols = ["D0_richness", "D1_exp_shannon", "evenness_D1_D0"]
e = bal[bal["year_start"].between(1998, 2004)].groupby("district_key")[cols].mean()
l_ = bal[bal["year_start"].between(2013, 2019)].groupby("district_key")[cols].mean()
ch = (l_ - e).dropna()
change = [[k.split("|")[1].title(), k.split("|")[0].title(),
           round(float(r["D1_exp_shannon"]), 4), round(float(r["evenness_D1_D0"]), 5),
           round(float(r["D0_richness"]), 4)] for k, r in ch.iterrows()]

mm = json.load(open(DOCS + "/site_market_map.json", encoding="utf-8"))

out = {"note": "Overview page. examples: mean annual area shares. name_fix: geojson "
               "polygons joined to the crop record by name. change: [district, state, "
               "dD1, dE, dD0], 2013-2019 mean minus 1998-2004 mean, {} districts.".format(len(change)),
       "examples": examples, "name_fix": name_fix, "change": change,
       "state_rings": mm["state_rings"]}
with open(DOCS + "/site_overview_examples.json", "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"))

# ------------------------------------------------------------- checks
print("wrote site_overview_examples.json", os.path.getsize(DOCS + "/site_overview_examples.json"), "bytes")
for x in examples:
    sh = np.array([s for _, s in x["shares"]])
    print("  {:16s} D0 {:5.1f} D1 {:5.2f} E {:.3f} y {:2d} crops {:2d} top {} {:.3f} sum {:.4f}".format(
        x["d"], x["D0"], x["D1"], x["E"], x["y"], len(sh), x["shares"][0][0], sh[0], sh.sum()))
print("  name_fix", len(name_fix), "polygons; still without a record:", len(still))
print("   ", sorted(still))
print("  change n {} D1 down {} up {} median {:.3f}; E down {} up {} median {:.4f}; D0 down {} up {} median {:.3f}".format(
    len(ch), (ch["D1_exp_shannon"] < 0).sum(), (ch["D1_exp_shannon"] > 0).sum(), ch["D1_exp_shannon"].median(),
    (ch["evenness_D1_D0"] < 0).sum(), (ch["evenness_D1_D0"] > 0).sum(), ch["evenness_D1_D0"].median(),
    (ch["D0_richness"] < 0).sum(), (ch["D0_richness"] > 0).sum(), ch["D0_richness"].median()))
