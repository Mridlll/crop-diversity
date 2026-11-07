"""
104_irrigation_panel_build.py

Builds a district-year irrigation panel, 1998-99 to 2019-20, and joins it to the
district-year crop diversity panel built from the same APY file the site uses.

Irrigation source
  DES Land Use Statistics (LUS), district-wise, as redistributed by the India Data
  Portal (CKAN package `land-use-statistics`, resources "Sources of Irrigation"
  and "Classification of Area", retrieved 2026-11-07 into
  outputs/revision_nov8/raw_lus/). Net irrigated area by source (government canal,
  private canal, tank, tubewell, other well, other source), gross irrigated area,
  net area sown and gross cropped area. Years 1998-99 to 2023-24.

Irrigation share
  NIA / NSA, net irrigated over net sown. Same concept as the cross-section's
  Antyodaya irr / (irr + unirr), which reconstructs net sown (script 72).
  GIA / GCA kept as an alternative.

Diversity
  Rebuilt per district-year from the raw APY file with script 76's cleaning
  (bogus pairs, state remaps, 2020-21 stub dropped, duplicates collapsed by max).
  Adds D0_05: count of crops holding at least 0.5 percent of the district-year's
  cropped area.

Districts
  APY district_key is matched to LUS (state, district) by normalised name within
  state, then fuzzy within state (difflib >= 0.85, logged). The 2011 parent
  (shrug_key) comes from outputs/shrug_covariates/district_crosswalk.csv, and the
  606-district analysis sample from final_panel.csv `in_final`.

Outputs (outputs/revision_nov8/)
  irrigation_panel_district_year.csv   one row per APY district-year
  irrigation_panel_name_match.csv      APY -> LUS name match log
  irrigation_panel_build_log.txt
"""
import difflib
import re

import numpy as np
import pandas as pd

RAW = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
REPO = r"D:/crop-diversity"
OUT = REPO + "/outputs/revision_nov8"
LUS = OUT + "/raw_lus"
SH = REPO + "/outputs/shrug_covariates"
LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


# ================================================================ APY diversity
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
df = df[df["year_start"] <= 2019]
df["district_key"] = df["state_name"].str.upper() + "|" + df["district_name"].str.upper()
K = ["district_key", "year", "season", "crop_name"]
df = df.sort_values("area", ascending=False).drop_duplicates(subset=K, keep="first")

dy = df.groupby(["district_key", "year_start", "crop_name"])["area"].sum().reset_index()
dy["p"] = dy["area"] / dy.groupby(["district_key", "year_start"])["area"].transform("sum")
dy["plogp"] = -dy["p"] * np.log(dy["p"])
dy["p2"] = dy["p"] ** 2
dy["ge05"] = (dy["p"] >= 0.005).astype(int)
dy["cereal"] = 0.0
ann = dy.groupby(["district_key", "year_start"]).agg(
    D0=("crop_name", "size"), D0_05=("ge05", "sum"), H=("plogp", "sum"),
    S=("p2", "sum"), apy_area=("area", "sum")).reset_index()
ann["D1"] = np.exp(ann["H"])
ann["D2"] = 1 / ann["S"]
ann["even"] = ann["D1"] / ann["D0"]
# cereal share of cropped area, crop TYPE based
ct = df.groupby(["district_key", "year_start", "crop_type"])["area"].sum().unstack(fill_value=0)
ct = ct.div(ct.sum(axis=1), axis=0)
ann = ann.merge(ct[["Cereals"]].rename(columns={"Cereals": "cereal_share"}).reset_index(),
                on=["district_key", "year_start"], how="left")
ann["state_name"] = ann["district_key"].str.split("|").str[0]
w("APY district-years: {:,}, districts {}".format(len(ann), ann["district_key"].nunique()))
w("Check D0>=D1>=D2 violations: {}".format(
    int(((ann.D0 < ann.D1 - 1e-9) | (ann.D1 < ann.D2 - 1e-9)).sum())))

# ================================================================ LUS irrigation
s = pd.read_csv(LUS + "/sources-of-irrigation.csv")
c = pd.read_csv(LUS + "/classification-of-area.csv")
for t in (s, c):
    t["year_start"] = t["year"].str[:4].astype(int)
    t["state_name"] = t["state_name"].str.strip()
    t["district_name"] = t["district_name"].astype(str).str.strip()
SRC = {("Canal", "Government"): "canal", ("Canal", "Private"): "canal",
       ("Tank", "Tank"): "tank", ("Well", "Tubewell"): "tubewell",
       ("Well", "Other Well"): "otherwell", ("Other Source", "Other Source"): "other"}
s["src"] = [SRC.get((a, b)) for a, b in zip(s["area_classification"], s["area_sub_classification"])]
s["typ"] = s["irrigated_area_type"].map({"Net Irrigated Area": "nia", "Gross Irrigated Area": "gia"})
idx = ["state_name", "district_name", "year_start"]
tot = (s.groupby(idx + ["typ"])["irrigated_area"].sum(min_count=1).unstack("typ"))
bysrc = (s[s["typ"] == "nia"].groupby(idx + ["src"])["irrigated_area"]
         .sum(min_count=1).unstack("src").add_prefix("nia_"))
lu = (c[c["land_use_metrics"].isin(["Net Area Sown", "Cropped Area", "Reporting Area"])]
      .groupby(idx + ["land_use_metrics"])["area"].sum(min_count=1).unstack("land_use_metrics")
      .rename(columns={"Net Area Sown": "nsa", "Cropped Area": "gca",
                       "Reporting Area": "reporting_area"}))
L = tot.join(bysrc, how="outer").join(lu, how="outer").reset_index()
L = L[(L["year_start"] >= 1997) & (L["year_start"] <= 2019)]
L["irr_nsa"] = L["nia"] / L["nsa"]
L["irr_gca"] = L["gia"] / L["gca"]
w("LUS district-years 1998-2019: {:,}; with NIA and NSA: {:,}".format(
    len(L), int((L["nia"].notna() & (L["nsa"] > 0)).sum())))
w("LUS irr_nsa > 1.05 (impossible): {}".format(int((L["irr_nsa"] > 1.05).sum())))

# ================================================================ name matching


def norm(x):
    x = x.upper().replace("&", "AND")
    x = re.sub(r"[^A-Z ]", " ", x)
    x = re.sub(r"\b(DISTRICT|DIST)\b", " ", x)
    return re.sub(r"\s+", " ", x).strip()


L["stU"] = L["state_name"].str.upper()
L["dN"] = L["district_name"].map(norm)
lus_names = L.groupby("stU")["dN"].unique().to_dict()
apyd = ann[["district_key", "state_name"]].drop_duplicates()
ALIAS = {"ANANTAPUR": "ANANTHAPURAMU", "KADAPA": "Y S R KADAPA", "DOHAD": "DAHOD",
         "GURGAON": "GURUGRAM", "MEWAT": "NUH", "BADGAM": "BUDGAM",
         "BANGALORE RURAL": "BENGALURU RURAL", "RAMANAGARA": "BENGALURU SOUTH",
         "HOSHANGABAD": "NARMADAPURAM", "KHANDWA": "KHANDWA EAST NIMAR",
         "KHARGONE": "KHARGONE WEST NIMAR", "NARSINGHPUR": "NARSIMHAPUR",
         "AHMEDNAGAR": "AHILYANAGAR", "AURANGABAD|MAHARASHTRA": "CHHATRAPATI SAMBHAJINAGAR",
         "OSMANABAD": "DHARASHIV", "SAIHA": "SIAHA", "BALESHWAR": "BALASORE",
         "KENDUJHAR": "KEONJHAR", "PONDICHERRY": "PUDUCHERRY",
         "MUKTSAR": "SRI MUKTSAR SAHIB", "TUTICORIN": "THOOTHUKKUDI",
         "BHADRADRI": "BHADRADRI KOTHAGUDEM", "JAYASHANKAR": "JAYASHANKAR BHUPALAPALLY",
         "JOGULAMBA": "JOGULAMBA GADWAL", "RAJANNA": "RAJANNA SIRCILLA",
         "YADADRI": "YADADRI BHUVANAGIRI", "ALLAHABAD": "PRAYAGRAJ", "FAIZABAD": "AYODHYA",
         "SANT RAVIDAS NAGAR": "BHADOHI", "BALODA BAZAR": "BALODABAZAR BHATAPARA",
         "BALRAMPUR|CHHATTISGARH": "BALRAMPUR RAMANUJGANJ", "KABIRDHAM": "KABEERDHAM",
         "KANKER": "UTTAR BASTAR KANKER", "GAURELLA PENDRA MARWAHI": "GAURELA PENDRA MARWAHI",
         "KARIMGANJ": "SRIBHUMI", "RANGAREDDI": "RANGA REDDY"}
# old Telangana districts filed under Andhra Pradesh in APY; Kargil now in Ladakh
XSTATE = {"ANDHRA PRADESH": ["TELANGANA"], "JAMMU AND KASHMIR": ["LADAKH"]}
rows = []
for k, st in zip(apyd["district_key"], apyd["state_name"]):
    dn = norm(k.split("|")[1])
    dn = ALIAS.get(dn + "|" + st, ALIAS.get(dn, dn))
    cands = list(lus_names.get(st, []))
    hit = None
    for st2 in XSTATE.get(st, []):
        if dn not in cands and dn in set(lus_names.get(st2, [])):
            hit = st2
    if hit:
        rows.append((k, hit, dn, "exact_xstate", 1.0))
        continue
    if dn in cands:
        rows.append((k, st, dn, "exact", 1.0))
        continue
    best = difflib.get_close_matches(dn, cands, n=1, cutoff=0.0)
    sc = difflib.SequenceMatcher(None, dn, best[0]).ratio() if best else 0
    if best and sc >= 0.85:
        rows.append((k, st, best[0], "fuzzy", round(sc, 3)))
    else:
        rows.append((k, st, best[0] if best else None, "none", round(sc, 3)))
nm = pd.DataFrame(rows, columns=["district_key", "stU", "dN", "method", "score"])
nm.to_csv(OUT + "/irrigation_panel_name_match.csv", index=False)
w("APY->LUS name match: " + str(nm["method"].value_counts().to_dict()))
nm_ok = nm[nm["method"] != "none"]
# two APY keys must not share one LUS name
Lm = L.merge(nm_ok[["district_key", "stU", "dN"]], on=["stU", "dN"], how="inner")
# an LUS district-year claimed by two APY keys (e.g. AP|X and TELANGANA|X) is
# assigned to whichever APY key actually has diversity data that year; if both
# do, it is ambiguous and dropped
have = set(zip(ann["district_key"], ann["year_start"]))
Lm = Lm[[(a, b) in have for a, b in zip(Lm["district_key"], Lm["year_start"])]]
amb = Lm.duplicated(["stU", "dN", "year_start"], keep=False)
w("LUS district-years claimed by two APY districts in the same year (dropped): {}".format(int(amb.sum())))
Lm = Lm[~amb]
Lm = Lm.groupby(["district_key", "year_start"]).first().reset_index()
P = ann.merge(Lm.drop(columns=["state_name", "district_name", "stU", "dN"]),
              on=["district_key", "year_start"], how="left")

# ================================================================ 2011 parent, sample
cw = pd.read_csv(SH + "/district_crosswalk.csv")
fp = pd.read_csv(SH + "/final_panel.csv")[["div_key", "in_final", "irr_share",
                                           "flag_shared_parent"]]
P = P.merge(cw[["div_key", "shrug_key"]], left_on="district_key", right_on="div_key",
            how="left").drop(columns="div_key")
P = P.merge(fp.rename(columns={"irr_share": "irr_antyodaya"}),
            left_on="district_key", right_on="div_key", how="left").drop(columns="div_key")
P["in_final"] = P["in_final"].fillna(False).astype(bool)

# ================================================================ quality flags
P = P.sort_values(["district_key", "year_start"])
P["irr_ok"] = P["irr_nsa"].between(0, 1.05) & (P["nsa"] > 0)
P.loc[~P["irr_ok"], "irr_nsa"] = np.nan
P["irr_nsa"] = P["irr_nsa"].clip(upper=1.0)
# spike: more than 0.20 away from the district's own 5-year centred median
med = (P.groupby("district_key")["irr_nsa"]
       .transform(lambda x: x.rolling(5, center=True, min_periods=3).median()))
P["irr_spike"] = (P["irr_nsa"] - med).abs() > 0.20
# territory break: NSA or APY area moves >35 percent year on year
for v in ["nsa", "apy_area"]:
    ch = P.groupby("district_key")[v].transform(lambda x: np.log(x).diff().abs())
    P["brk_" + v] = ch > np.log(1.35)
P["irr_ma3"] = (P.groupby("district_key")["irr_nsa"]
                .transform(lambda x: x.rolling(3, center=True, min_periods=2).mean()))
P.to_csv(OUT + "/irrigation_panel_district_year.csv", index=False)

# ================================================================ coverage report
f = P[P["in_final"]]
w("")
w("Analysis-sample (606) coverage")
w("  districts with any LUS irrigation: {}".format(
    f.loc[f["irr_nsa"].notna(), "district_key"].nunique()))
w("  district-years with diversity: {:,}; with irrigation: {:,} ({:.1%})".format(
    len(f), int(f["irr_nsa"].notna().sum()), f["irr_nsa"].notna().mean()))
w("  spike-flagged district-years: {:,}".format(int(f["irr_spike"].sum())))
w("  NSA breaks: {:,}; APY-area breaks: {:,}".format(
    int(f["brk_nsa"].sum()), int(f["brk_apy_area"].sum())))
cov = f.groupby("year_start").agg(n=("D1", "size"), irr=("irr_nsa", lambda x: x.notna().sum()),
                                  mean_irr=("irr_nsa", "mean"))
w("")
w("| year | districts with diversity | with irrigation | mean NIA/NSA |")
w("|---|---|---|---|")
for y, r in cov.iterrows():
    w("| {}-{} | {} | {} | {:.3f} |".format(y, str(y + 1)[-2:], int(r.n), int(r.irr), r.mean_irr))
st = f.groupby("state_name").agg(dy=("D1", "size"), irr=("irr_nsa", lambda x: x.notna().mean()))
w("")
w("States with under 50% irrigation coverage in the sample:")
for k_, r in st[st["irr"] < 0.5].iterrows():
    w("  {}: {:.0%} of {} district-years".format(k_, r.irr, int(r.dy)))
# district mean LUS vs Antyodaya
dm = f.groupby("district_key").agg(lus=("irr_nsa", "mean"), ay=("irr_antyodaya", "first"),
                                   lus1719=("irr_nsa", lambda x: x[f.loc[x.index, "year_start"] >= 2017].mean()))
w("")
w("Agreement with the Antyodaya 2019-20 share used by the site:")
w("  corr(LUS 1998-2019 mean, Antyodaya) = {:.3f}, n={}".format(
    dm[["lus", "ay"]].dropna().corr().iloc[0, 1], len(dm[["lus", "ay"]].dropna())))
w("  corr(LUS 2017-2019 mean, Antyodaya) = {:.3f}; mean LUS 2017-19 {:.3f} vs Antyodaya {:.3f}".format(
    dm[["lus1719", "ay"]].dropna().corr().iloc[0, 1], dm["lus1719"].mean(), dm["ay"].mean()))
nat = P.groupby("year_start")[["nia", "nsa"]].sum()
w("National check, LUS NIA/NSA summed over matched districts: 1998 {:.3f}, 2019 {:.3f}".format(
    nat.loc[1998, "nia"] / nat.loc[1998, "nsa"], nat.loc[2019, "nia"] / nat.loc[2019, "nsa"]))
open(OUT + "/irrigation_panel_build_log.txt", "w", encoding="utf-8").write("\n".join(LOG))
