"""
98_crop_group_panel.py  (titled 97_crop_group_panel.py in earlier copies; renumbered, file name is 98)

District x year x season crop-group panel on constant 2011 (PC11) district
boundaries, 1997-98 to 2019-20, with group area, shares of gross cropped area,
annual totals and data-defect flags. The panel keeps 1997-98; every analysis
script drops it through _years.py.

Cleaning is script 76's (and 96's), unchanged:
  bogus state-district pairs dropped; Uttarakhand, Chhattisgarh and Jharkhand
  districts remapped; 2020-21 dropped; duplicate (district, year, season, crop)
  rows collapsed by maximum.

Constant boundaries:
  every crop-record district (725) is mapped to its 2011 parent with
  outputs/shrug_covariates/district_crosswalk.csv (script 72) and areas are
  summed to the parent. The parent key (`unit_key`, the SHRUG `shrug_key`) is
  the panel unit and is also the clustering group. The three districts with no
  2011 parent (Mumbai, Daman, Diu) keep their own key and carry
  `unit_no_2011_parent = 1`.

Crop groups (exactly these nine):
  rice, wheat, maize, millets (jowar, bajra, ragi, small millets),
  pulses, oilseeds, cotton, sugarcane, other.
  Coconut is NOT in oilseeds: the oilseed list is the nine annual oilseeds plus
  APY's own 'Other Oilseeds' line; coconut is a plantation crop and goes to other.

Seasons:
  kharif     = APY Kharif + Autumn + Winter (autumn and winter rice are the
               monsoon-season aus and aman crops of the east)
  rabi       = Rabi
  summer     = Summer (boro rice and other dry-season crops)
  whole_year = Whole Year rows (perennials and states that do not split seasons)
  annual     = sum of the four

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/97_crop_group_panel.py

Outputs (outputs/crop_groups/):
  crop_group_panel.csv              unit x year x season, wide on the nine groups
  crop_group_panel_annual.csv       the season == 'annual' rows only
  crop_group_panel_long.csv.gz      unit x year x season x group, long
  crop_group_flags.csv              unit x year defect flags (also merged on the panel)
  crop_group_units.csv              unit attributes (state, children, in_606, years)
  crop_group_child_map.csv          crop-record district -> unit
  crop_group_crop_mapping.csv       every APY crop name -> group, with area
  crop_group_presplit_donors.csv    late-entrant units and the neighbour they were reported in
  crop_group_panel_log.txt          console log

Run scripts/97_crop_group_adjacency.py first: the pre-split donor flag uses it.
"""
import os

import numpy as np
import pandas as pd

RAW = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
REPO = r"D:/crop-diversity"
COV = REPO + "/outputs/shrug_covariates"
OUT = REPO + "/outputs/crop_groups"
os.makedirs(OUT, exist_ok=True)

LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


# ------------------------------------------------------------------ load, script 76 cleaning
df = pd.read_csv(RAW)
for c in ["year", "season", "state_name", "district_name", "crop_name", "crop_type"]:
    df[c] = df[c].astype(str).str.strip()
assert (df["area_unit"].dropna() == "Hectare").all()
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
df = df[df["year_start"] <= 2019]
K = ["district_key", "year", "season", "crop_name"]
dupmask = df.duplicated(subset=K, keep=False)
df["_dup"] = dupmask
n_dup_rows = int(df.duplicated(subset=K).sum())
df = df.sort_values("area", ascending=False).drop_duplicates(subset=K, keep="first")
w("Rows after script-76 cleaning: {:,}; duplicate rows collapsed: {}.".format(len(df), n_dup_rows))

# ------------------------------------------------------------------ crop groups
GROUP = {
    "Rice": "rice",
    "Wheat": "wheat",
    "Maize": "maize",
    "Jowar": "millets", "Bajra": "millets", "Ragi": "millets", "Small Millets": "millets",
    "Arhar/Tur": "pulses", "Gram": "pulses", "Moong(Green Gram)": "pulses",
    "Urad": "pulses", "Masoor": "pulses", "Moth": "pulses", "Horse-Gram": "pulses",
    "Khesari": "pulses", "Cowpea(Lobia)": "pulses", "Peas & Beans (Pulses)": "pulses",
    "Other  Rabi Pulses": "pulses", "Other Kharif Pulses": "pulses",
    "Other Summer Pulses": "pulses",
    "Groundnut": "oilseeds", "Rapeseed &Mustard": "oilseeds", "Soyabean": "oilseeds",
    "Sunflower": "oilseeds", "Sesamum": "oilseeds", "Castor Seed": "oilseeds",
    "Linseed": "oilseeds", "Niger Seed": "oilseeds", "Safflower": "oilseeds",
    "Other Oilseeds": "oilseeds",
    "Cotton(Lint)": "cotton",
    "Sugarcane": "sugarcane",
}
GROUPS = ["rice", "wheat", "maize", "millets", "pulses", "oilseeds", "cotton",
          "sugarcane", "other"]
df["group"] = df["crop_name"].map(GROUP).fillna("other")

cm = (df.groupby(["crop_name", "crop_type", "group"])
        .agg(rows=("area", "size"), area_mha_summed_all_years=("area", lambda s: s.sum() / 1e6))
        .reset_index().sort_values(["group", "area_mha_summed_all_years"], ascending=[True, False]))
cm["note"] = ""
cm.loc[cm["crop_name"] == "Coconut", "note"] = ("APY crop_type Oilseeds; plantation crop, "
                                                "not in the specified oilseed list, so other")
cm.loc[cm["crop_name"] == "Guar Seed", "note"] = "APY crop_type Fodder; a legume but not a pulse"
cm.loc[cm["crop_name"].isin(["Barley", "Other Cereals"]), "note"] = "cereal outside the named groups"
cm.loc[cm["crop_name"].isin(["Moth", "Horse-Gram", "Khesari", "Cowpea(Lobia)",
                             "Peas & Beans (Pulses)"]), "note"] = "other pulses"
cm["area_mha_summed_all_years"] = cm["area_mha_summed_all_years"].round(3)
cm.to_csv(OUT + "/crop_group_crop_mapping.csv", index=False)
w("")
w("Crop mapping ({} APY crop names):".format(len(cm)))
for g in GROUPS:
    w("  {:<10} {}".format(g, ", ".join(cm.loc[cm["group"] == g, "crop_name"])))

# ------------------------------------------------------------------ seasons
SEASON = {"Kharif": "kharif", "Autumn": "kharif", "Winter": "kharif",
          "Rabi": "rabi", "Summer": "summer", "Whole Year": "whole_year"}
df["season_group"] = df["season"].map(SEASON)
assert df["season_group"].notna().all(), df.loc[df["season_group"].isna(), "season"].unique()
SEASONS = ["kharif", "rabi", "summer", "whole_year"]

# ------------------------------------------------------------------ crosswalk to 2011 parents
cw = pd.read_csv(COV + "/district_crosswalk.csv")
fp = pd.read_csv(COV + "/final_panel.csv")
S606 = set(fp.loc[fp["in_final"] == True, "div_key"])
assert len(S606) == 606
missing = set(df["district_key"]) - set(cw["div_key"])
assert not missing, missing
cw["unit_key"] = cw["shrug_key"].where(cw["shrug_key"].notna(), cw["div_key"])
cw["unit_no_2011_parent"] = cw["shrug_key"].isna().astype(int)
df = df.merge(cw[["div_key", "unit_key", "unit_no_2011_parent"]],
              left_on="district_key", right_on="div_key", how="left")
assert df["unit_key"].notna().all()
w("")
w("Crop-record districts: {}; 2011 units: {} ({} without a 2011 parent kept as their own unit).".format(
    df["district_key"].nunique(), df["unit_key"].nunique(),
    df.loc[df["unit_no_2011_parent"] == 1, "unit_key"].nunique()))

# ------------------------------------------------------------------ defect flags, row level
st, y, crop, sea = df["state_name"], df["year_start"], df["crop_name"], df["season"]
df["f_odisha_season_collapsed"] = ((st == "Odisha") & y.between(2001, 2003)).astype(int)
df["f_odisha_summer_rice_2012"] = ((st == "Odisha") & (y == 2012) & (crop == "Rice")
                                   & (sea == "Summer")).astype(int)
df["f_bihar_rice_season_swap"] = ((st == "Bihar") & y.between(1998, 2001)
                                  & (crop == "Rice")).astype(int)
df["f_bihar_1999_rice_missing"] = ((st == "Bihar") & (y == 1999)).astype(int)
df["f_tn_rice_single_season"] = ((st == "Tamil Nadu") & (y <= 2016)).astype(int)

UY = ["unit_key", "year_start"]
fl = df.groupby(UY).agg(
    n_child_reporting=("district_key", "nunique"),
    child_districts=("district_key", lambda s: ";".join(sorted(set(s)))),
    dup_rows_collapsed=("_dup", "sum"),
    f_odisha_season_collapsed=("f_odisha_season_collapsed", "max"),
    f_odisha_summer_rice_2012=("f_odisha_summer_rice_2012", "max"),
    f_bihar_rice_season_swap=("f_bihar_rice_season_swap", "max"),
    f_bihar_1999_rice_missing=("f_bihar_1999_rice_missing", "max"),
    f_tn_rice_single_season=("f_tn_rice_single_season", "max"),
    gca_ha=("area", "sum")).reset_index()
# dup_rows_collapsed = number of (district, year, season, crop) cells that had duplicate rows
wy = df[df["season_group"] == "whole_year"].groupby(UY)["area"].sum()
fl = fl.merge(wy.rename("whole_year_ha").reset_index(), on=UY, how="left")
fl["whole_year_ha"] = fl["whole_year_ha"].fillna(0)
fl["whole_year_share"] = fl["whole_year_ha"] / fl["gca_ha"]
fl["f_whole_year_ge50"] = (fl["whole_year_share"] >= 0.5).astype(int)
fl["f_whole_year_ge20"] = (fl["whole_year_share"] >= 0.2).astype(int)

# GCA against the unit's own median over the years it is observed
med = fl.groupby("unit_key")["gca_ha"].transform("median")
fl["gca_ratio_to_unit_median"] = fl["gca_ha"] / med
r_ = fl["gca_ratio_to_unit_median"]
fl["f_gca_jump_severe"] = ((r_ < 0.5) | (r_ > 2.0)).astype(int)
fl["f_gca_jump_mild"] = ((r_ < 2 / 3) | (r_ > 1.5)).astype(int)

# children of the unit not reporting in a year when they report both before and after
ch = df.groupby(["unit_key", "district_key"])["year_start"].agg(["min", "max"]).reset_index()
yrs_ch = df.groupby(["unit_key", "district_key"])["year_start"].apply(set)
gap = []
for (u, d), ys in yrs_ch.items():
    lo, hi = min(ys), max(ys)
    for yy in range(lo + 1, hi):
        if yy not in ys:
            gap.append((u, yy))
gap = pd.DataFrame(gap, columns=UY).drop_duplicates()
gap["f_child_gap"] = 1
fl = fl.merge(gap, on=UY, how="left")
fl["f_child_gap"] = fl["f_child_gap"].fillna(0).astype(int)

# pre-2011 splits the 2011 crosswalk cannot fix: a unit that first appears after
# 1998-99 was usually reported inside a neighbouring unit before that. Among the
# late unit's queen neighbours (script 97 adjacency), any whose GCA steps down by
# at least 10 percent when the late unit appears, by an amount at least a fifth of
# the late unit's own area, is taken as a donor and its earlier years are flagged.
# Entry is the first year the unit holds at least a fifth of its own median GCA,
# so one-off stub rows (a few hectares) do not count as entry.
unit_state = df.groupby("unit_key")["state_name"].agg(lambda s: s.value_counts().index[0])
gw = fl.pivot_table(index="unit_key", columns="year_start", values="gca_ha")
umed = gw.median(axis=1)
first = fl.groupby("unit_key")["year_start"].min()
entry = gw.ge(0.2 * umed, axis=0).idxmax(axis=1)
late = entry[entry > 1998]
adj = pd.read_csv(OUT + "/crop_group_adjacency_edges.csv")
NB = adj.groupby("unit_i")["unit_j"].apply(set).to_dict()


def win(u, lo, hi):
    return gw.loc[u, [c for c in gw.columns if lo <= c < hi]].dropna()


donor = []
for u, t in late.items():
    late_size = win(u, t, t + 3).median()
    for p in sorted(NB.get(u, set())):
        if p not in gw.index or entry[p] >= t:
            continue
        before, after = win(p, t - 3, t), win(p, t, t + 3)
        if len(before) == 0 or len(after) == 0:
            continue
        drop = before.median() - after.median()
        if after.median() / before.median() < 0.9 and drop > 0.2 * late_size:
            donor.append({"late_unit": u, "entry_year": int(t), "donor_unit": p,
                          "donor_drop_ha": round(drop), "late_unit_area_ha": round(late_size),
                          "donor_ratio_after_before": round(after.median() / before.median(), 3)})
donor = pd.DataFrame(donor, columns=["late_unit", "entry_year", "donor_unit", "donor_drop_ha",
                                     "late_unit_area_ha", "donor_ratio_after_before"])
fl["f_pre_split_donor"] = 0
fl["f_late_entrant"] = fl["unit_key"].map(entry > 1998).astype(int)
fl["f_before_entry"] = (fl["year_start"] < fl["unit_key"].map(entry)).astype(int)
for _, d in donor.iterrows():
    m = (fl["unit_key"] == d["donor_unit"]) & (fl["year_start"] < d["entry_year"])
    fl.loc[m, "f_pre_split_donor"] = 1
donor.to_csv(OUT + "/crop_group_presplit_donors.csv", index=False)

fl["flag_season_unreliable"] = (fl[["f_odisha_season_collapsed", "f_bihar_rice_season_swap",
                                    "f_tn_rice_single_season", "f_whole_year_ge50",
                                    "f_odisha_summer_rice_2012"]].max(axis=1)).astype(int)
fl["flag_annual_unreliable"] = (fl[["f_odisha_summer_rice_2012", "f_bihar_1999_rice_missing",
                                    "f_gca_jump_severe", "f_pre_split_donor",
                                    "f_child_gap", "f_before_entry"]].max(axis=1)).astype(int)
FLAGS = [c for c in fl.columns if c.startswith("f_") or c.startswith("flag_")]
fl["flag_any"] = fl[FLAGS].max(axis=1).astype(int)
FLAGS.append("flag_any")

# ------------------------------------------------------------------ units
units = cw.groupby("unit_key").agg(
    n_child_districts=("div_key", "size"),
    child_districts=("div_key", lambda s: ";".join(sorted(s))),
    unit_no_2011_parent=("unit_no_2011_parent", "max"),
    in_606=("div_key", lambda s: int(any(k in S606 for k in s)))).reset_index()
units = units[units["unit_key"].isin(df["unit_key"])]
units["state"] = units["unit_key"].str.split("|").str[0]
units["district"] = units["unit_key"].str.split("|").str[1]
units["state_crop_records"] = units["unit_key"].map(unit_state)
units["first_year"] = units["unit_key"].map(first)
units["last_year"] = units["unit_key"].map(fl.groupby("unit_key")["year_start"].max())
units["n_years"] = units["unit_key"].map(fl.groupby("unit_key")["year_start"].nunique())
units["mean_gca_ha"] = units["unit_key"].map(fl.groupby("unit_key")["gca_ha"].mean()).round(1)
units = units[["unit_key", "state", "district", "state_crop_records", "n_child_districts",
               "child_districts", "unit_no_2011_parent", "in_606", "first_year",
               "last_year", "n_years", "mean_gca_ha"]]
units.to_csv(OUT + "/crop_group_units.csv", index=False)
cw[["div_key", "unit_key", "method", "score", "unit_no_2011_parent"]].assign(
    in_606=cw["div_key"].isin(S606).astype(int)).to_csv(
    OUT + "/crop_group_child_map.csv", index=False)

# ------------------------------------------------------------------ the panel
a = df.groupby(UY + ["season_group", "group"])["area"].sum()
uy = fl[UY]
grid = (uy.assign(_k=1)
          .merge(pd.DataFrame({"season": SEASONS + ["annual"], "_k": 1}), on="_k")
          .merge(pd.DataFrame({"group": GROUPS, "_k": 1}), on="_k").drop(columns="_k"))
a = a.reset_index().rename(columns={"season_group": "season", "area": "area_ha"})
ann_a = a.groupby(UY + ["group"])["area_ha"].sum().reset_index().assign(season="annual")
a = pd.concat([a, ann_a], ignore_index=True)
L = grid.merge(a, on=UY + ["season", "group"], how="left")
L["area_ha"] = L["area_ha"].fillna(0.0)
L["season_area_ha"] = L.groupby(UY + ["season"])["area_ha"].transform("sum")
L = L.merge(fl[UY + ["gca_ha"]], on=UY, how="left")
L["share_season"] = np.where(L["season_area_ha"] > 0, L["area_ha"] / L["season_area_ha"], np.nan)
L["share_gca"] = L["area_ha"] / L["gca_ha"]
L["ln_area_ha"] = np.where(L["area_ha"] > 0, np.log(L["area_ha"].clip(lower=1e-9)), np.nan)
L["ln1p_area_ha"] = np.log1p(L["area_ha"])
L["year"] = L["year_start"].map(lambda v: "{}-{}".format(v, str(v + 1)[-2:]))

# checks
chk = L[L["season"] == "annual"].groupby(UY)["share_gca"].sum()
assert np.allclose(chk, 1.0), "annual shares do not sum to 1"
chk2 = L[L["season"] != "annual"].groupby(UY)["area_ha"].sum()
assert np.allclose(chk2.values, fl.set_index(UY).loc[chk2.index, "gca_ha"].values)
w("")
w("Checks passed: annual group shares sum to 1; the four seasons sum to annual GCA.")

L = L.merge(fl[UY + FLAGS], on=UY, how="left")
L = L.merge(units[["unit_key", "state", "in_606", "unit_no_2011_parent"]], on="unit_key", how="left")
LONGCOLS = ["unit_key", "state", "year_start", "year", "season", "group", "area_ha",
            "ln_area_ha", "ln1p_area_ha", "share_season", "share_gca", "season_area_ha",
            "gca_ha", "in_606", "unit_no_2011_parent"] + FLAGS
L = L[LONGCOLS].sort_values(["unit_key", "year_start", "season", "group"])
L.to_csv(OUT + "/crop_group_panel_long.csv.gz", index=False, compression="gzip",
         float_format="%.6g")

# wide
Wd = L.pivot_table(index=["unit_key", "year_start", "season"], columns="group",
                   values=["area_ha", "share_season", "share_gca", "ln_area_ha"],
                   aggfunc="first", dropna=False)
Wd.columns = ["{}_{}".format(v, g) for v, g in Wd.columns]
Wd = Wd.reset_index()
SEA_ORDER = {s: i for i, s in enumerate(SEASONS + ["annual"])}
base = L.drop_duplicates(["unit_key", "year_start", "season"])[
    ["unit_key", "state", "year_start", "year", "season", "season_area_ha", "gca_ha",
     "in_606", "unit_no_2011_parent"]]
Wd = base.merge(Wd, on=["unit_key", "year_start", "season"], how="left")
Wd = Wd.merge(fl[UY + ["n_child_reporting", "dup_rows_collapsed", "whole_year_share",
                       "gca_ratio_to_unit_median"] + FLAGS], on=UY, how="left")
order = (["unit_key", "state", "year_start", "year", "season", "season_area_ha", "gca_ha"]
         + ["area_ha_" + g for g in GROUPS] + ["share_season_" + g for g in GROUPS]
         + ["share_gca_" + g for g in GROUPS] + ["ln_area_ha_" + g for g in GROUPS]
         + ["in_606", "unit_no_2011_parent", "n_child_reporting", "dup_rows_collapsed",
            "whole_year_share", "gca_ratio_to_unit_median"] + FLAGS)
Wd = Wd[order]
Wd["_s"] = Wd["season"].map(SEA_ORDER)
Wd = Wd.sort_values(["unit_key", "year_start", "_s"]).drop(columns="_s")
Wd.to_csv(OUT + "/crop_group_panel.csv", index=False, float_format="%.6g")
Wd[Wd["season"] == "annual"].to_csv(OUT + "/crop_group_panel_annual.csv", index=False,
                                    float_format="%.6g")
fl.drop(columns=["child_districts"]).merge(
    fl[UY + ["child_districts"]], on=UY).to_csv(OUT + "/crop_group_flags.csv", index=False,
                                                float_format="%.6g")

# ------------------------------------------------------------------ report
w("")
w("Panel: {} units, {} unit-years, {} unit-year-season rows (incl. annual).".format(
    fl["unit_key"].nunique(), len(fl), len(Wd)))
s606 = units[units["in_606"] == 1]
w("Units holding a 606-sample district: {} (the 605 clustering groups expected).".format(len(s606)))
w("Unit-years in the 606 units: {}.".format(int(fl["unit_key"].isin(s606["unit_key"]).sum())))
w("")
w("Unit-years by year (all units / 606 units):")
cov = fl.assign(s=fl["unit_key"].isin(s606["unit_key"])).groupby("year_start").agg(
    all=("unit_key", "size"), s606=("s", "sum"), gca_mha=("gca_ha", lambda v: v.sum() / 1e6))
for yy, rr in cov.iterrows():
    w("  {}-{}  {:>4} {:>4}  GCA {:>6.1f} Mha".format(yy, str(yy + 1)[-2:], int(rr["all"]),
                                                     int(rr["s606"]), rr["gca_mha"]))
w("")
w("Years observed per unit: " + str(units["n_years"].describe().round(1).to_dict()))
w("Units observed all 23 years: {}.".format(int((units["n_years"] == 23).sum())))
w("")
w("Unit-years with at least one season row (all units):")
for s in SEASONS:
    n = int((L[(L["season"] == s) & (L["group"] == "rice")]["season_area_ha"] > 0).sum())
    w("  {:<11} {}".format(s, n))
w("")
w("Flags (unit-years flagged, of {}; in 606 units):".format(len(fl)))
in6 = fl["unit_key"].isin(s606["unit_key"])
for f in FLAGS:
    w("  {:<28} {:>5}  {:>5}".format(f, int(fl[f].sum()), int(fl.loc[in6, f].sum())))
w("")
w("Pre-2011 split donors detected: {}".format(len(donor)))
for _, d in donor.iterrows():
    w("  {} (entry {}) <- {}: donor drop {:,} ha (ratio {}) vs late unit {:,} ha".format(
        d["late_unit"], d["entry_year"], d["donor_unit"], d["donor_drop_ha"],
        d["donor_ratio_after_before"], d["late_unit_area_ha"]))
late_nodonor = sorted(set(late.index) - set(donor["late_unit"]))
w("Late-entrant units (entry after 1998-99) with no donor found: {}".format(len(late_nodonor)))
w("  " + "; ".join("{} ({})".format(u, int(late[u])) for u in late_nodonor))
w("")
nat = L[(L["season"] == "annual")].groupby(["year_start", "group"])["area_ha"].sum().unstack()
nat = nat.div(nat.sum(axis=1), axis=0)
w("National share of GCA by group, selected years (unflagged and flagged together):")
w(nat.loc[[1998, 2005, 2012, 2019]].round(3).to_string())
clean = fl.loc[fl["flag_annual_unreliable"] == 0, UY]
nc = L[(L["season"] == "annual")].merge(clean, on=UY).groupby(["year_start", "group"])["area_ha"].sum().unstack()
nc = nc.div(nc.sum(axis=1), axis=0)
w("")
w("Same, annual-reliable unit-years only:")
w(nc.loc[[1998, 2005, 2012, 2019]].round(3).to_string())

with open(OUT + "/crop_group_panel_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
print("\nWROTE outputs to", OUT)
