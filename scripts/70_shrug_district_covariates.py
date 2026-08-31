"""
70_shrug_district_covariates.py

Builds a district-level covariate table from SHRUG 2.1 for the crop-diversity analysis.

Sources (all village/shrid level, aggregated up to PC11 district):
  Mission Antyodaya (2019-20)  -> irrigation, cropping seasons, markets, farmer counts
  PC11 Village Directory       -> land use, irrigation by source (independent 2011 check)
  PC11 Population Abstract     -> SC/ST population, cultivators vs agricultural labourers
  SECC 2012 (MoRD rural)       -> land ownership, owned acres, caste shares

Design rules enforced here:
  - Areas are SUMMED to district, never averaged.
  - Binary village indicators are summed (village count) AND divided (village share).
  - Irrigation share denominator is GROSS cropped area, not net sown.
  - Every merge asserts row count and logs unmatched keys on both sides.
  - Village-level outliers are flagged and reported, not silently dropped.

Outputs:
  outputs/shrug_covariates/shrug_district_covariates.csv
  outputs/shrug_covariates/build_log.txt
  outputs/shrug_covariates/provenance.csv
"""
import os
import numpy as np
import pandas as pd

SHRUG = r"D:/SHRUG_2.1_Data/extracted"
OUT = r"D:/crop-diversity/outputs/shrug_covariates"
os.makedirs(OUT, exist_ok=True)

LOG = []


def log(msg):
    print(msg)
    LOG.append(str(msg))


def section(t):
    log("")
    log("=" * 78)
    log(t)
    log("=" * 78)


PROV = []


def prov(var, module, file, formula, note=""):
    PROV.append(dict(variable=var, module=module, source_file=file,
                     formula=formula, note=note))


DKEY = ["pc11_state_id", "pc11_district_id"]


def check_merge(left, right, on, how, name):
    """Merge with row-count and match-rate assertions."""
    nl, nr = len(left), len(right)
    m = left.merge(right, on=on, how=how, indicator=True)
    both = (m["_merge"] == "both").sum()
    lonly = (m["_merge"] == "left_only").sum()
    ronly = (m["_merge"] == "right_only").sum()
    log("  merge {}: left={} right={} -> {} (both={} left_only={} right_only={})".format(
        name, nl, nr, len(m), both, lonly, ronly))
    if how == "left":
        assert len(m) == nl, "{}: LEFT MERGE MULTIPLIED ROWS {} -> {}".format(name, nl, len(m))
    return m.drop(columns="_merge")


# ---------------------------------------------------------------- 1. keys
section("1. KEYS")
key = pd.read_stata(SHRUG + "/shrug-pc-keys-dta/shrid_pc11dist_key.dta")
key = key.dropna(subset=DKEY).drop_duplicates("shrid2")
log("shrid -> pc11 district key: {:,} shrids, {} districts".format(
    len(key), key.groupby(DKEY).ngroups))

names = pd.read_stata(SHRUG + "/shrug-shrid-keys-dta/shrid_loc_names.dta",
                      columns=["shrid2", "state_name", "district_name"])
log("loc names: {:,} shrids".format(len(names)))

# modal (most frequent) name per district id, so one district id -> one name
nm = names.merge(key, on="shrid2", how="inner")
dnames = (nm.groupby(DKEY + ["state_name", "district_name"])
            .size().rename("n").reset_index()
            .sort_values("n", ascending=False)
            .drop_duplicates(DKEY)[DKEY + ["state_name", "district_name"]])
log("district name table: {} districts, {} states".format(
    len(dnames), dnames["state_name"].nunique()))


# ------------------------------------------------------------ 2. antyodaya
section("2. MISSION ANTYODAYA (2019-20)")
AY_SUM = ["total_cultivable_area_in_hac", "net_sown_area_in_hac",
          "net_sown_area_kharif_in_hac", "net_sown_area_rabi_in_hac",
          "net_sown_area_other_in_hac", "area_irrigated_in_hac",
          "total_unirrigated_land_area_in_h", "total_no_of_farmers",
          "total_no_farmers_adopted_organic", "no_of_farmers_using_drip_sprinkl",
          "total_hhd", "total_hhd_engaged_in_farm_activi", "total_population"]
AY_BIN = ["mandi", "regular_market", "weekly_haat", "fpo",
          "canal", "surface_water", "ground_water", "other_irrigation",
          "is_soil_testing_centre_available", "is_fertilizer_shop_available",
          "is_govt_seed_centre_available", "availability_of_food_storage_war",
          "availability_of_farm_gate_proces", "availability_of_custom_hiring_ce",
          "is_common_pastures_available", "livestock_ext_services"]

ay = pd.read_stata(SHRUG + "/shrug-antyodaya-dta/antyodaya_shrid.dta",
                   columns=["shrid2"] + AY_SUM + AY_BIN)
log("antyodaya shrid rows: {:,}".format(len(ay)))

# --- village-level data quality flags (report, do not silently drop)
ay["_gca"] = ay[["net_sown_area_kharif_in_hac", "net_sown_area_rabi_in_hac",
                 "net_sown_area_other_in_hac"]].sum(axis=1)
bad_cult = int((ay["total_cultivable_area_in_hac"] > 100000).sum())
bad_irr = int((ay["area_irrigated_in_hac"] > ay["_gca"] * 3).sum())
bad_nsa = int((ay["net_sown_area_in_hac"] > ay["total_cultivable_area_in_hac"] * 1.05).sum())
log("  QC village outliers: cultivable>100k ha = {}; irrigated>3x GCA = {}; "
    "net sown>cultivable = {:,}".format(bad_cult, bad_irr, bad_nsa))

ay.loc[ay["total_cultivable_area_in_hac"] > 100000, "total_cultivable_area_in_hac"] = np.nan
log("  set {} impossible cultivable-area values to NaN".format(bad_cult))

# do the 4 irrigation-source flags behave like a partition?
src = ay[["canal", "surface_water", "ground_water", "other_irrigation"]].sum(axis=1)
log("  irrigation source flags sum: p50={:.2f}  share==0: {:.3f}  "
    "share in (0,1.05]: {:.3f}  share>1.05: {:.3f}".format(
        src.median(), (src == 0).mean(),
        ((src > 0) & (src <= 1.05)).mean(), (src > 1.05).mean()))

ay = check_merge(ay, key, "shrid2", "left", "antyodaya x key")
ay = ay.dropna(subset=DKEY)

ay_d = ay.groupby(DKEY).agg({c: "sum" for c in AY_SUM + AY_BIN})
ay_d["ay_n_villages"] = ay.groupby(DKEY).size()
ay_d = ay_d.reset_index()
log("antyodaya districts: {}".format(len(ay_d)))

for c in AY_SUM:
    prov("ay_" + c, "shrug-antyodaya-dta", "antyodaya_shrid.dta",
         "sum over shrids in district")
for c in AY_BIN:
    prov("ay_" + c + "_n", "shrug-antyodaya-dta", "antyodaya_shrid.dta",
         "sum of village-share flag = est. village count")

ay_d = ay_d.rename(columns={c: "ay_" + c for c in AY_SUM})
ay_d = ay_d.rename(columns={c: "ay_" + c + "_n" for c in AY_BIN})


# ------------------------------------------------------------- 3. vd11
section("3. PC11 VILLAGE DIRECTORY (land use, 2011)")
VD = ["pc11_vd_land_nt_swn", "pc11_vd_land_un_irr", "pc11_vd_land_src_irr",
      "pc11_vd_land_canal_irr", "pc11_vd_land_wl_tw_irr", "pc11_vd_land_tnk_lk_irr",
      "pc11_vd_land_w_fall_irr", "pc11_vd_land_oth_src_irr",
      "pc11_vd_land_fores", "pc11_vd_land_non_agri", "pc11_vd_land_cur_fal",
      "pc11_vd_land_fallow", "pc11_vd_land_cult_waste", "pc11_vd_land_pst_grz",
      "pc11_vd_power_agr", "pc11_vd_comm_bank", "pc11_vd_coop_bank"]
vd = pd.read_stata(SHRUG + "/shrug-vd11-dta/pc11_vd_clean_shrid.dta",
                   columns=["shrid2"] + VD)
log("vd11 shrid rows: {:,}".format(len(vd)))
vd = check_merge(vd, key, "shrid2", "left", "vd11 x key").dropna(subset=DKEY)
vd_d = vd.groupby(DKEY)[VD].sum()
vd_d["vd_n_villages"] = vd.groupby(DKEY).size()
vd_d = vd_d.reset_index()
log("vd11 districts: {}".format(len(vd_d)))
for c in VD:
    prov(c, "shrug-vd11-dta", "pc11_vd_clean_shrid.dta", "sum over shrids")


# ------------------------------------------------------------- 4. pca11
section("4. PC11 POPULATION ABSTRACT (caste, workers, 2011)")
PCA = ["pc11_pca_tot_p", "pc11_pca_p_sc", "pc11_pca_p_st",
       "pc11_pca_main_cl_p", "pc11_pca_main_al_p",
       "pc11_pca_marg_cl_p", "pc11_pca_marg_al_p",
       "pc11_pca_tot_work_p", "pc11_pca_no_hh"]
pca = pd.read_stata(SHRUG + "/shrug-pca11-dta/pc11_pca_clean_shrid.dta",
                    columns=["shrid2"] + PCA)
log("pca11 shrid rows: {:,}".format(len(pca)))
pca = check_merge(pca, key, "shrid2", "left", "pca11 x key").dropna(subset=DKEY)
pca_d = pca.groupby(DKEY)[PCA].sum().reset_index()
log("pca11 districts: {}".format(len(pca_d)))
for c in PCA:
    prov(c, "shrug-pca11-dta", "pc11_pca_clean_shrid.dta", "sum over shrids")


# -------------------------------------------------------------- 5. secc
section("5. SECC 2012 (MoRD rural)")
SECC_W = ["land_own_share", "sc_share", "st_share", "nco2d_cultiv_share",
          "inc_source_cultiv_share", "inc_source_manlab_share", "kisan_cc"]
SECC_S = ["unirr_land_acre_sum", "two_crop_acre_sum", "other_irr_acre_sum"]
secc = pd.read_stata(SHRUG + "/shrug-secc-mord-rural-dta/secc_rural_shrid.dta",
                     columns=["shrid2", "secc_hh"] + SECC_W + SECC_S)
log("secc shrid rows: {:,}".format(len(secc)))
secc = check_merge(secc, key, "shrid2", "left", "secc x key").dropna(subset=DKEY)

# household-weighted means for shares, plain sums for acre totals
for c in SECC_W:
    secc["_w_" + c] = secc[c] * secc["secc_hh"]
g = secc.groupby(DKEY)
secc_d = g[["secc_hh"] + SECC_S].sum()
for c in SECC_W:
    secc_d["secc_" + c] = g["_w_" + c].sum() / g["secc_hh"].sum()
    prov("secc_" + c, "shrug-secc-mord-rural-dta", "secc_rural_shrid.dta",
         "household-weighted mean over shrids")
secc_d = secc_d.rename(columns={c: "secc_" + c for c in SECC_S}).reset_index()
for c in SECC_S:
    prov("secc_" + c, "shrug-secc-mord-rural-dta", "secc_rural_shrid.dta",
         "sum over shrids")
log("secc districts: {}".format(len(secc_d)))


# -------------------------------------------------------------- 6. merge
section("6. ASSEMBLE DISTRICT TABLE")
df = dnames.copy()
for tab, nm_ in [(ay_d, "antyodaya"), (vd_d, "vd11"), (pca_d, "pca11"), (secc_d, "secc")]:
    df = check_merge(df, tab, DKEY, "left", "base x " + nm_)
log("assembled: {}".format(df.shape))


# ------------------------------------------------------------ 7. derive
section("7. DERIVED VARIABLES")


def safe_div(a, b, floor=0.0):
    aa = pd.to_numeric(df[a], errors="coerce")
    bb = pd.to_numeric(df[b], errors="coerce")
    return np.where(bb > floor, aa / bb, np.nan)


# --- cropping
df["gca_ha"] = df[["ay_net_sown_area_kharif_in_hac", "ay_net_sown_area_rabi_in_hac",
                   "ay_net_sown_area_other_in_hac"]].sum(axis=1)
df["nsa_ha"] = df["ay_net_sown_area_in_hac"]
df["cropping_intensity"] = safe_div("gca_ha", "nsa_ha")
prov("gca_ha", "derived", "-", "kharif + rabi + other net sown area")
prov("cropping_intensity", "derived", "-", "gca_ha / nsa_ha")

# --- irrigation, both denominators kept so the choice is auditable
df["irr_share_gca"] = safe_div("ay_area_irrigated_in_hac", "gca_ha")
df["irr_share_nsa"] = safe_div("ay_area_irrigated_in_hac", "nsa_ha")
prov("irr_share_gca", "derived", "-", "ay_area_irrigated / gca_ha", "PREFERRED")
prov("irr_share_nsa", "derived", "-", "ay_area_irrigated / nsa_ha", "diagnostic only")

# --- independent PC11 irrigation check
df["vd_irr_share"] = safe_div("pc11_vd_land_src_irr", "pc11_vd_land_nt_swn")
prov("vd_irr_share", "derived", "-", "src_irr / net sown (PC11 2011)",
     "independent of Antyodaya")

# --- irrigation source mix, two independent versions
for s, lab in [("canal", "canal"), ("ground_water", "ground"),
               ("surface_water", "surface"), ("other_irrigation", "other")]:
    df["ay_src_" + lab + "_vshare"] = safe_div("ay_" + s + "_n", "ay_n_villages")
    prov("ay_src_" + lab + "_vshare", "derived", "-",
         "villages with " + s + " as source / total villages")

vd_src = ["pc11_vd_land_canal_irr", "pc11_vd_land_wl_tw_irr",
          "pc11_vd_land_tnk_lk_irr", "pc11_vd_land_w_fall_irr",
          "pc11_vd_land_oth_src_irr"]
df["_vd_src_tot"] = df[vd_src].sum(axis=1)
for c, lab in zip(vd_src, ["canal", "tubewell", "tank", "waterfall", "othersrc"]):
    df["vd_src_" + lab + "_ashare"] = safe_div(c, "_vd_src_tot")
    prov("vd_src_" + lab + "_ashare", "derived", "-",
         c + " / sum of VD irrigated area by source", "AREA share, not village share")

# --- markets and services
SHORT = {"availability_of_food_storage_war": "storage",
         "availability_of_farm_gate_proces": "farmgate_proc",
         "is_fertilizer_shop_available": "fert_shop",
         "is_soil_testing_centre_available": "soil_test",
         "availability_of_custom_hiring_ce": "custom_hire",
         "livestock_ext_services": "livestock_ext"}
for s in ["mandi", "regular_market", "weekly_haat", "fpo",
          "availability_of_food_storage_war", "availability_of_farm_gate_proces",
          "is_fertilizer_shop_available", "is_soil_testing_centre_available",
          "availability_of_custom_hiring_ce", "livestock_ext_services"]:
    short = SHORT.get(s, s)
    df[short + "_vshare"] = safe_div("ay_" + s + "_n", "ay_n_villages")
    df[short + "_count"] = df["ay_" + s + "_n"]
    prov(short + "_vshare", "derived", "-",
         "villages with " + s + " / total villages")

# --- agrarian structure
df["mean_holding_ha"] = safe_div("ay_total_cultivable_area_in_hac", "ay_total_no_of_farmers")
df["gca_per_farmer_ha"] = safe_div("gca_ha", "ay_total_no_of_farmers")
df["organic_farmer_share"] = safe_div("ay_total_no_farmers_adopted_organic",
                                      "ay_total_no_of_farmers")
df["drip_farmer_share"] = safe_div("ay_no_of_farmers_using_drip_sprinkl",
                                   "ay_total_no_of_farmers")
df["farm_hh_share"] = safe_div("ay_total_hhd_engaged_in_farm_activi", "ay_total_hhd")
prov("mean_holding_ha", "derived", "-", "cultivable area / number of farmers",
     "proxy for average operational holding")
prov("organic_farmer_share", "derived", "-", "organic farmers / total farmers")

# --- caste and labour
df["pca_sc_share"] = safe_div("pc11_pca_p_sc", "pc11_pca_tot_p")
df["pca_st_share"] = safe_div("pc11_pca_p_st", "pc11_pca_tot_p")
df["_cl"] = df["pc11_pca_main_cl_p"] + df["pc11_pca_marg_cl_p"]
df["_al"] = df["pc11_pca_main_al_p"] + df["pc11_pca_marg_al_p"]
df["cultivator_share_agwork"] = np.where((df["_cl"] + df["_al"]) > 0,
                                         df["_cl"] / (df["_cl"] + df["_al"]), np.nan)
df["agwork_share_totwork"] = safe_div("_cl", "pc11_pca_tot_work_p") + \
                             safe_div("_al", "pc11_pca_tot_work_p")
prov("cultivator_share_agwork", "derived", "-",
     "cultivators / (cultivators + ag labourers), main+marginal",
     "proxy for owner-operator vs landless structure")

# --- SECC land
df["secc_landless_share"] = 1 - df["secc_land_own_share"]
df["secc_unirr_acre_per_hh"] = safe_div("secc_unirr_land_acre_sum", "secc_hh")
df["secc_twocrop_acre_per_hh"] = safe_div("secc_two_crop_acre_sum", "secc_hh")
prov("secc_landless_share", "derived", "-", "1 - land_own_share")

df = df.drop(columns=[c for c in df.columns if c.startswith("_")])


# ------------------------------------------------------------ 8. flags
section("8. COVERAGE FLAGS")
df["flag_low_ay_coverage"] = df["ay_n_villages"] < 0.25 * df["vd_n_villages"]
df["flag_no_antyodaya"] = df["ay_n_villages"].isna()
df["flag_irr_gt_1"] = df["irr_share_gca"] > 1.0
log("  low antyodaya coverage (<25% of PC11 villages): {}".format(
    int(df["flag_low_ay_coverage"].sum())))
log("  no antyodaya at all: {}".format(int(df["flag_no_antyodaya"].sum())))
log("  irrigation share > 1 even on GCA: {}".format(int(df["flag_irr_gt_1"].sum())))

df["district_key"] = (df["state_name"].str.upper().str.strip() + "|" +
                      df["district_name"].str.upper().str.strip())

front = ["pc11_state_id", "pc11_district_id", "state_name", "district_name",
         "district_key", "ay_n_villages", "vd_n_villages"]
df = df[front + [c for c in df.columns if c not in front]]

df.to_csv(OUT + "/shrug_district_covariates.csv", index=False)
pd.DataFrame(PROV).to_csv(OUT + "/provenance.csv", index=False)
log("")
log("WROTE {}/shrug_district_covariates.csv  {}".format(OUT, df.shape))
log("WROTE {}/provenance.csv  {} variables documented".format(OUT, len(PROV)))
with open(OUT + "/build_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
