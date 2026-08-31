"""
79_market_covariates.py

Extends the district covariate table with the market, input-supply and
connectivity variables needed for the market layer, plus the development
controls that layer cannot do without.

Why the controls matter. Mandis, fertiliser shops and cold stores are all built
where there is surplus to trade. Without a development control, any association
between market infrastructure and cropping pattern is partly just "richer, denser,
better-connected districts look different". So this script adds:

  - VIIRS nightlights at district level (the standard SHRUG development proxy)
  - Economic Census 2013 establishment and employment counts (non-farm density)
  - road, transport, bank, electricity and railway access from Antyodaya

New market variables, all as share of a district's villages:
  output markets   mandi, regular market, weekly haat
  input supply     fertiliser shop, government seed centre, soil testing centre,
                   custom hiring centre
  post-harvest     food storage warehouse, farm-gate processing
  institutions     FPO, self-help groups, PDS, common pastures

Outputs:
  outputs/shrug_covariates/market_covariates.csv
  outputs/shrug_covariates/market_build_log.txt
"""
import numpy as np
import pandas as pd

SHRUG = r"D:/SHRUG_2.1_Data/extracted"
OUT = r"D:/crop-diversity/outputs/shrug_covariates"
DKEY = ["pc11_state_id", "pc11_district_id"]
LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


# ------------------------------------------------------------------ keys
key = pd.read_stata(SHRUG + "/shrug-pc-keys-dta/shrid_pc11dist_key.dta")
key = key.dropna(subset=DKEY).drop_duplicates("shrid2")

# ------------------------------------------------------------- antyodaya
BIN = ["mandi", "regular_market", "weekly_haat", "fpo",
       "is_fertilizer_shop_available", "is_govt_seed_centre_available",
       "is_soil_testing_centre_available", "availability_of_custom_hiring_ce",
       "availability_of_food_storage_war", "availability_of_farm_gate_proces",
       "is_common_pastures_available", "livestock_ext_services",
       "is_pds_available", "internal_pucca_road", "public_transport",
       "is_bank_available", "is_atm_available", "availability_of_railway_station",
       "no_electricity", "is_veterinary_hospital_available",
       "availability_of_milk_routes", "csc"]
NUM = ["total_hhd", "total_population", "total_shg", "total_no_of_farmers",
       "total_hhd_having_bpl_cards", "total_hhd_engaged_in_farm_activi"]

ay = pd.read_stata(SHRUG + "/shrug-antyodaya-dta/antyodaya_shrid.dta",
                   columns=["shrid2"] + BIN + NUM)
w("antyodaya shrids: {:,}".format(len(ay)))
ay = ay.merge(key, on="shrid2", how="left").dropna(subset=DKEY)

g = ay.groupby(DKEY)
d = g[BIN + NUM].sum()
d["n_villages"] = g.size()
d = d.reset_index()

SHORT = {
    "is_fertilizer_shop_available": "fert_shop",
    "is_govt_seed_centre_available": "seed_centre",
    "is_soil_testing_centre_available": "soil_test",
    "availability_of_custom_hiring_ce": "custom_hire",
    "availability_of_food_storage_war": "storage",
    "availability_of_farm_gate_proces": "farmgate_proc",
    "is_common_pastures_available": "pasture",
    "livestock_ext_services": "livestock_ext",
    "is_pds_available": "pds",
    "internal_pucca_road": "pucca_road",
    "public_transport": "public_transport",
    "is_bank_available": "bank",
    "is_atm_available": "atm",
    "availability_of_railway_station": "railway",
    "no_electricity": "no_electricity",
    "is_veterinary_hospital_available": "vet",
    "availability_of_milk_routes": "milk_route",
    "csc": "csc",
}
for b in BIN:
    s = SHORT.get(b, b)
    d["m_" + s + "_vshare"] = d[b] / d["n_villages"]
    d["m_" + s + "_n"] = d[b]

d["shg_per_1000hh"] = 1000 * d["total_shg"] / d["total_hhd"]
d["bpl_share"] = d["total_hhd_having_bpl_cards"] / d["total_hhd"]
d["villages_per_100k_pop"] = 1e5 * d["n_villages"] / d["total_population"]
w("antyodaya districts: {}".format(len(d)))

# ------------------------------------------------------------- nightlights
vi = pd.read_stata(SHRUG + "/shrug-viirs-annual-dta/viirs_annual_pc11dist.dta")
w("")
w("viirs district rows: {:,}, years {}-{}, categories {}".format(
    len(vi), int(vi["year"].min()), int(vi["year"].max()),
    sorted(vi["category"].astype(str).unique())[:6]))
# take the latest year, and whichever category is the full-district one
vi["year"] = vi["year"].astype(int)
YR = int(vi["year"].max())
v = vi[vi["year"] == YR].copy()
cat_n = v.groupby(v["category"].astype(str)).size().sort_values(ascending=False)
w("rows per category in {}: {}".format(YR, cat_n.to_dict()))
CAT = cat_n.index[0]
v = v[v["category"].astype(str) == CAT]
v = v.groupby(DKEY, as_index=False).agg(
    viirs_mean=("viirs_annual_mean", "mean"),
    viirs_sum=("viirs_annual_sum", "sum"),
    viirs_cells=("viirs_annual_num_cells", "sum"))
w("viirs districts ({} , category '{}'): {}".format(YR, CAT, len(v)))

# ------------------------------------------------------------ econ census
ec = pd.read_stata(SHRUG + "/shrug-ec13-dta/ec13_pc11dist.dta",
                   columns=DKEY + ["ec13_count_all", "ec13_emp_all"])
ec = ec.groupby(DKEY, as_index=False).sum()
w("economic census districts: {}".format(len(ec)))

# ------------------------------------------------------------------ merge
for t in (v, ec):
    for c in DKEY:
        t[c] = t[c].astype(str).str.strip()
for c in DKEY:
    d[c] = d[c].astype(str).str.strip()

n0 = len(d)
m = d.merge(v, on=DKEY, how="left").merge(ec, on=DKEY, how="left")
assert len(m) == n0, "merge multiplied rows"
w("")
w("merged: {} districts. nightlights missing {}, econ census missing {}.".format(
    len(m), int(m["viirs_mean"].isna().sum()), int(m["ec13_count_all"].isna().sum())))

m["estab_per_1000pop"] = 1000 * m["ec13_count_all"] / m["total_population"]
m["nonfarm_emp_per_1000pop"] = 1000 * m["ec13_emp_all"] / m["total_population"]
m["log_viirs"] = np.log1p(m["viirs_mean"])

# ------------------------------------------------------------------ indices
def z(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / s.std()


GROUPS = {
    "idx_output_market": ["m_mandi_vshare", "m_regular_market_vshare", "m_weekly_haat_vshare"],
    "idx_input_supply": ["m_fert_shop_vshare", "m_seed_centre_vshare",
                         "m_soil_test_vshare", "m_custom_hire_vshare"],
    "idx_postharvest": ["m_storage_vshare", "m_farmgate_proc_vshare"],
    "idx_connectivity": ["m_pucca_road_vshare", "m_public_transport_vshare",
                         "m_bank_vshare", "m_railway_vshare"],
}
for name, cols in GROUPS.items():
    m[name] = pd.concat([z(m[c]) for c in cols], axis=1).mean(axis=1)
    w("{:20s} from {}".format(name, ", ".join(c.replace("m_", "").replace("_vshare", "")
                                              for c in cols)))

keep = DKEY + ["n_villages", "shg_per_1000hh", "bpl_share", "villages_per_100k_pop",
               "viirs_mean", "log_viirs", "ec13_count_all", "ec13_emp_all",
               "estab_per_1000pop", "nonfarm_emp_per_1000pop"] + list(GROUPS)
keep += [c for c in m.columns if c.startswith("m_")]
m[keep].to_csv(OUT + "/market_covariates.csv", index=False)

w("")
w("Village share of districts, mean across {} districts:".format(len(m)))
w("")
w("| facility | mean village share | districts with none |")
w("|---|---|---|")
for b in BIN:
    s = SHORT.get(b, b)
    col = "m_" + s + "_vshare"
    w("| {} | {:.3f} | {} |".format(s, m[col].mean(), int((m[col] == 0).sum())))

with open(OUT + "/market_build_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
print("\nWROTE market_covariates.csv {}".format(m[keep].shape))
