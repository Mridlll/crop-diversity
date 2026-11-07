"""
USAGE
    python scripts/95_data_page_export.py [--out docs/data]

Exports the six JSON files read by docs/data.html (Data and methods page):
    site_data_sources.json    source -> measure-group flows, with variable labels
    site_data_coverage.json   districts in the crop record per crop year (all / the 606)
    site_data_crosswalk.json  one row per district in the crop record: matching method,
                              2011 district, whether it is in the 606
    site_data_funnel.json     the sample funnel 725 -> 606, with dropped districts by state
    site_data_checks.json     cross-source agreement figures (validation, C3/C4/C9)
    site_data_models.json     which controls each analysis holds constant, robustness versions

The sample filter repeats scripts/77_final_results_corrected.py: matched to a SHRUG
district, has an Antyodaya record, Antyodaya coverage >= 25 percent, irrigation share
and D1 present, and at least 10 years of crop data.
"""
import argparse
import json
import os

import pandas as pd

from _years import FIRST_YEAR, LAST_YEAR, N_YEARS, WINDOW_LABEL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "shrug_covariates")
DIV = os.path.join(ROOT, "outputs", "crop_diversity_analysis")

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(ROOT, "docs", "data"))
args = ap.parse_args()


def dump(name, obj):
    p = os.path.join(args.out, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    print("wrote", p, os.path.getsize(p), "bytes")


# ------------------------------------------------------------------ sample
corr = pd.read_csv(os.path.join(DIV, "district_diversity_indices_corrected.csv"))
cw = pd.read_csv(os.path.join(OUT, "district_crosswalk.csv"))
sh = pd.read_csv(os.path.join(OUT, "shrug_district_covariates.csv"))

p = corr.merge(cw, left_on="district_key", right_on="div_key", how="left")
p = p.merge(sh.rename(columns={"district_key": "shrug_key", "state_name": "sh_state",
                               "district_name": "sh_district"}),
            on="shrug_key", how="left")
s0 = pd.Series(True, index=p.index)
s1 = p["shrug_key"].notna()
noay = p["flag_no_antyodaya"].astype(object).where(p["flag_no_antyodaya"].notna(), True).astype(bool)
low = p["flag_low_ay_coverage"].astype(object).where(p["flag_low_ay_coverage"].notna(), True).astype(bool)
s2 = s1 & ~noay
s3 = s2 & ~low
s4 = s3 & p["irr_share"].notna() & p["D1_exp_shannon"].notna()
s5 = s4 & (p["n_years"] >= 10)
assert s5.sum() == 606, s5.sum()
assert s4.sum() == s3.sum(), "irrigation/D1 filter is expected to drop nobody"
in606 = set(p.loc[s5, "district_key"])


def title(s):
    small = {"and", "of", "the"}
    out = []
    for i, w in enumerate(str(s).lower().split()):
        out.append(w if (i and w in small) else w[:1].upper() + w[1:])
    return " ".join(out)


def by_state(df, extra=None):
    g = []
    for st, d in df.groupby("state_name"):
        rows = []
        for _, r in d.sort_values("district_name").iterrows():
            row = {"d": title(r["district_name"])}
            if extra:
                row.update(extra(r))
            rows.append(row)
        g.append({"state": title(st), "districts": rows})
    g.sort(key=lambda x: (-len(x["districts"]), x["state"]))
    return g


def cov_ratio(r):
    a, v = r.get("ay_n_villages"), r.get("vd_n_villages")
    if pd.notna(a) and pd.notna(v) and v > 0:
        return {"cov": round(float(a) / float(v), 3)}
    return {}


steps = [
    {"label": "in the crop record", "rule": "", "n": int(s0.sum()), "dropped": []},
    {"label": "match a 2011 census district", "rule": "unmatched",
     "n": int(s1.sum()), "dropped": by_state(p[s0 & ~s1])},
    {"label": "have a Mission Antyodaya record", "rule": "no village survey record",
     "n": int(s2.sum()), "dropped": by_state(p[s1 & ~s2])},
    {"label": "survey reaches at least a quarter of villages", "rule": "survey reaches under a quarter of villages",
     "n": int(s3.sum()), "dropped": by_state(p[s2 & ~s3], cov_ratio)},
    {"label": "at least ten years of crop data", "rule": "under ten years of crop data",
     "n": int(s5.sum()), "dropped": by_state(p[s4 & ~s5], lambda r: {"y": int(r["n_years"])})},
]
short = p[s4 & ~s5]
funnel = {
    "steps": steps,
    "short_new": int((short["method"] == "parent_map").sum()),
    "keys606": sorted(in606),
}
dump("site_data_funnel.json", funnel)

# ------------------------------------------------------------------ crosswalk
cwrows = []
for _, r in p.iterrows():
    par = r["shrug_key"]
    cwrows.append({
        "k": r["district_key"], "s": title(r["state_name"]), "d": title(r["district_name"]),
        "m": r["method"] if pd.notna(r["method"]) else "unmatched",
        "p": title(str(par).split("|")[1]) if pd.notna(par) else "",
        "ps": title(str(par).split("|")[0]) if pd.notna(par) else "",
        "in": 1 if r["district_key"] in in606 else 0,
        "y": int(r["n_years"]),
    })
dump("site_data_crosswalk.json", cwrows)

# ------------------------------------------------------------------ coverage
a = pd.read_csv(os.path.join(DIV, "district_year_diversity_panel_corrected.csv"),
                usecols=["district_key", "year_start"])
a = a[a["year_start"] <= 2019]
years = list(range(FIRST_YEAR, LAST_YEAR + 1))
n_all = a.groupby("year_start")["district_key"].nunique().reindex(years, fill_value=0)
n_606 = a[a["district_key"].isin(in606)].groupby("year_start")["district_key"].nunique().reindex(years, fill_value=0)
ny = p.loc[s5, "n_years"]
coverage = {
    "years": years, "n_all": [int(x) for x in n_all], "n_606": [int(x) for x in n_606],
    "years_606": {"all_years": int((ny == N_YEARS).sum()), "all_but_one": int((ny >= N_YEARS - 1).sum()),
                  "lt20": int((ny < 20).sum()), "median": float(ny.median())},
    "sources": [
        {"name": "Census village directory", "year": 2011, "label": "2011"},
        {"name": "Census abstract", "year": 2011, "label": "2011"},
        {"name": "Socio-Economic and Caste Census", "year": 2012, "label": "2012"},
        {"name": "Economic Census", "year": 2013, "label": "2013"},
        {"name": "Mission Antyodaya", "year": 2019, "label": "2019–20"},
        {"name": "Night lights", "year": 2023, "label": "2023"},
    ],
}
dump("site_data_coverage.json", coverage)

# ------------------------------------------------------------------ sources
pv = pd.read_csv(os.path.join(OUT, "provenance.csv"))
assert len(pv) == 93

SRC_OF_DERIVED = {
    "vd_irr_share": "vd", "cultivator_share_agwork": "pca", "secc_landless_share": "secc",
}
for v in pv.loc[pv.module == "derived", "variable"]:
    if v.startswith("vd_src_"):
        SRC_OF_DERIVED[v] = "vd"
MOD = {"shrug-antyodaya-dta": "ay", "shrug-vd11-dta": "vd", "shrug-pca11-dta": "pca",
       "shrug-secc-mord-rural-dta": "secc"}

LBL = {
    "ay_total_cultivable_area_in_hac": ("cultivable area", "farm"),
    "ay_net_sown_area_in_hac": ("net sown area", "land"),
    "ay_net_sown_area_kharif_in_hac": ("kharif sown area", "land"),
    "ay_net_sown_area_rabi_in_hac": ("rabi sown area", "land"),
    "ay_net_sown_area_other_in_hac": ("other-season sown area", "land"),
    "ay_area_irrigated_in_hac": ("irrigated area", "land"),
    "ay_total_unirrigated_land_area_in_h": ("unirrigated area", "land"),
    "ay_total_no_of_farmers": ("number of farmers", "farm"),
    "ay_total_no_farmers_adopted_organic": ("farmers practising organic farming", "mkt"),
    "ay_no_of_farmers_using_drip_sprinkl": ("farmers using drip or sprinkler", "land"),
    "ay_total_hhd": ("households", "work"),
    "ay_total_hhd_engaged_in_farm_activi": ("households in farm work", "work"),
    "ay_total_population": ("population", "work"),
    "ay_mandi_n": ("villages with a mandi", "mkt"),
    "ay_regular_market_n": ("villages with a regular market", "mkt"),
    "ay_weekly_haat_n": ("villages with a weekly haat", "mkt"),
    "ay_fpo_n": ("villages with a producer organisation", "mkt"),
    "ay_canal_n": ("villages irrigated mainly by canal", "land"),
    "ay_surface_water_n": ("villages irrigated mainly by surface water", "land"),
    "ay_ground_water_n": ("villages irrigated mainly by groundwater", "land"),
    "ay_other_irrigation_n": ("villages irrigated mainly by other sources", "land"),
    "ay_is_soil_testing_centre_available_n": ("villages with soil testing", "mkt"),
    "ay_is_fertilizer_shop_available_n": ("villages with a fertiliser shop", "mkt"),
    "ay_is_govt_seed_centre_available_n": ("villages with a seed centre", "mkt"),
    "ay_availability_of_food_storage_war_n": ("villages with a food storage warehouse", "mkt"),
    "ay_availability_of_farm_gate_proces_n": ("villages with farm-gate processing", "mkt"),
    "ay_availability_of_custom_hiring_ce_n": ("villages with custom hiring of machinery", "mkt"),
    "ay_is_common_pastures_available_n": ("villages with common pasture", "land"),
    "ay_livestock_ext_services_n": ("villages with livestock extension", "mkt"),
    "pc11_vd_land_nt_swn": ("net sown area", "land"),
    "pc11_vd_land_un_irr": ("unirrigated area", "land"),
    "pc11_vd_land_src_irr": ("irrigated area", "land"),
    "pc11_vd_land_canal_irr": ("area irrigated by canal", "land"),
    "pc11_vd_land_wl_tw_irr": ("area irrigated by well and tubewell", "land"),
    "pc11_vd_land_tnk_lk_irr": ("area irrigated by tank and lake", "land"),
    "pc11_vd_land_w_fall_irr": ("area irrigated by waterfall", "land"),
    "pc11_vd_land_oth_src_irr": ("area irrigated by other sources", "land"),
    "pc11_vd_land_fores": ("forest area", "land"),
    "pc11_vd_land_non_agri": ("land in non-farm use", "land"),
    "pc11_vd_land_cur_fal": ("current fallow", "land"),
    "pc11_vd_land_fallow": ("other fallow", "land"),
    "pc11_vd_land_cult_waste": ("cultivable waste", "land"),
    "pc11_vd_land_pst_grz": ("pasture and grazing land", "land"),
    "pc11_vd_power_agr": ("power supply for agriculture", "land"),
    "pc11_vd_comm_bank": ("villages with a commercial bank", "dev"),
    "pc11_vd_coop_bank": ("villages with a cooperative bank", "dev"),
    "pc11_pca_tot_p": ("population", "work"),
    "pc11_pca_p_sc": ("Scheduled Caste population", "work"),
    "pc11_pca_p_st": ("Scheduled Tribe population", "work"),
    "pc11_pca_main_cl_p": ("main cultivators", "work"),
    "pc11_pca_main_al_p": ("main agricultural labourers", "work"),
    "pc11_pca_marg_cl_p": ("marginal cultivators", "work"),
    "pc11_pca_marg_al_p": ("marginal agricultural labourers", "work"),
    "pc11_pca_tot_work_p": ("workers", "work"),
    "pc11_pca_no_hh": ("households", "work"),
    "secc_land_own_share": ("share of households owning land", "farm"),
    "secc_sc_share": ("Scheduled Caste share", "work"),
    "secc_st_share": ("Scheduled Tribe share", "work"),
    "secc_nco2d_cultiv_share": ("share working as cultivators", "work"),
    "secc_inc_source_cultiv_share": ("share earning mainly from cultivation", "work"),
    "secc_inc_source_manlab_share": ("share earning mainly from manual labour", "work"),
    "secc_kisan_cc": ("share holding a Kisan credit card", "dev"),
    "secc_unirr_land_acre_sum": ("unirrigated land owned", "farm"),
    "secc_two_crop_acre_sum": ("land owned that carries two crops", "farm"),
    "secc_other_irr_acre_sum": ("other irrigated land owned", "farm"),
    "gca_ha": ("gross cropped area", "land"),
    "cropping_intensity": ("cropping intensity", "land"),
    "irr_share_gca": ("irrigated share of cropped area", "land"),
    "irr_share_nsa": ("irrigated share of sown area", "land"),
    "vd_irr_share": ("irrigated share of sown area, 2011", "land"),
    "ay_src_canal_vshare": ("share of villages irrigated by canal", "land"),
    "ay_src_ground_vshare": ("share of villages irrigated by groundwater", "land"),
    "ay_src_surface_vshare": ("share of villages irrigated by surface water", "land"),
    "ay_src_other_vshare": ("share of villages irrigated by other sources", "land"),
    "vd_src_canal_ashare": ("canal share of irrigated area", "land"),
    "vd_src_tubewell_ashare": ("well and tubewell share of irrigated area", "land"),
    "vd_src_tank_ashare": ("tank share of irrigated area", "land"),
    "vd_src_waterfall_ashare": ("waterfall share of irrigated area", "land"),
    "vd_src_othersrc_ashare": ("other-source share of irrigated area", "land"),
    "mandi_vshare": ("share of villages with a mandi", "mkt"),
    "regular_market_vshare": ("share of villages with a regular market", "mkt"),
    "weekly_haat_vshare": ("share of villages with a weekly haat", "mkt"),
    "fpo_vshare": ("share of villages with a producer organisation", "mkt"),
    "storage_vshare": ("share of villages with a food storage warehouse", "mkt"),
    "farmgate_proc_vshare": ("share of villages with farm-gate processing", "mkt"),
    "fert_shop_vshare": ("share of villages with a fertiliser shop", "mkt"),
    "soil_test_vshare": ("share of villages with soil testing", "mkt"),
    "custom_hire_vshare": ("share of villages with custom hiring", "mkt"),
    "livestock_ext_vshare": ("share of villages with livestock extension", "mkt"),
    "mean_holding_ha": ("mean holding size", "farm"),
    "organic_farmer_share": ("share of farmers practising organic farming", "mkt"),
    "cultivator_share_agwork": ("cultivators as a share of farm workers", "work"),
    "secc_landless_share": ("share of households without land", "farm"),
}
for _v in list(LBL):
    if _v.startswith("ay_") and _v.endswith("_n"):
        LBL[_v] = ("number of " + LBL[_v][0], LBL[_v][1])
missing = set(pv.variable) - set(LBL)
assert not missing, missing

sources = [
    {"id": "crop", "name": "Crop statistics", "year": "1998–99 to 2019–20", "level": "district"},
    {"id": "ay", "name": "Mission Antyodaya", "year": "2019–20", "level": "village"},
    {"id": "vd", "name": "Census village directory", "year": "2011", "level": "village"},
    {"id": "pca", "name": "Census abstract", "year": "2011", "level": "village"},
    {"id": "secc", "name": "SECC", "year": "2012", "level": "village"},
    {"id": "ec", "name": "Economic Census", "year": "2013", "level": "district"},
    {"id": "nl", "name": "Night lights", "year": "2023", "level": "district"},
]
groups = [
    {"id": "div", "name": "crop diversity"},
    {"id": "land", "name": "irrigation and land use"},
    {"id": "mkt", "name": "markets, inputs and post-harvest"},
    {"id": "work", "name": "workforce and caste"},
    {"id": "farm", "name": "land ownership and farm size"},
    {"id": "dev", "name": "development measures"},
]
flows = {}


def add(s, g, label, in93=True):
    flows.setdefault((s, g), {"s": s, "g": g, "n": 0, "n93": 0, "vars": []})
    f = flows[(s, g)]
    f["n"] += 1
    f["n93"] += 1 if in93 else 0
    f["vars"].append(label)


for _, r in pv.iterrows():
    v = r["variable"]
    src = MOD.get(r["module"]) or SRC_OF_DERIVED.get(v, "ay")
    lab, grp = LBL[v]
    add(src, grp, lab)
for lab in ["count of crops", "effective number of crops", "effective number weighted to the main crops",
            "evenness", "cereal share", "pulse share", "oilseed share", "fibre crop share", "fruit share",
            "vegetable share", "spice share", "sugar share", "fodder share", "drugs and narcotics share"]:
    add("crop", "div", lab, in93=False)
for lab in ["share of villages with a paved road", "share of villages with public transport",
            "share of villages with a bank", "share of villages with a railway station"]:
    add("ay", "dev", lab, in93=False)
add("ec", "dev", "non-farm businesses per 1,000 people", in93=False)
add("nl", "dev", "night-time brightness, log", in93=False)
for s in sources:
    s["n_vars"] = sum(f["n"] for f in flows.values() if f["s"] == s["id"])
    s["n93"] = sum(f["n93"] for f in flows.values() if f["s"] == s["id"])
assert sum(s["n93"] for s in sources) == 93
dump("site_data_sources.json", {"sources": sources, "groups": groups, "flows": list(flows.values()),
                               "modules_direct": {k: int(v) for k, v in pv.module.value_counts().items()}})

# ------------------------------------------------------------------ checks
# from the validation run (scripts/71): C3, C4 and C9. The state-level comparisons
# (C5, C6) rest on reference values that have not been confirmed and are left out.
checks = [
    {"measure": "Irrigated share", "against": "Mission Antyodaya 2019–20 against census village records 2011",
     "stat": "rho", "value": 0.812, "r": 0.816, "rho": 0.812, "n": 582, "mean_a": 0.554, "mean_b": 0.474,
     "la": "survey", "lb": "census"},
    {"measure": "Scheduled Caste share", "against": "census 2011 against SECC 2012",
     "stat": "r", "value": 0.942, "r": 0.942, "n": 609, "mean_a": 0.151, "mean_b": 0.051,
     "la": "census", "lb": "SECC"},
    {"measure": "Scheduled Tribe share", "against": "census 2011 against SECC 2012",
     "stat": "r", "value": 0.990, "r": 0.990, "n": 609, "mean_a": 0.183, "mean_b": 0.065,
     "la": "census", "lb": "SECC"},
    {"measure": "Irrigated share", "against": "village records summed here against SHRUG's district file",
     "stat": "r", "value": 0.9923, "r": 0.9923, "n": 604},
    {"measure": "Cropping intensity", "against": "village records summed here against SHRUG's district file",
     "stat": "r", "value": 0.9955, "r": 0.9955, "n": 604},
    {"measure": "Mean holding size", "against": "village records summed here against SHRUG's district file",
     "stat": "r", "value": 0.9904, "r": 0.9904, "n": 603},
]
dump("site_data_checks.json", checks)

# ------------------------------------------------------------------ models
rb = pd.read_csv(os.path.join(OUT, "final_robustness.csv"))
rb = rb[rb.spec.str.startswith("D1:")]
NAMES = {
    "D1: state fixed effects": "within state, the main estimate",
    "D1: PC11 VD irrigation": "census 2011 irrigated share",
    "D1: drop shared parents": "without the 64 districts that share a 2011 district",
    "D1: inverse-parent weighted": "weighted by one over the districts sharing a 2011 district",
    "D1: drop source conflicts": "without districts where the two irrigation sources differ by more than 0.30",
    "D1: full panel only": "districts missing at most one year",
    "D1: area weighted": "weighted by cropped area",
    "D1: no fixed effects": "without the within-state comparison",
}
rob = [{"name": NAMES[r.spec], "n": int(r.n), "tp": round(float(r.tp), 3), "p2": float(r.p2),
        "r2": round(float(r.r2), 3)} for r in rb.itertuples()]
cols = [
    {"id": "irr", "name": "irrigated share and its square",
     "def": "Share of sown area irrigated, from the village survey, with its square so the line can rise and then fall."},
    {"id": "farm", "name": "average farm size",
     "def": "Cultivable area divided by the number of farmers, on a log scale."},
    {"id": "st", "name": "Scheduled Tribe share",
     "def": "Share of the population from Scheduled Tribes, census 2011."},
    {"id": "nl", "name": "night lights",
     "def": "Night-time brightness seen from satellite, 2023, on a log scale."},
    {"id": "ec", "name": "non-farm businesses",
     "def": "Establishments outside farming per 1,000 people, Economic Census 2013."},
    {"id": "conn", "name": "connectivity average",
     "def": "Average of standardised shares of villages with a paved road, public transport, a bank and a railway station."},
    {"id": "fe", "name": "within state",
     "def": "Each state gets its own intercept, so a district is set only against districts in the same state."},
    {"id": "cl", "name": "grouped by 2011 district",
     "def": "Districts carved from the same 2011 district are treated as one group when measuring uncertainty."},
]
rows = [
    {"name": "irrigation curve", "n": 606, "has": ["irr", "fe", "cl"],
     "note": "Effective number of crops on irrigated share and its square. Across eight versions of the model and sample the peak runs from 0.245 to 0.513."},
    {"name": "irrigation source", "n": 606, "has": ["irr", "farm", "st", "fe", "cl"],
     "note": "Canal and surface-water shares of villages, groundwater left out as the comparison."},
    {"name": "single district measures", "n": 606, "has": ["irr", "farm", "fe", "cl"],
     "note": "One district measure at a time, first on its own and then holding irrigation and farm size constant within state."},
    {"name": "facility comparisons", "n": 606, "has": ["irr", "farm", "st", "nl", "ec", "conn", "fe", "cl"],
     "note": "One facility at a time. Benjamini-Hochberg correction across the ten crop categories."},
]
dump("site_data_models.json", {"cols": cols, "rows": rows, "robust": rob,
                               "r2": {"none": 0.065, "fe_only": 0.49, "fe_irr": 0.530},
                               "tp": {"none": 0.513, "fe": 0.282}})
print("states in 606:", p.loc[s5, "state_name"].nunique())
