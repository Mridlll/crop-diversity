"""
121_census_irrigation_panel.py

District irrigation-by-source panel from the SHRUG village directories
(Census of India 1991, 2001, 2011), on constant 2011 (PC11) district boundaries.

Unit of aggregation
  SHRUG 2.1 stores all three village directories on shrid2 units, and every shrid2
  is a PC11 geography (99% of 1991/2001 shrid2 appear in shrid_pc11dist_key). So a
  village-directory row is assigned to its PC11 district through shrid_pc11dist_key
  and areas are SUMMED. No boundary re-weighting is needed or applied.

Denominator
  irrigated share = irrigated area / (irrigated area + unirrigated area).
  This is net irrigated area over net area sown, which is what the DES land-use
  net-irrigated / net-sown ratio and the Antyodaya share in scripts 72 measure.
  All areas are hectares in all three rounds (labels: "in Hectares").

Source harmonisation (see MAPPING below; written verbatim into the audit)
  canal        1991, 2001: canal_govt + canal_pvt             2011: land_canal_irr
  groundwater  1991: well_wo_el + well_w_el + tw_wo_el + tw_w_el
               2001: well_wo_el + tw_wo_el + tw_w_el  (no "well with electricity" field)
               2011: wl_tw_irr (wells and tube wells are lumped)
  gw_elec      1991: well_w_el + tw_w_el    2001: tw_w_el only (partial)   2011: n/a
  tank_lake    1991, 2001: tank_irr + lake_irr                2011: tnk_lk_irr
  river        1991, 2001: river_irr                          2011: n/a (not a category)
  other_misc   1991, 2001: w_fall + oth_irr (+ river in river_other)
               2011: w_fall_irr + oth_src_irr
  river_other  river + waterfall + other, the only river-containing set comparable
               across all three rounds
  unattributed total irrigated minus the listed sources (>=0); tracks what the
               source columns fail to explain, mainly the missing 2001 electrified wells

Outputs (outputs/revision_nov8/census_irrigation/)
  census_irrigation_panel.csv        district x round, long
  census_irrigation_panel_wide.csv   one row per district
  census_irrigation_state_round.csv  state x round coverage and flags
  census_irrigation_audit.md
  census_irrigation_log.txt
"""
import os
import sys
import numpy as np
import pandas as pd

R = "D:/SHRUG_2.1_Data/extracted/"
REPO = "D:/crop-diversity/"
OUT = REPO + "outputs/revision_nov8/census_irrigation/"
os.makedirs(OUT, exist_ok=True)

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


# ------------------------------------------------------------------ thresholds
PLAUS_MULT = 1.5          # village dropped if irrigated+unirrigated > 1.5 x reference area
MIN_AREA_COV = 0.60       # state-round needs >= this share of PC11 rural area with valid data
MIN_DIST_COV = 0.50       # district-round needs >= this share of its PC11 rural area covered
MAX_ZERO_IRR = 0.97       # state-round with >97% of valid villages at zero irrigation = zeros everywhere
MAX_DROPPED = 0.10        # state-round with >10% of valid-area implausible villages
MAX_GAP_DES = 0.25        # census vs DES mean gap on the same districts
MIN_DES_DISTRICTS = 3     # DES check needs this many matched districts in the state
MAX_GAP_AY = 0.25         # census 2011 vs Antyodaya 2019-20, only where DES cannot be matched
MAX_JUMP_0111 = 0.30      # 2001 vs 2011 share, only where DES cannot be matched and the other round passes DES
MAX_JUMP_9101 = 0.25
MIN_RATIO_9101 = 0.5      # 1991 share below 0.5x (or above 2x) the clean 2001 share, with a gap over 0.05

# ------------------------------------------------------------------ keys
log("=" * 78)
log("KEYS")
key = pd.read_stata(R + "shrug-pc-keys-dta/shrid_pc11dist_key.dta")
key = key.dropna(subset=["pc11_state_id", "pc11_district_id"]).drop_duplicates("shrid2")
rk = pd.read_stata(R + "shrug-pc-keys-dta/pc11r_shrid_key.dta")[["shrid2", "pc11_land_area"]]
# a shrid2 can fuse several PC11 villages (duplicate rows in the key): the village-directory row covers them all, so sum area
rk = rk.groupby("shrid2", as_index=False).pc11_land_area.sum(min_count=1)
key = key.merge(rk, on="shrid2", how="left")
key["st"] = key.pc11_state_id.astype(int)
key["dt"] = key.pc11_district_id.astype(int)
key["ref_ha_pc11"] = key.pc11_land_area * 100.0      # km2 -> ha
log("shrids with district key: {:,}; districts: {}".format(len(key), key.groupby(["st", "dt"]).ngroups))

cov = pd.read_csv(REPO + "outputs/shrug_covariates/shrug_district_covariates.csv")
cov["st"] = cov.pc11_state_id.astype(int)
cov["dt"] = cov.pc11_district_id.astype(int)
names = cov[["st", "dt", "state_name", "district_name", "district_key"]].rename(
    columns={"district_key": "shrug_key"})
assert names.duplicated(["st", "dt"]).sum() == 0
univ = key[key.shrid2.isin(rk.shrid2)].copy()      # rural shrid universe (those with a PC11 rural key)
log("rural shrid universe: {:,}".format(len(univ)))
univ["area_u"] = univ.pc11_land_area.fillna(0) * 100.0

# ------------------------------------------------------------------ MAPPING
MAPPING = [
    # category, 1991, 2001, 2011
    ("canal", "canal_govt + canal_pvt", "canal_govt + canal_pvt", "land_canal_irr"),
    ("groundwater", "well_wo_el + well_w_el + tw_wo_el + tw_w_el",
     "well_wo_el + tw_wo_el + tw_w_el (no well-with-electricity field exists in the 2001 file)",
     "land_wl_tw_irr (wells and tube wells lumped)"),
    ("gw_elec (electrified wells and tube wells)", "well_w_el + tw_w_el",
     "tw_w_el only (partial: electrified open wells are not recorded)", "not recorded"),
    ("tank_lake", "tank_irr + lake_irr", "tank_irr + lake_irr", "land_tnk_lk_irr (lumped)"),
    ("river", "river_irr", "river_irr", "not a category (falls in 'other')"),
    ("other_misc (waterfall + other)", "w_fall + oth_irr", "w_fall + oth_irr",
     "land_w_fall_irr + land_oth_src_irr"),
    ("river_other (river + waterfall + other)", "river_irr + w_fall + oth_irr",
     "river_irr + w_fall + oth_irr", "land_w_fall_irr + land_oth_src_irr"),
    ("irrigated total", "tot_irr; if missing, sum of the source columns",
     "tot_irr; if missing, sum of the source columns", "max(land_src_irr, sum of sources)"),
    ("unirrigated", "un_irr", "un_irr", "land_un_irr"),
    ("unattributed", "tot - sum(listed sources), floored at 0",
     "tot - sum(listed sources), floored at 0", "src_irr - sum(listed sources), floored at 0"),
]

# ------------------------------------------------------------------ village tables
def num(d, c):
    return pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series(np.nan, index=d.index)


def load_round(yr):
    if yr == 1991:
        d = pd.read_stata(R + "shrug-vd91-dta/pc91_vd_clean_shrid.dta")
        p = "pc91_vd_"
        s = {c: num(d, p + c) for c in ["canal_govt", "canal_pvt", "well_wo_el", "well_w_el", "tw_wo_el",
                                        "tw_w_el", "tank_irr", "lake_irr", "river_irr", "w_fall", "oth_irr"]}
        tot_f, un, own = num(d, p + "tot_irr"), num(d, p + "un_irr"), num(d, p + "area")
        groups = dict(
            canal=["canal_govt", "canal_pvt"],
            gw=["well_wo_el", "well_w_el", "tw_wo_el", "tw_w_el"],
            gw_elec=["well_w_el", "tw_w_el"],
            tank=["tank_irr", "lake_irr"], river=["river_irr"], other=["w_fall", "oth_irr"])
    elif yr == 2001:
        d = pd.read_stata(R + "shrug-vd01-dta/pc01_vd_clean_shrid.dta")
        p = "pc01_vd_"
        s = {c: num(d, p + c) for c in ["canal_govt", "canal_pvt", "well_wo_el", "tw_wo_el",
                                        "tw_w_el", "tank_irr", "lake_irr", "river_irr", "w_fall", "oth_irr"]}
        tot_f, un, own = num(d, p + "tot_irr"), num(d, p + "un_irr"), num(d, p + "area")
        groups = dict(
            canal=["canal_govt", "canal_pvt"],
            gw=["well_wo_el", "tw_wo_el", "tw_w_el"],
            gw_elec=["tw_w_el"],
            tank=["tank_irr", "lake_irr"], river=["river_irr"], other=["w_fall", "oth_irr"])
    else:
        d = pd.read_stata(R + "shrug-vd11-dta/pc11_vd_clean_shrid.dta")
        p = "pc11_vd_land_"
        s = {"canal": num(d, p + "canal_irr"), "gw": num(d, p + "wl_tw_irr"),
             "tank": num(d, p + "tnk_lk_irr"), "w_fall": num(d, p + "w_fall_irr"),
             "oth": num(d, p + "oth_src_irr")}
        tot_f, un, own = num(d, p + "src_irr"), num(d, p + "un_irr"), num(d, "pc11_vd_area")
        groups = None
    v = pd.DataFrame({"shrid2": d.shrid2})
    srcs = pd.DataFrame(s)
    if yr != 2011:
        ssum = srcs.sum(axis=1, min_count=1)
    else:
        ssum = srcs.sum(axis=1, min_count=1)
    # Total irrigated = the larger of the reported total and the sum of the source fields (NaN-safe). The 1991 directory
    # carries a total of zero or blank in many villages whose source fields are filled; the source fields then exceed the total.
    tot = pd.Series(np.fmax(tot_f.values, ssum.values), index=tot_f.index)
    repaired = (ssum > tot_f.fillna(0) + 0.5)
    v_repair = (ssum - tot_f.fillna(0)).where(repaired, 0)
    log("  total repaired from source sum in {:,} villages ({:.1%} of villages with a source), adding {:.2f} Mha "
        "({:.1%} of the reported total)".format(int(repaired.sum()), repaired.sum() / max(ssum.notna().sum(), 1),
                                                v_repair.sum() / 1e6, v_repair.sum() / max(tot_f.sum(), 1)))
    v["raw_tot_irr"], v["raw_un"], v["own_area"] = tot_f, un, own
    v["tot_irr"], v["un_irr"] = tot, un
    zero = srcs.fillna(0)
    if yr != 2011:
        for g, cols in groups.items():
            v[g] = zero[cols].sum(axis=1)
        v["other_misc"] = v["other"]
        v["river_other"] = v["river"] + v["other"]
        v["tank_lake"] = v["tank"]
        v["groundwater"] = v["gw"]
        v["gw_elec"] = v["gw_elec"]
        v = v.drop(columns=["gw", "tank", "other"])
        listed = v[["canal", "groundwater", "tank_lake", "river", "other_misc"]].sum(axis=1)
        v["river"] = v["river"]
    else:
        v["canal"] = zero["canal"]
        v["groundwater"] = zero["gw"]
        v["gw_elec"] = np.nan
        v["tank_lake"] = zero["tank"]
        v["river"] = np.nan
        v["other_misc"] = zero["w_fall"] + zero["oth"]
        v["river_other"] = v["other_misc"]
        listed = v[["canal", "groundwater", "tank_lake", "other_misc"]].sum(axis=1)
    v["unattributed"] = (v["tot_irr"] - listed).clip(lower=0)
    # sources only meaningful where total is observed; leave NaN elsewhere
    return v


def prepare(yr):
    log("")
    log("=" * 78)
    log("ROUND {}".format(yr))
    v = load_round(yr)
    n_file = len(v)
    v = v.merge(key[["shrid2", "st", "dt", "ref_ha_pc11"]], on="shrid2", how="left")
    log("village rows {:,}; without PC11 district key {:,} ({:.2%})".format(
        n_file, v.st.isna().sum(), v.st.isna().mean()))
    v = v.dropna(subset=["st"]).copy()
    v["st"], v["dt"] = v.st.astype(int), v.dt.astype(int)
    v = v.merge(univ[["shrid2", "area_u"]], on="shrid2", how="left")
    v["area_u"] = v.area_u.fillna(0.0)               # urban-only shrids contribute no rural area
    ref = np.fmax(v.own_area.fillna(0), v.ref_ha_pc11.fillna(0))
    v["has_land"] = v.tot_irr.notna() & v.un_irr.notna()
    v["area_irr_un"] = v.tot_irr.fillna(0) + v.un_irr.fillna(0)
    v["implausible"] = v.has_land & (v.area_irr_un > PLAUS_MULT * ref) & (v.area_irr_un > 5)
    v["valid"] = v.has_land & ~v.implausible
    v["in_univ"] = v.area_u > 0
    log("rows with non-missing irrigated and unirrigated: {:,} ({:.1%})".format(
        v.has_land.sum(), v.has_land.mean()))
    log("rows implausible (irr+unirr > {}x reference area): {:,} ({:.2%} of has_land)".format(
        PLAUS_MULT, v.implausible.sum(), v.implausible.sum() / max(v.has_land.sum(), 1)))
    return v


# ------------------------------------------------------------------ aggregation
SRC = ["canal", "groundwater", "gw_elec", "tank_lake", "river", "other_misc", "river_other", "unattributed"]


def aggregate(v, by):
    g = v.groupby(by)
    out = pd.DataFrame({
        "n_villages_file": g.size(),
        "n_has_land": g.has_land.sum(),
        "n_implausible": g.implausible.sum(),
        "n_valid": g.valid.sum(),
    })
    out["area_valid_ha"] = v[v.valid].groupby(by).area_u.sum()
    out["area_nonmissing_ha"] = v[v.has_land].groupby(by).area_u.sum()
    out["area_implausible_ha"] = v[v.implausible].groupby(by).area_u.sum()
    vv = v[v.valid]
    gv = vv.groupby(by)
    out["irr_ha"] = gv.tot_irr.sum()
    out["unirr_ha"] = gv.un_irr.sum()
    for c in SRC:
        out[c + "_ha"] = gv[c].sum(min_count=1) if c not in ("river", "gw_elec") or vv[c].notna().any() else np.nan
    out["n_valid_zero_irr"] = vv[vv.tot_irr == 0].groupby(by).size()
    out["n_valid_pos_area"] = vv[vv.area_irr_un > 0].groupby(by).size()
    out["n_valid_zero_irr_posarea"] = vv[(vv.tot_irr == 0) & (vv.area_irr_un > 0)].groupby(by).size()
    out["n_src_gt_tot"] = vv[(vv.canal + vv.groundwater + vv.tank_lake + vv.other_misc +
                              vv.river.fillna(0)) > vv.tot_irr * 1.01 + 0.5].groupby(by).size()
    out = out.fillna({"n_valid_zero_irr": 0, "n_valid_pos_area": 0, "n_valid_zero_irr_posarea": 0, "n_src_gt_tot": 0,
                      "area_valid_ha": 0, "area_nonmissing_ha": 0, "area_implausible_ha": 0,
                      "irr_ha": 0, "unirr_ha": 0})
    return out


def shares(df):
    den = df.irr_ha + df.unirr_ha
    df["irr_unirr_ha"] = den
    df["irr_share"] = np.where(den > 0, df.irr_ha / den, np.nan)
    for c in SRC:
        df["sh_" + c] = np.where(den > 0, df[c + "_ha"] / den, np.nan)
    for c in SRC:
        df["comp_" + c] = np.where(df.irr_ha > 0, df[c + "_ha"] / df.irr_ha, np.nan)
    return df


rounds = {}
dist_rows, state_rows, vil = [], [], {}
for yr in (1991, 2001, 2011):
    v = prepare(yr)
    vil[yr] = v
    a = shares(aggregate(v, ["st", "dt"])).reset_index()
    a["round"] = yr
    dist_rows.append(a)
    s = shares(aggregate(v, ["st"])).reset_index()
    s["round"] = yr
    state_rows.append(s)

# universe (all PC11 rural shrids, constant across rounds)
U_d = univ.groupby(["st", "dt"]).agg(n_univ_shrids=("shrid2", "size"), area_univ_ha=("area_u", "sum")).reset_index()
U_s = univ.groupby("st").agg(n_univ_shrids=("shrid2", "size"), area_univ_ha=("area_u", "sum")).reset_index()

D = pd.concat(dist_rows, ignore_index=True)
Sx = pd.concat(state_rows, ignore_index=True)
# make a complete district x round grid so absent districts are explicit
grid = U_d.merge(pd.DataFrame({"round": [1991, 2001, 2011]}), how="cross")
D = grid.merge(D, on=["st", "dt", "round"], how="left")
D = D.merge(names, on=["st", "dt"], how="left")
Sg = U_s.merge(pd.DataFrame({"round": [1991, 2001, 2011]}), how="cross")
Sx = Sg.merge(Sx, on=["st", "round"], how="left")
st_names = names.groupby("st").state_name.first().reset_index()
Sx = Sx.merge(st_names, on="st", how="left")
for df in (D, Sx):
    for c in ["n_villages_file", "n_has_land", "n_valid", "area_valid_ha", "area_nonmissing_ha"]:
        df[c] = df[c].fillna(0)
    df["cov_villages"] = df.n_has_land / df.n_univ_shrids
    df["cov_area"] = df.area_valid_ha / df.area_univ_ha
    df["cov_area_nonmissing"] = df.area_nonmissing_ha / df.area_univ_ha

# ------------------------------------------------------------------ audit of Sx
log("")
log("=" * 78)
log("STATE x ROUND COVERAGE (before validation flags)")
Sx["zero_irr_share_of_villages"] = Sx.n_valid_zero_irr_posarea / Sx.n_valid_pos_area.replace(0, np.nan)
Sx["dropped_area_share"] = Sx.area_implausible_ha / Sx.area_nonmissing_ha.add(Sx.area_implausible_ha, fill_value=0).replace(0, np.nan)
Sx["src_gt_tot_share"] = Sx.n_src_gt_tot / Sx.n_valid.replace(0, np.nan)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
log(Sx[["state_name", "round", "n_univ_shrids", "cov_villages", "cov_area", "irr_share",
        "zero_irr_share_of_villages", "dropped_area_share"]].round(3).to_string())

# ------------------------------------------------------------------ DES + Antyodaya
des = pd.read_csv(REPO + "outputs/revision_nov8/irrigation_panel_district_year.csv")
fp = pd.read_csv(REPO + "outputs/shrug_covariates/final_panel.csv")
shk = names.set_index("shrug_key")[["st", "dt"]]


def des_window(y0, y1, uniq_only):
    d = des[des.year_start.between(y0, y1)].dropna(subset=["shrug_key", "irr_nsa"]).copy()
    if uniq_only:
        d = d[~d.flag_shared_parent.astype(bool)]
    m = d.groupby(["shrug_key", "district_key"]).agg(irr_nsa=("irr_nsa", "mean"), n=("irr_nsa", "size"),
                                                      nia=("nia", "sum"), nsa=("nsa", "sum")).reset_index()
    m = m[m.n >= 2]                                       # at least two of the three years
    m = m.groupby("shrug_key").agg(des_irr_nsa=("irr_nsa", "mean"), des_n_div=("district_key", "size")).reset_index()
    return m.merge(names, on="shrug_key", how="left")


ay = cov.copy()
ay["ay_share"] = ay.ay_area_irrigated_in_hac / (ay.ay_area_irrigated_in_hac + ay.ay_total_unirrigated_land_area_in_h)
ay = ay.replace([np.inf, -np.inf], np.nan)
site_keys = set(fp.loc[fp.in_final.astype(bool), "shrug_key"].dropna())
log("site sample: {} districts in final_panel with in_final, {} unique SHRUG parents".format(
    int(fp.in_final.astype(bool).sum()), len(site_keys)))

# ------------------------------------------------------------------ state-round flags
# Step 1: coverage / zeros / dropped-area rules
Sx["flag_coverage"] = Sx.cov_area < MIN_AREA_COV
Sx["flag_zeros"] = Sx.zero_irr_share_of_villages > MAX_ZERO_IRR
Sx["flag_dropped"] = Sx.dropped_area_share > MAX_DROPPED
Sx["flag_missing_round"] = Sx.n_has_land == 0
# Step 2: external check against DES, on the SAME districts: mean census district share vs mean DES share
win = {1991: None, 2001: (2000, 2002), 2011: (2010, 2012)}
Sx["des_state_share"] = np.nan
Sx["census_share_on_des_districts"] = np.nan
Sx["n_des_districts"] = 0
for yr, w_ in win.items():
    if w_ is None:
        continue
    d = des_window(w_[0], w_[1], uniq_only=False)
    c = D[D["round"] == yr][["st", "dt", "irr_share", "cov_area"]]
    m = d.merge(c, on=["st", "dt"]).dropna(subset=["irr_share", "des_irr_nsa"])
    m = m[m.cov_area >= MIN_DIST_COV]
    g = m.groupby("st").agg(des=("des_irr_nsa", "mean"), cen=("irr_share", "mean"), n=("st", "size"))
    for st_, r in g.iterrows():
        if r.n >= MIN_DES_DISTRICTS:
            ix = (Sx["round"] == yr) & (Sx.st == st_)
            Sx.loc[ix, ["des_state_share", "census_share_on_des_districts", "n_des_districts"]] = [r.des, r.cen, r.n]
Sx["gap_vs_des"] = Sx.census_share_on_des_districts - Sx.des_state_share
Sx["flag_des_gap"] = Sx.gap_vs_des.abs() > MAX_GAP_DES

# Fallback where DES cannot be matched (fewer than MIN_DES_DISTRICTS districts): 2011 against Antyodaya 2019-20 on the
# same districts. Antyodaya is eight years later and noisy (r about 0.8 with the census), so only large gaps count.
Sx["ay_state_share"] = np.nan
c11 = D[(D["round"] == 2011) & (D.cov_area >= MIN_DIST_COV)][["st", "dt", "irr_share"]].merge(
    ay[["st", "dt", "ay_share"]], on=["st", "dt"]).dropna()
g11 = c11.groupby("st").agg(ay=("ay_share", "mean"), cen=("irr_share", "mean"), n=("st", "size"))
g11 = g11[g11.n >= MIN_DES_DISTRICTS]
no_des = (Sx.n_des_districts < MIN_DES_DISTRICTS)
m11 = (Sx["round"] == 2011) & Sx.st.isin(g11.index) & no_des
Sx.loc[m11, "ay_state_share"] = Sx.loc[m11, "st"].map(g11.ay)
Sx["census_share_on_ay_districts"] = np.where(m11, Sx.st.map(g11.cen), np.nan)
Sx["gap_vs_ay"] = np.where(m11, Sx.st.map(g11.cen) - Sx.ay_state_share, np.nan)
Sx["flag_ay_gap"] = Sx.gap_vs_ay.abs() > MAX_GAP_AY

# no cultivated area recorded at all (every land cell zero): Meghalaya, Mizoram and parts of the NE
Sx["flag_no_land"] = Sx.irr_share.isna() & (Sx.n_has_land > 0)
Sx["flag_zeros"] = Sx.flag_zeros | ((Sx.zero_irr_share_of_villages > 0.95) & (Sx.irr_share < 0.01))

# 1991 has no DES panel: compare with 2001 when the 2001 state-round is itself unaffected by the DES/coverage checks
piv = Sx.pivot(index="st", columns="round", values="irr_share")
Sx = Sx.merge(piv.rename(columns=lambda c: "irr_share_{}".format(c)).reset_index(), on="st", how="left")
# 2001 vs 2011 jump where one of the two has no DES check but the other passes one
chk_ok = (Sx.n_des_districts >= MIN_DES_DISTRICTS) & ~Sx.flag_des_gap & ~Sx.flag_coverage
Sx["des_clean"] = chk_ok
pv = Sx.pivot(index="st", columns="round", values="des_clean")
Sx["flag_jump_0111"] = False
for yr, other in ((2001, 2011), (2011, 2001)):
    ix = (Sx["round"] == yr) & no_des
    ref = Sx.st.map(piv[other]); refclean = Sx.st.map(pv[other]).fillna(False).astype(bool)
    Sx.loc[ix, "flag_jump_0111"] = ((Sx.irr_share - ref).abs() > MAX_JUMP_0111)[ix] & refclean[ix]
ok01 = Sx[Sx["round"] == 2001].set_index("st").eval("~(flag_coverage | flag_des_gap | flag_no_land)")
Sx["flag_jump_1991"] = ((Sx["round"] == 1991) & Sx.st.map(ok01).fillna(False)
                       & (((Sx.irr_share_1991 - Sx.irr_share_2001).abs() > MAX_JUMP_9101)
                          | (((Sx.irr_share_1991 / Sx.irr_share_2001) < MIN_RATIO_9101)
                             & ((Sx.irr_share_1991 - Sx.irr_share_2001).abs() > 0.05))
                          | (((Sx.irr_share_1991 / Sx.irr_share_2001) > 1 / MIN_RATIO_9101)
                             & ((Sx.irr_share_1991 - Sx.irr_share_2001).abs() > 0.05))))


def classify(r):
    if r.flag_missing_round:
        return "round absent for this state"
    reasons = []
    if r.flag_no_land:
        reasons.append("no cultivated land recorded (all land cells zero)")
    if r.flag_coverage:
        reasons.append("area coverage {:.0%} < {:.0%}".format(r.cov_area, MIN_AREA_COV))
    if r.flag_zeros:
        reasons.append("{:.0%} of cultivated villages report zero irrigation".format(r.zero_irr_share_of_villages))
    if r.flag_dropped:
        reasons.append("{:.0%} of reported area in implausible villages".format(r.dropped_area_share))
    if r.flag_des_gap:
        reasons.append("census share {:.2f} vs DES {:.2f} on the same districts".format(
            r.census_share_on_des_districts, r.des_state_share))
    if r.flag_ay_gap:
        reasons.append("census share {:.2f} vs Antyodaya 2019-20 {:.2f} on the same districts (no DES match)".format(
            r.census_share_on_ay_districts, r.ay_state_share))
    if r.flag_jump_0111:
        reasons.append("share {:.2f} vs {:.2f} in the other census round that passes the DES check (no DES match here)".format(
            r.irr_share, r.irr_share_2001 if r["round"] == 2011 else r.irr_share_2011))
    if r.flag_jump_1991:
        reasons.append("1991 share {:.2f} vs 2001 share {:.2f}".format(r.irr_share_1991, r.irr_share_2001))
    return "; ".join(reasons)


Sx["flag_reason"] = Sx.apply(classify, axis=1)
Sx["state_round_usable"] = Sx.flag_reason == ""
Sx["state_round_cov_only"] = (Sx.flag_coverage & ~Sx.flag_no_land & ~Sx.flag_zeros & ~Sx.flag_dropped
                              & ~Sx.flag_des_gap & ~Sx.flag_ay_gap & ~Sx.flag_jump_0111 & ~Sx.flag_jump_1991 & ~Sx.flag_missing_round)

# ------------------------------------------------------------------ attach terrain
log("")
log("=" * 78)
log("TERRAIN")
el = pd.read_stata(R + "shrug-elevation-dta/elevation_shrid.dta")
tr = pd.read_stata(R + "shrug-rugged-dta/tri_shrid.dta")
t = key[["shrid2", "st", "dt"]].merge(el[["shrid2", "elevation_mean", "elevation_num_cells", "elevation_std"]],
                                      on="shrid2", how="left").merge(
    tr[["shrid2", "tri_mean", "tri_num_cells"]], on="shrid2", how="left")
t["w_e"] = t.elevation_num_cells.fillna(0)
t["w_t"] = t.tri_num_cells.fillna(0)


def wmean(g, c, w):
    ww = g[w]
    ok = g[c].notna() & (ww > 0)
    return np.average(g.loc[ok, c], weights=ww[ok]) if ok.any() else np.nan


terr = t.groupby(["st", "dt"]).apply(lambda g: pd.Series({
    "elevation_mean_m": wmean(g, "elevation_mean", "w_e"),
    "tri_mean": wmean(g, "tri_mean", "w_t"),
    "terrain_cells": g.w_e.sum(),
})).reset_index()
ed = pd.read_stata(R + "shrug-elevation-dta/elevation_pc11dist.dta")
td = pd.read_stata(R + "shrug-rugged-dta/tri_pc11dist.dta")
ed["st"], ed["dt"] = ed.pc11_state_id.astype(int), ed.pc11_district_id.astype(int)
td["st"], td["dt"] = td.pc11_state_id.astype(int), td.pc11_district_id.astype(int)
chk = terr.merge(ed[["st", "dt", "elevation_mean"]], on=["st", "dt"]).merge(td[["st", "dt", "tri_mean"]],
                                                                          on=["st", "dt"], suffixes=("", "_official"))
log("area-weighted shrid elevation vs SHRUG district file: corr {:.4f}, median abs diff {:.2f} m (n={})".format(
    chk.elevation_mean_m.corr(chk.elevation_mean), (chk.elevation_mean_m - chk.elevation_mean).abs().median(), len(chk)))
log("area-weighted shrid TRI vs SHRUG district file: corr {:.4f}, median abs diff {:.3f} (n={})".format(
    chk.tri_mean.corr(chk.tri_mean_official), (chk.tri_mean - chk.tri_mean_official).abs().median(), len(chk)))

# ------------------------------------------------------------------ district flags
D = D.merge(Sx[["st", "round", "state_round_usable", "state_round_cov_only", "flag_reason"]], on=["st", "round"], how="left")
# a state flagged ONLY for low coverage still has usable districts where the district itself is covered
D["district_usable"] = ((D.state_round_usable.fillna(False) | D.state_round_cov_only.fillna(False))
                        & (D.cov_area >= MIN_DIST_COV) & D.irr_share.notna())
D = D.merge(terr, on=["st", "dt"], how="left")
D = D.rename(columns={"st": "pc11_state_id", "dt": "pc11_district_id"})

# ------------------------------------------------------------------ national shares
log("")
log("=" * 78)
log("NATIONAL SHARES (all valid villages / usable state-rounds only)")
nat_rows = []
for yr in (1991, 2001, 2011):
    for lab, mask_states in (("all valid villages", None), ("usable state-rounds only", "usable")):
        v = vil[yr]
        vv = v[v.valid]
        if mask_states == "usable":
            ok = Sx[(Sx["round"] == yr) & Sx.state_round_usable].st
            vv = vv[vv.st.isin(ok)]
        den = vv.tot_irr.sum() + vv.un_irr.sum()
        row = {"round": yr, "basis": lab, "n_states": vv.st.nunique(), "irr_unirr_Mha": den / 1e6,
               "irr_share": vv.tot_irr.sum() / den}
        for c in ["canal", "groundwater", "gw_elec", "tank_lake", "river", "other_misc", "river_other", "unattributed"]:
            row["sh_" + c] = vv[c].sum() / den if vv[c].notna().any() else np.nan
        for c in ["canal", "groundwater", "tank_lake", "river_other", "unattributed"]:
            row["comp_" + c] = vv[c].sum() / vv.tot_irr.sum()
        nat_rows.append(row)
NAT = pd.DataFrame(nat_rows)
log(NAT.round(3).to_string())

# ------------------------------------------------------------------ cross-check vs district files
log("")
log("=" * 78)
log("CROSS-CHECK AGAINST SHRUG DISTRICT FILES")
xc = []
for yr, f, p in [(1991, "shrug-vd91-dta/pc91_vd_clean_pc91dist.dta", "pc91_vd_"),
                 (2001, "shrug-vd01-dta/pc01_vd_clean_pc01dist.dta", "pc01_vd_"),
                 (2011, "shrug-vd11-dta/pc11_vd_clean_pc11dist.dta", "pc11_vd_land_")]:
    dd = pd.read_stata(R + f)
    tcol, ucol = (p + "tot_irr", p + "un_irr") if yr != 2011 else (p + "src_irr", p + "un_irr")
    v = vil[yr]
    raw_t = v.loc[v.has_land, "tot_irr"].sum()
    raw_u = v.loc[v.has_land, "un_irr"].sum()
    fl_t, fl_u = dd[tcol].sum(), dd[ucol].sum()
    r_ = {"round": yr, "dist_file_irr_Mha": fl_t / 1e6, "village_irr_all_Mha": raw_t / 1e6,
          "dist_file_unirr_Mha": fl_u / 1e6, "village_unirr_all_Mha": raw_u / 1e6,
          "dist_file_share": fl_t / (fl_t + fl_u), "village_all_share": raw_t / (raw_t + raw_u),
          "village_valid_share": v[v.valid].tot_irr.sum() / (v[v.valid].tot_irr.sum() + v[v.valid].un_irr.sum())}
    if yr == 2011:
        dd["st"], dd["dt"] = dd.pc11_state_id.astype(int), dd.pc11_district_id.astype(int)
        m = dd[["st", "dt", tcol, ucol]].assign(fs=lambda x: x[tcol] / (x[tcol] + x[ucol])).merge(
            D[D["round"] == 2011][["pc11_state_id", "pc11_district_id", "irr_share"]].rename(
                columns={"pc11_state_id": "st", "pc11_district_id": "dt"}), on=["st", "dt"])
        r_["district_corr_vs_own_file_2011"] = m.fs.corr(m.irr_share)
        r_["district_median_absdiff_2011"] = (m.fs - m.irr_share).abs().median()
    xc.append(r_)
XC = pd.DataFrame(xc)
log(XC.round(4).T.to_string())

# ------------------------------------------------------------------ validation
log("")
log("=" * 78)
log("VALIDATION")
VAL = []


def corr_row(label, x, y, extra=None):
    m = pd.DataFrame({"x": x, "y": y}).dropna()
    r = {"comparison": label, "n": len(m)}
    if len(m) > 3:
        r.update(pearson=m.x.corr(m.y), spearman=m.x.corr(m.y, method="spearman"),
                 mean_x=m.x.mean(), mean_y=m.y.mean(), mean_diff_x_minus_y=(m.x - m.y).mean(),
                 mean_abs_diff=(m.x - m.y).abs().mean(),
                 slope=np.polyfit(m.y, m.x, 1)[0])
    if extra:
        r.update(extra)
    VAL.append(r)


c11 = D[D["round"] == 2011].set_index(["pc11_state_id", "pc11_district_id"])
c11u = c11[c11.district_usable]
c01 = D[D["round"] == 2001].set_index(["pc11_state_id", "pc11_district_id"])
c01u = c01[c01.district_usable]
c91 = D[D["round"] == 1991].set_index(["pc11_state_id", "pc11_district_id"])

ay_i = ay.set_index(["st", "dt"])["ay_share"]
ay_i.index.names = ["pc11_state_id", "pc11_district_id"]
site_idx = ay.loc[ay.district_key.isin(site_keys), ["st", "dt"]].apply(tuple, axis=1)
site_mi = pd.MultiIndex.from_tuples(list(site_idx), names=["pc11_state_id", "pc11_district_id"])

for lab, cdf in (("all districts", c11), ("usable districts", c11u)):
    corr_row("census 2011 vs Antyodaya 2019-20, " + lab, cdf.irr_share, ay_i.reindex(cdf.index))
    ci = cdf.index.intersection(site_mi)
    corr_row("census 2011 vs Antyodaya 2019-20, site sample (" + lab + ")", cdf.loc[ci].irr_share, ay_i.reindex(ci))

for uniq, tag in ((True, "unshared parents only"), (False, "all parents, mean of child districts")):
    d11 = des_window(2010, 2012, uniq).set_index(["st", "dt"]).des_irr_nsa
    d11.index.names = ["pc11_state_id", "pc11_district_id"]
    d01 = des_window(2000, 2002, uniq).set_index(["st", "dt"]).des_irr_nsa
    d01.index.names = ["pc11_state_id", "pc11_district_id"]
    corr_row("census 2011 vs DES 2010-12, usable districts, " + tag, c11u.irr_share, d11.reindex(c11u.index))
    corr_row("census 2011 vs DES 2010-12, all districts, " + tag, c11.irr_share, d11.reindex(c11.index))
    corr_row("census 2001 vs DES 2000-02, usable districts, " + tag, c01u.irr_share, d01.reindex(c01u.index))
    corr_row("census 2001 vs DES 2000-02, all districts, " + tag, c01.irr_share, d01.reindex(c01.index))
corr_row("census 2001 vs census 2011, usable in both", c01u.irr_share, c11u.irr_share.reindex(c01u.index))
corr_row("census 1991 vs census 2001, all", c91.irr_share, c01.irr_share)
corr_row("census 1991 vs census 2011, all", c91.irr_share, c11.irr_share)
VALDF = pd.DataFrame(VAL)
log(VALDF.round(3).to_string())

# state-level validation of 2001 against DES (where 2001 is patchy)
d01s = des_window(2000, 2002, False)
d11s = des_window(2010, 2012, False)
st_val = []
for yr, dd in ((2001, d01s), (2011, d11s)):
    cdf = D[(D["round"] == yr)].merge(dd[["st", "dt", "des_irr_nsa"]].rename(
        columns={"st": "pc11_state_id", "dt": "pc11_district_id"}), on=["pc11_state_id", "pc11_district_id"])
    for s_, g in cdf.groupby("pc11_state_id"):
        gg = g.dropna(subset=["irr_share", "des_irr_nsa"])
        st_val.append({"round": yr, "pc11_state_id": s_, "n": len(gg),
                       "corr": gg.irr_share.corr(gg.des_irr_nsa) if len(gg) >= 5 else np.nan,
                       "census_mean": gg.irr_share.mean(), "des_mean": gg.des_irr_nsa.mean()})
STV = pd.DataFrame(st_val).merge(st_names.rename(columns={"st": "pc11_state_id"}), on="pc11_state_id")

# ------------------------------------------------------------------ write panel
keep_long = ["pc11_state_id", "pc11_district_id", "state_name", "district_name", "shrug_key", "round",
             "n_univ_shrids", "area_univ_ha", "n_villages_file", "n_has_land", "n_implausible", "n_valid",
             "cov_villages", "cov_area", "irr_ha", "unirr_ha", "irr_unirr_ha", "irr_share"]
keep_long += ["sh_" + c for c in SRC] + ["comp_" + c for c in SRC] + [c + "_ha" for c in SRC]
keep_long += ["elevation_mean_m", "tri_mean", "state_round_usable", "flag_reason", "district_usable"]
P = D[keep_long].sort_values(["pc11_state_id", "pc11_district_id", "round"])
P.to_csv(OUT + "census_irrigation_panel.csv", index=False)

wide_cols = ["n_valid", "cov_villages", "cov_area", "irr_unirr_ha", "irr_share"] + \
            ["sh_" + c for c in SRC] + ["district_usable"]
W = P.pivot(index=["pc11_state_id", "pc11_district_id", "state_name", "district_name", "shrug_key",
                   "elevation_mean_m", "tri_mean"], columns="round", values=wide_cols)
W.columns = ["{}_{}".format(a, b) for a, b in W.columns]
W = W.reset_index()
W.to_csv(OUT + "census_irrigation_panel_wide.csv", index=False)
Sx_out = Sx.rename(columns={"st": "pc11_state_id"})
Sx_out.to_csv(OUT + "census_irrigation_state_round.csv", index=False)
NAT.to_csv(OUT + "census_irrigation_national.csv", index=False)
VALDF.to_csv(OUT + "census_irrigation_validation.csv", index=False)
log("wrote panel {} wide {}".format(P.shape, W.shape))

# ------------------------------------------------------------------ audit markdown
def md_table(df, fmt=None):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            x = r[c]
            if isinstance(x, (float, np.floating)):
                cells.append("" if np.isnan(x) else ("{:.3f}".format(x) if abs(x) < 100 else "{:,.0f}".format(x)))
            else:
                cells.append(str(x))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def pct(x):
    return "" if pd.isna(x) else "{:.0%}".format(x)


cov_tab = Sx.assign(cov_area=Sx.cov_area.clip(upper=1)).pivot(index="state_name", columns="round", values="cov_area").map(pct)
cov_tab.columns = ["area cov {}".format(c) for c in cov_tab.columns]
vil_tab = Sx.assign(cov_villages=Sx.cov_villages.clip(upper=1)).pivot(index="state_name", columns="round", values="cov_villages").map(pct)
vil_tab.columns = ["village cov {}".format(c) for c in vil_tab.columns]
sh_tab = Sx.pivot(index="state_name", columns="round", values="irr_share").map(lambda x: "" if pd.isna(x) else "{:.2f}".format(x))
sh_tab.columns = ["share {}".format(c) for c in sh_tab.columns]
st_tab = pd.concat([cov_tab, vil_tab, sh_tab], axis=1).reset_index()
fitflag = Sx.pivot(index="state_name", columns="round", values="state_round_usable").map(lambda x: "ok" if x is True or x == True else "FLAG")
fitflag.columns = ["fit {}".format(c) for c in fitflag.columns]
st_tab = st_tab.merge(fitflag.reset_index(), on="state_name")
st_tab = st_tab[["state_name", "area cov 1991", "area cov 2001", "area cov 2011",
                 "village cov 1991", "village cov 2001", "village cov 2011",
                 "share 1991", "share 2001", "share 2011", "fit 1991", "fit 2001", "fit 2011"]]

nat_cov = []
for yr in (1991, 2001, 2011):
    v = vil[yr]
    nat_cov.append({"round": yr, "rows in file": len(v), "non-missing land (share of PC11 rural shrids)":
                    v.has_land.sum() / len(univ), "valid area / PC11 rural area": v[v.valid].area_u.sum() / univ.area_u.sum(),
                    "implausible villages": int(v.implausible.sum())})
nat_cov = pd.DataFrame(nat_cov)
nat_cov["round"] = nat_cov["round"].astype(str)

flagged = Sx[~Sx.state_round_usable][["state_name", "round", "cov_area", "irr_share", "des_state_share", "flag_reason"]]
flagged = flagged.assign(cov_area=flagged.cov_area.clip(upper=1))
flagged = flagged.sort_values(["round", "state_name"])
fit = {}
for yr in (1991, 2001, 2011):
    ok = sorted(Sx[(Sx["round"] == yr) & Sx.state_round_usable].state_name)
    bad = sorted(Sx[(Sx["round"] == yr) & ~Sx.state_round_usable].state_name)
    fit[yr] = (ok, bad)

mdl = []
w = mdl.append
w("# Census irrigation panel by source: audit")
w("")
w("Script 121_census_irrigation_panel.py. Source: SHRUG 2.1 village directories (PC91, PC01, PC11), "
  "aggregated to constant 2011 districts through shrid_pc11dist_key. Areas are hectares in all three rounds.")
w("")
w("## 1. What the panel holds")
w("")
w("- `census_irrigation_panel.csv`: {} district-round rows ({} districts x 3 rounds), long.".format(len(P), P.groupby(["pc11_state_id", "pc11_district_id"]).ngroups))
w("- `census_irrigation_panel_wide.csv`: one row per district, round suffixes _1991, _2001, _2011.")
w("- `census_irrigation_state_round.csv`: the state x round coverage table behind section 3.")
w("- `census_irrigation_national.csv`, `census_irrigation_validation.csv`: tables reproduced below.")
w("- Irrigated share = irrigated / (irrigated + unirrigated), net irrigated over net area sown. "
  "`sh_*` columns are source area over that same denominator; `comp_*` are source area over total irrigated area.")
w("- `district_usable` is True when the district's state-round passes the checks in section 3 and the district has at "
  "least {:.0%} of its PC11 rural area in valid villages.".format(MIN_DIST_COV))
w("- Terrain: `elevation_mean_m` and `tri_mean` are area-weighted (by raster cell count) over the district's shrids; they "
  "match the SHRUG district files (correlation {:.4f} for elevation, {:.4f} for TRI).".format(
      chk.elevation_mean_m.corr(chk.elevation_mean), chk.tri_mean.corr(chk.tri_mean_official)))
w("")
w("## 2. Source harmonisation (every mapping)")
w("")
w(md_table(pd.DataFrame(MAPPING, columns=["category", "1991 fields", "2001 fields", "2011 fields"])))
w("")
w("Rules applied at village level:")
w("")
w("1. Total irrigated = the larger of the village's total field and the sum of its source fields (blank fields ignored). The 1991 total is blank or zero in many villages whose source fields are filled; sources then exceed the total, and the repair adds the difference to irrigated area. The repair is quantified per round in the log, and the 2001 and 2011 totals are almost never below their sources. A village "
  "needs both a total irrigated and an unirrigated value to count as having land data.")
w("2. Missing source fields inside a village that has a total are set to zero (a blank source is a zero in these directories).")
w("3. A village is dropped as implausible when irrigated + unirrigated exceeds {}x its reference area "
  "(the larger of the round's own village area and the PC11 shrid area) and is above 5 ha. "
  "These are mostly merged or split villages and keying errors (the worst 2001 row reports 2.4 million ha of unirrigated land).".format(PLAUS_MULT))
w("4. Shares come from summed hectares, never from averaging village or district ratios.")
w("5. Comparability limits that no mapping can repair: 2011 has no river category (river irrigation sits inside other), "
  "2011 lumps wells with tube wells, 2001 has no electrified open wells, and electrification is not recorded in 2011. "
  "The comparable three-round groups are canal, groundwater, tank and lake, and river_other.")
w("")
w("## 3. Coverage audit")
w("")
w("National coverage:")
w("")
w(md_table(nat_cov))
w("")
w("Share of PC11 rural villages (shrids) with non-missing land data, and share of PC11 rural area in valid villages, by state and round. "
  "`fit` is the verdict from the state-round rules below.")
w("")
w(md_table(st_tab))
w("")
w("State-round rules (a state-round is flagged if any applies):")
w("")
w("- the round has no land rows for the state;")
w("- every land cell in the state is zero, so no cultivated area is recorded;")
w("- valid villages cover under {:.0%} of the state's PC11 rural area (a coverage-only flag still leaves districts "
  "usable where the district itself has at least {:.0%} coverage);".format(MIN_AREA_COV, MIN_DIST_COV))
w("- zeros everywhere: over {:.0%} of cultivated valid villages report zero irrigation, or over 95% do and the "
  "state share is under 1%;".format(MAX_ZERO_IRR))
w("- more than {:.0%} of the state's reported area sits in implausible villages;".format(MAX_DROPPED))
w("- the mean census district share differs from the DES mean on the same districts (2000-02 for the 2001 round, "
  "2010-12 for 2011, at least {} matched districts) by more than {:.2f};".format(MIN_DES_DISTRICTS, MAX_GAP_DES))
w("- where DES cannot be matched in 2001 or 2011: 2011 differs from Antyodaya 2019-20 on the same districts by more than {:.2f}, "
  "or the share differs by more than {:.2f} from the other of 2001/2011 when that one passes its DES check;".format(MAX_GAP_AY, MAX_JUMP_0111))
w("- for 1991, where DES cannot help (its panel starts in 1997): the 1991 state share differs from the 2001 share by more "
  "than {:.2f}, or is under {:.1f}x (or over {:.2f}x) the 2001 share with a gap above 0.05, when the 2001 state-round passes its own "
  "checks. Net irrigated area rose by roughly a sixth between the two censuses nationally, so a 1991 state share far below its "
  "2001 share is under-recording.".format(MAX_JUMP_9101, MIN_RATIO_9101, 1 / MIN_RATIO_9101))
w("")
w("Flagged state-rounds:")
w("")
w(md_table(flagged.assign(cov_area=flagged.cov_area.map(pct), round=flagged["round"].astype(str))) if len(flagged) else "None.")
w("")
w("Coverage of a share above 100% means the round lists more village rows than the PC11 rural shrid universe "
  "(villages that split or merge between rounds); those shares are capped at 100% in the table above.")
w("")
w("## 4. National shares by round and source")
w("")
w("Share of (irrigated + unirrigated) area. `comp_*` columns are shares of total irrigated area.")
w("")
w(md_table(NAT.assign(**{"round": NAT["round"].astype(str)})))
w("")
w("Cross-check against the SHRUG district files (sums of district totals vs village-level sums, hectares in millions):")
w("")
w(md_table(XC.assign(**{"round": XC["round"].astype(str)}).drop(columns=[c for c in XC.columns if c.startswith("district_")])))
w("")
w("For 2011 the village aggregation reproduces the SHRUG district file at district level: correlation {:.4f}, median absolute "
  "difference {:.4f} in share. The 1991 district file totals are lower than the village sums because it omits the irrigated "
  "total for villages where only the source fields are filled (about 13% of 1991 villages), which the village build recovers.".format(
      XC.loc[XC["round"] == 2011, "district_corr_vs_own_file_2011"].iloc[0],
      XC.loc[XC["round"] == 2011, "district_median_absdiff_2011"].iloc[0]))
w("")
w("## 5. Validation")
w("")
w(md_table(VALDF))
w("")
w("Reading the table: DES and the census agree closely (r about 0.94) in both overlap rounds, which validates the "
  "village build for 2001 and 2011. Antyodaya 2019-20 agrees less (r about 0.82; 0.87 on usable districts) and runs about 9 points "
  "higher, as expected from a later year and a self-reported village survey. Census 1991 correlates only about 0.5 with 2001 "
  "because the 1991 directory under-records irrigation in many states.")
w("")
w("State-level agreement with DES in the two overlapping rounds (district-level correlation within state, n at least 5):")
w("")
w(md_table(STV.sort_values(["round", "corr"])[["round", "state_name", "n", "corr", "census_mean", "des_mean"]].assign(
    round=lambda x: x["round"].astype(str))))
w("")
w("## 6. Fit for analysis")
w("")
for yr in (1991, 2001, 2011):
    ok, bad = fit[yr]
    w("**{}**: {} states fit. Not fit: {}.".format(yr, len(ok), ", ".join(bad) if bad else "none"))
    w("")
    w("Fit: " + ", ".join(ok))
    w("")
open(OUT + "census_irrigation_audit.md", "w", encoding="utf-8").write("\n".join(mdl))
open(OUT + "census_irrigation_log.txt", "w", encoding="utf-8").write("\n".join(LOG))
log("DONE")

# ------------------------------------------------------------------ site-sample coverage (appended to audit)
site_rows = []
sk = names[names.shrug_key.isin(site_keys)][["st", "dt"]]
for yr in (1991, 2001, 2011):
    d_ = D[(D["round"] == yr)].merge(sk.rename(columns={"st": "pc11_state_id", "dt": "pc11_district_id"}),
                                     on=["pc11_state_id", "pc11_district_id"])
    site_rows.append({"round": str(yr), "site SHRUG parents": len(d_), "with a census share": int(d_.irr_share.notna().sum()),
                      "usable": int(d_.district_usable.sum())})
SITE = pd.DataFrame(site_rows)
with open(OUT + "census_irrigation_audit.md", "a", encoding="utf-8") as fh:
    fh.write("\n## 7. Coverage of the site's regression sample\n\n")
    fh.write("Unique SHRUG parent districts behind the site's 606 districts, and how many carry a usable census share.\n\n")
    fh.write(md_table(SITE) + "\n")
log(SITE.to_string())
