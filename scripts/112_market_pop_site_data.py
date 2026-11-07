"""
112_market_pop_site_data.py

Puts the markets page on population-weighted facility measures.

Each facility is the share of a district's rural population (Census 2011 village
population) living in a village that reports it, from 104_market_pop_weighted.py
(version "W": every facility measure population-weighted, including connectivity and
the group indices). Village shares stay available for comparison.

Steps
  1. Write outputs/revision_nov8/market_covariates_pop.csv: market_covariates.csv with
     the ten m_*_vshare columns replaced by the population shares (the idx_* columns are
     only used as a sample filter downstream and are left as they are).
  2. Re-run 89_market_typology.py and 90_market_map_layers.py on that file
     (CROPDIV_MARKET_COV), which rewrites docs/data/site_typology.json and
     docs/data/site_market_map.json on population shares.
  3. Write docs/data/site_markets_pop.json: descriptives, correlations, standard
     deviations, one-facility-at-a-time coefficients (population and village), block
     coefficients with BH q, irrigation interactions, the weekly haat robustness ladder,
     within-state band tests, and a scatter, all on population shares.
  4. Print the state and band numbers quoted in the page prose.

site_markets.json (village shares, written by 81) is left untouched; the measurement
page still reads it.

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/112_market_pop_site_data.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = r"D:/crop-diversity"
COV = REPO + "/outputs/shrug_covariates"
REV = REPO + "/outputs/revision_nov8"
DOCS = REPO + "/docs/data"
PY = sys.executable
KEY = ["pc11_state_id", "pc11_district_id"]

FAC = [("mandi", "mandi"), ("regular market", "regular_market"), ("weekly haat", "weekly_haat"),
       ("fertiliser shop", "fert_shop"), ("seed centre", "seed_centre"), ("soil testing", "soil_test"),
       ("custom hiring", "custom_hire"), ("cold storage", "storage"),
       ("farm-gate processing", "farmgate_proc"), ("FPO", "fpo")]
LAB = {s: l for l, s in FAC}


def r6(v):
    if v is None:
        return None
    v = float(v)
    return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)


# ------------------------------------------------------------ 1. covariates on population shares
mk = pd.read_csv(COV + "/market_covariates.csv")
ws = pd.read_csv(REV + "/market_pw_weighted_district_shares.csv")
for t in (mk, ws):
    for c in KEY:
        t[c] = pd.to_numeric(t[c], errors="coerce")
n0 = len(mk)
# Sultanpur and Amethi share one 2011 district and carry identical shares
m = mk.merge(ws[KEY + ["m_%s_w" % s for _, s in FAC]].drop_duplicates(KEY), on=KEY, how="left")
assert len(m) == n0
for _, s in FAC:
    m["m_%s_vshare_village" % s] = m["m_%s_vshare" % s]
    m["m_%s_vshare" % s] = m["m_%s_w" % s].where(m["m_%s_w" % s].notna(), np.nan)
m = m.drop(columns=["m_%s_w" % s for _, s in FAC])
POP_CSV = REV + "/market_covariates_pop.csv"
m.to_csv(POP_CSV, index=False)
print("wrote", POP_CSV)

# ------------------------------------------------------------ 2. typology and map layers
env = dict(os.environ, CROPDIV_MARKET_COV=POP_CSV,
           CROPDIV_TYPOLOGY_LOG=REV + "/market_typology_pop_log.txt")
for sc in ["89_market_typology.py", "90_market_map_layers.py"]:
    r = subprocess.run([PY, REPO + "/scripts/" + sc], env=env, capture_output=True, text=True)
    print("ran", sc, "exit", r.returncode)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(1)

for fn, note in [("site_typology.json",
                  "Market typology on the 606-district analysis set, on POPULATION shares: each "
                  "facility is the share of the district's rural population living in a village "
                  "that reports it. 'density' is the mean percentile rank of the nine facilities "
                  "other than the weekly haat; types cross density terciles with weekly haat share "
                  "above or below the median. Keys match districts.geojson 'k'."),
                 ("site_market_map.json",
                  "Ten facility POPULATION shares (share of rural population living in a village "
                  "with the facility; order as in fac), D0 count of crops and D1 effective number "
                  "of crops for the 606 market-analysis districts, keyed as districts.geojson k; "
                  "m = has a polygon. state_rings: simplified state outlines.")]:
    p = DOCS + "/" + fn
    j = json.load(open(p, encoding="utf-8"))
    j["note"] = note
    j["measure"] = "population share"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(j, f, separators=(",", ":"))

# check the typology against 104's version W assignment
ty = json.load(open(DOCS + "/site_typology.json", encoding="utf-8"))
pw = pd.read_csv(REV + "/market_pw_typology.csv")
print("typology cuts (population):", ty["cuts"])
print("type sizes:", {t["code"]: t["n"] for t in ty["types"]})
sizes104 = pw["type_W"].value_counts().sort_index().tolist()
print("104 type_W sizes (by code 0..5):", sizes104)
assert sorted(sizes104) == sorted(t["n"] for t in ty["types"]), "typology does not match 104 version W"

# ------------------------------------------------------------ 3. site_markets_pop.json
fp = pd.read_csv(COV + "/final_panel.csv")
for c in KEY:
    fp[c] = pd.to_numeric(fp[c], errors="coerce")
dm = fp.merge(m, on=KEY, how="left")
dm = dm[dm["in_final"] & dm["idx_output_market"].notna()].copy()
assert len(dm) == 606
corr_div = pd.read_csv(REPO + "/outputs/crop_diversity_analysis/district_diversity_indices_corrected.csv")
DV = ["D0_richness", "D1_exp_shannon"]
dm = dm.drop(columns=[c for c in DV if c in dm]).merge(corr_div[["district_key"] + DV], on="district_key", how="left")

C = ["m_%s_vshare" % s for _, s in FAC]
CV = ["m_%s_vshare_village" % s for _, s in FAC]
desc = []
for (lab, s), c, cv in zip(FAC, C, CV):
    desc.append({"name": lab, "mean": r6(dm[c].mean()), "p10": r6(dm[c].quantile(.1)),
                 "p90": r6(dm[c].quantile(.9)), "zero": int((dm[c] == 0).sum()),
                 "sd": r6(dm[c].std()),
                 "viirs": r6(dm[[c, "log_viirs"]].corr().iloc[0, 1]),
                 "estab": r6(dm[[c, "estab_per_1000pop"]].corr().iloc[0, 1]),
                 "mean_village": r6(dm[cv].mean()), "sd_village": r6(dm[cv].std())})
cm = dm[C].corr()
labels = [l for l, _ in FAC]

coef = pd.read_csv(REV + "/market_pw_coefs.csv")


def block(blk, ver):
    s = coef[(coef["block"] == blk) & (coef["version"] == ver)]
    return [{"facility": LAB[r.facility], "outcome": r.outcome, "b": r6(r.b), "se": r6(r.se),
             "p": r6(r.p), "q": r6(r.q) if pd.notna(r.q) else None, "n": int(r.n)}
            for r in s.itertuples()]


inter = pd.read_csv(REV + "/market_pw_interaction.csv")
IKEY = {"mandi": "mandi village share", "fertiliser shop": "fertiliser shop village share",
        "output-market index": "output-market index", "input-supply index": "input-supply index",
        "post-harvest index": "post-harvest index"}
inter_w = [{"measure": IKEY[r.measure], "curv_lo": r6(r.curv_lo), "curv_hi": r6(r.curv_hi),
            "b2_inter": r6(r.inter), "se_inter": r6(r.se), "p_inter": r6(r.p),
            "tp_lo": r6(r.tp_lo), "tp_hi": r6(r.tp_hi), "sd": r6(r.sd_raw), "n": int(r.n)}
           for r in inter[inter["version"] == "W"].itertuples()]

rob = pd.read_csv(REV + "/market_pw_robust_haat.csv")
LLAB = {"no adjustment": "No adjustment", "state baseline only": "Within states only",
        "+ irrigation": "Adding irrigation", "all measures": "All measures held constant",
        "drop shared parents": "Without districts carved from a split 2011 district",
        "drop districts with no haat": "Without districts that have no weekly haat",
        "with nine other facilities": "With all nine other facilities"}
ladder = [{"label": LLAB[r.spec], "b": r6(r.b), "se": r6(r.se), "p": r6(r.p), "n": int(r.n),
           "b_village": r6(rob[(rob["version"] == "U") & (rob["spec"] == r.spec)]["b"].iloc[0])}
          for r in rob[rob["version"] == "W"].itertuples()]

tt = pd.read_csv(REV + "/market_pw_typology_tests.csv")
band_tests = [{"outcome": r.outcome, "measure": "population" if r.version == "W" else "village",
               "mid": r6(r.mid), "p_mid": r6(r.p_mid), "top": r6(r.top), "p_top": r6(r.p_top),
               "haat": r6(r.h), "p_haat": r6(r.p_h), "p_joint": r6(r.p_joint)}
              for r in tt.itertuples() if r.version in ("U", "W")]

scatter = [{"haat": r6(r["m_weekly_haat_vshare"]), "mandi": r6(r["m_mandi_vshare"]),
            "fert": r6(r["m_fert_shop_vshare"]), "D0": r6(r["D0_richness"]),
            "D1": r6(r["D1_exp_shannon"]), "s": r["state_name"]} for _, r in dm.iterrows()]

out = {"note": ("Market facilities as POPULATION shares: the share of a district's rural population "
                "(Census 2011 village population) living in a village that reports the facility, "
                "Mission Antyodaya 2019-20. Coefficients are the change from no resident to every "
                "resident living in such a village, with state fixed effects and all controls "
                "(also population-weighted). solo_village keeps the village-share estimates."),
       "measure": "population share", "n": int(len(dm)),
       "descriptives": desc,
       "corr": {"labels": labels, "matrix": [[r6(v) for v in row] for row in cm.values]},
       "solo": block("solo", "W"), "solo_village": block("solo", "U"),
       "output_market": block("output", "W"), "input_supply": block("input", "W"),
       "post_harvest": block("post", "W"),
       "interaction": inter_w, "ladder": ladder, "band_tests": band_tests, "scatter": scatter}
p = DOCS + "/site_markets_pop.json"
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"))
print("WROTE {} ({:,} bytes)".format(p, os.path.getsize(p)))

# ------------------------------------------------------------ 4. numbers quoted in the prose
print("\n== descriptives (population share; village share in brackets)")
for d in desc:
    print("  {:22s} mean {:.3f} ({:.3f})  p10 {:.3f} p90 {:.3f} zero {}  sd {:.3f}  viirs {:+.2f} estab {:+.2f}".format(
        d["name"], d["mean"], d["mean_village"], d["p10"], d["p90"], d["zero"], d["sd"], d["viirs"], d["estab"]))
print("\n== correlations")
off = cm.where(~np.eye(len(cm), dtype=bool))
nh = [c for c in C if "haat" not in c]
sub = off.loc[nh, nh].stack()
print("  nine non-haat: {:.2f} to {:.2f}".format(sub.min(), sub.max()))
print("  top pair:", sub.idxmax(), round(sub.max(), 3))
hc = off.loc["m_weekly_haat_vshare"].dropna()
print("  haat vs others: {:.2f} to {:.2f}".format(hc.min(), hc.max()), hc.round(2).to_dict())
six = ["m_seed_centre_vshare", "m_soil_test_vshare", "m_custom_hire_vshare", "m_fert_shop_vshare",
       "m_farmgate_proc_vshare", "m_fpo_vshare"]
print("  six input/processing min:", round(off.loc[six, six].stack().min(), 3))
print("\n== state averages (all 606 districts)")
dm["st"] = dm["state_name"].str.title()
sm = dm.groupby("st")[C].mean()
sm.columns = labels
print(sm[["weekly haat"]].sort_values("weekly haat", ascending=False).head(8).round(3).to_string())
print(sm.loc[["Kerala", "Gujarat"], ["fertiliser shop", "FPO"]].round(3).to_string())
print("\n== variance split (states >= 8)")
for v in ty["variance_split"]["vars"]:
    print("  {:22s} between {:.3f}".format(v["var"], v["between"]))
print("\n== types")
for t in ty["types"]:
    print("  {:12s} n={:3d} D0 {:.1f} D1 {:.2f} even {:.3f} cereal {:.3f} fert {:.3f} fpo {:.3f} haat {:.3f}".format(
        t["code"], t["n"], t["D0"], t["D1"], t["evenness"], t["cereal_share"], t["fac"]["fertiliser shop"],
        t["fac"]["FPO"], t["fac"]["weekly haat"]))
print("  variance explained", ty["variance_explained"])
print("  stability", ty["type_stability"])
print("  kmeans silhouette range {:.2f} to {:.2f}".format(
    min(k["silhouette"] for k in ty["kmeans"]["diagnostics"]), max(k["silhouette"] for k in ty["kmeans"]["diagnostics"])))
print("  dense group", ty["kmeans"]["dense_group"]["n"], ty["kmeans"]["dense_group"]["states"])
print("  corr(density, haat)", round(np.corrcoef([v["dens"] for v in ty["districts"].values()],
                                                   [v["haat"] for v in ty["districts"].values()])[0, 1], 3))
