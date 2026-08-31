"""
81_export_site_data.py

Exports the analysis results to JSON for the site under docs/.

Everything the pages draw comes from here, so a figure can never drift from the
numbers in the reports. Re-run this after any change to scripts 76 to 80.

Outputs (all under docs/data/):
  site_indices.json       D0/D1/D2/evenness per district, plus the ABI for contrast
  site_irrigation.json    decile profile, robustness ladder, source decomposition
  site_markets.json       facility descriptives, correlations, coefficients
  site_audit.json         the construction defects, quantified
  site_meta.json          counts, coverage, provenance summary
"""
import json
import os

import numpy as np
import pandas as pd

REPO = r"D:/crop-diversity"
COV = REPO + "/outputs/shrug_covariates"
DIV = REPO + "/outputs/crop_diversity_analysis"
DOCS = REPO + "/docs/data"
os.makedirs(DOCS, exist_ok=True)


def dump(obj, name):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else round(float(o), 6)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(str(type(o)))
    with open(DOCS + "/" + name, "w", encoding="utf-8") as f:
        json.dump(obj, f, default=default, separators=(",", ":"))
    print("  {:26s} {:>9,} bytes".format(name, os.path.getsize(DOCS + "/" + name)))


def clean(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), 6)


print("Exporting site data")

fp = pd.read_csv(COV + "/final_panel.csv")
d = fp[fp["in_final"]].copy()
corr = pd.read_csv(DIV + "/district_diversity_indices_corrected.csv")
orig = pd.read_csv(DIV + "/district_diversity_indices.csv")

# ------------------------------------------------------------- indices
m = corr.merge(orig[["district_key", "agro_biodiversity_index", "crop_richness",
                     "shannon_index"]],
               on="district_key", how="left", suffixes=("", "_orig"))
idx = [{
    "k": r["district_key"], "s": r["state_name"], "d": r["district_name"],
    "y": int(r["n_years"]),
    "D0": clean(r["D0_richness"]), "D1": clean(r["D1_exp_shannon"]),
    "D2": clean(r["D2_inv_simpson"]), "E": clean(r["evenness_D1_D0"]),
    "abi": clean(r["agro_biodiversity_index"]),
    "abi_o": clean(r["agro_biodiversity_index_orig"]),
    "D0_o": clean(r["crop_richness_orig"]),
} for _, r in m.iterrows()]
dump({"districts": idx,
      "note": "D0 crops grown, D1 exp(Shannon), D2 inverse Simpson, E = D1/D0. "
              "_o suffix is the original pooled construction."},
     "site_indices.json")

# ------------------------------------------------------------ overview
# The front page is a description of what Indian agrobiodiversity looks like,
# before any explanation is attempted. Everything it needs is built here.
panel = pd.read_csv(DIV + "/district_year_diversity_panel_corrected.csv")
cc = corr.copy()
cc[["state_name", "district_name"]] = cc["district_key"].str.split("|", expand=True)

st = (cc.groupby("state_name")
        .agg(n=("district_key", "size"), D0=("D0_richness", "mean"),
             D1=("D1_exp_shannon", "mean"), D2=("D2_inv_simpson", "mean"),
             E=("evenness_D1_D0", "mean"),
             area=("mean_annual_cropped_area", "sum"))
        .query("n >= 4").sort_values("D1", ascending=False).reset_index())

# national trend: area-weighted so a small district does not swing it
panel = panel.merge(cc[["district_key"]], on="district_key", how="inner")
tr = []
for y, g in panel.groupby("year_start"):
    wgt = g["cropped_area"].fillna(0)
    if wgt.sum() <= 0:
        continue
    tr.append({"year": int(y), "n": int(len(g)),
               "D0": clean(np.average(g["D0_richness"], weights=wgt)),
               "D1": clean(np.average(g["D1_exp_shannon"], weights=wgt)),
               "D2": clean(np.average(g["D2_inv_simpson"], weights=wgt)),
               "E": clean(np.average(g["evenness_D1_D0"], weights=wgt)),
               "area": clean(wgt.sum() / 1e6)})
tr = [t for t in tr if t["n"] >= 300]

# The unbalanced trend is not like-for-like: the district count climbs from under
# 500 to nearly 700 across the period, so a rise could be new districts entering
# rather than existing ones changing. A balanced panel of districts observed in
# every year is the honest series.
yrs_all = sorted(panel["year_start"].unique())
cnt = panel.groupby("district_key")["year_start"].nunique()
bal_keys = cnt[cnt == len(yrs_all)].index
bal_panel = panel[panel["district_key"].isin(bal_keys)]
trb = []
for y, g in bal_panel.groupby("year_start"):
    wgt = g["cropped_area"].fillna(0)
    if wgt.sum() <= 0:
        continue
    trb.append({"year": int(y), "n": int(len(g)),
                "D0": clean(np.average(g["D0_richness"], weights=wgt)),
                "D1": clean(np.average(g["D1_exp_shannon"], weights=wgt)),
                "D2": clean(np.average(g["D2_inv_simpson"], weights=wgt)),
                "E": clean(np.average(g["evenness_D1_D0"], weights=wgt))})
print("  balanced panel: {} districts x {} years".format(len(bal_keys), len(yrs_all)))

# District-level change. The national series is area-weighted, so it can sit flat
# while most districts move. Those are different questions and both get answered.
# 1997 is dropped from the change window: it sits far below every later year on
# every index, which is a reporting artefact rather than an agronomic fact.
early = bal_panel[bal_panel["year_start"].between(1998, 2004)]
late = bal_panel[bal_panel["year_start"].between(2013, 2019)]
ch = (early.groupby("district_key")[["D0_richness", "D1_exp_shannon", "evenness_D1_D0"]]
           .mean()
           .join(late.groupby("district_key")[["D0_richness", "D1_exp_shannon",
                                               "evenness_D1_D0"]].mean(),
                 lsuffix="_e", rsuffix="_l").dropna())
ch["dD1"] = ch["D1_exp_shannon_l"] - ch["D1_exp_shannon_e"]
ch["dD0"] = ch["D0_richness_l"] - ch["D0_richness_e"]
ch["dE"] = ch["evenness_D1_D0_l"] - ch["evenness_D1_D0_e"]
change = {
    "n": int(len(ch)),
    "window": "1998-2004 against 2013-2019",
    "D1_up": int((ch["dD1"] > 0).sum()), "D1_down": int((ch["dD1"] < 0).sum()),
    "D0_up": int((ch["dD0"] > 0).sum()), "D0_down": int((ch["dD0"] < 0).sum()),
    "E_up": int((ch["dE"] > 0).sum()), "E_down": int((ch["dE"] < 0).sum()),
    "dD1_median": clean(ch["dD1"].median()), "dD0_median": clean(ch["dD0"].median()),
    "dE_median": clean(ch["dE"].median()),
    "hist_dD1": [clean(v) for v in ch["dD1"]],
    "hist_dE": [clean(v) for v in ch["dE"]],
}
print("  district change: D1 up {} down {}, D0 up {} down {}".format(
    change["D1_up"], change["D1_down"], change["D0_up"], change["D0_down"]))

dom = (cc.groupby("dominant_crop")
         .agg(n=("district_key", "size"), share=("dominant_crop_share", "mean"))
         .sort_values("n", ascending=False).head(12).reset_index())

CATS = [c for c in cc.columns if c.startswith("share_")]
cc["_q"] = pd.qcut(cc["D1_exp_shannon"], 4, labels=["Q1 least diverse", "Q2", "Q3",
                                                    "Q4 most diverse"])
catq = cc.groupby("_q", observed=True)[CATS].mean().reset_index()
catq["_q"] = catq["_q"].astype(str)

dump({"states": st.replace({np.nan: None}).to_dict("records"),
      "trend": tr,
      "trend_balanced": trb,
      "change": change,
      "dominant": dom.replace({np.nan: None}).to_dict("records"),
      "cat_by_quartile": catq.replace({np.nan: None}).to_dict("records"),
      "cat_labels": [c.replace("share_", "").replace("_", " ") for c in CATS],
      "cat_keys": CATS,
      "national": {"D0": clean(cc["D0_richness"].mean()),
                   "D1": clean(cc["D1_exp_shannon"].mean()),
                   "D2": clean(cc["D2_inv_simpson"].mean()),
                   "E": clean(cc["evenness_D1_D0"].mean()),
                   "n": int(len(cc))},
      "top": cc.nlargest(8, "D1_exp_shannon")[
          ["district_key", "D0_richness", "D1_exp_shannon", "dominant_crop",
           "dominant_crop_share"]].replace({np.nan: None}).to_dict("records"),
      "bottom": cc.nsmallest(8, "D1_exp_shannon")[
          ["district_key", "D0_richness", "D1_exp_shannon", "dominant_crop",
           "dominant_crop_share"]].replace({np.nan: None}).to_dict("records")},
     "site_overview.json")

# ---------------------------------------------------------- irrigation
d["dec"] = pd.qcut(d["irr_share"], 10, labels=False, duplicates="drop") + 1
prof = d.groupby("dec").agg(
    n=("D1_exp_shannon", "size"), irr=("irr_share", "mean"),
    D0=("D0_richness", "mean"), D1=("D1_exp_shannon", "mean"),
    D2=("D2_inv_simpson", "mean"), E=("evenness_D1_D0", "mean"),
    D1se=("D1_exp_shannon", lambda s: s.std() / np.sqrt(len(s))),
    D0se=("D0_richness", lambda s: s.std() / np.sqrt(len(s))),
    cereal=("share_cereals", "mean")).reset_index()

fam = pd.read_csv(COV + "/final_dimension_results.csv")
src = []
for line in open(COV + "/final_results.md", encoding="utf-8"):
    if line.startswith("| ") and ("_exp_shannon" in line or "_richness" in line
                                  or "share_" in line or "inv_simpson" in line
                                  or "evenness" in line):
        src.append([c.strip() for c in line.strip().strip("|").split("|")])

scat = [{"x": clean(r["irr_share"]), "y": clean(r["D1_exp_shannon"]),
         "d0": clean(r["D0_richness"]), "s": r["state_name"], "k": r["div_key"]}
        for _, r in d.iterrows()]

rob = pd.read_csv(COV + "/final_robustness.csv")
srcres = pd.read_csv(COV + "/final_source_results.csv")
srcmix = {"canal": clean(d["ay_src_canal_vshare"].mean()),
          "ground": clean(d["ay_src_ground_vshare"].mean()),
          "surface": clean(d["ay_src_surface_vshare"].mean()),
          "other": clean(d["ay_src_other_vshare"].mean())}
dump({"profile": prof.replace({np.nan: None}).to_dict("records"),
      "scatter": scat,
      "dimensions": fam.replace({np.nan: None}).to_dict("records"),
      "robustness": rob.replace({np.nan: None}).to_dict("records"),
      "source": srcres.replace({np.nan: None}).to_dict("records"),
      "source_mix": srcmix,
      "n": len(d), "states": int(d["state_name"].nunique())},
     "site_irrigation.json")

# ------------------------------------------------------------- markets
mk = pd.read_csv(COV + "/market_covariates.csv")
for t in (fp, mk):
    for c in ["pc11_state_id", "pc11_district_id"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
dm = fp.merge(mk, on=["pc11_state_id", "pc11_district_id"], how="left")
dm = dm[dm["in_final"] & dm["idx_output_market"].notna()].copy()

FAC = [("mandi", "m_mandi_vshare"), ("regular market", "m_regular_market_vshare"),
       ("weekly haat", "m_weekly_haat_vshare"), ("fertiliser shop", "m_fert_shop_vshare"),
       ("seed centre", "m_seed_centre_vshare"), ("soil testing", "m_soil_test_vshare"),
       ("custom hiring", "m_custom_hire_vshare"), ("cold storage", "m_storage_vshare"),
       ("farm-gate processing", "m_farmgate_proc_vshare"), ("FPO", "m_fpo_vshare")]
desc = [{"name": lab, "mean": clean(dm[c].mean()),
         "p10": clean(dm[c].quantile(.1)), "p90": clean(dm[c].quantile(.9)),
         "zero": int((dm[c] == 0).sum()),
         "viirs": clean(dm[[c, "log_viirs"]].corr().iloc[0, 1]),
         "estab": clean(dm[[c, "estab_per_1000pop"]].corr().iloc[0, 1])}
        for lab, c in FAC]
cmat = dm[[c for _, c in FAC]].corr()
cmat.index = [l for l, _ in FAC]
cmat.columns = [l for l, _ in FAC]

solo = pd.read_csv(COV + "/market_results_solo.csv")
outp = pd.read_csv(COV + "/market_results_output.csv")
inp = pd.read_csv(COV + "/market_results_input.csv")
inter = pd.read_csv(COV + "/market_results_interaction.csv")

haat_scatter = [{"haat": clean(r["m_weekly_haat_vshare"]),
                 "mandi": clean(r["m_mandi_vshare"]),
                 "fert": clean(r["m_fert_shop_vshare"]),
                 "D0": clean(r["D0_richness"]), "D1": clean(r["D1_exp_shannon"]),
                 "s": r["state_name"]} for _, r in dm.iterrows()]

dump({"descriptives": desc,
      "corr": {"labels": list(cmat.columns),
               "matrix": [[clean(v) for v in row] for row in cmat.values]},
      "solo": solo.replace({np.nan: None}).to_dict("records"),
      "output_market": outp.replace({np.nan: None}).to_dict("records"),
      "input_supply": inp.replace({np.nan: None}).to_dict("records"),
      "interaction": inter.replace({np.nan: None}).to_dict("records"),
      "scatter": haat_scatter, "n": len(dm)},
     "site_markets.json")

# --------------------------------------------------------------- audit
cd = pd.read_csv(COV + "/diversity_corrected_annual.csv")
raw_years = []
for line in open(COV + "/corrected_vs_original.md", encoding="utf-8"):
    if line.startswith("| 20") or line.startswith("| 19"):
        p = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(p) == 4:
            raw_years.append({"year": p[0], "rows": p[1], "mha": p[2], "dist": p[3]})

cov_rich = cd.groupby("n_years").agg(
    pooled=("rich_pooled", "mean"), annual=("rich_annual_mean", "mean"),
    n=("district_key", "size")).reset_index()
cov_rich = cov_rich[cov_rich["n"] >= 5]

mv = corr.merge(orig[["district_key", "agro_biodiversity_index"]],
                on="district_key", suffixes=("", "_o"))
mv["r_new"] = mv["agro_biodiversity_index"].rank(ascending=False)
mv["r_old"] = mv["agro_biodiversity_index_o"].rank(ascending=False)
mv["move"] = (mv["r_old"] - mv["r_new"]).abs()

dump({"years": raw_years,
      "coverage_richness": cov_rich.replace({np.nan: None}).to_dict("records"),
      "richness_pair": [{"a": clean(a), "p": clean(b)} for a, b in
                        zip(cd["rich_annual_mean"], cd["rich_pooled"])],
      "rank_moves": [clean(v) for v in mv["move"]],
      "abi_pair": [{"o": clean(a), "n": clean(b)} for a, b in
                   zip(mv["agro_biodiversity_index_o"], mv["agro_biodiversity_index"])],
      "movers": mv.nlargest(10, "move")[
          ["district_key", "n_years", "agro_biodiversity_index_o",
           "agro_biodiversity_index", "move"]].replace({np.nan: None}).to_dict("records")},
     "site_audit.json")

# ---------------------------------------------------------------- meta
prov = pd.read_csv(COV + "/provenance.csv")
cw = pd.read_csv(COV + "/district_crosswalk.csv")
checks = []
for line in open(COV + "/validation_report.md", encoding="utf-8"):
    if line.startswith("| C") and "|" in line:
        p = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(p) == 3 and p[1] in ("PASS", "WARN", "FAIL"):
            checks.append({"check": p[0], "status": p[1], "detail": p[2]})

dump({"n_analysis": len(d), "n_states": int(d["state_name"].nunique()),
      "n_covariates": int(len(prov)),
      "n_districts_shrug": 631,
      "crosswalk": cw["method"].value_counts().to_dict(),
      "match_rate": clean(cw["shrug_key"].notna().mean()),
      "shared_parent": int(cw["n_sharing_parent"].fillna(1).gt(1).sum()),
      "checks": checks,
      "modules": prov.groupby("module").size().to_dict()},
     "site_meta.json")

print("\nDone. {} files in docs/data/.".format(
    len([f for f in os.listdir(DOCS) if f.startswith("site_")])))
