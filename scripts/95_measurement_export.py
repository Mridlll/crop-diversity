"""
95_measurement_export.py

USAGE:
    python scripts/95_measurement_export.py

Exports the data behind docs/measurement.html to
docs/data/site_measurement_data.json. Reuses the logic of scripts 76 and 81 and
leaves both unedited.

  subsample_moves   the 200 mean absolute shifts in the composite score when it
                    is recomputed on random 80 percent draws of districts (seed 11,
                    same loop as script 76, on the same district table)
  composite         SD of the composite, Shannon-Simpson correlation
  panel             district-year count and the D0 >= D1 >= D2 check
  in_sample         keys of the 606 market-analysis districts
  stats             median, p10, p90, mean of D0, D1, D2, E for the 725 and 606 sets,
                    and the correlations between them
  pooled            pooled vs within-year count of crops: ratio, rank movement
  examples          crop-share vectors for two real district-years (2019-20)
"""
import json
import numpy as np
import pandas as pd

REPO = r"D:/crop-diversity"
DIV = REPO + "/outputs/crop_diversity_analysis"
COV = REPO + "/outputs/shrug_covariates"
RAW = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
OUTF = REPO + "/docs/data/site_measurement_data.json"

out = {}

# ------------------------------------------------ composite, as in script 76
dist = pd.read_csv(DIV + "/district_diversity_indices_corrected.csv")
rng = np.random.default_rng(11)
moves = []
for _ in range(200):
    sub = dist.sample(frac=0.80, random_state=int(rng.integers(1e9)))
    a = sub.copy()
    for c in ["shannon_index", "simpson_index", "crop_richness"]:
        v = a[c]
        a[c + "_n2"] = (v - v.min()) / (v.max() - v.min())
    a["abi2"] = a[[c + "_n2" for c in
                   ["shannon_index", "simpson_index", "crop_richness"]]].mean(axis=1)
    j = a[["district_key", "abi2"]].merge(
        dist[["district_key", "agro_biodiversity_index"]], on="district_key")
    moves.append(float((j["abi2"] - j["agro_biodiversity_index"]).abs().mean()))
out["subsample_moves"] = [round(m, 5) for m in moves]
out["composite"] = {
    "n_districts": int(len(dist)),
    "move_mean": round(float(np.mean(moves)), 4),
    "move_max": round(float(np.max(moves)), 4),
    "sd": round(float(dist["agro_biodiversity_index"].std()), 4),
    "shannon_simpson_r": round(float(dist[["shannon_index", "simpson_index"]].corr().iloc[0, 1]), 3),
}

# ------------------------------------------------ district-year panel
ann = pd.read_csv(DIV + "/district_year_diversity_panel_corrected.csv")
viol = int(((ann["D0_richness"] < ann["D1_exp_shannon"] - 1e-9) |
            (ann["D1_exp_shannon"] < ann["D2_inv_simpson"] - 1e-9)).sum())
out["panel"] = {"n_district_years": int(len(ann)), "violations": viol,
                "first_year": "1997-98", "last_year": "2019-20"}

# ------------------------------------------------ sample flag and stats
ind = json.load(open(REPO + "/docs/data/site_indices.json"))["districts"]
I = pd.DataFrame(ind)
typ = pd.read_csv(COV + "/market_typology.csv")
samp = set(typ["div_key"])
out["in_sample"] = sorted(k for k in I["k"] if k in samp)


def summ(df):
    r = {"n": int(len(df))}
    for c in ["D0", "D1", "D2", "E"]:
        v = df[c]
        r[c] = {"mean": round(float(v.mean()), 3), "median": round(float(v.median()), 3),
                "p10": round(float(v.quantile(.1)), 3), "p90": round(float(v.quantile(.9)), 3),
                "min": round(float(v.min()), 3), "max": round(float(v.max()), 3)}
    r["corr"] = {"D0_D1": round(float(df["D0"].corr(df["D1"])), 3),
                 "D1_D2": round(float(df["D1"].corr(df["D2"])), 3),
                 "D0_D2": round(float(df["D0"].corr(df["D2"])), 3),
                 "E_D0": round(float(df["E"].corr(df["D0"])), 3),
                 "E_D1": round(float(df["E"].corr(df["D1"])), 3)}
    r["violations"] = int(((df["D0"] < df["D1"] - 1e-9) | (df["D1"] < df["D2"] - 1e-9)).sum())
    return r


out["stats"] = {"all": summ(I), "sample": summ(I[I["k"].isin(samp)])}

# ------------------------------------------------ pooled vs within-year count
P = I.dropna(subset=["D0_o"]).copy()
P["ratio"] = P["D0_o"] / P["D0"]
P["r_pool"] = P["D0_o"].rank(ascending=False)
P["r_ann"] = P["D0"].rank(ascending=False)
P["move"] = (P["r_pool"] - P["r_ann"]).abs()
top = P.nlargest(10, "move")
out["pooled"] = {
    "n": int(len(P)),
    "ratio_median": round(float(P["ratio"].median()), 3),
    "ratio_mean": round(float(P["ratio"].mean()), 3),
    "under20_years": int((P["y"] < 20).sum()),
    "corr_years_pooled": round(float(P["D0_o"].corr(P["y"])), 3),
    "corr_years_annual": round(float(P["D0"].corr(P["y"])), 3),
    "rank_move_median": round(float(P["move"].median()), 1),
    "rank_move_p90": round(float(P["move"].quantile(.9)), 1),
    "rank_move_max": round(float(P["move"].max()), 1),
    "rank_move_over100": int((P["move"] > 100).sum()),
    "top_movers_ratio_median": round(float(top["ratio"].median()), 3),
    "top_movers": [{"k": r.k, "y": int(r.y), "D0": round(r.D0, 2), "D0_o": r.D0_o,
                    "move": r.move} for r in top.itertuples()],
}

# ------------------------------------------------ real district-years, script 76 cleaning
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
pair = list(zip(df["state_name"], df["district_name"]))
df = df[[p not in BOGUS for p in pair]]
df["year_start"] = df["year"].str.split("-").str[0].astype(int)
df["district_key"] = df["state_name"].str.upper() + "|" + df["district_name"].str.upper()
df = df[df["year_start"] <= 2019]
K = ["district_key", "year", "season", "crop_name"]
df = df.sort_values("area", ascending=False).drop_duplicates(subset=K, keep="first")

EX = [("WEST BENGAL|BANKURA", 2019, "Bankura, West Bengal"),
      ("KARNATAKA|CHAMARAJANAGARA", 2019, "Chamarajanagara, Karnataka")]
ex = []
for key, yr, name in EX:
    s = (df[(df["district_key"] == key) & (df["year_start"] == yr)]
         .groupby("crop_name")["area"].sum().sort_values(ascending=False))
    if s.empty:
        print("MISSING example", key, yr)
        continue
    p = s / s.sum()
    d1 = float(np.exp(-(p * np.log(p)).sum()))
    d2 = float(1 / (p ** 2).sum())
    ex.append({"key": key, "name": name, "year": "{}-{}".format(yr, str(yr + 1)[-2:]),
               "crops": [c for c in p.index], "shares": [round(float(v) * 100, 3) for v in p.values],
               "D0": int(len(p)), "D1": round(d1, 3), "D2": round(d2, 3)})
out["examples"] = ex

out["note"] = ("Data for measurement.html. Shares in examples are percent of gross cropped area. "
               "stats.sample is the 606-district analysis set used on markets.html.")
with open(OUTF, "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"))
print(json.dumps({k: v for k, v in out.items() if k not in ("subsample_moves", "in_sample")}, indent=1)[:6000])
print("in_sample", len(out["in_sample"]))
