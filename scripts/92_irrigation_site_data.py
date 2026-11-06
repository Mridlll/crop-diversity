"""
92_irrigation_site_data.py

Data for the scene visuals on docs/irrigation.html.

Rebuilds, from final_panel.csv (the 606-district analysis set of script 77),
every quantity the irrigation page draws: the fitted irrigation curves with
their covariance, district points, the decile profile with standard errors and
crop-group and source shares, the three irrigation groups on both irrigation
measures, the fourteen robustness rows, the source coefficients and the
district-characteristic coefficients with standard deviations. Each regression
is re-run exactly as in script 77 and checked against its CSVs.

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/92_irrigation_site_data.py

Output:
  docs/data/site_irrigation_story.json
"""
import json
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REPO = r"D:/crop-diversity"
COV = REPO + "/outputs/shrug_covariates"
GEO = REPO + "/docs/data/districts.geojson"
OUT = REPO + "/docs/data/site_irrigation_story.json"

ALIAS = {"orissa": "odisha", "pondicherry": "puducherry", "uttaranchal": "uttarakhand",
         "jammu and kashmir": "jammu kashmir", "nct of delhi": "delhi",
         "andaman and nicobar islands": "andaman nicobar islands",
         "andaman and nicobar": "andaman nicobar islands",
         "dadra and nagar haveli": "dadra nagar haveli", "daman and diu": "daman diu",
         "the dadra and nagar haveli and daman and diu": "dadra nagar haveli"}


def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
    return re.sub(r"\s+", " ", s)


def key(state, dist):
    return ALIAS.get(norm(state), norm(state)) + "|" + norm(dist)


def title(s):
    return " ".join(x if x in ("and", "of") else x.capitalize() for x in str(s).lower().split())


def r(v, k=4):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), k)


p = pd.read_csv(COV + "/final_panel.csv")
d = p[p["in_final"] == True].copy().reset_index(drop=True)
assert len(d) == 606 and d["state_name"].nunique() == 30

# ---------------------------------------------------------------- sample accounting
few = p["n_years"] < 10
cov_bad = (p["flag_low_ay_coverage"].fillna(True).astype(bool)
           | p["flag_no_antyodaya"].fillna(True).astype(bool) | p["shrug_key"].isna()
           | p["irr_share"].isna())
acct = dict(total=int(len(p)), few_years=int(few.sum()),
            coverage_only=int((~few & cov_bad).sum()), final=606,
            median_years=float(d["n_years"].median()),
            years_23=int((d["n_years"] == 23).sum()), years_22plus=int((d["n_years"] >= 22).sum()),
            shared_parent=int(d["flag_shared_parent"].sum()), groups=int(d["shrug_key"].nunique()))
assert acct["total"] - acct["few_years"] - acct["coverage_only"] == 606
print("accounting", acct)
print("coverage exclusions by state",
      p[~few & cov_bad]["state_name"].value_counts().to_dict())

Y = {"D0": "D0_richness", "D1": "D1_exp_shannon", "D2": "D2_inv_simpson", "E": "evenness_D1_D0"}
CL = lambda df: dict(cov_type="cluster", cov_kwds={"groups": df["shrug_key"]})
SQ = "I(irr_share ** 2)"


# ---------------------------------------------------------------- curves
def curve(y):
    out = {}
    m0 = smf.ols(f"{y} ~ irr_share + I(irr_share**2)", d).fit(**CL(d))
    V0 = m0.cov_params().loc[["Intercept", "irr_share", SQ], ["Intercept", "irr_share", SQ]].values
    b0, b1, b2 = m0.params["Intercept"], m0.params["irr_share"], m0.params[SQ]
    out["pooled"] = dict(b0=b0, b1=b1, b2=b2, V=V0.tolist(), tp=-b1 / (2 * b2), r2=m0.rsquared)
    m1 = smf.ols(f"{y} ~ irr_share + I(irr_share**2) + C(state_name)", d).fit(**CL(d))
    V1 = m1.cov_params().loc[["irr_share", SQ], ["irr_share", SQ]].values
    b1, b2 = m1.params["irr_share"], m1.params[SQ]
    xm, x2m = d["irr_share"].mean(), (d["irr_share"] ** 2).mean()
    c = d[y].mean() - b1 * xm - b2 * x2m
    out["state"] = dict(c=c, b1=b1, b2=b2, xm=xm, x2m=x2m, V=V1.tolist(), tp=-b1 / (2 * b2),
                        r2=m1.rsquared)
    for kk, v in out.items():
        f = lambda x: (v.get("b0", v.get("c")) + v["b1"] * x + v["b2"] * x * x)
        v["at0"], v["peak"], v["at1"] = f(0.0), f(v["tp"]), f(1.0)
        print(y, kk, {a: round(b, 4) for a, b in v.items() if not isinstance(b, list)})
    return out


curves = {"D1": curve(Y["D1"]), "D0": curve(Y["D0"])}
assert abs(curves["D1"]["state"]["b2"] + 6.876783413117592) < 1e-9
assert abs(curves["D1"]["pooled"]["b2"] + 10.185227455305288) < 1e-9

# ---------------------------------------------------------------- regimes and keys
cut = lambda s: pd.cut(s, [-0.01, 0.40, 0.60, 1.01], labels=[0, 1, 2])
d["reg"] = cut(d["irr_share"]).astype(int)
d["regvd"] = cut(d["irr_share_vd11"])
d["k"] = [key(*v.split("|", 1)) for v in d["div_key"]]
assert d["k"].is_unique
g = gpd.read_file(GEO)
names = dict(zip(g["k"], g["n"]))
gstate = dict(zip(g["k"], g["s"]))
# polygons whose geojson key differs from the district key: renamed districts,
# Telangana and Uttarakhand parents filed under their pre-split state names
OVR = {
    "andaman nicobar islands|north and middle andaman": ["North  & Middle Andaman|Andaman & Nicobar"],
    "andaman nicobar islands|south andamans": ["South Andaman|Andaman & Nicobar"],
    "andhra pradesh|kadapa": ["Y.S.R.|Andhra Pradesh"],
    "andhra pradesh|karimnagar": ["Karimnagar|Telangana"], "andhra pradesh|khammam": ["Khammam|Telangana"],
    "andhra pradesh|medak": ["Medak|Telangana"], "andhra pradesh|nalgonda": ["Nalgonda|Telangana"],
    "andhra pradesh|nizamabad": ["Nizamabad|Telangana"], "andhra pradesh|rangareddi": ["Ranga Reddy|Telangana"],
    "andhra pradesh|warangal": ["Warangal Rural|Telangana", "Warangal Urban|Telangana"],
    "arunachal pradesh|dibang valley": ["Upper Dibang Valley|Arunachal Pradesh"],
    "haryana|gurgaon": ["Gurugram|Haryana"], "haryana|mewat": ["Nuh|Haryana"],
    "karnataka|bangalore rural": ["Bengaluru Rural|Karnataka"], "ladakh|leh ladakh": ["Leh|Ladakh"],
    "punjab|firozepur": ["Firozpur|Punjab"], "punjab|muktsar": ["Sri Muktsar Sahib|Punjab"],
    "punjab|s a s nagar": ["Sahibzada Ajit Singh Nag*|Punjab"],
    "tamil nadu|kanchipuram": ["Kancheepuram|Tamil Nadu"], "tamil nadu|villupuram": ["Viluppuram|Tamil Nadu"],
    "uttar pradesh|allahabad": ["Prayagraj|Uttar Pradesh"], "uttar pradesh|sant ravidas nagar": ["Bhadohi|Uttar Pradesh"],
    "uttarakhand|rudra prayag": ["Rudraprayag|Uttarakhand"], "uttarakhand|udam singh nagar": ["Udham Singh Nagar|Uttarakhand"],
    "uttarakhand|uttar kashi": ["Uttarkashi|Uttarakhand"],
    "west bengal|north 24 parganas": ["North Twenty Four Pargan*|West Bengal"],
    "west bengal|south 24 parganas": ["South Twenty Four Pargan*|West Bengal"],
    "west bengal|pashchim medinipur": ["Medinipur West|West Bengal"],
}
fnames = set(g["n"] + "|" + g["s"])
geo_by_name = {}
for k_, lst in OVR.items():
    assert k_ in set(d["k"]), k_
    for nm in lst:
        assert nm in fnames, nm
        geo_by_name[nm] = k_
on_map = set(d["k"]) & (set(names) | set(OVR))
print("districts on map", len(on_map), "of", len(d))
for k_, nm in OVR.items():
    names.setdefault(k_, nm[0].split("|")[0].replace("*", "").strip())
    gstate.setdefault(k_, nm[0].split("|")[1])


def regimes(col):
    rows = []
    for gi, s in d.dropna(subset=[col]).groupby(col, observed=True):
        rows.append(dict(g=int(gi), n=int(len(s)), irr=r(s["irr_share"].mean() if col == "reg"
                                                          else s["irr_share_vd11"].mean()),
                         D1=r(s[Y["D1"]].mean()), D0=r(s[Y["D0"]].mean(), 3),
                         D2=r(s[Y["D2"]].mean()), E=r(s[Y["E"]].mean()),
                         cereal=r(s["share_cereals"].mean())))
    return rows


reg = {"ay": regimes("reg"), "vd": regimes("regvd")}
print("regimes", reg)

# ---------------------------------------------------------------- districts
SRC = {"ground": "ay_src_ground_vshare", "canal": "ay_src_canal_vshare",
       "surface": "ay_src_surface_vshare", "other": "ay_src_other_vshare"}
dist = []
for _, x in d.iterrows():
    dist.append(dict(k=x["k"], n=names.get(x["k"]) or title(x["div_key"].split("|", 1)[1]),
                     s=gstate.get(x["k"]) or title(x["state_name"]),
                     x=r(x["irr_share"]), D0=r(x[Y["D0"]], 2), D1=r(x[Y["D1"]], 3),
                     D2=r(x[Y["D2"]], 3), E=r(x[Y["E"]]), ce=r(x["share_cereals"], 3),
                     can=r(x["ay_src_canal_vshare"], 3), g=int(x["reg"]),
                     sp=bool(x["flag_shared_parent"])))

# ---------------------------------------------------------------- deciles
d["dec"] = pd.qcut(d["irr_share"], 10, labels=False, duplicates="drop") + 1
prof = []
se = lambda s: float(s.std(ddof=1) / np.sqrt(len(s)))
for dec, s in d.groupby("dec"):
    row = dict(dec=int(dec), n=int(len(s)), irr=r(s["irr_share"].mean()),
               lo=r(s["irr_share"].min(), 3), hi=r(s["irr_share"].max(), 3))
    for kk, c in Y.items():
        row[kk] = r(s[c].mean()); row[kk + "se"] = r(se(s[c]))
    for kk, c in [("cereal", "share_cereals"), ("pulse", "share_pulses"), ("oilseed", "share_oilseeds")]:
        row[kk] = r(s[c].mean()); row[kk + "se"] = r(se(s[c]))
    for kk, c in SRC.items():
        row[kk] = r(s[c].mean(), 3)
    prof.append(row)
    print("dec", row)
old = json.load(open(REPO + "/docs/data/site_irrigation.json"))["profile"]
for a, b in zip(prof, old):
    assert abs(a["D1"] - b["D1"]) < 1e-3 and abs(a["cereal"] - b["cereal"]) < 1e-3 and a["n"] == b["n"]
mix = {kk: r(d[c].mean(), 4) for kk, c in SRC.items()}
print("source mix", mix)


# ---------------------------------------------------------------- robustness (script 77 SPECS)
def fit(df, y="D1_exp_shannon", x="irr_share", fe=True, wt=None):
    df = df.dropna(subset=[y, x, "shrug_key"]).copy()
    if wt:
        df = df.dropna(subset=[wt])
    f = "{} ~ {} + I({}**2)".format(y, x, x) + (" + C(state_name)" if fe else "")
    m = (smf.wls(f, data=df, weights=df[wt]) if wt else smf.ols(f, data=df)).fit(**CL(df))
    k2 = "I({} ** 2)".format(x)
    b1, b2 = m.params[x], m.params[k2]
    return dict(n=int(m.nobs), b1=b1, b2=b2, se2=m.bse[k2], p2=m.pvalues[k2], tp=-b1 / (2 * b2),
                r2=m.rsquared)


conflict = d["flag_irr_source_conflict"].fillna(False).astype(bool)
SPECS = [
    ("no state baseline", "D1", "comparison", lambda: fit(d, fe=False)),
    ("state baseline (headline)", "D1", "comparison", lambda: fit(d)),
    ("irrigation from the 2011 village directory", "D1", "sample", lambda: fit(d, x="irr_share_vd11")),
    ("without districts sharing 2011 records", "D1", "sample", lambda: fit(d[~d["flag_shared_parent"]])),
    ("districts sharing 2011 records weighted down", "D1", "weighting", lambda: fit(d, wt="w_parent")),
    ("without source conflicts", "D1", "sample", lambda: fit(d[~conflict])),
    ("at least 22 years of records", "D1", "sample", lambda: fit(d[d["n_years"] >= 22])),
    ("weighted by cropped area", "D1", "weighting", lambda: fit(d, wt="mean_annual_cropped_area")),
    ("count of crops", "D0", "measure", lambda: fit(d, y="D0_richness")),
    ("number of main crops", "D2", "measure", lambda: fit(d, y="D2_inv_simpson")),
    ("evenness", "E", "measure", lambda: fit(d, y="evenness_D1_D0")),
    ("Shannon index", "H", "measure", lambda: fit(d, y="shannon_index")),
    ("Simpson index", "S", "measure", lambda: fit(d, y="simpson_index")),
    ("composite index, for comparison", "ABI", "measure", lambda: fit(d, y="agro_biodiversity_index")),
]
ref = pd.read_csv(COV + "/final_robustness.csv")
rob = []
for i, (lab, y, kind, fn) in enumerate(SPECS):
    o = fn()
    assert abs(o["b2"] - ref.loc[i, "b2"]) < 1e-9 and o["n"] == ref.loc[i, "n"], lab
    rob.append(dict(lab=lab, y=y, kind=kind, n=o["n"], b1=r(o["b1"], 5), b2=r(o["b2"], 5),
                    se2=r(o["se2"], 5), p2=float("%.4g" % o["p2"]), tp=r(o["tp"], 4), r2=r(o["r2"], 4)))
print("robustness ok; D1 tp range",
      min(x["tp"] for x in rob[:8]), max(x["tp"] for x in rob[:8]),
      "median", float(np.median([x["tp"] for x in rob[:8]])))

# drop the three states where the irrigated share runs well above published figures
s3 = d[~d["state_name"].str.upper().isin(["MAHARASHTRA", "JHARKHAND", "ASSAM"])]
o3 = fit(s3)
drop3 = dict(n=o3["n"], b2=r(o3["b2"], 4), p2=float("%.3g" % o3["p2"]), tp=r(o3["tp"], 4))
print("drop MH/JH/AS", drop3)

# ---------------------------------------------------------------- source
OUTC = [("D1", "D1_exp_shannon"), ("D0", "D0_richness"), ("D2", "D2_inv_simpson"),
        ("E", "evenness_D1_D0"), ("cereal", "share_cereals"), ("pulse", "share_pulses"),
        ("oilseed", "share_oilseeds")]
sref = pd.read_csv(COV + "/final_source_results.csv")
src = []
for i, (kk, dep) in enumerate(OUTC):
    s2 = d.dropna(subset=[dep, "irr_share", "ay_src_canal_vshare", "ay_src_surface_vshare",
                          "mean_holding_ha", "shrug_key"])
    m = smf.ols(dep + " ~ irr_share + I(irr_share**2) + ay_src_canal_vshare + "
                "ay_src_surface_vshare + np.log(mean_holding_ha) + pca_st_share + "
                "C(state_name)", data=s2).fit(**CL(s2))
    assert abs(m.params["ay_src_canal_vshare"] - sref.loc[i, "b_canal"]) < 1e-9
    assert abs(m.params["ay_src_surface_vshare"] - sref.loc[i, "b_surface"]) < 1e-9
    row = dict(y=kk, n=int(m.nobs))
    for sk, c in [("canal", "ay_src_canal_vshare"), ("surface", "ay_src_surface_vshare")]:
        row[sk] = dict(b=r(m.params[c], 5), se=r(m.bse[c], 5), p=float("%.4g" % m.pvalues[c]))
    src.append(row)
print("source", src)

# ---------------------------------------------------------------- district characteristics
SPEC = [
    ("irr", "irrigated share of sown area", "irr_share"),
    ("canal", "canal village share", "ay_src_canal_vshare"),
    ("hold", "mean holding size (log)", "np.log(mean_holding_ha)"),
    ("cult", "cultivator share of agricultural workers", "cultivator_share_agwork"),
    ("landless", "landless share of rural households", "secc_landless_share"),
    ("mandi", "villages with a mandi", "mandi_vshare"),
    ("haat", "villages with a weekly haat", "weekly_haat_vshare"),
    ("regular", "villages with a regular market", "regular_market_vshare"),
    ("fpo", "villages with a producer organisation", "fpo_vshare"),
    ("cold", "villages with a food storage warehouse", "storage_vshare"),
    ("sc", "Scheduled Caste population share", "pca_sc_share"),
    ("st", "Scheduled Tribe population share", "pca_st_share"),
    ("ci", "cropping intensity", "cropping_intensity"),
]
dref = pd.read_csv(COV + "/final_dimension_results.csv")
dims = []
for i, (kk, lab, v) in enumerate(SPEC):
    base = v.replace("np.log(", "").replace(")", "")
    s2 = d.dropna(subset=[base, "D1_exp_shannon", "shrug_key", "mean_holding_ha"])
    m0 = smf.ols("D1_exp_shannon ~ " + v, data=s2).fit(**CL(s2))
    f1 = ("D1_exp_shannon ~ " + v +
          (" + irr_share + I(irr_share**2)" if v != "irr_share" else " + I(irr_share**2)") +
          ("" if "mean_holding" in v else " + np.log(mean_holding_ha)") + " + C(state_name)")
    m1 = smf.ols(f1, data=s2).fit(**CL(s2))
    k0 = m0.params.index[1]
    assert abs(m0.params[k0] - dref.loc[i, "raw_b"]) < 1e-9 and abs(m1.params[v] - dref.loc[i, "adj_b"]) < 1e-9
    xv = np.log(s2[base]) if "np.log" in v else s2[base]
    dims.append(dict(key=kk, lab=lab, n=int(m1.nobs), sd=r(xv.std(ddof=1), 4),
                     raw=dict(b=r(m0.params[k0], 5), se=r(m0.bse[k0], 5), p=float("%.4g" % m0.pvalues[k0])),
                     adj=dict(b=r(m1.params[v], 5), se=r(m1.bse[v], 5), p=float("%.4g" % m1.pvalues[v]))))
for x in dims:
    print("dim", x["key"], x["sd"], round(x["raw"]["b"] * x["sd"], 3), round(x["adj"]["b"] * x["sd"], 3), x["adj"]["p"])

# ---------------------------------------------------------------- state outlines and state means
gg = g.copy()
gg["geometry"] = gg["geometry"].buffer(0)
st = gg.dissolve(by="s").reset_index()
st["geometry"] = st["geometry"].simplify(0.02, preserve_topology=True)
rings = []
for _, row in st.iterrows():
    geom = row["geometry"]
    for pg in ([geom] if geom.geom_type == "Polygon" else list(geom.geoms)):
        if pg.area >= 0.01:
            rings.append([[round(a, 3), round(b, 3)] for a, b in pg.exterior.coords])
smeans = d.groupby("state_name")["irr_share"].agg(["mean", "size"]).round(3)
print(smeans.sort_values("mean").to_string())

out = dict(
    note="Irrigation page scene data, 606 districts. x = irrigated share of net sown area "
         "(Mission Antyodaya 2019-20); D0 count of crops, D1 effective number, D2 dominant-"
         "weighted, E evenness, annual means 1997-98 to 2019-20. g = irrigation group "
         "(0 under 40%, 1 40-60%, 2 over 60%). k joins districts.geojson.",
    sample=acct, means={kk: r(d[c].mean(), 3) for kk, c in Y.items()},
    curve=curves, districts=dist, geo_by_name=geo_by_name, profile=prof, source_mix=mix, regimes=reg,
    robustness=rob, drop3=drop3, source=src, dims=dims,
    states={s.title(): dict(irr=float(v["mean"]), n=int(v["size"])) for s, v in smeans.iterrows()},
    state_rings=rings)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float("%.6g" % o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


with open(OUT, "w", encoding="utf-8") as f:
    json.dump(clean(out), f, separators=(",", ":"))
print("WROTE", OUT)
