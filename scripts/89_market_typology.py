"""
89_market_typology.py

State comparison and a district typology for the market page.

Uses the same 606-district analysis set and the same ten facility village shares
as site_markets.json (final_panel in_final, with market covariates present), and
the corrected diversity indices behind site_indices.json.

  1. State comparison. For every state with at least 8 districts: mean and spread
     of D0 (count of crops) and D1 (effective number of crops), and the mean
     village share of each of the ten facilities. For D0, D1, evenness and each
     facility, the share of district variance that lies between states rather
     than within them (eta squared, plus a small-group adjusted version).

  2. Typology. k-means on standardised facility shares, k = 3 to 6, checked
     across seeds, bootstrap resamples, three scalings (z, square root, rank) and
     with and without the weekly haat. Then a rule-based alternative on the two
     axes the data actually has: overall facility density (mean percentile rank
     of the nine facilities other than the haat, in terciles) crossed with weekly
     haat share (above or below the median). The rule-based types are what get
     exported; the k-means diagnostics are written to the log so the choice is
     auditable.

Outputs:
  docs/data/site_typology.json
  outputs/shrug_covariates/market_typology_log.txt
"""
import json
import os
import re
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "3")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari, silhouette_score

REPO = r"D:/crop-diversity"
COV = REPO + "/outputs/shrug_covariates"
DIV = REPO + "/outputs/crop_diversity_analysis"
OUT_JSON = REPO + "/docs/data/site_typology.json"
OUT_LOG = os.environ.get("CROPDIV_TYPOLOGY_LOG", COV + "/market_typology_log.txt")
# facility shares: village shares by default; 112_market_pop_site_data.py points this at
# the population-weighted shares the site uses
MK_CSV = os.environ.get("CROPDIV_MARKET_COV", COV + "/market_covariates.csv")
GEO = REPO + "/docs/data/districts.geojson"
MIN_STATE = 8
LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


def r6(v):
    if v is None:
        return None
    v = float(v)
    return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)


# ------------------------------------------------------------- the map key
# identical to 85_rebuild_map_layers.py, so the ids match districts.geojson "k"
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
    a = ALIAS.get(norm(state), norm(state))
    return a + "|" + norm(dist)


# ------------------------------------------------------------------ data
fp = pd.read_csv(COV + "/final_panel.csv")
mk = pd.read_csv(MK_CSV)
for t in (fp, mk):
    for c in ["pc11_state_id", "pc11_district_id"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
n0 = len(fp)
d = fp.merge(mk, on=["pc11_state_id", "pc11_district_id"], how="left")
assert len(d) == n0, "merge multiplied rows"
d = d[d["in_final"] & d["idx_output_market"].notna()].copy().reset_index(drop=True)
assert len(d) == 606, "analysis set is not the 606 districts behind site_markets.json"

# diversity from the corrected indices (the source of site_indices.json)
corr = pd.read_csv(DIV + "/district_diversity_indices_corrected.csv")
DV = ["D0_richness", "D1_exp_shannon", "evenness_D1_D0", "share_cereals", "share_pulses"]
d = d.drop(columns=DV).merge(corr[["district_key"] + DV], on="district_key", how="left")
assert d["D1_exp_shannon"].notna().all()

d["k"] = [key(*v.split("|", 1)) for v in d["div_key"]]
geo_k = {f["properties"].get("k") for f in json.load(open(GEO, encoding="utf-8"))["features"]}
on_map = d["k"].isin(geo_k)
w("Analysis set: {} districts, {} states; {} have a polygon in districts.geojson.".format(
    len(d), d["state_name"].nunique(), int(on_map.sum())))
assert d["k"].is_unique

FAC = [("mandi", "m_mandi_vshare"), ("regular market", "m_regular_market_vshare"),
       ("weekly haat", "m_weekly_haat_vshare"), ("fertiliser shop", "m_fert_shop_vshare"),
       ("seed centre", "m_seed_centre_vshare"), ("soil testing", "m_soil_test_vshare"),
       ("custom hiring", "m_custom_hire_vshare"), ("cold storage", "m_storage_vshare"),
       ("farm-gate processing", "m_farmgate_proc_vshare"), ("FPO", "m_fpo_vshare")]
FL = [a for a, _ in FAC]
FC = [b for _, b in FAC]
NH = [c for c in FC if "haat" not in c]
SHORT = {c: c.replace("m_", "").replace("_vshare", "") for c in FC}


def state_label(s):
    return " ".join(x if x in ("and",) else x.capitalize() for x in s.lower().split())


# ================================================== 1. state comparison
w("")
w("## 1. States")
big = d.groupby("state_name").size()
big = big[big >= MIN_STATE].index
sb = d[d["state_name"].isin(big)].copy()
w("States with >= {} districts: {} states, {} districts.".format(MIN_STATE, len(big), len(sb)))


def decomp(df, col):
    y = df[col].astype(float)
    g = df["state_name"]
    n, G = len(y), g.nunique()
    sst = ((y - y.mean()) ** 2).sum()
    ssb = (df.groupby("state_name")[col].transform("mean") - y.mean()).pow(2).sum()
    ssw = sst - ssb
    eta = ssb / sst
    adj = 1 - (ssw / (n - G)) / (sst / (n - 1))
    return eta, adj


VARS = [("D0", "D0_richness"), ("D1", "D1_exp_shannon"), ("evenness", "evenness_D1_D0")] + \
       [(a, b) for a, b in FAC]
vdec = []
w("")
w("| variable | between-state share (states >= 8) | adjusted | all 30 states |")
w("|---|---|---|---|")
for lab, c in VARS:
    e, a = decomp(sb, c)
    e_all, _ = decomp(d, c)
    vdec.append({"var": lab, "between": r6(e), "between_adj": r6(a), "within": r6(1 - e),
                 "between_all_states": r6(e_all)})
    w("| {} | {:.3f} | {:.3f} | {:.3f} |".format(lab, e, a, e_all))

states = []
for s, g in sb.groupby("state_name"):
    rec = {"state": state_label(s), "n": int(len(g))}
    for lab, c in [("D0", "D0_richness"), ("D1", "D1_exp_shannon")]:
        v = g[c]
        rec[lab] = {"mean": r6(v.mean()), "sd": r6(v.std()), "p10": r6(v.quantile(.1)),
                    "p90": r6(v.quantile(.9)), "min": r6(v.min()), "max": r6(v.max())}
    rec["evenness"] = r6(g["evenness_D1_D0"].mean())
    rec["fac"] = {lab: r6(g[c].mean()) for lab, c in FAC}
    states.append(rec)
states.sort(key=lambda r: -r["D1"]["mean"])
w("")
w("| state | n | D0 mean | D0 sd | D1 mean | D1 sd | haat | fert shop | FPO | mandi |")
w("|---|---|---|---|---|---|---|---|---|---|")
for r in states:
    w("| {} | {} | {:.1f} | {:.1f} | {:.2f} | {:.2f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
        r["state"], r["n"], r["D0"]["mean"], r["D0"]["sd"], r["D1"]["mean"], r["D1"]["sd"],
        r["fac"]["weekly haat"], r["fac"]["fertiliser shop"], r["fac"]["FPO"], r["fac"]["mandi"]))


# ================================================== 2a. k-means diagnostics
w("")
w("## 2a. k-means on facility shares")


def zs(M):
    return (M - M.mean(0)) / M.std(0)


SCALE = {
    "z": zs(d[FC].values),
    "sqrt": zs(np.sqrt(d[FC].values)),
    "rank": zs(d[FC].rank(pct=True).values),
    "z, no haat": zs(d[NH].values),
}


def fit(M, k, seed=0, n_init=50):
    return KMeans(k, n_init=n_init, random_state=seed).fit(M)


ref = {}
km_diag = []
w("")
w("| scaling | k | silhouette | ARI across 20 seeds | bootstrap ARI mean | p10 | sizes |")
w("|---|---|---|---|---|---|---|")
for nm, M in SCALE.items():
    for k in range(3, 7):
        base = fit(M, k)
        lab = base.labels_
        ref[(nm, k)] = lab
        seeds = [ari(lab, fit(M, k, s, 10).labels_) for s in range(1, 21)]
        rng = np.random.default_rng(k)
        bs = []
        for b in range(40):
            i = rng.choice(len(M), len(M))
            bs.append(ari(lab, fit(M[i], k, b, 10).predict(M)))
        sizes = sorted(np.bincount(lab).tolist(), reverse=True)
        km_diag.append({"scaling": nm, "k": k, "silhouette": r6(silhouette_score(M, lab)),
                        "ari_seeds": r6(np.mean(seeds)), "ari_boot": r6(np.mean(bs)),
                        "ari_boot_p10": r6(np.percentile(bs, 10)), "sizes": sizes})
        w("| {} | {} | {:.3f} | {:.2f} | {:.2f} | {:.2f} | {} |".format(
            nm, k, km_diag[-1]["silhouette"], np.mean(seeds), np.mean(bs),
            np.percentile(bs, 10), sizes))

w("")
w("Agreement between scalings at k = 3 (ARI):")
cross = {}
for a in SCALE:
    for b in SCALE:
        if a < b:
            v = ari(ref[(a, 3)], ref[(b, 3)])
            cross[a + " vs " + b] = r6(v)
            w("  {:22s} {:.2f}".format(a + " vs " + b, v))

# the one k-means group that recurs: the densest cluster under z scaling
lab = ref[("z", 3)]
dense_c = int(pd.Series(d["m_fert_shop_vshare"]).groupby(lab).mean().idxmax())
dense = set(np.where(lab == dense_c)[0])
w("")
w("Densest z, k=3 cluster: {} districts, states {}".format(
    len(dense), d.loc[sorted(dense), "state_name"].value_counts().to_dict()))
recur = []
for (nm, k), L in ref.items():
    best = max(len(dense & set(np.where(L == c)[0])) / len(dense | set(np.where(L == c)[0]))
               for c in np.unique(L))
    recur.append(best)
    w("  best Jaccard match in {:12s} k={}: {:.2f}".format(nm, k, best))
d["km_dense"] = d.index.isin(dense)


# ================================================== 2b. rule-based types
w("")
w("## 2b. Rule-based types: facility density x weekly haat")
d["density"] = d[NH].rank(pct=True).mean(axis=1)
d["density_z"] = pd.Series(zs(d[NH].values).mean(axis=1))
w("corr(density, haat share) = {:.3f}".format(np.corrcoef(d["density"], d["m_weekly_haat_vshare"])[0, 1]))
cut = d["density"].quantile([1 / 3, 2 / 3]).values
hmed = d["m_weekly_haat_vshare"].median()
w("density tercile cuts {:.3f}, {:.3f}; haat median {:.3f}".format(cut[0], cut[1], hmed))


def assign(dens, haat, c1, c2, hm):
    band = np.where(dens <= c1, 0, np.where(dens <= c2, 1, 2))
    hi = (haat > hm).astype(int)
    return band * 2 + hi


TYPES = [
    ("few", "Few facilities", "bottom third on shops, markets and FPOs, and few haats"),
    ("haat_only", "Weekly haats, little else", "bottom third on shops, markets and FPOs, more haats than the median"),
    ("mid", "Some shops and FPOs, few haats", "middle third on shops, markets and FPOs, few haats"),
    ("mid_haat", "Some shops and FPOs, with haats", "middle third on shops, markets and FPOs, more haats than the median"),
    ("served", "Shops, FPOs and markets, few haats", "top third on shops, markets and FPOs, few haats"),
    ("served_haat", "Shops, FPOs and markets, with haats", "top third on shops, markets and FPOs, more haats than the median"),
]
d["type_i"] = assign(d["density"].values, d["m_weekly_haat_vshare"].values, cut[0], cut[1], hmed)
d["type"] = [TYPES[i][0] for i in d["type_i"]]

# stability of the rule
rng = np.random.default_rng(7)
moved = []
for b in range(500):
    s = d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))
    c = s["density"].quantile([1 / 3, 2 / 3]).values
    t = assign(d["density"].values, d["m_weekly_haat_vshare"].values, c[0], c[1],
               s["m_weekly_haat_vshare"].median())
    moved.append((t != d["type_i"].values).mean())
zc = d["density_z"].quantile([1 / 3, 2 / 3]).values
t_z = assign(d["density_z"].values, d["m_weekly_haat_vshare"].values, zc[0], zc[1], hmed)
band_rank = d["type_i"] // 2
band_z = t_z // 2
stab = {
    "boot_share_moved_mean": r6(np.mean(moved)),
    "boot_share_moved_p90": r6(np.percentile(moved, 90)),
    "same_band_if_density_is_mean_z": r6((band_rank == band_z).mean()),
    "same_type_if_density_is_mean_z": r6((d["type_i"].values == t_z).mean()),
}
w("bootstrap the cut points (500): mean {:.1%} of districts change type, 90th pct {:.1%}".format(
    np.mean(moved), np.percentile(moved, 90)))
w("density as mean z instead of mean rank: {:.1%} same band, {:.1%} same type".format(
    stab["same_band_if_density_is_mean_z"], stab["same_type_if_density_is_mean_z"]))
w("rule band vs k-means (rank, k=3) ARI {:.2f}; vs k-means (z, k=3) ARI {:.2f}".format(
    ari(band_rank, ref[("rank", 3)]), ari(band_rank, ref[("z", 3)])))
stab["ari_band_vs_kmeans_rank3"] = r6(ari(band_rank, ref[("rank", 3)]))
stab["ari_band_vs_kmeans_z3"] = r6(ari(band_rank, ref[("z", 3)]))

# share of variance the types explain, against what state explains
def eta(df, col, grp):
    y = df[col]
    return ((df.groupby(grp)[col].transform("mean") - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum()


expl = {}
for lab, c in [("D0", "D0_richness"), ("D1", "D1_exp_shannon"), ("evenness", "evenness_D1_D0")]:
    expl[lab] = {"type": r6(eta(d, c, "type")), "state": r6(eta(d, c, "state_name"))}
    w("{} variance explained: type {:.3f}, state {:.3f}".format(lab, expl[lab]["type"], expl[lab]["state"]))
# state composition of the types
type_state_eta = None
dd = pd.get_dummies(d["type"]).astype(float)
cv = [eta(d.assign(_y=dd[c]), "_y", "state_name") for c in dd.columns]
w("mean between-state share of type membership: {:.3f}".format(np.mean(cv)))

PROF = [("D0", "D0_richness"), ("D1", "D1_exp_shannon"), ("evenness", "evenness_D1_D0"),
        ("cereal_share", "share_cereals"), ("pulse_share", "share_pulses"),
        ("irrigation_share", "irr_share")]
types = []
w("")
w("| type | n | top states | D0 | D1 | evenness | cereal | pulse | irrigation | nightlights mean | median |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
for code, name, rule in TYPES:
    g = d[d["type"] == code]
    vc = g["state_name"].value_counts()
    top = [{"state": state_label(s), "n": int(n), "share_of_type": r6(n / len(g)),
            "share_of_state": r6(n / (d["state_name"] == s).sum())} for s, n in vc.head(5).items()]
    rec = {"code": code, "name": name, "rule": rule, "n": int(len(g)),
           "top_states": top,
           "nightlights_mean": r6(g["viirs_mean"].mean()),
           "nightlights_median": r6(g["viirs_mean"].median()),
           "fac": {lab: r6(g[c].mean()) for lab, c in FAC}}
    for lab, c in PROF:
        rec[lab] = r6(g[c].mean())
    types.append(rec)
    w("| {} | {} | {} | {:.1f} | {:.2f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.2f} |".format(
        name, len(g), ", ".join("{} {}".format(state_label(s), n) for s, n in vc.head(3).items()),
        rec["D0"], rec["D1"], rec["evenness"], rec["cereal_share"], rec["pulse_share"],
        rec["irrigation_share"], rec["nightlights_mean"], rec["nightlights_median"]))
w("")
w("Facility means by type:")
w(pd.DataFrame({t["name"]: t["fac"] for t in types}).T.round(3).to_string())

# the k-means dense group, kept as a flag only
kd = d[d["km_dense"]]
dense_rec = {"n": int(len(kd)),
             "states": {state_label(s): int(n) for s, n in kd["state_name"].value_counts().items()},
             "D0": r6(kd["D0_richness"].mean()), "D1": r6(kd["D1_exp_shannon"].mean()),
             "fac": {lab: r6(kd[c].mean()) for lab, c in FAC},
             "best_jaccard_across_solutions": {"min": r6(min(recur)), "median": r6(np.median(recur))}}
w("")
w("k-means dense group: {}".format(dense_rec))

# per-state type mix
for rec in states:
    s = [k for k in big if state_label(k) == rec["state"]][0]
    g = d[d["state_name"] == s]["type"].value_counts()
    rec["types"] = {code: int(g.get(code, 0)) for code, _, _ in TYPES}

# ================================================== 3. export
dist = {r["k"]: {"t": r["type"], "dens": r6(r["density"]), "haat": r6(r["m_weekly_haat_vshare"]),
                 "kd": bool(r["km_dense"])}
        for _, r in d.iterrows()}
out = {
    "note": ("Market typology on the 606-district analysis set. 'density' is the mean "
             "percentile rank of nine facility village shares (all but the weekly haat); "
             "types cross density terciles with weekly haat share above or below the median. "
             "District keys match the 'k' property in districts.geojson. 'kd' flags the "
             "dense-facility group that k-means finds under z scaling (mostly Gujarat and Kerala)."),
    "n_districts": int(len(d)),
    "n_on_map": int(on_map.sum()),
    "cuts": {"density_tercile_1": r6(cut[0]), "density_tercile_2": r6(cut[1]),
             "haat_median": r6(hmed)},
    "facilities": FL,
    "types": types,
    "type_stability": stab,
    "variance_explained": expl,
    "kmeans": {"diagnostics": km_diag, "agreement_k3": cross, "dense_group": dense_rec},
    "variance_split": {"min_districts": MIN_STATE, "n_states": int(len(big)),
                       "n_districts": int(len(sb)), "vars": vdec},
    "states": states,
    "districts": dist,
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"))
with open(OUT_LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
print("\nWROTE {} ({:,} bytes)".format(OUT_JSON, os.path.getsize(OUT_JSON)))
