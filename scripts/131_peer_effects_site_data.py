"""
131_peer_effects_site_data.py

Builds docs/data/site_neighbours.json for docs/neighbours.html.

Reads only finished outputs; estimates nothing:
  - outputs/revision_nov8/peer_effects/chart_data/v1..v8 (scripts 125-129)
  - outputs/crop_groups/neighbours_estimates.csv (script 99, year-on-year co-movement
    and the same-state distance placebo), used for the opening comparison only
  - docs/data/site_market_map.json for the state outlines (same as the crop-groups page)
  - outputs/revision_nov8/peer_effects/referee/ (script 132, referee checks): the numbers the page
    states after the checks (border-km interval, placebo borders, fragility of the geography model,
    crop-group shares of the gap, widening interval, Moran's I without AP-TS links, suitability-matched
    wave benchmarks, alternative rings, Telangana timing)
Map polygons are the site's docs/data/districts.geojson, joined through
chart_data/v4_polygon_to_unit.csv (geojson property k -> 2011 unit key).
No APCNF, PMDS or RySS data.

Usage:  python scripts/131_peer_effects_site_data.py [--out docs/data/site_neighbours.json]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

REPO = "D:/crop-diversity"
PE = REPO + "/outputs/revision_nov8/peer_effects"
CD = PE + "/chart_data"
G = ["rice", "wheat", "maize", "millets", "pulses", "oilseeds", "cotton", "sugarcane", "other"]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=REPO + "/docs/data/site_neighbours.json")
args = ap.parse_args()


def r(v, d=2):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, d) if np.isfinite(f) else None


TITLE = lambda s: " ".join(w.capitalize() if w not in ("and",) else w for w in str(s).lower().split())
J = {"groups": G, "sources": {}}

# ------------------------------------------------------------------ 1. similarity vs distance
v1 = json.load(open(CD + "/v1_similarity_vs_distance.json", encoding="utf-8"))
J["sim"] = [dict(w=c["window"], t=c["pair_type"], km=c["km"], n=c["n_pairs"], m=c["sim_mean"],
                 p25=c["sim_p25"], p75=c["sim_p75"]) for c in v1["curve"]]
ref = {}
for p in v1["reference_points"]:
    w = "2015-19" if p["window"] == "w1519" else "1998-2002"
    ref.setdefault(w, {})[p["pairs"]] = dict(n=p["n_pairs"], km=r(p["mean_km"], 1), sim=r(100 * p["sim_mean"], 1),
                                             dD1=r(p["dD1_mean"], 2))
J["sim_ref"] = ref
# distance at which the same-state curve falls to the adjacent cross-border mean (linear interpolation)
cur = pd.DataFrame(v1["curve"])
bk = {}
for w in ["2015-19", "1998-2002"]:
    target = ref[w]["adjacent, across a border"]["sim"]
    ss = cur[(cur.window == w) & (cur.pair_type == "same_state")].sort_values("km")
    km_cross = None
    for (k0, s0), (k1, s1) in zip(ss[["km", "sim_mean"]].values[:-1], ss[["km", "sim_mean"]].values[1:]):
        if s0 >= target >= s1:
            km_cross = k0 + (s0 - target) / (s0 - s1) * (k1 - k0)
            break
    km_adj = ref[w]["adjacent, same state"]["km"]
    bk[w] = dict(target=target, km_cross=r(km_cross, 0), km_adj=km_adj,
                 border_km=r(km_cross - km_adj, 0) if km_cross else None)
J["border_km"] = bk
J["sources"]["sim"] = "chart_data/v1_similarity_vs_distance.json (curve, reference_points); border_km interpolated here"

# ------------------------------------------------------------------ year-on-year co-movement (earlier work)
ne = pd.read_csv(REPO + "/outputs/crop_groups/neighbours_estimates.csv")
yoy = {}
for g in ["pulses", "millets", "oilseeds"]:
    q = ne[ne.group == g]
    pick = lambda spec, var: q[(q.spec == spec) & (q["var"] == var)].iloc[0]
    a, b = pick("ref contemporaneous W dS_t", "WD_t"), pick("M2 main", "WD_l1")
    hn, hp, hd = (pick("P2 lag+placebo", "WD_l1"), pick("P2 lag+placebo", "PD_l1"),
                  pick("P2 lag+placebo", "WD_l1 minus PD_l1"))
    yoy[g] = dict(same_year=r(a.b, 2), same_year_se=r(a.se, 2), lag=r(b.b, 3), lag_se=r(b.se, 3), lag_p=r(b.p, 3),
                  hr_nbr=r(hn.b, 3), hr_nbr_se=r(hn.se, 3), hr_ring=r(hp.b, 3), hr_ring_se=r(hp.se, 3),
                  hr_diff=r(hd.b, 3), hr_diff_se=r(hd.se, 3), hr_diff_p=r(hd.p, 2))
J["yoy"] = yoy
J["sources"]["yoy"] = "outputs/crop_groups/neighbours_estimates.csv (ref contemporaneous; M2 main; P2 lag+placebo)"

# ------------------------------------------------------------------ 2. border dumbbells
v2 = pd.read_csv(CD + "/v2_border_dumbbells.csv")
big = v2[(v2.window == "2015-19") & (v2.few_pairs == 0)].groupby("border").n_pairs.max()
order = big.sort_values(ascending=False).index.tolist()
rows = []
for b in order:
    for v in ["rice", "wheat", "pulses", "millets"]:
        for w in ["1998-2002", "2015-19"]:
            x = v2[(v2.border == b) & (v2["var"] == v) & (v2.window == w)]
            if not len(x):
                continue
            x = x.iloc[0]
            rows.append(dict(border=b, a=TITLE(x.state_a), b_=TITLE(x.state_b), v=v, w=w, n=int(x.n_pairs),
                             na=int(x.n_units_a), nb=int(x.n_units_b), ma=r(x.mean_a, 1), mb=r(x.mean_b, 1),
                             gap=r(x.gap_pairFE, 1), se=r(x.se_unit, 1), geo=r(x.geo_predicted_gap, 1)))
J["borders"] = dict(order=order, rows=rows)
J["sources"]["borders"] = "chart_data/v2_border_dumbbells.csv (borders with 8+ adjacent pairs in 2015-19)"

# ------------------------------------------------------------------ 3. border gap over time
v3 = pd.read_csv(CD + "/v3_border_gap_over_time.csv")
# Telangana | Karnataka cotton (7 pairs, 3 Telangana districts) dropped after the referee checks (script 132):
# its gap cannot be tested on so few districts. Punjab | Haryana rice (12 pairs) shows a border gap that held.
CHOSEN = [("Telangana | Andhra Pradesh", "cotton"), ("Punjab | Haryana", "rice"),
          ("Telangana | Andhra Pradesh", "rice"), ("Chhattisgarh | Odisha", "rice"),
          ("Jharkhand | Odisha", "rice"), ("Haryana | Rajasthan", "wheat")]
gs = []
for s, v in CHOSEN:
    x = v3[(v3.series == s) & (v3["var"] == v)].sort_values("year")
    nmax = int(x.n_pairs.max())
    a, b = s.split(" | ")
    gs.append(dict(series=s, a=a, b=b, v=v, nmax=nmax,
                   pts=[[int(y), r(g, 1), int(n)] for y, g, n in zip(x.year, x.gap, x.n_pairs)]))
ctx = v3[v3.kind == "adjacent pairs, mean"]
J["gap_time"] = dict(series=gs, context={k: [[int(y), r(s, 1)] for y, s in zip(d.year, d.sim)]
                                         for k, d in ctx[ctx.series.isin(["same state", "cross-border"])].groupby("series")})
J["sources"]["gap_time"] = "chart_data/v3_border_gap_over_time.csv"

# ------------------------------------------------------------------ border coefficients (prose and method)
xs = pd.read_csv(PE + "/q2_crosssection.csv").drop_duplicates()
xsec = []
for _, x in xs[(xs.outcome.isin(["sim", "g_rice", "g_wheat", "g_pulses", "g_millets", "dD1"]))].iterrows():
    xsec.append(dict(w="2015-19" if x.window == "w1519" else "1998-2002", o=x.outcome, spec=x.spec.split(" ")[0],
                     cl=x.cluster, b=r(x.b_cross, 2), se=r(x.se, 2), p=r(x.p, 3), n=int(x.n_pairs),
                     mean_same=r(x.mean_same_state, 1)))
J["xsec"] = xsec
J["sources"]["xsec"] = "outputs/revision_nov8/peer_effects/q2_crosssection.csv"
pw = pd.read_csv(PE + "/q2_panel_widening.csv").drop_duplicates()
J["widening"] = [dict(o=x.outcome, spec=x.spec, term=x.term, cl=x.cluster, b=r(x.b, 2), se=r(x.se, 2), p=r(x.p, 3), n=int(x.n))
                 for _, x in pw[pw.outcome == "sim"].iterrows()]
J["sources"]["widening"] = "outputs/revision_nov8/peer_effects/q2_panel_widening.csv"
# geography gaps between the two districts of an adjacent pair, by pair type (AP-TS pairs left out, as in script 127)
pp = pd.read_csv(PE + "/pe_pairs.csv.gz")
adj = pp[pp.adjacent == 1].copy()
apts = adj.state_i.isin(["ANDHRA PRADESH", "TELANGANA"]) & adj.state_j.isin(["ANDHRA PRADESH", "TELANGANA"]) & (adj.state_i != adj.state_j)
adj = adj[~apts]
geo = {}
for ss, d in adj.groupby("same_state"):
    geo["same" if ss == 1 else "cross"] = dict(n=int(len(d)), km=r(d.dist_km.mean(), 0), rain_mm=r(100 * d.d_rain.mean(), 0),
                                               elev_m=r(100 * d.d_elev.mean(), 0), tri=r(d.d_tri.mean(), 1),
                                               irr_pts=r(100 * d.d_irr.mean(), 1))
J["geo_gaps"] = geo
J["sources"]["geo_gaps"] = "outputs/revision_nov8/peer_effects/pe_pairs.csv.gz, adjacent pairs, means by same_state"

# ------------------------------------------------------------------ Moran's I (prose numbers)
v7 = pd.read_csv(CD + "/v7_moran.csv")
J["moran"] = [dict(m=x.measure, v=x["var"], l=x.links, I=r(x.I, 2), p=r(x.p_perm, 3), n=int(x.n)) for _, x in v7.iterrows()]
J["sources"]["moran"] = "chart_data/v7_moran.csv"

# ------------------------------------------------------------------ 4. spread maps
v4 = json.load(open(CD + "/v4_spread_maps.json", encoding="utf-8"))
yrs = v4["years"]
PANEL = [2001, 2005, 2009, 2013, 2017, 2019]
sp = {"panel_years": PANEL, "episodes": {}}
for ep in ["chickpea_surge", "soybean_entry"]:
    E = v4["episodes"][ep]
    units = {}
    for k, u in E["units"].items():
        units[k] = dict(n=u["name"], s=u["state"], st=u["status"], y=u["event_year"],
                        sh=[r(u["share_by_year"][yrs.index(y)], 1) for y in PANEL], b=r(u["base"], 1))
    sp["episodes"][ep] = dict(threshold=E["threshold_pts"], units=units)
J["spread"] = sp
pm = pd.read_csv(CD + "/v4_polygon_to_unit.csv")
J["poly"] = {x.k: x.unit_key for x in pm.itertuples() if isinstance(x.unit_key, str)}
mm = json.load(open(REPO + "/docs/data/site_market_map.json", encoding="utf-8"))
J["state_rings"] = mm["state_rings"]
J["sources"]["spread"] = "chart_data/v4_spread_maps.json, v4_polygon_to_unit.csv"

v8 = pd.read_csv(CD + "/v8_wave.csv")
J["wave"] = [dict(ep=x.episode, mode=x["mode"], n=int(x.events_with_prior_grower), adj=r(100 * x.share_contiguous, 0),
                  rnd_nat=r(100 * x.share_contiguous_random_national, 0),
                  rnd_state=r(100 * x.share_contiguous_random_within_state, 0), p_state=r(x.p_within_state, 3),
                  km=r(x.median_km_nearest_grower, 0), jump=r(100 * x.share_jumps_gt150km, 0)) for _, x in v8.iterrows()]
J["sources"]["wave"] = "chart_data/v8_wave.csv"

# ------------------------------------------------------------------ 5. neighbour vs placebo ring
v5 = json.load(open(CD + "/v5_neighbour_vs_placebo.json", encoding="utf-8"))
J["hazard"] = [dict(ep=x["episode"], mode=x["mode"], nb=x["nbr_b"], nbse=x["nbr_se"], rg=x["ring_b"], rgse=x["ring_se"],
                    d=x["diff_b"], dse=x["diff_se"], dp=x["diff_p"], q=x["diff_q_bh14"], ev=x["n_events"],
                    base=x["base_rate"], ss=x["nbr_same_state_b"], ssse=x["nbr_same_state_se"],
                    os=x["nbr_other_state_b"], osse=x["nbr_other_state_se"]) for x in v5["rows"]]
J["sources"]["hazard"] = "chart_data/v5_neighbour_vs_placebo.json"

# ------------------------------------------------------------------ 6. explorer
v6 = json.load(open(CD + "/v6_neighbour_explorer.json", encoding="utf-8"))
ex = {}
for k, d in v6["districts"].items():
    m = d["mix"].get("2015-19")
    if not m:
        continue
    ex[k] = dict(n=d["name"], s=d["state"], D1=m.get("D1"), mix=[r(m.get(g), 1) for g in G],
                 nb=[dict(k=x["unit_key"], km=x["km"], ss=x["same_state"], sim=x["sim_2015_19"]) for x in d["neighbours"]],
                 msim=d.get("mean_nbr_sim_2015_19"), pct=d.get("pct_rank_mean_nbr_sim"))
# neighbours that have no 2015-19 mix of their own are kept for name lookup only
names = {k: dict(n=d["name"], s=d["state"]) for k, d in v6["districts"].items() if k not in ex}
J["explorer"] = dict(districts=ex, names=names)
# districts with neighbours on both sides of a state line: how often is the least similar neighbour across it,
# against the share of their neighbours that lie across it
nboth = nleast = 0
exp_share = []
for d in ex.values():
    nb = [x for x in d["nb"] if x["sim"] is not None]
    if any(x["ss"] == 0 for x in nb) and any(x["ss"] == 1 for x in nb):
        nboth += 1
        nleast += int(min(nb, key=lambda x: x["sim"])["ss"] == 0)
        exp_share.append(np.mean([x["ss"] == 0 for x in nb]))
J["explorer"]["least_similar"] = dict(n_districts=nboth, n_least_cross=nleast, pct_least_cross=r(100 * nleast / nboth, 0),
                                      pct_nbrs_cross=r(100 * np.mean(exp_share), 0))
J["sources"]["explorer"] = "chart_data/v6_neighbour_explorer.json"

# ------------------------------------------------------------------ referee checks (script 132)
RF = PE + "/referee"
ck = {}
h1 = pd.read_csv(RF + "/h1_border_km_equivalent.csv").iloc[0]
ck["border_km"] = dict(interp=r(h1.equiv_interp, 0), lo=r(h1.equiv_interp_lo, 0), hi=r(h1.equiv_interp_hi, 0),
                       linear=r(h1.equiv_linear, 0), reg=r(h1.equiv_reg, 0), km_cross_bin=r(h1.bin_rule_km, 0))
pl = pd.read_csv(RF + "/h2_placebo_partitions.csv")
ck["placebo"] = dict(n=int(len(pl)), m1_mean=r(pl.M1.mean(), 2), m1_lo=r(pl.M1.quantile(.025), 2),
                     m1_hi=r(pl.M1.quantile(.975), 2), m1_min=r(pl.M1.min(), 2))
bm = pd.read_csv(RF + "/h2_border_by_measure.csv")
ck["by_measure"] = [dict(w=x.window, m=x.measure, spec=x.spec, b=r(x.b, 2), se=r(x.se_unit, 2), se_st=r(x.se_state, 2),
                         p_st=r(x.p_state, 3), p_wild=r(x.p_wild_state, 3)) for _, x in bm.iterrows()]
lo = pd.read_csv(RF + "/h2_leave_out.csv")
ck["leave_out"] = [dict(m=x.measure, drop=x["drop"], kind=x.kind, spec=x.spec, b=r(x.b, 2), se=r(x.se_unit, 2))
                   for _, x in lo.iterrows()]
g3 = pd.read_csv(RF + "/h3_group_border_gaps.csv")
g3 = g3[g3.window == "w1519"]
ck["groups"] = [dict(spec=x.spec, o=x.outcome, b=r(x.b, 2), se_st=r(x.se_state, 2), lo=r(x.lo95_state, 1), hi=r(x.hi95_state, 1),
                     share=r(x.share_of_sim_gap, 2), p_wild=r(x.p_wild_state, 3), q_wild=r(x.q_bh_wild_state, 2))
                for _, x in g3.iterrows()]
wd = pd.read_csv(RF + "/h4_widening.csv")
ck["widening"] = [dict(o=x.outcome, spec=x.spec, term=x.term, cl=x.cluster, b=r(x.b, 2), lo=r(x.lo95, 1), hi=r(x.hi95, 1))
                  for _, x in wd.iterrows() if x.spec == "periods" and x.term == "x_2013"]
tc = pd.read_csv(RF + "/h4_telangana_cotton_pairs.csv")
u = tc.drop_duplicates("ts_unit")
gain_pre = (u.ts_2010_13 - u.ts_9802).sum()
gain_post = (u.ts_2014_19 - u.ts_2010_13).sum()
ck["telangana_cotton"] = dict(share_gain_before_2014=r(gain_pre / (gain_pre + gain_post), 2),
                              ts_units_per_border=3, sign_flip_p_min=0.25)
mo = pd.read_csv(RF + "/h5_moran_checks.csv")
ck["moran"] = [dict(v=x["var"], l=x.links, I=r(x.I, 2), p=r(x.p_perm, 3), lo=r(x.I_lo, 2), hi=r(x.I_hi, 2))
               for _, x in mo.iterrows()]
wv = pd.read_csv(RF + "/h6_wave_benchmarks.csv")
ck["wave"] = [dict(ep=x.episode, mode=x["mode"], sub=x.subset, n=int(x.events), obs=r(100 * x.obs, 0),
                   rnd_state=r(100 * x.p_state, 0), matched=r(100 * x.p_covnn_state, 0), p_matched=r(x.pval_covnn_state, 3),
                   bin=r(100 * x.p_state_bin, 0), prop=r(100 * x.p_prop_state, 0))
              for _, x in wv.iterrows()]
ar = pd.read_csv(RF + "/h7_alternative_rings.csv")
ar = ar[ar["var"] == "nbr minus ring"]
ck["rings"] = [dict(ep=x.episode, mode=x["mode"], ring=x.ring, cl=x.cluster, b=r(x.b, 3), se=r(x.se, 3), p=r(x.p, 3))
               for _, x in ar.iterrows()]
# number of crops reported: adjacent cross-border pairs differ by this many more crops (D0), after geography (M2)
ck["crop_count_gap"] = 4.7
J["checks"] = ck
J["sources"]["checks"] = ("outputs/revision_nov8/peer_effects/referee/ (script 132): h1_border_km_equivalent, "
                          "h2_placebo_partitions, h2_border_by_measure, h2_leave_out, h3_group_border_gaps, h4_widening, "
                          "h4_telangana_cotton_pairs, h5_moran_checks, h6_wave_benchmarks, h7_alternative_rings; "
                          "crop_count_gap from referee_log.txt (neg_dD0, M2, 2015-19: -4.68)")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(J, f, separators=(",", ":"), ensure_ascii=False)
print("wrote", args.out, round(os.path.getsize(args.out) / 1024), "KB;", len(ex), "explorer districts,",
      len(J["poly"]), "polygons; border km", bk)
