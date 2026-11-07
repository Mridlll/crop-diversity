"""
141_pe_v2_site_data.py

Adds the November reviewer-round results (scripts 138, 139, 140) to the Neighbours page
data, docs/data/site_neighbours.json, under three keys. Every other key is left as
script 131 wrote it, so running this after 131 is safe and running it twice changes
nothing.

  leadlag        138: neighbours and the same-state ring at k = -3..3 (k > 0 = years
                 before), spec "no_own" (the figure) and the sum-of-lags-minus-leads
                 tests for every spec
  within_cross   140: same-state against other-state neighbours in take-up
  common_border  139: border gap on the eight crops every state records

Run after 131, 138, 139, 140:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/141_pe_v2_site_data.py
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(ROOT, "outputs", "revision_nov8", "peer_effects", "v2")
SITE = os.path.join(ROOT, "docs", "data", "site_neighbours.json")
r = lambda v, d=4: None if v is None or not np.isfinite(v) else round(float(v), d)

D = json.load(open(SITE, encoding="utf-8"))

# ---------------------------------------------------------------- leads and lags
L = pd.read_csv(os.path.join(V2, "138_leads_lags.csv"))
T = pd.read_csv(os.path.join(V2, "138_leads_lags_tests.csv"))
ll = {"spec": "no_own", "outcomes": {}}
for o in ["chickpea", "soybean", "cotton", "pulses"]:
    d = L[(L.outcome == o) & (L.spec == "no_own")]
    rec = {"n": int(d.n.iloc[0])}
    for s, key in [("neighbours", "nb"), ("ring", "rg")]:
        q = d[d.set == s].sort_values("k")
        rec[key] = [[int(k), r(b), r(se)] for k, b, se in zip(q.k, q.b, q.se)]
    tt = {}
    for spec in ["no_own", "main", "no_k0"]:
        x = T[(T.outcome == o) & (T.spec == spec) & (T.set == "neighbours")
              & (T.test == "sum of lags 1-3 minus sum of leads 1-3")].iloc[0]
        tt[spec] = {"b": r(x.b, 3), "se": r(x.se, 3), "p": r(x.p, 3)}
    rec["lag_minus_lead"] = tt
    ll["outcomes"][o] = rec
D["leadlag"] = ll

# ---------------------------------------------------------------- within / across state
C = pd.read_csv(os.path.join(V2, "140_within_cross_hazard.csv"))
TT = pd.read_csv(os.path.join(V2, "140_within_cross_tests.csv"))
NF = pd.read_csv(os.path.join(V2, "140_nfsm_chickpea.csv"))
wc = []
for ep, mode in [("chickpea", "surge"), ("soybean", "entry"), ("soybean", "surge"), ("cotton", "surge"),
                 ("chickpea", "entry")]:
    g = lambda spec, var, cl="unit": C[(C.episode == ep) & (C["mode"] == mode) & C.spec.str.startswith(spec)
                                        & (C["var"] == var) & (C.cluster == cl)].iloc[0]
    t = lambda spec, test, cl: TT[(TT.episode == ep) & (TT["mode"] == mode) & TT.spec.str.startswith(spec)
                                  & (TT.test == test) & (TT.cluster == cl)].iloc[0]
    a, b = g("S2", "nbr_same"), g("S2", "nbr_other")
    d2u, d2s = t("S2", "nbr_same minus nbr_other", "unit"), t("S2", "nbr_same minus nbr_other", "state")
    e3u, e3s = t("S3", "edge within minus edge across", "unit"), t("S3", "edge within minus edge across", "state")
    wc.append({"ep": ep, "mode": mode, "ev": int(a.n_events),
               "same": r(a.b), "same_se": r(a.se), "other": r(b.b), "other_se": r(b.se),
               "diff": r(d2u.b), "diff_se": r(d2u.se), "diff_p": r(d2u.p, 3), "diff_p_state": r(d2s.p, 3),
               "edge_diff": r(e3u.b), "edge_diff_se": r(e3u.se), "edge_diff_p": r(e3u.p, 3),
               "edge_diff_p_state": r(e3s.p, 3), "edge_ev": int(e3u.n_events)})
nf = NF[(NF["mode"] == "surge") & (NF.cluster == "unit") & (NF["var"] == "nbr minus ring")]
D["within_cross"] = {"rows": wc, "nfsm_chickpea_surge": {
    "base": r(nf[nf.spec == "base: nbr + ring"].b.iloc[0]),
    "with_nfsm": r(nf[nf.spec == "+ NFSM own, nbr share, ring share"].b.iloc[0]),
    "with_nfsm_se": r(nf[nf.spec == "+ NFSM own, nbr share, ring share"].se.iloc[0])}}

# ---------------------------------------------------------------- common crops border
G = pd.read_csv(os.path.join(V2, "139_border_gap.csv"))
K = pd.read_csv(os.path.join(V2, "139_border_km.csv"))
cb = {"crops": ["rice", "wheat", "maize", "chickpea", "pigeon pea", "groundnut", "rapeseed and mustard", "cotton"]}
for meas in ["sim8", "sim9"]:
    x = {}
    for spec in ["M1", "M2", "M3"]:
        q = G[(G.window == "w1519") & (G.measure == meas) & (G.spec == spec)].iloc[0]
        x[spec] = {"b": r(q.b, 2), "se": r(q.se_unit, 2), "se_state": r(q.se_state, 2), "p_wild": r(q.p_wild_state, 3)}
    k = K[K.measure == meas].iloc[0]
    x["km"] = {"interp": r(k.equiv_interp, 0), "lo": r(k.equiv_interp_lo, 0), "hi": r(k.equiv_interp_hi, 0),
               "linear": r(k.equiv_linear, 0), "reg": r(k.equiv_reg, 0)}
    x["adj_same"], x["adj_cross"] = r(k.same_adjacent_mean, 1), r(k.cross_adjacent_mean, 1)
    cb[meas] = x
D["common_border"] = cb

D.setdefault("sources", {})
D["sources"]["leadlag"] = "outputs/revision_nov8/peer_effects/v2/138_leads_lags.csv, 138_leads_lags_tests.csv"
D["sources"]["within_cross"] = "outputs/revision_nov8/peer_effects/v2/140_within_cross_*.csv, 140_nfsm_chickpea.csv"
D["sources"]["common_border"] = "outputs/revision_nov8/peer_effects/v2/139_border_gap.csv, 139_border_km.csv"
with open(SITE, "w", encoding="utf-8") as f:
    json.dump(D, f, separators=(",", ":"), ensure_ascii=False)
print("written", SITE, {k: (len(v) if hasattr(v, "__len__") else v) for k, v in
                        [("leadlag", D["leadlag"]["outcomes"]), ("within_cross", D["within_cross"]["rows"])]})
print(json.dumps(D["within_cross"], indent=0)[:1500])
print(json.dumps(D["common_border"]))
print(json.dumps({o: v["lag_minus_lead"] for o, v in D["leadlag"]["outcomes"].items()}))
