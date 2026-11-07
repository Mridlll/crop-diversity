"""
140_pe_within_cross_hazard.py

Does the neighbour edge in take-up stop at the state line? Reviewer item: the chickpea
and soybean neighbour edge (script 128, "nbr minus ring") is carried by same-state
neighbours only, which would point to programmes run inside a state (NFSM-type cluster
demonstrations, state seed supply) rather than farmers learning from farmers, who
cross state lines.

Reuses script 128's definitions (executed read-only up to its main loop, as script 132
does): units, adjacency, distance, the at-risk / event rules, controls, state-by-year FE.

Exposure sets, all measured at t-1 as the share of the set that already grows the crop:
  nbr_same   adjacent units in the same present-day state
  nbr_other  adjacent units in another state
  ring_same  non-adjacent units at 0.75-1.5x the unit's mean neighbour distance, same state
  ring_other the same ring, other states

Specs (linear probability hazard, script 128 controls, state-by-year FE; SE clustered on
unit and on state):
  S1  full sample: nbr + ring                                   (reproduces script 128)
  S2  border units (both nbr sets defined): nbr_same + nbr_other + ring   (reproduces 128 split)
  S3  border units with both rings defined: nbr_same + nbr_other + ring_same + ring_other
      tests: nbr_same - nbr_other; nbr_same - ring_same; nbr_other - ring_other;
             (nbr_same - ring_same) - (nbr_other - ring_other)
  S4  full sample, same-state comparison only: nbr_same + ring_same
  S5  S3 with Andhra Pradesh and Telangana units dropped (they shared a state to 2014)
  S6  S3 with the 2011 state (Telangana inside Andhra Pradesh) defining "same state"
      and the state-by-year FE before 2014
  S7  chickpea only: S1 + NFSM pulses coverage of the unit at t, share of neighbours and
      of the ring covered at t (NFSM district lists, script 97). From 2010-11 the pulses
      component covers whole states in most of India, so inside state-by-year FE the
      coverage terms are identified mainly by the 2007-08 to 2009-10 district lists
      and the partly covered states.
Timing: for every pair of units that both have an event, the gap in event years,
  by pair type (adjacent same state, adjacent other state, ring same state, ring other
  state). Programme roll-outs tend to put events in the same year; copying puts the
  neighbour's event a year or more later.

Run: C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/140_pe_within_cross_hazard.py
Outputs (outputs/revision_nov8/peer_effects/v2/):
  140_within_cross_hazard.csv, 140_within_cross_tests.csv, 140_event_timing.csv,
  140_nfsm_chickpea.csv, 140_log.txt
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
V2 = os.path.join(ROOT, "outputs", "revision_nov8", "peer_effects", "v2")
os.makedirs(V2, exist_ok=True)

src = open(os.path.join(HERE, "128_peer_effects_spread.py"), encoding="utf-8").read()
cut = src.index("rng = np.random.default_rng(20261107)\nallH")
ns = {"__file__": os.path.join(HERE, "128_peer_effects_spread.py"), "__name__": "s128"}
exec(compile(src[:cut], "128_header", "exec"), ns)
LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


ADJ, DIST, RING, SAME, EYE = ns["ADJ"], ns["DIST"], ns["RING"], ns["SAME"], ns["EYE"]
st, keys, NU, NY, YEARS = ns["st"], ns["keys"], ns["NU"], ns["NY"], ns["YEARS"]
covs, CTRL, EP, demean = ns["covs"], ns["CTRL"], ns["EP"], ns["demean"]
grid, event_years, grower_matrix, setmean = ns["grid"], ns["event_years"], ns["grower_matrix"], ns["setmean"]
st11 = ns["U"].set_index("unit_key").state.reindex(keys).values
st11 = np.array([str(s).upper().replace("&", "AND") for s in st11])
SAME11 = st11[:, None] == st11[None]
APTS = np.isin(st, ["ANDHRA PRADESH", "TELANGANA"])

# NFSM pulses coverage by unit-year (max over child districts), 0 before 2007-08
nf = pd.read_csv(os.path.join(ROOT, "outputs", "crop_groups", "policy_nfsm_district_year.csv"))
nf = nf[nf.component == "pulses"].copy()
nf["year_start"] = nf.year.str[:4].astype(int)
cm = pd.read_csv(os.path.join(ROOT, "outputs", "crop_groups", "crop_group_child_map.csv"))[["div_key", "unit_key"]]
nf = nf.merge(cm, left_on="district_key", right_on="div_key")
nfu = nf.groupby(["unit_key", "year_start"]).treated.max().reset_index()
NF = np.zeros((NU, NY))
ix = {k: i for i, k in enumerate(keys)}
d_ = nfu[nfu.unit_key.isin(ix) & nfu.year_start.between(1998, 2019)]
NF[d_.unit_key.map(ix).values, (d_.year_start - 1998).values] = d_.treated.fillna(0).values


def setmean_x(X, M):
    """mean of a unit x year matrix over a set (row mask matrix M); NaN if empty."""
    den = M.sum(1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den[:, None] > 0, (M.astype(float) @ X) / den[:, None], np.nan)


def build(name, mode, same=SAME):
    col, L, flag = EP[name]
    S = grid(col, flag)
    base, T, status = event_years(S, L, mode)
    G = grower_matrix(T, status)
    sets = {"nbr": ADJ, "ring": RING, "nbr_same": ADJ & same, "nbr_other": ADJ & ~same,
            "ring_same": RING & same, "ring_other": RING & ~same}
    E = {k: setmean(G, M) for k, M in sets.items()}
    E["nf_own"] = NF
    E["nf_nbr"] = setmean_x(NF, ADJ)
    E["nf_ring"] = setmean_x(NF, RING)
    E["nf_nbr_samestatus"] = None  # filled per row below
    rows = []
    for i in np.where(status == "at_risk")[0]:
        for t in range(3, NY):
            y = YEARS[t]
            if np.isfinite(T[i]) and y > T[i]:
                break
            r = {"unit_key": keys[i], "st": st[i], "st11": st11[i], "year": y, "base": base[i],
                 "entry": int(T[i] == y) if np.isfinite(T[i]) else 0}
            for k in ["nbr", "ring", "nbr_same", "nbr_other", "ring_same", "ring_other"]:
                r[k] = E[k][i, t - 1]
            r["nf_own"], r["nf_nbr"], r["nf_ring"] = NF[i, t], E["nf_nbr"][i, t], E["nf_ring"][i, t]
            # same-state neighbours split by whether they share the unit's NFSM status at t
            m_same = ADJ[i] & SAME[i] & np.isfinite(G[:, t - 1])
            ms = m_same & (NF[:, t] == NF[i, t])
            md = m_same & (NF[:, t] != NF[i, t])
            r["nbr_same_nfmatch"] = G[ms, t - 1].mean() if ms.any() else np.nan
            r["nbr_same_nfdiff"] = G[md, t - 1].mean() if md.any() else np.nan
            rows.append(r)
    d = pd.DataFrame(rows).join(covs, on="unit_key")
    return d, T, status


def fit(d, xs, fe_state="st", extra=None, label=""):
    xs_all = xs + (extra or [])
    dd = d.dropna(subset=["entry"] + xs_all + CTRL).copy()
    if fe_state == "st11pre":
        fe_lab = np.where(dd.year < 2014, dd.st11, dd.st)
    else:
        fe_lab = dd[fe_state].values
    sy = pd.factorize(pd.Series(fe_lab).astype(str).values + "_" + dd.year.astype(str).values)[0]
    keep = np.bincount(sy)[sy] > 1
    dd, sy = dd[keep], pd.factorize(sy[keep])[0]
    if dd.entry.sum() < 5:
        return None
    Z = demean(dd[["entry"] + xs_all + CTRL].values.astype(float), [sy])
    out = {"n": len(dd), "n_units": dd.unit_key.nunique(), "n_events": int(dd.entry.sum()),
           "base_rate": dd.entry.mean(), "res": {}}
    for cl in ["unit", "state"]:
        g = pd.factorize(dd.unit_key if cl == "unit" else dd.st)[0]
        out["res"][cl] = (sm.OLS(Z[:, 0], Z[:, 1:]).fit(cov_type="cluster", cov_kwds={"groups": g}),
                          len(np.unique(g)))
    out["xs"] = xs_all
    return out


def contrast(o, cl, wts):
    r, _ = o["res"][cl]
    c = np.zeros(len(o["xs"]) + len(CTRL))
    for k, v in wts.items():
        c[o["xs"].index(k)] = v
    t = r.t_test(c)
    return float(np.squeeze(t.effect)), float(np.squeeze(t.sd)), float(np.squeeze(t.pvalue))


SPECS = {
    "S1 nbr + ring (128)": (["nbr", "ring"], "st", None),
    "S2 border units: nbr_same + nbr_other + ring (128 split)": (["nbr_same", "nbr_other", "ring"], "st", None),
    "S3 border units: four sets": (["nbr_same", "nbr_other", "ring_same", "ring_other"], "st", None),
    "S4 same-state only: nbr_same + ring_same": (["nbr_same", "ring_same"], "st", None),
}
TESTS = {
    "S1 nbr + ring (128)": {"nbr minus ring": {"nbr": 1, "ring": -1}},
    "S2 border units: nbr_same + nbr_other + ring (128 split)": {"nbr_same minus nbr_other": {"nbr_same": 1, "nbr_other": -1}},
    "S3 border units: four sets": {
        "nbr_same minus nbr_other": {"nbr_same": 1, "nbr_other": -1},
        "nbr_same minus ring_same": {"nbr_same": 1, "ring_same": -1},
        "nbr_other minus ring_other": {"nbr_other": 1, "ring_other": -1},
        "edge within minus edge across": {"nbr_same": 1, "ring_same": -1, "nbr_other": -1, "ring_other": 1},
        "ring_same minus ring_other": {"ring_same": 1, "ring_other": -1}},
    "S4 same-state only: nbr_same + ring_same": {"nbr_same minus ring_same": {"nbr_same": 1, "ring_same": -1}},
}
TESTS["S5 S3 without AP and TS"] = TESTS["S3 border units: four sets"]
TESTS["S6 S3 with 2011 states before 2014"] = TESTS["S3 border units: four sets"]

coef_rows, test_rows = [], []
EPIS = [("chickpea", "surge"), ("chickpea", "entry"), ("soybean", "entry"), ("soybean", "surge"),
        ("cotton", "surge"), ("cotton", "entry"), ("maize", "entry"), ("maize", "surge"),
        ("moong", "entry"), ("moong", "surge"), ("rabi_maize", "entry"), ("rabi_maize", "surge"),
        ("rice", "surge")]
datasets = {}
for name, mode in EPIS:
    d, T, status = build(name, mode)
    d11, _, _ = build(name, mode, same=SAME11)
    datasets[(name, mode)] = (d, T, status)
    runs = dict(SPECS)
    runs["S5 S3 without AP and TS"] = (["nbr_same", "nbr_other", "ring_same", "ring_other"], "st", None)
    runs["S6 S3 with 2011 states before 2014"] = (["nbr_same", "nbr_other", "ring_same", "ring_other"], "st11pre", None)
    for spec, (xs, fe, ex) in runs.items():
        dd = d
        if spec.startswith("S5"):
            dd = d[~d.st.isin(["ANDHRA PRADESH", "TELANGANA"])]
        if spec.startswith("S6"):
            # same / other defined on 2011 states before 2014, present-day states after
            dd = d.copy()
            pre = dd.year < 2014
            for k in ["nbr_same", "nbr_other", "ring_same", "ring_other"]:
                dd.loc[pre, k] = d11.loc[pre, k].values
        o = fit(dd, xs, fe, ex)
        if o is None:
            continue
        for cl in ["unit", "state"]:
            r, ncl = o["res"][cl]
            for k, x in enumerate(xs):
                coef_rows.append({"episode": name, "mode": mode, "spec": spec, "cluster": cl, "var": x,
                                  "b": r.params[k], "se": r.bse[k], "p": r.pvalues[k], "n": o["n"],
                                  "n_units": o["n_units"], "n_events": o["n_events"], "base_rate": o["base_rate"],
                                  "n_clusters": ncl})
            for tn, wts in TESTS.get(spec, {}).items():
                b, se, p = contrast(o, cl, wts)
                test_rows.append({"episode": name, "mode": mode, "spec": spec, "cluster": cl, "test": tn,
                                  "b": b, "se": se, "p": p, "n": o["n"], "n_events": o["n_events"]})
    w(f"{name} {mode}: done")

C = pd.DataFrame(coef_rows)
TT = pd.DataFrame(test_rows)
C.to_csv(os.path.join(V2, "140_within_cross_hazard.csv"), index=False)
TT.to_csv(os.path.join(V2, "140_within_cross_tests.csv"), index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
focus = [("chickpea", "surge"), ("chickpea", "entry"), ("soybean", "entry"), ("soybean", "surge"), ("cotton", "surge")]
for name, mode in focus:
    w(f"\n=== {name} {mode} ===")
    c = C[(C.episode == name) & (C["mode"] == mode)]
    w(c.pivot_table(index=["spec", "var"], columns="cluster", values=["b", "se"], sort=False).round(4).to_string())
    w(c.groupby("spec", sort=False)[["n", "n_units", "n_events"]].first().to_string())
    t = TT[(TT.episode == name) & (TT["mode"] == mode)]
    w(t.pivot_table(index=["spec", "test"], columns="cluster", values=["b", "se", "p"], sort=False).round(4).to_string())

# reproduction of script 128
H = pd.read_csv(os.path.join(ROOT, "outputs", "revision_nov8", "peer_effects", "q3_hazard.csv"))
H = H[(H.tag == "main") & (H.spec == "nbr + ring") & (H.cluster == "unit") & H["var"].isin(["nbr", "ring"])]
chk = C[(C.spec.str.startswith("S1")) & (C.cluster == "unit")].merge(
    H, left_on=["episode", "mode", "var"], right_on=["episode", "mode", "var"], suffixes=("", "_128"))
w(f"\nreproduction of script 128 nbr + ring: max |b - b_128| = {(chk.b - chk.b_128).abs().max():.2e} over {len(chk)} terms")
H2 = pd.read_csv(os.path.join(ROOT, "outputs", "revision_nov8", "peer_effects", "q3_hazard.csv"))
H2 = H2[(H2.tag == "main") & (H2.spec == "nbr split by border") & (H2.cluster == "unit")]
chk2 = C[(C.spec.str.startswith("S2")) & (C.cluster == "unit")].merge(
    H2, on=["episode", "mode", "var"], suffixes=("", "_128"))
w(f"reproduction of script 128 split: max |b - b_128| = {(chk2.b - chk2.b_128).abs().max():.2e} over {len(chk2)} terms")

# ------------------------------------------------------------------ NFSM, chickpea
nfr = []
for mode in ["surge", "entry"]:
    d, T, status = datasets[("chickpea", mode)]
    ev = d[d.entry == 1]
    w(f"\nchickpea {mode}: events by year: " + ", ".join(f"{k}: {v}" for k, v in ev.year.value_counts().sort_index().items()))
    w(f"  events in an NFSM-pulses unit that year: {int(ev.nf_own.sum())} of {len(ev)}; at-risk unit-years covered "
      f"{d.nf_own.mean():.2f}; events 2007-2009 {int(ev.year.between(2007, 2009).sum())}")
    # within-state variation in coverage, by year
    v = d.groupby(["year", "st"]).nf_own.agg(lambda x: 0 < x.mean() < 1).groupby("year").sum()
    w("  states with within-state variation in NFSM pulses coverage, by year: "
      + ", ".join(f"{k}: {int(x)}" for k, x in v.items() if k >= 2007))
    for spec, xs, ex in [("base: nbr + ring", ["nbr", "ring"], None),
                         ("+ NFSM own, nbr share, ring share", ["nbr", "ring"], ["nf_own", "nf_nbr", "nf_ring"]),
                         ("base, 2007-2019 only", ["nbr", "ring"], None),
                         ("same-state nbrs split by shared NFSM status", ["nbr_same_nfmatch", "nbr_same_nfdiff"], None)]:
        dd = d if "2007" not in spec else d[d.year >= 2007]
        o = fit(dd, xs, "st", ex)
        if o is None:
            continue
        for cl in ["unit", "state"]:
            r, ncl = o["res"][cl]
            for k, x in enumerate(o["xs"]):
                nfr.append({"mode": mode, "spec": spec, "cluster": cl, "var": x, "b": r.params[k], "se": r.bse[k],
                            "p": r.pvalues[k], "n": o["n"], "n_events": o["n_events"]})
            if "nbr" in xs and "ring" in xs:
                b, se, p = contrast(o, cl, {"nbr": 1, "ring": -1})
                nfr.append({"mode": mode, "spec": spec, "cluster": cl, "var": "nbr minus ring", "b": b, "se": se,
                            "p": p, "n": o["n"], "n_events": o["n_events"]})
NFR = pd.DataFrame(nfr)
NFR.to_csv(os.path.join(V2, "140_nfsm_chickpea.csv"), index=False)
w(NFR.pivot_table(index=["mode", "spec", "var"], columns="cluster", values=["b", "se", "p"], sort=False).round(4).to_string())

# ------------------------------------------------------------------ event timing by pair type
trows = []
iu = np.triu_indices(NU, 1)
for name, mode in EPIS:
    _, T, status = datasets[(name, mode)]
    ok = np.isfinite(T)
    for lab, M in [("adjacent, same state", ADJ & SAME), ("adjacent, other state", ADJ & ~SAME),
                   ("ring, same state", RING & SAME & RING.T), ("ring, other state", RING & ~SAME & RING.T)]:
        Mu = np.triu(M | M.T, 1) & ok[:, None] & ok[None, :]
        a, b = np.nonzero(Mu)
        if len(a) == 0:
            trows.append({"episode": name, "mode": mode, "pairs": lab, "n_pairs": 0})
            continue
        gap = np.abs(T[a] - T[b])
        trows.append({"episode": name, "mode": mode, "pairs": lab, "n_pairs": len(a),
                      "share_same_year": float((gap == 0).mean()), "share_within_1y": float((gap <= 1).mean()),
                      "share_within_2y": float((gap <= 2).mean()), "median_gap": float(np.median(gap))})
TM = pd.DataFrame(trows)
TM.to_csv(os.path.join(V2, "140_event_timing.csv"), index=False)
w("\nEvent-year gap for pairs where both units have the event:")
w(TM[TM.episode.isin(["chickpea", "soybean", "cotton"])].round(3).to_string(index=False))

with open(os.path.join(V2, "140_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
