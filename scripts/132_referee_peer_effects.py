"""
132_referee_peer_effects.py

Referee checks on the district peer-effects results (scripts 125-129,
outputs/revision_nov8/peer_effects/peer_effects_results.md). Tries to refute each
headline. Reads the outputs of 125-128 and reuses the definitions in 128 (executed
read-only up to its main loop); edits nothing upstream.

  H1  similarity falls with distance; 77% for adjacent same-state pairs
      - unit-bootstrap CIs, alternative similarity measures, "110 km" computed three ways
  H2  border effect -7.2 raw, -3.4 / -4.1 after geography
      - dyadic SEs (unit, state) reproduced; wild-cluster bootstrap with state-level
        weights (Webb, restricted null; a cross-border pair gets (v_a + v_b)/sqrt 2)
      - placebo partitions: 500 random k-means regions with as many regions as states;
        how large a "border" coefficient does an arbitrary line give?
      - one-district-inside placebo: same-state adjacent pairs where exactly one unit
        touches another state, against interior pairs
      - similarity measure: L1 (base), L1 without "other", L1 with shares < 1% zeroed,
        both, Pearson correlation of shares, Bhattacharyya affinity, |D1 gap|, |D0 gap|
      - leave one state out, leave one border out, drop Punjab, drop Telangana and AP
  H3  per-group border gaps, M1-M3, unit and state SEs, wild bootstrap p, BH and Holm
      over the nine groups; exact decomposition of the similarity gap into groups
  H4  widening: pair-year panel reproduced and re-run with the "other"-free and 1%
      similarity; Telangana cotton borders: pair by pair, timing, sign-flip p
  H5  Moran's I of changes: within vs cross with AP-TS links excluded, within-state links
      restricted to border units, within-state link subsets matched in size to the
      cross-border set, and both link sets on state-demeaned values
  H6  wave: benchmarks within state x baseline-share bin, covariate nearest neighbours
      (national and within state), propensity-weighted within state; entry noise flags
      (crop absent from records before entry, state crop-list additions, GCA jumps,
      not sustained to 2015-19)
  H7  neighbour minus ring: reproduce, BH with unit and state SEs, alternative rings

Run: C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/132_referee_peer_effects.py
Outputs: outputs/revision_nov8/peer_effects/referee/ (csv per check + referee_log.txt)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _pe_common import OUT, GROUPS, load, dyadic_ols, dyadic_vcov, demean, moran  # noqa: E402

ROOT = os.path.dirname(HERE)
RO = os.path.join(OUT, "referee")
os.makedirs(RO, exist_ok=True)
LOG = []
FAST = "--fast" in sys.argv


def w(s=""):
    print(s, flush=True)
    LOG.append(str(s))


def save(df, name):
    df.to_csv(os.path.join(RO, name), index=False)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def holm(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    a = np.maximum.accumulate(p[o] * (n - np.arange(n)))
    out = np.empty(n)
    out[o] = np.minimum(a, 1)
    return out


rng = np.random.default_rng(132)
units, uy, up, P = load()
upw = {lab: up[up.window == lab].set_index("unit_key") for lab in ["w9802", "w1519"]}
codes_u = {u: i for i, u in enumerate(sorted(units.unit_key))}
codes_s = {s: i for i, s in enumerate(sorted(units.st.dropna().unique()))}
GEO = ["d_rain", "d_raincv", "d_elev", "d_tri", "d_irr"]
SG = [f"s_{g}" for g in GROUPS]


# ------------------------------------------------------------------ similarity measures
def shares(lab, unit_keys):
    return upw[lab].reindex(unit_keys)[SG].values


def renorm(M):
    s = M.sum(1, keepdims=True)
    return np.where(s > 0, M / np.where(s > 0, s, 1), np.nan)


def measures(A_, B_, D1a, D1b, D0a, D0b):
    out = {}
    out["sim_L1"] = (1 - 0.5 * np.abs(A_ - B_).sum(1)) * 100
    a8, b8 = renorm(A_[:, :8]), renorm(B_[:, :8])
    out["sim_L1_no_other"] = (1 - 0.5 * np.abs(a8 - b8).sum(1)) * 100
    a1, b1 = renorm(np.where(A_ >= 0.01, A_, 0)), renorm(np.where(B_ >= 0.01, B_, 0))
    out["sim_L1_gt1pct"] = (1 - 0.5 * np.abs(a1 - b1).sum(1)) * 100
    a81, b81 = renorm(np.where(a8 >= 0.01, a8, 0)), renorm(np.where(b8 >= 0.01, b8, 0))
    out["sim_L1_no_other_gt1pct"] = (1 - 0.5 * np.abs(a81 - b81).sum(1)) * 100
    am, bm = A_ - A_.mean(1, keepdims=True), B_ - B_.mean(1, keepdims=True)
    out["sim_corr"] = (am * bm).sum(1) / np.sqrt((am ** 2).sum(1) * (bm ** 2).sum(1)) * 100
    out["sim_bhatt"] = np.sqrt(np.clip(A_, 0, None) * np.clip(B_, 0, None)).sum(1) * 100
    out["neg_dD1"] = -np.abs(D1a - D1b)
    out["neg_dD0"] = -np.abs(D0a - D0b)
    for k, g in enumerate(GROUPS):
        out[f"g_{g}"] = np.abs(A_[:, k] - B_[:, k]) * 100
    return out


# window-mean D0 (crops reported) per unit
d0w = {}
for lab, (a, b) in [("w9802", (1998, 2002)), ("w1519", (2015, 2019))]:
    d0w[lab] = uy[(uy.reliable == 1) & uy.year_start.between(a, b)].groupby("unit_key").D0.mean()

for lab in ["w9802", "w1519"]:
    A_, B_ = shares(lab, P.unit_i), shares(lab, P.unit_j)
    m = measures(A_, B_, upw[lab].reindex(P.unit_i).D1.values, upw[lab].reindex(P.unit_j).D1.values,
                 d0w[lab].reindex(P.unit_i).values, d0w[lab].reindex(P.unit_j).values)
    for k, v in m.items():
        P[f"{k}_{lab}"] = v
assert np.allclose(P.sim_L1_w1519.dropna(), P.sim_w1519.dropna() * 100)
P["ui"], P["uj"] = P.unit_i.map(codes_u), P.unit_j.map(codes_u)
P["si"], P["sj"] = P.st_i.map(codes_s), P.st_j.map(codes_s)
Pn = P[P.new_border == 0].copy()
MEAS = ["sim_L1", "sim_L1_no_other", "sim_L1_gt1pct", "sim_L1_no_other_gt1pct", "sim_corr", "sim_bhatt",
        "neg_dD1", "neg_dD0"]

# ================================================================== H1
w("=" * 80)
w("H1. Similarity falls with distance; adjacent same-state pairs share 77%")
CLS = [("adjacent, same state", lambda d: (d.adjacent == 1) & (d.same_state == 1)),
       ("adjacent, across a border", lambda d: (d.adjacent == 1) & (d.same_state == 0)),
       ("not adjacent, 60-100 km, same state",
        lambda d: (d.adjacent == 0) & d.dist_km.between(60, 100) & (d.same_state == 1)),
       ("200-300 km, same state", lambda d: d.dist_km.between(200, 300) & (d.same_state == 1)),
       ("200-300 km, across a border", lambda d: d.dist_km.between(200, 300) & (d.same_state == 0)),
       ("500-600 km, any", lambda d: d.dist_km.between(500, 600))]
lab = "w1519"
NB = 200 if FAST else 400
ukeys = sorted(set(Pn.unit_i) | set(Pn.unit_j))
uidx = {u: k for k, u in enumerate(ukeys)}
pi_, pj_ = Pn.unit_i.map(uidx).values, Pn.unit_j.map(uidx).values
boot_counts = rng.multinomial(len(ukeys), np.ones(len(ukeys)) / len(ukeys), size=NB)
h1 = []
for meas in ["sim_L1", "sim_L1_no_other", "sim_L1_gt1pct", "sim_corr", "sim_bhatt"]:
    y = Pn[f"{meas}_{lab}"].values
    for nm, f in CLS:
        msk = f(Pn).values & np.isfinite(y)
        est = y[msk].mean()
        bs = []
        for c in boot_counts:
            wt = c[pi_[msk]] * c[pj_[msk]]
            bs.append((wt * y[msk]).sum() / wt.sum() if wt.sum() > 0 else np.nan)
        h1.append({"measure": meas, "pairs": nm, "n": int(msk.sum()), "mean": est,
                   "lo95_unit_boot": np.nanpercentile(bs, 2.5), "hi95_unit_boot": np.nanpercentile(bs, 97.5)})
h1 = pd.DataFrame(h1)
save(h1, "h1_similarity_classes.csv")
w(h1.pivot_table(index="pairs", columns="measure", values="mean", sort=False).round(1).to_string())
w("unit-bootstrap 95% CI, L1 measure:")
w(h1[h1.measure == "sim_L1"][["pairs", "n", "mean", "lo95_unit_boot", "hi95_unit_boot"]].round(1).to_string(index=False))

# monotone fall with distance? same-state and cross-border, 50 km bins
bins = np.arange(0, 650, 50)
mono = []
for meas in ["sim_L1", "sim_L1_no_other", "sim_corr"]:
    for ss in [1, 0]:
        d = Pn[Pn.same_state == ss]
        g = d.groupby(pd.cut(d.dist_km, bins))[f"{meas}_{lab}"].mean()
        mono.append({"measure": meas, "same_state": ss, **{str(k.right): v for k, v in g.items()}})
mono = pd.DataFrame(mono)
save(mono, "h1_distance_profile_50km.csv")
w("mean similarity by 50 km bin (upper edge):")
w(mono.round(1).to_string(index=False))

# "110 km": three ways
xb = Pn[(Pn.adjacent == 1) & (Pn.same_state == 0)][f"sim_L1_{lab}"].mean()
xa = Pn[(Pn.adjacent == 1) & (Pn.same_state == 1)]
xa_km, xa_sim = xa.dist_km.mean(), xa[f"sim_L1_{lab}"].mean()
ss = Pn[(Pn.same_state == 1)].dropna(subset=[f"sim_L1_{lab}"])


def km_cross(dd, target):
    # (a) 20 km bins, first bin whose mean falls to target (the original rule)
    b = (dd.dist_km // 20) * 20 + 10
    g = dd.groupby(b)[f"sim_L1_{lab}"].agg(["mean", "size"])
    g = g[g["size"] >= 10]
    hit = g[g["mean"] <= target]
    a = hit.index[0] if len(hit) else np.nan
    # (b) linear interpolation of 20 km bin means
    gi = g["mean"].values
    xs = g.index.values
    k = np.where(gi <= target)[0]
    if len(k) and k[0] > 0:
        j = k[0]
        bb = xs[j - 1] + (gi[j - 1] - target) / (gi[j - 1] - gi[j]) * (xs[j] - xs[j - 1])
    else:
        bb = np.nan
    # (c) linear fit 0-300 km
    q = dd[dd.dist_km <= 300]
    sl, ic = np.polyfit(q.dist_km, q[f"sim_L1_{lab}"], 1)
    c = (target - ic) / sl
    return a, bb, c, sl


a_, b_, c_, sl = km_cross(ss, xb)
w(f"\ncross-border adjacent similarity {xb:.2f}; same-state adjacent {xa_sim:.2f} at {xa_km:.0f} km")
w(f"distance at which same-state pairs fall to {xb:.1f}: 20 km-bin rule {a_:.0f} km, interpolated {b_:.0f} km, "
  f"linear 0-300 km fit {c_:.0f} km (slope {sl * 100:.2f} pts per 100 km)")
kb = []
for c in boot_counts[: (100 if FAST else 200)]:
    wt_u = pd.Series(c, index=ukeys)
    dd = ss.assign(_w=wt_u.reindex(ss.unit_i).values * wt_u.reindex(ss.unit_j).values)
    dd = dd.loc[dd.index.repeat(dd._w.astype(int))]
    xadj = Pn[(Pn.adjacent == 1) & (Pn.same_state == 0)].dropna(subset=[f"sim_L1_{lab}"])
    wx = wt_u.reindex(xadj.unit_i).values * wt_u.reindex(xadj.unit_j).values
    tgt = (wx * xadj[f"sim_L1_{lab}"].values).sum() / wx.sum()
    aa = dd[dd.adjacent == 1]
    try:
        r = km_cross(dd, tgt)
        kb.append((r[1] - aa.dist_km.mean(), r[2] - aa.dist_km.mean()))
    except Exception:
        pass
kb = np.array(kb)
w(f"border-equivalent km (crossing minus adjacent same-state mean km), unit bootstrap 95%: "
  f"interpolated {b_ - xa_km:.0f} [{np.nanpercentile(kb[:, 0], 2.5):.0f}, {np.nanpercentile(kb[:, 0], 97.5):.0f}], "
  f"linear {c_ - xa_km:.0f} [{np.nanpercentile(kb[:, 1], 2.5):.0f}, {np.nanpercentile(kb[:, 1], 97.5):.0f}]")
# (d) regression-based: border coef / distance slope among all pairs <= 300 km with both terms
q = Pn[Pn.dist_km <= 300].dropna(subset=[f"sim_L1_{lab}"])
o = dyadic_ols(q[f"sim_L1_{lab}"].values, np.column_stack([q.cross, q.dist_km / 100, q.adjacent]),
               ["cross", "dist100", "adj"], q.ui.values, q.uj.values)
w(f"pairs <= 300 km, sim on cross + distance + adjacent: cross {o['cross'][0]:.2f} ({o['cross'][1]:.2f}), "
  f"per 100 km {o['dist100'][0]:.2f} ({o['dist100'][1]:.2f}) -> border = {100 * o['cross'][0] / o['dist100'][0]:.0f} km")
save(pd.DataFrame([{"target_cross_adjacent": xb, "same_state_adjacent_km": xa_km, "bin_rule_km": a_,
                    "interp_km": b_, "linear_km": c_, "equiv_interp": b_ - xa_km, "equiv_linear": c_ - xa_km,
                    "equiv_interp_lo": np.nanpercentile(kb[:, 0], 2.5), "equiv_interp_hi": np.nanpercentile(kb[:, 0], 97.5),
                    "equiv_linear_lo": np.nanpercentile(kb[:, 1], 2.5), "equiv_linear_hi": np.nanpercentile(kb[:, 1], 97.5),
                    "reg_cross": o["cross"][0], "reg_per100km": o["dist100"][0],
                    "equiv_reg": 100 * o["cross"][0] / o["dist100"][0]}]), "h1_border_km_equivalent.csv")

# ================================================================== H2
w("\n" + "=" * 80)
w("H2. Border effect on similarity, adjacent pairs")
A = Pn[Pn.adjacent == 1].copy()


def design(base, spec):
    X = {"cross": base.cross.values.astype(float), "dist100": base.dist_km.values / 100}
    if spec in ("M2", "M3"):
        X.update({g: base[g].values for g in GEO})
    Xm = np.column_stack(list(X.values()))
    names = list(X)
    if spec == "M3":
        S = np.zeros((len(base), len(codes_s)))
        np.add.at(S, (np.arange(len(base)), base.si.values), 0.5)
        np.add.at(S, (np.arange(len(base)), base.sj.values), 0.5)
        S = S[:, S.sum(0) > 0][:, 1:]
        Xm = np.column_stack([Xm, S])
        names += [f"_s{k}" for k in range(S.shape[1])]
    return Xm, names


def reg(base, ycol, spec, cl="unit"):
    Xm, names = design(base, spec)
    mi, mj = (base.ui.values, base.uj.values) if cl == "unit" else (base.si.values, base.sj.values)
    o = dyadic_ols(base[ycol].values.astype(float), Xm, names, mi, mj, conservative=(cl == "state"))
    return o


WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])


def wild_state(base, ycol, spec, B=999):
    """wild-cluster restricted bootstrap; state weights; cross-border pair weight
    (v_a + v_b)/sqrt(2). t uses the conservative state-dyadic vcov."""
    y = base[ycol].values.astype(float)
    Xm, names = design(base, spec)
    X = np.column_stack([np.ones(len(y)), Xm])
    k0 = 1  # cross is column 1
    br = np.linalg.lstsq(np.delete(X, k0, 1), y, rcond=None)[0]
    yr = np.delete(X, k0, 1) @ br
    er = y - yr
    si, sj = base.si.values, base.sj.values
    XtXi = np.linalg.pinv(X.T @ X)

    def tstat(yy):
        b = XtXi @ X.T @ yy
        e = yy - X @ b
        V, _ = dyadic_vcov(X, e, si, sj, conservative=True)
        return b[k0] / np.sqrt(V[k0, k0])

    t0 = tstat(y)
    ns = int(max(si.max(), sj.max())) + 1
    same = si == sj
    ts = np.empty(B)
    for r in range(B):
        v = rng.choice(WEBB, ns)
        wt = np.where(same, v[si], (v[si] + v[sj]) / np.sqrt(2))
        ts[r] = tstat(yr + wt * er)
    return t0, (1 + np.sum(np.abs(ts) >= abs(t0))) / (B + 1)


B_WILD = 199 if FAST else 999
h2 = []
for lab in ["w1519", "w9802"]:
    for meas in MEAS:
        ycol = f"{meas}_{lab}"
        base = A.dropna(subset=[ycol] + GEO).copy()
        sd_same = base.loc[base.cross == 0, ycol].std()
        for spec in ["M1", "M2", "M3"]:
            ou, os_ = reg(base, ycol, spec, "unit"), reg(base, ycol, spec, "state")
            rec = {"window": lab, "measure": meas, "spec": spec, "n": ou["_n"], "b": ou["cross"][0],
                   "se_unit": ou["cross"][1], "p_unit": ou["cross"][2], "se_state": os_["cross"][1],
                   "p_state": os_["cross"][2], "n_states": os_["_ncl"],
                   "mean_same": base.loc[base.cross == 0, ycol].mean(), "sd_same": sd_same,
                   "b_in_sd": ou["cross"][0] / sd_same}
            if lab == "w1519" and (meas in ("sim_L1", "sim_L1_no_other", "sim_L1_no_other_gt1pct", "sim_corr")
                                   or spec == "M2"):
                t0, pw = wild_state(base, ycol, spec, B=B_WILD)
                rec.update({"t_state": t0, "p_wild_state": pw})
            h2.append(rec)
h2 = pd.DataFrame(h2)
save(h2, "h2_border_by_measure.csv")
w(h2[h2.window == "w1519"][["measure", "spec", "n", "b", "se_unit", "se_state", "p_state", "p_wild_state",
                            "mean_same", "b_in_sd"]].round(3).to_string(index=False))
w("1998-2002:")
w(h2[(h2.window == "w9802") & h2.measure.isin(["sim_L1", "sim_L1_no_other", "sim_L1_no_other_gt1pct"])]
  [["measure", "spec", "n", "b", "se_unit", "se_state", "p_state"]].round(3).to_string(index=False))

# ---- placebo partitions
from scipy.cluster.vq import kmeans2  # noqa: E402

lab = "w1519"
ycol = f"sim_L1_{lab}"
base = A.dropna(subset=[ycol] + GEO).copy()
uu = units[units.unit_key.isin(set(Pn.unit_i) | set(Pn.unit_j))].set_index("unit_key")
XY = uu[["cx", "cy"]].values / 1000
nst = len(set(base.st_i) | set(base.st_j))
real = {s: reg(base, ycol, s)["cross"][0] for s in ["M1", "M2"]}
NPL = 200 if FAST else 500
pl = []
for r in range(NPL):
    _, labs = kmeans2(XY, nst, minit="++", iter=10, seed=int(rng.integers(1e9)))
    lu = pd.Series(labs, index=uu.index)
    b2 = base.copy()
    b2["cross"] = (lu.reindex(b2.unit_i).values != lu.reindex(b2.unit_j).values).astype(int)
    rec = {"draw": r, "share_cross": b2.cross.mean()}
    for s in ["M1", "M2"]:
        rec[s] = reg(b2, ycol, s)["cross"][0]
    # placebo lines inside true states only: same-state pairs, placebo cross
    b3 = b2[b2.same_state == 1]
    if b3.cross.sum() > 20:
        rec["M2_within_true_states"] = reg(b3, ycol, "M2")["cross"][0]
    pl.append(rec)
pl = pd.DataFrame(pl)
save(pl, "h2_placebo_partitions.csv")
for s in ["M1", "M2"]:
    w(f"placebo k-means partitions ({NPL}, {nst} regions): {s} placebo border coef mean {pl[s].mean():.2f}, "
      f"sd {pl[s].std():.2f}, 2.5-97.5% [{pl[s].quantile(.025):.2f}, {pl[s].quantile(.975):.2f}]; real {real[s]:.2f}; "
      f"share of placebos at or below real {(pl[s] <= real[s]).mean():.3f}")
w(f"placebo lines drawn inside true states (same-state pairs only), M2: mean {pl.M2_within_true_states.mean():.2f}, "
  f"sd {pl.M2_within_true_states.std():.2f}, share <= real M2 {(pl.M2_within_true_states <= real['M2']).mean():.3f}")

# ---- one-district-inside placebo
touch = set(A[A.cross == 1].unit_i) | set(A[A.cross == 1].unit_j)
A["bi"] = A.unit_i.isin(touch).astype(int) + A.unit_j.isin(touch).astype(int)
A["ptype"] = np.where(A.cross == 1, "cross-border", np.where(A.bi == 2, "same state, both border districts",
                      np.where(A.bi == 1, "same state, border-interior", "same state, interior")))
b4 = A.dropna(subset=[ycol] + GEO).copy()
w("\nsimilarity by adjacent pair type, 2015-19:")
w(b4.groupby("ptype")[[ycol, "d_irr", "d_tri", "dist_km"]].agg(["mean", "size"]).round(2).to_string())
for nm, sub in [("border-interior vs interior", b4[b4.ptype.isin(["same state, border-interior", "same state, interior"])]),
                ("both-border vs interior", b4[b4.ptype.isin(["same state, both border districts", "same state, interior"])])]:
    sub = sub.copy()
    sub["cross"] = (sub.ptype != "same state, interior").astype(int)
    for s in ["M1", "M2"]:
        o = reg(sub, ycol, s)
        w(f"  placebo '{nm}' {s}: {o['cross'][0]:.2f} ({o['cross'][1]:.2f}), n {o['_n']}")
# real border effect against only border-district same-state pairs as the comparison
sub = b4[b4.ptype.isin(["cross-border", "same state, both border districts", "same state, border-interior"])]
for s in ["M1", "M2", "M3"]:
    o = reg(sub, ycol, s)
    w(f"  cross-border vs same-state pairs that include a border district, {s}: {o['cross'][0]:.2f} ({o['cross'][1]:.2f})")

# ---- leave-outs
lo = []
for meas in ["sim_L1", "sim_L1_no_other_gt1pct"]:
    ycol_ = f"{meas}_{lab}"
    base_ = A.dropna(subset=[ycol_] + GEO).copy()
    sts = sorted(set(base_.st_i) | set(base_.st_j))
    drops = [("none", [])] + [(f"state {s}", [s]) for s in sts] + [
        ("Punjab", ["PUNJAB"]), ("Telangana + Andhra Pradesh", ["TELANGANA", "ANDHRA PRADESH"]),
        ("Punjab + Haryana", ["PUNJAB", "HARYANA"]), ("Punjab + Haryana + Rajasthan", ["PUNJAB", "HARYANA", "RAJASTHAN"])]
    for nm, ds in drops:
        sub = base_[~base_.st_i.isin(ds) & ~base_.st_j.isin(ds)]
        for s in ["M1", "M2", "M3"]:
            o = reg(sub, ycol_, s)
            lo.append({"measure": meas, "drop": nm, "kind": "state", "spec": s, "b": o["cross"][0],
                       "se_unit": o["cross"][1], "n": o["_n"]})
    xk = base_[base_.cross == 1]
    keys = sorted(set(tuple(sorted(t)) for t in zip(xk.st_i, xk.st_j)))
    for kk in keys:
        sub = base_[~((base_.cross == 1) & ([tuple(sorted(t)) == kk for t in zip(base_.st_i, base_.st_j)]))]
        for s in ["M2", "M3"]:
            o = reg(sub, ycol_, s)
            lo.append({"measure": meas, "drop": f"border {kk[0]} | {kk[1]}", "kind": "border", "spec": s,
                       "b": o["cross"][0], "se_unit": o["cross"][1], "n": o["_n"]})
lo = pd.DataFrame(lo)
save(lo, "h2_leave_out.csv")
for meas in ["sim_L1", "sim_L1_no_other_gt1pct"]:
    for s in ["M1", "M2", "M3"]:
        d = lo[(lo.measure == meas) & (lo.spec == s)]
        if d.empty:
            continue
        full = d[d["drop"] == "none"].b
        full = full.iloc[0] if len(full) else np.nan
        ds = d[d.kind == "state"]
        db = d[d.kind == "border"]
        msg = (f"{meas} {s}: full {full:.2f}; drop one state range [{ds.b.min():.2f} ({ds.loc[ds.b.idxmin(), 'drop']}), "
               f"{ds.b.max():.2f} ({ds.loc[ds.b.idxmax(), 'drop']})]")
        if len(db):
            msg += f"; drop one border range [{db.b.min():.2f}, {db.b.max():.2f}]"
        w(msg)
    w(lo[(lo.measure == meas) & lo["drop"].isin(["Punjab", "Telangana + Andhra Pradesh", "Punjab + Haryana",
                                                  "Punjab + Haryana + Rajasthan"])]
      .round(2).to_string(index=False))

# ================================================================== H3
w("\n" + "=" * 80)
w("H3. Per-group border gaps (absolute share gaps, points), adjacent pairs, 2015-19")
h3 = []
for lab in ["w1519", "w9802"]:
    for spec in ["M1", "M2", "M3"]:
        bsim = None
        for g in ["sim_L1"] + [f"g_{x}" for x in GROUPS]:
            ycol = f"{g}_{lab}"
            base = A.dropna(subset=[f"sim_L1_{lab}"] + GEO).copy()
            ou, os_ = reg(base, ycol, spec, "unit"), reg(base, ycol, spec, "state")
            rec = {"window": lab, "spec": spec, "outcome": g, "b": ou["cross"][0], "se_unit": ou["cross"][1],
                   "p_unit": ou["cross"][2], "se_state": os_["cross"][1], "p_state": os_["cross"][2],
                   "mean_same": base.loc[base.cross == 0, ycol].mean(), "n": ou["_n"]}
            if lab == "w1519" and spec in ("M2", "M3"):
                rec["p_wild_state"] = wild_state(base, ycol, spec, B=B_WILD)[1]
            if g == "sim_L1":
                bsim = ou["cross"][0]
            else:
                rec["share_of_sim_gap"] = -0.5 * ou["cross"][0] / bsim
            rec["lo95_state"] = rec["b"] - 1.96 * rec["se_state"]
            rec["hi95_state"] = rec["b"] + 1.96 * rec["se_state"]
            h3.append(rec)
h3 = pd.DataFrame(h3)
for (lab, spec), d in h3[h3.outcome != "sim_L1"].groupby(["window", "spec"]):
    for c in ["p_unit", "p_state", "p_wild_state"]:
        if c in d and d[c].notna().all():
            h3.loc[d.index, c.replace("p_", "q_bh_")] = bh(d[c].values)
            h3.loc[d.index, c.replace("p_", "holm_")] = holm(d[c].values)
save(h3, "h3_group_border_gaps.csv")
cols = ["spec", "outcome", "b", "se_unit", "se_state", "p_unit", "p_state", "p_wild_state", "q_bh_unit",
        "q_bh_state", "q_bh_wild_state", "holm_wild_state", "share_of_sim_gap", "mean_same"]
w(h3[h3.window == "w1519"][[c for c in cols if c in h3]].round(3).to_string(index=False))
d = h3[(h3.window == "w1519") & (h3.outcome != "sim_L1")]
for spec in ["M1", "M2", "M3"]:
    dd = d[d.spec == spec].set_index("outcome").share_of_sim_gap
    w(f"{spec}: share of the similarity border gap from rice+wheat {dd[['g_rice', 'g_wheat']].sum():.2f}, "
      f"pulses+millets {dd[['g_pulses', 'g_millets']].sum():.2f}, other {dd['g_other']:.2f}, "
      f"cotton {dd['g_cotton']:.2f}, oilseeds {dd['g_oilseeds']:.2f}, maize {dd['g_maize']:.2f}, sugarcane {dd['g_sugarcane']:.2f}")
# relative to the same-state mean gap
w("border coefficient as a share of the same-state mean gap, M2: " + ", ".join(
    f"{r.outcome[2:]} {r.b / r.mean_same:+.2f}" for r in d[d.spec == "M2"].itertuples()))

# ================================================================== H4
w("\n" + "=" * 80)
w("H4. Widening, pair-year panel")
sh = uy[uy.reliable == 1].set_index(["unit_key", "year_start"])
SH = [f"share_gca_{g}" for g in GROUPS]
AA = P[P.adjacent == 1].reset_index(drop=True)
years = sorted(uy.year_start.unique())
pp = []
for t in years:
    a = sh.reindex(list(zip(AA.unit_i, [t] * len(AA))))[SH].values
    b = sh.reindex(list(zip(AA.unit_j, [t] * len(AA))))[SH].values
    m = measures(a, b, np.zeros(len(a)), np.zeros(len(a)), np.zeros(len(a)), np.zeros(len(a)))
    d = pd.DataFrame({"pid": np.arange(len(AA)), "year": t, "sim_L1": m["sim_L1"],
                      "sim_L1_no_other": m["sim_L1_no_other"], "sim_L1_no_other_gt1pct": m["sim_L1_no_other_gt1pct"],
                      "sim_corr": m["sim_corr"], "g_rice": m["g_rice"], "g_cotton": m["g_cotton"]})
    pp.append(d)
pp = pd.concat(pp, ignore_index=True).merge(
    AA[["cross", "new_border", "ui", "uj", "si", "sj"]].rename_axis("pid").reset_index(), on="pid")
pp = pp.dropna(subset=["sim_L1"])
pp["oldx"] = ((pp.cross == 1) & (pp.new_border == 0)).astype(int)
PER = [(2003, 2007), (2008, 2012), (2013, 2019)]
for a_, b_ in PER:
    pp[f"x_{a_}"] = pp.oldx * pp.year.between(a_, b_)
pp["x_trend"] = pp.oldx * (pp.year - 1998) / 10
pp["nb_post14"] = pp.new_border * (pp.year >= 2014)
gp, gy = pd.factorize(pp.pid)[0], pd.factorize(pp.year)[0]
h4 = []
for yv in ["sim_L1", "sim_L1_no_other", "sim_L1_no_other_gt1pct", "sim_corr", "g_rice", "g_cotton"]:
    dd = pp.dropna(subset=[yv])
    gp_, gy_ = pd.factorize(dd.pid)[0], pd.factorize(dd.year)[0]
    for spec, xs in [("periods", ["x_2003", "x_2008", "x_2013", "nb_post14"]), ("trend", ["x_trend", "nb_post14"])]:
        Z = demean(dd[[yv] + xs].values.astype(float), [gp_, gy_])
        for cl, (mi, mj) in [("unit", (dd.ui.values, dd.uj.values)), ("state", (dd.si.values, dd.sj.values))]:
            o = dyadic_ols(Z[:, 0], Z[:, 1:], xs, mi, mj, add_const=False, conservative=(cl == "state"))
            for x in xs[:-1]:
                h4.append({"outcome": yv, "spec": spec, "term": x, "cluster": cl, "b": o[x][0], "se": o[x][1],
                           "lo95": o[x][0] - 1.96 * o[x][1], "hi95": o[x][0] + 1.96 * o[x][1], "p": o[x][2]})
h4 = pd.DataFrame(h4)
save(h4, "h4_widening.csv")
w(h4[h4.term.isin(["x_2013", "x_trend"])].round(2).to_string(index=False))

# Telangana cotton: pair by pair
w("\nTelangana cotton borders, pair by pair (cotton share, Telangana side minus other side, points):")
TS = []
X = P[(P.adjacent == 1) & (P.cross == 1)]
cot = uy[uy.reliable == 1].pivot_table(index="unit_key", columns="year_start", values="sh_cotton")
for other in ["ANDHRA PRADESH", "KARNATAKA", "MAHARASHTRA"]:
    d = X[((X.st_i == "TELANGANA") & (X.st_j == other)) | ((X.st_j == "TELANGANA") & (X.st_i == other))]
    for r in d.itertuples():
        ua, ub = (r.unit_i, r.unit_j) if r.st_i == "TELANGANA" else (r.unit_j, r.unit_i)
        ser = (cot.reindex([ua]).iloc[0] - cot.reindex([ub]).iloc[0]) * 100
        e = ser[[y for y in range(1998, 2003) if y in ser.index]].mean()
        l = ser[[y for y in range(2015, 2020) if y in ser.index]].mean()
        ca = cot.reindex([ua]).iloc[0] * 100
        cb = cot.reindex([ub]).iloc[0] * 100
        TS.append({"border": f"Telangana | {other.title()}", "ts_unit": ua, "other_unit": ub,
                   "gap_9802": e, "gap_1519": l, "change": l - e,
                   "ts_9802": ca.loc[1998:2002].mean(), "ts_1519": ca.loc[2015:2019].mean(),
                   "oth_9802": cb.loc[1998:2002].mean(), "oth_1519": cb.loc[2015:2019].mean(),
                   "ts_2010_13": ca.loc[2010:2013].mean(), "ts_2014_19": ca.loc[2014:2019].mean(),
                   "n_years_9802": int(ser.loc[1998:2002].notna().sum())})
TS = pd.DataFrame(TS)
save(TS, "h4_telangana_cotton_pairs.csv")
w(TS.round(1).to_string(index=False))
tsu_ = TS.drop_duplicates("ts_unit")
pre = (tsu_.ts_2010_13 - tsu_.ts_9802).sum()
post = (tsu_.ts_2014_19 - tsu_.ts_2010_13).sum()
w(f"Telangana border units' cotton share gain 1998-2002 -> 2010-13: {pre:.1f} pts summed over {len(tsu_)} units; "
  f"2010-13 -> 2014-19: {post:.1f}; share of the gain before the 2014 split {pre / (pre + post):.2f}")
w(tsu_[["ts_unit", "ts_9802", "ts_2010_13", "ts_2014_19", "ts_1519"]].round(1).to_string(index=False))
for bdr, d in TS.groupby("border"):
    ch = d.change.dropna().values
    # sign-flip at the level of the Telangana unit (pairs sharing a TS unit flip together)
    dd = d.dropna(subset=["change"])
    tsu = pd.factorize(dd.ts_unit)[0]
    flips = np.array([[1 if (k >> j) & 1 else -1 for j in range(tsu.max() + 1)] for k in range(2 ** (tsu.max() + 1))])
    stat = ch.mean()
    null = np.array([(f[tsu] * ch).mean() for f in flips])
    pexact = (np.abs(null) >= abs(stat) - 1e-12).mean()
    w(f"  {bdr}: {len(ch)} pairs with both windows, {tsu.max() + 1} Telangana units, {len(set(dd.other_unit))} other units; "
      f"mean change {stat:+.1f}, positive in {(ch > 0).sum()} of {len(ch)}; exact sign-flip p over TS units {pexact:.3f} "
      f"(smallest attainable {1 / len(flips) * 2:.3f})")
# timing: Telangana state-wide cotton share by year (all TS units, area-weighted)
ts_all = uy[(uy.st == "TELANGANA") & (uy.reliable == 1)].groupby("year_start").apply(
    lambda d: (d.sh_cotton * d.gca_ha).sum() / d.gca_ha.sum() * 100)
w("Telangana cotton share of GCA, all units, by year: " + ", ".join(f"{k}: {v:.0f}" for k, v in ts_all.items()))
for stn in ["KARNATAKA", "MAHARASHTRA", "ANDHRA PRADESH"]:
    s_ = uy[(uy.st == stn) & (uy.reliable == 1)].groupby("year_start").apply(
        lambda d: (d.sh_cotton * d.gca_ha).sum() / d.gca_ha.sum() * 100)
    w(f"{stn.title()} cotton share by year: " + ", ".join(f"{k}: {v:.0f}" for k, v in s_.items() if k in (1998, 2002, 2008, 2013, 2015, 2019)))

# ================================================================== H5
w("\n" + "=" * 80)
w("H5. Moran's I of changes 1998-2002 -> 2015-19")
ed = pd.read_csv(os.path.join(ROOT, "outputs", "crop_groups", "crop_group_adjacency_edges.csv"))
st_u = units.set_index("unit_key").st
in606 = set(units[units.in_606 == 1].unit_key)
h5 = []
NPERM = 499 if FAST else 999
for v in ["D1", "s_rice", "s_wheat", "s_pulses", "s_millets", "s_oilseeds"]:
    x = (upw["w1519"][v] - upw["w9802"][v]).dropna()
    x = x[x.index.isin(in606) & x.index.isin(st_u.index)]
    keys = list(x.index)
    ix = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    Wb = np.zeros((n, n))
    for a, b in zip(ed.unit_i, ed.unit_j):
        if a in ix and b in ix:
            Wb[ix[a], ix[b]] = 1
    sarr = st_u.reindex(keys).values
    same = sarr[:, None] == sarr[None, :]
    apts = (np.isin(sarr, ["ANDHRA PRADESH", "TELANGANA"])[:, None] & np.isin(sarr, ["ANDHRA PRADESH", "TELANGANA"])[None, :]) & ~same
    xv = x.values.astype(float)
    xdm = xv - pd.Series(xv).groupby(sarr).transform("mean").values
    crossW = Wb * (~same)
    crossW_noapts = Wb * (~same) * (~apts)
    border_units = crossW.sum(1) > 0
    withinW = Wb * same
    withinW_border = withinW * border_units[:, None] * border_units[None, :]
    sets = [("within-state links", withinW, xv), ("cross-border links", crossW, xv),
            ("cross-border links, AP-TS excluded", crossW_noapts, xv),
            ("within-state links among border units", withinW_border, xv),
            ("within-state links, state means removed", withinW, xdm),
            ("cross-border links, state means removed", crossW_noapts, xdm)]
    for nm, Wl, vals in sets:
        r = moran(vals, Wl, nperm=NPERM, rng=rng)
        h5.append({"var": v, "links": nm, "I": r["I"], "p_perm": r["p_perm"], "n_units": r["n"],
                   "n_links": int(Wl.sum() / 2)})
    # within-state links subsampled to the number of cross-border links (AP-TS excluded), 300 draws
    iu = np.argwhere(np.triu(withinW, 1) > 0)
    ncross = int(np.triu(crossW_noapts, 1).sum())
    Is = []
    for _ in range(100 if FAST else 300):
        sel = iu[rng.choice(len(iu), ncross, replace=False)]
        Ws = np.zeros((n, n))
        Ws[sel[:, 0], sel[:, 1]] = 1
        Ws = Ws + Ws.T
        Is.append(moran(xv, Ws, nperm=0 if False else 1, rng=rng)["I"])
    Is = np.array(Is)
    h5.append({"var": v, "links": f"within-state links, random {ncross}-link subsets (mean; 2.5/97.5 pct)",
               "I": Is.mean(), "I_lo": np.percentile(Is, 2.5), "I_hi": np.percentile(Is, 97.5),
               "n_links": ncross})
    # link-level: correlation of the pair values over links (both orders), using all-unit standardisation
    z = (xv - xv.mean()) / xv.std()
    for nm, Wl in [("within", withinW), ("cross_noapts", crossW_noapts)]:
        ii, jj = np.nonzero(np.triu(Wl, 1))
        r_ = np.corrcoef(np.r_[z[ii], z[jj]], np.r_[z[jj], z[ii]])[0, 1]
        h5.append({"var": v, "links": f"pair correlation over {nm} links", "I": r_, "n_links": len(ii)})
    # difference test: within-border-units minus cross, unit block bootstrap
    bdu = np.where(border_units)[0]
    diffs = []
    for _ in range(200 if FAST else 500):
        cnt = np.bincount(rng.choice(n, n), minlength=n).astype(float)
        out = []
        for Wl in [withinW_border, crossW_noapts]:
            Ww = Wl * cnt[:, None] * cnt[None, :]
            keep = Ww.sum(1) > 0
            if keep.sum() < 10:
                out.append(np.nan)
                continue
            Wr = Ww[np.ix_(keep, keep)]
            Wr = Wr / Wr.sum(1, keepdims=True)
            zz = xv[keep] - np.average(xv[keep], weights=cnt[keep])
            wk = cnt[keep]
            out.append((wk * zz * (Wr @ zz)).sum() / (wk * zz ** 2).sum())
        diffs.append(out[0] - out[1])
    diffs = np.array(diffs)
    h5.append({"var": v, "links": "within(border units) minus cross(no AP-TS), unit bootstrap",
               "I": np.nanmean(diffs), "I_lo": np.nanpercentile(diffs, 2.5), "I_hi": np.nanpercentile(diffs, 97.5)})
h5 = pd.DataFrame(h5)
save(h5, "h5_moran_checks.csv")
w(h5.round(3).to_string(index=False))

# ================================================================== H6 and H7: reuse script 128 definitions
w("\n" + "=" * 80)
w("H6/H7. Spread episodes (script 128 definitions, executed read-only up to its main loop)")
src = open(os.path.join(HERE, "128_peer_effects_spread.py"), encoding="utf-8").read()
cut = src.index("rng = np.random.default_rng(20261107)\nallH")
ns = {"__file__": os.path.join(HERE, "128_peer_effects_spread.py"), "__name__": "s128"}
_LOGsave = list(LOG)
exec(compile(src[:cut], "128_header", "exec"), ns)
ADJ, DIST, ST, KEYS, NU, YEARS = ns["ADJ"], ns["DIST"], ns["st"], ns["keys"], ns["NU"], ns["YEARS"]
Ucov = ns["U"].set_index("unit_key")[["rain_ann_mean", "rain_ann_cv", "elevation_mean_m", "tri_mean", "irr_share"]]
COVZ = Ucov.reindex(KEYS).values.astype(float)
COVZ = (COVZ - np.nanmean(COVZ, 0)) / np.nanstd(COVZ, 0)
COVZ = np.where(np.isfinite(COVZ), COVZ, 0.0)
CROPNAME = {"cotton": "Cotton(Lint)", "soybean": "Soyabean", "maize": "Maize", "rabi_maize": "Maize",
            "chickpea": "Gram", "moong": "Moong(Green Gram)", "rice": "Rice"}
jumps = pd.read_csv(os.path.join(ROOT, "outputs", "revision_nov8", "d0_state_jumps.csv"))
jumps["st_up"] = jumps.state.str.upper().str.replace("&", "AND", regex=False)
st11 = ns["U"].set_index("unit_key").state.reindex(KEYS).values  # 2011 state, APY reporting state pre-2014
gca = uy.pivot_table(index="unit_key", columns="year_start", values="gca_ha").reindex(KEYS)
gca = gca.reindex(columns=YEARS).values
d0 = uy.pivot_table(index="unit_key", columns="year_start", values="D0").reindex(KEYS).reindex(columns=YEARS).values


def bern_p(obs, probs, nsim=20000):
    probs = np.asarray(probs)
    sims = (rng.random((nsim, len(probs))) < probs).sum(1)
    return (1 + (sims >= obs).sum()) / (nsim + 1)


h6, h6e = [], []
for name, (col, L, flag) in ns["EP"].items():
    S = ns["grid"](col, flag)
    for mode in ["entry", "surge"]:
        base, T, status, = ns["event_years"](S, L, mode)
        G = ns["grower_matrix"](T, status)
        atrisk_all = status == "at_risk"
        # propensity of ever having the event, at-risk units, logit on covariates + baseline
        Xp = np.column_stack([np.ones(NU), COVZ, np.nan_to_num(base), np.nan_to_num(base) ** 2])
        yv = np.isfinite(T).astype(float)
        import statsmodels.api as sm
        try:
            lm = sm.Logit(yv[atrisk_all], Xp[atrisk_all]).fit_regularized(alpha=1.0, disp=0)
            prop = 1 / (1 + np.exp(-Xp @ lm.params))
        except Exception:
            prop = np.ones(NU)
        if mode == "entry":
            bedges = np.array([-1, 0.2 * L, 0.5 * L, 10])
        else:
            qq = np.nanpercentile(base[atrisk_all], [25, 50, 75])
            bedges = np.r_[-1, qq, 10]
        bbin = np.digitize(np.nan_to_num(base), bedges)
        ev = []
        for t in range(3, NY := len(YEARS)):
            y = YEARS[t]
            Gp = np.nan_to_num(G[:, t - 1]) > 0
            if not Gp.any():
                continue
            atrisk = atrisk_all & ~(np.isfinite(T) & (T < y))
            ent = atrisk & np.isfinite(T) & (T == y)
            if not ent.any():
                continue
            contig = (ADJ & Gp[None, :]).any(1)
            ar = np.where(atrisk)[0]
            for i in np.where(ent)[0]:
                cand_st = ar[ST[ar] == ST[i]]
                cand_stb = cand_st[bbin[cand_st] == bbin[i]]
                if len(cand_stb) < 3:
                    cand_stb = cand_st
                dcov = np.sqrt((((COVZ[ar] - COVZ[i]) ** 2).sum(1)) + ((np.nan_to_num(base[ar]) - np.nan_to_num(base[i])) / max(L, 1e-9)) ** 2)
                cand_nn = ar[np.argsort(dcov)[:10]]
                dcs = dcov[ST[ar] == ST[i]]
                cand_nn_st = cand_st[np.argsort(dcs)[:5]]
                pw = prop[cand_st]
                # noise flags
                s_i = S[i]
                prev = s_i[:t][np.isfinite(s_i[:t])]
                absent_before = bool(len(prev) and np.all(prev == 0))
                late = s_i[17:][np.isfinite(s_i[17:])]
                sustained = bool(len(late) and late.mean() >= (L if mode == "entry" else base[i] + L))
                g_prev = gca[i, t - 1] if np.isfinite(gca[i, t - 1]) else np.nan
                gjump = gca[i, t] / g_prev if np.isfinite(g_prev) and g_prev > 0 else np.nan
                jj = jumps[(jumps.st_up == str(st11[i]).upper()) & (jumps.year.between(y - 1, y))]
                listadd = bool(any(CROPNAME[name] in str(s).split("; ") for s in jj.new_crops))
                d0j = d0[i, t] - d0[i, t - 1] if np.isfinite(d0[i, t - 1]) else np.nan
                ev.append({"episode": name, "mode": mode, "unit_key": KEYS[i], "st": ST[i], "year": y,
                           "contig": int(contig[i]), "baseline": base[i], "share_t": s_i[t],
                           "share_tm1": s_i[t - 1], "mean_2015_19": late.mean() if len(late) else np.nan,
                           "p_state": contig[cand_st].mean(), "p_state_bin": contig[cand_stb].mean(),
                           "p_covnn_nat": contig[cand_nn].mean(), "p_covnn_state": contig[cand_nn_st].mean(),
                           "p_prop_state": (pw * contig[cand_st]).sum() / pw.sum(),
                           "absent_before": int(absent_before), "sustained": int(sustained),
                           "gca_ratio": gjump, "state_list_added": int(listadd), "d0_jump": d0j})
        ev = pd.DataFrame(ev)
        if ev.empty:
            continue
        ev["suspect"] = ((ev.absent_before == 1) | (ev.state_list_added == 1) |
                         (ev.gca_ratio.sub(1).abs() > 0.3) | (ev.sustained == 0)).astype(int)
        h6e.append(ev)
        for subset, e in [("all", ev), ("clean", ev[ev.suspect == 0])]:
            if len(e) == 0:
                continue
            rec = {"episode": name, "mode": mode, "subset": subset, "events": len(e), "obs": e.contig.mean()}
            for c in ["p_state", "p_state_bin", "p_covnn_nat", "p_covnn_state", "p_prop_state"]:
                rec[c] = e[c].mean()
                rec["pval_" + c[2:]] = bern_p(e.contig.sum(), e[c].values)
            if subset == "all":
                rec.update({"n_absent_before": int(ev.absent_before.sum()), "n_list_added": int(ev.state_list_added.sum()),
                            "n_gca_jump30": int((ev.gca_ratio.sub(1).abs() > 0.3).sum()),
                            "n_not_sustained": int((ev.sustained == 0).sum()), "n_suspect": int(ev.suspect.sum())})
            h6.append(rec)
h6 = pd.DataFrame(h6)
save(h6, "h6_wave_benchmarks.csv")
save(pd.concat(h6e), "h6_wave_events.csv")
w("share of events adjacent to a grower: observed vs benchmarks (expected share under each draw rule):")
w(h6.round(3).to_string(index=False))

# ---------------------------------------------------------------- H7
w("\n" + "=" * 80)
w("H7. Neighbour minus ring")
H = pd.read_csv(os.path.join(OUT, "q3_hazard.csv"))
dmr = H[(H.tag == "main") & (H["var"] == "nbr minus ring")]
for cl in ["unit", "state"]:
    d = dmr[dmr.cluster == cl].copy()
    d["q_bh"] = bh(d.p.values)
    d["holm"] = holm(d.p.values)
    w(f"cluster = {cl}: " + "; ".join(f"{r.episode} {r.mode} {r.b:.3f} ({r.se:.3f}) p {r.p:.3f} q {r.q_bh:.2f}"
                                       for r in d.sort_values("p").itertuples()))
    save(d, f"h7_nbr_minus_ring_{cl}.csv")
# reproduce + alternative rings for the two candidates and cotton surge
RING0 = ns["RING"].copy()
md = np.array([DIST[i, ADJ[i]].mean() if ADJ[i].any() else np.nan for i in range(NU)])
EYE = np.eye(NU, dtype=bool)
SAME = ST[:, None] == ST[None]
rings = {"0.75-1.5x (main)": RING0,
         "0.5-1.0x": (~ADJ) & (~EYE) & (DIST >= 0.5 * md[:, None]) & (DIST <= 1.0 * md[:, None]),
         "1.0-2.0x": (~ADJ) & (~EYE) & (DIST >= 1.0 * md[:, None]) & (DIST <= 2.0 * md[:, None]),
         "0.75-1.5x same state": RING0 & SAME,
         "second-order neighbours": (~ADJ) & (~EYE) & ((ADJ.astype(int) @ ADJ.astype(int)) > 0)}
h7 = []
for (name, mode) in [("chickpea", "surge"), ("soybean", "entry"), ("soybean", "surge"), ("cotton", "surge"),
                     ("maize", "surge")]:
    col, L, flag = ns["EP"][name]
    S = ns["grid"](col, flag)
    for rn, R in rings.items():
        ns["RING"] = R
        out = ns["hazard"](name, S, L, "main", mode)[4]
        for r in out:
            if r["spec"] == "nbr + ring":
                h7.append({"episode": name, "mode": mode, "ring": rn, "var": r["var"], "cluster": r["cluster"],
                           "b": r["b"], "se": r["se"], "p": r["p"], "n_entries": r["n_entries"]})
ns["RING"] = RING0
h7 = pd.DataFrame(h7)
save(h7, "h7_alternative_rings.csv")
w(h7[h7["var"] == "nbr minus ring"].pivot_table(index=["episode", "mode", "ring"], columns="cluster",
                                                 values=["b", "se", "p"]).round(3).to_string())
chk = h7[(h7.ring == "0.75-1.5x (main)") & (h7["var"] == "nbr minus ring") & (h7.cluster == "unit")].merge(
    dmr[dmr.cluster == "unit"], on=["episode", "mode"], suffixes=("", "_orig"))
w("reproduction, max |b - b_orig| = " + f"{(chk.b - chk.b_orig).abs().max():.2e}")

LOG = _LOGsave + LOG[len(_LOGsave):]
with open(os.path.join(RO, "referee_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
