"""
100_referee_neighbours_checks.py

Referee checks on 99_crop_groups_neighbours.py (Q4 neighbour spread). Reuses that
script's data build (everything above its estimation loop) unchanged, then:
  A. reproduce the main lag coefficients
  B. power of the lag-vs-lead "placebo": simulate genuine neighbour spread on the real
     adjacency graph and real sample mask, run the same specs
  C. spatially robust inference: two-way cluster (unit + state-year), state cluster
     with restricted wild cluster bootstrap
  D. finer regional shocks: sub-state spatial region x year FE (k-means on centroids)
Outputs: outputs/crop_groups/referee_neighbours_log.txt, referee_neighbours_estimates.csv
No APCNF/PMDS/RySS data.
Run: C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/100_referee_neighbours_checks.py
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "scripts", "99_crop_groups_neighbours.py")
src = open(SRC, encoding="utf-8").read().split("rows = []")[0]
G = {"__file__": SRC, "__name__": "ref"}
exec(compile(src, "99_head", "exec"), G)
import statsmodels.api as sm

S, GROUPS, nb_idx, NU, NY, OUT = G["S"], G["GROUPS"], G["nb_idx"], G["NU"], G["NY"], G["OUT"]
diff, shift, wavg, fit, demean = G["diff"], G["shift"], G["wavg"], G["fit"], G["demean"]
RAIN, RAIN_NAMES, UI, YI, SI, s606, xy, st_arr = (G["RAIN"], G["RAIN_NAMES"], G["UI"], G["YI"],
                                                   G["SI"], G["s606"], G["xy"], G["st_arr"])
L = []


def lg(s=""):
    print(s)
    L.append(str(s))


def rc():
    return {k: RAIN[k] for k in RAIN_NAMES}


out = []

# ---------------- A. reproduce
lg("A. reproduce M2 main WD_l1")
for g in GROUPS:
    D = diff(S[g]); WD = wavg(D, nb_idx)
    o = fit(D, {"WD_l1": shift(WD, 1), "own_l1": shift(D, 1), **rc()})
    lg(f"  {g:9s} {o['WD_l1_b']:.4f} ({o['WD_l1_se']:.4f}) p={o['WD_l1_p']:.3f} n={o['n']}")
    out.append({"block": "A", "group": g, "spec": "main", "b": o["WD_l1_b"], "se": o["WD_l1_se"],
                "p": o["WD_l1_p"], "n": o["n"]})

# ---------------- B. simulation of true spread
lg("\nB. simulated spread on the real graph and real pulses sample mask; true b on W dS(t-1)")
mask = np.isfinite(diff(S["pulses"]))


def Wm(v):
    return wavg(v[:, None], nb_idx, min_share=0.0)[:, 0]


def sim(b, kind, rng):
    T = NY + 30
    if kind == "iid":            # dS_t = b W dS_{t-1} + e_t
        Dm = np.zeros((NU, T))
        for t in range(1, T):
            Dm[:, t] = b * Wm(Dm[:, t - 1]) + rng.normal(0, 0.04, NU)
        Dm = Dm[:, -NY:]
    else:                        # latent level with spread, plus iid measurement noise
        X = np.zeros((NU, T)); dX = np.zeros((NU, T))
        for t in range(1, T):
            dX[:, t] = b * Wm(dX[:, t - 1]) + rng.normal(0, 0.025, NU)
            X[:, t] = X[:, t - 1] + dX[:, t]
        Dm = diff(X[:, -NY:] + rng.normal(0, 0.02, (NU, NY)))
    return np.where(mask, Dm, np.nan)


rng = np.random.default_rng(7)
for kind in ["iid", "noisy_level"]:
    for b in [0.0, 0.07, 0.25]:
        acc = []
        for r in range(30):
            Dm = sim(b, kind, rng); WD = wavg(Dm, nb_idx)
            o1 = fit(Dm, {"WD_l1": shift(WD, 1), "own_l1": shift(Dm, 1)})
            o2 = fit(Dm, {"WD_f1": shift(WD, -1), "own_f1": shift(Dm, -1)},
                     extra_mask=np.isfinite(shift(WD, 1)) & np.isfinite(shift(Dm, 1)))
            o3 = fit(Dm, {"WD_l1": shift(WD, 1), "WD_f1": shift(WD, -1),
                          "own_l1": shift(Dm, 1), "own_f1": shift(Dm, -1)})
            acc.append([o1["WD_l1_b"], o2["WD_f1_b"], o3["WD_l1_b"] - o3["WD_f1_b"]])
        a = np.array(acc)
        lg(f"  {kind:11s} true b={b:.2f}: lag {a[:, 0].mean():.3f} (sd {a[:, 0].std():.3f}); "
           f"mirrored lead {a[:, 1].mean():.3f} (sd {a[:, 1].std():.3f}); joint lag-lead {a[:, 2].mean():.3f}")
        out.append({"block": "B", "group": kind, "spec": f"sim b={b}", "b": a[:, 0].mean(),
                    "se": a[:, 0].std(), "p": np.nan, "n": 30, "lead_mean": a[:, 1].mean(),
                    "diff_mean": a[:, 2].mean()})

# ---------------- C. inference robust to spatial correlation
lg("\nC. inference: two-way cluster unit + state-year; state cluster restricted wild bootstrap (Webb, 999)")
SY0 = G["SY"]


def design(y, Xd):
    names = list(Xd)
    Z = np.column_stack([y.ravel()] + [Xd[k].ravel() for k in names])
    m = np.isfinite(Z).all(1) & s606[UI]
    Z = Z[m]; ui, sy, si = UI[m], SY0[m], SI[m]
    Zd = demean(Z, [pd.factorize(ui)[0], pd.factorize(sy)[0]])
    return Zd[:, 0], Zd[:, 1:], ui, sy, si


webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])


def wcr(y, X, cl, k=0, B=999, rng=np.random.default_rng(1)):
    cid = pd.factorize(cl)[0]; ng = cid.max() + 1
    full = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": cid})
    t0 = full.params[k] / full.bse[k]
    rr = sm.OLS(y, np.delete(X, k, 1)).fit()
    ts = []
    for _ in range(B):
        yb = rr.fittedvalues + rr.resid * rng.choice(webb, ng)[cid]
        fb = sm.OLS(yb, X).fit(cov_type="cluster", cov_kwds={"groups": cid})
        ts.append(fb.params[k] / fb.bse[k])
    return full.params[k], full.bse[k], np.mean(np.abs(ts) >= abs(t0))


for g in ["pulses", "millets", "oilseeds", "maize", "other"]:
    D = diff(S[g]); WD = wavg(D, nb_idx)
    y, X, ui, sy, si = design(D, {"WD_l1": shift(WD, 1), "own_l1": shift(D, 1), **rc()})
    tw = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={
        "groups": np.column_stack([pd.factorize(ui)[0], pd.factorize(sy)[0]])})
    b, se_s, pb = wcr(y, X, si)
    lg(f"  {g:9s} b={tw.params[0]:.4f}  two-way SE {tw.bse[0]:.4f} p={tw.pvalues[0]:.3f} | "
       f"state SE {se_s:.4f}, wild bootstrap p={pb:.3f}  (n={len(y)}; FE kept singletons, so n can differ by a few)")
    out.append({"block": "C", "group": g, "spec": "two-way unit+state-year", "b": tw.params[0],
                "se": tw.bse[0], "p": tw.pvalues[0], "n": len(y)})
    out.append({"block": "C", "group": g, "spec": "state cluster wild bootstrap", "b": b, "se": se_s,
                "p": pb, "n": len(y)})

# ---------------- D. sub-state region x year FE
lg("\nD. sub-state spatial regions (k-means on centroids within state, about 6 units each) x year FE")
from scipy.cluster.vq import kmeans2
reg = np.zeros(NU, int); nxt = 0
for s in np.unique(st_arr):
    ii = np.where(st_arr == s)[0]
    k = max(1, int(round(len(ii) / 6)))
    if k == 1:
        reg[ii] = nxt; nxt += 1; continue
    _, lab = kmeans2(xy[ii] / 1000.0, k, seed=3, minit="++")
    reg[ii] = nxt + pd.factorize(lab)[0]; nxt += k
lg(f"  regions: {len(np.unique(reg))}")
nb_same = np.mean([np.mean(reg[nb_idx[i]] == reg[i]) for i in range(NU) if s606[i] and len(nb_idx[i])])
lg(f"  share of neighbours in the same region: {nb_same:.2f}")
G["SY"] = reg[UI] * NY + YI          # fit() reads SY from the exec namespace
for g in ["pulses", "millets", "oilseeds", "maize", "other"]:
    D = diff(S[g]); WD = wavg(D, nb_idx)
    o = fit(D, {"WD_l1": shift(WD, 1), "own_l1": shift(D, 1), **rc()})
    o2 = fit(D, {"WD_f1": shift(WD, -1), "own_f1": shift(D, -1), **rc()},
             extra_mask=np.isfinite(shift(WD, 1)) & np.isfinite(shift(D, 1)))
    lg(f"  {g:9s} lag {o['WD_l1_b']:.4f} ({o['WD_l1_se']:.4f}) p={o['WD_l1_p']:.3f} | "
       f"mirrored lead {o2['WD_f1_b']:.4f} ({o2['WD_f1_se']:.4f}) p={o2['WD_f1_p']:.3f} n={o['n']}")
    out.append({"block": "D", "group": g, "spec": "region-year FE lag", "b": o["WD_l1_b"],
                "se": o["WD_l1_se"], "p": o["WD_l1_p"], "n": o["n"]})
    out.append({"block": "D", "group": g, "spec": "region-year FE mirrored lead", "b": o2["WD_f1_b"],
                "se": o2["WD_f1_se"], "p": o2["WD_f1_p"], "n": o2["n"]})
G["SY"] = SY0

pd.DataFrame(out).to_csv(os.path.join(OUT, "referee_neighbours_estimates.csv"), index=False)
with open(os.path.join(OUT, "referee_neighbours_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))
