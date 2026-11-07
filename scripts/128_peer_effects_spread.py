"""
128_peer_effects_spread.py

Q3. Spread episodes: does a district take up a crop after its neighbours have?

Episodes (shares of annual GCA unless stated; reliable unit-years only):
  cotton        cotton share        (the Bt cotton years; APY has no Bt/non-Bt split)
  soybean       soybean share
  maize         maize share, all seasons
  rabi_maize    rabi-season maize as share of GCA (needs a reliable rabi split)
  chickpea      gram share          (the move of chickpea into the south)
  moong         green gram share
  rice          rice share
Telangana paddy is a statewide push. State-by-year FE absorb it, so it is handled as
a border contrast (script 127) and as chart data (script 129).

Two event types, both run for every episode:
Entry (mode "entry"): a unit is "at risk" if its 1998-2000 mean share is below the
  threshold L. Its event year T_i is the first year t >= 2001 with s_t >= L, the next
  reliable value >= L, and the mean of all values from t on >= L. Units at or above L
  in 1998-2000 are established growers from the start.
Surge (mode "surge"): every unit with a baseline is at risk; the event is the first
  year the share is L above the unit's own 1998-2000 mean, by the same sustained rule.
  Surge fits Bt cotton, which grew mostly inside districts that already grew cotton.
L = 5 points of GCA (2 for rabi maize); sensitivity at 0.6x and 2x.
"Grower" in the exposure terms = has had its event by t-1 (or is established).

Hazard (linear probability, risk set = at-risk units before their event, t = 2001-2019):
  event_it = b * Nbr_i,t-1 + p1 * Ring_i,t-1 + p2 * SameState_i,t-1
             + baseline share + rain mean, rain CV, elevation, TRI, irrigated share
             + state-by-year FE + e
  Nbr       = share of adjacent units that are growers at t-1
  Ring      = same, for non-adjacent units whose centroid is 0.75-1.5x the unit's mean
              distance to its own neighbours (any state): about the same distance, no
              shared border
  SameState = same, for all same-state units that are not adjacent
  SEs clustered on the 2011 parent unit, and on state. Nbr is also split into
  same-state and other-state neighbours.

Wave or jump: for each event, is the unit contiguous to a grower at t-1, and how far
is the nearest grower? Compared with 999 draws that reassign each year's events at
random among that year's at-risk units (nationally, and within state keeping each
state's count). Events in a year with no prior grower anywhere are left out.

Run: C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/128_peer_effects_spread.py
Outputs (outputs/revision_nov8/peer_effects/):
  q3_entry_years.csv, q3_hazard.csv, q3_wave.csv, q3_episode_summary.csv,
  q3_entries_detail.csv, q3_unit_year_status.csv.gz, q3_log.txt
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pe_common import OUT, ROOT, load, demean  # noqa: E402

warnings.filterwarnings("ignore")
LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


units, uy, up, P = load()
uy["sh_maize"] = uy.share_gca_maize
EP = {"cotton": ("sh_cotton", 0.05, "reliable"), "soybean": ("sh_soybean", 0.05, "reliable"),
      "maize": ("sh_maize", 0.05, "reliable"), "rabi_maize": ("sh_maize_rabi", 0.02, "reliable_rabi"),
      "chickpea": ("sh_gram", 0.05, "reliable"), "moong": ("sh_moong", 0.05, "reliable"),
      "rice": ("sh_rice", 0.05, "reliable")}
YEARS = list(range(1998, 2020))
NY = len(YEARS)

U = units[(units.in_606 == 1) & units.cx.notna()].reset_index(drop=True)
keys = list(U.unit_key)
ix = {k: i for i, k in enumerate(keys)}
NU = len(keys)
xy = U[["cx", "cy"]].values
DIST = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)) / 1000
ed = pd.read_csv(os.path.join(ROOT, "outputs", "crop_groups", "crop_group_adjacency_edges.csv"))
ADJ = np.zeros((NU, NU), bool)
for a, b in zip(ed.unit_i, ed.unit_j):
    if a in ix and b in ix:
        ADJ[ix[a], ix[b]] = True
st = U.st.values
SAME = st[:, None] == st[None]
EYE = np.eye(NU, dtype=bool)
md = np.array([DIST[i, ADJ[i]].mean() if ADJ[i].any() else np.nan for i in range(NU)])
RING = (~ADJ) & (~EYE) & (DIST >= 0.75 * md[:, None]) & (DIST <= 1.5 * md[:, None])
SSNA = SAME & (~ADJ) & (~EYE)
w(f"units {NU}; mean neighbours {ADJ.sum(1).mean():.2f} at {np.nanmean(md):.0f} km; ring set mean size "
  f"{RING.sum(1).mean():.1f} (same state {(RING & SAME).sum(1).mean():.1f}) at "
  f"{np.nanmean([DIST[i, RING[i]].mean() for i in range(NU) if RING[i].any()]):.0f} km; "
  f"same-state non-adjacent mean size {SSNA.sum(1).mean():.1f}")
covs = U.set_index("unit_key")[["rain_ann_mean", "rain_ann_cv", "elevation_mean_m", "tri_mean", "irr_share"]]


def grid(col, flag):
    M = np.full((NU, NY), np.nan)
    d = uy[(uy[flag] == 1) & uy.unit_key.isin(ix)]
    M[d.unit_key.map(ix).values, (d.year_start - 1998).values] = d[col].values
    return M


def event_years(S, L, mode):
    base = np.nanmean(S[:, :3], axis=1)
    T = np.full(NU, np.nan)
    if mode == "entry":
        status = np.where(base >= L, "established", np.where(np.isfinite(base), "at_risk", "no_baseline"))
        thr = np.full(NU, L)
    else:
        status = np.where(np.isfinite(base), "at_risk", "no_baseline")
        thr = base + L
    for i in np.where(status == "at_risk")[0]:
        s = S[i]
        for t in range(3, NY):
            if not np.isfinite(s[t]) or s[t] < thr[i]:
                continue
            nxt = s[t + 1:][np.isfinite(s[t + 1:])]
            if len(nxt) == 0 or nxt[0] < thr[i]:
                continue
            rest = s[t:][np.isfinite(s[t:])]
            if rest.mean() >= thr[i]:
                T[i] = YEARS[t]
                break
    return base, T, status


def grower_matrix(T, status):
    G = np.zeros((NU, NY))
    G[status == "established"] = 1
    for i in np.where(np.isfinite(T))[0]:
        G[i, YEARS.index(int(T[i])):] = 1
    G[status == "no_baseline"] = np.nan
    return G


def setmean(G, M):
    Gf = np.nan_to_num(G)
    ok = np.isfinite(G).astype(float)
    num = M.astype(float) @ Gf
    den = M.astype(float) @ ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


CTRL = ["base", "rain_ann_mean", "rain_ann_cv", "elevation_mean_m", "tri_mean", "irr_share"]
SPECS = [("nbr only", ["nbr"]), ("nbr + ring", ["nbr", "ring"]),
         ("nbr + ring + same-state", ["nbr", "ring", "samestate"]),
         ("ring only", ["ring"]), ("same-state only", ["samestate"]),
         ("nbr split by border", ["nbr_same", "nbr_other", "ring"])]


def hazard(name, S, L, tag, mode):
    base, T, status = event_years(S, L, mode)
    G = grower_matrix(T, status)
    N, R, Q = setmean(G, ADJ), setmean(G, RING), setmean(G, SSNA)
    Nss, Nxs = setmean(G, ADJ & SAME), setmean(G, ADJ & ~SAME)
    rows = []
    for i in np.where(status == "at_risk")[0]:
        for t in range(3, NY):
            y = YEARS[t]
            if np.isfinite(T[i]) and y > T[i]:
                break
            rows.append({"unit_key": keys[i], "st": st[i], "year": y,
                         "entry": int(T[i] == y) if np.isfinite(T[i]) else 0,
                         "nbr": N[i, t - 1], "nbr_same": Nss[i, t - 1], "nbr_other": Nxs[i, t - 1],
                         "ring": R[i, t - 1], "samestate": Q[i, t - 1], "base": base[i]})
    d = pd.DataFrame(rows).join(covs, on="unit_key")
    out = []
    for spec, xs in SPECS:
        dd = d.dropna(subset=["entry"] + xs + CTRL).copy()
        sy = pd.factorize(dd.st + "_" + dd.year.astype(str))[0]
        keep = np.bincount(sy)[sy] > 1
        dd, sy = dd[keep], pd.factorize(sy[keep])[0]
        if dd.entry.sum() < 5:
            continue
        Z = demean(dd[["entry"] + xs + CTRL].values.astype(float), [sy])
        base_rec = {"episode": name, "mode": mode, "threshold": L, "tag": tag, "spec": spec,
                    "n": len(dd), "n_units": dd.unit_key.nunique(), "n_entries": int(dd.entry.sum()),
                    "mean_entry": dd.entry.mean()}
        for cl in ["unit", "state"]:
            g = pd.factorize(dd.unit_key if cl == "unit" else dd.st)[0]
            r = sm.OLS(Z[:, 0], Z[:, 1:]).fit(cov_type="cluster", cov_kwds={"groups": g})
            for k, x in enumerate(xs):
                out.append({**base_rec, "var": x, "cluster": cl, "b": r.params[k], "se": r.bse[k],
                            "p": r.pvalues[k], "n_clusters": len(np.unique(g)), "sd_x": dd[x].std(),
                            "mean_x_at_event": dd.loc[dd.entry == 1, x].mean(),
                            "mean_x_no_event": dd.loc[dd.entry == 0, x].mean()})
            if spec == "nbr + ring":
                c = np.zeros(Z.shape[1] - 1)
                c[0], c[1] = 1, -1
                tt = r.t_test(c)
                out.append({**base_rec, "var": "nbr minus ring", "cluster": cl, "b": float(np.squeeze(tt.effect)),
                            "se": float(np.squeeze(tt.sd)), "p": float(np.squeeze(tt.pvalue)),
                            "n_clusters": len(np.unique(g)), "sd_x": np.nan})
    return base, T, status, G, out


def wave(name, mode, T, status, G, rng, ndraw=999):
    rows = []
    obs_c, obs_d = [], []
    sims_nat, sims_st = np.zeros(ndraw), np.zeros(ndraw)
    n_ent, risk_contig = 0, []
    for t in range(3, NY):
        y = YEARS[t]
        Gp = np.nan_to_num(G[:, t - 1]) > 0
        if not Gp.any():
            continue
        atrisk = (status == "at_risk") & ~(np.isfinite(T) & (T < y))
        ent = atrisk & np.isfinite(T) & (T == y)
        if not ent.any():
            continue
        contig = (ADJ & Gp[None, :]).any(1)
        dmin = np.where(Gp[None, :], DIST, np.inf).min(1)
        for i in np.where(ent)[0]:
            jn = np.argmin(np.where(Gp, DIST[i], np.inf))
            rows.append({"episode": name, "mode": mode, "unit_key": keys[i], "st": st[i], "year": y,
                         "contiguous_to_grower": int(contig[i]), "km_to_nearest_grower": dmin[i],
                         "nearest_grower": keys[jn], "nearest_grower_same_state": int(st[jn] == st[i])})
        obs_c.append(contig[ent].sum())
        obs_d.append(dmin[ent])
        risk_contig.append(contig[atrisk].mean())
        n_ent += ent.sum()
        ar = np.where(atrisk)[0]
        k = ent.sum()
        for s_ in range(ndraw):
            sims_nat[s_] += contig[rng.choice(ar, k, replace=False)].sum()
        for s_ in range(ndraw):
            tot = 0
            for sname in np.unique(st[ent]):
                arS = ar[st[ar] == sname]
                kS = int((ent & (st == sname)).sum())
                tot += contig[rng.choice(arS, kS, replace=False)].sum()
            sims_st[s_] += tot
    if n_ent == 0:
        return pd.DataFrame(rows), {}
    oc = sum(obs_c)
    dists = np.concatenate(obs_d)
    return pd.DataFrame(rows), {
        "episode": name, "mode": mode, "events_with_prior_grower": int(n_ent),
        "share_contiguous": oc / n_ent,
        "share_contiguous_random_national": sims_nat.mean() / n_ent,
        "p_national": (1 + (sims_nat >= oc).sum()) / (ndraw + 1),
        "share_contiguous_random_within_state": sims_st.mean() / n_ent,
        "p_within_state": (1 + (sims_st >= oc).sum()) / (ndraw + 1),
        "median_km_nearest_grower": float(np.median(dists)),
        "share_jumps_gt150km": float((dists > 150).mean()),
        "share_atrisk_contiguous_mean": float(np.mean(risk_contig))}


rng = np.random.default_rng(20261107)
allH, allW, allE, waveS, summ, status_rows = [], [], [], [], [], []
for name, (col, L, flag) in EP.items():
    S = grid(col, flag)
    for mode in ["entry", "surge"]:
        for mult, tag in [(1.0, "main"), (0.6, "low threshold"), (2.0, "high threshold")]:
            base, T, status, G, out = hazard(name, S, L * mult, tag, mode)
            allH += out
            if tag != "main":
                continue
            wr, ws = wave(name, mode, T, status, G, rng)
            allW.append(wr)
            if ws:
                waveS.append(ws)
            e = pd.DataFrame({"episode": name, "mode": mode, "unit_key": keys, "st": st, "baseline_share": base,
                              "status_1998_2000": status, "event_year": T, "threshold": L,
                              "share_2015_19": np.nanmean(S[:, 17:], axis=1)})
            allE.append(e)
            for i in range(NU):
                for t in range(NY):
                    status_rows.append((name, mode, keys[i], YEARS[t], S[i, t], G[i, t]))
            ent = e[e.event_year.notna()]
            summ.append({"episode": name, "mode": mode, "threshold": L,
                         "established_1998_2000": int((status == "established").sum()),
                         "at_risk": int((status == "at_risk").sum()), "events": len(ent),
                         "median_event_year": ent.event_year.median() if len(ent) else np.nan,
                         "events_by_period": "; ".join(f"{a}-{b}: {int(ent.event_year.between(a, b).sum())}"
                                                        for a, b in [(2001, 2004), (2005, 2009), (2010, 2014), (2015, 2019)]),
                         "states_with_events": ent.st.nunique(),
                         "top_states": "; ".join(f"{k.title()} {v}" for k, v in ent.st.value_counts().head(4).items())})
            w(f"{name} [{mode}]: L={L}, established {summ[-1]['established_1998_2000']}, at risk "
              f"{summ[-1]['at_risk']}, events {len(ent)} ({summ[-1]['top_states']}); {summ[-1]['events_by_period']}")

H = pd.DataFrame(allH)
H.to_csv(os.path.join(OUT, "q3_hazard.csv"), index=False)
pd.concat(allE).to_csv(os.path.join(OUT, "q3_entry_years.csv"), index=False)
pd.concat(allW).to_csv(os.path.join(OUT, "q3_entries_detail.csv"), index=False)
WS = pd.DataFrame(waveS)
WS.to_csv(os.path.join(OUT, "q3_wave.csv"), index=False)
pd.DataFrame(summ).to_csv(os.path.join(OUT, "q3_episode_summary.csv"), index=False)
pd.DataFrame(status_rows, columns=["episode", "mode", "unit_key", "year", "share", "grower"]).to_csv(
    os.path.join(OUT, "q3_unit_year_status.csv.gz"), index=False, compression="gzip")

w("\nHazard, main thresholds (coef = change in yearly event probability for 0 -> 1 share of growers):")
m = H[(H.tag == "main") & H.spec.isin(["nbr only", "nbr + ring", "nbr split by border"])]
w(m.pivot_table(index=["mode", "episode", "spec", "var"], columns="cluster", values=["b", "se"]).round(3).to_string())
w(H[H.tag == "main"].groupby(["mode", "episode"])[["n", "n_units", "n_entries", "mean_entry"]].first().to_string())
w("\nSensitivity (nbr + ring, unit clusters):")
s = H[(H.spec == "nbr + ring") & (H.cluster == "unit")]
w(s.pivot_table(index=["mode", "episode", "var"], columns="tag", values=["b", "se", "n_entries"]).round(3).to_string())
w("\nWave or jump:")
w(WS.round(3).to_string(index=False))

with open(os.path.join(OUT, "q3_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
