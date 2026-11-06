"""
99_crop_groups_neighbours.py

Q4 Neighbour spread. Does a district's change in a crop group's share of gross
cropped area follow its neighbours' change of the year before?

    dS_it = b * W dS_i,t-1 + r * dS_i,t-1 + rain controls + a_i + d_state,t + e_it

dS is the year-on-year change in the share of annual gross cropped area (share_gca_<g>).
W is the row-standardised queen adjacency (500 m tolerance, script 97 adjacency).
a_i = 2011-parent unit FE, d_state,t = state-by-year FE.
SEs clustered on the 2011 parent unit (605 groups in the 606 sample); state clustering
reported as a robustness check.

Placebos
  lead      : neighbours' FUTURE change W dS_i,t+1
  distance  : same-state, NON-adjacent units whose centroid distance is within
              0.75-1.5x the unit's mean centroid distance to its own neighbours
  2nd order : neighbours of neighbours that are not neighbours, any state
  permutation: 200 draws giving each unit as many random same-state non-adjacent
              "neighbours" as it has real ones

Inputs (all built earlier, no APCNF/PMDS data):
  outputs/crop_groups/crop_group_panel_annual.csv, crop_group_panel.csv
  outputs/crop_groups/crop_group_adjacency_edges.csv, crop_group_polygon_match.csv
  outputs/crop_groups/crop_group_child_map.csv, rainfall_district_year.csv
  docs/data/districts.geojson
Outputs: outputs/crop_groups/neighbours_*.csv, neighbours_coefplot.png,
         neighbours_log.txt (results note neighbours_results.md is written by hand
         from these tables)

Run: C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/99_crop_groups_neighbours.py
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "crop_groups")
GROUPS = ["rice", "wheat", "maize", "millets", "pulses", "oilseeds", "cotton",
          "sugarcane", "other"]
HEAD = ["pulses", "millets", "oilseeds"]
RNG = np.random.default_rng(20261107)
LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


# ----------------------------------------------------------------- panel
pa = pd.read_csv(os.path.join(OUT, "crop_group_panel_annual.csv"))
pa = pa[pa.season == "annual"].copy()
units_all = sorted(pa.unit_key.unique())
uix = {u: i for i, u in enumerate(units_all)}
years = list(range(pa.year_start.min(), pa.year_start.max() + 1))
NU, NY = len(units_all), len(years)
w(f"units {NU}, years {years[0]}-{years[-1]} ({NY})")

state_of = pa.drop_duplicates("unit_key").set_index("unit_key").state
in606 = pa.drop_duplicates("unit_key").set_index("unit_key").in_606


def grid(df, col, valid=None):
    """unit x year matrix of a column; NaN where absent or invalid."""
    M = np.full((NU, NY), np.nan)
    d = df if valid is None else df[valid]
    M[d.unit_key.map(uix).values, (d.year_start - years[0]).values] = d[col].values
    return M


valid_ann = pa.flag_annual_unreliable == 0
S = {g: grid(pa, f"share_gca_{g}", valid_ann) for g in GROUPS}       # flagged -> NaN
S_raw = {g: grid(pa, f"share_gca_{g}") for g in GROUPS}              # no flag filter
LA = {g: grid(pa, f"ln_area_ha_{g}", valid_ann) for g in GROUPS}

# seasonal shares (kharif, rabi) of that season's area, season split reliable only
pf = pd.read_csv(os.path.join(OUT, "crop_group_panel.csv"),
                 usecols=["unit_key", "year_start", "season", "season_area_ha",
                          "flag_season_unreliable", "flag_annual_unreliable"]
                 + [f"share_season_{g}" for g in GROUPS])
SS = {}
for sea in ["kharif", "rabi"]:
    d = pf[(pf.season == sea)]
    ok = (d.flag_season_unreliable == 0) & (d.flag_annual_unreliable == 0) & (d.season_area_ha > 0)
    for g in GROUPS:
        SS[(sea, g)] = grid(d, f"share_season_{g}", ok)


def diff(M, k=1):
    D = np.full_like(M, np.nan)
    D[:, k:] = M[:, k:] - M[:, :-k]
    return D


def shift(M, k):
    """value at t-k placed at t (k>0 lag, k<0 lead)."""
    R = np.full_like(M, np.nan)
    if k > 0:
        R[:, k:] = M[:, :-k]
    elif k < 0:
        R[:, :k] = M[:, -k:]
    else:
        R = M.copy()
    return R


# ----------------------------------------------------------------- neighbour sets
ed = pd.read_csv(os.path.join(OUT, "crop_group_adjacency_edges.csv"))
NB = {u: set() for u in units_all}
for a, b in zip(ed.unit_i, ed.unit_j):
    if a in NB and b in NB:
        NB[a].add(b)
        NB[b].add(a)

# centroids of dissolved unit polygons, EPSG 7755 metres
gj = gpd.read_file(os.path.join(ROOT, "docs", "data", "districts.geojson"))[["n", "s", "geometry"]]
pm = pd.read_csv(os.path.join(OUT, "crop_group_polygon_match.csv"))[["n", "s", "unit_key"]]
gj = gj.merge(pm, on=["n", "s"], how="left")
gj = gj[gj.unit_key.notna()]
gj["geometry"] = gj.geometry.buffer(0)
U = gj.dissolve(by="unit_key", as_index=False).to_crs(7755)
cen = U.set_index("unit_key").geometry.centroid
xy = np.array([[cen[u].x, cen[u].y] if u in cen.index else [np.nan, np.nan] for u in units_all])
DIST = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) / 1000.0  # km
w(f"units with centroid: {np.isfinite(xy[:, 0]).sum()} of {NU}")

nb_idx = [np.array(sorted(uix[v] for v in NB[u]), dtype=int) for u in units_all]
st_arr = np.array([state_of[u] for u in units_all])
A = np.zeros((NU, NU), bool)
for i, nb in enumerate(nb_idx):
    A[i, nb] = True

plc_idx, so_idx, nb_meand = [], [], np.full(NU, np.nan)
for i in range(NU):
    nb = nb_idx[i]
    if len(nb) == 0:
        plc_idx.append(np.array([], int)); so_idx.append(np.array([], int)); continue
    md = DIST[i, nb].mean()
    nb_meand[i] = md
    cand = np.where((st_arr == st_arr[i]) & (~A[i]) & (np.arange(NU) != i)
                    & (DIST[i] >= 0.75 * md) & (DIST[i] <= 1.5 * md))[0]
    plc_idx.append(cand)
    so = set()
    for j in nb:
        so.update(nb_idx[j].tolist())
    so.discard(i)
    so = np.array(sorted(x for x in so if not A[i, x]), dtype=int)
    so_idx.append(so)

s606 = np.array([in606[u] == 1 for u in units_all])
w(f"mean neighbours (606 units): {np.mean([len(nb_idx[i]) for i in range(NU) if s606[i]]):.2f}")
w(f"mean centroid distance to neighbours (606 units): {np.nanmean(nb_meand[s606]):.1f} km")
npl = np.array([len(p) for p in plc_idx])
w(f"distance-placebo set: 606 units with >=1 placebo unit {int((npl[s606] > 0).sum())}, "
  f"mean size {npl[s606 & (npl > 0)].mean():.2f}, mean distance "
  f"{np.mean([DIST[i, plc_idx[i]].mean() for i in range(NU) if s606[i] and len(plc_idx[i])]):.1f} km")
nso = np.array([len(p) for p in so_idx])
w(f"second-order set: mean size {nso[s606].mean():.2f}")


def wavg(M, sets, min_share=0.5):
    """mean over the set of the non-missing values; NaN if fewer than min_share present."""
    R = np.full_like(M, np.nan)
    for i, idx in enumerate(sets):
        if len(idx) == 0:
            continue
        sub = M[idx]
        n = np.isfinite(sub).sum(0)
        m = np.nanmean(np.where(np.isfinite(sub), sub, np.nan), axis=0) if True else None
        ok = n >= max(1, int(np.ceil(min_share * len(idx))))
        R[i, ok] = m[ok]
    return R


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------------- rainfall to units
rf = pd.read_csv(os.path.join(OUT, "rainfall_district_year.csv"),
                 usecols=["district_key", "year", "rain_jjas", "qc_jjas", "rain_jfmam_next",
                          "qc_jfmam_next", "area_km2"])
cm = pd.read_csv(os.path.join(OUT, "crop_group_child_map.csv"))[["div_key", "unit_key"]]
rf = rf.merge(cm, left_on="district_key", right_on="div_key", how="inner")
rf = rf.drop_duplicates(["unit_key", "district_key", "year"])
rf["qc"] = ((rf.qc_jjas == 1) | rf.rain_jjas.isna()).astype(int)
rf["qcr"] = ((rf.qc_jfmam_next == 1) | rf.rain_jfmam_next.isna()).astype(int)


def agg(d):
    wgt = d.area_km2.fillna(1.0)
    return pd.Series({
        "rain_jjas": np.average(d.rain_jjas.fillna(0), weights=wgt) if d.rain_jjas.notna().all() else np.nan,
        "rain_rabi": np.average(d.rain_jfmam_next.fillna(0), weights=wgt) if d.rain_jfmam_next.notna().all() else np.nan,
        "qc": d.qc.max(), "qcr": d.qcr.max()})


ru = rf.groupby(["unit_key", "year"])[["rain_jjas", "rain_jfmam_next", "qc", "qcr", "area_km2"]].apply(agg).reset_index()
for v, q in [("rain_jjas", "qc"), ("rain_rabi", "qcr")]:
    ru.loc[ru[q] == 1, v] = np.nan
    gstat = ru.groupby("unit_key")[v].agg(["mean", "std"])
    ru = ru.merge(gstat.rename(columns={"mean": "m_" + v, "std": "s_" + v}), on="unit_key")
    ru["z_" + v] = (ru[v] - ru["m_" + v]) / ru["s_" + v]
ru = ru.rename(columns={"year": "year_start"})
ru = ru[ru.year_start.between(years[0], years[-1]) & ru.unit_key.isin(uix)]
ZJ = grid(ru, "z_rain_jjas")
ZR = grid(ru, "z_rain_rabi")
w(f"rain unit-years (606 units, 1997-2019) with a usable monsoon value: "
  f"{np.isfinite(ZJ[s606]).sum()} of {s606.sum() * NY}")
RAIN = {
    "zj": ZJ, "zj2": ZJ ** 2, "zr": ZR, "zj_l1": shift(ZJ, 1), "zr_l1": shift(ZR, 1),
    "Wzj": wavg(ZJ, nb_idx), "Wzr": wavg(ZR, nb_idx),
    "Wzj_l1": shift(wavg(ZJ, nb_idx), 1), "Wzr_l1": shift(wavg(ZR, nb_idx), 1),
}
RAIN_NAMES = list(RAIN)

# ----------------------------------------------------------------- estimation
UI = np.repeat(np.arange(NU), NY)
YI = np.tile(np.arange(NY), NU)
state_codes = pd.factorize(st_arr)[0]
SI = state_codes[UI]
SY = SI * NY + YI


def demean(X, groups, tol=1e-10, maxit=500):
    """alternating projections over several FE groupings (columns of X)."""
    X = X.copy()
    for _ in range(maxit):
        Xold = X.copy()
        for g in groups:
            # subtract group means
            cnt = np.bincount(g)
            for c in range(X.shape[1]):
                s = np.bincount(g, weights=X[:, c])
                X[:, c] -= (s / np.where(cnt == 0, 1, cnt))[g]
        if np.max(np.abs(X - Xold)) < tol:
            break
    return X


def fit(y, Xd, extra_mask=None, fe="unit+sy", cluster="unit", label=""):
    """y: NU x NY matrix; Xd: dict name -> NU x NY. Sample: 606 units, all finite."""
    names = list(Xd)
    cols = [y.ravel()] + [Xd[k].ravel() for k in names]
    Z = np.column_stack(cols)
    m = np.isfinite(Z).all(1) & s606[UI]
    if extra_mask is not None:
        m &= extra_mask.ravel()
    Z = Z[m]
    groups = []
    ui, sy, yi = UI[m], SY[m], YI[m]
    if "unit" in fe:
        groups.append(pd.factorize(ui)[0])
    if "sy" in fe:
        groups.append(pd.factorize(sy)[0])
    if fe == "unit+year":
        groups.append(pd.factorize(yi)[0])
    # drop singletons iteratively
    keep = np.ones(len(Z), bool)
    for _ in range(10):
        ch = False
        for g in groups:
            c = np.bincount(g[keep] if False else g, weights=keep.astype(float))
            bad = keep & (c[g] <= 1)
            if bad.any():
                keep &= ~bad
                ch = True
        if not ch:
            break
    Z, ui, sy, si = Z[keep], ui[keep], sy[keep], SI[m][keep]
    groups = [pd.factorize(g[keep])[0] for g in groups]
    Zd = demean(Z, groups) if groups else Z - Z.mean(0)
    yd, Xm = Zd[:, 0], Zd[:, 1:]
    cl = ui if cluster == "unit" else si
    res = sm.OLS(yd, Xm).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(cl)[0]})
    out = {"label": label, "n": int(len(yd)), "n_units": int(len(np.unique(ui))),
           "n_clusters": int(len(np.unique(cl)))}
    for k, nm in enumerate(names):
        out[nm + "_b"] = res.params[k]
        out[nm + "_se"] = res.bse[k]
        out[nm + "_p"] = res.pvalues[k]
    out["sd_y"] = float(np.std(Z[:, 0]))
    out["cov01"] = float(np.asarray(res.cov_params())[0, 1]) if len(names) > 1 else np.nan
    out["_mask"] = m
    return out


def row(o, var, group, spec, extra=""):
    return {"group": group, "spec": spec, "var": var, "b": o[var + "_b"], "se": o[var + "_se"],
            "lo": o[var + "_b"] - 1.96 * o[var + "_se"], "hi": o[var + "_b"] + 1.96 * o[var + "_se"],
            "p": o[var + "_p"], "n": o["n"], "n_units": o["n_units"], "n_clusters": o["n_clusters"],
            "sd_dep": o["sd_y"], "note": extra}


rows = []
desc = []
rain_ctrl = lambda: {k: RAIN[k] for k in RAIN_NAMES}

for g in GROUPS:
    D = diff(S[g])
    WD = wavg(D, nb_idx)
    PD = wavg(D, plc_idx)
    SD2 = wavg(D, so_idx)
    base = {"WD_l1": shift(WD, 1), "own_l1": shift(D, 1)}

    # M1 no rain
    o = fit(D, dict(base), label="M1")
    rows.append(row(o, "WD_l1", g, "M1 no rain")); rows.append(row(o, "own_l1", g, "M1 no rain"))
    # M2 main
    X2 = {**base, **rain_ctrl()}
    o2 = fit(D, X2, label="M2")
    rows.append(row(o2, "WD_l1", g, "M2 main")); rows.append(row(o2, "own_l1", g, "M2 main"))
    m_main = o2["_mask"]
    sdx = np.nanstd(base["WD_l1"].ravel()[m_main])
    desc.append({"group": g, "mean_share": np.nanmean(S[g][s606]), "sd_dS": np.nanstd(D[s606]),
                 "sd_WD_l1_main": sdx, "n_main": o2["n"]})
    # contemporaneous, for reference
    o = fit(D, {"WD_t": WD, "own_l1": shift(D, 1), **rain_ctrl()})
    rows.append(row(o, "WD_t", g, "ref contemporaneous W dS_t"))
    # lead placebo, alone and with lag; same sample as main where possible
    o = fit(D, {"WD_f1": shift(WD, -1), "own_l1": shift(D, 1), **rain_ctrl()},
            extra_mask=np.isfinite(shift(WD, 1)))
    rows.append(row(o, "WD_f1", g, "P1 lead alone"))
    o = fit(D, {"WD_l1": shift(WD, 1), "WD_f1": shift(WD, -1), "own_l1": shift(D, 1), **rain_ctrl()})
    rows.append(row(o, "WD_l1", g, "P1 lag+lead")); rows.append(row(o, "WD_f1", g, "P1 lag+lead"))
    # mirrored lead: control own FUTURE change, as the lag spec controls own past change.
    # A transitory shock common to i and its neighbours at t raises dS_i,t and lowers
    # dS_j,t+1, so the raw t+1 lead is pushed negative mechanically; the mirror removes it.
    o = fit(D, {"WD_f1": shift(WD, -1), "own_f1": shift(D, -1), **rain_ctrl()},
            extra_mask=np.isfinite(shift(WD, 1)) & np.isfinite(shift(D, 1)))
    rows.append(row(o, "WD_f1", g, "P1b lead, own-lead control (mirror)"))
    o = fit(D, {"WD_l1": shift(WD, 1), "own_l1": shift(D, 1), **rain_ctrl()},
            extra_mask=np.isfinite(shift(WD, -1)) & np.isfinite(shift(D, -1)))
    rows.append(row(o, "WD_l1", g, "P1b lag, same sample as mirror"))
    # P1d: lag and lead together, own lag and lead controlled; test lag = lead.
    # Neighbour-led spread predicts lag > lead; a shared regional process predicts lag = lead.
    o = fit(D, {"WD_l1": shift(WD, 1), "WD_f1": shift(WD, -1), "own_l1": shift(D, 1),
                "own_f1": shift(D, -1), **rain_ctrl()})
    rows.append(row(o, "WD_l1", g, "P1d lag+lead, own lag+lead")); rows.append(row(o, "WD_f1", g, "P1d lag+lead, own lag+lead"))
    from scipy.stats import norm as _n
    dif = o["WD_l1_b"] - o["WD_f1_b"]
    dse = np.sqrt(o["WD_l1_se"] ** 2 + o["WD_f1_se"] ** 2 - 2 * o["cov01"])
    rows.append({"group": g, "spec": "P1d lag+lead, own lag+lead", "var": "WD_l1 minus WD_f1", "b": dif, "se": dse,
                 "lo": dif - 1.96 * dse, "hi": dif + 1.96 * dse, "p": 2 * (1 - _n.cdf(abs(dif / dse))),
                 "n": o["n"], "n_units": o["n_units"], "n_clusters": o["n_clusters"], "sd_dep": o["sd_y"], "note": ""})
    # two-year lead and two-year lag: no level-year shared with own change
    o = fit(D, {"WD_f2": shift(WD, -2), "own_l1": shift(D, 1), **rain_ctrl()},
            extra_mask=np.isfinite(shift(WD, 2)))
    rows.append(row(o, "WD_f2", g, "P1c lead t+2 alone"))
    o = fit(D, {"WD_l2": shift(WD, 2), "own_l1": shift(D, 1), **rain_ctrl()},
            extra_mask=np.isfinite(shift(WD, -2)))
    rows.append(row(o, "WD_l2", g, "P1c lag t-2 alone, same sample"))
    # distance placebo
    cm_ = np.isfinite(shift(PD, 1))
    o = fit(D, {"WD_l1": shift(WD, 1), "own_l1": shift(D, 1), **rain_ctrl()}, extra_mask=cm_)
    rows.append(row(o, "WD_l1", g, "P2 lag, placebo sample"))
    o = fit(D, {"PD_l1": shift(PD, 1), "own_l1": shift(D, 1), **rain_ctrl()},
            extra_mask=np.isfinite(shift(WD, 1)))
    rows.append(row(o, "PD_l1", g, "P2 distance placebo alone"))
    o = fit(D, {"WD_l1": shift(WD, 1), "PD_l1": shift(PD, 1), "own_l1": shift(D, 1), **rain_ctrl()})
    rows.append(row(o, "WD_l1", g, "P2 lag+placebo")); rows.append(row(o, "PD_l1", g, "P2 lag+placebo"))
    dif = o["WD_l1_b"] - o["PD_l1_b"]
    dse = np.sqrt(o["WD_l1_se"] ** 2 + o["PD_l1_se"] ** 2 - 2 * o["cov01"])
    from scipy.stats import norm as _n
    rows.append({"group": g, "spec": "P2 lag+placebo", "var": "WD_l1 minus PD_l1", "b": dif, "se": dse,
                 "lo": dif - 1.96 * dse, "hi": dif + 1.96 * dse, "p": 2 * (1 - _n.cdf(abs(dif / dse))),
                 "n": o["n"], "n_units": o["n_units"], "n_clusters": o["n_clusters"], "sd_dep": o["sd_y"], "note": ""})
    # second order
    o = fit(D, {"WD_l1": shift(WD, 1), "SO_l1": shift(SD2, 1), "own_l1": shift(D, 1), **rain_ctrl()})
    rows.append(row(o, "WD_l1", g, "P3 lag+2nd-order")); rows.append(row(o, "SO_l1", g, "P3 lag+2nd-order"))

    # robustness on main
    o = fit(D, X2, cluster="state"); rows.append(row(o, "WD_l1", g, "R1 cluster state"))
    o = fit(D, X2, fe="unit+year"); rows.append(row(o, "WD_l1", g, "R2 unit+year FE only"))
    o = fit(D, X2, fe="sy"); rows.append(row(o, "WD_l1", g, "R3 state-year FE, no unit FE"))
    o = fit(D, {"WD_l1": base["WD_l1"], **rain_ctrl()}); rows.append(row(o, "WD_l1", g, "R4 no own lag"))
    lateok = np.zeros((NU, NY), bool); lateok[:, 2:] = True  # drop 1997-98 and 1998-99 as t
    o = fit(D, X2, extra_mask=lateok); rows.append(row(o, "WD_l1", g, "R5 t>=1999-00"))
    Dr = diff(S_raw[g]); WDr = wavg(Dr, nb_idx)
    o = fit(Dr, {"WD_l1": shift(WDr, 1), "own_l1": shift(Dr, 1), **rain_ctrl()})
    rows.append(row(o, "WD_l1", g, "R6 no defect filter"))
    # take-up: neighbours' rises and falls separately
    WDp = wavg(np.where(np.isfinite(D), np.maximum(D, 0), np.nan), nb_idx)
    WDn = wavg(np.where(np.isfinite(D), np.minimum(D, 0), np.nan), nb_idx)
    o = fit(D, {"WDpos_l1": shift(WDp, 1), "WDneg_l1": shift(WDn, 1), "own_l1": shift(D, 1), **rain_ctrl()})
    rows.append(row(o, "WDpos_l1", g, "R7 rises/falls split")); rows.append(row(o, "WDneg_l1", g, "R7 rises/falls split"))
    # two lags
    o = fit(D, {"WD_l1": shift(WD, 1), "WD_l2": shift(WD, 2), "own_l1": shift(D, 1), "own_l2": shift(D, 2), **rain_ctrl()})
    rows.append(row(o, "WD_l1", g, "R8 two lags")); rows.append(row(o, "WD_l2", g, "R8 two lags"))
    # 3-year changes: own t-3..t on neighbours' t-6..t-3
    D3 = diff(S[g], 3); WD3 = wavg(D3, nb_idx)
    yrmask = np.zeros((NU, NY), bool); yrmask[:, [k for k in range(NY) if (years[k] - years[0]) % 3 == 0]] = True
    o = fit(D3, {"WD3_l3": shift(WD3, 3), "own3_l3": shift(D3, 3), **rain_ctrl()}, extra_mask=yrmask)
    rows.append(row(o, "WD3_l3", g, "R9 3-yr changes, non-overlapping"))
    o = fit(D3, {"WD3_f3": shift(WD3, -3), "own3_l3": shift(D3, 3), **rain_ctrl()},
            extra_mask=yrmask & np.isfinite(shift(WD3, 3)))
    rows.append(row(o, "WD3_f3", g, "R9 3-yr lead placebo"))
    o = fit(D3, {"WD3_f3": shift(WD3, -3), "own3_f3": shift(D3, -3), **rain_ctrl()},
            extra_mask=yrmask & np.isfinite(shift(WD3, 3)) & np.isfinite(shift(D3, 3)))
    rows.append(row(o, "WD3_f3", g, "R9 3-yr lead, own-lead control (mirror)"))
    o = fit(D3, {"WD3_l3": shift(WD3, 3), "own3_l3": shift(D3, 3), **rain_ctrl()},
            extra_mask=yrmask & np.isfinite(shift(WD3, -3)) & np.isfinite(shift(D3, -3)))
    rows.append(row(o, "WD3_l3", g, "R9 3-yr lag, same sample as mirror"))
    # log area change
    DL = diff(LA[g]); WDL = wavg(DL, nb_idx)
    o = fit(DL, {"WD_l1": shift(WDL, 1), "own_l1": shift(DL, 1), **rain_ctrl()})
    rows.append(row(o, "WD_l1", g, "R10 dlog area"))
    o = fit(DL, {"WD_f1": shift(WDL, -1), "own_l1": shift(DL, 1), **rain_ctrl()},
            extra_mask=np.isfinite(shift(WDL, 1)))
    rows.append(row(o, "WD_f1", g, "R10 dlog area lead placebo"))
    # neighbours' lagged LEVEL (does being next to high-share districts predict rises?)
    WS = wavg(S[g], nb_idx)
    o = fit(D, {"WS_l1": shift(WS, 1), "own_lvl_l1": shift(S[g], 1), "own_l1": shift(D, 1), **rain_ctrl()})
    rows.append(row(o, "WS_l1", g, "R11 neighbours' lagged level"))
    # seasons
    for sea in ["kharif", "rabi"]:
        Ds = diff(SS[(sea, g)]); WDs = wavg(Ds, nb_idx)
        if np.nansum(np.abs(Ds[s606])) < 1e-9:
            w(f"season {sea} {g}: no variation, skipped"); continue
        try:
            o = fit(Ds, {"WD_l1": shift(WDs, 1), "own_l1": shift(Ds, 1), **rain_ctrl()})
            rows.append(row(o, "WD_l1", g, f"S {sea} share of season area"))
            o = fit(Ds, {"WD_f1": shift(WDs, -1), "own_l1": shift(Ds, 1), **rain_ctrl()},
                    extra_mask=np.isfinite(shift(WDs, 1)))
            rows.append(row(o, "WD_f1", g, f"S {sea} lead placebo"))
        except Exception as e:
            w(f"season {sea} {g} failed: {e}")
    w(f"done {g}")

res = pd.DataFrame(rows)
from statsmodels.stats.multitest import multipletests
mm = (res.spec == "M2 main") & (res["var"] == "WD_l1")
res.loc[mm, "q_bh_9groups"] = multipletests(res.loc[mm, "p"], method="fdr_bh")[1]
res.to_csv(os.path.join(OUT, "neighbours_estimates.csv"), index=False)
pd.DataFrame(desc).to_csv(os.path.join(OUT, "neighbours_descriptives.csv"), index=False)

# ----------------------------------------------------------------- permutation placebo
w("permutation placebo: random same-state non-adjacent 'neighbours', same count")
perm_rows = []
NPERM = 200
for g in HEAD:
    D = diff(S[g])
    WD = wavg(D, nb_idx)
    obs = res[(res.group == g) & (res.spec == "M2 main") & (res["var"] == "WD_l1")].b.iloc[0]
    draws = []
    for r in range(NPERM):
        sets = []
        for i in range(NU):
            k = len(nb_idx[i])
            cand = np.where((st_arr == st_arr[i]) & (~A[i]) & (np.arange(NU) != i))[0]
            sets.append(RNG.choice(cand, size=min(k, len(cand)), replace=False) if k and len(cand) else np.array([], int))
        RD = wavg(D, sets)
        o = fit(D, {"RD_l1": shift(RD, 1), "own_l1": shift(D, 1), **rain_ctrl()},
                extra_mask=np.isfinite(shift(WD, 1)))
        draws.append(o["RD_l1_b"])
    draws = np.array(draws)
    p2 = (np.sum(np.abs(draws - draws.mean()) >= abs(obs - draws.mean())) + 1) / (NPERM + 1)
    perm_rows.append({"group": g, "observed_b": obs, "perm_mean": draws.mean(), "perm_sd": draws.std(),
                      "perm_p05": np.percentile(draws, 5), "perm_p95": np.percentile(draws, 95),
                      "share_draws_ge_obs": np.mean(draws >= obs), "p_two_sided_centered": p2,
                      "n_draws": NPERM})
    w(f"  {g}: observed {obs:.4f}; random same-state sets mean {draws.mean():.4f}, "
      f"sd {draws.std():.4f}, 5-95% [{np.percentile(draws,5):.4f}, {np.percentile(draws,95):.4f}]")
pd.DataFrame(perm_rows).to_csv(os.path.join(OUT, "neighbours_permutation.csv"), index=False)

# ----------------------------------------------------------------- print headline
pd.set_option("display.width", 220)
show = res[res.group.isin(HEAD)][["group", "spec", "var", "b", "se", "p", "n", "n_clusters"]]
w(show.round(4).to_string(index=False))
oth = res[(~res.group.isin(HEAD)) & res.spec.isin(["M2 main", "P1 lead alone", "P2 distance placebo alone"])
          & res["var"].isin(["WD_l1", "WD_f1", "PD_l1"])]
w(oth[["group", "spec", "var", "b", "se", "p", "n"]].round(4).to_string(index=False))

# ----------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True)
specs = [("M2 main", "WD_l1", "", "o", "0.1"),
         ("P1b lead, own-lead control (mirror)", "WD_f1", "", "s", "0.35"),
         ("P2 distance placebo alone", "PD_l1", "", "^", "0.6")]
for ax, g in zip(axes, HEAD):
    for k, (sp, v, lab, mk, col) in enumerate(specs):
        r = res[(res.group == g) & (res.spec == sp) & (res["var"] == v)].iloc[0]
        ax.errorbar(k, r.b, yerr=1.96 * r.se, fmt=mk, color=col, ms=7, capsize=4, lw=1.5)
    ax.axhline(0, color="0.75", lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["neighbours,\nlast year", "neighbours,\nnext year\n(placebo)", "same-state\nnon-neighbours,\nlast year (placebo)"], fontsize=8)
    ax.set_title(g.capitalize(), fontsize=10)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("change in own share per unit change\nin comparison districts' share")
fig.suptitle("Neighbours' next-year change predicts a district's change as well as their last-year change does\n"
             "(year-on-year change in share of gross cropped area; 95% CI; district and state-by-year fixed effects)", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "neighbours_coefplot.png"), dpi=200)

with open(os.path.join(OUT, "neighbours_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
w("written")
