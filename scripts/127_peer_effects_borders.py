"""
127_peer_effects_borders.py

Q2. State borders as a test of geography against policy.

A. Cross-section, adjacent pairs (queen, script 97), both windows:
     y_ij = a + b * cross_ij + f(distance) [+ |gaps in rainfall mean, rainfall CV,
            elevation, ruggedness, irrigated share|] [+ state-membership FE] + e
   y = crop-mix similarity, the |D1| gap, and |share gap| for single groups.
   Andhra Pradesh-Telangana pairs are dropped (a border only from 2014; see C).
   SEs dyadic-robust: by 2011 parent unit, and by state.

B. Pair-year panel, adjacent pairs, annual values (both units reliable):
     y_ijt = pair FE + year FE + sum_p c_p * cross_ij * period_p
             + d * newborder_ij * post2014_t + e
   Pair FE hold every fixed geographic difference between the two districts, so c_p
   asks whether cross-border pairs drifted apart relative to same-state pairs.
   Periods: 1998-2002 (base), 2003-07, 2008-12, 2013-19. A linear version uses
   cross * (year - 1998)/10.

C. Border by border: for every pair of states with at least 5 adjacent cross-border
   district pairs, the mean within-pair difference (state A minus state B) in rice,
   wheat, maize, millets, pulses, oilseeds, cotton and sugarcane shares and D1, in
   each window. This is the border gap with pair fixed effects. A geography-adjusted
   gap subtracts what the rainfall/elevation/ruggedness/irrigation differences across
   the pair predict, with the prediction taken from same-state adjacent pairs (signed
   differences, no constant). Side means for dumbbells. Annual gaps for the
   highlighted borders, with a within-pair trend.

Run: C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/127_peer_effects_borders.py
Outputs (outputs/revision_nov8/peer_effects/):
  q2_crosssection.csv, q2_panel_widening.csv, q2_border_gaps.csv, q2_border_annual.csv,
  q2_border_trends.csv, q2_geography_gaps.csv, q2_log.txt
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pe_common import OUT, GROUPS, load, dyadic_ols, demean, dyadic_vcov  # noqa: E402
from scipy import stats  # noqa: E402

LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


units, uy, up, P = load()
A = P[P.adjacent == 1].copy()
codes_u = {u: i for i, u in enumerate(sorted(units.unit_key))}
codes_s = {s: i for i, s in enumerate(sorted(units.st.dropna().unique()))}
A["ui"], A["uj"] = A.unit_i.map(codes_u), A.unit_j.map(codes_u)
A["si"], A["sj"] = A.st_i.map(codes_s), A.st_j.map(codes_s)
GEO = ["d_rain", "d_raincv", "d_elev", "d_tri", "d_irr"]
w(f"adjacent pairs {len(A)}: same state {int(A.same_state.sum())}, cross-border "
  f"{int(((A.same_state == 0) & (A.new_border == 0)).sum())}, Andhra Pradesh-Telangana {int(A.new_border.sum())}")
w("geography gaps, mean by pair type (adjacent, AP-TS dropped):")
w(A[A.new_border == 0].groupby("cross")[["dist_km"] + GEO].mean().round(3).to_string())

# ------------------------------------------------------------------ A. cross-section
OUTCOMES = ["sim", "dD1"] + [f"g_{g}" for g in ["rice", "wheat", "maize", "millets", "pulses", "oilseeds", "cotton"]]
rows = []
for lab in ["w9802", "w1519"]:
    for yv in OUTCOMES:
        col = f"{yv}_{lab}"
        base = A[(A.new_border == 0)].dropna(subset=[col] + GEO).copy()
        scale = 1.0 if yv == "dD1" else 100.0  # similarity and shares in points
        y = base[col].values * scale
        Xd = {"cross": base.cross.values, "dist100": base.dist_km.values / 100}
        Xg = {**Xd, **{g: base[g].values for g in GEO}}
        # state-membership FE: one dummy per state touched by the pair
        # a same-state pair carries 1 on its state; a cross-border pair 0.5 on each,
        # so b compares it with the average internal similarity of its two states
        S = np.zeros((len(base), len(codes_s)))
        np.add.at(S, (np.arange(len(base)), base.si.values), 0.5)
        np.add.at(S, (np.arange(len(base)), base.sj.values), 0.5)
        S = S[:, S.sum(0) > 0][:, 1:]
        for spec, X in [("M1 distance", Xd), ("M2 + geography gaps", Xg), ("M3 + state-membership FE", Xg)]:
            Xm = np.column_stack(list(X.values()))
            names = list(X)
            if spec.startswith("M3"):
                Xm = np.column_stack([Xm, S])
                names = names + [f"_s{k}" for k in range(S.shape[1])]
            for cl, (mi, mj) in [("unit", (base.ui.values, base.uj.values)),
                                 ("state", (base.si.values, base.sj.values))]:
                o = dyadic_ols(y, Xm, names, mi, mj, conservative=(cl == "state"))
                b, se, p = o["cross"]
                rows.append({"window": lab, "outcome": yv, "spec": spec, "cluster": cl, "b_cross": b,
                             "se": se, "p": p, "n_pairs": o["_n"], "n_clusters": o["_ncl"],
                             "mean_same_state": y[base.cross.values == 0].mean(),
                             "mean_cross": y[base.cross.values == 1].mean(),
                             **{f"b_{g}": o[g][0] for g in GEO if g in o},
                             "b_dist100": o["dist100"][0]})
cs = pd.DataFrame(rows)
cs.to_csv(os.path.join(OUT, "q2_crosssection.csv"), index=False)
w("")
w("A. Cross-border minus same-state, adjacent pairs (similarity and share gaps in points; D1 in crops):")
w(cs[cs.outcome.isin(["sim", "dD1", "g_rice", "g_wheat", "g_pulses", "g_millets"])]
  .pivot_table(index=["window", "outcome", "spec"], columns="cluster", values=["b_cross", "se"]).round(3).to_string())
w(cs.groupby(["window", "spec"]).n_pairs.first().to_string())

# ------------------------------------------------------------------ B. pair-year panel
sh = uy[uy.reliable == 1].set_index(["unit_key", "year_start"])
SH = [f"share_gca_{g}" for g in GROUPS]
years = sorted(uy.year_start.unique())
pp = []
for t in years:
    a = sh.reindex(list(zip(A.unit_i, [t] * len(A))))
    b = sh.reindex(list(zip(A.unit_j, [t] * len(A))))
    d = pd.DataFrame({"pid": np.arange(len(A)), "year": t})
    d["sim"] = (1 - 0.5 * np.abs(a[SH].values - b[SH].values).sum(1)) * 100
    d["dD1"] = np.abs(a.D1.values - b.D1.values)
    for g in ["rice", "wheat", "maize", "millets", "pulses", "oilseeds", "cotton"]:
        d[f"g_{g}"] = np.abs(a[f"share_gca_{g}"].values - b[f"share_gca_{g}"].values) * 100
    pp.append(d)
pp = pd.concat(pp, ignore_index=True)
pp = pp.merge(A[["cross", "new_border", "ui", "uj", "si", "sj", "dist_km"]].reset_index(drop=True)
              .rename_axis("pid").reset_index(), on="pid")
pp = pp.dropna(subset=["sim"])
pp["oldx"] = ((pp.cross == 1) & (pp.new_border == 0)).astype(int)
PER = [(2003, 2007), (2008, 2012), (2013, 2019)]
for a_, b_ in PER:
    pp[f"x_{a_}"] = pp.oldx * pp.year.between(a_, b_)
pp["x_trend"] = pp.oldx * (pp.year - 1998) / 10
pp["nb_post14"] = pp.new_border * (pp.year >= 2014)
pp["nb_trend"] = pp.new_border * (pp.year - 1998) / 10
w(f"\nB. pair-years {len(pp)}, pairs {pp.pid.nunique()} (cross {pp[pp.oldx == 1].pid.nunique()}, "
  f"AP-TS {pp[pp.new_border == 1].pid.nunique()})")
prow = []
gp, gy = pd.factorize(pp.pid)[0], pd.factorize(pp.year)[0]
for yv in OUTCOMES:
    for spec, xs in [("periods", [f"x_{a}" for a, _ in PER] + ["nb_post14"]),
                     ("trend", ["x_trend", "nb_post14"])]:
        Z = demean(pp[[yv] + xs].values, [gp, gy])
        for cl, (mi, mj) in [("unit", (pp.ui.values, pp.uj.values)), ("state", (pp.si.values, pp.sj.values))]:
            o = dyadic_ols(Z[:, 0], Z[:, 1:], xs, mi, mj, add_const=False, conservative=(cl == "state"))
            for x in xs:
                prow.append({"outcome": yv, "spec": spec, "term": x, "cluster": cl, "b": o[x][0],
                             "se": o[x][1], "p": o[x][2], "n": o["_n"], "n_clusters": o["_ncl"]})
pw = pd.DataFrame(prow)
pw.to_csv(os.path.join(OUT, "q2_panel_widening.csv"), index=False)
w("Pair FE + year FE; cross-border x period (base 1998-2002), AP-TS x post-2014:")
w(pw[pw.outcome.isin(["sim", "dD1", "g_rice", "g_wheat", "g_pulses", "g_millets"])]
  .pivot_table(index=["outcome", "spec", "term"], columns="cluster", values=["b", "se"]).round(3).to_string())
# raw means by period for the chart
pp["period"] = pd.cut(pp.year, [1997, 2002, 2007, 2012, 2019], labels=["1998-02", "2003-07", "2008-12", "2013-19"])
pp["ptype"] = np.where(pp.new_border == 1, "Andhra Pradesh-Telangana", np.where(pp.cross == 1, "cross-border", "same state"))
pr = pp.groupby(["ptype", "year"])[OUTCOMES].mean().reset_index()
pr["n_pairs"] = pp.groupby(["ptype", "year"]).size().values
pr.to_csv(os.path.join(OUT, "q2_pair_type_year_means.csv"), index=False)

# ------------------------------------------------------------------ C. border by border
VARS = ["rice", "wheat", "maize", "millets", "pulses", "oilseeds", "cotton", "sugarcane"]
COV = ["rain_ann_mean", "rain_ann_cv", "elevation_mean_m", "tri_mean", "irr_share"]
cov = units.set_index("unit_key")[COV].copy()
cov["rain_ann_mean"] /= 100
cov["elevation_mean_m"] /= 100
upw = {lab: up[up.window == lab].set_index("unit_key") for lab in ["w9802", "w1519"]}

# geography -> crop-mix slope from same-state adjacent pairs, signed differences, no constant
gam = {}
ss = A[(A.same_state == 1)]
for lab in ["w9802", "w1519"]:
    t = upw[lab]
    for v in VARS + ["D1"]:
        col = "D1" if v == "D1" else f"s_{v}"
        dy = t.reindex(ss.unit_i)[col].values - t.reindex(ss.unit_j)[col].values
        dx = cov.reindex(ss.unit_i).values - cov.reindex(ss.unit_j).values
        ok = np.isfinite(dy) & np.isfinite(dx).all(1)
        gam[(lab, v)] = np.linalg.lstsq(dx[ok], dy[ok], rcond=None)[0]
        r2 = 1 - ((dy[ok] - dx[ok] @ gam[(lab, v)]) ** 2).sum() / (dy[ok] ** 2).sum()
        if lab == "w1519" and v in ["rice", "wheat", "pulses", "millets", "D1"]:
            w(f"  geography slope {lab} {v}: R2 of signed same-state gaps on geography gaps {r2:.2f}")

ORIENT = [("PUNJAB", "HARYANA"), ("HARYANA", "RAJASTHAN"), ("PUNJAB", "RAJASTHAN"),
          ("CHHATTISGARH", "ODISHA"), ("JHARKHAND", "ODISHA"), ("CHHATTISGARH", "JHARKHAND"),
          ("TELANGANA", "KARNATAKA"), ("TELANGANA", "MAHARASHTRA"), ("ANDHRA PRADESH", "KARNATAKA"),
          ("KARNATAKA", "MAHARASHTRA"), ("MADHYA PRADESH", "UTTAR PRADESH"), ("WEST BENGAL", "BIHAR"),
          ("TELANGANA", "ANDHRA PRADESH"), ("MADHYA PRADESH", "RAJASTHAN"), ("BIHAR", "UTTAR PRADESH"),
          ("BIHAR", "JHARKHAND"), ("MADHYA PRADESH", "MAHARASHTRA"), ("CHHATTISGARH", "MADHYA PRADESH"),
          ("HARYANA", "UTTAR PRADESH"), ("GUJARAT", "RAJASTHAN"), ("GUJARAT", "MAHARASHTRA"),
          ("KERALA", "TAMIL NADU"), ("KARNATAKA", "TAMIL NADU"), ("JHARKHAND", "WEST BENGAL"),
          ("UTTARAKHAND", "UTTAR PRADESH"), ("ANDHRA PRADESH", "ODISHA"), ("PUNJAB", "HIMACHAL PRADESH")]
HIGHLIGHT = ORIENT[:13]
X = A[A.cross == 1].copy()
X["key"] = [tuple(sorted((a, b))) for a, b in zip(X.st_i, X.st_j)]
cnt = X.key.value_counts()
borders = [o for o in ORIENT if tuple(sorted(o)) in cnt.index and cnt[tuple(sorted(o))] >= 3]
borders += [tuple(k) for k in cnt[cnt >= 5].index if not any(tuple(sorted(o)) == k for o in ORIENT)]
grow, side_rows = [], []
for sa, sb in borders:
    d = X[X.key == tuple(sorted((sa, sb)))].copy()
    # orient so that unit_a is in state sa
    flip = d.st_i != sa
    d["ua"] = np.where(flip, d.unit_j, d.unit_i)
    d["ub"] = np.where(flip, d.unit_i, d.unit_j)
    mi, mj = d.ua.map(codes_u).values, d.ub.map(codes_u).values
    for lab in ["w9802", "w1519", "change"]:
        for v in VARS + ["D1"]:
            col = "D1" if v == "D1" else f"s_{v}"
            sc = 1 if v == "D1" else 100
            if lab == "change":
                ya = (upw["w1519"].reindex(d.ua)[col].values - upw["w9802"].reindex(d.ua)[col].values)
                yb = (upw["w1519"].reindex(d.ub)[col].values - upw["w9802"].reindex(d.ub)[col].values)
            else:
                ya, yb = upw[lab].reindex(d.ua)[col].values, upw[lab].reindex(d.ub)[col].values
            dy = (ya - yb) * sc
            ok = np.isfinite(dy)
            if ok.sum() < 3:
                continue
            o = dyadic_ols(dy[ok], np.empty((ok.sum(), 0)), [], mi[ok], mj[ok])
            rec = {"state_a": sa, "state_b": sb, "border": f"{sa.title()} | {sb.title()}",
                   "highlight": int((sa, sb) in HIGHLIGHT), "window": lab, "var": v,
                   "n_pairs": int(ok.sum()), "n_units_a": d.ua[ok].nunique(), "n_units_b": d.ub[ok].nunique(),
                   "mean_a": np.nanmean(pd.Series(ya[ok], index=d.ua[ok]).groupby(level=0).first()) * sc,
                   "mean_b": np.nanmean(pd.Series(yb[ok], index=d.ub[ok]).groupby(level=0).first()) * sc,
                   "gap_pairFE": o["const"][0], "se_unit": o["const"][1], "p_unit": o["const"][2],
                   "few_pairs": int(ok.sum() < 8)}
            if lab != "change":
                dx = cov.reindex(d.ua).values - cov.reindex(d.ub).values
                okx = ok & np.isfinite(dx).all(1)
                pred = (dx @ gam[(lab, v)]) * sc
                if okx.sum() >= 3:
                    o2 = dyadic_ols((dy - pred)[okx], np.empty((okx.sum(), 0)), [], mi[okx], mj[okx])
                    rec.update({"geo_predicted_gap": np.nanmean(pred[okx]), "gap_net_of_geography": o2["const"][0],
                                "se_net": o2["const"][1], "p_net": o2["const"][2], "n_pairs_net": int(okx.sum())})
            grow.append(rec)
bg = pd.DataFrame(grow)
bg.to_csv(os.path.join(OUT, "q2_border_gaps.csv"), index=False)
w("\nC. Border gaps (state A minus state B, points of GCA; D1 in crops), 2015-19, highlighted borders:")
show = bg[(bg.window == "w1519") & (bg.highlight == 1) & bg["var"].isin(["rice", "wheat", "pulses", "millets", "D1"])]
w(show[["border", "var", "n_pairs", "mean_a", "mean_b", "gap_pairFE", "se_unit", "geo_predicted_gap",
        "gap_net_of_geography", "se_net"]].round(2).to_string(index=False))

# fraction of the absolute border gap that geography predicts, pooled over borders and crops
g2 = bg[(bg.window == "w1519") & (bg["var"] != "D1")].dropna(subset=["geo_predicted_gap"])
w(f"\nAcross {len(g2)} border x crop cells in 2015-19: mean |raw gap| {g2.gap_pairFE.abs().mean():.2f} pts, "
  f"mean |geography-predicted| {g2.geo_predicted_gap.abs().mean():.2f}, mean |net| {g2.gap_net_of_geography.abs().mean():.2f}; "
  f"corr(raw, predicted) {np.corrcoef(g2.gap_pairFE, g2.geo_predicted_gap)[0, 1]:.2f}; "
  f"cells with |net| significant at 5% (unit-dyadic): {(g2.p_net < 0.05).sum()}")
geo = []
for lab in ["w9802", "w1519"]:
    g2 = bg[(bg.window == lab) & (bg["var"] != "D1")].dropna(subset=["geo_predicted_gap"])
    geo.append({"window": lab, "cells": len(g2), "mean_abs_raw": g2.gap_pairFE.abs().mean(),
                "mean_abs_geo_pred": g2.geo_predicted_gap.abs().mean(),
                "mean_abs_net": g2.gap_net_of_geography.abs().mean(),
                "corr_raw_pred": np.corrcoef(g2.gap_pairFE, g2.geo_predicted_gap)[0, 1],
                "share_sig_raw": (g2.p_unit < 0.05).mean(), "share_sig_net": (g2.p_net < 0.05).mean()})
pd.DataFrame(geo).to_csv(os.path.join(OUT, "q2_geography_gaps.csv"), index=False)

# annual gaps and within-pair trend for the highlighted borders
ann, trn = [], []
shy = uy[uy.reliable == 1].set_index(["unit_key", "year_start"])
for sa, sb in [b for b in borders if b in HIGHLIGHT] + [b for b in borders if b not in HIGHLIGHT]:
    d = X[X.key == tuple(sorted((sa, sb)))]
    flip = d.st_i != sa
    ua = np.where(flip, d.unit_j, d.unit_i)
    ub = np.where(flip, d.unit_i, d.unit_j)
    recs = []
    for t in years:
        a = shy.reindex(list(zip(ua, [t] * len(ua))))
        b = shy.reindex(list(zip(ub, [t] * len(ub))))
        for v in ["rice", "wheat", "maize", "millets", "pulses", "oilseeds", "cotton"]:
            dd = (a[f"share_gca_{v}"].values - b[f"share_gca_{v}"].values) * 100
            for k in range(len(ua)):
                recs.append((v, t, k, ua[k], ub[k], dd[k]))
    r = pd.DataFrame(recs, columns=["var", "year", "pair", "ua", "ub", "gap"]).dropna()
    for v, rv in r.groupby("var"):
        m = rv.groupby("year").gap.agg(["mean", "count"]).reset_index()
        for _, q in m.iterrows():
            ann.append({"border": f"{sa.title()} | {sb.title()}", "state_a": sa, "state_b": sb, "var": v,
                        "year": int(q.year), "gap": q["mean"], "n_pairs": int(q["count"])})
        # within-pair trend: gap_pt = pair FE + beta * (t-1998)/10
        if rv.pair.nunique() >= 3 and rv.year.nunique() >= 10:
            gpp = pd.factorize(rv.pair)[0]
            Z = demean(np.column_stack([rv.gap.values, (rv.year.values - 1998) / 10]), [gpp])
            mi = rv.ua.map(codes_u).values
            mj = rv.ub.map(codes_u).values
            o = dyadic_ols(Z[:, 0], Z[:, 1:], ["trend"], mi, mj, add_const=False)
            trn.append({"border": f"{sa.title()} | {sb.title()}", "highlight": int((sa, sb) in HIGHLIGHT),
                        "var": v, "gap_first5": rv[rv.year <= 2002].gap.mean(),
                        "gap_last5": rv[rv.year >= 2015].gap.mean(),
                        "trend_per_decade": o["trend"][0], "se_unit": o["trend"][1], "p": o["trend"][2],
                        "n_pair_years": len(rv), "n_pairs": rv.pair.nunique()})
pd.DataFrame(ann).to_csv(os.path.join(OUT, "q2_border_annual.csv"), index=False)
tr = pd.DataFrame(trn)
tr.to_csv(os.path.join(OUT, "q2_border_trends.csv"), index=False)
w("\nWithin-pair trend in the signed border gap (points per decade), highlighted borders, |trend| > 2 SE:")
w(tr[(tr.highlight == 1) & (tr.trend_per_decade.abs() > 2 * tr.se_unit)].round(2).to_string(index=False))

with open(os.path.join(OUT, "q2_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
