"""
139_pe_common_crops_border.py

Is the state-border gap in crop-mix likeness a reporting artefact? Reviewer item: each
state keeps its own list of minor crops, and adjacent districts across a state line
differ by about five crops in how many they report. Restrict likeness to crops that
every state reports and rerun the border gap.

Crops: rice, wheat, maize, gram (chickpea), arhar/tur, groundnut, rapeseed and mustard,
cotton (lint). APY names: Rice, Wheat, Maize, Gram, Arhar/Tur, Groundnut,
Rapeseed &Mustard, Cotton(Lint).

Steps
  1. APY cleaned exactly as script 125 (script 98 rules), crop area summed over seasons
     per 2011 parent unit and year; window shares = crop area / GCA summed over the
     unit's reliable years (same rule as pe_unit_period.csv; checked on rice).
  2. Reporting check: for each present-day state and crop, the share of reliable
     unit-years with positive area and the first and last year the crop appears.
  3. Likeness on the eight crops, two ways:
       sim8      100 minus half the summed absolute gap in the eight crops' shares of
                 their own combined area (composition among common crops)
       sim8_gca  share of gross cropped area the two districts have in common in the
                 eight crops: 100 * sum_k min(a_k, b_k), shares of GCA
  4. Border gap on adjacent pairs (Andhra Pradesh-Telangana dropped), as scripts 127
     and 132: M1 distance; M2 + gaps in rainfall mean, rainfall CV, elevation,
     ruggedness, irrigated share; M3 M2 + state-membership FE. Dyadic SE by unit and by
     state; restricted wild cluster bootstrap by state (Webb weights, 999 draws).
  5. Border-equivalent km as script 132: distance at which same-state pairs fall to
     the cross-border adjacent mean (interpolated 20 km bins, linear 0-300 km fit),
     minus the mean distance of adjacent same-state pairs; unit bootstrap 200 draws;
     and the regression version (cross / per-100 km slope, pairs <= 300 km).

Run: C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/139_pe_common_crops_border.py
Outputs (outputs/revision_nov8/peer_effects/v2/):
  139_reporting_by_state.csv, 139_border_gap.csv, 139_border_km.csv,
  139_similarity_classes.csv, 139_log.txt
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _pe_common import OUT, load, dyadic_ols, dyadic_vcov  # noqa: E402
from _years import FIRST_YEAR, LAST_YEAR  # noqa: E402

ROOT = os.path.dirname(HERE)
V2 = os.path.join(OUT, "v2")
os.makedirs(V2, exist_ok=True)
RAW = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
CG = os.path.join(ROOT, "outputs", "crop_groups")
LOG = []
rng = np.random.default_rng(139)


def w(s=""):
    print(s)
    LOG.append(str(s))


CROPS = {"Rice": "rice", "Wheat": "wheat", "Maize": "maize", "Gram": "gram", "Arhar/Tur": "tur",
         "Groundnut": "groundnut", "Rapeseed &Mustard": "mustard", "Cotton(Lint)": "cotton"}
C8 = list(CROPS.values())

# ------------------------------------------------------------------ 1. APY, script 125 cleaning
df = pd.read_csv(RAW)
for c in ["year", "season", "state_name", "district_name", "crop_name"]:
    df[c] = df[c].astype(str).str.strip()
df = df.dropna(subset=["area"])
df = df[df["area"] > 0]
BOGUS = [("Delhi", "Chandigarh"), ("Goa", "Chandigarh"),
         ("The Dadra And Nagar Haveli And Daman And Diu", "Chandigarh"),
         ("Delhi", "Surguja"), ("Goa", "Surguja"),
         ("The Dadra And Nagar Haveli And Daman And Diu", "Surguja"),
         ("Andaman And Nicobar Islands", "Purulia")]
REMAP = {("Uttar Pradesh", d): "Uttarakhand" for d in
         ["Almora", "Bageshwar", "Chamoli", "Champawat", "Dehradun", "Haridwar",
          "Nainital", "Pauri Garhwal", "Pithoragarh"]}
REMAP.update({("Madhya Pradesh", d): "Chhattisgarh" for d in
              ["Bastar", "Bilaspur", "Dhamtari", "Durg", "Jashpur", "Kanker",
               "Korba", "Mahasamund", "Raipur", "Rajnandgaon", "Surguja"]})
REMAP.update({("Bihar", d): "Jharkhand" for d in ["Deoghar", "Dhanbad", "Garhwa"]})
pair = list(zip(df["state_name"], df["district_name"]))
df = df[[p not in BOGUS for p in pair]]
for (s_, d_), new in REMAP.items():
    m = (df["state_name"] == s_) & (df["district_name"] == d_)
    df.loc[m, "state_name"] = new
df["year_start"] = df["year"].str.split("-").str[0].astype(int)
df["district_key"] = df["state_name"].str.upper() + "|" + df["district_name"].str.upper()
df = df[df["year_start"].between(FIRST_YEAR, LAST_YEAR)]
K = ["district_key", "year", "season", "crop_name"]
df = df.sort_values("area", ascending=False).drop_duplicates(subset=K, keep="first")
cm = pd.read_csv(os.path.join(CG, "crop_group_child_map.csv"))[["div_key", "unit_key"]]
df = df.merge(cm, left_on="district_key", right_on="div_key", how="left")
df = df[df.unit_key.notna() & df.crop_name.isin(CROPS)]
ca = df.groupby(["unit_key", "year_start", "crop_name"]).area.sum().unstack().rename(columns=CROPS)
ca = ca.reindex(columns=C8).fillna(0).reset_index()
w(f"APY rows for the eight crops {FIRST_YEAR}-{LAST_YEAR}: {len(df):,}")

units, uy, up, P = load()
u = uy[["unit_key", "year_start", "gca_ha", "reliable", "st", "share_gca_rice"]].merge(
    ca, on=["unit_key", "year_start"], how="left")
for c in C8:
    u[c] = u[c].fillna(0)
ur = u[u.reliable == 1]

# ------------------------------------------------------------------ 2. reporting
rep = []
in606 = set(units[units.in_606 == 1].unit_key)
for stn, d in ur[ur.unit_key.isin(in606)].groupby("st"):
    for c in C8:
        pos = d[d[c] > 0]
        rep.append({"state": stn, "crop": c, "unit_years": len(d), "share_unit_years_positive": len(pos) / len(d),
                    "share_units_ever": pos.unit_key.nunique() / d.unit_key.nunique(),
                    "first_year": pos.year_start.min() if len(pos) else np.nan,
                    "last_year": pos.year_start.max() if len(pos) else np.nan,
                    "state_area_share_pct": 100 * d[c].sum() / d.gca_ha.sum()})
REP = pd.DataFrame(rep)
REP.to_csv(os.path.join(V2, "139_reporting_by_state.csv"), index=False)
pv = REP.pivot_table(index="state", columns="crop", values="share_units_ever").reindex(columns=C8)
w("\nshare of a state's units that ever report the crop (606 sample, reliable years):")
w(pv.round(2).to_string())
pa = REP.pivot_table(index="state", columns="crop", values="state_area_share_pct").reindex(columns=C8)
w("\ncrop share of the state's GCA, percent:")
w(pa.round(1).to_string())
absent = REP[REP.share_units_ever == 0][["state", "crop"]]
w("state x crop never reported: " + "; ".join(f"{a.title()} {b}" for a, b in absent.values))
# a crop that a state never reports but its neighbours across the line grow at >1% would be a
# reporting suspect; list such cases for the states that share a border
A = P[(P.adjacent == 1) & (P.new_border == 0) & (P.cross == 1)]
sus = []
for a, b in A[["st_i", "st_j"]].drop_duplicates().values:
    for c in C8:
        for x, y in [(a, b), (b, a)]:
            ex = REP[(REP.state == x) & (REP.crop == c)]
            ey = REP[(REP.state == y) & (REP.crop == c)]
            if len(ex) and len(ey) and ex.share_units_ever.iloc[0] == 0 and ey.state_area_share_pct.iloc[0] > 1:
                sus.append((x, y, c, ey.state_area_share_pct.iloc[0]))
w("crop absent from a state's record while the state across the line grows it on >1% of GCA: "
  + ("; ".join(f"{x.title()} (vs {y.title()}) {c} {v:.1f}%" for x, y, c, v in sorted(set(sus))) or "none"))

# ------------------------------------------------------------------ window shares
win = {}
for lab, (a, b) in [("w9802", (1998, 2002)), ("w1519", (2015, 2019))]:
    d = ur[ur.year_start.between(a, b)].groupby("unit_key")[C8 + ["gca_ha"]].sum()
    win[lab] = d[C8].div(d.gca_ha, axis=0)
chk = win["w1519"].rice.reindex(up[up.window == "w1519"].unit_key).values - up[up.window == "w1519"].s_rice.values
w(f"\nrice window share vs pe_unit_period, 2015-19: max |diff| {np.nanmax(np.abs(chk)):.2e}")
for lab in win:
    w(f"{lab}: eight crops' mean share of GCA across 606 units "
      f"{100 * win[lab].reindex(list(in606)).sum(1).mean():.1f} percent")

for lab in ["w9802", "w1519"]:
    Ai = win[lab].reindex(P.unit_i).values
    Bj = win[lab].reindex(P.unit_j).values
    sa, sb = Ai.sum(1, keepdims=True), Bj.sum(1, keepdims=True)
    an = np.where(sa > 0, Ai / np.where(sa > 0, sa, 1), np.nan)
    bn = np.where(sb > 0, Bj / np.where(sb > 0, sb, 1), np.nan)
    P[f"sim8_{lab}"] = (1 - 0.5 * np.abs(an - bn).sum(1)) * 100
    P[f"sim8_gca_{lab}"] = np.minimum(Ai, Bj).sum(1) * 100
    P[f"sim9_{lab}"] = P[f"sim_{lab}"] * 100

codes_u = {k: i for i, k in enumerate(sorted(units.unit_key))}
codes_s = {s: i for i, s in enumerate(sorted(units.st.dropna().unique()))}
P["ui"], P["uj"] = P.unit_i.map(codes_u), P.unit_j.map(codes_u)
P["si"], P["sj"] = P.st_i.map(codes_s), P.st_j.map(codes_s)
Pn = P[P.new_border == 0].copy()
GEO = ["d_rain", "d_raincv", "d_elev", "d_tri", "d_irr"]
MEAS = ["sim9", "sim8", "sim8_gca"]

# similarity classes
cls = []
for meas in MEAS:
    for lab in ["w1519", "w9802"]:
        y = Pn[f"{meas}_{lab}"]
        for nm, msk in [("adjacent, same state", (Pn.adjacent == 1) & (Pn.same_state == 1)),
                        ("adjacent, across a border", (Pn.adjacent == 1) & (Pn.same_state == 0)),
                        ("not adjacent, 60-100 km, same state", (Pn.adjacent == 0) & Pn.dist_km.between(60, 100) & (Pn.same_state == 1)),
                        ("not adjacent, 60-100 km, across a border", (Pn.adjacent == 0) & Pn.dist_km.between(60, 100) & (Pn.same_state == 0)),
                        ("200-300 km, same state", Pn.dist_km.between(200, 300) & (Pn.same_state == 1)),
                        ("500-600 km, any", Pn.dist_km.between(500, 600))]:
            cls.append({"measure": meas, "window": lab, "pairs": nm, "n": int((msk & y.notna()).sum()),
                        "mean": y[msk].mean()})
CL = pd.DataFrame(cls)
CL.to_csv(os.path.join(V2, "139_similarity_classes.csv"), index=False)
w("\nlikeness by pair class:")
w(CL.pivot_table(index=["window", "pairs"], columns="measure", values="mean", sort=False).round(1).to_string())


# ------------------------------------------------------------------ 4. border gap
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


def reg(base, ycol, spec, cl):
    Xm, names = design(base, spec)
    mi, mj = (base.ui.values, base.uj.values) if cl == "unit" else (base.si.values, base.sj.values)
    return dyadic_ols(base[ycol].values.astype(float), Xm, names, mi, mj, conservative=(cl == "state"))


WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])


def wild_state(base, ycol, spec, B=999):
    y = base[ycol].values.astype(float)
    Xm, _ = design(base, spec)
    X = np.column_stack([np.ones(len(y)), Xm])
    k0 = 1
    Xr = np.delete(X, k0, 1)
    yr = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
    er = y - yr
    si, sj = base.si.values, base.sj.values
    XtXi = np.linalg.pinv(X.T @ X)

    def tstat(yy):
        b = XtXi @ X.T @ yy
        V, _ = dyadic_vcov(X, yy - X @ b, si, sj, conservative=True)
        return b[k0] / np.sqrt(V[k0, k0])

    t0 = tstat(y)
    ns_ = int(max(si.max(), sj.max())) + 1
    same = si == sj
    ts = np.empty(B)
    for r in range(B):
        v = rng.choice(WEBB, ns_)
        ts[r] = tstat(yr + np.where(same, v[si], (v[si] + v[sj]) / np.sqrt(2)) * er)
    return (1 + np.sum(np.abs(ts) >= abs(t0))) / (B + 1)


Aadj = Pn[Pn.adjacent == 1].copy()
gap = []
for lab in ["w1519", "w9802"]:
    for meas in MEAS:
        ycol = f"{meas}_{lab}"
        base = Aadj.dropna(subset=[ycol, "sim9_" + lab, "sim8_" + lab] + GEO).copy()
        for spec in ["M1", "M2", "M3"]:
            ou, os_ = reg(base, ycol, spec, "unit"), reg(base, ycol, spec, "state")
            rec = {"window": lab, "measure": meas, "spec": spec, "n": ou["_n"], "b": ou["cross"][0],
                   "se_unit": ou["cross"][1], "p_unit": ou["cross"][2], "se_state": os_["cross"][1],
                   "p_state": os_["cross"][2], "mean_same": base.loc[base.cross == 0, ycol].mean(),
                   "mean_cross": base.loc[base.cross == 1, ycol].mean()}
            if lab == "w1519":
                rec["p_wild_state"] = wild_state(base, ycol, spec)
            gap.append(rec)
# robustness: drop pairs touching a state flagged above as a reporting suspect
sus_states = sorted({x for x, _, _, _ in sus})
base = Aadj[~(Aadj.st_i.isin(sus_states) | Aadj.st_j.isin(sus_states))].dropna(
    subset=["sim8_w1519", "sim9_w1519"] + GEO).copy()
for spec in ["M1", "M2", "M3"]:
    ou, os_ = reg(base, "sim8_w1519", spec, "unit"), reg(base, "sim8_w1519", spec, "state")
    gap.append({"window": "w1519", "measure": "sim8, suspect states dropped (" + ", ".join(s.title() for s in sus_states) + ")",
                "spec": spec, "n": ou["_n"], "b": ou["cross"][0], "se_unit": ou["cross"][1], "p_unit": ou["cross"][2],
                "se_state": os_["cross"][1], "p_state": os_["cross"][2],
                "mean_same": base.loc[base.cross == 0, "sim8_w1519"].mean(),
                "mean_cross": base.loc[base.cross == 1, "sim8_w1519"].mean()})
G = pd.DataFrame(gap)
G.to_csv(os.path.join(V2, "139_border_gap.csv"), index=False)
w("\nborder gap, adjacent pairs (same sample for all three measures):")
w(G.round(3).to_string(index=False))

# ------------------------------------------------------------------ 5. border-equivalent km
lab = "w1519"


def km_cross(dd, target, col):
    b = (dd.dist_km // 20) * 20 + 10
    g = dd.groupby(b)[col].agg(["mean", "size"])
    g = g[g["size"] >= 10]
    gi, xs = g["mean"].values, g.index.values
    k = np.where(gi <= target)[0]
    bb = np.nan
    if len(k) and k[0] > 0:
        j = k[0]
        bb = xs[j - 1] + (gi[j - 1] - target) / (gi[j - 1] - gi[j]) * (xs[j] - xs[j - 1])
    q = dd[dd.dist_km <= 300]
    sl, ic = np.polyfit(q.dist_km, q[col], 1)
    return bb, (target - ic) / sl


ukeys = sorted(set(Pn.unit_i) | set(Pn.unit_j))
boot = rng.multinomial(len(ukeys), np.ones(len(ukeys)) / len(ukeys), size=200)
kmrows = []
for meas in MEAS:
    col = f"{meas}_{lab}"
    ss = Pn[Pn.same_state == 1].dropna(subset=[col])
    xadj = Pn[(Pn.adjacent == 1) & (Pn.same_state == 0)].dropna(subset=[col])
    xa = ss[ss.adjacent == 1]
    tgt = xadj[col].mean()
    bi, bl = km_cross(ss, tgt, col)
    kb = []
    for c in boot:
        wu = pd.Series(c, index=ukeys)
        wt = wu.reindex(ss.unit_i).values * wu.reindex(ss.unit_j).values
        dd = ss.loc[ss.index.repeat(wt.astype(int))]
        wx = wu.reindex(xadj.unit_i).values * wu.reindex(xadj.unit_j).values
        t_ = (wx * xadj[col].values).sum() / wx.sum()
        aa = dd[dd.adjacent == 1]
        try:
            r = km_cross(dd, t_, col)
            kb.append((r[0] - aa.dist_km.mean(), r[1] - aa.dist_km.mean()))
        except Exception:
            pass
    kb = np.array(kb)
    q = Pn[Pn.dist_km <= 300].dropna(subset=[col])
    o = dyadic_ols(q[col].values, np.column_stack([q.cross, q.dist_km / 100, q.adjacent]),
                   ["cross", "dist100", "adj"], q.ui.values, q.uj.values)
    kmrows.append({"measure": meas, "cross_adjacent_mean": tgt, "same_adjacent_mean": xa[col].mean(),
                   "same_adjacent_km": xa.dist_km.mean(), "equiv_interp": bi - xa.dist_km.mean(),
                   "equiv_interp_lo": np.nanpercentile(kb[:, 0], 2.5), "equiv_interp_hi": np.nanpercentile(kb[:, 0], 97.5),
                   "equiv_linear": bl - xa.dist_km.mean(),
                   "equiv_linear_lo": np.nanpercentile(kb[:, 1], 2.5), "equiv_linear_hi": np.nanpercentile(kb[:, 1], 97.5),
                   "reg_cross": o["cross"][0], "reg_per100km": o["dist100"][0],
                   "equiv_reg": 100 * o["cross"][0] / o["dist100"][0]})
KM = pd.DataFrame(kmrows)
KM.to_csv(os.path.join(V2, "139_border_km.csv"), index=False)
w("\nborder-equivalent km, 2015-19:")
w(KM.round(2).to_string(index=False))
ref = pd.read_csv(os.path.join(OUT, "referee", "h1_border_km_equivalent.csv"))
w(f"script 132 reference (nine groups): interp {ref.equiv_interp.iloc[0]:.0f}, linear {ref.equiv_linear.iloc[0]:.0f}, "
  f"regression {ref.equiv_reg.iloc[0]:.0f}")

with open(os.path.join(V2, "139_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
