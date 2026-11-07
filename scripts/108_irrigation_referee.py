"""
108_irrigation_referee.py

Adversarial re-check of the irrigation panel results (scripts 104-107).
Reads the panel from 104 and re-estimates with the 105 helper. Writes nothing
outside outputs/revision_nov8/.

  R1  reproduce the main D1 specification
  R2  is the "within curvature" within? decompose irr^2 around the district mean:
        irr^2 = m^2 + 2 m dev + dev^2  (m^2 absorbed by district FE)
  R3  mechanical denominator / area co-movement: control log NSA and log APY
      cropped area; keep only years with small NSA moves; APY area vs LUS GCA
      consistency filter
  R4  crop-data defect flags from outputs/crop_groups (2011-unit level)
  R5  district-specific linear trends
  R6  timing: lead-only and lag-only quadratics, and both together
  R7  inference: cluster by state (28) instead of 2011 parent
  R8  D0 threshold sensitivity (0.25%, 0.5%, 1%, 2%), within and long-difference,
      long differences with baseline irrigation and baseline-outcome controls
  R9  year-by-year cross-sections of the LUS share (is the 0.27 peak stable?)

Output: outputs/revision_nov8/irrigation_referee_raw.md, irrigation_referee.csv
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:/crop-diversity/scripts")
from _revision_fe import fe_ols, turning_point, pval, demean  # noqa: E402

RAW = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
REPO = r"D:/crop-diversity"
OUT = REPO + "/outputs/revision_nov8"
R, ROWS = [], []


def w(s=""):
    print(s)
    R.append(str(s))


# ---------------------------------------------------------------- extra D0 thresholds
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
df = df[df["year_start"] <= 2019]
df["district_key"] = df["state_name"].str.upper() + "|" + df["district_name"].str.upper()
df = df.sort_values("area", ascending=False).drop_duplicates(
    subset=["district_key", "year", "season", "crop_name"], keep="first")
dy = df.groupby(["district_key", "year_start", "crop_name"])["area"].sum().reset_index()
dy["p"] = dy["area"] / dy.groupby(["district_key", "year_start"])["area"].transform("sum")
TH = {"D0_025": 0.0025, "D0_05chk": 0.005, "D0_1": 0.01, "D0_2": 0.02}
for k, t in TH.items():
    dy[k] = (dy["p"] >= t).astype(int)
dy["top1"] = dy["p"]
thr = dy.groupby(["district_key", "year_start"]).agg(
    **{k: (k, "sum") for k in TH}, top_share=("top1", "max")).reset_index()

P = pd.read_csv(OUT + "/irrigation_panel_district_year.csv")
P = P.merge(thr, on=["district_key", "year_start"], how="left")
assert (P["D0_05chk"] == P["D0_05"]).all(), "D0_05 rebuild mismatch"
fl = pd.read_csv(REPO + "/outputs/crop_groups/crop_group_flags.csv",
                 usecols=["unit_key", "year_start", "flag_annual_unreliable", "flag_season_unreliable",
                          "f_pre_split_donor", "f_bihar_1999_rice_missing", "f_odisha_summer_rice_2012",
                          "f_whole_year_ge50"])
P = P.merge(fl.rename(columns={"unit_key": "shrug_key"}), on=["shrug_key", "year_start"], how="left")

P = P[P["in_final"] & P["year_start"].between(1998, 2019)].copy()
P["irr"] = P["irr_nsa"]
P["irr2"] = P["irr"] ** 2
P["sy"] = P["state_name"] + "_" + P["year_start"].astype(str)
P["t"] = (P["year_start"] - 2008) / 10
P = P.sort_values(["district_key", "year_start"])
g = P.groupby("district_key")
consec_prev = g["year_start"].diff() == 1
consec_next = g["year_start"].diff(-1) == -1
P["irr_lag"] = g["irr"].shift(1).where(consec_prev)
P["irr_lead"] = g["irr"].shift(-1).where(consec_next)
P["irr_lag2"], P["irr_lead2"] = P["irr_lag"] ** 2, P["irr_lead"] ** 2
P["lnsa"] = np.log(P["nsa"])
P["lapy"] = np.log(P["apy_area"])
P["lnia"] = np.log(P["nia"].where(P["nia"] > 0))
P["dlnsa"] = g["lnsa"].diff().abs().where(consec_prev)
P["apy_gca"] = P["apy_area"] / P["gca"]
P["apy_gca_dev"] = (np.log(P["apy_gca"]) - P.groupby("district_key")["apy_gca"].transform(
    lambda x: np.log(x).median())).abs()
P["state_cl"] = P["state_name"]
E = P[P["irr"].notna()].copy()
E["m"] = E.groupby("district_key")["irr"].transform("mean")
E["dev"] = E["irr"] - E["m"]
E["mdev"] = E["m"] * E["dev"]
E["dev2"] = E["dev"] ** 2
FES = ["district_key", "sy"]
NEST = ("district_key",)


def est(lab, d, y="D1", xs=("irr", "irr2"), fes=FES, cl="shrug_key", nested=NEST, quad=True):
    xs = list(xs)
    r = fe_ols(d, y, xs, list(fes), cl, nested=nested)
    o = dict(test=lab, y=y, n=r["n"], G=r["G"])
    for x in xs:
        o["b_" + x], o["se_" + x] = r["params"][x], r["se"][x]
        o["p_" + x] = pval(r["params"][x], r["se"][x], r["G"])
    txt = "; ".join("{} {:+.3f} ({:.3f}, p={:.3f})".format(x, o["b_" + x], o["se_" + x], o["p_" + x])
                    for x in xs[:4])
    if quad and len(xs) >= 2 and xs[1].endswith("2"):
        tp, tse = turning_point(r, xs[0], xs[1])
        o["tp"], o["tp_se"] = tp, tse
        txt += "; peak {:.3f} ({:.3f})".format(tp, tse)
    ROWS.append(o)
    w("- {} [{}] n={:,}, G={}: {}".format(lab, y, r["n"], r["G"], txt))
    return r


w("# Irrigation panel: referee re-check")
w("")
w("## R1 reproduce main")
est("main", E)
est("main linear", E, xs=["irr"], quad=False)

w("")
w("## R2 decomposition around the district mean (m absorbed; m*dev = between-district slope heterogeneity)")
dm = demean(E.dropna(subset=["D1"]), ["irr2", "mdev", "dev2"], FES)
w("- share of residual irr^2 variance explained by 2*m*dev: R2 = {:.3f}".format(
    np.corrcoef(dm["irr2"], dm["mdev"])[0, 1] ** 2))
for y in ["D1", "D0_05", "D2", "even", "D0"]:
    est("decomp irr + m*dev + dev^2", E, y=y, xs=["irr", "mdev", "dev2"], quad=False)

w("")
w("## R3 mechanical area / denominator")
for y in ["D1", "D0_05"]:
    est("main + log NSA + log APY area", E, y=y, xs=["irr", "irr2", "lnsa", "lapy"])
    est("log NIA and log NSA entered separately (linear)", E[E["lnia"].notna()], y=y,
        xs=["lnia", "lnsa", "lapy"], quad=False)
st = E[(E["dlnsa"] < 0.05)]
est("only years with |dlog NSA| < 0.05", st)
est("only years with |dlog NSA| < 0.05, linear", st, xs=["irr"], quad=False)
cons = E[E["apy_gca_dev"] < np.log(1.2)]
w("  APY area / LUS GCA: median {:.3f}; district-years within 20% of own median ratio: {:.1%}".format(
    E["apy_gca"].median(), (E["apy_gca_dev"] < np.log(1.2)).mean()))
est("APY area within 20% of its usual ratio to LUS GCA", cons)
est("APY area within 20% of its usual ratio to LUS GCA", cons, y="D0_05", xs=["irr"], quad=False)

w("")
w("## R4 crop-data defect flags (crop_groups, 2011-unit level)")
w("  flagged annual-unreliable district-years in E: {:,} of {:,}".format(
    int((E["flag_annual_unreliable"] == 1).sum()), len(E)))
cl4 = E[(E["flag_annual_unreliable"] != 1)]
cl5 = E[(E["flag_annual_unreliable"] != 1) & (E["flag_season_unreliable"] != 1)]
for y in ["D1", "D0_05"]:
    est("drop flag_annual_unreliable", cl4, y=y)
    est("drop flag_annual_unreliable, linear", cl4, y=y, xs=["irr"], quad=False)
est("drop annual and season flags", cl5)

w("")
w("## R5 district-specific linear trends")
def detrend_sy(d, cols, it=200):
    """Alternating projections: district intercept+slope on t, and state x year means."""
    X = d[cols].to_numpy(float).copy()
    code = pd.factorize(d["district_key"])[0]
    sy = pd.factorize(d["sy"])[0]
    t = d["t"].to_numpy()
    n_d = np.bincount(code)
    for _ in range(it):
        X0 = X.copy()
        tm = np.bincount(code, t) / n_d
        tc = t - tm[code]
        stt = np.bincount(code, tc * tc)
        for j in range(X.shape[1]):
            xm = np.bincount(code, X[:, j]) / n_d
            xc = X[:, j] - xm[code]
            bta = np.where(stt > 0, np.bincount(code, tc * xc) / np.where(stt > 0, stt, 1), 0)
            X[:, j] = xc - bta[code] * tc
            X[:, j] -= (np.bincount(sy, X[:, j]) / np.bincount(sy))[sy]
        if np.max(np.abs(X - X0)) < 1e-9:
            break
    return pd.DataFrame(X, columns=[c + "_dt" for c in cols], index=d.index)


for y in ["D1", "D0_05", "D2", "even"]:
    d = E.dropna(subset=[y, "irr"]).copy()
    d = d[d.groupby("district_key")["irr"].transform("size") >= 4]
    d = d.join(detrend_sy(d, [y, "irr", "irr2"]))
    d["irr_dt2"] = d["irr2_dt"]
    est("district FE + district trends + state x year FE", d, y=y + "_dt",
        xs=["irr_dt", "irr_dt2"], fes=[], nested=())
    est("same, linear", d.join(detrend_sy(d, [y, "irr"]).add_suffix("L")), y=y + "_dtL",
        xs=["irr_dtL"], fes=[], nested=(), quad=False)

w("")
w("## R6 timing")
for y in ["D1"]:
    est("lead only (t+1 irrigation)", E[E["irr_lead"].notna()], y=y, xs=["irr_lead", "irr_lead2"])
    est("current, same sample as lead", E[E["irr_lead"].notna()], y=y)
    est("lag only (t-1)", E[E["irr_lag"].notna()], y=y, xs=["irr_lag", "irr_lag2"])
    est("current, same sample as lag", E[E["irr_lag"].notna()], y=y)
    est("lag, current, lead together", E[E["irr_lag"].notna() & E["irr_lead"].notna()], y=y,
        xs=["irr_lag", "irr_lag2", "irr", "irr2", "irr_lead", "irr_lead2"], quad=False)

w("")
w("## R7 inference: cluster by state")
for y in ["D1", "D0_05", "D0", "D2", "even", "cereal_share"]:
    est("main, state clusters", E, y=y, cl="state_cl", nested=())
for y in ["D1", "D0_05"]:
    est("linear, state clusters", E, y=y, xs=["irr"], cl="state_cl", nested=(), quad=False)

w("")
w("## R8 D0 thresholds")
for y in ["D0_025", "D0_05", "D0_1", "D0_2", "top_share"]:
    est("within linear", E, y=y, xs=["irr"], quad=False)
    est("within linear, drop flags", cl4, y=y, xs=["irr"], quad=False)


def period(lo, hi):
    q = P[P["year_start"].between(lo, hi)]
    gq = q.groupby("district_key")
    o = gq[["D1", "D0", "D0_025", "D0_05", "D0_1", "D0_2", "D2", "even", "top_share", "irr"]].mean()
    o["n_irr"], o["n_y"] = gq["irr"].count(), gq["D1"].count()
    return o


a, b = period(1998, 2000), period(2017, 2019)
L = a.join(b, lsuffix="_0", rsuffix="_1", how="inner")
L = L[(L["n_irr_0"] >= 2) & (L["n_irr_1"] >= 2) & (L["n_y_0"] >= 2) & (L["n_y_1"] >= 2)]
meta = P.groupby("district_key").agg(state_name=("state_name", "first"), shrug_key=("shrug_key", "first"),
                                     brk=("brk_nsa", "sum"),
                                     flagged=("flag_annual_unreliable", lambda s: (s == 1).sum()))
L = L.join(meta).reset_index()
L["dirr"] = L["irr_1"] - L["irr_0"]
w("")
w("Long differences: n={}".format(len(L)))
for y in ["D1", "D0", "D0_025", "D0_05", "D0_1", "D0_2", "top_share"]:
    L["d" + y] = L[y + "_1"] - L[y + "_0"]
    est("LD state FE", L, y="d" + y, xs=["dirr"], fes=["state_name"], nested=(), quad=False)
    est("LD state FE + baseline irr", L, y="d" + y, xs=["dirr", "irr_0"], fes=["state_name"],
        nested=(), quad=False)
    est("LD state FE + baseline irr + baseline outcome", L, y="d" + y,
        xs=["dirr", "irr_0", y + "_0"], fes=["state_name"], nested=(), quad=False)
Lc = L[(L["brk"] == 0) & (L["flagged"] == 0)]
w("  no NSA break and no defect flag in any year: n={}".format(len(Lc)))
for y in ["D1", "D0_05", "D0_1"]:
    est("LD clean", Lc, y="d" + y, xs=["dirr", "irr_0"], fes=["state_name"], nested=(), quad=False)

w("")
w("## R9 cross-section of the LUS share, year by year (state FE, district clusters on 2011 parent)")
cs = []
for yr in range(1998, 2020):
    d = E[E["year_start"] == yr].copy()
    if d["irr"].notna().sum() < 300:
        continue
    r = fe_ols(d, "D1", ["irr", "irr2"], ["state_name"], "shrug_key")
    tp, tse = turning_point(r, "irr", "irr2")
    cs.append((yr, r["n"], r["params"]["irr2"], r["se"]["irr2"], tp, tse))
    w("- {}: n={}, irr2 {:+.2f} ({:.2f}), peak {:.3f} ({:.3f})".format(yr, r["n"], r["params"]["irr2"],
                                                                     r["se"]["irr2"], tp, tse))
cs = pd.DataFrame(cs, columns=["yr", "n", "b2", "se2", "tp", "tpse"])
w("  peaks: median {:.3f}, range {:.3f} to {:.3f}".format(cs["tp"].median(), cs["tp"].min(), cs["tp"].max()))

pd.DataFrame(ROWS).to_csv(OUT + "/irrigation_referee.csv", index=False)
open(OUT + "/irrigation_referee_raw.md", "w", encoding="utf-8").write("\n".join(R))
