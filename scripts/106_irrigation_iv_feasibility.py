"""
106_irrigation_iv_feasibility.py

Instruments for district irrigation that can be built from data on disk.
Neither is the Duflo-Pande river-gradient design or a CGWB aquifer design;
those need data this repo does not hold (see the plan in irrigation_panel_results.md).

  IV-1 shift-share on irrigation SOURCE.
       Z_it = sum_s share_{i,s,base} x G_{s,t,-state}
       share_{i,s,base}: district NIA from source s over NSA, 1998-2000 mean
       G_{s,t,-state}:   growth of NIA from source s since 1998-2000 in all OTHER
                         states, on a balanced set of districts
       It predicts irrigation from where national canal, tank and tubewell
       irrigation went, weighted by how the district irrigated at baseline.

  IV-2 terrain x time. SHRUG elevation and terrain ruggedness (pc11 district
       means) times a linear trend. Flat, low districts are where tubewells and
       canals spread most easily. Inside state x year FE this is a within-state
       differential trend.

Both enter with district FE and state x year FE, errors clustered on shared
2011 parents. The quadratic is instrumented with the instrument and its square.

Outputs: outputs/revision_nov8/irrigation_iv_results.csv, irrigation_iv_raw.md
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:/crop-diversity/scripts")
from _revision_fe import fe_iv, fe_ols, turning_point, pval  # noqa: E402

OUT = r"D:/crop-diversity/outputs/revision_nov8"
SHR = r"D:/SHRUG_2.1_Data/extracted"
R = []


def w(s=""):
    print(s)
    R.append(str(s))


P = pd.read_csv(OUT + "/irrigation_panel_district_year.csv")
P = P[P["year_start"].between(1998, 2019)].copy()
SRC = ["nia_canal", "nia_tank", "nia_tubewell", "nia_otherwell", "nia_other"]

# ---------------------------------------------------------------- shifts
# a source cell left blank in a district-year whose total NIA is reported is a zero
for s in SRC:
    P[s] = np.where(P["nia"].notna(), P[s].fillna(0), np.nan)
w("Source cells blank but total reported are read as zero.")
# balanced set: total NIA reported in >= 20 of the 22 years
P["src_ok"] = P["nia"] > 0
nyr = P.groupby("district_key")["src_ok"].transform("sum")
Bal = P[(nyr >= 20) & P["src_ok"]].copy()
w("# IV feasibility: raw")
w("")
w("Balanced source panel: {} districts reporting NIA in >= 20 of 22 years (others interpolated out of the shift).".format(
    Bal["district_key"].nunique()))
tot = Bal.groupby(["state_name", "year_start"])[SRC].sum()
nat = tot.groupby("year_start").sum()
grow = {}
for st in P["state_name"].unique():
    o = nat - (tot.xs(st, level="state_name") if st in tot.index.get_level_values(0) else 0)
    o = o.reindex(nat.index)
    b = o.loc[1998:2000].mean()
    grow[st] = o / b
w("National NIA by source, 2017-19 over 1998-2000 (balanced districts): " + ", ".join(
    "{} {:.2f}".format(s.replace("nia_", ""), nat.loc[2017:2019, s].mean() / nat.loc[1998:2000, s].mean())
    for s in SRC))

base = (P[P["year_start"].between(1998, 2000)].groupby("district_key")
        .apply(lambda g: pd.Series({s: g[s].sum() / g["nsa"][g[s].notna()].sum()
                                    if g[s].notna().any() else np.nan for s in SRC})))
base.columns = ["sh_" + s.replace("nia_", "") for s in SRC]
P = P.join(base, on="district_key")
Z = np.zeros(len(P))
ok = np.ones(len(P), bool)
for s in SRC:
    g = np.array([grow[st].loc[y, s] if y in grow[st].index else np.nan
                  for st, y in zip(P["state_name"], P["year_start"])])
    sh = P["sh_" + s.replace("nia_", "")].to_numpy()
    Z = Z + np.nan_to_num(sh * g)
    ok &= ~np.isnan(sh)
P["z_ss"] = np.where(ok, Z, np.nan)
P["z_ss2"] = P["z_ss"] ** 2

# ---------------------------------------------------------------- terrain
fp = pd.read_csv(r"D:/crop-diversity/outputs/shrug_covariates/final_panel.csv",
                 usecols=["div_key", "pc11_state_id", "pc11_district_id"])
el = pd.read_stata(SHR + "/shrug-elevation-dta/elevation_pc11dist.dta")
tr = pd.read_stata(SHR + "/shrug-rugged-dta/tri_pc11dist.dta")
for t_ in (el, tr):
    t_["pc11_district_id"] = t_["pc11_district_id"].astype(int)
fp = fp.dropna(subset=["pc11_district_id"])
fp["pc11_district_id"] = fp["pc11_district_id"].astype(int)
fp = fp.merge(el[["pc11_district_id", "elevation_mean"]], on="pc11_district_id", how="left")
fp = fp.merge(tr[["pc11_district_id", "tri_mean"]], on="pc11_district_id", how="left")
P = P.merge(fp[["div_key", "elevation_mean", "tri_mean"]], left_on="district_key",
            right_on="div_key", how="left")
P["t"] = (P["year_start"] - 1998) / 10
for v, nmv in [("elevation_mean", "elev"), ("tri_mean", "tri")]:
    sd = P.drop_duplicates("district_key")[v].std()
    P["z_" + nmv + "_t"] = (P[v] - P[v].mean()) / sd * P["t"]
    P["z_" + nmv + "_t2"] = P["z_" + nmv + "_t"] ** 2
w("Districts with SHRUG terrain: {}".format(P.dropna(subset=["tri_mean"])["district_key"].nunique()))

# ---------------------------------------------------------------- estimation
E = P[P["in_final"] & P["irr_nsa"].notna() & (P["year_start"] >= 2001)].copy()
E["irr"] = E["irr_nsa"]
E["irr2"] = E["irr"] ** 2
E["sy"] = E["state_name"] + "_" + E["year_start"].astype(str)
FES = ["district_key", "sy"]
NEST = ("district_key",)
rows = []
w("")
w("Estimation years 2001-02 to 2019-20 (baseline years excluded).")

for lab, inst in [("shift-share on source", ["z_ss", "z_ss2"]),
                  ("terrain x trend", ["z_tri_t", "z_elev_t", "z_tri_t2", "z_elev_t2"])]:
    d = E.dropna(subset=inst)
    w("")
    w("## " + lab)
    # first stage, linear endogenous only
    fs = fe_ols(d, "irr", inst, FES, "shrug_key", nested=NEST)
    q = len(inst)
    b = fs["params"][inst].to_numpy()
    V = fs["vcov"].loc[inst, inst].to_numpy()
    F = float(b @ np.linalg.inv(V) @ b) / q
    w("First stage for irrigation (n={:,}, districts {}, clusters {}): ".format(
        fs["n"], fs["n_units"], fs["G"]) + "; ".join(
        "{} {:+.4f} ({:.4f})".format(i, fs["params"][i], fs["se"][i]) for i in inst)
      + "; cluster-robust F = {:.2f}".format(F))
    rows.append(dict(iv=lab, stage="first, irr", n=fs["n"], F=F,
                     **{"b_" + i: fs["params"][i] for i in inst}))
    for y in ["D1", "D0_05", "D0", "even"]:
        # linear IV
        r = fe_iv(d, y, ["irr"], [], inst, FES, "shrug_key", nested=NEST)
        ols = fe_ols(d, y, ["irr"], FES, "shrug_key", nested=NEST)
        w("- {} linear: 2SLS {:+.3f} ({:.3f}, p={:.3f}) vs OLS {:+.3f} ({:.3f}); first-stage F {:.2f}".format(
            y, r["params"]["irr"], r["se"]["irr"], pval(r["params"]["irr"], r["se"]["irr"], r["G"]),
            ols["params"]["irr"], ols["se"]["irr"], r["first"][0]["F"]))
        rows.append(dict(iv=lab, stage="2sls linear", y=y, n=r["n"], b=r["params"]["irr"],
                         se=r["se"]["irr"], F=r["first"][0]["F"], ols=ols["params"]["irr"]))
        # quadratic IV
        rq = fe_iv(d, y, ["irr", "irr2"], [], inst, FES, "shrug_key", nested=NEST)
        tp, tse = turning_point(rq, "irr", "irr2")
        w("  {} quadratic: irr {:+.3f} ({:.3f}), irr2 {:+.3f} ({:.3f}), peak {:.3f} ({:.3f}); first-stage F irr {:.2f}, irr2 {:.2f}".format(
            y, rq["params"]["irr"], rq["se"]["irr"], rq["params"]["irr2"], rq["se"]["irr2"],
            tp, tse, rq["first"][0]["F"], rq["first"][1]["F"]))
        rows.append(dict(iv=lab, stage="2sls quadratic", y=y, n=rq["n"], b=rq["params"]["irr"],
                         se=rq["se"]["irr"], b2=rq["params"]["irr2"], se2=rq["se"]["irr2"],
                         tp=tp, tp_se=tse, F=rq["first"][0]["F"], F2=rq["first"][1]["F"]))

pd.DataFrame(rows).to_csv(OUT + "/irrigation_iv_results.csv", index=False)
open(OUT + "/irrigation_iv_raw.md", "w", encoding="utf-8").write("\n".join(R))

# ---------------------------------------------------------------- just-identified shift-share
w("")
w("## shift-share, just identified (z_ss only), and the reduced form")
d = E.dropna(subset=["z_ss"])
fs = fe_ols(d, "irr", ["z_ss"], FES, "shrug_key", nested=NEST)
F1 = (fs["params"]["z_ss"] / fs["se"]["z_ss"]) ** 2
w("First stage: z_ss {:+.4f} ({:.4f}), F = {:.2f}, n={:,}".format(
    fs["params"]["z_ss"], fs["se"]["z_ss"], F1, fs["n"]))
# correlation of the shift-share with baseline tubewell share
dd = d.drop_duplicates("district_key")
w("Corr of baseline tubewell share with the district's z_ss change 2001->2019: {:.3f}".format(
    np.corrcoef(dd["sh_tubewell"].fillna(0), (d.groupby("district_key")["z_ss"].last()
                                               - d.groupby("district_key")["z_ss"].first()
                                               ).reindex(dd["district_key"]).fillna(0))[0, 1]))
rows2 = []
for y in ["D1", "D0_05", "D0", "D2", "even", "cereal_share"]:
    rf = fe_ols(d, y, ["z_ss"], FES, "shrug_key", nested=NEST)
    r = fe_iv(d, y, ["irr"], [], ["z_ss"], FES, "shrug_key", nested=NEST)
    w("- {}: reduced form {:+.3f} ({:.3f}); 2SLS {:+.3f} ({:.3f}, p={:.3f}), n={:,}".format(
        y, rf["params"]["z_ss"], rf["se"]["z_ss"], r["params"]["irr"], r["se"]["irr"],
        pval(r["params"]["irr"], r["se"]["irr"], r["G"]), r["n"]))
    rows2.append(dict(iv="shift-share just-identified", stage="2sls linear", y=y, n=r["n"],
                      b=r["params"]["irr"], se=r["se"]["irr"], F=F1,
                      rf=rf["params"]["z_ss"], rf_se=rf["se"]["z_ss"]))
pd.concat([pd.DataFrame(rows), pd.DataFrame(rows2)]).to_csv(
    OUT + "/irrigation_iv_results.csv", index=False)
open(OUT + "/irrigation_iv_raw.md", "w", encoding="utf-8").write("\n".join(R))
