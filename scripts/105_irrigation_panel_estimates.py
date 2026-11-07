"""
105_irrigation_panel_estimates.py

Within-district irrigation and crop diversity, on the panel from script 104.

  A. Cross-section on the same districts (state baseline), for comparison:
     Antyodaya 2019-20 share (the site's) and the LUS 1998-2019 mean share.
  B. Panel, quadratic in NIA/NSA:
       pooled with state x year FE (between + within variation)
       district FE + year FE
       district FE + state x year FE           <- main
     errors clustered on shared 2011 parents.
  C. Robustness of the main spec.
  D. Binned within-district version.
  E. Long differences, 1998-2000 to 2017-2019.

Outputs (outputs/revision_nov8/):
  irrigation_panel_estimates.csv, irrigation_panel_bins.csv,
  irrigation_panel_longdiff.csv, irrigation_panel_results_raw.md
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, r"D:/crop-diversity/scripts")
from _revision_fe import fe_ols, turning_point, pval  # noqa: E402

OUT = r"D:/crop-diversity/outputs/revision_nov8"
R = []


def w(s=""):
    print(s)
    R.append(str(s))


P = pd.read_csv(OUT + "/irrigation_panel_district_year.csv")
P = P[P["in_final"] & P["year_start"].between(1998, 2019)].copy()
P["irr"] = P["irr_nsa"]
P["irr2"] = P["irr"] ** 2
P["sy"] = P["state_name"] + "_" + P["year_start"].astype(str)
P["yr"] = P["year_start"].astype(str)
# district spells: a new spell starts at every NSA territory break
P = P.sort_values(["district_key", "year_start"])
P["spell"] = P["district_key"] + "_" + P.groupby("district_key")["brk_nsa"].cumsum().astype(str)
P["irr_lag"] = P.groupby("district_key")["irr"].shift(1)
P.loc[P.groupby("district_key")["year_start"].diff() != 1, "irr_lag"] = np.nan
P["irr_lag2"] = P["irr_lag"] ** 2
P["irr_ma3_2"] = P["irr_ma3"] ** 2
P["irrg"] = P["irr_gca"].where(P["irr_gca"].between(0, 1.05)).clip(upper=1)
P["irrg2"] = P["irrg"] ** 2
P["w_area"] = P["apy_area"]
E = P[P["irr"].notna()].copy()
YS = [("D1", "effective number of crops (D1)"), ("D0", "crops grown (D0)"),
      ("D0_05", "crops with >=0.5% of area (D0 thresholded)"),
      ("D2", "main crops (D2)"), ("even", "evenness D1/D0"),
      ("cereal_share", "cereal share of area")]
NESTED = ("district_key", "spell")

w("# Irrigation panel: raw results")
w("")
w("Estimation sample: {:,} district-years, {} districts, {} clusters (2011 parents), {} states,"
  " 1998-99 to 2019-20.".format(len(E), E["district_key"].nunique(), E["shrug_key"].nunique(),
                                 E["state_name"].nunique()))
w("Irrigation is LUS net irrigated / net sown. Mean {:.3f}, SD {:.3f}.".format(
    E["irr"].mean(), E["irr"].std()))
# variation decomposition
wd = E["irr"] - E.groupby("district_key")["irr"].transform("mean")
w("Within-district SD {:.3f} ({:.0%} of total variance); within district after state x year"
  " FE: see below.".format(wd.std(), wd.var() / E["irr"].var()))
rows = []


def rec(spec, y, r, quad=True, x="irr", x2="irr2"):
    b1, s1 = r["params"][x], r["se"][x]
    out = dict(spec=spec, y=y, n=r["n"], units=r["n_units"], clusters=r["G"],
               b1=b1, se1=s1, p1=pval(b1, s1, r["G"]))
    if quad:
        b2, s2 = r["params"][x2], r["se"][x2]
        tp, tse = turning_point(r, x, x2)
        out.update(b2=b2, se2=s2, p2=pval(b2, s2, r["G"]), tp=tp, tp_se=tse)
    out["within_r2"] = r.get("within_r2", np.nan)
    rows.append(out)
    return out


# ============================================================ A. cross-section
w("")
w("## A. Cross-section, district means, state baseline")
w("")
X = (P.groupby("district_key")
     .agg(D1=("D1", "mean"), D0=("D0", "mean"), D0_05=("D0_05", "mean"), D2=("D2", "mean"),
          even=("even", "mean"), cereal_share=("cereal_share", "mean"),
          irr_lus=("irr", "mean"), irr_ay=("irr_antyodaya", "first"),
          state_name=("state_name", "first"), shrug_key=("shrug_key", "first"))
     .reset_index())
for lab, xv in [("Antyodaya 2019-20 (site)", "irr_ay"), ("LUS 1998-2019 mean", "irr_lus")]:
    for y, _ in YS:
        d = X.dropna(subset=[y, xv]).copy()
        d["x"], d["x2"] = d[xv], d[xv] ** 2
        d["one"] = 1
        r = fe_ols(d, y, ["x", "x2"], ["state_name"], "shrug_key")
        r["n_units"] = r["n"]
        o = rec("XS state FE: " + lab, y, r, x="x", x2="x2")
        if y == "D1":
            w("{}: D1 = {:+.3f}({:.3f}) irr {:+.3f}({:.3f}) irr^2, p(sq)={:.4f}, peak {:.3f} (se {:.3f}), n={}".format(
                lab, o["b1"], o["se1"], o["b2"], o["se2"], o["p2"], o["tp"], o["tp_se"], o["n"]))
# check against statsmodels with dummies (cross-section LUS)
d = X.dropna(subset=["D1", "irr_lus"])
m = smf.ols("D1 ~ irr_lus + I(irr_lus**2) + C(state_name)", data=d).fit(
    cov_type="cluster", cov_kwds={"groups": d["shrug_key"]})
w("statsmodels check (LUS XS): b2 {:+.4f} se {:.4f}".format(
    m.params["I(irr_lus ** 2)"], m.bse["I(irr_lus ** 2)"]))

# ============================================================ B. panel
w("")
w("## B. Panel, quadratic")
w("")
w("| spec | outcome | n | districts | clusters | irr | irr^2 | p(irr^2) | peak (se) |")
w("|---|---|---|---|---|---|---|---|---|")
SPECS_B = [("pooled, state x year FE", ["sy"]),
           ("district FE + year FE", ["district_key", "yr"]),
           ("district FE + state x year FE", ["district_key", "sy"])]
for lab, fes in SPECS_B:
    for y, _ in YS:
        r = fe_ols(E, y, ["irr", "irr2"], fes, "shrug_key", nested=NESTED)
        if lab.startswith("pooled"):
            r["n_units"] = r["data"]["district_key"].nunique()
        o = rec(lab, y, r)
        w("| {} | {} | {:,} | {} | {} | {:+.3f} ({:.3f}) | {:+.3f} ({:.3f}) | {:.4f} | {:.3f} ({:.3f}) |".format(
            lab, y, o["n"], o["units"], o["clusters"], o["b1"], o["se1"], o["b2"], o["se2"],
            o["p2"], o["tp"], o["tp_se"]))
# check one spec against dummies
d = E.dropna(subset=["D1", "irr"])
d = d[d.groupby("district_key")["D1"].transform("size") > 1]
m = smf.ols("D1 ~ irr + irr2 + C(district_key) + C(sy)", data=d).fit(
    cov_type="cluster", cov_kwds={"groups": d["shrug_key"]})
w("")
w("statsmodels dummy check (main D1): irr {:+.4f} ({:.4f}), irr2 {:+.4f} ({:.4f})".format(
    m.params["irr"], m.bse["irr"], m.params["irr2"], m.bse["irr2"]))

# linear within slope
w("")
w("Linear within-district slope (district FE + state x year FE):")
for y, _ in YS:
    r = fe_ols(E, y, ["irr"], ["district_key", "sy"], "shrug_key", nested=NESTED)
    o = rec("main, linear", y, r, quad=False)
    w("  {}: {:+.3f} ({:.3f}), p={:.4f}, n={:,}".format(y, o["b1"], o["se1"], o["p1"], o["n"]))
    if y == "D1":
        rd = r["data"]
        from _revision_fe import demean
        dm = demean(rd, ["irr"], ["district_key", "sy"])
        w("  residual SD of irrigation after district and state x year FE: {:.3f}".format(dm["irr"].std()))

# ============================================================ C. robustness
w("")
w("## C. Robustness of the main specification (D1)")
w("")
w("| variant | n | irr | irr^2 | p(irr^2) | peak (se) | linear slope (se) |")
w("|---|---|---|---|---|---|---|")


def variant(lab, d, x="irr", x2="irr2", fes=("district_key", "sy"), wt=None, y="D1"):
    r = fe_ols(d, y, [x, x2], list(fes), "shrug_key", nested=NESTED, weights=wt)
    o = rec("robust: " + lab, y, r, x=x, x2=x2)
    rl = fe_ols(d, y, [x], list(fes), "shrug_key", nested=NESTED, weights=wt)
    o["lin"], o["lin_se"] = rl["params"][x], rl["se"][x]
    rows[-1].update(lin=o["lin"], lin_se=o["lin_se"])
    w("| {} | {:,} | {:+.3f} ({:.3f}) | {:+.3f} ({:.3f}) | {:.4f} | {:.3f} ({:.3f}) | {:+.3f} ({:.3f}) |".format(
        lab, o["n"], o["b1"], o["se1"], o["b2"], o["se2"], o["p2"], o["tp"], o["tp_se"],
        o["lin"], o["lin_se"]))


variant("main", E)
variant("drop irrigation spikes", E[~E["irr_spike"]])
variant("district-spell FE (split at NSA territory breaks)", E, fes=("spell", "sy"))
variant("drop district-years with an APY area break", E[~E["brk_apy_area"]])
variant("3-year centred mean of irrigation", P[P["irr_ma3"].notna()], x="irr_ma3", x2="irr_ma3_2")
variant("irrigation lagged one year", P[P["irr_lag"].notna()], x="irr_lag", x2="irr_lag2")
variant("gross irrigated / gross cropped", P[P["irrg"].notna()], x="irrg", x2="irrg2")
nyr = E.groupby("district_key")["irr"].transform("count")
variant("districts with >=15 irrigation years", E[nyr >= 15])
variant("cropped-area weighted", E, wt="w_area")
variant("drop shared 2011 parents", E[~E["flag_shared_parent"].fillna(False).astype(bool)])
for y in ["D0", "D0_05", "D2", "even"]:
    variant("district-spell FE, drop spikes [{}]".format(y), E[~E["irr_spike"]],
            fes=("spell", "sy"), y=y)

# ============================================================ D. bins
w("")
w("## D. Binned within-district version (district FE + state x year FE)")
w("")
edges = [0, .1, .2, .3, .4, .5, .6, .7, .8, 1.0001]
labs = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-100"]
E["bin"] = pd.cut(E["irr"], edges, right=False, labels=labs)
bcols = []
for b in labs[1:]:
    c = "b_" + b.replace("-", "_")
    E[c] = (E["bin"] == b).astype(float)
    bcols.append(c)
# switchers: district-years in a bin that the district does not always occupy
E["n_bins_d"] = E.groupby("district_key")["bin"].transform("nunique")
brows = []
for y in ["D1", "D0", "D0_05", "D2", "even"]:
    r = fe_ols(E, y, bcols, ["district_key", "sy"], "shrug_key", nested=NESTED)
    for b, c in zip(labs[1:], bcols):
        brows.append(dict(y=y, bin=b, coef=r["params"][c], se=r["se"][c],
                          p=pval(r["params"][c], r["se"][c], r["G"]),
                          n_obs=int(E[c].sum()),
                          n_switch=int(E.loc[(E[c] == 1) & (E["n_bins_d"] > 1), "district_key"].nunique())))
B = pd.DataFrame(brows)
B.to_csv(OUT + "/irrigation_panel_bins.csv", index=False)
w("Reference bin 0-10%. Coefficients in units of the outcome.")
w("")
w("| bin | n obs | districts moving through | D1 | D0 | D0_05 | evenness |")
w("|---|---|---|---|---|---|---|")
for b in labs[1:]:
    g = B[B["bin"] == b].set_index("y")
    w("| {} | {} | {} | {:+.3f} ({:.3f}) | {:+.2f} ({:.2f}) | {:+.2f} ({:.2f}) | {:+.4f} ({:.4f}) |".format(
        b, int(g.loc["D1", "n_obs"]), int(g.loc["D1", "n_switch"]),
        g.loc["D1", "coef"], g.loc["D1", "se"], g.loc["D0", "coef"], g.loc["D0", "se"],
        g.loc["D0_05", "coef"], g.loc["D0_05", "se"], g.loc["even", "coef"], g.loc["even", "se"]))
w("")
w("Districts never leaving the 0-10% bin: {}; districts in one bin throughout: {} of {}.".format(
    E.groupby("district_key")["bin"].apply(lambda s: (s == "0-10").all()).sum(),
    int((E.groupby("district_key")["bin"].nunique() == 1).sum()), E["district_key"].nunique()))

# ============================================================ E. long differences
w("")
w("## E. Long differences, 1998-2000 to 2017-2019")
w("")


def period(lo, hi):
    q = P[P["year_start"].between(lo, hi)]
    g = q.groupby("district_key")
    o = g[["D1", "D0", "D0_05", "D2", "even", "cereal_share", "irr"]].mean()
    o["n_irr"] = g["irr"].count()
    o["n_y"] = g["D1"].count()
    return o


a, b = period(1998, 2000), period(2017, 2019)
L = a.join(b, lsuffix="_0", rsuffix="_1", how="inner")
L = L[(L["n_irr_0"] >= 2) & (L["n_irr_1"] >= 2) & (L["n_y_0"] >= 2) & (L["n_y_1"] >= 2)]
meta = P.groupby("district_key")[["state_name", "shrug_key", "brk_nsa"]].agg(
    {"state_name": "first", "shrug_key": "first", "brk_nsa": "sum"})
L = L.join(meta).reset_index()
L["dirr"] = L["irr_1"] - L["irr_0"]
L["dirr2"] = L["irr_1"] ** 2 - L["irr_0"] ** 2
for y in ["D1", "D0", "D0_05", "D2", "even", "cereal_share"]:
    L["d" + y] = L[y + "_1"] - L[y + "_0"]
w("Districts with both windows: {}; mean change in NIA/NSA {:+.3f} (SD {:.3f}); "
  "{} rose by more than 0.10, {} fell by more than 0.10.".format(
      len(L), L["dirr"].mean(), L["dirr"].std(), int((L["dirr"] > .1).sum()),
      int((L["dirr"] < -.1).sum())))
w("Mean change: D1 {:+.3f}, D0 {:+.2f}, D0_05 {:+.2f}.".format(
    L["dD1"].mean(), L["dD0"].mean(), L["dD0_05"].mean()))
ldrows = []
w("")
w("| outcome | sample | n | dIrr | d(Irr^2) | p(quad) | peak (se) | linear dIrr (se) |")
w("|---|---|---|---|---|---|---|---|")
for samp, Ls in [("all", L), ("no NSA territory break", L[L["brk_nsa"] == 0])]:
    for y in ["D1", "D0", "D0_05", "D2", "even", "cereal_share"]:
        r = fe_ols(Ls, "d" + y, ["dirr", "dirr2"], ["state_name"], "shrug_key")
        rl = fe_ols(Ls, "d" + y, ["dirr"], ["state_name"], "shrug_key")
        tp, tse = turning_point(r, "dirr", "dirr2")
        o = dict(sample=samp, y=y, n=r["n"], b1=r["params"]["dirr"], se1=r["se"]["dirr"],
                 b2=r["params"]["dirr2"], se2=r["se"]["dirr2"],
                 p2=pval(r["params"]["dirr2"], r["se"]["dirr2"], r["G"]), tp=tp, tp_se=tse,
                 lin=rl["params"]["dirr"], lin_se=rl["se"]["dirr"],
                 lin_p=pval(rl["params"]["dirr"], rl["se"]["dirr"], rl["G"]))
        ldrows.append(o)
        w("| {} | {} | {} | {:+.3f} ({:.3f}) | {:+.3f} ({:.3f}) | {:.4f} | {:.3f} ({:.3f}) | {:+.3f} ({:.3f}) p={:.3f} |".format(
            y, samp, o["n"], o["b1"], o["se1"], o["b2"], o["se2"], o["p2"], tp, tse,
            o["lin"], o["lin_se"], o["lin_p"]))
# slope of dD1 on dIrr by baseline irrigation tercile: inverted U implies + at low, - at high
w("")
w("Linear long-difference slope of D1 on dIrr, by baseline irrigation group (state FE):")
L["base_grp"] = pd.cut(L["irr_0"], [0, .2, .4, .6, 1.01], right=False,
                       labels=["<20%", "20-40%", "40-60%", ">=60%"])
for gname, Ls in L.groupby("base_grp", observed=True):
    if len(Ls) < 30:
        continue
    rl = fe_ols(Ls, "dD1", ["dirr"], ["state_name"], "shrug_key")
    ldrows.append(dict(sample="base " + str(gname), y="D1", n=rl["n"], lin=rl["params"]["dirr"],
                       lin_se=rl["se"]["dirr"],
                       lin_p=pval(rl["params"]["dirr"], rl["se"]["dirr"], rl["G"])))
    w("  baseline {}: {:+.3f} ({:.3f}), p={:.3f}, n={}".format(
        gname, rl["params"]["dirr"], rl["se"]["dirr"],
        pval(rl["params"]["dirr"], rl["se"]["dirr"], rl["G"]), rl["n"]))
pd.DataFrame(ldrows).to_csv(OUT + "/irrigation_panel_longdiff.csv", index=False)
L.to_csv(OUT + "/irrigation_panel_longdiff_data.csv", index=False)

# ============================================================ F. what drives the within hump
w("")
w("## F. Threats inside the within estimate (D1, district FE + state x year FE)")
w("")
RF = pd.read_csv(r"D:/crop-diversity/outputs/crop_groups/rainfall_district_year.csv",
                 usecols=["district_key", "year", "z_jjas", "z_ond", "z_jfmam_next"])
RF = RF.rename(columns={"year": "year_start"}).drop_duplicates(["district_key", "year_start"])
F = E.merge(RF, on=["district_key", "year_start"], how="left")
F["z_jjas2"] = F["z_jjas"] ** 2
# fixed denominator: NIA over the district's own mean NSA, so a drought that
# shrinks the sown area cannot raise the share on its own
F["nsa_bar"] = F.groupby("district_key")["nsa"].transform("mean")
F["irr_fix"] = (F["nia"] / F["nsa_bar"]).clip(upper=1.2)
F["irr_fix2"] = F["irr_fix"] ** 2
# baseline = first three irrigation years of the district
F = F.sort_values(["district_key", "year_start"])
F["rk"] = F.groupby("district_key").cumcount()
base = F[F["rk"] < 3].groupby("district_key")["irr"].mean().rename("irr_base")
F = F.join(base, on="district_key")
F["irr_x_base"] = F["irr"] * F["irr_base"]
F["dev2"] = (F["irr"] - F["irr_base"]) ** 2
F["irr_lead"] = F.groupby("district_key")["irr"].shift(-1)
F.loc[F.groupby("district_key")["year_start"].diff(-1) != -1, "irr_lead"] = np.nan
F["irr_lead2"] = F["irr_lead"] ** 2
fes = ["district_key", "sy"]


def show(lab, xs, d=F, y="D1"):
    r = fe_ols(d, y, xs, fes, "shrug_key", nested=NESTED)
    rows.append(dict(spec="threat: " + lab, y=y, n=r["n"], clusters=r["G"],
                     **{"b_" + x: r["params"][x] for x in xs},
                     **{"se_" + x: r["se"][x] for x in xs}))
    w("- {} (n={:,}): ".format(lab, r["n"]) + "; ".join(
        "{} {:+.3f} ({:.3f}, p={:.3f})".format(x, r["params"][x], r["se"][x],
                                                pval(r["params"][x], r["se"][x], r["G"])) for x in xs))
    return r


FR = F[F["z_jjas"].notna()]
show("main, rainfall sample", ["irr", "irr2"], FR)
show("plus monsoon, post-monsoon and next-spring rainfall z-scores", ["irr", "irr2", "z_jjas", "z_jjas2", "z_ond", "z_jfmam_next"], FR)
r = show("fixed-denominator irrigation, NIA / own mean NSA", ["irr_fix", "irr_fix2"])
tp, tse = turning_point(r, "irr_fix", "irr_fix2")
w("  peak {:.3f} (se {:.3f})".format(tp, tse))
show("decomposition: slope varies with baseline level vs curvature around own baseline",
     ["irr", "irr_x_base", "dev2"])
show("lead (next year's irrigation) added", ["irr", "irr2", "irr_lead", "irr_lead2"],
     F[F["irr_lead"].notna()])
# slow vs fast: centred 7-year mean against the year's deviation from it
F["irr_slow"] = F.groupby("district_key")["irr"].transform(
    lambda x: x.rolling(7, center=True, min_periods=4).mean())
F["irr_fast"] = F["irr"] - F["irr_slow"]
F["irr_slow2"] = F["irr_slow"] ** 2
F["fast_x_base"] = F["irr_fast"] * F["irr_base"]
F["slow_x_base"] = F["irr_slow"] * F["irr_base"]
FS = F[F["irr_slow"].notna()]
dmS = __import__("_revision_fe").demean(FS, ["irr_slow", "irr_fast"], fes)
w("- within variation split: SD of slow (7-year mean) {:.3f}, SD of fast (deviation) {:.3f}".format(
    dmS["irr_slow"].std(), dmS["irr_fast"].std()))
r = show("slow component quadratic", ["irr_slow", "irr_slow2"], FS)
tp, tse = turning_point(r, "irr_slow", "irr_slow2")
w("  peak {:.3f} (se {:.3f})".format(tp, tse))
show("slow and fast, each with a baseline interaction",
     ["irr_slow", "slow_x_base", "irr_fast", "fast_x_base"], FS)
w("")
w("Reading the decomposition: a negative `irr_x_base` means irrigation gains raise D1 where")
w("a district started dry and lower it where it started wet, which is the cross-district")
w("version of the hump. A negative `dev2` would be curvature inside a single district's")
w("own path.")

pd.DataFrame(rows).to_csv(OUT + "/irrigation_panel_estimates.csv", index=False)
open(OUT + "/irrigation_panel_results_raw.md", "w", encoding="utf-8").write("\n".join(R))
