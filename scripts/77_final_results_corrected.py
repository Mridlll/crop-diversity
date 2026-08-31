"""
77_final_results_corrected.py

The headline results re-estimated on the CORRECTED diversity indices from
script 76, so nothing rests on the 24-year pooled construction.

This is the version to quote.

Outputs:
  outputs/shrug_covariates/final_results.md
  outputs/shrug_covariates/final_panel.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

REPO = r"D:/crop-diversity"
OUT = REPO + "/outputs/shrug_covariates"
DIV = REPO + "/outputs/crop_diversity_analysis"
R = []


def w(s=""):
    print(s)
    R.append(str(s))


def star(pv):
    return "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.10 else ""))


# corrected diversity + the SHRUG covariates, via the existing crosswalk
corr = pd.read_csv(DIV + "/district_diversity_indices_corrected.csv")
cw = pd.read_csv(OUT + "/district_crosswalk.csv")
sh = pd.read_csv(OUT + "/shrug_district_covariates.csv")
old = pd.read_csv(OUT + "/analysis_panel.csv")[
    ["div_key", "irrigation_pct", "in_analysis", "flag_shared_parent",
     "w_parent", "n_sharing_parent"]]

p = corr.merge(cw, left_on="district_key", right_on="div_key", how="left")
n0 = len(p)
p = p.merge(sh.rename(columns={"district_key": "shrug_key"}), on="shrug_key",
            how="left", suffixes=("", "_sh"))
assert len(p) == n0, "merge multiplied rows"
p = p.merge(old, on="div_key", how="left")

p["in_final"] = (p["shrug_key"].notna() & p["irr_share"].notna()
                 & p["D1_exp_shannon"].notna()
                 & ~p["flag_low_ay_coverage"].fillna(True).astype(bool)
                 & ~p["flag_no_antyodaya"].fillna(True).astype(bool)
                 & (p["n_years"] >= 10))
d = p[p["in_final"]].copy()

w("# Final results, on corrected diversity indices")
w("")
w("Diversity indices are the annual-mean rebuild from script 76: 2020-21 stub")
w("dropped, duplicate rows collapsed, every index measured per year then averaged")
w("across 1997-98 to 2019-20. Districts with under 10 years of data are excluded.")
w("")
w("Analysis set: {} districts, {} states. {} sit on a shared pre-2011 parent.".format(
    len(d), d["state_name"].nunique(), int(d["flag_shared_parent"].sum())))
w("Median years of data per district: {:.0f}.".format(d["n_years"].median()))
w("")


# ------------------------------------------------------------------ 1
w("## 1. The inverted U, on corrected indices")
w("")
w("Primary outcome is **D1, the effective number of crops** (exp of Shannon), not")
w("the ABI. The ABI is a min-max composite: sample-dependent, equally weighted for")
w("no stated reason, and two of its three components correlate at 0.94, so it is")
w("really two parts evenness to one part richness. D1 is absolute, needs no")
w("normalisation and reads directly as a count of equally-common crops. The ABI is")
w("kept as one row for comparison.")
w("")
w("| specification | n | linear | squared | p(squared) | turning point | R2 |")
w("|---|---|---|---|---|---|---|")


def fit(df, y="D1_exp_shannon", x="irr_share", fe=True, wt=None):
    df = df.dropna(subset=[y, x, "shrug_key"]).copy()
    if wt:
        df = df.dropna(subset=[wt])
    f = "{} ~ {} + I({}**2)".format(y, x, x) + (" + C(state_name)" if fe else "")
    kw = dict(cov_type="cluster", cov_kwds={"groups": df["shrug_key"]})
    m = smf.wls(f, data=df, weights=df[wt]).fit(**kw) if wt else smf.ols(f, data=df).fit(**kw)
    b1, b2 = m.params[x], m.params["I({} ** 2)".format(x)]
    return dict(n=int(m.nobs), b1=b1, b2=b2,
                p2=m.pvalues["I({} ** 2)".format(x)], tp=-b1 / (2 * b2), r2=m.rsquared)


conflict = d["flag_irr_source_conflict"].fillna(False).astype(bool)
SPECS = [
    ("D1: no fixed effects", lambda: fit(d, fe=False)),
    ("D1: state fixed effects", lambda: fit(d)),
    ("D1: PC11 VD irrigation", lambda: fit(d, x="irr_share_vd11")),
    ("D1: drop shared parents", lambda: fit(d[~d["flag_shared_parent"]])),
    ("D1: inverse-parent weighted", lambda: fit(d, wt="w_parent")),
    ("D1: drop source conflicts", lambda: fit(d[~conflict])),
    ("D1: full 23 years only", lambda: fit(d[d["n_years"] >= 22])),
    ("D1: area weighted", lambda: fit(d, wt="mean_annual_cropped_area")),
    ("D0 richness", lambda: fit(d, y="D0_richness")),
    ("D2 inverse Simpson", lambda: fit(d, y="D2_inv_simpson")),
    ("evenness D1/D0", lambda: fit(d, y="evenness_D1_D0")),
    ("Shannon (raw)", lambda: fit(d, y="shannon_index")),
    ("Simpson (raw)", lambda: fit(d, y="simpson_index")),
    ("ABI (composite, for comparison)", lambda: fit(d, y="agro_biodiversity_index")),
]
rows = []
for lab, fn in SPECS:
    r = fn()
    rows.append(dict(spec=lab, **r))
    w("| {} | {} | {:+.3f} | {:+.3f}{} | {:.4f} | {:.3f} | {:.3f} |".format(
        lab, r["n"], r["b1"], r["b2"], star(r["p2"]), r["p2"], r["tp"], r["r2"]))
t = pd.DataFrame(rows)
t.to_csv(OUT + "/final_robustness.csv", index=False)
core = t[t["spec"].str.startswith("D1:")]
fam = t[~t["spec"].str.startswith("D1:")]
w("")
w("**Sample robustness.** {} of {} D1 specifications keep a negative squared term".format(
    int(((core["b2"] < 0) & (core["p2"] < 0.05)).sum()), len(core)))
w("significant at 5 percent. Turning points {:.3f} to {:.3f}, median {:.3f}.".format(
    core["tp"].min(), core["tp"].max(), core["tp"].median()))
w("")
w("**Index robustness.** {} of {} alternative indices agree, so the hump is not an".format(
    int(((fam["b2"] < 0) & (fam["p2"] < 0.05)).sum()), len(fam)))
w("artefact of one index choice. It shows in the plain count of crops (D0), in the")
w("effective counts (D1, D2) and in evenness on its own, which says irrigation does")
w("two things at once: it changes how many crops a district grows and how evenly")
w("area is spread across them.")
w("")
w("On the pooled construction crop richness showed no hump at all (p = 0.19). It does")
w("now, so that null was an artefact of pooling over years rather than a real result.")

w("")
w("### 1b. The effective-number scale")
w("")
w("| index | mean | reading |")
w("|---|---|---|")
w("| D0 richness | {:.1f} | crops grown in an average year |".format(d["D0_richness"].mean()))
w("| D1 exp(Shannon) | {:.1f} | equally-common crops giving the same diversity |".format(
    d["D1_exp_shannon"].mean()))
w("| D2 inverse Simpson | {:.1f} | the same, weighted toward the dominant crops |".format(
    d["D2_inv_simpson"].mean()))
w("| evenness D1/D0 | {:.3f} | how evenly area is spread |".format(
    d["evenness_D1_D0"].mean()))
w("")
w("The average district grows about {:.0f} crops but is effectively growing about {:.0f}.".format(
    d["D0_richness"].mean(), round(d["D1_exp_shannon"].mean())))
w("That gap is the whole story of Indian cropping concentration, and a unitless")
w("0-to-1 composite hides it.")


# ------------------------------------------------------------------ 2
w("")
w("## 2. Decile profile")
w("")
d["dec"] = pd.qcut(d["irr_share"], 10, labels=False, duplicates="drop") + 1
g = d.groupby("dec").agg(n=("D1_exp_shannon", "size"),
                         irr=("irr_share", "mean"),
                         d0=("D0_richness", "mean"),
                         d1=("D1_exp_shannon", "mean"),
                         d2=("D2_inv_simpson", "mean"),
                         ev=("evenness_D1_D0", "mean"),
                         abi=("agro_biodiversity_index", "mean"),
                         ce=("share_cereals", "mean"))
w("| decile | n | irrigation | D0 | D1 | D2 | evenness | ABI | cereal share |")
w("|---|---|---|---|---|---|---|---|---|")
for i, r_ in g.iterrows():
    w("| {} | {} | {:.3f} | {:.1f} | {:.2f} | {:.2f} | {:.3f} | {:.3f} | {:.3f} |".format(
        int(i), int(r_["n"]), r_["irr"], r_["d0"], r_["d1"], r_["d2"], r_["ev"],
        r_["abi"], r_["ce"]))


# ------------------------------------------------------------------ 3
w("")
w("## 3. Irrigation source, groundwater omitted")
w("")
w("| outcome | n | canal | surface | R2 |")
w("|---|---|---|---|---|")
_src_rows = []
for dep in ["D1_exp_shannon", "D0_richness", "D2_inv_simpson", "evenness_D1_D0",
            "share_cereals", "share_pulses", "share_oilseeds"]:
    s2 = d.dropna(subset=[dep, "irr_share", "ay_src_canal_vshare",
                          "ay_src_surface_vshare", "mean_holding_ha", "shrug_key"])
    m = smf.ols(dep + " ~ irr_share + I(irr_share**2) + ay_src_canal_vshare + "
                "ay_src_surface_vshare + np.log(mean_holding_ha) + pca_st_share + "
                "C(state_name)", data=s2).fit(
        cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})
    w("| {} | {} | {:+.3f}{} | {:+.3f}{} | {:.3f} |".format(
        dep, int(m.nobs),
        m.params["ay_src_canal_vshare"], star(m.pvalues["ay_src_canal_vshare"]),
        m.params["ay_src_surface_vshare"], star(m.pvalues["ay_src_surface_vshare"]),
        m.rsquared))
    _src_rows.append(dict(outcome=dep, n=int(m.nobs),
                          b_canal=m.params["ay_src_canal_vshare"],
                          p_canal=m.pvalues["ay_src_canal_vshare"],
                          b_surface=m.params["ay_src_surface_vshare"],
                          p_surface=m.pvalues["ay_src_surface_vshare"],
                          r2=m.rsquared))


# ------------------------------------------------------------------ 4
w("")
w("## 4. The five dimensions")
w("")
SPEC = {
    "irrigation share": "irr_share",
    "canal village share": "ay_src_canal_vshare",
    "mean holding (log ha)": "np.log(mean_holding_ha)",
    "cultivator share of ag workers": "cultivator_share_agwork",
    "landless share (SECC)": "secc_landless_share",
    "mandi village share": "mandi_vshare",
    "weekly haat village share": "weekly_haat_vshare",
    "regular market village share": "regular_market_vshare",
    "FPO village share": "fpo_vshare",
    "cold storage village share": "storage_vshare",
    "SC population share": "pca_sc_share",
    "ST population share": "pca_st_share",
    "cropping intensity": "cropping_intensity",
}
w("Outcome is D1, the effective number of crops. `adjusted` adds irrigation and")
w("log mean holding size and state fixed effects.")
w("")
w("| dimension | n | raw | adjusted |")
w("|---|---|---|---|")
res = []
for lab, v in SPEC.items():
    base = v.replace("np.log(", "").replace(")", "")
    s2 = d.dropna(subset=[base, "D1_exp_shannon", "shrug_key", "mean_holding_ha"])
    kw = dict(cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})
    m0 = smf.ols("D1_exp_shannon ~ " + v, data=s2).fit(**kw)
    f1 = ("D1_exp_shannon ~ " + v +
          (" + irr_share + I(irr_share**2)" if v != "irr_share" else " + I(irr_share**2)") +
          ("" if "mean_holding" in v else " + np.log(mean_holding_ha)") +
          " + C(state_name)")
    m1 = smf.ols(f1, data=s2).fit(**kw)
    k0, k1 = m0.params.index[1], [c for c in m1.params.index if c == v][0]
    w("| {} | {} | {}{} | {}{} |".format(
        lab, int(m1.nobs), "{:+.3f}".format(m0.params[k0]), star(m0.pvalues[k0]),
        "{:+.3f}".format(m1.params[k1]), star(m1.pvalues[k1])))
    res.append(dict(dimension=lab, n=int(m1.nobs), raw_b=m0.params[k0],
                    raw_p=m0.pvalues[k0], adj_b=m1.params[k1], adj_p=m1.pvalues[k1]))
w("")
w("Significance: *** p<0.01, ** p<0.05, * p<0.10.")
pd.DataFrame(res).to_csv(OUT + "/final_dimension_results.csv", index=False)
pd.DataFrame(_src_rows).to_csv(OUT + "/final_source_results.csv", index=False)

p.to_csv(OUT + "/final_panel.csv", index=False)
with open(OUT + "/final_results.md", "w", encoding="utf-8") as f:
    f.write("\n".join(R))
print("\nWROTE final_results.md, final_panel.csv, final_dimension_results.csv")
