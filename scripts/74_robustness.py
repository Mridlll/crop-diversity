"""
74_robustness.py

The central claim is now: agrobiodiversity rises with irrigation up to about
60 percent of sown area and falls after that. Script 73 found the quadratic
term significant at p = 0.0001 with state fixed effects.

This script tries to break it. Each row is the same quadratic re-estimated on
a different sample or a different irrigation measure. The claim survives only
if the squared term stays negative and significant and the turning point stays
in a narrow band.

Specifications tested:
  1  full analysis set
  2  state fixed effects
  3  PC11 Village Directory irrigation instead of Antyodaya
  4  drop the 155 districts sharing a pre-2011 parent
  5  inverse-parent weighted
  6  drop districts where the two irrigation sources conflict by >0.30
  7  drop Maharashtra, Jharkhand and Assam, where Antyodaya irrigation is
     furthest from published figures
  8  drop the smallest decile of gross cropped area (thin districts)
  9  Shannon index instead of the composite ABI
 10  Simpson index instead of the composite ABI
 11  crop richness instead of the composite ABI
 12  weighted by gross cropped area

Also fixes the SECC landless-share row that script 73 skipped.

Output: outputs/shrug_covariates/robustness.md
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

OUT = r"D:/crop-diversity/outputs/shrug_covariates"
R = []


def w(s=""):
    print(s)
    R.append(str(s))


p = pd.read_csv(OUT + "/analysis_panel.csv")
d0 = p[p["in_analysis"]].copy()

w("# Robustness of the irrigation-diversity inverted U")
w("")
w("Every row is `outcome ~ irrigation + irrigation^2`, standard errors clustered")
w("on the pre-2011 parent district. The claim survives only if the squared term")
w("stays negative and significant with a stable turning point.")
w("")


def fit(df, y="agro_biodiversity_index", x="irr_share", fe=False, wt=None):
    df = df.dropna(subset=[y, x, "shrug_key"]).copy()
    if wt is not None:
        df = df.dropna(subset=[wt])
    f = "{} ~ {} + I({}**2)".format(y, x, x)
    if fe:
        f += " + C(state_name)"
    kw = dict(cov_type="cluster", cov_kwds={"groups": df["shrug_key"]})
    m = (smf.wls(f, data=df, weights=df[wt]).fit(**kw) if wt
         else smf.ols(f, data=df).fit(**kw))
    b1 = m.params[x]
    b2 = m.params["I({} ** 2)".format(x)]
    tp = -b1 / (2 * b2) if b2 != 0 else np.nan
    return dict(n=int(m.nobs), b1=b1, p1=m.pvalues[x], b2=b2,
                p2=m.pvalues["I({} ** 2)".format(x)], tp=tp, r2=m.rsquared)


conflict = d0["flag_irr_source_conflict"].fillna(False).astype(bool)
gca_p10 = d0["gca_ha"].quantile(0.10)
BAD_STATES = ["Maharashtra", "Jharkhand", "Assam"]
assert d0["state_name"].isin(BAD_STATES).any(), "state filter matches nothing"

SPECS = [
    ("1  full set", lambda: fit(d0)),
    ("2  state fixed effects", lambda: fit(d0, fe=True)),
    ("3  PC11 VD irrigation instead", lambda: fit(d0, x="irr_share_vd11", fe=True)),
    ("4  drop shared parents", lambda: fit(d0[~d0["flag_shared_parent"]], fe=True)),
    ("5  inverse-parent weighted", lambda: fit(d0, fe=True, wt="w_parent")),
    ("6  drop source-conflict districts", lambda: fit(d0[~conflict], fe=True)),
    ("7  drop MH, JH, AS", lambda: fit(d0[~d0["state_name"].isin(BAD_STATES)], fe=True)),
    ("8  drop thinnest 10% by GCA", lambda: fit(d0[d0["gca_ha"] > gca_p10], fe=True)),
    ("9  Shannon index", lambda: fit(d0, y="shannon_index", fe=True)),
    ("10 Simpson index", lambda: fit(d0, y="simpson_index", fe=True)),
    ("11 crop richness", lambda: fit(d0, y="crop_richness", fe=True)),
    ("12 weighted by cropped area", lambda: fit(d0, fe=True, wt="gca_ha")),
]

w("| specification | n | linear | squared | p(squared) | turning point | R2 |")
w("|---|---|---|---|---|---|---|")
rows = []
for lab, fn in SPECS:
    try:
        r = fn()
        rows.append(dict(spec=lab, **r))
        star = "***" if r["p2"] < 0.01 else ("**" if r["p2"] < 0.05 else
                                             ("*" if r["p2"] < 0.10 else ""))
        w("| {} | {} | {:+.3f} | {:+.3f}{} | {:.4f} | {:.3f} | {:.3f} |".format(
            lab, r["n"], r["b1"], r["b2"], star, r["p2"], r["tp"], r["r2"]))
    except Exception as e:
        w("| {} | failed | | | | | |".format(lab))
        print("   ", e)

t = pd.DataFrame(rows)
core = t[t["spec"].str.startswith(("1 ", "2 ", "3 ", "4 ", "5 ", "6 ", "7 ", "8 ", "12"))]
neg_sig = int(((core["b2"] < 0) & (core["p2"] < 0.05)).sum())
w("")
w("Of the {} specifications using the composite index, {} keep a negative squared".format(
    len(core), neg_sig))
w("term significant at the 5 percent level.")
if len(core[(core["b2"] < 0) & (core["p2"] < 0.05)]):
    tps = core[(core["b2"] < 0) & (core["p2"] < 0.05)]["tp"]
    w("Turning points across those: min {:.3f}, median {:.3f}, max {:.3f}.".format(
        tps.min(), tps.median(), tps.max()))
t.to_csv(OUT + "/robustness.csv", index=False)


# ------------------------------------------------------------ landless fix
w("")
w("## SECC land ownership (the row script 73 dropped)")
w("")
sub = d0.dropna(subset=["secc_landless_share", "agro_biodiversity_index", "shrug_key"])
w("Non-missing on SECC landless share: {} of {} districts.".format(len(sub), len(d0)))
for lab, f in [("raw", "agro_biodiversity_index ~ secc_landless_share"),
               ("adjusted",
                "agro_biodiversity_index ~ secc_landless_share + irr_share + "
                "np.log(mean_holding_ha) + C(state_name)")]:
    s2 = sub.dropna(subset=["mean_holding_ha"]) if lab == "adjusted" else sub
    m = smf.ols(f, data=s2).fit(cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})
    w("{:9s} coefficient on landless share = {:+.4f} (p = {:.4f}, n = {})".format(
        lab, m.params["secc_landless_share"], m.pvalues["secc_landless_share"],
        int(m.nobs)))
w("")
w("Also SECC owned-acre measures:")
for v in ["secc_unirr_acre_per_hh", "secc_twocrop_acre_per_hh"]:
    s2 = d0.dropna(subset=[v, "agro_biodiversity_index", "shrug_key", "mean_holding_ha"])
    m = smf.ols("agro_biodiversity_index ~ {} + irr_share + C(state_name)".format(v),
                data=s2).fit(cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})
    w("  {:26s} {:+.4f} (p = {:.4f}, n = {})".format(
        v, m.params[v], m.pvalues[v], int(m.nobs)))


# --------------------------------------------------- source, with quadratic
w("")
w("## Irrigation source, with the quadratic in irrigation level included")
w("")
w("Script 73 controlled for irrigation linearly, which is inconsistent with a")
w("quadratic relationship. Re-run with the squared term in place.")
w("")
w("| outcome | n | canal (vs groundwater) | surface (vs groundwater) | R2 |")
w("|---|---|---|---|---|")
for dep in ["agro_biodiversity_index", "shannon_index", "crop_richness",
            "share_cereals", "share_pulses", "share_oilseeds"]:
    s2 = d0.dropna(subset=[dep, "irr_share", "ay_src_canal_vshare",
                           "ay_src_surface_vshare", "mean_holding_ha", "shrug_key"])
    m = smf.ols(dep + " ~ irr_share + I(irr_share**2) + ay_src_canal_vshare + "
                "ay_src_surface_vshare + np.log(mean_holding_ha) + "
                "pca_st_share + C(state_name)", data=s2).fit(
        cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})

    def st(b, pv):
        s = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.10 else ""))
        return "{:+.3f}{}".format(b, s)

    w("| {} | {} | {} | {} | {:.3f} |".format(
        dep, int(m.nobs),
        st(m.params["ay_src_canal_vshare"], m.pvalues["ay_src_canal_vshare"]),
        st(m.params["ay_src_surface_vshare"], m.pvalues["ay_src_surface_vshare"]),
        m.rsquared))
w("")
w("Significance: *** p<0.01, ** p<0.05, * p<0.10.")

with open(OUT + "/robustness.md", "w", encoding="utf-8") as f:
    f.write("\n".join(R))
print("\nWROTE robustness.md and robustness.csv")
