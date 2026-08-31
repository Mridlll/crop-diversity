"""
73_irrigation_diversity_rebuilt.py

Re-tests the repository's headline finding with the SHRUG-built irrigation
measure, then decomposes irrigation by source.

The finding under test (README, point 1):
  "Semi-irrigated districts are the most diverse (ABI 0.69), more than both
   rainfed (0.62) and fully irrigated (0.67)."
It was built on a scraped irrigation variable covering 503 districts that
agrees with the SHRUG measure at only rho = 0.55.

Robustness dimensions carried through every table:
  - two independent irrigation measures (Mission Antyodaya 2019, PC11 VD 2011)
  - with and without the 155 districts that share a pre-2011 parent
  - inverse-parent weighting and standard errors clustered on the parent
  - with and without districts where the two irrigation sources conflict

Outputs:
  outputs/shrug_covariates/abi_by_regime_rebuilt.csv
  outputs/shrug_covariates/irrigation_source_results.csv
  outputs/shrug_covariates/findings.md
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
d = p[p["in_analysis"]].copy()
w("# Irrigation and agrobiodiversity, rebuilt on SHRUG")
w("")
w("Analysis set: {} districts across {} states.".format(
    len(d), d["state_name"].nunique()))
w("{} of them sit on a shared pre-2011 parent district.".format(
    int(d["flag_shared_parent"].sum())))
w("")


def regime(x):
    return pd.cut(x, [-0.01, 0.40, 0.60, 1.01],
                  labels=["Rainfed", "Semi-irrigated", "Irrigated"])


d["regime_new"] = regime(d["irr_share"])
d["regime_vd11"] = regime(d["irr_share_vd11"])
old = pd.to_numeric(d["irrigation_pct"], errors="coerce")
d["irr_old"] = np.where(old > 1.5, old / 100.0, old)
d["regime_old"] = regime(d["irr_old"])


# ------------------------------------------------------------------ table 1
w("## 1. The headline finding, old measure and new")
w("")
rows = []
for lab, col, wt in [
        ("scraped irrigation_pct (original)", "regime_old", None),
        ("SHRUG Antyodaya 2019", "regime_new", None),
        ("SHRUG Antyodaya, parent-weighted", "regime_new", "w_parent"),
        ("SHRUG Antyodaya, drop shared parents", "regime_new", "drop"),
        ("PC11 Village Directory 2011", "regime_vd11", None)]:
    t = d.copy()
    if wt == "drop":
        t = t[~t["flag_shared_parent"]]
    g = {}
    for r_ in ["Rainfed", "Semi-irrigated", "Irrigated"]:
        s = t[t[col] == r_]
        if len(s) == 0:
            g[r_], g[r_ + "_n"] = np.nan, 0
            continue
        if wt == "w_parent":
            g[r_] = np.average(s["agro_biodiversity_index"], weights=s["w_parent"])
        else:
            g[r_] = s["agro_biodiversity_index"].mean()
        g[r_ + "_n"] = len(s)
    g["measure"] = lab
    g["inverted_U"] = (g["Semi-irrigated"] > g["Rainfed"]) and \
                      (g["Semi-irrigated"] > g["Irrigated"])
    rows.append(g)

t1 = pd.DataFrame(rows)[["measure", "Rainfed", "Rainfed_n", "Semi-irrigated",
                         "Semi-irrigated_n", "Irrigated", "Irrigated_n", "inverted_U"]]
w("| measure | rainfed ABI | n | semi ABI | n | irrigated ABI | n | inverted U? |")
w("|---|---|---|---|---|---|---|---|")
for _, r_ in t1.iterrows():
    w("| {} | {:.3f} | {} | {:.3f} | {} | {:.3f} | {} | {} |".format(
        r_["measure"], r_["Rainfed"], int(r_["Rainfed_n"]), r_["Semi-irrigated"],
        int(r_["Semi-irrigated_n"]), r_["Irrigated"], int(r_["Irrigated_n"]),
        "yes" if r_["inverted_U"] else "NO"))
t1.to_csv(OUT + "/abi_by_regime_rebuilt.csv", index=False)


# ------------------------------------------------------------------ table 2
w("")
w("## 2. Shape of the relationship, without imposing cut-points")
w("")
d["irr_decile"] = pd.qcut(d["irr_share"], 10, labels=False, duplicates="drop") + 1
g2 = d.groupby("irr_decile").agg(
    n=("agro_biodiversity_index", "size"),
    irr=("irr_share", "mean"),
    abi=("agro_biodiversity_index", "mean"),
    shannon=("shannon_index", "mean"),
    richness=("crop_richness", "mean"),
    cereal=("share_cereals", "mean"))
w("| decile of irrigation | n | mean irrigation | ABI | Shannon | richness | cereal share |")
w("|---|---|---|---|---|---|---|")
for i, r_ in g2.iterrows():
    w("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.1f} | {:.3f} |".format(
        int(i), int(r_["n"]), r_["irr"], r_["abi"], r_["shannon"],
        r_["richness"], r_["cereal"]))

w("")
m_lin = smf.ols("agro_biodiversity_index ~ irr_share", data=d).fit(
    cov_type="cluster", cov_kwds={"groups": d["shrug_key"]})
m_qua = smf.ols("agro_biodiversity_index ~ irr_share + I(irr_share**2)", data=d).fit(
    cov_type="cluster", cov_kwds={"groups": d["shrug_key"]})
b1, b2 = m_qua.params["irr_share"], m_qua.params["I(irr_share ** 2)"]
w("Linear:    ABI = {:.4f} {:+.4f} x irrigation   (p = {:.4f}, R2 = {:.3f})".format(
    m_lin.params["Intercept"], m_lin.params["irr_share"],
    m_lin.pvalues["irr_share"], m_lin.rsquared))
w("Quadratic: linear term {:+.4f} (p={:.4f}), squared term {:+.4f} (p={:.4f}), R2 = {:.3f}".format(
    b1, m_qua.pvalues["irr_share"], b2, m_qua.pvalues["I(irr_share ** 2)"], m_qua.rsquared))
if b2 < 0 and m_qua.pvalues["I(irr_share ** 2)"] < 0.10:
    w("Turning point at irrigation = {:.3f} (concave, so an interior maximum).".format(-b1 / (2 * b2)))
else:
    w("No statistically supported interior maximum: the squared term is "
      "{} (p = {:.3f}).".format("positive" if b2 > 0 else "negative",
                                m_qua.pvalues["I(irr_share ** 2)"]))
w("")
mfe = smf.ols("agro_biodiversity_index ~ irr_share + I(irr_share**2) + C(state_name)",
              data=d).fit(cov_type="cluster", cov_kwds={"groups": d["shrug_key"]})
w("With state fixed effects: linear {:+.4f} (p={:.4f}), squared {:+.4f} (p={:.4f}).".format(
    mfe.params["irr_share"], mfe.pvalues["irr_share"],
    mfe.params["I(irr_share ** 2)"], mfe.pvalues["I(irr_share ** 2)"]))


# ------------------------------------------------------------------ table 3
w("")
w("## 3. Irrigation source, holding the level of irrigation constant")
w("")
d["src_canal"] = d["ay_src_canal_vshare"]
d["src_ground"] = d["ay_src_ground_vshare"]
d["src_surface"] = d["ay_src_surface_vshare"]

w("Village-share of each dominant irrigation source, mean across districts:")
w("canal {:.3f}, groundwater {:.3f}, surface {:.3f}, other {:.3f}.".format(
    d["src_canal"].mean(), d["src_ground"].mean(), d["src_surface"].mean(),
    d["ay_src_other_vshare"].mean()))
w("")

g3 = d.groupby(pd.qcut(d["src_canal"], 4, labels=["Q1 least canal", "Q2", "Q3",
                                                  "Q4 most canal"], duplicates="drop")).agg(
    n=("agro_biodiversity_index", "size"), irr=("irr_share", "mean"),
    abi=("agro_biodiversity_index", "mean"), cereal=("share_cereals", "mean"),
    pulses=("share_pulses", "mean"), rich=("crop_richness", "mean"))
w("| canal quartile | n | irrigation | ABI | cereal share | pulse share | richness |")
w("|---|---|---|---|---|---|---|")
for i, r_ in g3.iterrows():
    w("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.1f} |".format(
        i, int(r_["n"]), r_["irr"], r_["abi"], r_["cereal"], r_["pulses"], r_["rich"]))

w("")
res = []
for dep in ["agro_biodiversity_index", "shannon_index", "crop_richness",
            "share_cereals", "share_pulses"]:
    m = smf.ols(dep + " ~ irr_share + src_canal + src_surface + "
                "np.log(mean_holding_ha) + pca_st_share + C(state_name)",
                data=d).fit(cov_type="cluster", cov_kwds={"groups": d["shrug_key"]})
    res.append(dict(outcome=dep, n=int(m.nobs),
                    b_irr=m.params["irr_share"], p_irr=m.pvalues["irr_share"],
                    b_canal=m.params["src_canal"], p_canal=m.pvalues["src_canal"],
                    b_surface=m.params["src_surface"], p_surface=m.pvalues["src_surface"],
                    r2=m.rsquared))
t3 = pd.DataFrame(res)
w("State fixed effects, groundwater is the omitted source, SEs clustered on parent district.")
w("")
w("| outcome | n | irrigation level | canal (vs groundwater) | surface (vs groundwater) | R2 |")
w("|---|---|---|---|---|---|")


def st(b, pv):
    s = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.10 else ""))
    return "{:+.3f}{}".format(b, s)


for _, r_ in t3.iterrows():
    w("| {} | {} | {} | {} | {} | {:.3f} |".format(
        r_["outcome"], int(r_["n"]), st(r_["b_irr"], r_["p_irr"]),
        st(r_["b_canal"], r_["p_canal"]), st(r_["b_surface"], r_["p_surface"]), r_["r2"]))
w("")
w("Significance: *** p<0.01, ** p<0.05, * p<0.10.")
t3.to_csv(OUT + "/irrigation_source_results.csv", index=False)


# ------------------------------------------------------------------ table 4
w("")
w("## 4. The other four dimensions")
w("")
SPECS = {
    "irrigation share": "irr_share",
    "canal village share": "src_canal",
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
    "organic farmer share": "organic_farmer_share",
    "cropping intensity": "cropping_intensity",
}
rows = []
for lab, v in SPECS.items():
    try:
        m0 = smf.ols("agro_biodiversity_index ~ " + v, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["shrug_key"]})
        m1 = smf.ols("agro_biodiversity_index ~ " + v + " + irr_share + "
                     "np.log(mean_holding_ha) + C(state_name)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["shrug_key"]})
        k = [c for c in m1.params.index if c.startswith(v.split("(")[0][:12])][0]
        rows.append(dict(dimension=lab, n=int(m1.nobs),
                         raw_b=m0.params[m0.params.index[1]],
                         raw_p=m0.pvalues[m0.pvalues.index[1]],
                         adj_b=m1.params[k], adj_p=m1.pvalues[k]))
    except Exception as e:
        rows.append(dict(dimension=lab, n=0, raw_b=np.nan, raw_p=np.nan,
                         adj_b=np.nan, adj_p=np.nan))
        print("  skipped {}: {}".format(lab, e))
t4 = pd.DataFrame(rows)
w("Outcome is the Agro-Biodiversity Index. `raw` is bivariate. `adjusted` adds")
w("irrigation share, log mean holding size and state fixed effects.")
w("")
w("| dimension | n | raw | adjusted |")
w("|---|---|---|---|")
for _, r_ in t4.iterrows():
    if np.isnan(r_["raw_b"]):
        w("| {} | - | - | - |".format(r_["dimension"])); continue
    w("| {} | {} | {} | {} |".format(r_["dimension"], int(r_["n"]),
                                     st(r_["raw_b"], r_["raw_p"]),
                                     st(r_["adj_b"], r_["adj_p"])))
t4.to_csv(OUT + "/dimension_results.csv", index=False)

with open(OUT + "/findings.md", "w", encoding="utf-8") as f:
    f.write("\n".join(R))
print("\nWROTE findings.md, abi_by_regime_rebuilt.csv, irrigation_source_results.csv, "
      "dimension_results.csv")
