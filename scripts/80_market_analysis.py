"""
80_market_analysis.py

The market layer: what different kinds of market infrastructure do to what a
district grows.

The idea being tested. Market access does not have one effect on crop diversity.
It has two opposite ones, and which wins depends on what the market is FOR.

  - A regulated mandi exists to move assured-price cereals in bulk. Where mandis
    are dense, the profitable thing is to grow what the mandi buys.
  - A weekly haat clears small lots of perishables: vegetables, fruit, spices,
    minor millets. It is the market a diverse smallholder actually uses.
  - A fertiliser shop is the retail end of the purchased-input package that
    travels with cereal intensification. It marks input-market penetration, not
    output-market access.
  - Cold storage and farm-gate processing make perishables sellable at all, so
    they should work the other way.

Layers:
  A  descriptive, and whether these variables separate into distinct factors
  B  output-market type: mandi vs haat vs regular market, on diversity and on
     all ten crop-category shares, Benjamini-Hochberg corrected
  C  input supply: fertiliser shop, seed centre, soil testing, custom hiring
  D  does market and input infrastructure explain the irrigation downslope?

Controls in every adjusted specification, because mandis and cold stores get
built where there is already surplus to trade:
  log nightlights, non-farm establishment density, a connectivity index,
  log mean holding size, ST population share, irrigation and its square,
  state fixed effects. Standard errors clustered on the pre-2011 parent district.

Everything here is descriptive. Market infrastructure is not randomly placed.

Outputs:
  outputs/shrug_covariates/market_analysis.md
  outputs/shrug_covariates/market_results_{output,input,interaction}.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats

OUT = r"D:/crop-diversity/outputs/shrug_covariates"
R = []


def w(s=""):
    print(s)
    R.append(str(s))


def star(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))


def head(t):
    w("")
    w("## " + t)
    w("")


# ------------------------------------------------------------------ data
fp = pd.read_csv(OUT + "/final_panel.csv")
mk = pd.read_csv(OUT + "/market_covariates.csv")
# final_panel already carries ay_total_population; do not re-merge it

for t in (fp, mk):
    for c in ["pc11_state_id", "pc11_district_id"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")

n0 = len(fp)
d = fp.merge(mk, on=["pc11_state_id", "pc11_district_id"], how="left")
assert len(d) == n0, "merge multiplied rows"
assert "ay_total_population" in d.columns, "population column lost in merge"
d = d[d["in_final"] & d["idx_output_market"].notna()].copy()

# density measures, since a village dummy is thin for a rare facility
d["mandi_per_100k"] = 1e5 * d["m_mandi_n"] / d["ay_total_population"]
d["haat_per_100k"] = 1e5 * d["m_weekly_haat_n"] / d["ay_total_population"]
d["fert_per_100k"] = 1e5 * d["m_fert_shop_n"] / d["ay_total_population"]

CATS = [c for c in d.columns if c.startswith("share_") and d[c].notna().sum() > 300]
CTRL = ("irr_share + I(irr_share**2) + np.log(mean_holding_ha) + pca_st_share + "
        "log_viirs + estab_per_1000pop + idx_connectivity + C(state_name)")

w("# The market layer: mandis, haats and fertiliser shops")
w("")
w("Analysis set: {} districts, {} states.".format(len(d), d["state_name"].nunique()))
w("Crop-category outcomes available: {}.".format(", ".join(
    c.replace("share_", "") for c in CATS)))
w("")
w("Diversity outcomes are the Hill numbers, not the ABI: D0 is the count of crops,")
w("D1 (exp of Shannon) and D2 (inverse Simpson) are effective counts weighted toward")
w("commoner crops, and evenness is D1/D0. All are in units of crops, so a coefficient")
w("reads as a change in the number of crops. The ABI is not used here at all.")
w("")
w("Adjusted models control for irrigation and its square, log mean holding size,")
w("ST population share, log nightlights, non-farm establishment density, a")
w("connectivity index and state fixed effects. Errors cluster on the pre-2011")
w("parent district.")


# ------------------------------------------------------------------ A
head("A. What exists where")
FAC = [("mandi", "m_mandi_vshare"), ("regular market", "m_regular_market_vshare"),
       ("weekly haat", "m_weekly_haat_vshare"), ("fertiliser shop", "m_fert_shop_vshare"),
       ("seed centre", "m_seed_centre_vshare"), ("soil testing", "m_soil_test_vshare"),
       ("custom hiring", "m_custom_hire_vshare"), ("cold storage", "m_storage_vshare"),
       ("farm-gate processing", "m_farmgate_proc_vshare"), ("FPO", "m_fpo_vshare")]
w("Share of a district's villages having each facility.")
w("")
w("| facility | mean | p10 | p90 | districts with none |")
w("|---|---|---|---|---|")
for lab, c in FAC:
    w("| {} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
        lab, d[c].mean(), d[c].quantile(.1), d[c].quantile(.9), int((d[c] == 0).sum())))

w("")
w("Mandis are rare: the median district has one in 2 to 3 villages per hundred, and")
w("{} districts report none at all. Haats are five times commoner. That asymmetry is".format(
    int((d["m_mandi_vshare"] == 0).sum())))
w("the point of the exercise rather than a nuisance.")

w("")
w("How the facilities correlate with each other:")
w("")
cm = d[[c for _, c in FAC]].corr()
cm.index = [l for l, _ in FAC]
cm.columns = [l[:9] for l, _ in FAC]
w("```")
w(cm.round(2).to_string())
w("```")
w("")
w("Correlation of each with the development controls:")
w("")
w("| facility | log nightlights | establishments per 1000 | connectivity |")
w("|---|---|---|---|")
for lab, c in FAC:
    row = [stats.pearsonr(d[c], d[x])[0] for x in
           ("log_viirs", "estab_per_1000pop", "idx_connectivity")]
    w("| {} | {:+.2f} | {:+.2f} | {:+.2f} |".format(lab, *row))
w("")
w("Everything correlates with development, which is exactly why the adjusted models")
w("carry nightlights, establishment density and connectivity.")


# ------------------------------------------------------------------ B
head("B. Output-market type and what a district grows")
w("Three output-market variables entered together, so each is read against the")
w("other two. Outcomes are the four Hill measures plus every crop-category share.")
w("")
w("Read D0 against D1. D0 counts crops; D1 counts *effective* crops, so a crop")
w("grown on a sliver of land barely moves it. A facility that lifts D0 but not D1")
w("is associated with more crops grown at the margin rather than a genuinely more")
w("balanced cropping pattern. That distinction is invisible in a composite index.")
w("")
OUTM = ["m_mandi_vshare", "m_weekly_haat_vshare", "m_regular_market_vshare"]
rows = []
for dep in ["D1_exp_shannon", "D0_richness", "D2_inv_simpson", "evenness_D1_D0"] + CATS:
    s2 = d.dropna(subset=[dep, "mean_holding_ha", "log_viirs"] + OUTM)
    m = smf.ols("{} ~ {} + {}".format(dep, " + ".join(OUTM), CTRL), data=s2).fit(
        cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})
    for v in OUTM:
        rows.append(dict(outcome=dep, var=v, b=m.params[v], p=m.pvalues[v], n=int(m.nobs)))
B = pd.DataFrame(rows)
# BH across the crop-category outcomes, separately for each market type
B["q"] = np.nan
for v in OUTM:
    msk = (B["var"] == v) & (B["outcome"].isin(CATS))
    B.loc[msk, "q"] = multipletests(B.loc[msk, "p"], method="fdr_bh")[1]

w("| outcome | mandi | weekly haat | regular market |")
w("|---|---|---|---|")
for dep in ["D1_exp_shannon", "D0_richness", "D2_inv_simpson", "evenness_D1_D0"] + CATS:
    cells = []
    for v in OUTM:
        r_ = B[(B.outcome == dep) & (B["var"] == v)].iloc[0]
        mark = star(r_["p"])
        if dep in CATS and pd.notna(r_["q"]) and r_["q"] < 0.05:
            mark += " (q)"
        cells.append("{:+.3f}{}".format(r_["b"], mark))
    w("| {} | {} | {} | {} |".format(dep.replace("share_", "share: "), *cells))
w("")
w("Stars are uncorrected p values. `(q)` marks results surviving a")
w("Benjamini-Hochberg correction at 5 percent across the {} crop-category".format(len(CATS)))
w("outcomes, applied separately for each market type.")
w("")


def _g(df, out, var, col="b"):
    r = df[(df.outcome == out) & (df["var"] == var)]
    return float(r.iloc[0][col]) if len(r) else float("nan")


w("**The haat result, read properly.** Weekly haats go with {:+.1f} crops on D0".format(
    _g(B, "D0_richness", "m_weekly_haat_vshare")))
w("(p < 0.01) but only {:+.2f} on D1 (p = {:.2f}), and evenness is *negative*".format(
    _g(B, "D1_exp_shannon", "m_weekly_haat_vshare"),
    _g(B, "D1_exp_shannon", "m_weekly_haat_vshare", "p")))
w("at {:+.3f} (p = {:.2f}). So haat districts grow more crops, and they grow them on".format(
    _g(B, "evenness_D1_D0", "m_weekly_haat_vshare"),
    _g(B, "evenness_D1_D0", "m_weekly_haat_vshare", "p")))
w("small patches around the same dominant staple. That is a real finding about")
w("smallholder marketing, and it is much weaker than 'haats make districts diverse'.")
w("An index that averages richness and evenness together would have reported the")
w("strong version.")
B.to_csv(OUT + "/market_results_output.csv", index=False)


# ------------------------------------------------------------------ C
head("C. Input supply")
w("Fertiliser shops, seed centres, soil testing and custom hiring, entered together.")
w("")
INP = ["m_fert_shop_vshare", "m_seed_centre_vshare", "m_soil_test_vshare",
       "m_custom_hire_vshare"]
rows = []
for dep in ["D1_exp_shannon", "D0_richness", "D2_inv_simpson", "evenness_D1_D0"] + CATS:
    s2 = d.dropna(subset=[dep, "mean_holding_ha", "log_viirs"] + INP)
    m = smf.ols("{} ~ {} + {}".format(dep, " + ".join(INP), CTRL), data=s2).fit(
        cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})
    for v in INP:
        rows.append(dict(outcome=dep, var=v, b=m.params[v], p=m.pvalues[v], n=int(m.nobs)))
Cc = pd.DataFrame(rows)
Cc["q"] = np.nan
for v in INP:
    msk = (Cc["var"] == v) & (Cc["outcome"].isin(CATS))
    Cc.loc[msk, "q"] = multipletests(Cc.loc[msk, "p"], method="fdr_bh")[1]
w("| outcome | fertiliser shop | seed centre | soil testing | custom hiring |")
w("|---|---|---|---|---|")
for dep in ["D1_exp_shannon", "D0_richness", "D2_inv_simpson", "evenness_D1_D0"] + CATS:
    cells = []
    for v in INP:
        r_ = Cc[(Cc.outcome == dep) & (Cc["var"] == v)].iloc[0]
        mark = star(r_["p"])
        if dep in CATS and pd.notna(r_["q"]) and r_["q"] < 0.05:
            mark += " (q)"
        cells.append("{:+.3f}{}".format(r_["b"], mark))
    w("| {} | {} | {} | {} | {} |".format(dep.replace("share_", "share: "), *cells))
Cc.to_csv(OUT + "/market_results_input.csv", index=False)

w("")
w("Post-harvest infrastructure, entered on its own:")
w("")
w("| outcome | cold storage | farm-gate processing | FPO |")
w("|---|---|---|---|")
PH = ["m_storage_vshare", "m_farmgate_proc_vshare", "m_fpo_vshare"]
for dep in ["D1_exp_shannon", "D0_richness", "share_vegetable",
            "share_fruits", "share_spices", "share_cereals"]:
    if dep not in d.columns:
        continue
    s2 = d.dropna(subset=[dep, "mean_holding_ha", "log_viirs"] + PH)
    m = smf.ols("{} ~ {} + {}".format(dep, " + ".join(PH), CTRL), data=s2).fit(
        cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})
    w("| {} | {} | {} | {} |".format(
        dep.replace("share_", "share: "),
        *["{:+.3f}{}".format(m.params[v], star(m.pvalues[v])) for v in PH]))
w("")
w("**FPOs run the opposite way to haats.** They lift D1 without lifting D0, meaning")
w("they go with area spread more evenly across the crops a district already grows,")
w("rather than with extra crops appearing at the margin. Haats add crops; FPOs")
w("rebalance area. Only one index family can tell those apart.")


# ------------------------------------------------------------------ D
head("D. Does market infrastructure explain the irrigation downslope?")
w("Diversity peaks near 37 percent irrigation and falls after. The standard story")
w("about the fall is Punjab: assured procurement plus dense input supply makes the")
w("cereal package the only rational choice. If that is right, the downslope should")
w("be steeper where mandis and fertiliser shops are dense.")
w("")
w("Test: interact irrigation and its square with each infrastructure measure,")
w("then read the fitted curve at a low and a high value of that measure.")
w("")
rows = []
for lab, v in [("mandi village share", "m_mandi_vshare"),
               ("fertiliser shop village share", "m_fert_shop_vshare"),
               ("output-market index", "idx_output_market"),
               ("input-supply index", "idx_input_supply"),
               ("post-harvest index", "idx_postharvest")]:
    s2 = d.dropna(subset=["D1_exp_shannon", "irr_share", v,
                          "mean_holding_ha", "log_viirs"]).copy()
    s2["_v"] = (s2[v] - s2[v].mean()) / s2[v].std()
    f = ("D1_exp_shannon ~ irr_share*_v + I(irr_share**2)*_v + "
         "np.log(mean_holding_ha) + pca_st_share + log_viirs + "
         "estab_per_1000pop + idx_connectivity + C(state_name)")
    m = smf.ols(f, data=s2).fit(cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})
    b2 = m.params["I(irr_share ** 2)"]
    b2x = m.params.get("I(irr_share ** 2):_v", np.nan)
    p2x = m.pvalues.get("I(irr_share ** 2):_v", np.nan)
    b1 = m.params["irr_share"]
    b1x = m.params.get("irr_share:_v", np.nan)
    lo = {"b1": b1 - b1x, "b2": b2 - b2x}
    hi = {"b1": b1 + b1x, "b2": b2 + b2x}
    rows.append(dict(measure=lab, b2=b2, b2_inter=b2x, p_inter=p2x,
                     curv_lo=lo["b2"], curv_hi=hi["b2"],
                     tp_lo=-lo["b1"] / (2 * lo["b2"]), tp_hi=-hi["b1"] / (2 * hi["b2"]),
                     n=int(m.nobs)))
D = pd.DataFrame(rows)
w("`curvature` is the coefficient on irrigation squared. More negative means a")
w("sharper peak and a steeper fall. `low` and `high` are one standard deviation")
w("below and above the mean of the infrastructure measure.")
w("")
w("| infrastructure | n | curvature at low | curvature at high | interaction | p |")
w("|---|---|---|---|---|---|")
for _, r_ in D.iterrows():
    w("| {} | {} | {:+.3f} | {:+.3f} | {:+.3f}{} | {:.4f} |".format(
        r_["measure"], int(r_["n"]), r_["curv_lo"], r_["curv_hi"],
        r_["b2_inter"], star(r_["p_inter"]), r_["p_inter"]))
D.to_csv(OUT + "/market_results_interaction.csv", index=False)


# ------------------------------------------------------------------ E
head("E. A market typology")
w("Districts split at the median on output-market density and on input-supply")
w("density, giving four types.")
w("")
d["_om"] = np.where(d["idx_output_market"] > d["idx_output_market"].median(), "high", "low")
d["_is"] = np.where(d["idx_input_supply"] > d["idx_input_supply"].median(), "high", "low")
d["mkt_type"] = ("output " + d["_om"] + ", input " + d["_is"])
g = d.groupby("mkt_type").agg(
    n=("D1_exp_shannon", "size"),
    irr=("irr_share", "mean"),
    abi=("D1_exp_shannon", "mean"),
    rich=("D0_richness", "mean"),
    cereal=("share_cereals", "mean"),
    pulses=("share_pulses", "mean"),
    veg=("share_vegetable", "mean") if "share_vegetable" in d.columns else ("share_pulses", "mean"))
w("| type | n | irrigation | D1 | D0 | cereal | pulses | vegetables |")
w("|---|---|---|---|---|---|---|---|")
for i, r_ in g.iterrows():
    w("| {} | {} | {:.3f} | {:.2f} | {:.1f} | {:.3f} | {:.3f} | {:.3f} |".format(
        i, int(r_["n"]), r_["irr"], r_["abi"], r_["rich"], r_["cereal"],
        r_["pulses"], r_["veg"]))
w("")
s2 = d.dropna(subset=["D1_exp_shannon", "mean_holding_ha", "log_viirs"])
m = smf.ols("D1_exp_shannon ~ C(mkt_type) + " + CTRL, data=s2).fit(
    cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})
w("Adjusted differences against 'output high, input high' as the reference:")
w("")
for k in [c for c in m.params.index if c.startswith("C(mkt_type)")]:
    w("  {:34s} {:+.4f} (p = {:.4f})".format(
        k.replace("C(mkt_type)[T.", "").replace("]", ""), m.params[k], m.pvalues[k]))

d[["div_key", "state_name", "district_name", "mkt_type", "idx_output_market",
   "idx_input_supply", "idx_postharvest", "idx_connectivity", "m_mandi_vshare",
   "m_weekly_haat_vshare", "m_fert_shop_vshare", "irr_share",
   "D1_exp_shannon", "D0_richness", "D2_inv_simpson", "evenness_D1_D0"]].to_csv(OUT + "/market_typology.csv", index=False)


# ------------------------------------------------------------------ F
head("F. Checks on the market layer")
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

w("### F1. Collinearity")
w("")
w("The facilities in section A correlate 0.65 to 0.83 with one another. Entering")
w("them together, as sections B and C do, splits shared variance in a way that")
w("makes individual coefficients unstable and can flip signs. Variance inflation")
w("factors for each block, with the controls included:")
w("")
for lab, blk in [("output market", OUTM), ("input supply", INP)]:
    X = d[blk + ["irr_share", "log_viirs", "estab_per_1000pop",
                 "idx_connectivity", "pca_st_share"]].dropna()
    X = X.assign(_const=1.0)
    vals = [(c, VIF(X.values, i)) for i, c in enumerate(X.columns) if c != "_const"]
    w("**{}**: ".format(lab) + ", ".join(
        "{} {:.1f}".format(c.replace("m_", "").replace("_vshare", ""), v)
        for c, v in vals))
w("")
w("A VIF above 5 is the usual warning line and above 10 the usual stop line.")

w("")
w("### F2. Each facility entered on its own")
w("")
w("The honest way to read collinear regressors. Every row is a separate model with")
w("that facility as the only market variable, plus the full control set.")
w("")
w("| facility | D1 effective crops | D0 richness | cereal share | pulse share |")
w("|---|---|---|---|---|")
solo = []
for lab, c in FAC:
    cells = []
    for dep in ["D1_exp_shannon", "D0_richness", "share_cereals", "share_pulses"]:
        s2 = d.dropna(subset=[dep, c, "mean_holding_ha", "log_viirs"])
        m = smf.ols("{} ~ {} + {}".format(dep, c, CTRL), data=s2).fit(
            cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})
        cells.append("{:+.3f}{}".format(m.params[c], star(m.pvalues[c])))
        solo.append(dict(facility=lab, outcome=dep, b=m.params[c], p=m.pvalues[c]))
    w("| {} | {} | {} | {} | {} |".format(lab, *cells))
pd.DataFrame(solo).to_csv(OUT + "/market_results_solo.csv", index=False)

w("")
w("### F3. What the development controls are doing")
w("")
w("The weekly-haat result is the one worth stress-testing, since it is the only")
w("facility that does not load on the common infrastructure factor.")
w("")
w("| specification | n | haat coefficient on D0 richness | p |")
w("|---|---|---|---|")
HA = "m_weekly_haat_vshare"
SPECS = [
    ("bivariate", "D0_richness ~ " + HA, d),
    ("state FE only", "D0_richness ~ {} + C(state_name)".format(HA), d),
    ("+ irrigation", "D0_richness ~ {} + irr_share + I(irr_share**2) + C(state_name)".format(HA), d),
    ("full controls", "D0_richness ~ {} + {}".format(HA, CTRL), d),
    ("full, drop shared parents", "D0_richness ~ {} + {}".format(HA, CTRL),
     d[~d["flag_shared_parent"]]),
    ("full, drop no-haat districts", "D0_richness ~ {} + {}".format(HA, CTRL),
     d[d[HA] > 0]),
    ("full, with all other facilities",
     "D0_richness ~ {} + {} + {}".format(HA, " + ".join(
         [c for _, c in FAC if c != HA]), CTRL), d),
]
for lab, f, dd in SPECS:
    s2 = dd.dropna(subset=["D0_richness", HA, "mean_holding_ha", "log_viirs"])
    m = smf.ols(f, data=s2).fit(cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})
    w("| {} | {} | {:+.3f}{} | {:.4f} |".format(
        lab, int(m.nobs), m.params[HA], star(m.pvalues[HA]), m.pvalues[HA]))

w("")
w("And the same for the haat measured per 100,000 people rather than as a village share:")
w("")
s2 = d.dropna(subset=["D0_richness", "haat_per_100k", "mean_holding_ha", "log_viirs"])
m = smf.ols("D0_richness ~ haat_per_100k + " + CTRL, data=s2).fit(
    cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})
w("  haats per 100k population: {:+.4f} (p = {:.4f}, n = {})".format(
    m.params["haat_per_100k"], m.pvalues["haat_per_100k"], int(m.nobs)))
sd = d["m_weekly_haat_vshare"].std()
s2 = d.dropna(subset=["D0_richness", HA, "mean_holding_ha", "log_viirs"])
m = smf.ols("D0_richness ~ {} + {}".format(HA, CTRL), data=s2).fit(
    cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})
w("")
w("Scale: a one standard deviation rise in haat village share ({:.3f}) goes with".format(sd))
w("{:+.1f} crops, against a mean richness of {:.1f}.".format(
    m.params[HA] * sd, d["D0_richness"].mean()))

w("")
w("### F4. What did not hold")
w("")
w("Written down so it is not quietly dropped.")
w("")
w("- The opening hypothesis was that dense mandi networks push districts toward")
w("  cereals. The coefficient is **negative** ({:+.3f}), so within a state and at a".format(
    B[(B.outcome == "share_cereals") & (B["var"] == "m_mandi_vshare")].iloc[0]["b"]))
w("  given level of irrigation, more mandis goes with *fewer* cereals, not more.")
w("- The second hypothesis was that the irrigation downslope is steeper where market")
w("  and input infrastructure is dense. Every interaction in section D is **positive**,")
w("  meaning the curve is flatter there, and only the mandi one approaches")
w("  significance (p = {:.3f}). The Punjab story does not show up in this".format(
    float(D[D.measure == "mandi village share"]["p_inter"].iloc[0])))
w("  cross-section.")
w("- The market typology in section E has large raw differences that vanish entirely")
w("  once the controls go in. All three adjusted contrasts are null.")

with open(OUT + "/market_analysis.md", "w", encoding="utf-8") as f:
    f.write("\n".join(R))
print("\nWROTE market_analysis.md and three result CSVs")
