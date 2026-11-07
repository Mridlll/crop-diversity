"""
104_market_pop_weighted.py

Population-weighted market facilities. docs/markets.html measures each facility as
the share of a district's VILLAGES that report it. Here each facility is instead the
share of the district's RURAL POPULATION living in a village that reports it.

Weights (primary): Census 2011 total population of the village (SHRUG pc11_pca_clean_shrid,
column pc11_pca_tot_p), joined on shrid2 to the Mission Antyodaya village file.
Weights (robustness): Mission Antyodaya's own 2019-20 total_population.

Three versions of every specification are run on the same 606 districts:
  U   village shares everywhere (reproduces the site; checked against the saved CSVs)
  W   population shares for all facility measures, including the connectivity controls
      and the group indices that are built from them
  Wf  population shares for the facility regressors only; controls and indices stay as on the site

Outputs (outputs/revision_nov8/):
  market_pop_weighted_results.md
  market_pw_weighted_district_shares.csv
  market_pw_coefs.csv, market_pw_robust_haat.csv, market_pw_interaction.csv,
  market_pw_typology.csv, market_pw_correlations.csv
"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

SHRUG = r"D:/SHRUG_2.1_Data/extracted"
COV = r"D:/crop-diversity/outputs/shrug_covariates"
OUT = r"D:/crop-diversity/outputs/revision_nov8"
os.makedirs(OUT, exist_ok=True)
DKEY = ["pc11_state_id", "pc11_district_id"]
LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


# ================================================================ village table
key = pd.read_stata(SHRUG + "/shrug-pc-keys-dta/shrid_pc11dist_key.dta")
key = key.dropna(subset=DKEY).drop_duplicates("shrid2")

BIN = {  # raw antyodaya column -> short name used in market_covariates
    "mandi": "mandi", "regular_market": "regular_market", "weekly_haat": "weekly_haat",
    "fpo": "fpo", "is_fertilizer_shop_available": "fert_shop",
    "is_govt_seed_centre_available": "seed_centre",
    "is_soil_testing_centre_available": "soil_test",
    "availability_of_custom_hiring_ce": "custom_hire",
    "availability_of_food_storage_war": "storage",
    "availability_of_farm_gate_proces": "farmgate_proc",
    "internal_pucca_road": "pucca_road", "public_transport": "public_transport",
    "is_bank_available": "bank", "availability_of_railway_station": "railway"}

ay = pd.read_stata(SHRUG + "/shrug-antyodaya-dta/antyodaya_shrid.dta",
                   columns=["shrid2", "total_population"] + list(BIN))
pc = pd.read_stata(SHRUG + "/shrug-pca11-dta/pc11_pca_clean_shrid.dta",
                   columns=["shrid2", "pc11_pca_tot_p"])
ay = ay.merge(key, on="shrid2", how="left").dropna(subset=DKEY)
ay = ay.merge(pc, on="shrid2", how="left")
w("# Population-weighted market facilities")
w("")
w("## Data and method")
w("")
w("Antyodaya village rows with a district key: {:,}.".format(len(ay)))
w("Rows with a Census 2011 population: {:,} ({:.2%}); zero or missing: {:,}.".format(
    ay["pc11_pca_tot_p"].notna().sum(), ay["pc11_pca_tot_p"].notna().mean(),
    int((ay["pc11_pca_tot_p"].fillna(0) <= 0).sum())))
w("Rows with an Antyodaya 2019-20 population: {:,} ({:.2%}); zero or missing: {:,}.".format(
    ay["total_population"].notna().sum(), ay["total_population"].notna().mean(),
    int((ay["total_population"].fillna(0) <= 0).sum())))
both = ay.dropna(subset=["pc11_pca_tot_p", "total_population"])
w("Village-level correlation of the two population series: {:.3f} (log-log {:.3f}).".format(
    both["pc11_pca_tot_p"].corr(both["total_population"]),
    np.log(both.query("pc11_pca_tot_p>0 and total_population>0")["pc11_pca_tot_p"]).corr(
        np.log(both.query("pc11_pca_tot_p>0 and total_population>0")["total_population"]))))
nflag = {k: int(ay[k].isna().sum()) for k in BIN}
w("Missing facility flags (treated as no facility, as in script 79): {}".format(
    {BIN[k]: v for k, v in nflag.items() if v}))

# villages with no Census population cannot be weighted: they get weight 0 in W
# and are counted in the unweighted U exactly as script 79 counts them
ay["pop_c"] = ay["pc11_pca_tot_p"].where(ay["pc11_pca_tot_p"] > 0, 0).fillna(0)
ay["pop_a"] = ay["total_population"].where(ay["total_population"] > 0, 0).fillna(0)


def district_shares(wcol):
    out = {}
    g = ay.groupby(DKEY)
    den = g[wcol].sum() if wcol else g.size()
    for raw, s in BIN.items():
        f = ay[raw].fillna(0).astype(float)
        num = (f * ay[wcol]).groupby([ay[k] for k in DKEY]).sum() if wcol else g[raw].sum()
        out[s] = num / den
    o = pd.DataFrame(out)
    o.index = pd.MultiIndex.from_arrays(
        [pd.to_numeric(o.index.get_level_values(i).astype(str), errors="coerce") for i in range(2)],
        names=DKEY)
    return o


U = district_shares(None)
WC = district_shares("pop_c")
WA = district_shares("pop_a")
pop_dist = ay.groupby(DKEY)[["pop_c", "pop_a"]].sum()

mk = pd.read_csv(COV + "/market_covariates.csv")
fp = pd.read_csv(COV + "/final_panel.csv")
for t in (mk, fp):
    for c in DKEY:
        t[c] = pd.to_numeric(t[c], errors="coerce")


def attach(df, tag):
    t = df.reset_index()
    for c in DKEY:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.rename(columns={s: "m_" + s + "_" + tag for s in BIN.values()})
    return t


UU = attach(U, "u")
chk = mk[DKEY + ["m_" + s + "_vshare" for s in BIN.values()]].merge(UU, on=DKEY)
maxdiff = max(np.abs(chk["m_" + s + "_vshare"] - chk["m_" + s + "_u"]).max() for s in BIN.values())
w("Check: rebuilt unweighted village shares against market_covariates.csv, max abs difference "
  "{:.2e} over {} districts.".format(maxdiff, len(chk)))
assert maxdiff < 1e-9

# group indices, z-scored over the same district set as script 79
GROUPS = {
    "idx_output_market": ["mandi", "regular_market", "weekly_haat"],
    "idx_input_supply": ["fert_shop", "seed_centre", "soil_test", "custom_hire"],
    "idx_postharvest": ["storage", "farmgate_proc"],
    "idx_connectivity": ["pucca_road", "public_transport", "bank", "railway"],
}


def z(s):
    return (s - s.mean()) / s.std()


def build_idx(df_shares):
    # df_shares: district-indexed frame with short names; restrict to the districts in
    # market_covariates, which is the set script 79 standardised over
    ix = pd.MultiIndex.from_frame(mk[DKEY])
    s = df_shares.reindex(ix)
    return pd.DataFrame({n: pd.concat([z(s[c]) for c in cols], axis=1).mean(axis=1)
                         for n, cols in GROUPS.items()}).reset_index()


IU = build_idx(U)
chk2 = mk[DKEY + list(GROUPS)].merge(IU, on=DKEY, suffixes=("", "_re"))
for n in GROUPS:
    assert np.abs(chk2[n] - chk2[n + "_re"]).max() < 1e-9, n
w("Check: rebuilt unweighted group indices reproduce market_covariates.csv exactly.")


def wide(shares, idx, tag):
    t = attach(shares, tag)
    i = idx.rename(columns={n: n + "_" + tag for n in GROUPS})
    for c in DKEY:
        i[c] = pd.to_numeric(i[c], errors="coerce")
    return t.merge(i, on=DKEY)


WCw = wide(WC, build_idx(WC), "w")        # census-2011 population weights
WAw = wide(WA, build_idx(WA), "wa")       # antyodaya-2019 population weights

d = fp.merge(mk, on=DKEY, how="left").merge(WCw, on=DKEY, how="left").merge(WAw, on=DKEY, how="left")
assert len(d) == len(fp)
pdist = pop_dist.reset_index()
for c in DKEY:
    pdist[c] = pd.to_numeric(pdist[c], errors="coerce")
d = d.merge(pdist, on=DKEY, how="left")
d = d[d["in_final"] & d["idx_output_market"].notna()].copy().reset_index(drop=True)
assert len(d) == 606
w("Analysis set: {} districts, {} states, {} clusters (shared 2011 parent).".format(
    len(d), d["state_name"].nunique(), d["shrug_key"].nunique()))
w("Rural population in the analysis districts, Census 2011 weights: {:.0f} million; "
  "villages with weight 0: {}.".format(d["pop_c"].sum() / 1e6, int((ay["pop_c"] == 0).sum())))
d["mandi_per_100k"] = 1e5 * d["m_mandi_n"] / d["ay_total_population"]
d["haat_per_100k"] = 1e5 * d["m_weekly_haat_n"] / d["ay_total_population"]

FAC = [("mandi", "mandi"), ("regular market", "regular_market"), ("weekly haat", "weekly_haat"),
       ("fertiliser shop", "fert_shop"), ("seed centre", "seed_centre"),
       ("soil testing", "soil_test"), ("custom hiring", "custom_hire"),
       ("cold storage", "storage"), ("farm-gate processing", "farmgate_proc"), ("FPO", "fpo")]
LAB = {s: l for l, s in FAC}

# column maps per version: facility -> column, index -> column
def cols(version):
    if version == "U":
        f = {s: "m_" + s + "_vshare" for _, s in FAC}
        ix = {n: n for n in GROUPS}
    elif version == "W":
        f = {s: "m_" + s + "_w" for _, s in FAC}
        ix = {n: n + "_w" for n in GROUPS}
    elif version == "Wf":
        f = {s: "m_" + s + "_w" for _, s in FAC}
        ix = {n: n for n in GROUPS}
    elif version == "WA":
        f = {s: "m_" + s + "_wa" for _, s in FAC}
        ix = {n: n + "_wa" for n in GROUPS}
    return f, ix


CATS = [c for c in d.columns if c.startswith("share_") and d[c].notna().sum() > 300]


def prep(version):
    """Return a copy of d where the standard column names hold this version's measures."""
    f, ix = cols(version)
    x = d.copy()
    for s, c in f.items():
        x["F_" + s] = x[c]
    for n, c in ix.items():
        x[n + "_use"] = x[c]
    return x


CTRL = ("irr_share + I(irr_share**2) + np.log(mean_holding_ha) + pca_st_share + "
        "log_viirs + estab_per_1000pop + idx_connectivity_use + C(state_name)")


def fit(x, dep, regs, extra=""):
    s2 = x.dropna(subset=[dep, "mean_holding_ha", "log_viirs"] + regs)
    f = "{} ~ {} + {}{}".format(dep, " + ".join(regs), CTRL, extra)
    m = smf.ols(f, data=s2).fit(cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})
    return m, s2


def rec(m, v, **kw):
    ci = m.conf_int().loc[v]
    return dict(b=m.params[v], se=m.bse[v], p=m.pvalues[v], lo=ci[0], hi=ci[1],
                n=int(m.nobs), **kw)


VERS = ["U", "W", "Wf", "WA"]
X = {v: prep(v) for v in VERS}

# ================================================================ descriptives and correlations
head = lambda t: (w(""), w("## " + t), w(""))
head("Weighted against unweighted shares, facility by facility")
rows = []
for lab, s in FAC:
    cu, cw, ca = [cols(v)[0][s] for v in ("U", "W", "WA")]
    r = stats.pearsonr(d[cu], d[cw])[0]
    rs = stats.spearmanr(d[cu], d[cw])[0]
    ra = stats.pearsonr(d[cu], d[ca])[0]
    rows.append(dict(facility=lab, mean_unweighted=d[cu].mean(), mean_weighted=d[cw].mean(),
                     mean_weighted_ay_pop=d[ca].mean(),
                     sd_unweighted=d[cu].std(), sd_weighted=d[cw].std(),
                     r_pearson=r, r_spearman=rs, r_ay_pop_weights=ra,
                     zero_u=int((d[cu] == 0).sum()), zero_w=int((d[cw] == 0).sum()),
                     max_abs_gap=np.abs(d[cu] - d[cw]).max(),
                     districts_gap_gt_5pt=int((np.abs(d[cu] - d[cw]) > .05).sum())))
CR = pd.DataFrame(rows)
CR.to_csv(OUT + "/market_pw_correlations.csv", index=False)
w("| facility | mean village share | mean pop share | SD village | SD pop | Pearson r | Spearman r | r (Antyodaya pop weights) | districts with none (v / p) | districts gap > 5 pts |")
w("|---|---|---|---|---|---|---|---|---|---|")
for _, r in CR.iterrows():
    w("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} / {} | {} |".format(
        r.facility, r.mean_unweighted, r.mean_weighted, r.sd_unweighted, r.sd_weighted,
        r.r_pearson, r.r_spearman, r.r_ay_pop_weights, r.zero_u, r.zero_w, r.districts_gap_gt_5pt))
d[["pc11_state_id", "pc11_district_id", "state_name", "district_name", "pop_c", "pop_a"] +
  ["m_" + s + "_vshare" for _, s in FAC] + ["m_" + s + "_w" for _, s in FAC] +
  ["m_" + s + "_wa" for _, s in FAC]].to_csv(OUT + "/market_pw_weighted_district_shares.csv", index=False)

w("")
w("Correlation among the weighted facilities (the structure that the site's section 2 rests on):")
w("")
cmU = d[[cols("U")[0][s] for _, s in FAC]].corr()
cmW = d[[cols("W")[0][s] for _, s in FAC]].corr()
cmU.index = cmU.columns = cmW.index = cmW.columns = [l for l, _ in FAC]
iu = np.triu_indices(10, 1)
nh = [i for i, (l, _) in enumerate(FAC) if l != "weekly haat"]
pairs = [(i, j) for i, j in zip(*iu)]
nine = [(i, j) for i, j in pairs if i in nh and j in nh]
hh = [(2, j) for j in range(10) if j != 2]
w("Nine non-haat facilities, pairwise range: village {:.2f} to {:.2f}; population {:.2f} to {:.2f}.".format(
    min(cmU.values[i, j] for i, j in nine), max(cmU.values[i, j] for i, j in nine),
    min(cmW.values[i, j] for i, j in nine), max(cmW.values[i, j] for i, j in nine)))
w("Weekly haat against the other nine: village {:.2f} to {:.2f}; population {:.2f} to {:.2f}.".format(
    min(cmU.values[i, j] for i, j in hh), max(cmU.values[i, j] for i, j in hh),
    min(cmW.values[i, j] for i, j in hh), max(cmW.values[i, j] for i, j in hh)))
w("Haat vs regular market / mandi: village {:+.2f} / {:+.2f}; population {:+.2f} / {:+.2f}.".format(
    cmU.loc["weekly haat", "regular market"], cmU.loc["weekly haat", "mandi"],
    cmW.loc["weekly haat", "regular market"], cmW.loc["weekly haat", "mandi"]))
cor_rows = []
for l in cmU.columns:
    for m_ in cmU.columns:
        cor_rows.append(dict(a=l, b=m_, r_village=cmU.loc[l, m_], r_pop=cmW.loc[l, m_]))
pd.DataFrame(cor_rows).to_csv(OUT + "/market_pw_corr_matrix_long.csv", index=False)

# ================================================================ coefficient tables
RES = []


def add(block, outcome, var, version, m, extra=None):
    r = rec(m, "F_" + var, block=block, outcome=outcome, facility=var, version=version)
    if extra:
        r.update(extra)
    RES.append(r)


DEPS_H = ["D0_richness", "D1_exp_shannon", "D2_inv_simpson", "evenness_D1_D0"]
BLOCKS = {"output": ["mandi", "weekly_haat", "regular_market"],
          "input": ["fert_shop", "seed_centre", "soil_test", "custom_hire"],
          "post": ["storage", "farmgate_proc", "fpo"]}

for v in VERS:
    x = X[v]
    # solo (the site's main figure and table)
    for lab, s in FAC:
        for dep in DEPS_H + CATS:
            m, _ = fit(x, dep, ["F_" + s])
            add("solo", dep, s, v, m)
    # grouped blocks
    for blk, regs in BLOCKS.items():
        for dep in DEPS_H + CATS:
            m, _ = fit(x, dep, ["F_" + r for r in regs])
            for r in regs:
                add(blk, dep, r, v, m)
R = pd.DataFrame(RES)
# BH across the crop-category outcomes, within block x facility x version
R["q"] = np.nan
for (blk, fac, ver), g in R[R["outcome"].isin(CATS)].groupby(["block", "facility", "version"]):
    R.loc[g.index, "q"] = multipletests(g["p"], method="fdr_bh")[1]
R.to_csv(OUT + "/market_pw_coefs.csv", index=False)

# baseline check against the saved site CSVs
solo_saved = pd.read_csv(COV + "/market_results_solo.csv")
inv = {l: s for l, s in FAC}
chk = []
for _, r in solo_saved.iterrows():
    mine = R[(R.block == "solo") & (R.version == "U") & (R.facility == inv[r["facility"]]) &
             (R.outcome == r["outcome"])].iloc[0]
    chk.append((abs(mine.b - r["b"]), abs(mine.p - r["p"])))
w("")
w("Baseline check: the U version of the solo models reproduces market_results_solo.csv "
  "(max |db| {:.1e}, max |dp| {:.1e}, {} cells).".format(
      max(a for a, _ in chk), max(b for _, b in chk), len(chk)))
assert max(a for a, _ in chk) < 1e-6
out_saved = pd.read_csv(COV + "/market_results_output.csv")
chk = []
for _, r in out_saved.iterrows():
    s = r["var"].replace("m_", "").replace("_vshare", "")
    mine = R[(R.block == "output") & (R.version == "U") & (R.facility == s) &
             (R.outcome == r["outcome"])].iloc[0]
    chk.append(abs(mine.b - r["b"]))
w("Baseline check: the U version of the output-market block reproduces market_results_output.csv "
  "(max |db| {:.1e}).".format(max(chk)))
assert max(chk) < 1e-6


def stars(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))


def sig(p):
    return p < 0.05


def cmp_rows(block, outcomes, facs, ver="W", qcol=False):
    out = []
    for o in outcomes:
        for fc in facs:
            a = R[(R.block == block) & (R.version == "U") & (R.facility == fc) & (R.outcome == o)].iloc[0]
            b = R[(R.block == block) & (R.version == ver) & (R.facility == fc) & (R.outcome == o)].iloc[0]
            flag = []
            if np.sign(a.b) != np.sign(b.b):
                flag.append("SIGN")
            if sig(a.p) != sig(b.p):
                flag.append("SIGNIF " + ("lost" if sig(a.p) else "gained"))
            if qcol and pd.notna(a.q) and (a.q < .05) != (b.q < .05):
                flag.append("q " + ("lost" if a.q < .05 else "gained"))
            out.append((o, fc, a, b, " ".join(flag)))
    return out


def table(rows, qcol=False):
    w("| outcome | facility | village b (SE) | p | pop b (SE) | p | ratio | flag |")
    w("|---|---|---|---|---|---|---|---|")
    for o, fc, a, b, fl in rows:
        ratio = b.b / a.b if abs(a.b) > 1e-12 else np.nan
        ex = ""
        if qcol:
            ex = " q {:.3f}->{:.3f}".format(a.q, b.q) if pd.notna(a.q) else ""
        w("| {} | {} | {:+.3f} ({:.3f}) | {:.3f} | {:+.3f} ({:.3f}) | {:.3f} | {:.2f} | {}{} |".format(
            o.replace("share_", "share: "), LAB[fc], a.b, a.se, a.p, b.b, b.se, b.p, ratio, fl, ex))


MAIN = ["D0_richness", "D1_exp_shannon", "evenness_D1_D0", "share_cereals", "share_pulses"]
head("Headline: one facility at a time (the site's main figure and table)")
w("Each row is a separate model with the full control set and state fixed effects, SEs clustered on the "
  "shared 2011 parent. 'village' is the site; 'pop' is the Census-2011 population share (version W, "
  "controls also population-weighted). Coefficients are the change from no village to all villages (or "
  "no residents to all residents) having the facility.")
w("")
table(cmp_rows("solo", MAIN, [s for _, s in FAC]))

# flags summary across everything
def flags_all(version):
    rows = []
    for blk in ["solo", "output", "input", "post"]:
        for o in DEPS_H + CATS:
            for fc in sorted(R[R.block == blk].facility.unique()):
                a = R[(R.block == blk) & (R.version == "U") & (R.facility == fc) & (R.outcome == o)]
                b = R[(R.block == blk) & (R.version == version) & (R.facility == fc) & (R.outcome == o)]
                if a.empty or b.empty:
                    continue
                a, b = a.iloc[0], b.iloc[0]
                rows.append(dict(block=blk, outcome=o, facility=fc, b_u=a.b, p_u=a.p, q_u=a.q,
                                 b_w=b.b, p_w=b.p, q_w=b.q,
                                 sign=np.sign(a.b) != np.sign(b.b), sig_u=sig(a.p), sig_w=sig(b.p),
                                 sig_change=sig(a.p) != sig(b.p),
                                 q_change=(pd.notna(a.q) and (a.q < .05) != (b.q < .05))))
    return pd.DataFrame(rows)


F = flags_all("W")
F.to_csv(OUT + "/market_pw_flags_all.csv", index=False)
head("Every coefficient that changes sign or p = 0.05 significance (version W)")
w("Cells compared: {} (blocks solo, output, input, post-harvest; outcomes D0, D1, D2, evenness and {} crop "
  "shares).".format(len(F), len(CATS)))
w("Sign changes: {}. Significance changes at 5 percent: {}. BH q changes at 5 percent: {}.".format(
    int(F.sign.sum()), int(F.sig_change.sum()), int(F.q_change.sum())))
w("")
chg = F[F.sign | F.sig_change | F.q_change]
w("| block | outcome | facility | village b (p) | pop b (p) | change |")
w("|---|---|---|---|---|---|")
for _, r in chg.iterrows():
    w("| {} | {} | {} | {:+.3f} ({:.3f}) | {:+.3f} ({:.3f}) | {} |".format(
        r.block, r.outcome.replace("share_", "share: "), LAB[r.facility], r.b_u, r.p_u, r.b_w, r.p_w,
        ("sign " if r.sign else "") + ("signif " if r.sig_change else "") + ("q" if r.q_change else "")))
Fw = {v: flags_all(v) for v in ["Wf", "WA"]}
w("")
w("Same count for the other two weightings: Wf (regressors weighted, controls as on the site): sign "
  "changes {}, 5% significance changes {}; WA (Antyodaya 2019-20 population weights): sign changes {}, "
  "5% significance changes {}.".format(
      int(Fw["Wf"].sign.sum()), int(Fw["Wf"].sig_change.sum()),
      int(Fw["WA"].sign.sum()), int(Fw["WA"].sig_change.sum())))
chg2 = Fw["Wf"][Fw["Wf"].sign | Fw["Wf"].sig_change]
w("")
w("Wf changes in detail:")
w("")
w("| block | outcome | facility | village b (p) | Wf b (p) |")
w("|---|---|---|---|---|")
for _, r in chg2.iterrows():
    w("| {} | {} | {} | {:+.3f} ({:.3f}) | {:+.3f} ({:.3f}) |".format(
        r.block, r.outcome.replace("share_", "share: "), LAB[r.facility], r.b_u, r.p_u, r.b_w, r.p_w))

head("Output-market block (mandi, haat, regular market together): diversity outcomes")
table(cmp_rows("output", DEPS_H, BLOCKS["output"]))
head("Input-supply block: diversity outcomes and the pulse and cereal shares")
table(cmp_rows("input", ["D0_richness", "D1_exp_shannon", "share_cereals", "share_pulses"], BLOCKS["input"], qcol=True))
head("Post-harvest block (storage, farm-gate processing, FPO): the site's FPO crop-share result")
vegs = [c for c in CATS if c in ("share_vegetable", "share_fruits", "share_cereals", "share_pulses")]
table(cmp_rows("post", ["D0_richness", "D1_exp_shannon"] + vegs, BLOCKS["post"], qcol=True), qcol=True)

# BH-surviving crop-category results, village against population
head("Crop-share results that survive Benjamini-Hochberg (q < 0.05), by version")
for blk in ["solo", "output", "input", "post"]:
    for ver in ["U", "W"]:
        g = R[(R.block == blk) & (R.version == ver) & (R.outcome.isin(CATS)) & (R.q < 0.05)]
        w("- {} / {}: {}".format(blk, ver, "; ".join(
            "{} on {} ({:+.1f} pts, q {:.3f})".format(LAB[r.facility], r.outcome.replace("share_", ""),
                                                      100 * r.b, r.q) for _, r in g.iterrows()) or "none"))

# ================================================================ haat robustness (script 80 F3)
head("Weekly haat on the count of crops: the page's robustness table")
HA = "F_weekly_haat"
rows = []
for v in VERS:
    x = X[v]
    SP = [
        ("no adjustment", "D0_richness ~ " + HA, x),
        ("state baseline only", "D0_richness ~ {} + C(state_name)".format(HA), x),
        ("+ irrigation", "D0_richness ~ {} + irr_share + I(irr_share**2) + C(state_name)".format(HA), x),
        ("all measures", "D0_richness ~ {} + {}".format(HA, CTRL), x),
        ("drop shared parents", "D0_richness ~ {} + {}".format(HA, CTRL), x[~x["flag_shared_parent"]]),
        ("drop districts with no haat", "D0_richness ~ {} + {}".format(HA, CTRL), x[x[HA] > 0]),
        ("with nine other facilities", "D0_richness ~ {} + {} + {}".format(
            HA, " + ".join("F_" + s for _, s in FAC if s != "weekly_haat"), CTRL), x),
    ]
    for lab, f, dd in SP:
        s2 = dd.dropna(subset=["D0_richness", HA, "mean_holding_ha", "log_viirs"])
        m = smf.ols(f, data=s2).fit(cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})
        r = rec(m, HA, spec=lab, version=v)
        rows.append(r)
RH = pd.DataFrame(rows)
RH.to_csv(OUT + "/market_pw_robust_haat.csv", index=False)
w("| specification | n | village b (SE) [95% CI] | pop b (SE) [95% CI] | Wf b (p) | WA b (p) |")
w("|---|---|---|---|---|---|")
for lab in RH.spec.unique():
    a, b, c, e = [RH[(RH.spec == lab) & (RH.version == v)].iloc[0] for v in ("U", "W", "Wf", "WA")]
    w("| {} | {} / {} | {:+.2f} ({:.2f}) [{:+.2f}, {:+.2f}] | {:+.2f} ({:.2f}) [{:+.2f}, {:+.2f}] | {:+.2f} ({:.4f}) | {:+.2f} ({:.4f}) |".format(
        lab, a.n, b.n, a.b, a.se, a.lo, a.hi, b.b, b.se, b.lo, b.hi, c.b, c.p, e.b, e.p))
mp = smf.ols("D0_richness ~ haat_per_100k + " + CTRL, data=X["U"].dropna(subset=["haat_per_100k"])).fit(
    cov_type="cluster", cov_kwds={"groups": X["U"].dropna(subset=["haat_per_100k"])["shrug_key"]})
w("")
w("Haats per 100,000 people (a density, already population-normalised, so no weighted version exists): "
  "{:+.4f} (SE {:.4f}, p {:.4f}), n {}.".format(mp.params["haat_per_100k"], mp.bse["haat_per_100k"],
                                               mp.pvalues["haat_per_100k"], int(mp.nobs)))
sdU, sdW = X["U"]["F_weekly_haat"].std(), X["W"]["F_weekly_haat"].std()
bU = RH[(RH.spec == "all measures") & (RH.version == "U")].iloc[0]
bW = RH[(RH.spec == "all measures") & (RH.version == "W")].iloc[0]
w("One SD: village share {:.3f} -> {:+.2f} crops ({:+.2f} SE-scaled); population share {:.3f} -> {:+.2f} "
  "crops. Mean D0 {:.1f}.".format(sdU, bU.b * sdU, bU.se * sdU, sdW, bW.b * sdW, d["D0_richness"].mean()))

# ================================================================ irrigation interaction (script 80 D)
head("Markets and the irrigation downslope")
rows = []
INTER = [("mandi", "F_mandi", False), ("fertiliser shop", "F_fert_shop", False),
         ("output-market index", "idx_output_market_use", True),
         ("input-supply index", "idx_input_supply_use", True),
         ("post-harvest index", "idx_postharvest_use", True)]
for ver in VERS:
    x = X[ver]
    for lab, v, _ in INTER:
        s2 = x.dropna(subset=["D1_exp_shannon", "irr_share", v, "mean_holding_ha", "log_viirs"]).copy()
        s2["_v"] = (s2[v] - s2[v].mean()) / s2[v].std()
        f = ("D1_exp_shannon ~ irr_share*_v + I(irr_share**2)*_v + np.log(mean_holding_ha) + "
             "pca_st_share + log_viirs + estab_per_1000pop + idx_connectivity_use + C(state_name)")
        m = smf.ols(f, data=s2).fit(cov_type="cluster", cov_kwds={"groups": s2["shrug_key"]})
        k2, k1 = "I(irr_share ** 2)", "irr_share"
        b2, b1 = m.params[k2], m.params[k1]
        b2x, b1x = m.params[k2 + ":_v"], m.params[k1 + ":_v"]
        rows.append(dict(version=ver, measure=lab, n=int(m.nobs), curv_lo=b2 - b2x, curv_hi=b2 + b2x,
                         inter=b2x, se=m.bse[k2 + ":_v"], p=m.pvalues[k2 + ":_v"],
                         tp_lo=-(b1 - b1x) / (2 * (b2 - b2x)), tp_hi=-(b1 + b1x) / (2 * (b2 + b2x)),
                         sd_raw=s2[v].std()))
RI = pd.DataFrame(rows)
RI.to_csv(OUT + "/market_pw_interaction.csv", index=False)
w("Interaction of the facility measure (standardised) with irrigation squared, D1 outcome. 'curv' is the "
  "coefficient on irrigation squared one SD below and above the mean; 'peak' is the irrigation share at "
  "the turning point.")
w("")
w("| measure | version | curvature low | curvature high | interaction (SE) | p | peak low | peak high |")
w("|---|---|---|---|---|---|---|---|")
for _, r in RI.iterrows():
    w("| {} | {} | {:+.2f} | {:+.2f} | {:+.2f} ({:.2f}) | {:.3f} | {:.2f} | {:.2f} |".format(
        r.measure, r.version, r.curv_lo, r.curv_hi, r.inter, r.se, r.p, r.tp_lo, r.tp_hi))

# ================================================================ typology
head("The six groups of district")
w("Rule from script 89: facility supply = mean percentile rank across the nine non-haat facilities, cut "
  "into thirds; weekly haat share above or below its median. Recomputed with population shares.")
w("")
NH = [s for _, s in FAC if s != "weekly_haat"]


def typology(version):
    f, _ = cols(version)
    x = d.copy()
    dens = x[[f[s] for s in NH]].rank(pct=True).mean(axis=1)
    c1, c2 = dens.quantile([1 / 3, 2 / 3]).values
    hm = x[f["weekly_haat"]].median()
    band = np.where(dens <= c1, 0, np.where(dens <= c2, 1, 2))
    hi = (x[f["weekly_haat"]] > hm).astype(int).values
    return dens.values, band, hi, band * 2 + hi, (c1, c2, hm)


dU, bU_, hU, tU, cutU = typology("U")
TN = ["Few", "Haats, little else", "Some, few haats", "Some, with haats", "Served, few haats", "Served, with haats"]
typ = pd.DataFrame({"state": d["state_name"], "district": d["district_name"],
                    "band_u": bU_, "haat_hi_u": hU, "type_u": tU, "dens_u": dU})
for ver in ["W", "Wf", "WA"]:
    dd_, bb, hh_, tt, cut = typology(ver)
    typ["band_" + ver], typ["haat_hi_" + ver], typ["type_" + ver], typ["dens_" + ver] = bb, hh_, tt, dd_
typ.to_csv(OUT + "/market_pw_typology.csv", index=False)
w("| comparison | density band changes | haat half changes | six-group type changes | band moves by 2 |")
w("|---|---|---|---|---|")
for ver, lab in [("W", "Census-2011 population weights"), ("WA", "Antyodaya-2019 population weights")]:
    nb = int((typ["band_u"] != typ["band_" + ver]).sum())
    nh_ = int((typ["haat_hi_u"] != typ["haat_hi_" + ver]).sum())
    nt = int((typ["type_u"] != typ["type_" + ver]).sum())
    n2 = int((np.abs(typ["band_u"] - typ["band_" + ver]) == 2).sum())
    w("| {} | {} of 606 ({:.1%}) | {} | {} ({:.1%}) | {} |".format(lab, nb, nb / 606, nh_, nt, nt / 606, n2))
w("")
w("Density-rank correlation, village against population shares: Pearson {:.3f}, Spearman {:.3f}.".format(
    np.corrcoef(typ["dens_u"], typ["dens_W"])[0, 1], stats.spearmanr(typ["dens_u"], typ["dens_W"])[0]))
w("Cuts (density terciles, haat median): village {:.3f}, {:.3f}, {:.3f}; population {:.3f}, {:.3f}, {:.3f}.".format(
    *cutU, *typology("W")[4]))
w("")
w("Transition of density bands (rows village, columns population):")
w("")
ct = pd.crosstab(typ["band_u"].map({0: "bottom", 1: "middle", 2: "top"}),
                 typ["band_W"].map({0: "bottom", 1: "middle", 2: "top"}))
w("```")
w(ct.to_string())
w("```")
w("")
w("Six-group sizes, village against population:")
w("")
w("| group | village n | population n |")
w("|---|---|---|")
for i, nm in enumerate(TN):
    w("| {} | {} | {} |".format(nm, int((typ.type_u == i).sum()), int((typ.type_W == i).sum())))

# profile of types and within-state tests
w("")
w("What the bands show, village against population, unadjusted means of the three outcomes the page quotes:")
w("")
w("| group (population definition) | n | D0 | D1 | cereal share | fert-shop village share | fert-shop pop share |")
w("|---|---|---|---|---|---|---|")
for ver in ["U", "W"]:
    for i, nm in enumerate(TN):
        g = d[typ["type_" + ("u" if ver == "U" else ver)] == i]
        w("| {} ({}) | {} | {:.1f} | {:.2f} | {:.3f} | {:.3f} | {:.3f} |".format(
            nm, "village" if ver == "U" else "pop", len(g), g.D0_richness.mean(), g.D1_exp_shannon.mean(),
            g.share_cereals.mean(), g.m_fert_shop_vshare.mean(), g.m_fert_shop_w.mean()))

w("")
w("Within-state tests (state fixed effects, errors clustered by state, no other controls): outcome on "
  "middle-third and top-third dummies (bottom is the reference) and the haat-rich half.")
w("")
w("| outcome | version | middle third b (p) | top third b (p) | haat-rich half b (p) | joint p, thirds |")
w("|---|---|---|---|---|---|")
tests = []
for dep in ["D1_exp_shannon", "share_cereals", "D0_richness", "evenness_D1_D0"]:
    for ver in ["U", "W"]:
        k = "u" if ver == "U" else ver
        x = d.assign(b_mid=(typ["band_" + k] == 1).astype(float), b_top=(typ["band_" + k] == 2).astype(float),
                     h=typ["haat_hi_" + k].astype(float))
        m = smf.ols(dep + " ~ b_mid + b_top + h + C(state_name)", data=x).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(x["state_name"])[0]})
        jp = float(m.wald_test("b_mid = 0, b_top = 0", scalar=True).pvalue)
        tests.append(dict(outcome=dep, version=ver, mid=m.params["b_mid"], p_mid=m.pvalues["b_mid"],
                          top=m.params["b_top"], p_top=m.pvalues["b_top"], h=m.params["h"],
                          p_h=m.pvalues["h"], p_joint=jp))
        w("| {} | {} | {:+.3f} ({:.3f}) | {:+.3f} ({:.3f}) | {:+.3f} ({:.3f}) | {:.3f} |".format(
            dep, "village" if ver == "U" else "pop", m.params["b_mid"], m.pvalues["b_mid"],
            m.params["b_top"], m.pvalues["b_top"], m.params["h"], m.pvalues["h"], jp))
pd.DataFrame(tests).to_csv(OUT + "/market_pw_typology_tests.csv", index=False)

with open(OUT + "/market_pop_weighted_results.md", "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
print("\nWROTE market_pop_weighted_results.md")
