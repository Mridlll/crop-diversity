"""
103_crop_groups_site_export.py

Builds docs/data/site_crop_groups.json for docs/crop-groups.html.

Reads only the crop-group outputs that passed the referee checks:
  - trends: the referee_fix_* files (Andhra Pradesh niger seed 2015-19 and Gujarat moth
    1998-99 removed), main 2017-19 end window and the 2016-18 end window
  - Andhra Pradesh and Telangana rice: referee_fix_* year series and accounting
  - policy: NFSM 2007 event-time coefficients (script 99 policy) and the identified checks
    (referee_policy_es_checks.csv)
  - rainfall: dose-response bins re-estimated here on the cleaned samples the referee used
    (annual-unreliable unit-years and 1997-98 dropped; mild GCA jumps also dropped for total
    rabi area), reusing script 100's data build and estimator unchanged
Also copies the crop-name mapping and the state outlines used by the markets map.
No APCNF, PMDS or RySS data.

Usage:  python scripts/103_crop_groups_site_export.py [--no-rain]
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = "D:/crop-diversity"
OUTD = REPO + "/outputs/crop_groups"
G = ["rice", "wheat", "maize", "millets", "pulses", "oilseeds", "cotton", "sugarcane", "other"]

ap = argparse.ArgumentParser()
ap.add_argument("--no-rain", action="store_true", help="skip re-estimating the rainfall bins "
                "and keep the ones already in the JSON")
ap.add_argument("--out", default=REPO + "/docs/data/site_crop_groups.json")
args = ap.parse_args()

r1 = lambda v, d=3: None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), d)
J = {"groups": G}

# ------------------------------------------------------------------ crop mapping
cm = pd.read_csv(OUTD + "/crop_group_crop_mapping.csv")
J["mapping"] = {g: sorted(cm.loc[cm.group == g, "crop_name"].str.replace(r"\s+", " ", regex=True)
                          .str.strip().tolist()) for g in G}

# ------------------------------------------------------------------ map: district endpoint
ep = pd.read_csv(OUTD + "/referee_fix_trends_district_endpoint.csv")
pm = pd.read_csv(OUTD + "/crop_group_polygon_match.csv")
units = {}
for _, r in ep.iterrows():
    units[r.unit_key] = {
        "s1": [r1(r["s1_" + g], 2) for g in G],          # share of GCA, 2017-19, pp
        "ds": [r1(r["ds_" + g], 2) for g in G],          # change since 1998-2000, pp
        "gca": r1(r.gca1 / 1000, 1),                     # kha
    }
J["units"] = units
J["poly"] = {r.k: r.unit_key for r in pm.itertuples() if isinstance(r.unit_key, str) and r.unit_key in units}
mm = json.load(open(REPO + "/docs/data/site_market_map.json", encoding="utf-8"))
J["state_rings"] = mm["state_rings"]

# ------------------------------------------------------------------ national
ne = pd.read_csv(OUTD + "/referee_fix_trends_national_endpoint.csv")
ne18 = pd.read_csv(OUTD + "/referee_fix_end18_trends_national_endpoint.csv")
nat = {}
for g in G:
    a = ne[(ne.spec == "main") & (ne.group == g)].iloc[0]
    b = ne18[(ne18.spec == "main") & (ne18.group == g)].iloc[0]
    nat[g] = dict(s0=r1(a.agg_share0, 2), s1=r1(a.agg_share1, 2), d=r1(a.d_agg_pp, 2), se=r1(a.se_agg, 2),
                  dm=r1(a.d_mean_pp, 2), dmse=r1(a.se_mean, 2), da=r1(a.d_area_kha / 1000, 2),
                  dase=r1(a.se_area_kha / 1000, 2), up=int(a.n_gain), down=int(a.n_lose),
                  d18=r1(b.d_agg_pp, 2), se18=r1(b.se_agg, 2), dm18=r1(b.d_mean_pp, 2),
                  dm18se=r1(b.se_mean, 2), n=int(a.n_units))
J["national"] = nat
ns = pd.read_csv(OUTD + "/referee_fix_trends_national_series.csv")
st = ns[ns.panel == "strict"].sort_values("year_start")
J["series"] = {"years": st.year_start.astype(int).tolist(), "n": int(st.n_units.iloc[0]),
               "gca": [r1(v, 1) for v in st.gca_mha],
               "aw": {g: [r1(v, 2) for v in st["aw_" + g]] for g in G}}

# ------------------------------------------------------------------ AP and Telangana rice
ys = pd.read_csv(OUTD + "/referee_fix_trends_apts_year_series.csv")
apts = {}
for reg, key in (("ANDHRA PRADESH", "ap"), ("TELANGANA", "ts")):
    d = ys[ys.region == reg]
    yrs = sorted(d.year_start.unique())
    kh = d[d.season == "kharif"].set_index("year_start").reindex(yrs)
    rb = d[d.season == "rabi"].set_index("year_start").reindex(yrs)
    an = d[d.season == "annual"].set_index("year_start").reindex(yrs)
    gca = an[["area_ha_" + g for g in G]].sum(axis=1)
    apts[key] = {"years": [int(y) for y in yrs], "n": int(an.n_units.max()),
                 "kharif": [r1(v / 1000, 1) for v in kh.area_ha_rice],
                 "rabi": [r1(v / 1000, 1) for v in rb.area_ha_rice],
                 "share": [r1(100 * a / g, 2) for a, g in zip(an.area_ha_rice, gca)]}
acc = pd.read_csv(OUTD + "/referee_fix_trends_apts_accounting.csv")
acc18 = pd.read_csv(OUTD + "/referee_fix_end18_trends_apts_accounting.csv")
for reg, key in (("ANDHRA PRADESH", "ap"), ("TELANGANA", "ts")):
    for tag, A in (("e19", acc), ("e18", acc18)):
        sub = A[A.region == reg].set_index("group")
        apts[key][tag] = {g: dict(d=r1(sub.loc[g, "d_share_pp"], 2), se=r1(sub.loc[g, "se_share"], 2),
                                  da=r1(sub.loc[g, "d_area_kha"], 0), dase=r1(sub.loc[g, "se_area_kha"], 0))
                          for g in G}
J["apts"] = apts

# ------------------------------------------------------------------ policy event studies
pol = {}
for key, f, lab in (("pulses", "policy_es_event_P07_share_gca_pulses.csv", "NFSM-Pulses"),
                    ("rice", "policy_es_event_rice07_share_gca_rice.csv", "NFSM-Rice"),
                    ("wheat", "policy_es_event_wheat07_share_gca_wheat.csv", "NFSM-Wheat")):
    d = pd.read_csv(OUTD + "/" + f).sort_values("e")
    pol[key] = {"label": lab, "e": d.e.astype(int).tolist(),
                "b": [r1(100 * v, 3) for v in d.coef], "se": [r1(100 * v, 3) for v in d.se]}
sm = pd.read_csv(OUTD + "/policy_es_nfsm_summary.csv")
def srow(label, y):
    r = sm[(sm.label == label) & (sm.y == y)].iloc[0]
    return r
for key, label, y in (("pulses", "NFSM-Pulses 2007 [P07]", "share_gca_pulses"),
                      ("rice", "NFSM-Rice 2007", "share_gca_rice"),
                      ("wheat", "NFSM-Wheat 2007", "share_gca_wheat")):
    r = srow(label, y)
    pol[key].update(nt=int(r.n_treat_units), nc=int(r.n_ctrl_units), mean=r1(100 * r.ymean_treat_pre, 1),
                    did=r1(100 * r.did, 2), didse=r1(100 * r.did_se, 2))
ck = pd.read_csv(OUTD + "/referee_policy_es_checks.csv")
c = lambda chk, y: ck[(ck.check == chk) & (ck.y == y)].iloc[0]
w = c("R3 DiD window 2002-2009", "share_gca_pulses")
wl = c("R3 DiD window 2002-2009", "ln_area_ha_pulses")
pol["pulses"].update(w02=r1(100 * w.est, 2), w02se=r1(100 * w.se, 2), w02l=r1(wl.est, 3), w02lse=r1(wl.se, 3))
J["policy"] = pol

# ------------------------------------------------------------------ rainfall dose-response
RAIN_KEYS = ["total", "rice", "pulses", "millets"]
if args.no_rain and os.path.exists(args.out):
    J["rain"] = json.load(open(args.out, encoding="utf-8"))["rain"]
else:
    SRC = REPO + "/scripts/100_referee_rainshock_checks.py"
    code = io.open(SRC, encoding="utf-8").read().split(
        "# =========================================================================== REFEREE CHECKS")[0]
    scratch = os.environ.get("TEMP", REPO + "/outputs/crop_groups")
    code = code.replace('OUT + "/referee_rainshock_log.txt"',
                        repr(scratch.replace("\\", "/") + "/site_export_rain_log.txt"))
    ns_ = {"__name__": "rain_build"}
    exec(compile(code, SRC, "exec"), ns_)
    build, fe_ols, SAMPLES, BLAB, stats = (ns_[k] for k in ("build", "fe_ols", "SAMPLES", "BLAB", "stats"))
    D = {name: build(df, lag, ctrl) for name, (df, lag, ctrl) in SAMPLES.items()}
    XB = ["s_b_" + l for l in BLAB if l != "normal"]

    def bins(d, y, extra=(), shock="s_def"):
        rb = fe_ols(d, y, XB + list(extra))
        out = []
        for l in BLAB:
            if l == "normal":
                out.append(dict(bin=l, b=0.0, se=0.0, n=int(d["s_b_normal"].sum())))
                continue
            out.append(dict(bin=l, b=r1(rb["b"]["s_b_" + l], 4), se=r1(rb["se"]["s_b_" + l], 4),
                            n=int(rb["d"]["s_b_" + l].sum())))
        rd = fe_ols(d, y, ["s_def", "s_exc"] + list(extra))
        b, s = rd["b"][shock], rd["se"][shock]
        p = 2 * stats.t.sf(abs(b / s), rd["g"] - 1)
        return dict(bins=out, dummy=dict(term=shock, b=r1(b, 4), se=r1(s, 4), p=r1(p, 4)),
                    units=int(rd["units"]), n=int(rd["n"]), clusters=int(rd["g"]))

    # "clean" drops unit-years whose annual totals are flagged unreliable and 1997-98;
    # the total-area panel also drops milder GCA jumps. These are the referee's samples.
    clean = lambda d: d[(d.flag_annual_unreliable == 0) & (d.year_start > 1997)]
    clean_m = lambda d: clean(d)[clean(d).f_gca_jump_mild == 0]
    rain = {}
    # A. total rabi area, all districts
    rain["total"] = bins(clean_m(D["rabi_t"]), "la_total")
    # B. rabi rice area where rabi rice is grown (mean rabi rice share >= 2 pp)
    d = D["rabi_t"]
    msh = d.groupby("unit_key")["sh_rice"].transform("mean")
    dr = clean(d[msh >= 2])
    dr = dr[dr.area_ha_rice > 0].copy()
    dr["ln_rice"] = np.log(dr.area_ha_rice)
    rain["rice"] = bins(dr, "ln_rice")
    # C. rabi pulse area, least irrigated third: bins interacted with irrigation third,
    #    estimated on all districts so the state-year baseline is shared
    d = clean(D["rabi_t"].dropna(subset=["irr_t"])).copy()
    XT = []
    for l in BLAB:
        if l == "normal":
            continue
        for t in (1, 2, 3):
            d[f"b_{l}_T{t}"] = d["s_b_" + l] * (d.irr_t == t)
            XT.append(f"b_{l}_T{t}")
    rb = fe_ols(d, "la_pulses", XT)
    out = []
    for l in BLAB:
        if l == "normal":
            out.append(dict(bin=l, b=0.0, se=0.0, n=int((d.s_b_normal * (d.irr_t == 1)).sum())))
            continue
        k = f"b_{l}_T1"
        out.append(dict(bin=l, b=r1(rb["b"][k], 4), se=r1(rb["se"][k], 4), n=int(rb["d"][k].sum())))
    Xi = [f"def_T{t}" for t in (1, 2, 3)] + [f"exc_T{t}" for t in (1, 2, 3)]
    rd = fe_ols(d, "la_pulses", Xi)
    b, s_ = rd["b"]["def_T1"], rd["se"]["def_T1"]
    rain["pulses"] = dict(bins=out, dummy=dict(term="def_T1", b=r1(b, 4), se=r1(s_, 4),
                          p=r1(2 * stats.t.sf(abs(b / s_), rd["g"] - 1), 4)),
                          units=int(d[d.irr_t == 1].unit_key.nunique()), n=int(rd["n"]), clusters=int(rd["g"]))
    # D. next-kharif millet area in millet-growing districts, after the monsoon of t
    d = clean(D["kharif_t1"])
    m = d.groupby("unit_key")["sh_millets"].transform("mean")
    di = d[(m >= 2) & (d.area_ha_millets > 0)].copy()
    di["ln_m"] = np.log(di.area_ha_millets)
    rain["millets"] = bins(di, "ln_m", extra=["c_def", "c_exc"], shock="s_exc")
    rain["labels"] = ns_["BTXT"]
    J["rain"] = rain
    for k in RAIN_KEYS:
        print(k, rain[k]["dummy"], rain[k]["units"], [(b["bin"], b["b"], b["se"]) for b in rain[k]["bins"]])

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(J, f, separators=(",", ":"))
print("wrote", args.out, round(os.path.getsize(args.out) / 1024), "KB;", len(units), "units,",
      len(J["poly"]), "polygons")
