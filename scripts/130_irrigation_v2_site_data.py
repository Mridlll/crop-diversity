"""
130_irrigation_v2_site_data.py

Chart data for the irrigation page after the irrigation v2 note and its referee report
(outputs/revision_nov8/irrigation_v2/irrigation_v2_results.md and
outputs/revision_nov8/irrigation_v2/referee/referee_raw_repro_infer_q1x_q2_q3_q4_mt.md).

Builds the samples by running the data-construction head of 124_referee_irrigation_v2.py
(which in turn runs the head of 123), so every sample matches the referee's. Writes only
docs/data/site_irrigation_v2.json.

Contents
  fifths    within-state means by fifth (quintile) of the Antyodaya 2019-20 irrigated share,
            relative to the least irrigated fifth, for four windows; effective number of crops
  tenths    the same by tenth (decile), full period
  districts each district's effective number of crops net of its state, full period, so the
            mean of the dots in each fifth equals that fifth's coefficient
  spline    two-piece line with a knot at 0.30, read from the referee CSV
  y1991     fifths of the census 1991 irrigated share, full period and 2017-18 to 2021-22
  change    1991-2011 change in irrigated share against the change in the effective number
            of crops (long differences), read from the referee CSV
  sources   per within-state SD of each 2011 source share of sown area, at the mean
            (spec D of the note), plus the referee's checks on the tank term

Every interval uses state-clustered (CR1) standard errors with t(G-1) critical values; every
p-value shown on the site is the wild cluster restricted bootstrap by state (Webb weights,
9,999 draws), the referee's method.

Needs: outputs of 121 and 122 (and 124 for the referee CSV).

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/130_irrigation_v2_site_data.py
"""
import contextlib
import io
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
REPO = r"D:/crop-diversity"
SRC = REPO + "/scripts/"
OUTV = REPO + "/outputs/revision_nov8/irrigation_v2"
REF = OUTV + "/referee/referee_results_repro_infer_q1x_q2_q3_q4_mt.csv"
DST = REPO + "/docs/data/site_irrigation_v2.json"

# ---------------------------------------------------------------- referee head (data + tools)
src = open(SRC + "124_referee_irrigation_v2.py", encoding="utf-8").read()
head = src[:src.index("# ================================================================== repro")]
NS = {"__name__": "ref_124_head"}
_argv = sys.argv
sys.argv = [sys.argv[0], "none"]          # 124 reads its sections from argv
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, "124_head", "exec"), NS)
sys.argv = _argv
NS["RNG"] = np.random.default_rng(20261108)
d0, with_win, c2, c3b, TER, wcr = NS["d0"], NS["with_win"], NS["c2"], NS["c3b"], NS["TER"], NS["wcr"]
ld_frame = NS["ld_frame"]


def r4(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 4)


def bins(df, x, y, nb, extra="", controls=()):
    """Within-state means by quantile bin of x, relative to the lowest bin."""
    keep = [y, x, "state_name"] + list(controls)
    s = df.dropna(subset=keep).copy()
    s["q_"] = pd.qcut(s[x], nb, labels=False, duplicates="drop")
    f = "{} ~ C(q_){} + C(state_name)".format(y, extra)
    m = smf.ols(f, s).fit()
    G = s["state_name"].nunique()
    tcrit = stats.t.ppf(0.975, G - 1)
    out = []
    for k in range(nb):
        g = s[s["q_"] == k]
        row = dict(q=k + 1, n=int(len(g)), lo=r4(g[x].min()), hi=r4(g[x].max()), irr=r4(g[x].mean()))
        if k == 0:
            row.update(b=0.0, ci=[0.0, 0.0], p=None)
        else:
            r = wcr(s, f, "C(q_)[T.{}]".format(k), "state_name", keep + ["q_"])
            row.update(b=r4(r["b"]), se=r4(r["se"]), ci=[r4(r["b"] - tcrit * r["se"]), r4(r["b"] + tcrit * r["se"])],
                       p=r4(r["p_wcr"]), p_t=r4(r["p_tG1"]))
        out.append(row)
    return dict(n=int(len(s)), G=int(G), rows=out), s, m


# ---------------------------------------------------------------- fifths, four windows
dy = pd.read_csv(OUTV + "/div_district_year.csv")
x = dy[dy["year_start"].isin([2020, 2021]) & ~dy["flag_sy"]]
g = x.groupby("district_key").agg(D1=("D1_d6", "mean"), ny=("year_start", "nunique"))
g = g[g["ny"] == 2][["D1"]]
ext = d0.drop(columns=[c for c in ["D1", "D0", "D0_05", "E", "D2"] if c in d0]).merge(
    g, left_on="district_key", right_index=True, how="inner")

WINDOWS = [("full", "1998-99 to 2019-20", with_win(d0, "full", "all")),
           ("w1517", "2015-16 to 2019-20", with_win(d0, "w1517", "all")),
           ("w1721", "2017-18 to 2021-22", with_win(d0, "w1721", "flags_drop6")),
           ("ext", "2020-21 and 2021-22", ext)]
OUT = dict(note="Irrigation v2 chart data, built by scripts/130_irrigation_v2_site_data.py. "
                "Intervals: state-clustered, t(G-1). p: wild cluster restricted bootstrap by state.",
           fifths={}, tenths=None, districts=None)
print("fifths of the Antyodaya 2019-20 irrigated share, D1, within state")
for key, lab, df in WINDOWS:
    res, s, m = bins(df, "irr_share", "D1", 5)
    res["label"] = lab
    OUT["fifths"][key] = res
    print("  {:<22} n {} G {}: {}".format(lab, res["n"], res["G"], ", ".join(
        "{:+.2f} (p {:.3f})".format(r["b"], r["p"]) for r in res["rows"][1:])))
    if key == "full":
        # partial residuals: y minus the state effect, so bin means equal the coefficients
        fe = {k.split("[T.")[1][:-1]: v for k, v in m.params.items() if k.startswith("C(state_name)")}
        a = s["state_name"].map(lambda z: fe.get(z, 0.0)) + m.params["Intercept"]
        s = s.assign(r=s["D1"] - a)
        chk = s.groupby("q_")["r"].mean().values
        assert np.allclose(chk[1:], [r["b"] for r in res["rows"][1:]], atol=1e-3), chk
        assert abs(chk[0]) < 1e-6
        names = json.load(open(REPO + "/docs/data/site_irrigation_story.json", encoding="utf-8"))["districts"]
        nm = {d["k"]: (d["n"], d["s"]) for d in names}
        OUT["districts"] = [dict(k=k, n=nm.get(k, (k.split("|")[-1].title(), ""))[0],
                                 s=nm.get(k, ("", sn.title()))[1], x=r4(xx), y=r4(rr), q=int(q) + 1)
                            for k, sn, xx, rr, q in zip(s["district_key"], s["state_name"], s["irr_share"], s["r"], s["q_"])]

res, _, _ = bins(WINDOWS[0][2], "irr_share", "D1", 10)
OUT["tenths"] = res
print("  tenths, full: " + ", ".join("{:+.2f} (p {:.3f})".format(r["b"], r["p"]) for r in res["rows"][1:]))

# ---------------------------------------------------------------- referee CSV rows
RC = pd.read_csv(REF)


def ref(label):
    r = RC[RC["label"] == label]
    assert len(r) == 1, label
    return r.iloc[0]


OUT["spline"] = {}
for key, t in [("full", "full"), ("w1721", "w1721")]:
    lo, hi = ref("spline {} D1 below".format(t)), ref("spline {} D1 above".format(t))
    tc = stats.t.ppf(0.975, hi["G"] - 1)
    OUT["spline"][key] = dict(knot=0.30, below=r4(lo["b"]), below_p=r4(lo["p_wcr"]),
                              below_ci=[r4(lo["b"] - tc * lo["se"]), r4(lo["b"] + tc * lo["se"])],
                              above=r4(hi["b"]), above_p=r4(hi["p_wcr"]),
                              above_ci=[r4(hi["b"] - tc * hi["se"]), r4(hi["b"] + tc * hi["se"])],
                              n=int(lo["n"]), G=int(hi["G"]))
OUT["share_above_030"] = r4((WINDOWS[0][2]["irr_share"].dropna() > 0.30).mean())

# ---------------------------------------------------------------- 1991 irrigation, fifths
print("fifths of the census 1991 irrigated share, D1, terrain controls, within state")
c91 = c3b.dropna(subset=["irr91", "irr_share"])
OUT["y1991"] = {}
for key, lab, t, v in [("full", "1998-99 to 2019-20", "full", "all"),
                       ("w1721", "2017-18 to 2021-22", "w1721", "flags_drop6")]:
    res, s, _ = bins(with_win(c91, t, v), "irr91", "D1", 5, extra=TER, controls=("elev_km", "tri"))
    res["label"] = lab
    OUT["y1991"][key] = res
    print("  {:<22} n {} G {}: {}".format(lab, res["n"], res["G"], ", ".join(
        "{:+.2f} (p {:.3f})".format(r["b"], r["p"]) for r in res["rows"][1:])))
OUT["y1991_corr_2019"] = r4(c91[["irr91", "irr_share"]].corr().iloc[0, 1])
OUT["y1991_mean"] = r4(c91["irr91"].mean())

# ---------------------------------------------------------------- 1991-2011 change
L = ld_frame()
sub = L[L["ok_1991_2011"]]
OUT["change"] = dict(mean_change=r4(sub["dirr_1991_2011"].mean()), rows=[])
for lab, t, smp in [("Q4 dD1 to cs0913 all", "2009-10 to 2013-14", "all districts"),
                    ("Q4 dD1 to cs0913 area-stable", "2009-10 to 2013-14", "cropped area stable"),
                    ("Q4 dD1 to w1517 all", "2015-16 to 2019-20", "all districts"),
                    ("Q4 dD1 to w1517 area-stable", "2015-16 to 2019-20", "cropped area stable")]:
    r = ref(lab)
    tc = stats.t.ppf(0.975, r["G"] - 1)
    OUT["change"]["rows"].append(dict(to=t, sample=smp, n=int(r["n"]), G=int(r["G"]), b=r4(r["b"]),
                                      ci=[r4(r["b"] - tc * r["se"]), r4(r["b"] + tc * r["se"])],
                                      p=r4(r["p_wcr"])))

# ---------------------------------------------------------------- sources, per within-state SD
print("2011 source shares of sown area, D1, per within-state SD at the mean (spec D)")
KD = NS["KD"] + ["D1"]
s2 = c2.dropna(subset=KD).copy()
G2 = s2["state_name"].nunique()
tc2 = stats.t.ppf(0.975, G2 - 1)
for v in ["sh_can", "sh_gw"]:
    s2[v + "_c"] = s2[v] - s2[v].mean()
FDc = "D1 ~ sh_can_c + I(sh_can_c**2) + sh_gw_c + I(sh_gw_c**2) + sh_tk + sh_ro" + TER + " + C(state_name)"
keepc = ["D1", "sh_can_c", "sh_gw_c", "sh_tk", "sh_ro", "elev_km", "tri", "state_name"]
OUT["sources"] = dict(n=int(len(s2)), G=int(G2), rows=[])
for v, term, lab in [("sh_tk", "sh_tk", "tanks and lakes"), ("sh_can", "sh_can_c", "canals"),
                     ("sh_gw", "sh_gw_c", "wells and tubewells"), ("sh_ro", "sh_ro", "rivers and other")]:
    wsd = float((s2[v] - s2.groupby("state_name")[v].transform("mean")).std())
    r = wcr(s2, FDc, term, "state_name", keepc)
    OUT["sources"]["rows"].append(dict(key=v, name=lab, sd=r4(wsd), mean=r4(s2[v].mean()), b=r4(r["b"] * wsd),
                                       ci=[r4((r["b"] - tc2 * r["se"]) * wsd), r4((r["b"] + tc2 * r["se"]) * wsd)],
                                       p=r4(r["p_wcr"])))
    print("  {:<20} within-state SD {:.3f}: {:+.3f} per SD, 95% {:+.3f} to {:+.3f}, wild p {:.3f}".format(
        lab, wsd, r["b"] * wsd, (r["b"] - tc2 * r["se"]) * wsd, (r["b"] + tc2 * r["se"]) * wsd, r["p_wcr"]))
# Cite one bootstrap run for the tank term: the referee's (124), which also supplies the
# leave-four-states-out p below. This script's own wild draw gives 0.009 for the same term.
OUT["sources"]["rows"][0]["p"] = r4(ref("Q2 D tank share, D1")["p_wcr"])
OUT["sources"]["tank_p_source"] = "referee_results_repro_infer_q1x_q2_q3_q4_mt.csv (124), Webb, 9,999 draws"
inter = ref("Q2 C interaction irr^2 x gw, D1")
OUT["sources"]["gw_vs_canal_curve_p"] = r4(inter["p_wcr"])
raw = open(OUTV + "/referee/referee_raw_repro_infer_q1x_q2_q3_q4_mt.md", encoding="utf-8").read()
ln = [l for l in raw.splitlines() if l.strip().startswith("drop Tamil Nadu+Andhra Pradesh+Telangana+Karnataka")][0]
OUT["sources"]["tank_without_south_p"] = float(ln.split("p wild-cluster (Webb, 9999 draws)")[1].strip())
OUT["sources"]["tank_without_south_n"] = int(ln.split(" n ")[1].split(":")[0])

with open(DST, "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, separators=(",", ":"))
print("wrote " + DST + " ({:.0f} KB)".format(os.path.getsize(DST) / 1024))
