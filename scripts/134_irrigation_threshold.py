"""
134_irrigation_threshold.py

Where does the effective number of crops start to fall with irrigation? Estimates the knot
of the two-piece line that the irrigation page draws (until now fixed at 0.30), instead of
assuming it.

Model (same sample, controls and fixed effects as the referee spline in
124_referee_irrigation_v2.py, section q1x, which the page cites):
    D1_i = a_state + b_lo * min(irr_i, g) + b_hi * max(irr_i - g, 0) + e_i
irr is the Antyodaya 2019-20 irrigated share of net sown area; D1 the district mean of the
effective number of crops over 1998-99 to 2019-20 (scripts/_years.py). State fixed effects
are partialled out (FWL), so every SSR below equals the SSR of the dummy regression.

  A. SSR profile over the knot g = 0.20, 0.21, ..., 0.80; least-squares knot g_hat.
  B. Confidence sets for the knot
       - likelihood-ratio inversion, LR(g) = n (SSR(g) - SSR(g_hat)) / SSR(g_hat):
         chi2(1) cut 3.84 (a continuous kink has regular asymptotics, Hansen 2017, JBES)
         and Hansen (2000) threshold cut 7.35 (conservative);
       - wild cluster bootstrap by state (Webb weights, 1,999 draws): resample from the
         fitted kink model, re-estimate g on the same grid, percentile interval.
  C. Test of no kink (one straight line) against a kink anywhere on the grid: sup over g of
     the state-clustered Wald statistic on the hinge term, p from a restricted wild cluster
     bootstrap by state (null = straight line, Webb, 1,999 draws). Hansen (2000)/(2017) style.
  D. The other indices (D0, D0 at 0.5 percent, D2, evenness), same steps.
  E. Adversarial checks on the knot: leave one state out; recent windows (a) 2015-16 to
     2019-20 and (b) 2017-18 to 2021-22 (flags dropped, six crops removed) and the
     extension-only 2020-21 and 2021-22; other irrigation measures (census 2011 and 1991
     village directories, the land-use statistics 1998-2019 mean); terrain controls.
  F. Slopes below and above the chosen knot with wild cluster restricted bootstrap p
     (9,999 draws), written for 130 to put on the site.

Outputs (outputs/revision_nov8/irrigation_v3/):
  threshold_ssr_profile.csv     SSR, LR and slopes at every grid knot, every index
  threshold_summary.csv         knot, confidence sets, sup-Wald p for every run (incl. E)
  threshold_loso.csv            leave-one-state-out knots
  threshold_spline.csv          slopes at the chosen knot (read by 130)
  threshold_raw.md              log

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/134_irrigation_threshold.py
"""
import contextlib
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = r"D:/crop-diversity"
SRC = REPO + "/scripts/"
OUTV = REPO + "/outputs/revision_nov8/irrigation_v2"
OUT = REPO + "/outputs/revision_nov8/irrigation_v3"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, SRC)
from _years import FIRST_YEAR, LAST_YEAR  # noqa: E402

LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


# ---------------------------------------------------------------- data: referee head (124)
src = open(SRC + "124_referee_irrigation_v2.py", encoding="utf-8").read()
head = src[:src.index("# ================================================================== repro")]
NS = {"__name__": "ref_124_head"}
_argv = sys.argv
sys.argv = [sys.argv[0], "none"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, "124_head", "exec"), NS)
sys.argv = _argv
NS["RNG"] = np.random.default_rng(20261109)
d0, with_win, wcr = NS["d0"], NS["with_win"], NS["wcr"]
RNG = np.random.default_rng(20261109)
GRID = np.round(np.arange(0.20, 0.8001, 0.01), 2)
BK = 1999          # bootstrap draws for the knot and the sup-Wald test
BP = 9999          # draws for the slope p-values (as on the site)


# ---------------------------------------------------------------- tools
def demean_by(v, g):
    """subtract group means (columns of v) within codes g."""
    v = np.asarray(v, float)
    G = g.max() + 1
    cnt = np.bincount(g, minlength=G).astype(float)
    if v.ndim == 1:
        return v - (np.bincount(g, v, G) / cnt)[g]
    out = np.empty_like(v)
    for j in range(v.shape[1]):
        out[:, j] = v[:, j] - (np.bincount(g, v[:, j], G) / cnt)[g]
    return out


def prep(df, y, x, ctrl=()):
    keep = [y, x, "state_name"] + list(ctrl)
    s = df.dropna(subset=keep).copy()
    g = pd.factorize(s["state_name"])[0]
    return s, s[y].to_numpy(float), s[x].to_numpy(float), g, (s[list(ctrl)].to_numpy(float) if ctrl else None)


def design_at(x, gam, C):
    cols = [np.minimum(x, gam), np.maximum(x - gam, 0.0)]
    if C is not None:
        cols += [C[:, j] for j in range(C.shape[1])]
    return np.column_stack(cols)


def cl_se(Xd, u, g, j):
    """CR1 SE of coefficient j (Xd already within-state demeaned; FE dof counted)."""
    n, k = Xd.shape
    G = g.max() + 1
    XtXi = np.linalg.inv(Xd.T @ Xd)
    h = Xd @ XtXi[:, j]
    S = np.bincount(g, h * u, G)
    kk = k + G
    c = G / (G - 1) * (n - 1) / (n - kk)
    return np.sqrt(c * (S ** 2).sum())


def profile(y, x, g, C=None, grid=GRID, Ymat=None):
    """SSR (and slopes) at each knot. If Ymat (n x B) is given, returns SSR matrix (grid x B)."""
    yd = demean_by(y if Ymat is None else Ymat, g)
    out_ssr, rows = [], []
    for gam in grid:
        Xd = demean_by(design_at(x, gam, C), g)
        Q, R = np.linalg.qr(Xd)
        fit = Q @ (Q.T @ yd)
        res = yd - fit
        ssr = (res ** 2).sum(axis=0)
        out_ssr.append(ssr)
        if Ymat is None:
            b = np.linalg.solve(R, Q.T @ yd)
            rows.append(dict(knot=gam, ssr=float(ssr), b_lo=b[0], b_hi=b[1]))
    return np.array(out_ssr), rows


def sup_wald(y, x, g, C=None, grid=GRID, Ymat=None):
    """sup over grid of the state-clustered Wald (t^2) on the hinge term in
    y ~ x + (x-g)+ [+ C] + state FE. Vectorised over Ymat columns."""
    Y = y[:, None] if Ymat is None else Ymat
    yd = demean_by(Y, g)
    G = g.max() + 1
    best = np.zeros(Y.shape[1])
    for gam in grid:
        cols = [x, np.maximum(x - gam, 0.0)] + ([C[:, j] for j in range(C.shape[1])] if C is not None else [])
        Xd = demean_by(np.column_stack(cols), g)
        n, k = Xd.shape
        XtXi = np.linalg.inv(Xd.T @ Xd)
        B = XtXi @ (Xd.T @ yd)
        U = yd - Xd @ B
        h = Xd @ XtXi[:, 1]
        S = np.zeros((G, Y.shape[1]))
        np.add.at(S, g, h[:, None] * U)
        c = G / (G - 1) * (n - 1) / (n - k - G)
        se = np.sqrt(c * (S ** 2).sum(axis=0))
        best = np.maximum(best, (B[1] / se) ** 2)
    return best


def webb(G, B_):
    vals = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
    return vals[RNG.integers(0, 6, size=(G, B_))]


def threshold_run(df, y, x, label, ctrl=(), boot=True, grid=GRID):
    s, Y, X, g, C = prep(df, y, x, ctrl)
    n, G = len(Y), g.max() + 1
    ssr, rows = profile(Y, X, g, C, grid)
    i = int(np.argmin(ssr))
    gh = float(grid[i])
    lr = n * (ssr - ssr[i]) / ssr[i]
    for r, l in zip(rows, lr):
        r["lr"] = float(l)
        r["slope_above"] = r["b_hi"]
        r["slope_below"] = r["b_lo"]
    inset = lambda cut: (float(grid[lr <= cut].min()), float(grid[lr <= cut].max()),
                         bool(np.all(np.diff(np.where(lr <= cut)[0]) == 1)))
    ci84, ci735 = inset(3.84), inset(7.35)
    # straight line (no kink) SSR and the LR of the best kink against it
    Xl = demean_by(np.column_stack([X] + ([C[:, j] for j in range(C.shape[1])] if C is not None else [])), g)
    yd = demean_by(Y, g)
    ssr_lin = float(((yd - Xl @ np.linalg.lstsq(Xl, yd, rcond=None)[0]) ** 2).sum())
    out = dict(label=label, y=y, x=x, n=n, G=G, knot=gh, ssr_min=float(ssr[i]), ssr_linear=ssr_lin,
               lr_ci95_lo=ci84[0], lr_ci95_hi=ci84[1], lr_ci95_connected=ci84[2],
               hansen735_lo=ci735[0], hansen735_hi=ci735[1],
               slope_below=rows[i]["b_lo"], slope_above=rows[i]["b_hi"],
               share_above=float((X > gh).mean()),
               knot_at_edge=bool(gh in (grid[0], grid[-1])))
    if boot:
        # knot CI: wild cluster bootstrap from the fitted kink model
        Xd = demean_by(design_at(X, gh, C), g)
        yd = demean_by(Y, g)
        b = np.linalg.lstsq(Xd, yd, rcond=None)[0]
        # fitted values in levels (state FE included): y - residual
        u = yd - Xd @ b
        fit = Y - u
        V = webb(G, BK)
        Ys = fit[:, None] + u[:, None] * V[g, :]
        S, _ = profile(Y, X, g, C, grid, Ymat=Ys)
        ks = grid[np.argmin(S, axis=0)]
        out.update(boot_lo=float(np.quantile(ks, .025)), boot_hi=float(np.quantile(ks, .975)),
                   boot_sd=float(ks.std()), boot_share_edge=float(np.isin(ks, [grid[0], grid[-1]]).mean()),
                   boot_share_le050=float((ks <= 0.50).mean()))
        # sup-Wald test of a kink against a straight line, restricted wild bootstrap
        sw0 = float(sup_wald(Y, X, g, C, grid)[0])
        ul = yd - Xl @ np.linalg.lstsq(Xl, yd, rcond=None)[0]
        fl = Y - ul
        V = webb(G, BK)
        Ys = fl[:, None] + ul[:, None] * V[g, :]
        sws = sup_wald(Y, X, g, C, grid, Ymat=Ys)
        out.update(supwald=sw0, supwald_p=float((sws >= sw0).mean()))
    return out, rows, s


def fmt(o):
    t = ("{label} | {y} on {x}: n {n} (G {G}); knot {knot:.2f}; LR 95% set {lr_ci95_lo:.2f} to {lr_ci95_hi:.2f}"
         "{c}; Hansen 7.35 set {hansen735_lo:.2f} to {hansen735_hi:.2f}; slopes {slope_below:+.2f} below, "
         "{slope_above:+.2f} above; {sa:.0%} of districts above the knot").format(
        c="" if o["lr_ci95_connected"] else " (not connected)", sa=o["share_above"], **o)
    if "boot_lo" in o:
        t += ("; bootstrap 95% {boot_lo:.2f} to {boot_hi:.2f} (SD {boot_sd:.3f}, {e:.0%} of draws at a grid edge, "
              "{le:.0%} at or below 0.50); sup-Wald {supwald:.2f}, p {supwald_p:.3f}").format(
            e=o["boot_share_edge"], le=o["boot_share_le050"], **o)
    return t


SUM, PROF = [], []


def record(o, rows):
    SUM.append(o)
    for r in rows:
        PROF.append(dict(label=o["label"], y=o["y"], x=o["x"], **r))


w("# Irrigation threshold: where the two-piece line bends")
w("")
w("Window rule: {}-{} to {}-{} (scripts/_years.py). Grid {:.2f} to {:.2f}, step 0.01. "
  "Bootstrap: wild cluster by state, Webb weights, {:,} draws (knot, sup-Wald); slope p {:,} draws."
  .format(FIRST_YEAR, str(FIRST_YEAR + 1)[-2:], LAST_YEAR, str(LAST_YEAR + 1)[-2:], GRID[0], GRID[-1], BK, BP))

FULL = with_win(d0, "full", "all")

# ---------------------------------------------------------------- A-C main
w("")
w("## A-C. Main: D1, Antyodaya 2019-20 share, state FE, full period")
w("")
# reproduce the referee spline at 0.30 first
s, Y, X, g, C = prep(FULL, "D1", "irr_share")
_, r030 = profile(Y, X, g, None, np.array([0.30]))
w("Check: knot fixed at 0.30 gives slope below {:+.3f}, above {:+.3f} (referee CSV: +2.91, -4.10)".format(
    r030[0]["b_lo"], r030[0]["b_hi"]))
main, rows, s = threshold_run(FULL, "D1", "irr_share", "main")
record(main, rows)
w(fmt(main))
w("")
w("SSR profile (every fifth grid point; LR = n (SSR - SSR min) / SSR min):")
w("")
w("| knot | SSR | LR | slope below | slope above |")
w("|---|---|---|---|---|")
for r in rows:
    if abs(r["knot"] * 100 % 5) < 1e-6 or r["knot"] == main["knot"]:
        w("| {:.2f}{} | {:.2f} | {:.2f} | {:+.2f} | {:+.2f} |".format(
            r["knot"], " (min)" if r["knot"] == main["knot"] else "", r["ssr"], r["lr"], r["b_lo"], r["b_hi"]))
w("")
w("Straight line SSR {:.2f}; best kink SSR {:.2f}; LR of kink against line {:.2f}.".format(
    main["ssr_linear"], main["ssr_min"], main["n"] * (main["ssr_linear"] - main["ssr_min"]) / main["ssr_min"]))

# ---------------------------------------------------------------- D other indices
w("")
w("## D. Other indices (same sample and model)")
w("")
for y in ["D0", "D0_05", "D2", "E"]:
    o, rows, _ = threshold_run(FULL, y, "irr_share", "index " + y)
    record(o, rows)
    w("- " + fmt(o))

# ---------------------------------------------------------------- E adversarial
w("")
w("## E. Adversarial checks on the D1 knot")
w("")
w("### Recent windows")
ALT = []
for lab, t, v in [("window a 2015-16 to 2019-20", "w1517", "all"),
                  ("window b 2017-18 to 2021-22 (flags dropped, six crops removed)", "w1721", "flags_drop6")]:
    o, rows, _ = threshold_run(with_win(d0, t, v), "D1", "irr_share", lab)
    record(o, rows)
    w("- " + fmt(o))
dy = pd.read_csv(OUTV + "/div_district_year.csv")
xx = dy[dy["year_start"].isin([2020, 2021]) & ~dy["flag_sy"]]
gg = xx.groupby("district_key").agg(D1=("D1_d6", "mean"), ny=("year_start", "nunique"))
gg = gg[gg["ny"] == 2][["D1"]]
EXT = d0.drop(columns=[c for c in ["D1", "D0", "D0_05", "E", "D2"] if c in d0]).merge(
    gg, left_on="district_key", right_index=True, how="inner")
o, rows, _ = threshold_run(EXT, "D1", "irr_share", "extension only 2020-21 and 2021-22")
record(o, rows)
w("- " + fmt(o))

w("")
w("### Other irrigation measures (full-period D1)")
P = pd.read_csv(REPO + "/outputs/revision_nov8/irrigation_panel_district_year.csv")
P = P[P["year_start"].between(FIRST_YEAR, LAST_YEAR)]
lus = P.groupby("district_key")["irr_nsa"].mean().rename("irr_lus")
F2 = FULL.merge(lus, left_on="district_key", right_index=True, how="left")
F2["irr_c2011"] = F2["c2011_irr_share"].where(F2["c2011_district_usable"])
F2["irr_c1991"] = F2["c1991_irr_share"].where(F2["c1991_district_usable"])
for xv, lab in [("irr_c2011", "census 2011 village directory"), ("irr_lus", "land-use statistics, 1998-2019 mean"),
                ("irr_c1991", "census 1991 village directory")]:
    o, rows, _ = threshold_run(F2, "D1", xv, "measure: " + lab)
    record(o, rows)
    w("- " + fmt(o))

w("")
w("The measures differ in level (means: Antyodaya {:.2f}, census 2011 {:.2f}, land-use statistics {:.2f}, "
  "census 1991 {:.2f}), so a knot on one scale is not a knot on another. Two further checks: the grid "
  "widened to 0.05 for the measures whose knot sits on the 0.20 edge, and every measure turned into its "
  "percentile rank among the districts (knot then read as a rank, grid 0.20 to 0.80).".format(
      F2["irr_share"].mean(), F2["irr_c2011"].mean(), F2["irr_lus"].mean(), F2["irr_c1991"].mean()))
WIDE = np.round(np.arange(0.05, 0.8001, 0.01), 2)
for xv, lab in [("irr_c2011", "census 2011 village directory"), ("irr_lus", "land-use statistics, 1998-2019 mean"),
                ("irr_c1991", "census 1991 village directory")]:
    o, rows, _ = threshold_run(F2, "D1", xv, "measure, grid from 0.05: " + lab, grid=WIDE)
    record(o, rows)
    w("- " + fmt(o))
for xv, lab in [("irr_share", "Antyodaya 2019-20"), ("irr_c2011", "census 2011 village directory"),
                ("irr_lus", "land-use statistics, 1998-2019 mean"), ("irr_c1991", "census 1991 village directory")]:
    F3 = F2.dropna(subset=[xv, "D1"]).copy()
    F3["rank_"] = F3[xv].rank(pct=True)
    o, rows, _ = threshold_run(F3, "D1", "rank_", "percentile rank: " + lab)
    o["x"] = "rank of " + xv
    record(o, rows)
    w("- " + fmt(o))

w("")
w("### Controls")
o, rows, _ = threshold_run(FULL, "D1", "irr_share", "terrain controls (elevation, ruggedness)", ctrl=("elev_km", "tri"))
record(o, rows)
w("- " + fmt(o))
o, rows, _ = threshold_run(FULL[~FULL["flag_shared_parent"].astype(bool)] if "flag_shared_parent" in FULL else FULL,
                           "D1", "irr_share", "drop districts sharing a 2011 parent")
record(o, rows)
w("- " + fmt(o))

w("")
w("### Leave one state out (D1, full period; LR sets only)")
lo_rows = []
for st in sorted(FULL["state_name"].dropna().unique()):
    o, _, _ = threshold_run(FULL[FULL["state_name"] != st], "D1", "irr_share", "drop " + st, boot=False)
    lo_rows.append(dict(state=st, n=o["n"], knot=o["knot"], lr_lo=o["lr_ci95_lo"], lr_hi=o["lr_ci95_hi"],
                        slope_below=o["slope_below"], slope_above=o["slope_above"]))
LO = pd.DataFrame(lo_rows)
LO.to_csv(OUT + "/threshold_loso.csv", index=False)
w("Knot range {:.2f} to {:.2f}, median {:.2f}; {} of {} drops keep it within the main bootstrap interval.".format(
    LO.knot.min(), LO.knot.max(), LO.knot.median(),
    int(LO.knot.between(main["boot_lo"], main["boot_hi"]).sum()), len(LO)))
mv = LO.reindex((LO.knot - main["knot"]).abs().sort_values(ascending=False).index).head(5)
w("Largest moves: " + "; ".join("{} {:.2f}".format(a, b) for a, b in mv[["state", "knot"]].values))

# ---------------------------------------------------------------- choose the knot
w("")
w("## Choice of knot for the site")
width = main["boot_hi"] - main["boot_lo"]
in_ci = main["boot_lo"] <= 0.50 <= main["boot_hi"] and main["lr_ci95_lo"] <= 0.50 <= main["lr_ci95_hi"]
imprecise = width > 0.15
KNOT = 0.50 if (imprecise and in_ci) else main["knot"]
w("Estimated knot {:.2f}; bootstrap 95% {:.2f} to {:.2f} (width {:.2f}); LR 95% set {:.2f} to {:.2f}. "
  "Rule: use 0.50 if the bootstrap interval is wider than 0.15 and 0.50 lies inside both sets, "
  "otherwise the estimate. Imprecise: {}; 0.50 inside: {}. Knot used: {:.2f}.".format(
      main["knot"], main["boot_lo"], main["boot_hi"], width, main["lr_ci95_lo"], main["lr_ci95_hi"],
      imprecise, in_ci, KNOT))

# ---------------------------------------------------------------- F slopes at the chosen knot
w("")
w("## F. Slopes at the chosen knot (state-clustered, t(G-1) interval, wild restricted p, {:,} draws)".format(BP))
SP = []
for key, t, v in [("full", "full", "all"), ("w1721", "w1721", "flags_drop6")]:
    for kn in sorted({KNOT, 0.30}):
        dfb = with_win(d0, t, v)
        dfb["lo"] = dfb["irr_share"].clip(upper=kn)
        dfb["hi"] = (dfb["irr_share"] - kn).clip(lower=0)
        for y in ["D1", "D0_05"]:
            f = "{} ~ lo + hi + C(state_name)".format(y)
            keep = [y, "lo", "hi", "state_name"]
            r1 = wcr(dfb, f, "lo", "state_name", keep, B_=BP)
            r2 = wcr(dfb, f, "hi", "state_name", keep, B_=BP)
            tc = stats.t.ppf(0.975, r2["G"] - 1)
            SP.append(dict(window=key, knot=kn, y=y, n=r1["n"], G=r1["G"],
                           below=r1["b"], below_se=r1["se"], below_p=r1["p_wcr"], below_ptG1=r1["p_tG1"],
                           below_lo=r1["b"] - tc * r1["se"], below_hi=r1["b"] + tc * r1["se"],
                           above=r2["b"], above_se=r2["se"], above_p=r2["p_wcr"], above_ptG1=r2["p_tG1"],
                           above_lo=r2["b"] - tc * r2["se"], above_hi=r2["b"] + tc * r2["se"],
                           share_above=float((dfb.dropna(subset=keep)["irr_share"] > kn).mean()),
                           chosen=bool(kn == KNOT)))
            w("- {} knot {:.2f} {}: below {:+.2f} (95% {:+.2f} to {:+.2f}, wild p {:.3f}); above {:+.2f} "
              "(95% {:+.2f} to {:+.2f}, wild p {:.3f}); {:.0%} of districts above".format(
                  key, kn, y, SP[-1]["below"], SP[-1]["below_lo"], SP[-1]["below_hi"], SP[-1]["below_p"],
                  SP[-1]["above"], SP[-1]["above_lo"], SP[-1]["above_hi"], SP[-1]["above_p"], SP[-1]["share_above"]))
pd.DataFrame(SP).to_csv(OUT + "/threshold_spline.csv", index=False)
pd.DataFrame(SUM).to_csv(OUT + "/threshold_summary.csv", index=False)
pd.DataFrame(PROF).to_csv(OUT + "/threshold_ssr_profile.csv", index=False)
with open(OUT + "/threshold_raw.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG) + "\n")
print("wrote " + OUT)
