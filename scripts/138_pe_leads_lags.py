"""
138_pe_leads_lags.py

Leads and lags of the neighbour coefficient, t-3 to t+3. Reviewer item: the page states
the lead test in one sentence ("the link at a lag is about the same looking forward as
looking back"); show it as a figure.

Reuses script 99's data and helpers (executed read-only up to its estimation loop):
2011 parent units, queen adjacency, the same-state distance ring (non-adjacent units in
the same state at 0.75-1.5x the unit's mean distance to its neighbours, about 100 km),
rain controls, unit FE and state-by-year FE, SE clustered on the unit.

    dS_it = sum_k b_k * W dS_i,t-k + sum_k c_k * R dS_i,t-k
            + own dS_i,t-1 + own dS_i,t+1 + rain + a_i + d_state,t + e,   k = -3..3

W dS = mean year-on-year change of the adjacent units, R dS = same for the ring;
k > 0 is the neighbours' change k years BEFORE the district's own change (a lag),
k < 0 the change k years AFTER (a lead). Own change a year before and a year after are
both held, so the design is symmetric in time: a shared transitory shock that pushes
own and neighbour shares in year t pushes the neighbour's change at t+1 and t-1
down by the same mechanism, so it does not tilt lags against leads.
Spread from neighbours predicts b_k > c_k for k > 0 and b_k = c_k for k < 0.
A shared regional process predicts b_k about equal to b_-k.

Outcomes: pulses group share (the page's year-on-year sentence), and the crop shares of
the take-up section: chickpea (gram), soybean, cotton, plus maize. Each has
  main        joint k = -3..3 with ring terms, own t-1 and t+1 held
  no_ring     joint k = -3..3, neighbours only
  no_own      main without own lag and lead
  no_k0       main without the same-year terms
Window 1998-99 to 2019-20 (scripts/_years.py; script 99 itself ran from 1997-98).
Tests: sum of lags (k=1..3) minus sum of leads (k=-1..-3), neighbours and ring;
neighbour-minus-ring at each k.

Run: C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/138_pe_leads_lags.py
Outputs (outputs/revision_nov8/peer_effects/v2/):
  138_leads_lags.csv, 138_leads_lags_tests.csv, 138_log.txt
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
V2 = os.path.join(ROOT, "outputs", "revision_nov8", "peer_effects", "v2")
os.makedirs(V2, exist_ok=True)

src = open(os.path.join(HERE, "99_crop_groups_neighbours.py"), encoding="utf-8").read()
cut = src.index("rows = []\ndesc = []")
ns = {"__file__": os.path.join(HERE, "99_crop_groups_neighbours.py"), "__name__": "s99"}
exec(compile(src[:cut], "99_header", "exec"), ns)
LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


S, diff, shift, wavg, fit = ns["S"], ns["diff"], ns["shift"], ns["wavg"], ns["fit"]
nb_idx, plc_idx, RAIN, years, NU, NY = ns["nb_idx"], ns["plc_idx"], ns["RAIN"], ns["years"], ns["NU"], ns["NY"]
uix, units_all = ns["uix"], ns["units_all"]
rain = {k: RAIN[k] for k in RAIN}

# crop shares from the peer-effects unit-year panel (same 2011 parent units)
uy = pd.read_csv(os.path.join(ROOT, "outputs", "revision_nov8", "peer_effects", "pe_unit_year.csv"),
                 usecols=["unit_key", "year_start", "reliable", "sh_gram", "sh_soybean", "sh_cotton"])
uy = uy[(uy.reliable == 1) & uy.unit_key.isin(uix)]


def grid_pe(col):
    M = np.full((NU, NY), np.nan)
    M[uy.unit_key.map(uix).values, (uy.year_start - years[0]).values] = uy[col].values
    return M


OUTC = {"pulses": S["pulses"], "chickpea": grid_pe("sh_gram"), "soybean": grid_pe("sh_soybean"),
        "cotton": grid_pe("sh_cotton"), "maize": S["maize"], "millets": S["millets"], "oilseeds": S["oilseeds"]}
from _years import FIRST_YEAR  # noqa: E402
win = np.zeros((NU, NY), bool)
win[:, [k for k, y in enumerate(years) if y >= FIRST_YEAR]] = True

# reproduction of script 99 "P2 lag+placebo" for pulses (its full 1997-2019 run)
D = diff(S["pulses"])
o = fit(D, {"WD_l1": shift(wavg(D, nb_idx), 1), "PD_l1": shift(wavg(D, plc_idx), 1), "own_l1": shift(D, 1), **rain})
ref = pd.read_csv(os.path.join(ROOT, "outputs", "crop_groups", "neighbours_estimates.csv"))
r0 = ref[(ref.group == "pulses") & (ref.spec == "P2 lag+placebo") & (ref["var"] == "WD_l1")].iloc[0]
w(f"reproduce 99 P2 pulses: WD_l1 {o['WD_l1_b']:.4f} (99: {r0.b:.4f}), PD_l1 {o['PD_l1_b']:.4f}")

K = [-3, -2, -1, 0, 1, 2, 3]
rows, tests = [], []
for name, M in OUTC.items():
    D = diff(M)
    WD, PD = wavg(D, nb_idx), wavg(D, plc_idx)
    # ring alone needs every k defined so all specs share one sample
    allk = np.ones((NU, NY), bool)
    for k in K:
        allk &= np.isfinite(shift(WD, k)) & np.isfinite(shift(PD, k))
    allk &= np.isfinite(shift(D, 1)) & np.isfinite(shift(D, -1)) & win
    for spec in ["main", "no_ring", "no_own", "no_k0"]:
        X = {}
        ks = [k for k in K if not (spec == "no_k0" and k == 0)]
        for k in ks:
            X[f"W{k:+d}"] = shift(WD, k)
        if spec != "no_ring":
            for k in ks:
                X[f"R{k:+d}"] = shift(PD, k)
        if spec != "no_own":
            X["own_l1"], X["own_f1"] = shift(D, 1), shift(D, -1)
        X.update(rain)
        names = list(X)
        o = fit(D, X, extra_mask=allk)
        # full vcov for contrasts: refit is cheap; fit() keeps only cov01, so rebuild here
        Z = np.column_stack([D.ravel()] + [X[n].ravel() for n in names])
        m = o["_mask"]
        Zm = Z[m]
        ui = ns["UI"][m]
        sy = ns["SY"][m]
        g = [pd.factorize(ui)[0], pd.factorize(sy)[0]]
        Zd = ns["demean"](Zm, g)
        import statsmodels.api as sm
        res = sm.OLS(Zd[:, 0], Zd[:, 1:]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(ui)[0]})
        V = np.asarray(res.cov_params())
        for k in ks:
            for pre, lab in [("W", "neighbours"), ("R", "ring")]:
                nm = f"{pre}{k:+d}"
                if nm not in names:
                    continue
                j = names.index(nm)
                rows.append({"outcome": name, "spec": spec, "set": lab, "k": k, "b": res.params[j],
                             "se": res.bse[j], "p": res.pvalues[j], "n": int(m.sum()),
                             "n_units": len(np.unique(ui)), "sd_dep": float(np.nanstd(Zm[:, 0]))})
            if spec in ("main", "no_k0", "no_own") and f"R{k:+d}" in names:
                c = np.zeros(len(names)); c[names.index(f"W{k:+d}")] = 1; c[names.index(f"R{k:+d}")] = -1
                b, se = c @ res.params, np.sqrt(c @ V @ c)
                rows.append({"outcome": name, "spec": spec, "set": "neighbours minus ring", "k": k, "b": b,
                             "se": se, "p": 2 * (1 - norm.cdf(abs(b / se))), "n": int(m.sum()),
                             "n_units": len(np.unique(ui))})
        for pre, lab in [("W", "neighbours"), ("R", "ring")]:
            if f"{pre}+1" not in names:
                continue
            c = np.zeros(len(names))
            for k in [1, 2, 3]:
                c[names.index(f"{pre}{k:+d}")] += 1
                c[names.index(f"{pre}{-k:+d}")] -= 1
            b, se = c @ res.params, np.sqrt(c @ V @ c)
            tests.append({"outcome": name, "spec": spec, "set": lab, "test": "sum of lags 1-3 minus sum of leads 1-3",
                          "b": b, "se": se, "p": 2 * (1 - norm.cdf(abs(b / se)))})
            c1 = np.zeros(len(names)); c1[names.index(f"{pre}+1")] = 1; c1[names.index(f"{pre}-1")] = -1
            b, se = c1 @ res.params, np.sqrt(c1 @ V @ c1)
            tests.append({"outcome": name, "spec": spec, "set": lab, "test": "lag 1 minus lead 1",
                          "b": b, "se": se, "p": 2 * (1 - norm.cdf(abs(b / se)))})
        if spec == "main":
            # joint test of the three leads (neighbours)
            idx = [names.index(f"W{k:+d}") for k in [-1, -2, -3]]
            bb = res.params[idx]
            Vi = V[np.ix_(idx, idx)]
            chi = float(bb @ np.linalg.solve(Vi, bb))
            from scipy.stats import chi2
            tests.append({"outcome": name, "spec": spec, "set": "neighbours", "test": "leads jointly zero (chi2, 3 df)",
                          "b": chi, "se": np.nan, "p": float(1 - chi2.cdf(chi, 3))})
            idx = [names.index(f"W{k:+d}") for k in [1, 2, 3]]
            bb = res.params[idx]
            chi = float(bb @ np.linalg.solve(V[np.ix_(idx, idx)], bb))
            tests.append({"outcome": name, "spec": spec, "set": "neighbours", "test": "lags jointly zero (chi2, 3 df)",
                          "b": chi, "se": np.nan, "p": float(1 - chi2.cdf(chi, 3))})
    w(f"{name}: done, n {int(m.sum())}")

R = pd.DataFrame(rows)
TT = pd.DataFrame(tests)
R.to_csv(os.path.join(V2, "138_leads_lags.csv"), index=False)
TT.to_csv(os.path.join(V2, "138_leads_lags_tests.csv"), index=False)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 400)
w(R[R.spec == "main"].pivot_table(index=["outcome", "set"], columns="k", values="b", sort=False).round(3).to_string())
w(R[R.spec == "main"].pivot_table(index=["outcome", "set"], columns="k", values="se", sort=False).round(3).to_string())
w(R[R.spec == "no_k0"].pivot_table(index=["outcome", "set"], columns="k", values="b", sort=False).round(3).to_string())
w(R[R.spec == "no_own"].pivot_table(index=["outcome", "set"], columns="k", values="b", sort=False).round(3).to_string())
w(TT.round(4).to_string(index=False))
with open(os.path.join(V2, "138_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
