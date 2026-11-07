"""
_mkt2_common.py

Shared tools for the markets v2 revision (scripts 136, 137, 138).

  wcr()        state-clustered t(G-1) p and restricted wild cluster bootstrap p
               (Webb six-point weights), the same algorithm as script 124
  fit_par()    OLS clustered on the 2011 parent district (shrug_key)
  load_104()   runs the data-construction head of 104_market_pop_weighted.py and
               returns its namespace (the 606-district frame with population-share
               facility measures and the site's controls); writes nothing
  apy_base()   runs the cleaning head of 122_irrigation_v2_diversity.py and returns
               district x year x crop area (seasons summed), 1998-99 to 2022-23
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

REPO = r"D:/crop-diversity"
RNG = np.random.default_rng(20261108)
B = 9999


def design(df, formula, keep):
    s = df.dropna(subset=[k for k in keep if k in df]).copy()
    return smf.ols(formula, s), s


def cr1(X, u, h, g):
    G = g.max() + 1
    n, k = X.shape
    S = np.zeros((G,) + u.shape[1:])
    np.add.at(S, g, h[:, None] * u if u.ndim == 2 else h * u)
    c = G / (G - 1) * (n - 1) / (n - k)
    return np.sqrt(c * (S ** 2).sum(axis=0))


def webb(G, B_):
    vals = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
    return vals[RNG.integers(0, 6, size=(G, B_))]


def wcr(df, formula, term, keep, cluster="state_name", B_=B):
    """Restricted wild cluster bootstrap for H0: term = 0. Returns dict with the
    point estimate, the CR1 state SE, p from t(G-1) and the bootstrap p."""
    m, s = design(df, formula, keep)
    X, y = m.exog, m.endog
    j = m.exog_names.index(term)
    g = pd.factorize(s[cluster])[0]
    G = g.max() + 1
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    u = y - X @ b
    h = X @ XtXi[:, j]
    se = cr1(X, u, h, g)
    t0 = b[j] / se
    Xr = np.delete(X, j, axis=1)
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    ur = y - Xr @ br
    fr = Xr @ br
    V = webb(G, B_)
    Ys = fr[:, None] + ur[:, None] * V[g, :]
    bs = XtXi @ (X.T @ Ys)
    us = Ys - X @ bs
    ts = bs[j] / cr1(X, us, h, g)
    return dict(b=b[j], se_st=se, G=G, n=len(y), p_st=2 * stats.t.sf(abs(t0), G - 1),
                p_wcr=float(np.mean(np.abs(ts) >= abs(t0))))


def fit_par(df, formula, keep, cluster="shrug_key"):
    s = df.dropna(subset=[k for k in keep if k in df]).copy()
    m = smf.ols(formula, s).fit(cov_type="cluster",
                                cov_kwds={"groups": pd.factorize(s[cluster])[0]})
    return m, s


def load_104():
    src = open(REPO + "/scripts/104_market_pop_weighted.py", encoding="utf-8").read()
    cut = src.index("VERS = [")
    ns = {"__name__": "m104"}
    exec(compile(src[:cut], "104_head", "exec"), ns)
    return ns


def apy_base():
    src = open(REPO + "/scripts/122_irrigation_v2_diversity.py", encoding="utf-8").read()
    cut = src.index("v_all = indices(base)")
    ns = {"__name__": "m122"}
    exec(compile(src[:cut], "122_head", "exec"), ns)
    return ns
