"""
_revision_fe.py  -- helper for scripts 105 and 106 (revision_nov8)

High-dimensional fixed effects by alternating projections, OLS and 2SLS on the
demeaned data, cluster-robust (CR1) standard errors, and the delta-method
turning point of a quadratic.

Degrees of freedom: fixed effects nested within clusters (district within its
2011 parent) are not counted, as in reghdfe; non-nested ones (state x year) are.
"""
import numpy as np
import pandas as pd
from scipy import stats


def demean(df, cols, fes, tol=1e-10, maxit=5000):
    """Partial out several sets of group fixed effects from `cols`."""
    X = df[cols].to_numpy(dtype=float).copy()
    codes = [pd.factorize(df[f])[0] for f in fes]
    counts = [np.bincount(c) for c in codes]
    for _ in range(maxit):
        Xold = X.copy()
        for c, n in zip(codes, counts):
            for j in range(X.shape[1]):
                m = np.bincount(c, weights=X[:, j]) / n
                X[:, j] -= m[c]
        if np.max(np.abs(X - Xold)) < tol:
            break
    return pd.DataFrame(X, columns=cols, index=df.index)


def _cluster_vcov(Xs, u, cl, bread, n, k):
    G = len(np.unique(cl))
    codes = pd.factorize(cl)[0]
    S = np.zeros((Xs.shape[1], Xs.shape[1]))
    sc = np.zeros((G, Xs.shape[1]))
    np.add.at(sc, codes, Xs * u[:, None])
    S = sc.T @ sc
    adj = G / (G - 1) * (n - 1) / max(n - k, 1)
    return adj * bread @ S @ bread, G


def fe_ols(df, y, xs, fes, cluster, nested=(), weights=None):
    """OLS of y on xs absorbing `fes`; returns dict with params, se, vcov."""
    d = df.dropna(subset=[y] + xs + fes + [cluster]).copy()
    if weights is not None:
        d = d[d[weights] > 0]
    if fes:
        # drop singletons iteratively
        for _ in range(10):
            keep = np.ones(len(d), bool)
            for f in fes:
                keep &= d.groupby(f)[y].transform("size").to_numpy() > 1
            if keep.all():
                break
            d = d[keep]
        dm = demean(d, [y] + xs, fes)
    else:
        dm = d[[y] + xs].copy()
        dm["_c"] = 1.0
        xs = xs + ["_c"]
    Y = dm[y].to_numpy()
    X = dm[xs].to_numpy()
    if weights is not None:
        sw = np.sqrt(d[weights].to_numpy())
        Y, X = Y * sw, X * sw[:, None]
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ Y
    u = Y - X @ b
    n = len(Y)
    k = X.shape[1] + sum(d[f].nunique() for f in fes if f not in nested)
    V, G = _cluster_vcov(X, u, d[cluster].to_numpy(), XtX_inv, n, k)
    se = np.sqrt(np.diag(V))
    ss_w = float((u ** 2).sum())
    ss_t = float(((Y - Y.mean()) ** 2).sum())
    return dict(params=pd.Series(b, xs), se=pd.Series(se, xs),
                vcov=pd.DataFrame(V, xs, xs), n=n, G=G,
                n_units=d[fes[0]].nunique() if fes else n,
                within_r2=1 - ss_w / ss_t, data=d, resid=u)


def fe_iv(df, y, endog, exog, instr, fes, cluster, nested=()):
    """2SLS absorbing fes. Returns second stage and first-stage diagnostics."""
    cols = [y] + endog + exog + instr
    d = df.dropna(subset=cols + fes + [cluster]).copy()
    for _ in range(10):
        keep = np.ones(len(d), bool)
        for f in fes:
            keep &= d.groupby(f)[y].transform("size").to_numpy() > 1
        if keep.all():
            break
        d = d[keep]
    dm = demean(d, cols, fes)
    Y = dm[y].to_numpy()
    Xen = dm[endog].to_numpy()
    W = dm[exog].to_numpy() if exog else np.zeros((len(d), 0))
    Z = np.column_stack([dm[instr].to_numpy(), W])
    X = np.column_stack([Xen, W])
    n = len(Y)
    kfe = sum(d[f].nunique() for f in fes if f not in nested)
    cl = d[cluster].to_numpy()
    # first stages
    fs = []
    ZtZi = np.linalg.inv(Z.T @ Z)
    for j, e in enumerate(endog):
        g = ZtZi @ Z.T @ Xen[:, j]
        v = Xen[:, j] - Z @ g
        V, G = _cluster_vcov(Z, v, cl, ZtZi, n, Z.shape[1] + kfe)
        q = len(instr)
        gi, Vi = g[:q], V[:q, :q]
        F = float(gi @ np.linalg.inv(Vi) @ gi) / q
        fs.append(dict(endog=e, coefs=dict(zip(instr, gi)),
                       ses=dict(zip(instr, np.sqrt(np.diag(Vi)))), F=F))
    Xhat = Z @ ZtZi @ Z.T @ X
    A = np.linalg.inv(Xhat.T @ X)
    b = A @ Xhat.T @ Y
    u = Y - X @ b
    V, G = _cluster_vcov(Xhat, u, cl, A, n, X.shape[1] + kfe)
    names = endog + exog
    return dict(params=pd.Series(b, names), se=pd.Series(np.sqrt(np.diag(V)), names),
                vcov=pd.DataFrame(V, names, names), n=n, G=G, first=fs, data=d)


def turning_point(r, x, x2):
    b1, b2 = r["params"][x], r["params"][x2]
    tp = -b1 / (2 * b2)
    g = np.array([-1 / (2 * b2), b1 / (2 * b2 ** 2)])
    V = r["vcov"].loc[[x, x2], [x, x2]].to_numpy()
    se = float(np.sqrt(g @ V @ g))
    return tp, se


def pval(b, se, G=None):
    t = b / se
    if G:
        return 2 * stats.t.sf(abs(t), G - 1)
    return 2 * stats.norm.sf(abs(t))
