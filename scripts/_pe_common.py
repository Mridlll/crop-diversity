"""
_pe_common.py

Shared helpers for the peer-effects scripts (125-129). Not run on its own.

  load()            unit table, unit-year panel, window table, pair table, with the
                    analysis state `st` (present-day state; Ladakh folded into Jammu
                    and Kashmir because it was not a separate state in 1998-2019)
  dyadic_ols()      OLS with dyadic-robust variance (Cameron and Miller 2014; Aronow,
                    Samii and Assenova 2015): any two pairs that share a member are
                    allowed to be correlated. Run with members = 2011 parent units, or
                    members = states (a pair touches one or two states).
  demean()          alternating projections over several FE groupings
  moran()           Moran's I with a row-standardised weight matrix and a
                    permutation p-value
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "revision_nov8", "peer_effects")
GROUPS = ["rice", "wheat", "maize", "millets", "pulses", "oilseeds", "cotton",
          "sugarcane", "other"]


def fold_state(s):
    s = str(s)
    return "JAMMU AND KASHMIR" if s in ("LADAKH",) else s


def load():
    units = pd.read_csv(os.path.join(OUT, "pe_units.csv"))
    units["st"] = units.state_now.map(fold_state)
    uy = pd.read_csv(os.path.join(OUT, "pe_unit_year.csv"))
    uy["st"] = uy.unit_key.map(units.set_index("unit_key").st)
    up = pd.read_csv(os.path.join(OUT, "pe_unit_period.csv"))
    P = pd.read_csv(os.path.join(OUT, "pe_pairs.csv.gz"))
    P["st_i"], P["st_j"] = P.state_i.map(fold_state), P.state_j.map(fold_state)
    P["same_state"] = (P.st_i == P.st_j).astype(int)
    P["cross"] = 1 - P.same_state
    P["new_border"] = (((P.st_i == "ANDHRA PRADESH") & (P.st_j == "TELANGANA"))
                       | ((P.st_j == "ANDHRA PRADESH") & (P.st_i == "TELANGANA"))).astype(int)
    return units, uy, up, P


def dyadic_vcov(X, e, mi, mj, conservative=False):
    """Dyadic-robust sandwich. X n x k (already demeaned if FE), e residuals,
    mi, mj integer member codes of the two ends of each pair.
    conservative=True skips the correction for pairs counted twice (pairs sharing
    both members). The meat is then PSD by construction and weakly larger; used for
    state membership, where ~30 clusters make the corrected meat non-PSD in places.
    The corrected version falls back to the conservative one if any variance is
    negative."""
    XtXi = np.linalg.pinv(X.T @ X)
    S = X * e[:, None]
    k = X.shape[1]
    G = int(max(mi.max(), mj.max())) + 1
    acc = np.zeros((G, k))
    np.add.at(acc, mi, S)
    diff = mi != mj
    np.add.at(acc, mj[diff], S[diff])          # a same-member pair is added once
    meat = acc.T @ acc
    # pairs with the same unordered (mi, mj), mi != mj, were counted twice
    if diff.any() and not conservative:
        a, b = np.minimum(mi, mj)[diff], np.maximum(mi, mj)[diff]
        code = pd.factorize(a.astype(np.int64) * (G + 1) + b)[0]
        acc2 = np.zeros((code.max() + 1, k))
        np.add.at(acc2, code, S[diff])
        meat -= acc2.T @ acc2
    ncl = len(np.unique(np.concatenate([mi, mj])))
    n = X.shape[0]
    adj = ncl / max(ncl - 1, 1) * (n - 1) / max(n - k, 1)
    V = adj * XtXi @ meat @ XtXi
    if not conservative and (np.diag(V) <= 0).any():
        return dyadic_vcov(X, e, mi, mj, conservative=True)
    return V, ncl


def dyadic_ols(y, X, names, mi, mj, add_const=True, conservative=False):
    """returns dict name -> (b, se, p) plus n, n_clusters."""
    from scipy import stats
    X = np.asarray(X, float)
    if add_const:
        X = np.column_stack([np.ones(len(y)), X])
        names = ["const"] + list(names)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    V, ncl = dyadic_vcov(X, e, mi, mj, conservative=conservative)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    p = 2 * stats.t.sf(np.abs(b / np.where(se > 0, se, np.nan)), df=max(ncl - 1, 1))
    out = {nm: (b[k], se[k], p[k]) for k, nm in enumerate(names)}
    out["_n"], out["_ncl"] = len(y), ncl
    out["_r2"] = 1 - e.var() / y.var() if y.var() > 0 else np.nan
    return out


def demean(Z, groups, tol=1e-10, maxit=1000):
    Z = np.asarray(Z, float).copy()
    for _ in range(maxit):
        old = Z.copy()
        for g in groups:
            cnt = np.bincount(g)
            for c in range(Z.shape[1]):
                s = np.bincount(g, weights=Z[:, c])
                Z[:, c] -= (s / np.where(cnt == 0, 1, cnt))[g]
        if np.max(np.abs(Z - old)) < tol:
            break
    return Z


def moran(x, Wb, nperm=999, rng=None):
    """x: vector (finite), Wb: binary symmetric adjacency on the same units.
    Units with no neighbour inside the set are dropped. Returns I, E[I], z, p_perm, n."""
    keep = Wb.sum(1) > 0
    x, Wb = x[keep], Wb[np.ix_(keep, keep)]
    keep2 = Wb.sum(1) > 0
    x, Wb = x[keep2], Wb[np.ix_(keep2, keep2)]
    n = len(x)
    W = Wb / Wb.sum(1, keepdims=True)
    z = x - x.mean()
    I = (z @ W @ z) / (z @ z)
    rng = rng or np.random.default_rng(0)
    sims = np.empty(nperm)
    for k in range(nperm):
        zp = rng.permutation(z)
        sims[k] = (zp @ W @ zp) / (zp @ zp)
    p = (1 + np.sum(np.abs(sims - sims.mean()) >= abs(I - sims.mean()))) / (nperm + 1)
    return {"I": I, "EI": -1 / (n - 1), "sd_perm": sims.std(), "z": (I - sims.mean()) / sims.std(),
            "p_perm": p, "n": n}
