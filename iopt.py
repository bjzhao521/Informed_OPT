"""iOPT: informed optimal path thresholds for compound extreme events.

For a target event probability p_t, iOPT returns one threshold per variable
and a set of exactly N = round(p_t * n_total) event days on which all
variables exceed their thresholds (upper tails).

Steps
  1. Percentile grid evenly spaced in rarity r = -ln(1 - a)
     (a = 0.50 ... 0.999, 0.10 nat steps, 63 nodes per variable).
  2. Empirical joint exceedance probability P on the grid.
  3. Dynamic programming: the monotone path through the grid that maximizes
     the sum of P along the path (Zhao, Horvat & Gao, 2025, ERL 20, 024048).
  4. Days are ranked by how far along the path they still exceed all
     thresholds (ties broken by the margin to the next node), and the N
     highest-ranked days form the event set.
  5. Optional anchor: the first variable (temperature) is moved a fraction f
     of the way, in rarity, from its data-optimal node toward a physical
     threshold; the other thresholds are re-optimized with it fixed, and
     N days are selected again.

Only numpy is required.
"""
import numpy as np


# ------------------------------------------------------------------ grid
def make_grid(floor=0.50, amax=0.999, step=0.10):
    """Percentile nodes at equal steps of `step` nat in r = -ln(1 - a)."""
    r0, r1 = -np.log(1 - floor), -np.log(1 - amax)
    return np.unique(np.round(1 - np.exp(-np.arange(r0, r1 + 1e-9, step)), 8))


# ------------------------------------------------------------------ P matrix
def ranks_to_bins(x, grid):
    """Empirical CDF u of each day and its grid bin (day exceeds node i <=> u > grid[i])."""
    n = len(x)
    u = (np.argsort(np.argsort(x, kind="stable"), kind="stable") + 1.0) / n
    return np.searchsorted(grid, u, side="left"), u


def pmat(bins, M):
    """Joint exceedance probability on the grid, any dimension. bins: (d, n)."""
    d, n = bins.shape
    H = np.zeros((M + 1,) * d, dtype=np.float64)
    np.add.at(H, tuple(np.clip(bins, 0, M)), 1.0)
    C = H
    for ax in range(d):
        C = np.flip(np.cumsum(np.flip(C, axis=ax), axis=ax), axis=ax)
    sl = tuple(slice(1, M + 1) for _ in range(d))
    return C[sl] / n


# ------------------------------------------------------------------ DP
_WF_CACHE = {}


def _wavefronts(shape):
    """Node indices grouped by the sum of their coordinates (cached)."""
    key = tuple(shape)
    if key in _WF_CACHE:
        return _WF_CACHE[key]
    d = len(shape)
    grids = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    ssum = sum(g.ravel() for g in grids)
    order = np.argsort(ssum, kind="stable")
    bounds = np.searchsorted(ssum[order], np.arange(ssum.max() + 2))
    strides = [int(np.prod(shape[ax + 1:])) for ax in range(d)]
    coords = [g.ravel()[order] for g in grids]
    _WF_CACHE[key] = (order, bounds, strides, coords)
    return _WF_CACHE[key]


def dp_path(P):
    """Monotone path from the first to the last grid corner that maximizes sum(P).

    Returns (list of node indices, P along the path, cumulative DP table).
    """
    shape = P.shape
    d = P.ndim
    order, bounds, strides, coords = _wavefronts(shape)
    Pf = P.ravel()
    Df = np.full(Pf.shape, -np.inf)
    Df[0] = Pf[0]
    for s_ in range(1, len(bounds) - 1):
        lo, hi = bounds[s_], bounds[s_ + 1]
        if lo >= hi:
            continue
        f = order[lo:hi]
        cand = np.full(hi - lo, -np.inf)
        for ax in range(d):
            m = coords[ax][lo:hi] > 0
            if m.any():
                np.maximum(cand, np.where(m, Df[f - strides[ax]], -np.inf), out=cand)
        Df[f] = Pf[f] + cand
    D = Df.reshape(shape)

    idx = tuple(n - 1 for n in shape)
    path = [idx]
    while any(i > 0 for i in idx):
        best, bidx = -np.inf, None
        for ax in range(d):
            if idx[ax] > 0:
                prev = list(idx)
                prev[ax] -= 1
                v = D[tuple(prev)]
                if v > best:
                    best, bidx = v, tuple(prev)
        idx = bidx
        path.append(idx)
    path.reverse()
    return path, np.array([P[i] for i in path]), D


# ------------------------------------------------------------------ ranking
def path_depth(us, path, grid):
    """Last path node each day still exceeds on every axis, and its margin to the next node."""
    n = len(us[0])
    A = np.asarray([[grid[i] for i in nd] for nd in path], dtype=float)
    L = len(path)

    k = np.full(n, L, dtype=np.int64)
    for ax in range(len(us)):
        np.minimum(k, np.searchsorted(A[:, ax], us[ax], side="left"), out=k)
    depth = k - 1

    resid = np.full(n, np.inf)
    brk = depth + 1
    m = brk < L
    if m.any():
        idx = brk[m]
        gap = np.full(int(m.sum()), np.inf)
        for ax in range(len(us)):
            np.minimum(gap, us[ax][m] - A[idx, ax], out=gap)
        resid[m] = gap
    return depth.astype(np.int32), resid


def rank_days(us, path, grid):
    """Day order by path depth, then by margin to the next node (descending)."""
    depth, resid = path_depth(us, path, grid)
    return np.lexsort((resid, depth))[::-1]


def calibrate(order, path, pv, p_t, n):
    """Take exactly N = round(p_t * n) days; report the path node closest to p_t."""
    N = int(round(p_t * n))
    m = np.zeros(n, dtype=bool)
    m[order[:N]] = True
    j = int(np.argmin(np.abs(np.log(np.maximum(pv, 1e-12)) - np.log(p_t))))
    return m, path[j], float(pv[j]), N


def solve_pinned(P, i_star, us_red, elig, grid, p_t, n):
    """Temperature fixed at grid[i_star]; re-optimize the other axes and take N days."""
    Psl = P[i_star]
    path, pv, _ = dp_path(Psl)
    order = rank_days(us_red, path, grid)
    order = order[elig[order]]
    N = int(round(p_t * n))
    if len(order) < N:
        raise RuntimeError(f"only {len(order)} eligible days with T fixed at node {i_star}, need {N}")
    m = np.zeros(n, dtype=bool)
    m[order[:N]] = True
    j = int(np.argmin(np.abs(np.log(np.maximum(pv, 1e-12)) - np.log(p_t))))
    return m, path[j]


def q_at(xs_sorted, a):
    """Value of a sorted sample at percentile a."""
    n = len(xs_sorted)
    return float(xs_sorted[min(n - 1, int(np.ceil(a * n)) - 1)])


# ------------------------------------------------------------------ main entry
def iopt(V, p_t, n_total=None, anchor=None, f=0.0, grid=None):
    """Select compound extreme days with iOPT.

    V        list of 1-D arrays, all upper tails: [T] (heat only),
             [T, D] (hot-dry) or [T, D, W] (hot-dry-windy).
    p_t      target probability, relative to n_total days.
    n_total  number of days p_t refers to (default len(V[0])). Use the number
             of all days when V holds growing-season days only.
    anchor   physical threshold for the first variable (same units as V[0]).
    f        0 = data-optimal, 0.5 = half-anchored, 1 = anchored.

    Returns dict with
      mask    boolean array, True on the N event days
      a       threshold percentiles (one per variable)
      thr     thresholds in the units of V
      N       number of event days
      clip    '' / 'floor' (anchor below the median) / 'freq' (anchor too high to give N days)
    """
    grid = make_grid() if grid is None else grid
    d, n, M = len(V), len(V[0]), len(grid)
    n_total = n if n_total is None else n_total
    p_eff = p_t * (n_total / n)
    sorted_ax = [np.sort(x) for x in V]

    bins, us = [], []
    for x in V:
        b, u = ranks_to_bins(x, grid)
        bins.append(b)
        us.append(u)

    if d == 1:
        N = int(round(p_eff * n))
        top = np.argsort(us[0])[::-1][:N]
        m = np.zeros(n, dtype=bool)
        m[top] = True
        a = (float(us[0][top].min()),)
        return dict(mask=m, a=a, thr=(q_at(sorted_ax[0], a[0]),), N=N, clip="")

    P = pmat(np.vstack(bins), M)
    path0, pv0, _ = dp_path(P)
    order0 = rank_days(us, path0, grid)
    m0, nd0, _, N = calibrate(order0, path0, pv0, p_eff, n)
    i0 = nd0[0]

    clip = ""
    if anchor is None or f == 0:
        m, a = m0, tuple(grid[i] for i in nd0)
    else:
        sortedU0 = np.sort(us[0])
        elig_cnt = n - np.searchsorted(sortedU0, grid, side="right")
        i_feas = int(np.nonzero(elig_cnt >= N)[0].max())
        aA_raw = float(np.searchsorted(sorted_ax[0], anchor, side="right")) / n
        if aA_raw < grid[0]:
            clip = "floor"
        aA = min(max(aA_raw, grid[0]), grid[-1])
        r_grid = -np.log1p(-grid)
        iA = int(np.argmin(np.abs(r_grid - (-np.log1p(-aA)))))
        if iA > i_feas:
            iA, clip = i_feas, "freq"
        i_star = iA if f == 1 else int(round((1 - f) * i0 + f * iA))
        elig = us[0] > grid[i_star]
        m, nd_red = solve_pinned(P, i_star, us[1:], elig, grid, p_eff, n)
        a = (grid[i_star],) + tuple(grid[i] for i in nd_red)

    thr = tuple(q_at(sorted_ax[k], a[k]) for k in range(d))
    return dict(mask=m, a=tuple(float(x) for x in a), thr=thr, N=N, clip=clip)
