"""
Distributions described by three quantiles - the shape every forecast here has.

The forecaster and the delay model both state an outcome as its 10th, 50th and
90th percentiles. These helpers turn that into a CDF, or into samples, with
linear tails out to the 1st and 99th percentile.
"""
from __future__ import annotations

import numpy as np

_PROBS = np.array([0.01, 0.10, 0.50, 0.90, 0.99])


def _knots(lo, mid, hi):
    lo, mid, hi = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (lo, mid, hi))
    left = np.maximum(mid - lo, 1e-6)
    right = np.maximum(hi - mid, 1e-6)
    return np.stack([lo - 1.5 * left, lo, mid, hi, hi + 1.5 * right], axis=1)


def quantile_cdf(lo, mid, hi, x):
    """Predictive CDF through the 10th, 50th and 90th percentiles."""
    xs = _knots(lo, mid, hi)
    x = np.broadcast_to(np.atleast_1d(np.asarray(x, dtype=float)), (len(xs),))
    out = np.empty(len(xs))
    for i in range(len(xs)):
        out[i] = np.interp(x[i], xs[i], _PROBS)
    return out


def quantile_sample(lo, mid, hi, n: int, rng) -> np.ndarray:
    """n draws from each distribution; shape (len(lo), n)."""
    xs = _knots(lo, mid, hi)
    u = rng.uniform(_PROBS[0], _PROBS[-1], size=(len(xs), n))
    out = np.empty_like(u)
    for i in range(len(xs)):
        out[i] = np.interp(u[i], _PROBS, xs[i])
    return out
