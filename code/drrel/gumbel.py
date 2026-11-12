"""Studentised Gumbel pivot for items with 41-199 disjoint control windows (PLAN.md section 8.3).

Recommended on 2026-11-11 by an AI model (Claude Fable 5.2) consulted on this one question.

Each control value is Y = S^2 maximised over one window; the maxima of an exponential-tailed
statistic over 15-50 trials are close to Gumbel. Fit (mu, beta) by maximum likelihood to the n
control values, and form T = (Y_claim - mu) / beta. Because Gumbel is a location-scale family, the
null distribution of T depends on n only; its quantiles come from a sealed Monte Carlo table
(gumbel_table.json, built by build_table with a fixed seed), which removes the small-sample bias
of a plug-in p-value. No free-shape extreme-value fit is used.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

TABLE = pathlib.Path(__file__).with_name("gumbel_table.json")
LEVELS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
NS = tuple(range(40, 200, 10))
N_SIM = 100000
SEED = 20261202


def fit(y):
    """Gumbel (max) MLE. y: (..., n). Returns mu, beta with the leading shape of y."""
    y = np.asarray(y, float)
    m = y.mean(-1, keepdims=True)
    b = np.sqrt(6) * y.std(-1, keepdims=True) / np.pi
    lo = y.min(-1, keepdims=True)
    for _ in range(60):     # Newton on g(b) = b - mean(y) + sum(y w)/sum(w), w = exp(-y/b)
        w = np.exp(-(y - lo) / b)
        sw, swy, swy2 = w.sum(-1, keepdims=True), (w * y).sum(-1, keepdims=True), (w * y * y).sum(-1, keepdims=True)
        g = b - m + swy / sw
        dg = 1 + (swy2 / sw - (swy / sw) ** 2) / b ** 2
        b = np.maximum(b - g / dg, 1e-9 + 0 * b)
    mu = lo - b * np.log(np.mean(np.exp(-(y - lo) / b), -1, keepdims=True))
    return mu[..., 0], b[..., 0]


def build_table(n_sim=N_SIM, seed=SEED):
    rng = np.random.default_rng(seed)
    out = {}
    for n in NS:
        q = []
        for _ in range(n_sim // 10000):
            y = rng.gumbel(size=(10000, n + 1))
            mu, b = fit(y[:, :n])
            q.append((y[:, n] - mu) / b)
        q = np.concatenate(q)
        out[str(n)] = {str(a): float(np.quantile(q, 1 - a)) for a in LEVELS}
    TABLE.write_text(json.dumps(dict(seed=seed, n_sim=n_sim, levels=LEVELS, table=out), indent=1))
    return out


def _table():
    return json.loads(TABLE.read_text())["table"]


_T = None


def critical(n, alpha):
    """c(n, alpha): least-squares fit c_inf + a / n across the tabulated n (smooths Monte Carlo
    noise of about 0.05 at alpha = 0.005), evaluated at n clipped to [40, 200]."""
    global _T
    if _T is None:
        _T = _table()
    ns = np.array([int(k) for k in _T], float)
    vals = np.array([_T[str(int(k))][str(alpha)] for k in ns])
    A = np.column_stack([np.ones_like(ns), 1 / ns])
    c_inf, a = np.linalg.lstsq(A, vals, rcond=None)[0]
    n = float(np.clip(n, ns.min(), 200.0))       # the smooth fit is used up to n = 200
    return float(c_inf + a / n)


def pvalue(y_claim, y_controls):
    """Tabulated tail probability of the pivot, stepped to the sealed levels (conservative)."""
    y = np.asarray(y_controls, float)
    mu, b = fit(y)
    t = (y_claim - mu) / b
    p = 1.0
    for a in LEVELS:
        if t >= critical(len(y), a):
            p = a
    return float(p), float(t)


if __name__ == "__main__":
    print(json.dumps(build_table(), indent=1))
