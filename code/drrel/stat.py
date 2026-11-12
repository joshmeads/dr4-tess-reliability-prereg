"""The test statistic: epoch anchored at the TESS value, period scanned over a narrow window.

box tier : matched-filter depth significance of a box of the TESS duration
sine tier: significance of a sinusoid's semi-amplitude, with the minimum anchored at t0

Every function takes one source's series: t (BJD_TDB), x (relative flux minus 1), s (1-sigma).
"""
from __future__ import annotations

import numpy as np

from . import RULES


def robust_beta(x, s):
    """Error inflation for one series: robust scatter of the normalised residuals, at least 1."""
    z = (x - np.median(x)) / s
    return max(1.0, 1.4826 * float(np.median(np.abs(z - np.median(z)))))


def grid(period, sigma_p, duty, tier, dt_max, rules=RULES):
    """Period grid over P +- n sigma. Spacing keeps the phase drift at the farthest epoch below
    a fixed step. Returns (periods, n_eff), n_eff = independent periods (2 w_f dt_max)."""
    m_need = grid_size(period, sigma_p, duty, tier, dt_max, rules)
    m = int(np.clip(m_need, rules["grid_min"], rules["grid_max"]))
    if m % 2 == 0:
        m += 1                       # odd count keeps the claimed period on the grid
    w = rules["window_nsigma"] * sigma_p
    n_eff = max(1.0, 2 * w * dt_max / period ** 2)
    return np.linspace(period - w, period + w, m), n_eff


def grid_size(period, sigma_p, duty, tier, dt_max, rules=RULES):
    """Uncapped number of anchored grid periods the window needs."""
    w = rules["window_nsigma"] * sigma_p
    step = rules["grid_phase_step_box"] * duty if tier == "box" else rules["grid_phase_step_sine"]
    dp = step * period ** 2 / max(dt_max, 1e-9)
    return int(np.ceil(2 * w / dp) + 1)


def _phase(t, t0, periods):
    return ((t[None, :] - t0) / periods[:, None] + 0.5) % 1.0 - 0.5


def box_scan(t, x, s, periods, t0, duration, beta=None, rules=RULES):
    """Box statistic at each trial period. Returns dict of arrays: S, depth, sigma, n_in, n_cyc.

    S = (weighted mean outside - weighted mean inside) / sigma. Points within the box at phase
    0.5 (a possible secondary eclipse) are left out of the baseline. A trial period with fewer
    than min_in_points inside, or from fewer than min_in_cycles distinct cycles, gets S = -inf.
    """
    periods = np.atleast_1d(np.asarray(periods, float))
    durations = np.broadcast_to(np.asarray(duration, float), periods.shape)
    beta = robust_beta(x, s) if beta is None else beta
    w = 1.0 / s ** 2
    ph = _phase(t, t0, periods)
    half = (durations / (2 * periods))[:, None]
    inside = np.abs(ph) < half
    sec = np.abs(np.abs(ph) - 0.5) < half
    out = ~inside & ~sec
    W_in = inside @ w
    W_out = out @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        m_in = (inside @ (w * x)) / W_in
        m_out = (out @ (w * x)) / W_out
        sig = beta * np.sqrt(1.0 / W_in + 1.0 / W_out)
        depth = m_out - m_in
        S = depth / sig
    n_in = inside.sum(1)
    # distinct cycles among inside points; t is sorted, so cycle numbers never decrease along a row
    cyc = np.where(inside, np.floor((t[None, :] - t0) / periods[:, None] + 0.5), -np.inf)
    prev = np.concatenate([np.full((len(periods), 1), -np.inf), np.maximum.accumulate(cyc, axis=1)[:, :-1]], axis=1)
    n_cyc = (inside & (cyc > prev)).sum(1)
    bad = (n_in < rules["min_in_points"]) | (n_cyc < rules["min_in_cycles"]) | ~np.isfinite(S)
    S = np.where(bad, -np.inf, S)
    return dict(S=S, depth=depth, sigma=sig, n_in=n_in, n_cyc=n_cyc)


def sine_scan(t, x, s, periods, t0, beta=None):
    """Anchored sinusoid x = c - A cos(2 pi (t - t0) / P). Returns S = A / sigma_A, A, sigma."""
    periods = np.atleast_1d(np.asarray(periods, float))
    beta = robust_beta(x, s) if beta is None else beta
    w = 1.0 / s ** 2
    u = -np.cos(2 * np.pi * (t[None, :] - t0) / periods[:, None])   # basis with a minimum at t0
    Sw, Sx = w.sum(), (w * x).sum()
    Su = u @ w
    Suu = (u * u) @ w
    Sux = u @ (w * x)
    det = Sw * Suu - Su ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        A = (Sw * Sux - Su * Sx) / det
        sig = beta * np.sqrt(Sw / det)
        S = A / sig
    S = np.where(np.isfinite(S), S, -np.inf)
    return dict(S=S, depth=A, sigma=sig)


def sine_free_scan(t, x, s, periods, beta=None):
    """Free-phase sinusoid x = c + a sin + b cos at each period. S = sqrt(chi2 drop) / beta."""
    periods = np.atleast_1d(np.asarray(periods, float))
    beta = robust_beta(x, s) if beta is None else beta
    w = 1.0 / s ** 2
    tm = np.median(t)
    ph = 2 * np.pi * (t[None, :] - tm) / periods[:, None]
    c, sn = np.cos(ph), np.sin(ph)
    Sw = w.sum()
    A = np.empty((len(periods), 3, 3))
    A[:, 0, 0] = Sw
    A[:, 0, 1] = A[:, 1, 0] = c @ w
    A[:, 0, 2] = A[:, 2, 0] = sn @ w
    A[:, 1, 1] = (c * c) @ w
    A[:, 2, 2] = (sn * sn) @ w
    A[:, 1, 2] = A[:, 2, 1] = (c * sn) @ w
    b = np.stack([np.full(len(periods), (w * x).sum()), c @ (w * x), sn @ (w * x)], -1)
    try:
        coef = np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return dict(S=np.full(len(periods), -np.inf), depth=np.full(len(periods), np.nan),
                    sigma=np.full(len(periods), np.nan))
    xm = (w * x).sum() / Sw
    chi_drop = np.einsum("ij,ij->i", coef, b) - Sw * xm ** 2
    amp = np.hypot(coef[:, 1], coef[:, 2])
    with np.errstate(invalid="ignore"):
        S = np.sqrt(np.clip(chi_drop, 0, None)) / beta
        sig = np.where(S > 0, amp / S, np.nan)
    return dict(S=np.where(np.isfinite(S), S, -np.inf), depth=amp, sigma=sig)


def box_free_scan(t, x, s, periods, duration, beta=None, rules=RULES):
    """Box at a free epoch: for each period, the best of 4 epoch positions per duty cycle."""
    periods = np.atleast_1d(np.asarray(periods, float))
    durations = np.broadcast_to(np.asarray(duration, float), periods.shape)
    beta = robust_beta(x, s) if beta is None else beta
    w = 1.0 / s ** 2
    Wt, Xt = w.sum(), (w * x).sum()
    S_out = np.full(len(periods), -np.inf); d_out = np.full(len(periods), np.nan); s_out = d_out.copy()
    for i, (P, D) in enumerate(zip(periods, durations)):
        q = D / P
        ph = ((t - t[0]) / P) % 1.0
        o = np.argsort(ph)
        p2 = np.concatenate([ph[o], ph[o] + 1.0])
        cw = np.concatenate([[0.0], np.cumsum(np.concatenate([w[o], w[o]]))])
        cx = np.concatenate([[0.0], np.cumsum(np.concatenate([(w * x)[o], (w * x)[o]]))])
        starts = np.arange(0, 1, q / rules["free_phase_per_duty"])
        a = np.searchsorted(p2, starts)
        b = np.searchsorted(p2, starts + q)
        n_in = b - a
        W_in = cw[b] - cw[a]
        X_in = cx[b] - cx[a]
        W_o = Wt - W_in
        with np.errstate(invalid="ignore", divide="ignore"):
            dep = (Xt - X_in) / W_o - X_in / W_in
            sg = beta * np.sqrt(1 / W_in + 1 / W_o)
            S = np.where(n_in >= rules["free_min_in_points"], dep / sg, -np.inf)
        k = int(np.argmax(S))
        S_out[i], d_out[i], s_out[i] = S[k], dep[k], sg[k]
    return dict(S=np.where(np.isfinite(S_out), S_out, -np.inf), depth=d_out, sigma=s_out)


def scan(tier, t, x, s, periods, t0, duration, beta=None, mode="anchored"):
    if mode == "free":
        if tier == "box":
            return box_free_scan(t, x, s, periods, duration, beta)
        return sine_free_scan(t, x, s, periods, beta)
    if tier == "box":
        return box_scan(t, x, s, periods, t0, duration, beta)
    return sine_scan(t, x, s, periods, t0, beta)


def best_epoch(tier, t, x, s, period, duration, rules=RULES):
    """Free mode: the epoch of minimum light at a given period, from the data."""
    if tier == "box":
        w = 1.0 / s ** 2
        q = duration / period
        ph = ((t - t[0]) / period) % 1.0
        starts = np.arange(0, 1, q / rules["free_phase_per_duty"])
        best, b0 = -np.inf, 0.0
        for a in starts:
            m = ((ph - a) % 1.0) < q
            if m.sum() < rules["free_min_in_points"] or (~m).sum() < 2:
                continue
            d = np.average(x[~m], weights=w[~m]) - np.average(x[m], weights=w[m])
            if d > best:
                best, b0 = d, a
        return float(t[0] + (b0 + q / 2) * period)
    w = 1.0 / s ** 2
    ph = 2 * np.pi * (t - t[0]) / period
    X = np.column_stack([np.ones_like(t), np.sin(ph), np.cos(ph)])
    c = np.linalg.lstsq(X * np.sqrt(w)[:, None], x * np.sqrt(w), rcond=None)[0]
    phi = np.arctan2(c[1], c[2])
    return float(t[0] + (phi + np.pi) / (2 * np.pi) * period)


def choose_mode(tier, period, sigma_p, duty, dt_max, rules=RULES):
    """'anchored' when the anchored grid fits in grid_max periods at dt_max, else 'free'."""
    return "anchored" if grid_size(period, sigma_p, duty, tier, dt_max, rules) <= rules["grid_max"] else "free"


def free_grid(period, sigma_p, span, rules=RULES):
    """Frequency grid f +- n sigma_f at a step of 1 / (oversample x span). Returns periods, n_eff."""
    f = 1.0 / period
    wf = rules["window_nsigma"] * sigma_p / period ** 2
    step = 1.0 / (rules["free_oversample"] * span)
    m = int(np.clip(np.ceil(2 * wf / step) + 1, rules["grid_min"], rules["free_grid_max"]))
    if m % 2 == 0:
        m += 1
    fr = np.linspace(f - wf, f + wf, m)
    return 1.0 / fr, max(1.0, 2 * wf * span)


def window_max(tier, t, x, s, periods, t0, duration, beta=None):
    """Maximum of the statistic over one window, with the period and amplitude at the maximum."""
    r = scan(tier, t, x, s, periods, t0, duration, beta)
    k = int(np.argmax(r["S"]))
    return float(r["S"][k]), float(np.atleast_1d(periods)[k]), float(r["depth"][k]), float(r["sigma"][k])


def model(tier, t, period, t0, duration, amp):
    """The signal injected for power: a box dip of depth amp, or a sinusoid of semi-amplitude amp."""
    if tier == "box":
        ph = ((t - t0) / period + 0.5) % 1.0 - 0.5
        return np.where(np.abs(ph) < duration / (2 * period), -amp, 0.0)
    return -amp * np.cos(2 * np.pi * (t - t0) / period)


def odd_even(tier, t, x, s, period, t0, duration, beta=None):
    """P versus 2P: amplitude difference between alternate cycles at the best period.

    Returns (diff, sigma_diff, amp) where amp is the mean amplitude. For the box tier diff is the
    depth at even cycles minus odd cycles; for the sine tier it is the 2P cosine coefficient.
    """
    beta = robust_beta(x, s) if beta is None else beta
    if tier == "box":
        r0 = box_scan(t, x, s, [2 * period], t0, duration, beta, rules={**RULES, "min_in_cycles": 2})
        r1 = box_scan(t, x, s, [2 * period], t0 + period, duration, beta, rules={**RULES, "min_in_cycles": 2})
        if not (np.isfinite(r0["S"][0]) and np.isfinite(r1["S"][0])):
            return np.nan, np.nan, np.nan
        d0, d1 = r0["depth"][0], r1["depth"][0]
        return float(d0 - d1), float(np.hypot(r0["sigma"][0], r1["sigma"][0])), float(0.5 * (d0 + d1))
    w = 1.0 / s ** 2
    ph = 2 * np.pi * (t - t0) / period
    X = np.column_stack([np.ones_like(t), -np.cos(ph), np.sin(ph), np.cos(ph / 2), np.sin(ph / 2)])
    A = X.T @ (X * w[:, None])
    try:
        cov = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan
    c = cov @ (X.T @ (w * x))
    return float(c[3]), float(beta * np.sqrt(cov[3, 3])), float(c[1])
