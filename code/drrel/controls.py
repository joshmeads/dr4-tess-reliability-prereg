"""Same-star controls, empirical p-values and power by injection, for one source and one ephemeris.

The claimed window and every control window share the grid size, the duty cycle and the anchor
epoch, so each control has the same number of independent trials as the claim. The time stamps,
noise and scan-angle systematics are the source's own.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import RULES, gumbel
from .stat import best_epoch, free_grid, grid, model, robust_beta, scan


@dataclass(frozen=True)
class Ephemeris:
    tier: str          # "box" or "sine"
    period: float      # d
    sigma_p: float     # d, 1 sigma
    t0: float          # BJD_TDB of a minimum (dip centre)
    duration: float    # d (box); ignored for sine
    amp: float         # predicted depth (box) or semi-amplitude (sine), relative flux
    mode: str = "anchored"   # sealed per item: "anchored" (epoch fixed) or "free" (phase free)
    err_group: str = ""      # period-error group for injections (period_errors.json); "" = none

    @property
    def duty(self):
        return self.duration / self.period if self.tier == "box" else 0.1


def _excluded(r, f, w_rel, rules):
    """True where ratio r (control period / claimed period) sits on a small-integer ratio or on
    the Gaia alias comb of the claimed frequency f."""
    nmax = rules["exclude_ratios_max_int"]
    tol_r = rules["exclude_ratio_tol"] + 2 * w_rel   # the control window must not touch the harmonic window
    bad = np.zeros_like(r, bool)
    for a in range(1, nmax + 1):
        for b in range(1, nmax + 1):
            bad |= np.abs(r - a / b) < tol_r
    fc = f / r
    wf = w_rel * f                                  # window half-width in frequency (claim and controls)
    tol_f = 2.0 / rules["gaia_span_days"] + 2 * wf  # a control window must not touch an alias window
    fp = 1.0 / rules["p_prec"]
    # where windows are wider than the precession comb's spacing the comb cannot be avoided;
    # then only k = 0 (harmonics and spin aliases) is excluded, and the ratio rule covers |k| <= 2
    K = np.arange(-rules["alias_prec_k"], rules["alias_prec_k"] + 1) if 2 * tol_f < 0.8 * fp else np.array([0])
    for h in rules["alias_harmonics"]:
        for j in rules["alias_spin_offsets"]:
            al = np.abs(h * f + j * rules["f_spin"] + K * fp)       # includes reflections through 0
            if j == 0 and h == 1.0:
                al = al[K != 0]                                     # the claim itself is not a control
            if al.size == 0:
                continue
            bad |=np.min(np.abs(fc[:, None] - al[None, :]), axis=1) < tol_f
    return bad


def control_ratios(eph: Ephemeris, rules=RULES):
    """Up to n_control ratios whose windows avoid the claim's aliases and do not overlap one
    another (overlapping windows share maxima and would not be independent controls)."""
    lo, hi = rules["control_ratio"]
    r = np.exp(np.linspace(np.log(lo), np.log(hi), rules["control_candidates"]))
    w_rel = rules["window_nsigma"] * eph.sigma_p / eph.period
    keep = r[~_excluded(r, 1.0 / eph.period, w_rel, rules)]
    f = 1.0 / eph.period
    gap = 2 * w_rel * f + 2.0 / rules["gaia_span_days"]
    disjoint, last = [], -np.inf
    for rr in keep[::-1]:                       # ascending control frequency
        fc = f / rr
        if fc - last >= gap:
            disjoint.append(rr)
            last = fc
    keep = np.array(disjoint[::-1])
    n = rules["n_control"]
    if len(keep) <= n:
        return keep
    return keep[np.round(np.linspace(0, len(keep) - 1, n)).astype(int)]


def testable(eph: Ephemeris, rules=RULES):
    """An item is testable when at least min_controls_pivot disjoint control windows exist."""
    return len(control_ratios(eph, rules)) >= rules["min_controls_pivot"]


def method_for(n_valid, rules=RULES):
    """The p-value method, fixed by the claim's count of valid controls n and used unchanged for
    the claim (n controls) and for every leave-one-out injection (n - 1 controls):
    rank when n >= n_rank_min (the leave-one-out minimum p, 1/n, is then <= alpha); the Gumbel
    pivot when min_controls_pivot <= n < n_rank_min (leave-one-out keeps >= 40, the table's
    smallest n); otherwise None (untestable)."""
    if n_valid >= rules["n_rank_min"]:
        return "rank"
    if n_valid >= rules["min_controls_pivot"]:
        return "pivot"
    return None


def pvalue(S, null, method, rules=RULES):
    """p of statistic S against the finite values of null, by the given method."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if method in ("rank", "rank_extended"):
        return float((1 + np.sum(null >= S)) / (1 + len(null)))
    if method == "pivot":
        return gumbel.pvalue(max(S, 0.0) ** 2, np.maximum(null, 0.0) ** 2)[0]
    return np.nan


_ERR = None


def period_offsets(group, n):
    """n signed period errors in units of sigma_P: the (k + 0.5)/n quantiles of the empirical
    distribution sealed in period_errors.json for this group (calibration half of the catalogued
    eclipsing binaries). An unknown or empty group gives zeros."""
    global _ERR
    if not group:
        return np.zeros(n)
    if _ERR is None:
        import json
        import pathlib
        _ERR = json.loads(pathlib.Path(__file__).with_name("period_errors.json").read_text())["quantiles"]
    q = np.asarray(_ERR[group], float)
    levels = np.linspace(0, 1, len(q))
    return np.interp((np.arange(n) + 0.5) / n, levels, q)


def _window(eph, r, dt_max, rules, span=None):
    """Periods and duration of the window at ratio r (r = 1 is the claim). Every window has the
    claim's width in frequency, so the same number of independent trials."""
    if eph.mode == "free":
        periods, n_eff = free_grid(eph.period * r, eph.sigma_p * r ** 2, span, rules)
        return periods, eph.duration * r, n_eff
    base, n_eff = grid(eph.period, eph.sigma_p, eph.duty, eph.tier, dt_max, rules)
    return base * r ** 2 + eph.period * (r - r ** 2), eph.duration * r, n_eff
    # (P + d) -> r P + r^2 d: the grid offsets scale with P^2, so phase drift per step is unchanged


def test(t, x, s, eph: Ephemeris, amp_for_power=None, rules=RULES, with_power=True):
    """Claimed-window statistic, empirical p from the control windows, and power by injection.

    amp_for_power: amplitude injected for power (default eph.amp).
    Returns a dict; 'status' is OK or NO_TEST (too few data or valid controls).
    """
    order = np.argsort(t)
    t, x, s = t[order], x[order], s[order]
    out = dict(status="NO_TEST", S=np.nan, p=np.nan, power=np.nan, p_best=np.nan, amp=np.nan,
               amp_sigma=np.nan, n_eff=np.nan, n_valid=0, n_points=len(t))
    if len(t) < 20:
        return out
    beta = robust_beta(x, s)
    dt_max = float(np.max(np.abs(t - eph.t0)))
    span = float(t[-1] - t[0])
    periods, dur, n_eff = _window(eph, 1.0, dt_max, rules, span)
    r_claim = scan(eph.tier, t, x, s, periods, eph.t0, dur, beta, eph.mode)
    k = int(np.argmax(r_claim["S"]))
    out.update(mode=eph.mode, S=float(r_claim["S"][k]), p_best=float(periods[k]), amp=float(r_claim["depth"][k]),
               amp_sigma=float(r_claim["sigma"][k]), n_eff=float(n_eff))
    def controls(n_control):
        ratios = control_ratios(eph, {**rules, "n_control": n_control})
        S = np.full(len(ratios), -np.inf)
        ws = []
        for i, r in enumerate(ratios):
            pc, dc, _ = _window(eph, r, dt_max, rules, span)
            ws.append((pc, dc, r))
            S[i] = np.max(scan(eph.tier, t, x, s, pc, eph.t0, dc, beta, eph.mode)["S"])
        return S, ws
    S_ctrl, wins = controls(rules["n_control"])
    valid = np.isfinite(S_ctrl)
    method = method_for(int(valid.sum()), rules)
    if method == "pivot" and np.mean(S_ctrl[valid] <= 0) > rules["pivot_max_nonpositive"]:
        # zero-inflated controls: no Gumbel fits them; use the exact rank test on an extended
        # control set if it has enough valid windows, else no calibrated test exists
        S_ctrl, wins = controls(rules["n_control_extended"])
        valid = np.isfinite(S_ctrl)
        method = "rank_extended" if valid.sum() >= rules["n_rank_min"] else None
        if method is None:
            out["p_method"] = "pivot_inapplicable"
            out["n_valid"] = int(valid.sum())
            return out                                         # NO_TEST
    out["n_valid"] = int(valid.sum())
    out["p_method"] = method
    if method is None:
        return out                                             # NO_TEST
    if not np.isfinite(out["S"]):
        out["status"] = "NO_COVERAGE"                          # the claimed window has no usable data
        if with_power:
            out["power"] = 0.0
        return out
    out["p"] = pvalue(out["S"], S_ctrl, method, rules)
    out["status"] = "OK"
    out["S_controls"] = S_ctrl
    if with_power:
        # Power is the recovery probability IF the claimed signal is real, the same for every
        # source whether it is real or not:
        #  - xm: the series with the fitted claim subtracted, so a real signal is not in the
        #    injections' baseline;
        #  - the null: control maxima of xm plus the predicted signal at the claimed period, so
        #    every source carries the leakage a real signal would put into its control windows
        #    (for a real source this is close to the claim's own null).
        amp = eph.amp if amp_for_power is None else amp_for_power
        xm, t_ref = remove_claim(t, x, s, eph, out["p_best"], dur, out["amp"])
        x_if = xm + model(eph.tier, t, eph.period, t_ref, dur, amp)
        S_if = np.array([np.max(scan(eph.tier, t, x_if, s, pc, eph.t0, dc, beta, eph.mode)["S"])
                         for pc, dc, _ in wins])
        out["power"] = power(t, xm, s, eph, wins, S_if, beta, amp, method, rules)
    return out


def remove_claim(t, x, s, eph, p_best, dur, amp_fit):
    """(x with the fitted claim subtracted, the reference epoch). Box: the fitted depth at the best
    period is added back inside the box; sine: the fitted sinusoid is subtracted. The epoch is the
    TESS t0 (anchored) or the best epoch in the data (free). Applied to every source alike."""
    t_ref = eph.t0 if eph.mode == "anchored" else best_epoch(eph.tier, t, x, s, p_best, dur)
    if eph.tier == "box":
        a = amp_fit if np.isfinite(amp_fit) else 0.0
        return x - model("box", t, p_best, t_ref, dur, a), t_ref
    w = 1.0 / s ** 2
    u = -np.cos(2 * np.pi * (t - t_ref) / p_best)
    X = np.column_stack([np.ones_like(t), u])
    c = np.linalg.lstsq(X * np.sqrt(w)[:, None], x * np.sqrt(w), rcond=None)[0]
    return x - c[1] * u, t_ref


def power(t, x, s, eph, wins, S_ctrl, beta, amp, method, rules=RULES):
    """Fraction of injections recovered at p <= alpha, by the claim's method with the other
    control windows as the null (leave-one-out).

    Injection k goes into the source's own series in every 5th control window, at the window's
    central period plus z_k sigma_P (scaled to the window), where z_k are the quantiles of the
    sealed empirical period-error distribution. So power includes the chance that the true period
    lies outside the +-3 sigma window, as the heavy-tailed real errors require."""
    idx = [i for i in np.arange(0, len(wins), max(1, len(wins) // rules["n_inject"]))[: rules["n_inject"]]
           if np.isfinite(S_ctrl[i])]
    if not idx:
        return 0.0
    z = period_offsets(eph.err_group, len(idx))
    hits = 0
    for zk, i in zip(z, idx):
        pc, dc, r = wins[i]
        p_inj = pc[len(pc) // 2] + zk * eph.sigma_p * r ** 2
        xi = x + model(eph.tier, t, p_inj, eph.t0, dc, amp)
        Si = np.max(scan(eph.tier, t, xi, s, pc, eph.t0, dc, beta, eph.mode)["S"])
        hits += bool(pvalue(Si, np.delete(S_ctrl, i), method, rules) <= rules["alpha"])
    return hits / len(idx)
