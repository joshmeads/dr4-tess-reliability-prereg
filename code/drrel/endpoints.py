"""The five primary endpoints, the two validity checks, and their bootstrap intervals.

Rows are per-item result dicts from score.py. kind is 'item' (a sealed prediction), 'shuffled'
(a quiet star given an item's ephemeris) or 'positive' (a catalogued binary recovered blind).
"""
from __future__ import annotations

import numpy as np

from . import RULES

G_BINS = (13.5, 14.0, 14.5, 15.0, 15.6)
CROWD_BINS = (0.0, 0.05, 0.2, 1.0, np.inf)       # neighbour flux ratio within 21 arcsec
N_BOOT = 2000
SEED = 20261202


def _arr(rows, k, default=np.nan):
    return np.array([r.get(k, default) if r.get(k) is not None else default for r in rows])


def false_rates(shuffled):
    """False ON_STAR and ON_NEIGHBOUR rates on quiet stars given borrowed ephemerides,
    among tested controls (outcome not NO_TEST)."""
    t = [r for r in shuffled if r["outcome"] != "NO_TEST"]
    n = len(t)
    if not n:
        return dict(n=0, on=np.nan, nb=np.nan)
    return dict(n=n, on=sum(r["outcome"] == "ON_STAR" for r in t) / n,
                nb=sum(r["outcome"] == "ON_NEIGHBOUR" for r in t) / n)


def purity(rows, a_on):
    """(R - n a) / (sum(power) (1 - a)): recoveries over powers, corrected for false recoveries."""
    if not rows:
        return np.nan
    R = sum(r["outcome"] == "ON_STAR" for r in rows)
    pw = float(np.sum(_arr(rows, "power", 0.0)))
    return (R - len(rows) * a_on) / (pw * (1 - a_on)) if pw > 0 else np.nan


def _boot(fn, rows, shuffled, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        rr = [rows[i] for i in rng.integers(0, len(rows), len(rows))] if rows else []
        ss = [shuffled[i] for i in rng.integers(0, len(shuffled), len(shuffled))] if shuffled else []
        v = fn(rr, ss)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return [np.nan, np.nan]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _est(fn, rows, shuffled):
    return dict(value=float(fn(rows, shuffled)), ci95=_boot(fn, rows, shuffled), n=len(rows))


def e1_purity_clean_eb(items, shuffled):
    """Purity of the clean eb tier at power >= 0.8, by G bin."""
    out = {}
    base = [r for r in items if r.get("clean_eb") and r["outcome"] != "NO_TEST"
            and (r.get("power") or 0) >= RULES["power_strong"]]
    fn = lambda rr, ss: purity(rr, false_rates(ss)["on"] if ss else 0.0)  # noqa: E731
    out["all"] = _est(fn, base, shuffled)
    g = _arr(base, "g")
    for lo, hi in zip(G_BINS[:-1], G_BINS[1:]):
        sel = [r for r, gg in zip(base, g) if lo <= gg < hi]
        out[f"G{lo:.1f}-{hi:.1f}"] = _est(fn, sel, shuffled)
    return out


def _off(rr, ss):
    fr = false_rates(ss) if ss else dict(on=0.0, nb=0.0)
    n = len(rr)
    on = sum(r["outcome"] == "ON_STAR" for r in rr) - n * fr["on"]
    nb = sum(r["outcome"] == "ON_NEIGHBOUR" for r in rr) - n * fr["nb"]
    return nb / (on + nb) if (on + nb) > 0 else np.nan


def e2_base(items):
    """The one E2 population: tested items (any outcome but NO_TEST) whose star has power >= 0.5.
    nb_complete is deliberately not a condition: it turns false when any neighbour's power at its
    minimum implied amplitude is below 0.5, and a signal that really sits on a neighbour usually
    has more than that minimum, so filtering on it removes true off-target detections (in the
    end-to-end simulation it gave 0.175 against a true 0.266)."""
    return [r for r in items if r["outcome"] in ("ON_STAR", "ON_NEIGHBOUR", "NOWHERE", "UNDECIDED")
            and (r.get("power") or 0) >= RULES["power_decided"]]


def e2_off_target(items, shuffled):
    """Off-target fraction among recovered signals, ON_NEIGHBOUR / (ON_STAR + ON_NEIGHBOUR), each
    count corrected by n x its false rate on shuffled controls, over e2_base(items); by crowding
    (neighbour flux within 21 arcsec over the star's flux)."""
    base = e2_base(items)
    out = {"all": _est(_off, base, shuffled)}
    c = _arr(base, "f21", 0.0)
    for lo, hi in zip(CROWD_BINS[:-1], CROWD_BINS[1:]):
        sel = [r for r, cc in zip(base, c) if lo <= cc < hi]
        out[f"f21_{lo}-{hi}"] = _est(_off, sel, shuffled)
    return out


def gaia_unequal(r):
    """Gaia's P-versus-2P call, or None where it cannot decide."""
    d, s, a = r.get("p2p_diff"), r.get("p2p_sigma"), r.get("p2p_amp")
    if d is None or s is None or a is None or not (np.isfinite(d) and np.isfinite(s) and np.isfinite(a)):
        return None
    if a <= 0 or s > RULES["p2p_decidable_rel_sigma"] * abs(a):
        return None
    return bool(abs(d) / s >= RULES["p2p_unequal_nsigma"])


def e3_p2p(items, shuffled):
    """Accuracy of the sealed P-versus-2P call (unequal alternate minima) where Gaia decides."""
    base = [r for r in items if r["outcome"] == "ON_STAR" and gaia_unequal(r) is not None]
    fn = lambda rr, ss: (np.mean([bool(r["call_unequal"]) == gaia_unequal(r) for r in rr])  # noqa: E731
                         if rr else np.nan)
    out = _est(fn, base, shuffled)
    out["confusion"] = {f"call_{a}_gaia_{b}": sum(bool(r["call_unequal"]) == a and gaia_unequal(r) == b for r in base)
                        for a in (False, True) for b in (False, True)}
    return out


def forecast(r, a_on):
    """Probability of the observed outcome ON_STAR: p_real_on x power, plus false recoveries."""
    q = r["p_pred"] * r["power"]
    return q + (1 - q) * a_on


def _brier(rr, ss):
    a = false_rates(ss)["on"] if ss else 0.0
    if not rr:
        return np.nan
    q = np.array([forecast(r, a) for r in rr]); y = np.array([r["outcome"] == "ON_STAR" for r in rr], float)
    return float(np.mean((q - y) ** 2))


def _bss(rr, ss):
    if not rr:
        return np.nan
    y = np.array([r["outcome"] == "ON_STAR" for r in rr], float)
    ref = float(np.mean((y.mean() - y) ** 2))
    return 1 - _brier(rr, ss) / ref if ref > 0 else np.nan


def e4_calibration(items, shuffled):
    """Brier and log score of the sealed probabilities, with a reliability diagram.
    Items outside the Kepler calibration set, with power >= 0.5."""
    base = [r for r in items if r["outcome"] not in ("NO_TEST",) and not r.get("kepler_cal")
            and (r.get("power") or 0) >= RULES["power_decided"] and r.get("p_pred") is not None]
    a = false_rates(shuffled)["on"] if shuffled else 0.0
    out = dict(brier=_est(_brier, base, shuffled), brier_skill=_est(_bss, base, shuffled))
    if base:
        q = np.clip([forecast(r, a) for r in base], 1e-6, 1 - 1e-6)
        y = np.array([r["outcome"] == "ON_STAR" for r in base], float)
        out["log_score"] = float(np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
        edges = np.linspace(0, 1, 11)
        k = np.clip(np.digitize(q, edges) - 1, 0, 9)
        out["reliability"] = [dict(bin=[float(edges[i]), float(edges[i + 1])], n=int((k == i).sum()),
                                   mean_forecast=float(q[k == i].mean()) if (k == i).any() else None,
                                   observed=float(y[k == i].mean()) if (k == i).any() else None)
                              for i in range(10)]
    return out


def _z(rr):
    return np.array([(r["p_best"] - r["period"]) / r["sigma_p"] for r in rr])


def e5_period(items, shuffled):
    """(Gaia period - TESS period) / sigma_P for recovered items that were not used to calibrate
    sigma_P (sigma_cal false). The robust standard deviation is 1.4826 x the MAD about the median;
    the median (a shift) is reported separately."""
    base = [r for r in items if r["outcome"] == "ON_STAR" and (r.get("sigma_p") or 0) > 0 and not r.get("sigma_cal")]
    if not base:
        return dict(n=0)
    z = _z(base)
    edge = RULES["window_nsigma"] * 0.97
    def rstd(rr, ss):
        zz = _z(rr)
        return 1.4826 * float(np.median(np.abs(zz - np.median(zz)))) if len(zz) else np.nan
    return dict(n=len(z), median=_est(lambda rr, ss: float(np.median(_z(rr))) if rr else np.nan, base, shuffled),
                robust_std=_est(rstd, base, shuffled),
                frac_within=[float(np.mean(np.abs(z) <= k)) for k in (1, 2, 3)],
                frac_at_edge=float(np.mean(np.abs(z) >= edge)))


def check_false_rate(shuffled):
    fr = false_rates(shuffled)
    passed = bool(fr["n"] > 0 and fr["on"] <= 2 * RULES["alpha"])
    return dict(fr, passed=passed, rule="false ON_STAR rate on shuffled controls <= 2 alpha")


def check_positive(positive):
    """Observed recovery of catalogued binaries tracks predicted power."""
    t = [r for r in positive if r["outcome"] != "NO_TEST" and (r.get("power") or 0) >= RULES["power_decided"]]
    if len(t) < 30:
        return dict(n=len(t), passed=False, reason="fewer than 30 powered positive controls")
    pw = _arr(t, "power", 0.0); y = np.array([r["outcome"] == "ON_STAR" for r in t], float)
    ratio = y.sum() / pw.sum()
    bins = []
    ok = 0.8 <= ratio <= 1.15
    for lo, hi in ((0.5, 0.8), (0.8, 0.95), (0.95, 1.01)):
        m = (pw >= lo) & (pw < hi)
        if m.sum() >= 30:
            bins.append(dict(bin=[lo, hi], n=int(m.sum()), power=float(pw[m].mean()), observed=float(y[m].mean())))
            ok &= y[m].mean() >= pw[m].mean() - 0.2
    return dict(n=len(t), ratio=float(ratio), bins=bins, passed=bool(ok),
                rule="observed/predicted in [0.8, 1.15], and observed >= power - 0.2 in every bin with n >= 30")


def secondary(items, shuffled, purity_allowed=True):
    """Descriptive only (not endpoints): purity at power >= 0.8 and the off-target fraction per
    tier stratum, sigma_P/P bin, and whether DR3 epoch photometry was public."""
    out = {}
    a_on = false_rates(shuffled)["on"] if shuffled else 0.0
    for key in ("stratum", "sigma_rel_bin", "dr3_epoch_public"):
        groups = {}
        for r in items:
            groups.setdefault(str(r.get(key)), []).append(r)
        out[key] = {}
        for g, rr in sorted(groups.items()):
            strong = [r for r in rr if r["outcome"] != "NO_TEST" and (r.get("power") or 0) >= RULES["power_strong"]]
            tested = e2_base(rr)
            out[key][g] = dict(n=len(rr), n_strong=len(strong),
                               purity=(float(purity(strong, a_on)) if strong else None) if purity_allowed else "withheld",
                               off_target=float(_off(tested, shuffled)) if tested else None)
    return out


def all_endpoints(items, shuffled, positive):
    fr = check_false_rate(shuffled)
    pc = check_positive(positive)
    if pc["passed"]:
        e1 = e1_purity_clean_eb(items, shuffled)
    else:                                   # withheld: no number is computed or reported
        e1 = dict(withheld=True, reason="power model failed its positive controls (check b)")
    return dict(check_false_rate=fr, check_positive_controls=pc, E1_purity_clean_eb=e1,
                E2_off_target=e2_off_target(items, shuffled), E3_p_vs_2p=e3_p2p(items, shuffled),
                E4_calibration=e4_calibration(items, shuffled), E5_period=e5_period(items, shuffled),
                secondary_descriptive=secondary(items, shuffled, purity_allowed=pc["passed"]))
