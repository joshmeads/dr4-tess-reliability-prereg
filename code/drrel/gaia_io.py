"""Read Gaia epoch photometry (DR3 layout; DR4 by column mapping) into (t BJD_TDB, x, s).

Format adapter only: it never changes a statistic. Rows rejected by the photometric or
variability processing are dropped, as are non-positive or non-finite fluxes. Several rows
per FoV transit (per-CCD photometry) are combined into one point per transit.
"""
from __future__ import annotations

import numpy as np

from . import RULES

TIME = ("g_transit_time", "time", "g_obs_time", "obs_time")
FLUX = ("g_transit_flux", "flux", "g_flux")
ERR = ("g_transit_flux_error", "flux_error", "g_flux_error")
REJECT = ("variability_flag_g_reject", "rejected_by_variability", "rejected_by_photometry", "g_rejected")
TRANSIT = ("transit_id",)
BAND = ("band",)
L_B = 1.550519768e-8
T0_TCB = 2443144.5003725


def tcb_to_tdb(jd_tcb):
    """TDB = TCB - L_B (JD_TCB - T0) (IAU 2006 B3), dropping the 6.55e-5 s constant."""
    return jd_tcb - L_B * (jd_tcb - T0_TCB)


def _pick(cols, names):
    for n in names:
        if n in cols:
            return n
    return None


def _truthy(v):
    if v.dtype.kind == "b":
        return v
    if v.dtype.kind in "iuf":
        return np.nan_to_num(v.astype(float)) != 0
    return np.array([str(a).strip().lower() in ("true", "1", "t", "yes") for a in v])


def series(table: dict, time_zero_jd=RULES["gaia_time_zero_jd"]):
    """table: mapping column -> array (one source). Returns t, x, s (sorted) or None if unusable."""
    cols = set(table)
    tc, fc, ec = _pick(cols, TIME), _pick(cols, FLUX), _pick(cols, ERR)
    if not (tc and fc and ec):
        raise KeyError(f"missing time/flux/error columns in {sorted(cols)}")
    t = np.asarray(table[tc], float)
    f = np.asarray(table[fc], float)
    e = np.asarray(table[ec], float)
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (f > 0) & (e > 0)
    bc = _pick(cols, BAND)
    if bc:
        ok &= np.array([str(b).strip().upper() == "G" for b in table[bc]])
    for r in REJECT:
        if r in cols:
            ok &= ~_truthy(np.asarray(table[r]))
    t, f, e = t[ok], f[ok], e[ok]
    trc = _pick(cols, TRANSIT)
    if trc is not None and len(t):
        tid = np.asarray(table[trc])[ok]
        u, inv, cnt = np.unique(tid, return_inverse=True, return_counts=True)
        if (cnt > 1).any():
            w = 1 / e ** 2
            W = np.bincount(inv, w)
            t = np.bincount(inv, w * t) / W
            f = np.bincount(inv, w * f) / W
            e = 1 / np.sqrt(W)
    if len(t) < 20:
        return None
    jd = t + time_zero_jd if np.nanmedian(t) < 1e5 else t
    tdb = tcb_to_tdb(jd)
    med = np.median(f)
    order = np.argsort(tdb)
    return tdb[order], (f / med - 1.0)[order], (e / med)[order]
