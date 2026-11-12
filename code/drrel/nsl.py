"""A simplified Gaia scanning law, for simulation and coarse power planning only.

Not used to score. The real DR4 time stamps replace it after release. It keeps what matters
for the window function: a 6 h spin, two fields of view 106.5 deg apart, a 45 deg solar aspect
angle with a 63-day precession, and the yearly motion of the Sun. Uniform precession, no
dead time other than a random loss fraction.
"""
from __future__ import annotations

import numpy as np

SPIN = 0.25            # d
PREC = 63.12           # d
XI = np.radians(45.0)
BASIC = np.radians(106.5)
HALF_AC = np.radians(0.35)   # across-scan half width of the astrometric field
DR4_START = 2456863.94       # 2014-07-25, BJD
DR4_END = 2458869.0          # 2020-01-20
OBLIQ = np.radians(23.4392911)


def ecliptic_unit(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.array([x, y * np.cos(OBLIQ) + z * np.sin(OBLIQ), -y * np.sin(OBLIQ) + z * np.cos(OBLIQ)])


def _sun_lon(t):
    d = t - 2451545.0
    g = np.radians(357.528 + 0.9856003 * d)
    return np.radians(280.460 + 0.9856474 * d + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))


def _axes(t, nu0=0.0):
    lam = _sun_lon(t)
    s = np.stack([np.cos(lam), np.sin(lam), np.zeros_like(lam)], -1)
    q = np.stack([-np.sin(lam), np.cos(lam), np.zeros_like(lam)], -1)
    e = np.array([0.0, 0.0, 1.0])
    nu = nu0 + 2 * np.pi * (t - DR4_START) / PREC
    z = np.cos(XI) * s + np.sin(XI) * (np.cos(nu)[:, None] * e + np.sin(nu)[:, None] * q)
    a = np.cross(s, z)
    a /= np.linalg.norm(a, axis=1)[:, None]
    b = np.cross(z, a)
    return z, a, b


def transit_times(ra_deg, dec_deg, start=DR4_START, end=DR4_END, loss=0.1, seed=0, step=0.01):
    """FoV transit times (BJD) of a star under the simplified scanning law."""
    rng = np.random.default_rng(seed)
    u = ecliptic_unit(ra_deg, dec_deg)
    t = np.arange(start, end, step)
    z, a, b = _axes(t)
    c = z @ u
    inside = np.abs(c) < np.sin(HALF_AC)
    if not inside.any():
        return np.array([])
    d = np.diff(np.concatenate([[0], inside.astype(int), [0]]))
    starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    out = []
    omega = 2 * np.pi / SPIN
    for i0, i1 in zip(starts, stops):
        ta, tb = t[i0], t[i1] + step
        psi = np.arctan2(b[i0] @ u, a[i0] @ u)
        for off in (0.0, BASIC):
            first = ta + ((psi - omega * (ta - start) - off) % (2 * np.pi)) / omega
            out.extend(np.arange(first, tb, SPIN))
    out = np.sort(np.array(out))
    keep = rng.random(len(out)) > loss
    return out[keep]


def sigma_g(g):
    """Per-transit fractional G flux error, a smooth stand-in for the DR3 per-transit errors."""
    return np.sqrt(1.0e-3 ** 2 + (2.5e-3 * 10 ** (0.2 * (np.asarray(g) - 15.0))) ** 2)
