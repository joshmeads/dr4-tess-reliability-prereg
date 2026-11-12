"""Simulated DR4-like bundles in the scorer's input format, for validation and tests.

Time stamps come from the simplified scanning law; noise from a smooth per-transit G error
with a 1% outlier rate. Truth per row: 'on_star', 'on_neighbour' or 'none'. A real signal sits
at a true period drawn from N(P, sigma_P), so the ephemeris error is part of the test.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import RULES
from .gaia_io import L_B, T0_TCB
from .nsl import sigma_g, transit_times
from .stat import model


def _tcb_from_tdb(tdb):
    return (tdb - L_B * T0_TCB) / (1 - L_B)


def _flux_rows(source_id, t_tdb, rel, g, rng):
    f0 = 10 ** (-0.4 * (g - 25.6874))           # G zero point of DR3, e-/s
    sig = sigma_g(g) * np.ones_like(t_tdb)
    noise = rng.normal(0, sig)
    out = rng.random(len(t_tdb)) < 0.01
    noise[out] += rng.normal(0, 6 * sig[out])
    flux = f0 * (1 + rel + noise)
    return dict(source_id=np.full(len(t_tdb), source_id, np.int64),
                transit_id=np.arange(len(t_tdb), dtype=np.int64) + source_id % 100000 * 1000,
                g_transit_time=_tcb_from_tdb(t_tdb) - RULES["gaia_time_zero_jd"],
                g_transit_flux=flux, g_transit_flux_error=f0 * sig,
                rejected_by_variability=np.zeros(len(t_tdb), bool))


def make_bundle(out_dir, row, truth, rng, neighbours=None, leak=1.0, amp_scale=1.0, z_true=None):
    """Write one bundle. row: sealed-prediction dict. neighbours: list of (dg, sep_arcsec).
    truth 'on_neighbour' puts the signal on the first neighbour at amplitude
    TESS amp x flux ratio / leak (leak <= 1 is the share of the neighbour's light in TESS).
    The true period is P + z sigma_P, with z = z_true if given, else a standard normal draw."""
    out_dir = pathlib.Path(out_dir)
    neighbours = neighbours or []
    ra, dec, g = row["ra"], row["dec"], row["g"]
    p_true = row["period"] + (rng.normal() if z_true is None else z_true) * row["sigma_p"]
    tier = "box" if row["tier"] == "box" else "sine"
    sources = [dict(source_id=int(row["gaia_dr3"]), g=g, ra=ra, dec=dec, role="target")]
    cols = []
    t = transit_times(ra, dec, seed=int(rng.integers(1 << 30)))
    rel = model(tier, t, p_true, row["t0"], row["duration"], row["amp"] * amp_scale) if truth == "on_star" else 0.0
    cols.append(_flux_rows(int(row["gaia_dr3"]), t, rel, g, rng))
    for j, (dg, sep) in enumerate(neighbours):
        sid = int(row["gaia_dr3"]) + j + 1
        dd = sep / 3600.0
        sources.append(dict(source_id=sid, g=g + dg, ra=ra + dd / max(np.cos(np.radians(dec)), 1e-3), dec=dec,
                            role="neighbour"))
        tn = t + rng.normal(0, 1e-4, len(t))       # same scans, a moment apart
        reln = 0.0
        if truth == "on_neighbour" and j == 0:
            a = min(row["amp"] * 10 ** (0.4 * dg) / leak, 0.95)
            reln = model(tier, tn, p_true, row["t0"], row["duration"], a)
        cols.append(_flux_rows(sid, tn, reln, g + dg, rng))
    tab = {k: np.concatenate([c[k] for c in cols]) for k in cols[0]}
    pq.write_table(pa.table(tab), out_dir / f"{row['row_id']}.epochs.parquet")
    (out_dir / f"{row['row_id']}.sources.json").write_text(json.dumps(sources))
    return p_true
