"""Gaia DR3 columns for every searched star (IRSA TAP, upload join), resumable by batch.

usage: dr3_fetch.py cols      # gaia_source columns for all searched stars
       dr3_fetch.py nbr N     # neighbour context for signal stars + N quiet comparison stars

Neighbour context, per target, from gaia_dr3_source within 63 arcsec (3 TESS pixels) and
dG < 6 mag: flux ratio sums inside 10.5, 21, 42 and 63 arcsec, the count within 21 arcsec
with dG < 5, and the brightest neighbour's dG within 21 and 63 arcsec.
A batch is written only when the service answered; missing batches are re-tried on rerun.
"""
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import tap  # noqa: E402

D = pathlib.Path(__file__).resolve().parents[1] / "data"
COLS = ("source_id, phot_g_mean_mag, phot_g_n_obs, phot_g_mean_flux_over_error, bp_rp, "
        "phot_bp_rp_excess_factor, ruwe, ipd_frac_multi_peak, ipd_gof_harmonic_amplitude, "
        "matched_transits, visibility_periods_used, has_epoch_photometry, phot_variable_flag, "
        "non_single_star, ecl_lat, l, b, teff_gspphot, parallax, parallax_over_error")


QUIET_SAMPLE = 150000
SEED = 20261202


def needed_ids():
    """Signal stars, plus a seeded random sample of quiet stars (no significant peak at all)."""
    st = pq.read_table(D / "searched_stars.parquet", columns=["gaia_dr3", "has_signal", "has_peak"]).to_pydict()
    gaia = np.asarray(st["gaia_dr3"], dtype=np.int64)
    sig = np.asarray(st["has_signal"]); peak = np.asarray(st["has_peak"])
    quiet = np.sort(gaia[~peak])
    rng = np.random.default_rng(SEED)
    pick = rng.choice(quiet, size=min(QUIET_SAMPLE, len(quiet)), replace=False)
    return np.unique(np.concatenate([gaia[sig], pick])), np.sort(pick)


def cols(reverse=False):
    out = D / "dr3_parts"; out.mkdir(exist_ok=True)
    have = set()
    for p in out.glob("cols_*.parquet"):
        have.update(pq.read_table(p, columns=["source_id"])["source_id"].to_pylist())
    ids, _ = needed_ids()
    ids = np.array([i for i in ids if int(i) not in have], dtype=np.int64)
    print("still needed:", len(ids), flush=True)
    B = 50000
    starts = list(range(0, len(ids), B))
    for k in (starts[::-1] if reverse else starts):
        f = out / f"cols_id{int(ids[k])}_{min(B, len(ids) - k)}.parquet"
        if f.exists():
            continue
        chunk = ids[k:k + B]
        cl = ", ".join("g." + c.strip() for c in COLS.split(","))
        t, _ = tap.query_retry(tap.IRSA, f"SELECT {cl} FROM gaia_dr3_source g JOIN TAP_UPLOAD.up u "
                               "ON g.source_id = u.source_id", upload=tap.id_table(chunk))
        if len(t) < 0.99 * len(chunk):
            raise RuntimeError(f"batch {k // B}: {len(t)} rows for {len(chunk)} ids; refusing a short answer")
        pq.write_table(tap.to_arrow(t), f.with_suffix(".tmp"))
        f.with_suffix(".tmp").rename(f)
        print(f"cols batch {k // B}: {len(t)} rows", flush=True)


NBR = ("SELECT u.source_id, "
       "SUM(CASE WHEN d < 10.5/3600.0 THEN POWER(10.0, -0.4*dg) ELSE 0 END) AS f10, "
       "SUM(CASE WHEN d < 21.0/3600.0 THEN POWER(10.0, -0.4*dg) ELSE 0 END) AS f21, "
       "SUM(CASE WHEN d < 42.0/3600.0 THEN POWER(10.0, -0.4*dg) ELSE 0 END) AS f42, "
       "SUM(POWER(10.0, -0.4*dg)) AS f63, "
       "SUM(CASE WHEN d < 21.0/3600.0 AND dg < 5 THEN 1 ELSE 0 END) AS n21, "
       "MIN(CASE WHEN d < 21.0/3600.0 THEN dg ELSE 99 END) AS dgmin21, "
       "MIN(dg) AS dgmin63, COUNT(*) AS n63 "
       "FROM (SELECT u.source_id, DISTANCE(POINT('ICRS', g.ra, g.dec), POINT('ICRS', u.ra, u.dec)) AS d, "
       "g.phot_g_mean_mag - u.gmag AS dg FROM TAP_UPLOAD.up u JOIN gaia_dr3_source g "
       "ON 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', u.ra, u.dec, 0.0175)) "
       "WHERE g.source_id <> u.source_id AND g.phot_g_mean_mag < u.gmag + 6) u GROUP BY u.source_id")


def nbr(n_quiet):
    st = pq.read_table(D / "searched_stars.parquet").to_pydict()
    gaia = np.asarray(st["gaia_dr3"], dtype=np.int64)
    sig = np.asarray(st["has_signal"]); peak = np.asarray(st["has_peak"])
    qs = set(needed_ids()[1][:n_quiet].tolist())
    pick = np.array([i for i in np.nonzero(~peak)[0] if int(gaia[i]) in qs], dtype=int)
    idx = np.concatenate([np.nonzero(sig)[0], pick])
    idx = idx[np.argsort(gaia[idx])]
    ra = np.asarray(st["ra"])[idx]; dec = np.asarray(st["dec"])[idx]; g = np.asarray(st["gmag"])[idx]
    ids = gaia[idx]
    out = D / "nbr_parts"; out.mkdir(exist_ok=True)
    B = int(__import__("os").environ.get("NBR_BATCH", "4000"))
    for k in range(0, len(ids), B):
        f = out / f"nbr_{B}_{k // B:05d}.parquet"
        if f.exists():
            continue
        up = tap.id_table(ids[k:k + B], ra=ra[k:k + B], dec=dec[k:k + B], gmag=g[k:k + B])
        t, _ = tap.query_retry(tap.IRSA, NBR, upload=up, timeout=900)
        got = {int(x) for x in t["source_id"]}
        miss = [int(x) for x in ids[k:k + B] if int(x) not in got]  # no neighbour at all within 63"
        tab = tap.to_arrow(t)
        if miss:
            z = np.zeros(len(miss))
            tab = pa.concat_tables([tab.cast(pa.schema([(c, pa.float64() if c != "source_id" else pa.int64())
                                                         for c in tab.column_names])),
                                    pa.table({"source_id": pa.array(miss, pa.int64()), "f10": z, "f21": z, "f42": z,
                                              "f63": z, "n21": z, "dgmin21": z + 99, "dgmin63": z + 99, "n63": z})])
        else:
            tab = tab.cast(pa.schema([(c, pa.float64() if c != "source_id" else pa.int64()) for c in tab.column_names]))
        pq.write_table(tab, f.with_suffix(".tmp"))
        f.with_suffix(".tmp").rename(f)
        print(f"nbr batch {k // B}/{len(ids) // B}: {len(t)} with neighbours, {len(miss)} without", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "cols":
        cols(reverse=len(sys.argv) > 2 and sys.argv[2] == "reverse")
    else:
        nbr(int(sys.argv[2]))
