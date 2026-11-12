"""Kepler calibration labels, streamed and reduced (user-approved, 2026-11-11).

For each sampled signal on a Kepler target: fetch two long-cadence quarters from the public
stpubdata bucket into memory, condition them as the TESS search did (sliding median +-0.75 d,
positive clip at 8 MAD), and run drrel's same-star test with a free phase at the TESS period
window. Keep only the numbers; the raw bytes are never written to disk.

Label real_on = 1 when Kepler shows the signal at p <= 0.01 with an amplitude at least 0.3 of the
TESS amplitude. Kepler's 4" pixels make that "on this star (or within a few arcsec)".
Labelling rules (wider control range, 100 controls) differ from the sealed Gaia test on purpose:
the TESS window is often too wide for 200 disjoint controls in 0.6-1.7 P.

The frozen labels (data/inputs/kepler_labels.parquet) were computed on 2026-11-11 with drrel 1.0.0,
whose labelling p-values used the rank test at >= 95 valid controls and the Gumbel pivot at 40-94.
Known limitation of those labels: the rank test's smallest p is 1/(n + 1), above alpha = 0.01 at
95-98 valid controls, so the 137 labelled rows there could never be detected; 7 of them sat at that
floor, and one (a box item with Kepler amplitude 0.084 against a threshold of 0.022) would
otherwise have been real_on = 1. The labels are frozen inputs and are not recomputed. Rerunning this
script with the current code reproduces them except at exactly 40 controls and where the v1.1.1
pivot guard (zero-inflated controls) applies.

Safety: free disk is checked before each batch and the run stops under 40 GB; at most three
downloads run at once; downloaded bytes are parsed by astropy only (run with python -I).
usage: kepler_calib.py PLAN.parquet OUT.parquet
"""
import io
import json
import pathlib
import re
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from scipy.ndimage import median_filter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drrel import RULES  # noqa: E402
from drrel.controls import Ephemeris, test  # noqa: E402
from drrel.stat import best_epoch, odd_even  # noqa: E402

S3 = "https://stpubdata.s3.amazonaws.com/"
UA = {"User-Agent": "dr4-reliability/1.0 (python-urllib)"}
MIN_FREE_GB = 40.0
MAX_BYTES = 3 * 1024 ** 2
LABEL_RULES = {**RULES, "n_control": 100, "n_rank_min": 95, "control_ratio": (0.4, 2.5),
               "alpha": 0.01, "n_inject": 20, "gaia_span_days": 1400.0}
KEPLER_BJD0 = 2454833.0


def get(url, limit=MAX_BYTES):
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                b = r.read(limit + 1)
            if len(b) > limit:
                raise ValueError("file larger than expected")
            return b
        except (OSError, ValueError) as e:  # noqa: PERF203
            if k == 3:
                raise
            time.sleep(10 * (k + 1))
            last = e  # noqa: F841
    return None


def quarters(kic):
    url = f"{S3}?list-type=2&prefix=kepler/public/lightcurves/{kic // 100000:04d}/{kic:09d}/"
    body = get(url, 200_000).decode()
    keys = sorted(k for k in re.findall(r"<Key>([^<]+)</Key>", body) if k.endswith("_llc.fits"))
    if len(keys) < 4:
        return keys
    inner = keys[2:-1] if len(keys) > 6 else keys
    return [inner[int(round(q * (len(inner) - 1)))] for q in (0.3, 0.7)]


def condition(t, f):
    """The TESS 'long' view on a Kepler quarter: median filter over +-0.75 d (73 cadences)."""
    f = f / np.median(f)
    trend = median_filter(f, size=73, mode="nearest")
    y = f / trend
    mad = 1.4826 * np.median(np.abs(y - np.median(y)))
    keep = (y < 1 + 8 * max(mad, 1e-5)) & np.isfinite(y)
    return t[keep], y[keep]


def light_curve(kic):
    keys = quarters(kic)
    if not keys:
        return None
    ts, ys, es = [], [], []
    for k in keys:
        raw = get(S3 + k)
        with fits.open(io.BytesIO(raw), memmap=False) as h:
            if int(h[0].header.get("KEPLERID", -1)) != kic:
                raise ValueError(f"header KEPLERID mismatch for {kic}")
            d = h[1].data
            t = np.asarray(d["TIME"], float); f = np.asarray(d["PDCSAP_FLUX"], float)
            e = np.asarray(d["PDCSAP_FLUX_ERR"], float); q = np.asarray(d["SAP_QUALITY"])
        del raw
        ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (q == 0) & (f > 0)
        if ok.sum() < 500:
            continue
        t, f, e = t[ok], f[ok], e[ok]
        e = e / np.median(f)
        tc, yc = condition(t, f)
        ts.append(tc + KEPLER_BJD0); ys.append(yc - 1.0)
        es.append(np.interp(tc, t, e))
    if not ts:
        return None
    return np.concatenate(ts), np.concatenate(ys), np.concatenate(es), len(keys)


def label(row, lc):
    t, x, s, nq = lc
    eph = Ephemeris(tier=row["tier"], period=row["period"], sigma_p=row["sigma_p"], t0=row["t0"],
                    duration=row["duration"], amp=row["amp"], mode="free")
    r = test(t, x, s, eph, rules=LABEL_RULES, with_power=True)
    out = dict(row_id=row["row_id"], status=r["status"], p=r["p"], amp_kep=r["amp"], power=r["power"],
               p_best=r["p_best"], n_eff=r["n_eff"], n_valid=r["n_valid"], n_points=len(t), n_quarters=nq)
    out["real_on"] = (bool(r["status"] == "OK" and r["p"] <= LABEL_RULES["alpha"]
                           and r["amp"] >= 0.3 * row["amp"]) if r["status"] == "OK" else None)
    out["detected_any_amp"] = bool(r["status"] == "OK" and r["p"] <= LABEL_RULES["alpha"])
    if out["detected_any_amp"]:
        te = best_epoch(eph.tier, t, x, s, r["p_best"], eph.duration)
        d, sd, a = odd_even(eph.tier, t, x, s, r["p_best"], te, eph.duration)
        out.update(p2p_diff=d, p2p_sigma=sd, p2p_amp=a)
    return out


def main():
    plan = pq.read_table(sys.argv[1]).to_pylist()
    out_path = pathlib.Path(sys.argv[2])
    done = {}
    jl = out_path.with_suffix(".jsonl")
    if jl.exists():
        for line in jl.read_text().splitlines():
            r = json.loads(line); done[r["row_id"]] = r
    by_kic = {}
    for r in plan:
        if r["row_id"] not in done:
            by_kic.setdefault(int(r["kic"]), []).append(r)
    kics = sorted(by_kic)
    print("stars to do:", len(kics), "rows:", sum(len(v) for v in by_kic.values()), flush=True)
    B = 30
    with open(jl, "a") as fh, ThreadPoolExecutor(3) as pool:
        for k in range(0, len(kics), B):
            free = shutil.disk_usage(ROOT).free / 1e9
            if free < MIN_FREE_GB:
                print(f"STOP: {free:.1f} GB free, under {MIN_FREE_GB} GB", flush=True)
                break
            batch = kics[k:k + B]
            lcs = list(pool.map(lambda kk: _safe(light_curve, kk), batch))
            for kic, lc in zip(batch, lcs):
                for row in by_kic[kic]:
                    if isinstance(lc, Exception) or lc is None:
                        res = dict(row_id=row["row_id"], status="NO_DATA", error=repr(lc)[:200] if lc else None)
                    else:
                        res = label(row, lc)
                    fh.write(json.dumps({k2: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                         for k2, v in res.items()}) + "\n")
                del lc
            fh.flush()
            print(f"  {min(k + B, len(kics))}/{len(kics)} stars, {free:.0f} GB free", flush=True)
    rows = [json.loads(line) for line in jl.read_text().splitlines()]
    keys = sorted({k for r in rows for k in r})
    pq.write_table(pa.table({k: [r.get(k) for r in rows] for k in keys}), out_path)
    print("labels written:", len(rows))


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001 - one star's failure is recorded, not fatal
        return e


if __name__ == "__main__":
    main()
