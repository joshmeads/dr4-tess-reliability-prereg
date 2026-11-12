"""Refine every survey signal's ephemeris from its stored TESS light curve.

box: profile the box SNR over a fine (P, t0) grid at the signal's duration.
ls : weighted sinusoid fit (free phase) over a fine frequency grid.
The 1-sigma period interval is where the chi-square improvement drops by beta^2 from its
maximum (beta = red-noise factor of the residuals at 2 h). The epoch is moved to the event
nearest the light curve's middle, where period and epoch errors decouple.

Reads the ledger (mode=ro) for lc_ref -> part file, and the store's Parquet files read-only.
Output: data/ephem_refined.parquet. Usage: refine_ephem.py [WORKERS]
"""
import pathlib
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from paths import survey_state  # noqa: E402

STATE = str(survey_state())    # the survey state directory, from DR4REL_SURVEY_STATE
D = pathlib.Path(__file__).resolve().parents[1] / "data"


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def red_beta(t, r, bin_days=2 / 24):
    k = np.floor((t - t[0]) / bin_days).astype(int)
    cnt = np.bincount(k)
    sums = np.bincount(k, r)
    m = cnt >= 3
    if m.sum() < 10:
        return 1.0
    means = sums[m] / cnt[m]
    expect = np.std(r) / np.sqrt(np.mean(cnt[m]))
    return float(max(1.0, np.std(means) / expect)) if expect > 0 else 1.0


def interval(x, chi, drop):
    """Half-width of the region around the maximum of chi where chi >= max - drop."""
    k = int(np.argmax(chi))
    thr = chi[k] - drop
    lo = k
    while lo > 0 and chi[lo - 1] >= thr:
        lo -= 1
    hi = k
    while hi < len(chi) - 1 and chi[hi + 1] >= thr:
        hi += 1
    edge = lo == 0 or hi == len(chi) - 1
    xl = x[lo] if lo == 0 else np.interp(thr, [chi[lo - 1], chi[lo]], [x[lo - 1], x[lo]])
    xh = x[hi] if hi == len(chi) - 1 else np.interp(thr, [chi[hi + 1], chi[hi]], [x[hi + 1], x[hi]])
    step = x[1] - x[0]
    return max((xh - xl) / 2, step / 2), k, edge


def refine_box(t, y, P0, t0, dur):
    sw = max(mad(np.diff(y)) / np.sqrt(2), 1e-5)
    tm = float(np.median(t))
    t0m = t0 + np.round((tm - t0) / P0) * P0
    Ps = P0 * (1 + np.linspace(-5e-4, 5e-4, 31))
    dts = np.linspace(-dur / 2, dur / 2, 21)
    best = np.empty((len(Ps), len(dts)))
    for i, P in enumerate(Ps):
        ph = ((t[None, :] - (t0m + dts[:, None])) / P + 0.5) % 1 - 0.5
        inside = np.abs(ph) * P < dur / 2
        n_in = inside.sum(1)
        n_out = len(t) - n_in
        s_in = inside @ y
        with np.errstate(invalid="ignore", divide="ignore"):
            depth = (y.sum() - s_in) / n_out - s_in / n_in
            best[i] = np.where(n_in >= 3, depth / (sw * np.sqrt(1 / n_in + 1 / n_out)), -np.inf)
    i, j = np.unravel_index(np.argmax(best), best.shape)
    P, dt = Ps[i], dts[j]
    ph = ((t - (t0m + dt)) / P + 0.5) % 1 - 0.5
    inside = np.abs(ph) * P < dur / 2
    depth = float(y[~inside].mean() - y[inside].mean())
    r = y - np.where(inside, -depth, 0.0) - y[~inside].mean()
    beta = red_beta(t, r)
    chi_p = np.max(best, axis=1) ** 2
    sig_p, _, edge_p = interval(Ps, chi_p, beta ** 2)
    chi_t = best[i] ** 2
    sig_t, _, edge_t = interval(dts, chi_t, beta ** 2)
    return dict(P=float(P), sigma_p=float(sig_p), t0=float(t0m + dt), sigma_t0=float(sig_t),
                snr=float(best[i, j]), amp=depth, beta=beta, edge=bool(edge_p or edge_t),
                n=len(t), span=float(t[-1] - t[0]))


def refine_sine(t, y, P0):
    sw = max(mad(np.diff(y)) / np.sqrt(2), 1e-5)
    T = t[-1] - t[0]
    f0 = 1 / P0
    fs = f0 + np.linspace(-1.5, 1.5, 61) / (10 * T)
    tm = float(np.median(t))
    tt = t - tm
    yc = y - y.mean()
    chi = np.empty(len(fs)); coef = np.empty((len(fs), 2))
    for k, f in enumerate(fs):
        X = np.column_stack([np.ones_like(tt), np.sin(2 * np.pi * f * tt), np.cos(2 * np.pi * f * tt)])
        c, *_ = np.linalg.lstsq(X, yc, rcond=None)
        res = yc - X @ c
        chi[k] = (np.sum(yc ** 2) - np.sum(res ** 2)) / sw ** 2
        coef[k] = c[1:]
    k = int(np.argmax(chi))
    f = fs[k]; a, b = coef[k]
    R = float(np.hypot(a, b))
    X = np.column_stack([np.ones_like(tt), np.sin(2 * np.pi * f * tt), np.cos(2 * np.pi * f * tt)])
    c, *_ = np.linalg.lstsq(X, yc, rcond=None)
    beta = red_beta(t, yc - X @ c)
    sig_f, _, edge = interval(fs, chi, beta ** 2)
    # a sin(wt) + b cos(wt) = R cos(wt - phi), phi = atan2(a, b); minimum where wt - phi = pi
    phi = np.arctan2(a, b)
    t_min = tm + (phi + np.pi) / (2 * np.pi * f)
    t_min -= np.round((t_min - tm) * f) / f
    sig_phase = beta * sw / (R * np.sqrt(len(t) / 2)) if R > 0 else np.inf
    return dict(P=float(1 / f), sigma_p=float(sig_f / f ** 2), t0=float(t_min),
                sigma_t0=float(sig_phase / (2 * np.pi * f)), snr=float(np.sqrt(max(chi[k], 0)) / beta),
                amp=R, beta=beta, edge=bool(edge), n=len(t), span=float(T))


def work(args):
    path, rows = args
    tab = pq.read_table(f"{STATE}/store/{path}", columns=["time", "flux"])
    out = []
    for sid, rowi, tier, P0, t0, dur in rows:
        try:
            r = tab.slice(rowi, 1).to_pylist()[0]
            t = np.asarray(r["time"], float); y = np.asarray(r["flux"], float)
            ok = np.isfinite(t) & np.isfinite(y)
            t, y = t[ok], y[ok]
            o = np.argsort(t); t, y = t[o], y[o]
            res = refine_box(t, y, P0, t0, dur) if tier == "box" else refine_sine(t, y, P0)
            res["signal_id"] = sid
            out.append(res)
        except Exception as e:  # noqa: BLE001 - one bad curve must not stop the pass
            out.append(dict(signal_id=sid, error=repr(e)[:200]))
    return out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sig = pq.read_table(D / "signals.parquet", columns=["signal_id", "tier", "period", "t0_bjd", "duration",
                                                         "lc_ref", "source"]).to_pydict()
    con = sqlite3.connect(f"file:{STATE}/survey.db?mode=ro", uri=True)
    pin = {r[0]: (r[1], r[2]) for r in con.execute("SELECT lc_ref, part_id, row FROM lc_pin")}
    ppath = dict(con.execute("SELECT part_id, path FROM part WHERE kind='lc'"))
    jobs = {}
    for sid, tier, P, t0, dur, ref, src in zip(*[sig[k] for k in ("signal_id", "tier", "period", "t0_bjd",
                                                                    "duration", "lc_ref", "source")]):
        if ref is None or ref not in pin:
            continue
        part, rowi = pin[ref]
        if part not in ppath:
            continue
        jobs.setdefault(ppath[part], []).append((sid, rowi, tier, P, t0, dur))
    print("signals with a stored curve:", sum(len(v) for v in jobs.values()), "files:", len(jobs), flush=True)
    if len(sys.argv) > 2 and sys.argv[2] == "test":
        for path in list(jobs)[:2]:
            for r in work((path, jobs[path][:6])):
                print(r)
        return
    results = []
    with ProcessPoolExecutor(workers) as ex:
        for k, res in enumerate(ex.map(work, list(jobs.items()), chunksize=4)):
            results.extend(res)
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)} files, {len(results)} signals", flush=True)
    keys = ("signal_id", "P", "sigma_p", "t0", "sigma_t0", "snr", "amp", "beta", "edge", "n", "span", "error")
    cols = {k: [r.get(k) for r in results] for k in keys}
    pq.write_table(pa.table(cols), D / "ephem_refined.parquet", compression="zstd")
    print("done:", len(results), "errors:", sum(r.get("error") is not None for r in results))


if __name__ == "__main__":
    main()
