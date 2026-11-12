"""Pre-DR4 purity per tier from Gaia DR3 G-band scatter (declared in the seal as information used).

Per star, the per-CCD fractional scatter A = sqrt(phot_g_n_obs) / phot_g_mean_flux_over_error
(the amplitude proxy of Mowlavi et al. 2021). Quiet comparison stars (searched, no peak at all)
give the reference distribution Q of log A in strata of G (0.1 mag) and resolved-companion flag
(ipd_frac_multi_peak > 0). If an item's signal is real and on the star, its A is the quadrature
sum of a quiet A and the signal's rms, rms = depth sqrt(q (1 - q)) for a box of duty q and
A / sqrt(2) for a sinusoid of semi-amplitude A; that gives V_i. Each tier is fitted as
(1 - f) Q + f V by maximum likelihood; f is the DR3 estimate of "real and on the star".
Per item: the likelihood ratio log(V_i / Q_i) and the posterior at its tier's f.

Validation inside the same run: catalogued (VSX) eclipsing binaries must come out near f = 1,
and a random set of quiet stars given borrowed signals near f = 0.
Output: data/dr3_item_lr.parquet, results/purity_dr3.json.
"""
import json
import pathlib
import re
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "data"
EB = re.compile(r"^(EA|EB|EW|E)([|:+/ ]|$)")
BW = 0.08          # KDE bandwidth in log10 A
N_BOOT = 300


def load_dr3():
    parts = sorted((D / "dr3_parts").glob("cols_*.parquet"))
    t = pa.concat_tables([pq.read_table(p) for p in parts], promote_options="default").to_pydict()
    return {int(s): i for i, s in enumerate(t["source_id"])}, t


def stratum(g, multi):
    return np.clip(((g - 13.4) / 0.1).astype(int), 0, 25) * 2 + (multi > 0).astype(int)


def kde_logpdf(points, ref):
    """log density of points under a Gaussian KDE of ref (1-D), evaluated in chunks."""
    out = np.empty(len(points))
    for k in range(0, len(points), 2000):
        d = (points[k:k + 2000, None] - ref[None, :]) / BW
        out[k:k + 2000] = np.log(np.mean(np.exp(-0.5 * d * d), axis=1) / (BW * np.sqrt(2 * np.pi)) + 1e-300)
    return out


def item_densities(logA, rms, strata, ref_by_stratum, rng):
    """log Q_i and log V_i per item."""
    lq = np.full(len(logA), np.nan); lv = np.full(len(logA), np.nan)
    for st in np.unique(strata):
        m = np.nonzero(strata == st)[0]
        ref = ref_by_stratum.get(st)
        if ref is None or len(ref) < 200:
            continue
        lq[m] = kde_logpdf(logA[m], ref)
        sub = ref if len(ref) <= 1500 else rng.choice(ref, 1500, replace=False)
        Aq = 10 ** sub
        for i in m:
            v = np.log10(np.sqrt(Aq ** 2 + rms[i] ** 2))
            d = (logA[i] - v) / BW
            lv[i] = np.log(np.mean(np.exp(-0.5 * d * d)) / (BW * np.sqrt(2 * np.pi)) + 1e-300)
    return lq, lv


def fit_f(lq, lv):
    ok = np.isfinite(lq) & np.isfinite(lv)
    lq, lv = lq[ok], lv[ok]
    if len(lq) < 20:
        return np.nan
    fs = np.linspace(0, 1, 1001)
    m = np.maximum(lq, lv)
    ll = [np.sum(m + np.log((1 - f) * np.exp(lq - m) + f * np.exp(lv - m))) for f in fs]
    return float(fs[int(np.argmax(ll))])


def fit_with_ci(lq, lv, rng):
    f = fit_f(lq, lv)
    ok = np.isfinite(lq) & np.isfinite(lv)
    idx = np.nonzero(ok)[0]
    if len(idx) < 20:
        return dict(f=f, ci95=[np.nan, np.nan], n=int(len(idx)))
    bs = [fit_f(lq[b], lv[b]) for b in (rng.choice(idx, len(idx)) for _ in range(N_BOOT))]
    return dict(f=f, ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=int(len(idx)))


def main():
    rng = np.random.default_rng(20261202)
    row_of, dr3 = load_dr3()
    g_n = np.asarray(dr3["phot_g_n_obs"], float); foe = np.asarray(dr3["phot_g_mean_flux_over_error"], float)
    gmag = np.asarray(dr3["phot_g_mean_mag"], float); multi = np.asarray(dr3["ipd_frac_multi_peak"], float)
    logA_all = np.log10(np.sqrt(g_n) / foe)
    st_all = stratum(gmag, np.nan_to_num(multi))
    quiet = pq.read_table(D / "quiet_stars.parquet", columns=["gaia_dr3"])["gaia_dr3"].to_pylist()
    qrows = np.array([row_of[q] for q in quiet if q in row_of])
    qrows = qrows[np.isfinite(logA_all[qrows])]
    ref = {st: logA_all[qrows[st_all[qrows] == st]] for st in np.unique(st_all[qrows])}
    print("quiet reference stars:", len(qrows), flush=True)

    it = pq.read_table(D / "items.parquet").to_pydict()
    n = len(it["item_id"])
    rows = np.array([row_of.get(int(g), -1) for g in it["gaia_dr3"]])
    have = rows >= 0
    logA = np.where(have, logA_all[np.clip(rows, 0, None)], np.nan)
    strata = np.where(have, st_all[np.clip(rows, 0, None)], -1)
    amp = np.asarray(it["amp"], float); P = np.asarray(it["period"], float); dur = np.asarray(it["duration"], float)
    tier = np.array(it["tier"])
    q = np.clip(dur / P, 1e-3, 0.5)
    rms = np.where(tier == "box", np.abs(amp) * np.sqrt(q * (1 - q)), np.abs(amp) / np.sqrt(2))
    lq, lv = item_densities(logA, rms, strata, ref, rng)
    print("densities done", flush=True)

    # quiet stars given borrowed signals: f must come out near 0
    qs = rng.choice(qrows, size=min(20000, len(qrows)), replace=False)
    donors = rng.choice(np.nonzero(np.isfinite(lv))[0], size=len(qs))
    lq_n, lv_n = item_densities(logA_all[qs], rms[donors], st_all[qs], ref, rng)

    tiers = {
        "box (all)": tier == "box",
        "ls (all)": tier == "ls",
        "box, pool released": (tier == "box") & (np.array(it["pool_status"]) == "RELEASED"),
        "box, pool vetoed": (tier == "box") & (np.array(it["pool_status"]) == "COMMON_MODE"),
        "ls, pool released": (tier == "ls") & (np.array(it["pool_status"]) == "RELEASED"),
        "ls, pool vetoed": (tier == "ls") & (np.array(it["pool_status"]) == "COMMON_MODE"),
        "clean eb tier": np.array([e is not None for e in it["eb_state"]]) &
                         (np.array(it["pool_status"]) == "RELEASED") &
                         np.array(["cm_suspect" not in (f or "") for f in it["pool_flags"]]),
        "variable queue": np.array([v is not None for v in it["var_confidence"]]),
        "alias_window track": np.array(it["track"]) == "alias_window",
        "off_window track": np.array(it["track"]) == "off_window",
    }
    vsx_eb = np.array([bool(t and EB.match(t)) for t in it["vsx_type"]]) & np.isfinite(np.asarray(it["vsx_h"], float))
    out = {"method": __doc__.strip().splitlines()[0], "n_quiet_reference": int(len(qrows)), "tiers": {},
           "validation": {"vsx_eclipsing_binaries": fit_with_ci(lq[vsx_eb], lv[vsx_eb], rng),
                          "quiet_with_borrowed_signals": fit_with_ci(lq_n, lv_n, rng)}}
    for name, m in tiers.items():
        out["tiers"][name] = fit_with_ci(lq[m], lv[m], rng)
        print(name, out["tiers"][name], flush=True)
    for name in ("box (all)", "ls (all)"):
        m = tiers[name]
        for lo, hi in ((0.01, 0.02), (0.02, 0.05), (0.05, 0.16)):
            amp2 = np.where(tier == "box", amp, 2 * amp)          # box depth, or sine peak-to-peak
            mm = m & (amp2 >= lo) & (amp2 < hi)
            out["tiers"][f"{name}, amplitude {lo}-{hi}"] = fit_with_ci(lq[mm], lv[mm], rng)
    print("validation", out["validation"])
    # per-item log LR and posterior at the tier's own f
    f_tier = np.where(tier == "box", out["tiers"]["box (all)"]["f"], out["tiers"]["ls (all)"]["f"])
    with np.errstate(invalid="ignore", over="ignore"):
        lr = lv - lq
        post = f_tier * np.exp(lr) / (1 - f_tier + f_tier * np.exp(lr))
    pq.write_table(pa.table({"item_id": it["item_id"], "dr3_log_lr": lr, "dr3_post": post, "dr3_logA": logA,
                             "dr3_has": have}), D / "dr3_item_lr.parquet")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "purity_dr3.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
