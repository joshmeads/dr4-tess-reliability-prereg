"""Test items from the catalogue snapshot: ephemerides, VSX matches, period-error calibration, mode.

1. Each survey signal takes its refined ephemeris (refine_ephem.py). Legacy signals with no
   stored curve keep the ledger values and are not testable.
2. Signals of one star and tier whose periods agree within 3 sigma are one item. Where two
   sectors give an unambiguous cycle count (sigma of the count < 0.1) the item's period is the
   epoch difference over the count ("combined"); the anchor epoch is the member sector nearest
   to Gaia's span.
3. VSX within 2 arcsec, period agreeing at P, P/2 or 2P within 1%: a catalogued match.
4. The catalogued eclipsing binaries are split in two halves by a sealed hash of the Gaia ID
   (SHA-256 of "sigma-split-20261202:<gaia_dr3>", even first byte = calibration). sigma_P is
   calibrated on the calibration half only, per tier, separately for single-sector and combined
   items: it is multiplied by the robust scatter (1.4826 x MAD about the median) of
   (P - h P_vsx)/sigma_P when that exceeds 1 (never shrunk). The held-out half is used only to
   test: positive controls and endpoint 5.
5. The empirical distribution of (P - h P_vsx)/sigma_P after scaling, on the calibration half,
   per group (box/ls x single/combined), is sealed as quantiles in drrel/period_errors.json; the
   power injections draw their period offsets from it.
6. The sealed mode: anchored (grid <= 512 periods at the farthest DR4 epoch), free (>= 41 disjoint
   control windows), or untestable.
Output: data/items.parquet, results/sigma_calibration.json, drrel/period_errors.json.
Needs DR4REL_SURVEY_STATE (for the cached VSX table).
"""
import hashlib
import json
import pathlib
import re
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work"))
from drrel import RULES  # noqa: E402
from drrel.controls import Ephemeris, testable  # noqa: E402
from drrel.nsl import DR4_END, DR4_START  # noqa: E402
from drrel.stat import choose_mode, grid  # noqa: E402
from paths import survey_state  # noqa: E402

D = ROOT / "data"
EB = re.compile(r"^(EA|EB|EW|E)([|:+/ ]|$)")
SPLIT_SALT = "sigma-split-20261202:"


def xyz(ra, dec):
    r, d = np.radians(ra), np.radians(dec)
    return np.c_[np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)]


def in_calibration_half(gaia):
    return hashlib.sha256((SPLIT_SALT + str(int(gaia))).encode()).digest()[0] % 2 == 0


def robust_scale(z):
    return 1.4826 * float(np.median(np.abs(z - np.median(z)))) if len(z) > 20 else np.nan


def main():
    s = pq.read_table(D / "signals.parquet").to_pydict()
    n = len(s["signal_id"])
    ref = {r["signal_id"]: r for r in pq.read_table(D / "ephem_refined.parquet").to_pylist()}
    P = np.array(s["period"], float); t0 = np.array(s["t0_bjd"], float)
    sp = np.full(n, np.nan); st0 = np.full(n, np.nan); amp = np.full(n, np.nan); snr = np.full(n, np.nan)
    refined = np.zeros(n, bool); edge = np.zeros(n, bool)
    for i, sid in enumerate(s["signal_id"]):
        r = ref.get(sid)
        if r and r.get("error") is None and r.get("P") and np.isfinite(r["sigma_p"]) and r["amp"] > 0:
            P[i], sp[i], t0[i], st0[i], amp[i], snr[i] = r["P"], r["sigma_p"], r["t0"], r["sigma_t0"], r["amp"], r["snr"]
            refined[i] = True; edge[i] = r["edge"]
    tier = np.array(s["tier"])
    dur = np.array([d if d is not None else 0.1 * p for d, p in zip(s["duration"], P)], float)
    dur = np.where(tier == "ls", 0.1 * P, dur)

    # VSX match
    v = pq.read_table(survey_state() / "cache" / "vsx_table.parquet", columns=["name", "ra", "dec", "type", "period"])
    vra, vde = v["ra"].to_numpy(zero_copy_only=False), v["dec"].to_numpy(zero_copy_only=False)
    ok = np.isfinite(vra) & np.isfinite(vde)
    vidx = np.nonzero(ok)[0]
    tree = cKDTree(xyz(vra[ok], vde[ok]))
    dd, jj = tree.query(xyz(np.array(s["ra"]), np.array(s["dec"])), distance_upper_bound=np.radians(2 / 3600))
    vtype = v["type"].to_pylist(); vper = v["period"].to_numpy(zero_copy_only=False); vname = v["name"].to_pylist()
    vsx_type = [None] * n; vsx_period = np.full(n, np.nan); vsx_name = [None] * n; vsx_h = np.full(n, np.nan)
    for i in np.nonzero(np.isfinite(dd))[0]:
        k = vidx[jj[i]]
        vsx_type[i], vsx_name[i] = vtype[k], vname[k]
        if np.isfinite(vper[k]):
            vsx_period[i] = vper[k]
            for h in (1.0, 0.5, 2.0):
                if abs(P[i] - h * vper[k]) / (h * vper[k]) < 0.01:
                    vsx_h[i] = h
                    break
    is_eb = np.isfinite(vsx_h) & np.array([bool(t and EB.match(t)) for t in vsx_type])
    cal_half = np.array([in_calibration_half(g) for g in s["gaia_dr3"]])

    # single-sector calibration, on the calibration half only
    calib = dict(split=dict(rule=f"sha256('{SPLIT_SALT}' + gaia_dr3)[0] even -> calibration half"))
    for tr in ("box", "ls"):
        m = refined & (tier == tr) & is_eb & cal_half & ~edge
        z = (P[m] - vsx_h[m] * vsx_period[m]) / sp[m]
        rs = robust_scale(z)
        k = float(max(1.0, rs)) if np.isfinite(rs) else 1.0
        calib[f"{tr}_single"] = dict(n=int(m.sum()), robust_std_before=rs, scale=k)
        sp[tier == tr] *= k
    sp = np.where(edge, sp * 3.0, sp)                      # a fit that hit the grid edge is only bounded

    # items
    gaia_mid = 0.5 * (DR4_START + DR4_END)
    by_key = {}
    for i in range(n):
        by_key.setdefault((s["star_id"][i], tier[i]), []).append(i)
    items, item_of = [], np.full(n, -1)
    for key, idx in by_key.items():
        idx = sorted(idx, key=lambda j: -(snr[j] if np.isfinite(snr[j]) else -1))
        for j in idx:
            if item_of[j] >= 0:
                continue
            grp = [j]; item_of[j] = len(items)
            for k2 in idx:
                if item_of[k2] >= 0:
                    continue
                tol = 3 * np.hypot(sp[j], sp[k2]) if np.isfinite(sp[j]) and np.isfinite(sp[k2]) else 0.01 * P[j]
                if abs(P[k2] - P[j]) <= max(tol, 1e-9):
                    grp.append(k2); item_of[k2] = len(items)
            items.append(grp)
    rows = []
    for it, grp in enumerate(items):
        j = grp[0]
        Pi, spi, t0i = P[j], sp[j], t0[j]
        combined = False
        if len(grp) > 1 and all(refined[k] for k in grp):
            a = min(grp, key=lambda k: t0[k]); b = max(grp, key=lambda k: t0[k])
            Pm = np.average([P[a], P[b]], weights=[1 / sp[a] ** 2, 1 / sp[b] ** 2])
            spm = 1 / np.sqrt(1 / sp[a] ** 2 + 1 / sp[b] ** 2)
            dt = t0[b] - t0[a]
            N = dt / Pm
            sN = np.hypot(st0[a], st0[b]) / Pm + dt * spm / Pm ** 2
            if sN < 0.1 and round(N) > 0:
                Pc = dt / round(N)
                sc = np.hypot(st0[a], st0[b]) / round(N)
                if abs(Pc - Pm) < 3 * np.hypot(spm, sc):
                    Pi, spi, combined = Pc, sc, True
            near = min(grp, key=lambda k: abs(t0[k] - gaia_mid))
            t0i = t0[near]
            if not combined:
                Pi, spi = P[near], sp[near]
            j = near
        rows.append(dict(j=j, grp=grp, P=Pi, sp=spi, t0=t0i, combined=combined))

    # combined-item calibration, on the calibration half only
    def item_z(r):
        j = r["j"]
        return (r["P"] - vsx_h[j] * vsx_period[j]) / r["sp"]
    for tr in ("box", "ls"):
        sel = [r for r in rows if r["combined"] and tier[r["j"]] == tr and is_eb[r["j"]] and cal_half[r["j"]]]
        rs = robust_scale(np.array([item_z(r) for r in sel]))
        k = float(max(1.0, rs)) if np.isfinite(rs) else 1.0
        calib[f"{tr}_combined"] = dict(n=len(sel), robust_std_before=rs, scale=k)
        for r in rows:
            if r["combined"] and tier[r["j"]] == tr:
                r["sp"] *= k

    # empirical period-error distributions (calibration half, after scaling) and their checks
    quantiles, levels = {}, np.linspace(0, 1, 1001)
    for tr in ("box", "ls"):
        for kind in ("single", "combined"):
            g = f"{tr}_{kind}"
            for half, name in ((True, "calibration"), (False, "held_out")):
                sel = [r for r in rows if r["combined"] == (kind == "combined") and tier[r["j"]] == tr
                       and is_eb[r["j"]] and refined[r["j"]] and cal_half[r["j"]] == half]
                z = np.array([item_z(r) for r in sel])
                calib[g][f"{name}_after"] = dict(n=len(z), robust_std=robust_scale(z),
                                                 frac_abs_z_gt_3=float(np.mean(np.abs(z) > 3)) if len(z) else None)
                if half:
                    quantiles[g] = np.quantile(z, levels).tolist() if len(z) >= 20 else None
    # a combined group with too few calibration objects falls back to its tier's single group
    for tr in ("box", "ls"):
        if quantiles[f"{tr}_combined"] is None:
            quantiles[f"{tr}_combined"] = quantiles[f"{tr}_single"]
    (ROOT / "drrel" / "period_errors.json").write_text(json.dumps(dict(
        description="Quantiles (levels 0..1 in 1001 steps) of (P_TESS - h P_VSX) / sigma_P after calibration, "
                    "calibration half of the catalogued eclipsing binaries, per tier and single/combined.",
        split=calib["split"], quantiles=quantiles)))

    out = {k: [] for k in ("item_id", "signal_id", "signal_ids", "star_id", "gaia_dr3", "ra", "dec", "gmag", "tier",
                           "period", "sigma_p", "t0", "duration", "amp", "snr", "n_signals", "n_sectors",
                           "combined", "refined", "source", "mode", "n_trials_anchored", "dt_max",
                           "vsx_name", "vsx_type", "vsx_period", "vsx_h", "vsx_eb", "sigma_cal", "vsx_holdout",
                           "err_group", "pool_status", "pool_flags", "track",
                           "eb_confidence", "eb_state", "eb_final", "var_confidence", "var_final", "eb_hyp_factor",
                           "odd_even_sigma", "secondary_sigma", "sine_half_p", "sine_p", "sector")}
    for it, r in enumerate(rows):
        j, Pi, spi, t0i = r["j"], r["P"], r["sp"], r["t0"]
        tt = "box" if tier[j] == "box" else "sine"
        duty = dur[j] / Pi if tier[j] == "box" else 0.1
        dt_max = max(abs(t0i - DR4_START), abs(t0i - DR4_END))
        mode, ntr = "untestable", np.nan
        if np.isfinite(spi) and spi > 0 and refined[j]:
            _, ntr = grid(Pi, spi, duty, tt, dt_max)
            m = choose_mode(tt, Pi, spi, duty, dt_max)
            eph = Ephemeris(tt, Pi, spi, t0i, dur[j], amp[j], mode=m)
            f = 1 / Pi
            wf = RULES["window_nsigma"] * spi / Pi ** 2
            lo, hi = RULES["control_ratio"]
            fits = RULES["min_controls_pivot"] * (2 * wf + 2 / RULES["gaia_span_days"]) <= f * (1 / lo - 1 / hi)
            mode = m if (fits and testable(eph)) else "untestable"
        grp = r["grp"]
        out["item_id"].append(it); out["signal_id"].append(s["signal_id"][j])
        out["signal_ids"].append(json.dumps([int(s["signal_id"][k]) for k in grp]))
        for k in ("star_id", "gaia_dr3", "ra", "dec", "gmag", "source", "pool_status", "pool_flags", "track",
                  "eb_confidence", "eb_state", "eb_final", "var_confidence", "var_final", "eb_hyp_factor", "sector"):
            out[k].append(s[k][j])
        out["tier"].append(tier[j]); out["period"].append(float(Pi)); out["sigma_p"].append(float(spi))
        out["t0"].append(float(t0i)); out["duration"].append(float(dur[j]))
        out["amp"].append(float(amp[j]) if np.isfinite(amp[j]) else
                          float((s["depth"][j] or 0) if tier[j] == "box" else (s["f_amplitude"][j] or 0) / 2))
        out["snr"].append(float(snr[j])); out["n_signals"].append(len(grp))
        out["n_sectors"].append(len({s["sector"][k] for k in grp}))
        out["combined"].append(r["combined"]); out["refined"].append(bool(refined[j])); out["mode"].append(mode)
        out["n_trials_anchored"].append(float(ntr)); out["dt_max"].append(float(dt_max))
        out["vsx_name"].append(vsx_name[j]); out["vsx_type"].append(vsx_type[j])
        out["vsx_period"].append(float(vsx_period[j])); out["vsx_h"].append(float(vsx_h[j]))
        eb_ok = bool(is_eb[j] and refined[j])
        out["vsx_eb"].append(eb_ok)
        out["sigma_cal"].append(eb_ok and bool(cal_half[j]))
        out["vsx_holdout"].append(eb_ok and not bool(cal_half[j]))
        out["err_group"].append(f"{tier[j]}_{'combined' if r['combined'] else 'single'}")
        out["odd_even_sigma"].append(s["f_odd_even_sigma"][j]); out["secondary_sigma"].append(s["f_secondary_sigma"][j])
        out["sine_half_p"].append(s["f_sine_half_p"][j]); out["sine_p"].append(s["f_sine_p"][j])
    pq.write_table(pa.table(out), D / "items.parquet", compression="zstd")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "sigma_calibration.json").write_text(json.dumps(calib, indent=1))
    modes = {}
    for t_, m_ in zip(out["tier"], out["mode"]):
        modes.setdefault(t_, {}).setdefault(m_, 0)
        modes[t_][m_] += 1
    print("signals", n, "items", len(rows), "combined", sum(out["combined"]))
    print("sigma calibration", json.dumps(calib, indent=1))
    print("modes by tier", json.dumps(modes))


if __name__ == "__main__":
    main()
