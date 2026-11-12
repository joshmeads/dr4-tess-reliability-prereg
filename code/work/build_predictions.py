"""Sealed predictions: per-item probability, calls, primary sample, controls, endpoint predictions.

Per item: p_pred = P(real and on the named star), from a logistic model fitted to Kepler labels
(kepler_calib.py) on features known before DR4: tier, amplitude, refined SNR, G, period, pool
status and flags, alias proximity, number of sectors, a resolved-companion flag, and the DR3
excess-scatter log likelihood ratio (purity_dr3.py).
call_unequal: alternate minima unequal at 2P (TESS odd/even >= 3 sigma), the sealed P-versus-2P call.
Primary sample (seeded): every testable box item, every testable item routed to the eb or variable
queue, and 30,000 other testable LS items. Shuffled controls: for a seeded quarter of primary items (at most 15,000), a quiet star
matched in G, ecliptic latitude, Galactic latitude and companion flag, given that item's ephemeris.
Positive controls: testable items matching a catalogued (VSX) eclipsing binary in the held-out half
(build_items.py), each also scored on 50% and 30% of its Gaia epochs.
Coarse power (planning only): simplified scanning law, DR3 transit count x 66/34, smooth G errors,
on 3,000 primary items, carried to the rest by tier/mode/(amplitude / noise) bins.
Outputs: data/predictions.parquet, data/items_sealed.parquet, results/endpoint_predictions.json,
results/sample_summary.json, results/kepler_calibration.json.
"""
import json
import pathlib
import re
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drrel import RULES, nsl  # noqa: E402
from drrel.controls import Ephemeris, test  # noqa: E402
from drrel.endpoints import G_BINS  # noqa: E402

sys.path.insert(0, str(ROOT / "work"))
from kepmap import plan_to_items  # noqa: E402

D = ROOT / "data"
EB = re.compile(r"^(EA|EB|EW|E)([|:+/ ]|$)")
SEED = 20261202
POS_SUBSAMPLE = (0.5, 0.3)
ALIASES = (1 / 3, 0.5, 2 / 3, 1.0, 2.0, 3.0)


def ecl_gal(ra, dec):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c = SkyCoord(ra=np.asarray(ra) * u.deg, dec=np.asarray(dec) * u.deg)
    return c.barycentrictrueecliptic.lat.deg, c.galactic.b.deg


def features(it, lr):
    tier = np.array(it["tier"])
    P = np.asarray(it["period"], float)
    amp = np.abs(np.asarray(it["amp"], float))
    snr = np.asarray(it["snr"], float)
    pool = np.array(it["pool_status"], dtype=object)
    flags = np.array([f or "" for f in it["pool_flags"]])
    alias = np.min([np.abs(P / a - 1) for a in ALIASES], axis=0)
    X = np.column_stack([
        (tier == "box").astype(float),
        np.log10(np.clip(np.where(tier == "box", amp, 2 * amp), 1e-4, 1)),
        np.log10(np.clip(np.nan_to_num(snr, nan=3.0), 1, 1e3)),
        np.asarray(it["gmag"], float) - 14.5,
        np.log10(P),
        (pool == "RELEASED").astype(float),
        (pool == "COMMON_MODE").astype(float),
        np.array(["cm_suspect" in f for f in flags], float),
        (alias < 0.03).astype(float),
        (np.asarray(it["n_sectors"]) > 1).astype(float),
        np.clip(np.nan_to_num(lr["dr3_log_lr"], nan=0.0), -10, 10),
        np.asarray(lr["multi"], float),
    ])
    names = ["box", "log_amp", "log_snr", "g", "log_p", "released", "vetoed", "cm_suspect", "near_alias",
             "multi_sector", "dr3_log_lr", "resolved_companion"]
    return X, names


def logistic(X, y, lam=1.0, iters=50):
    Xb = np.column_stack([np.ones(len(X)), X])
    mu, sd = Xb[:, 1:].mean(0), Xb[:, 1:].std(0) + 1e-9
    Z = Xb.copy(); Z[:, 1:] = (Xb[:, 1:] - mu) / sd
    w = np.zeros(Z.shape[1])
    R = lam * np.eye(Z.shape[1]); R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + R
        g = Z.T @ (p - y) + R @ w
        w -= np.linalg.solve(H, g)
    return dict(w=w, mu=mu, sd=sd)


def predict(m, X):
    Z = np.column_stack([np.ones(len(X)), (X - m["mu"]) / m["sd"]])
    return 1 / (1 + np.exp(-Z @ m["w"]))


def coarse_power(rows, dr3, rng):
    out = []
    for r in rows:
        d = dr3.get(int(r["gaia_dr3"]))
        n_tr = int(round((d["matched_transits"] if d and d["matched_transits"] > 0 else 40) * 66 / 34))
        t = nsl.transit_times(r["ra"], r["dec"], loss=0.0, seed=int(r["item_id"]))
        if len(t) > n_tr:
            t = np.sort(rng.choice(t, n_tr, replace=False))
        s = np.full(len(t), nsl.sigma_g(r["gmag"]))
        x = rng.normal(0, s)
        eph = Ephemeris("box" if r["tier"] == "box" else "sine", r["period"], r["sigma_p"], r["t0"], r["duration"],
                        abs(r["amp"]), mode=r["mode"], err_group=r["err_group"])
        res = test(t, x, s, eph)
        out.append(res["power"] if res["status"] in ("OK", "NO_COVERAGE") else np.nan)
    return np.array(out, float)


def main():
    rng = np.random.default_rng(SEED)
    it = pq.read_table(D / "items.parquet").to_pydict()
    n = len(it["item_id"])
    lrt = pq.read_table(D / "dr3_item_lr.parquet").to_pydict()
    assert lrt["item_id"] == it["item_id"]
    parts = sorted((D / "dr3_parts").glob("cols_*.parquet"))
    dt = pa.concat_tables([pq.read_table(p) for p in parts], promote_options="default").to_pylist()
    dr3 = {int(r["source_id"]): r for r in dt}
    multi = np.array([(dr3.get(int(g)) or {}).get("ipd_frac_multi_peak", 0) or 0 for g in it["gaia_dr3"]]) > 0
    lr = dict(dr3_log_lr=np.asarray(lrt["dr3_log_lr"], float), multi=multi)
    X, names = features(it, lr)

    # Kepler labels -> model
    kp = pq.read_table(D / "kepler_plan.parquet").to_pydict()
    if (D / "kepler_labels.parquet").exists() and "partial" not in sys.argv:
        kl = {r["row_id"]: r for r in pq.read_table(D / "kepler_labels.parquet").to_pylist()}
    else:                                   # dry run on labels written so far
        kl = {r["row_id"]: r for r in map(json.loads, (D / "kepler_labels.jsonl").read_text().splitlines())}
    idx_of = {iid: i for i, iid in enumerate(it["item_id"])}
    plan_idx = plan_to_items(kp, it)          # by star, tier and period: item ids change on rebuilds
    lab_i, lab_y, kep_items = [], [], set()
    shuf, unmatched = [], 0
    for rid, kind, ii in zip(kp["row_id"], kp["kind"], plan_idx):
        r = kl.get(rid)
        if r is None or r.get("real_on") is None:
            continue
        if kind == "shuffled":
            shuf.append(bool(r["real_on"]))
            continue
        if ii is None:
            unmatched += 1
            continue
        lab_i.append(ii); lab_y.append(float(r["real_on"])); kep_items.add(it["item_id"][ii])
    lab_i, lab_y = np.array(lab_i), np.array(lab_y)
    k_false = float(np.mean(shuf)) if shuf else np.nan
    # 5-fold cross validation
    folds = rng.integers(0, 5, len(lab_i))
    cv = np.empty(len(lab_i))
    for k in range(5):
        m = logistic(X[lab_i[folds != k]], lab_y[folds != k])
        cv[folds == k] = predict(m, X[lab_i[folds == k]])
    model = logistic(X[lab_i], lab_y)
    p_pred = predict(model, X)
    brier_cv = float(np.mean((cv - lab_y) ** 2)); brier_clim = float(np.mean((lab_y.mean() - lab_y) ** 2))
    rel = []
    for lo in np.arange(0, 1, 0.1):
        mm = (cv >= lo) & (cv < lo + 0.1)
        rel.append(dict(bin=[round(lo, 1), round(lo + 0.1, 1)], n=int(mm.sum()),
                        forecast=float(cv[mm].mean()) if mm.any() else None,
                        observed=float(lab_y[mm].mean()) if mm.any() else None))
    tier = np.array(it["tier"])
    kep_summary = dict(
        n_labelled=int(len(lab_y)), n_real_on=int(lab_y.sum()), n_label_rows_without_item=unmatched,
        real_on_fraction={t: float(lab_y[tier[lab_i] == t].mean()) if (tier[lab_i] == t).any() else None
                          for t in ("box", "ls")},
        false_label_rate_quiet_stars=k_false, n_quiet_controls=len(shuf),
        cv_brier=brier_cv, climatology_brier=brier_clim, cv_brier_skill=1 - brier_cv / brier_clim,
        cv_reliability=rel, coefficients=dict(zip(["intercept"] + names, map(float, model["w"]))),
        standardisation=dict(mu=dict(zip(names, map(float, model["mu"]))), sd=dict(zip(names, map(float, model["sd"])))))

    # calls and flags
    oe = np.array([v if v is not None else np.nan for v in it["odd_even_sigma"]], float)
    call_unequal = np.nan_to_num(oe, nan=0) >= 3.0
    mode = np.array(it["mode"])
    testable = mode != "untestable"
    routed = np.array([(e is not None) or (v is not None) for e, v in zip(it["eb_state"], it["var_confidence"])])
    clean_eb = (np.array([e is not None for e in it["eb_state"]]) & (np.array(it["pool_status"]) == "RELEASED")
                & np.array(["cm_suspect" not in (f or "") for f in it["pool_flags"]]))
    refined = np.asarray(it["refined"])
    base = testable & refined
    primary = base & ((tier == "box") | routed)
    others = np.nonzero(base & ~primary)[0]
    primary[rng.choice(others, size=min(30000, len(others)), replace=False)] = True
    kepler_cal = np.array([iid in kep_items for iid in it["item_id"]])

    # rows
    def row(i, kind, rid, star=None, subsample=None):
        g = it["gaia_dr3"][i] if star is None else star["gaia"]
        d3 = dr3.get(int(g)) or {}
        rel = it["sigma_p"][i] / it["period"][i]
        return dict(dr3_epoch_public=bool(d3.get("has_epoch_photometry")),
                    dr3_variable_flag=str(d3.get("phot_variable_flag")),
                    sigma_rel_bin=("<3e-4" if rel < 3e-4 else "3e-4-1e-3" if rel < 1e-3 else "1e-3-3e-3" if rel < 3e-3 else ">=3e-3"),
                    row_id=rid, kind=kind, item_id=int(it["item_id"][i]), gaia_dr3=int(g),
                    ra=float(it["ra"][i] if star is None else star["ra"]), dec=float(it["dec"][i] if star is None else star["dec"]),
                    g=float(it["gmag"][i] if star is None else star["g"]), tier=str(tier[i]), mode=str(mode[i]),
                    period=float(it["period"][i]), sigma_p=float(it["sigma_p"][i]), t0=float(it["t0"][i]),
                    duration=float(it["duration"][i]), amp=float(abs(it["amp"][i])),
                    call_unequal=bool(call_unequal[i]), p_pred=float(p_pred[i]), primary=bool(primary[i]),
                    clean_eb=bool(clean_eb[i]), kepler_cal=bool(kepler_cal[i]), f21=None,
                    sigma_cal=bool(it["sigma_cal"][i]), err_group=str(it["err_group"][i]), subsample=subsample,
                    stratum=f"{tier[i]}|{mode[i]}|{'clean_eb' if clean_eb[i] else ('routed' if routed[i] else 'raw')}")
    rows = [row(i, "item", f"i{it['item_id'][i]}") for i in range(n) if testable[i] or primary[i]]
    # shuffled controls, matched
    quiet = pq.read_table(D / "quiet_stars.parquet", columns=["gaia_dr3", "ra", "dec", "gmag"]).to_pydict()
    qi = [k for k, g in enumerate(quiet["gaia_dr3"]) if int(g) in dr3]
    qra, qde = np.asarray(quiet["ra"])[qi], np.asarray(quiet["dec"])[qi]
    qg = np.asarray(quiet["gmag"])[qi]; qid = np.asarray(quiet["gaia_dr3"], np.int64)[qi]
    qel, qb = ecl_gal(qra, qde)
    qm = np.array([(dr3[int(g)]["ipd_frac_multi_peak"] or 0) > 0 for g in qid])
    key = lambda g, el, b, m: (int((g - 13.4) / 0.25), int((el + 90) / 10), int(min(abs(b), 30) / 5), int(m))  # noqa: E731
    pool = {}
    for k in rng.permutation(len(qid)):
        pool.setdefault(key(qg[k], qel[k], qb[k], qm[k]), []).append(k)
    prim = np.nonzero(primary)[0]
    el, b = ecl_gal(np.asarray(it["ra"])[prim], np.asarray(it["dec"])[prim])
    pick = rng.permutation(len(prim))[: min(15000, len(prim) // 4)]
    n_shuf = 0
    for j in pick:
        i = prim[j]
        bucket = pool.get(key(it["gmag"][i], el[j], b[j], multi[i]))
        if not bucket:
            continue
        k = bucket.pop()
        rows.append(row(i, "shuffled", f"s{it['item_id'][i]}",
                        dict(gaia=int(qid[k]), ra=float(qra[k]), dec=float(qde[k]), g=float(qg[k]))))
        n_shuf += 1
    # positive controls: catalogued eclipsing binaries of the held-out half only (never used to
    # calibrate sigma_P), each also at 50% and 30% of its Gaia epochs, so check (b) sees
    # intermediate power on real binaries
    pos = np.nonzero(np.asarray(it["vsx_holdout"]) & testable & refined)[0]
    for i in pos:
        rows.append(row(i, "positive", f"p{it['item_id'][i]}"))
        for frac in POS_SUBSAMPLE:
            rows.append(row(i, "positive", f"p{it['item_id'][i]}_s{int(frac * 100)}", subsample=frac))

    # coarse power on a sample of primary items
    samp = rng.choice(prim, size=min(3000, len(prim)), replace=False)
    srows = [{k: it[k][i] for k in ("item_id", "gaia_dr3", "ra", "dec", "gmag", "tier", "period", "sigma_p", "t0",
                                     "duration", "amp", "mode", "err_group")} for i in samp]
    pw = coarse_power(srows, dr3, rng)
    snr_g = np.abs(np.asarray(it["amp"], float)) / nsl.sigma_g(np.asarray(it["gmag"], float))
    def pbin(i):
        return (str(tier[i]), str(mode[i]), int(np.clip(np.log2(max(snr_g[i], 1e-3)), -3, 5)))
    table = {}
    for i, p in zip(samp, pw):
        if np.isfinite(p):
            table.setdefault(pbin(i), []).append(p)
    power_coarse = np.array([np.mean(table[pbin(i)]) if pbin(i) in table else np.nan for i in range(n)])
    for r in rows:
        r["power_coarse"] = float(power_coarse[idx_of[r["item_id"]]])
    pq.write_table(pa.Table.from_pylist(rows), D / "predictions.parquet", compression="zstd")
    sealed = dict(it); sealed.update(p_pred=p_pred, call_unequal=call_unequal, primary=primary, clean_eb=clean_eb,
                                     kepler_cal=kepler_cal, power_coarse=power_coarse)
    pq.write_table(pa.table(sealed), D / "items_sealed.parquet", compression="zstd")

    # endpoint predictions
    a_on = RULES["alpha"]
    pr = primary & ~kepler_cal                         # the E4 population excludes the Kepler labels
    pwc = np.nan_to_num(power_coarse)
    e1_pop = primary & clean_eb & (pwc >= RULES["power_strong"])   # the E1 population, as scored
    boot = []
    for _ in range(200):
        bi = rng.integers(0, len(lab_i), len(lab_i))
        boot.append(predict(logistic(X[lab_i[bi]], lab_y[bi]), X))
    boot = np.array(boot)
    def pred_mean(mask):
        """The estimator is recoveries over summed power, so its expectation is the
        power-weighted mean of p_pred: sum(p_pred x power) / sum(power)."""
        if not mask.any():
            return dict(n=0)
        w = pwc[mask]
        v = (boot[:, mask] * w).sum(1) / w.sum()
        return dict(n=int(mask.sum()), value=float((p_pred[mask] * w).sum() / w.sum()),
                    ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
    gm = np.asarray(it["gmag"], float)
    e1 = {"all": pred_mean(e1_pop)}
    for lo, hi in zip(G_BINS[:-1], G_BINS[1:]):
        e1[f"G{lo:.1f}-{hi:.1f}"] = pred_mean(e1_pop & (gm >= lo) & (gm < hi))
    q = p_pred * np.nan_to_num(power_coarse) + (1 - p_pred * np.nan_to_num(power_coarse)) * a_on
    e4m = pr & (np.nan_to_num(power_coarse) >= RULES["power_decided"])
    # P-versus-2P from Kepler: where Kepler detects and decides, agreement of the TESS call
    agree, dec_n = [], 0
    for rid, kind, ii in zip(kp["row_id"], kp["kind"], plan_idx):
        r = kl.get(rid)
        if kind != "signal" or ii is None or not r or not r.get("detected_any_amp") or r.get("p2p_sigma") in (None, 0):
            continue
        d, s, a = r.get("p2p_diff"), r.get("p2p_sigma"), r.get("p2p_amp")
        if None in (d, s, a) or a <= 0 or s > RULES["p2p_decidable_rel_sigma"] * abs(a):
            continue
        agree.append(bool(call_unequal[ii]) == bool(abs(d) / s >= RULES["p2p_unequal_nsigma"]))
    vc = json.loads((ROOT / "results" / "identity_counts.json").read_text())["counts"]
    e2_prior = vc.get("OFF_TARGET", 0) / max(1, sum(vc.get(k, 0) for k in ("OFF_TARGET", "KNOWN", "KNOWN_REVISE", "PASS")))
    raw_pw = pw[np.isfinite(pw)]
    preds = {
        "E1_purity_clean_eb_power_ge_0.8": dict(
            prediction=e1, population="primary clean-eb items (Kepler-labelled ones included, as in scoring) with "
                                      "power >= 0.8; before DR4 the coarse power stands in for the measured power",
            basis="power-weighted mean of the sealed p_pred, sum(p_pred x power)/sum(power), the expectation of "
                  "the estimator; 95% from refitting the Kepler model on bootstrap resamples"),
        "E2_off_target_fraction": dict(prediction=e2_prior, interval95=[0.10, 0.40],
                                       directional="the off-target fraction rises monotonically across the four "
                                                   "crowding bins of neighbour flux within 21 arcsec",
                                       basis="the survey's identity stage (results/identity_counts.json): eb items closed OFF_TARGET "
                                             "over those closed OFF_TARGET, KNOWN, KNOWN_REVISE or passing identity; "
                                             "a weakly informed prior, so the interval is wide"),
        "E3_p_vs_2p_accuracy": dict(prediction=float(np.mean(agree)) if agree else None, n_kepler=len(agree),
                                    basis="agreement of the sealed call with Kepler's own odd/even where Kepler decides"),
        "E4_calibration": dict(predicted_brier_if_calibrated=float(np.mean(q[e4m] * (1 - q[e4m]))) if e4m.any() else None,
                               n=int(e4m.sum()), kepler_cv_brier=brier_cv, kepler_cv_brier_skill=1 - brier_cv / brier_clim,
                               prediction="Brier skill score > 0, with a 95% interval excluding 0"),
        "E5_period": dict(prediction="over recovered items not in the sigma_P calibration half: robust std "
                                     "(1.4826 x MAD about the median) of (P_Gaia - P_TESS)/sigma_P in [0.8, 1.4], "
                                     "median in [-0.5, 0.5], at most 15% at the window edge",
                          basis="end-to-end simulation with heavy-tailed period errors (results/validation/e2e.json)"),
        "checks": dict(false_rate="false ON_STAR rate on shuffled controls <= 0.01",
                       positive_controls="observed/predicted in [0.8, 1.15]; observed >= power - 0.2 per bin with n >= 30"),
    }
    summary = dict(
        items=n, testable=int(testable.sum()), by_mode={m: int((mode == m).sum()) for m in np.unique(mode)},
        by_tier_mode={f"{t}|{m}": int(((tier == t) & (mode == m)).sum()) for t in ("box", "ls") for m in np.unique(mode)},
        primary=int(primary.sum()), primary_by_tier={t: int((primary & (tier == t)).sum()) for t in ("box", "ls")},
        clean_eb_primary=int((primary & clean_eb).sum()), shuffled=n_shuf, positive_items=int(len(pos)),
        positive_rows=int(len(pos) * (1 + len(POS_SUBSAMPLE))),
        rows=len(rows), kepler_cal_items=int(kepler_cal.sum()),
        coarse_power_sample=dict(n=int(len(raw_pw)), median=float(np.median(raw_pw)),
                                 frac_ge_0_5=float(np.mean(raw_pw >= 0.5)), frac_ge_0_8=float(np.mean(raw_pw >= 0.8)),
                                 by_tier={t: float(np.mean(raw_pw[np.array([tier[i] == t for i in samp])[np.isfinite(pw)]] >= 0.8))
                                          for t in ("box", "ls")}),
        p_pred_mean_primary={t: float(p_pred[primary & (tier == t)].mean()) for t in ("box", "ls")})
    R = ROOT / "results"
    (R / "endpoint_predictions.json").write_text(json.dumps(preds, indent=1))
    (R / "sample_summary.json").write_text(json.dumps(summary, indent=1))
    (R / "kepler_calibration.json").write_text(json.dumps(kep_summary, indent=1))
    print(json.dumps(summary, indent=1)); print(json.dumps(kep_summary, indent=1)[:3000])
    print(json.dumps(preds, indent=1)[:3000])


if __name__ == "__main__":
    main()
