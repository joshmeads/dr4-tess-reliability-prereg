"""End-to-end validation on simulated DR4-like data with known truth, through the sealed CLI.

Items: 45% on the star, 15% on a neighbour, 40% nowhere; box and sine tiers. The true period is
P + z sigma_P with z drawn from the HELD-OUT half's empirical period errors (heavy-tailed), while
the sealed power injections use the calibration half's quantiles (drrel/period_errors.json), so
the check is not circular. Shuffled controls: quiet stars with borrowed ephemerides. Positive
controls: real signals, each also at 50% and 30% of its epochs. Output: results/validation/e2e.json.
usage: validate_e2e.py N_ITEMS WORKERS
"""
import json
import pathlib
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drrel import score  # noqa: E402
from drrel.endpoints import purity  # noqa: E402
from drrel.simulate import make_bundle  # noqa: E402
from drrel.endpoints import e2_base  # noqa: E402
from drrel.stat import choose_mode  # noqa: E402

OUT = ROOT / "results" / "validation"


def main():
    n_items, workers = int(sys.argv[1]), int(sys.argv[2])
    rng = np.random.default_rng(20261202)
    bundles = OUT / "e2e_bundles"
    if bundles.exists():
        shutil.rmtree(bundles)
    bundles.mkdir(parents=True)
    rows, truth = [], {}
    it = pq.read_table(ROOT / "data" / "items.parquet",
                       columns=["tier", "period", "sigma_p", "vsx_period", "vsx_h", "vsx_holdout", "combined"]).to_pydict()
    z_held = {}
    for t_, P_, s_, vp, vh, ho, cb in zip(*(it[k] for k in ("tier", "period", "sigma_p", "vsx_period", "vsx_h",
                                                             "vsx_holdout", "combined"))):
        if ho and not cb and s_ and s_ > 0:
            z_held.setdefault(f"{t_}_single", []).append((P_ - vh * vp) / s_)
    z_held = {k: np.array(v) for k, v in z_held.items()}
    def ephem():
        tier = "box" if rng.random() < 0.5 else "sine"
        P = float(np.exp(rng.uniform(np.log(0.25), np.log(4.0))))
        D = float(rng.choice([0.03, 0.05, 0.08, 0.12])) if tier == "box" else 0.1 * P
        amp = float(np.exp(rng.uniform(np.log(0.004), np.log(0.06)))) if tier == "box" else \
            float(np.exp(rng.uniform(np.log(0.002), np.log(0.03))))
        # measured, calibrated sigma_P/P: box about 1e-4 to 4e-4, LS about 1e-3 to 4e-3
        sp = float(P * 10 ** (rng.uniform(-4.1, -3.4) if tier == "box" else rng.uniform(-3.4, -2.4)))
        return tier, P, D, amp, sp
    def row(kind, i, truth_kind, eph_src=None):
        tier, P, D, amp, sp = eph_src or ephem()
        g = float(rng.uniform(13.5, 15.5))
        ra, dec = float(rng.uniform(0, 360)), float(np.degrees(np.arcsin(rng.uniform(-0.4, 1.0))))
        t0 = float(rng.uniform(2458683, 2459824))
        mode = choose_mode(tier, P, sp, D / P if tier == "box" else 0.1, 3000.0)
        p_on = {"on_star": 0.8, "on_neighbour": 0.25, "none": 0.15}[truth_kind]  # an imperfect forecaster
        return dict(row_id=f"{kind}{i}", kind=kind, item_id=i, gaia_dr3=10_000_000 * (i + 1) + hash(kind) % 1000,
                    ra=ra, dec=dec, g=g, tier=tier, mode=mode, period=P, sigma_p=sp, t0=t0, duration=D, amp=amp,
                    call_unequal=False, p_pred=p_on, primary=True, clean_eb=bool(rng.random() < 0.5),
                    kepler_cal=False, sigma_cal=False, subsample=None,
                    err_group=("box" if tier == "box" else "ls") + "_single",
                    f21=float(rng.choice([0.01, 0.1, 0.5, 2.0])), stratum="sim"), (tier, P, D, amp, sp)
    def ztrue(r):
        return float(rng.choice(z_held[r["err_group"]]))
    for i in range(n_items):
        tk = rng.choice(["on_star", "on_neighbour", "none"], p=[0.45, 0.15, 0.40])
        r, e = row("item", i, tk)
        rows.append(r); truth[r["row_id"]] = tk
        nbs = [(float(rng.uniform(0.5, 3.0)), float(rng.uniform(5, 60))) for _ in range(rng.integers(1, 4))]
        make_bundle(bundles, r, tk, rng, neighbours=nbs, leak=float(rng.uniform(0.3, 1.0)), z_true=ztrue(r))
        if i % 2 == 0:                                    # a shuffled control for every second item
            c, _ = row("shuffled", i, "none", e)
            rows.append(c); truth[c["row_id"]] = "none"
            make_bundle(bundles, c, "none", rng, neighbours=nbs, z_true=ztrue(c))
        if i % 4 == 0:
            p, _ = row("positive", i, "on_star")
            rows.append(p); truth[p["row_id"]] = "on_star"
            make_bundle(bundles, p, "on_star", rng, neighbours=nbs, z_true=ztrue(p))
            for frac in (0.5, 0.3):
                q = dict(p, row_id=f"{p['row_id']}_s{int(frac * 100)}", subsample=frac)
                rows.append(q); truth[q["row_id"]] = "on_star"
                for ext in (".sources.json", ".epochs.parquet"):
                    shutil.copy(bundles / f"{p['row_id']}{ext}", bundles / f"{q['row_id']}{ext}")
    pq.write_table(pa.Table.from_pylist(rows), OUT / "e2e_predictions.parquet")
    (OUT / "e2e_truth.json").write_text(json.dumps(truth))
    score.main(["--predictions", str(OUT / "e2e_predictions.parquet"), "--bundles", str(bundles),
                "--out", str(OUT / "e2e_out"), "--workers", str(workers), "--unsealed"])
    res = [json.loads(line) for line in (OUT / "e2e_out" / "results.jsonl").read_text().splitlines()]
    ep = json.loads((OUT / "e2e_out" / "endpoints.json").read_text())["endpoints"]
    items = [r for r in res if r["kind"] == "item"]
    strong = [r for r in items if (r.get("power") or 0) >= 0.8 and r["outcome"] != "NO_TEST"]
    conf = {}
    for r in items:
        conf.setdefault(truth[r["row_id"]], {}).setdefault(r["outcome"], 0)
        conf[truth[r["row_id"]]][r["outcome"]] += 1
    true_on = np.mean([truth[r["row_id"]] == "on_star" for r in strong])
    sh = [r for r in res if r["kind"] == "shuffled"]
    a_on = ep["check_false_rate"]["on"]
    summary = dict(
        n_items=len(items), confusion_truth_by_outcome=conf,
        true_on_star_fraction_power_ge_0_8=float(true_on),
        estimated_purity_power_ge_0_8=float(purity(strong, a_on)),
        false_rate_shuffled=ep["check_false_rate"], positive_controls=ep["check_positive_controls"],
        e2_off_target_all=ep["E2_off_target"]["all"],
        true_off_target_among_real=float(np.mean([truth[r["row_id"]] == "on_neighbour" for r in items
                                                  if truth[r["row_id"]] != "none"])),
        true_off_target_in_e2_population=float(np.mean([truth[r["row_id"]] == "on_neighbour" for r in e2_base(items)
                                                        if truth[r["row_id"]] != "none"])),
        e4=dict(brier=ep["E4_calibration"]["brier"], reliability=ep["E4_calibration"].get("reliability")),
        e5=ep["E5_period"], n_shuffled=len(sh))
    (OUT / "e2e.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:4000])
    shutil.rmtree(bundles)


if __name__ == "__main__":
    main()
