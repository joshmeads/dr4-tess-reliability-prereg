"""Pre-seal check of the Gumbel pivot (recommended by an AI model, Claude Fable 5.2; PLAN.md 8.3), on simulated Gaia sampling.

1. Size: false recoveries (p <= 0.005) on pure noise for LS-like wide windows (40-189 disjoint
   controls), white noise and white + red noise (a random walk binned at 1 d, plus a 63 d term).
2. Agreement: on narrow-window items (>= 190 controls), subsample 40 controls and compare the
   pivot's decision with the rank decision, on noise and on injected signals.
Output: results/validation/pivot_check.json. usage: pivot_check.py N
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drrel import RULES, gumbel, nsl  # noqa: E402
from drrel.controls import Ephemeris, control_ratios, method_for, pvalue, test  # noqa: E402
from drrel.stat import model  # noqa: E402


def noise(t, s, rng, red):
    x = rng.normal(0, s)
    if red:
        day = np.floor(t - t[0]).astype(int)
        walk = np.cumsum(rng.normal(0, 0.3 * s[0], day.max() + 1))
        x += walk[day] - walk.mean() + 0.5 * s[0] * np.sin(2 * np.pi * t / 63.12 + rng.uniform(0, 6.3))
    return x


def main():
    n = int(sys.argv[1])
    rng = np.random.default_rng(20261202)
    res = {}
    for red in (False, True):
        hits, used, methods = 0, 0, {}
        for k in range(n):
            P = float(np.exp(rng.uniform(np.log(0.25), np.log(3.0))))
            eph = Ephemeris("sine", P, float(rng.uniform(1.1e-3, 4.4e-3) * P), 2459420.0, 0.1 * P, 0.005, mode="free")
            if method_for(len(control_ratios(eph))) != "pivot":
                continue
            ra, dec = float(rng.uniform(0, 360)), float(np.degrees(np.arcsin(rng.uniform(-0.4, 1.0))))
            t = nsl.transit_times(ra, dec, seed=k)
            s = np.full(len(t), nsl.sigma_g(rng.uniform(13.5, 15.5)))
            r = test(t, noise(t, s, rng, red), s, eph, with_power=False)
            if r["status"] != "OK":
                continue
            used += 1
            hits += r["p"] <= RULES["alpha"]
            methods[r["p_method"]] = methods.get(r["p_method"], 0) + 1
        res[f"size_{'red' if red else 'white'}"] = dict(n=used, false=hits, rate=hits / max(used, 1), methods=methods)
        print(res, flush=True)
    # agreement on narrow windows
    agree = dict(noise=[0, 0], signal=[0, 0])
    rank_hits = dict(noise=0, signal=0); piv_hits = dict(noise=0, signal=0)
    for k in range(n // 2):
        P = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
        eph = Ephemeris("box", P, 1.5e-4 * P, 2459420.0, 0.06, 0.006)
        t = nsl.transit_times(float(rng.uniform(0, 360)), float(rng.uniform(-20, 70)), seed=10000 + k)
        s = np.full(len(t), nsl.sigma_g(14.5))
        for kind in ("noise", "signal"):
            x = rng.normal(0, s) + (model("box", t, P, eph.t0, eph.duration, eph.amp) if kind == "signal" else 0)
            r = test(t, x, s, eph, with_power=False)
            if r["status"] != "OK" or r["p_method"] != "rank":
                continue
            sub = rng.choice(r["S_controls"][np.isfinite(r["S_controls"])], 40, replace=False)
            pp = pvalue(r["S"], sub, "pivot")
            a, b = r["p"] <= RULES["alpha"], pp <= RULES["alpha"]
            agree[kind][0] += a == b; agree[kind][1] += 1
            rank_hits[kind] += a; piv_hits[kind] += b
    res["agreement_40_subsample"] = dict(agree=agree, rank_recoveries=rank_hits, pivot_recoveries=piv_hits)
    res["critical_0.005"] = {str(m): gumbel.critical(m, 0.005) for m in (40, 100, 189)}
    out = ROOT / "results" / "validation" / "pivot_check.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
