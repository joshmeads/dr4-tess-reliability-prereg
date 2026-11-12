"""Pre-seal check of the pivot guard (pivot_max_nonpositive, then the extended rank test), on simulated
Gaia sampling, pure noise.

A. Sol's round-2 setting: the narrowest-window box ephemeris (6 sigma_P x 3000 / P^2 < 1, duty
   0.12-0.3, anchored), 40 sampled transits: before the guard the pivot gave 38 false of 879.
B. All primary box items whose window spans < 1 independent period, natural sampling (all
   transits): method used and false rate.
C. A random sample of primary items (box and LS), natural sampling: how often the guard fires.
Output: results/validation/narrow_window_check.json. usage: narrow_window_check.py N_A N_B N_C
"""
import collections
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drrel import RULES, nsl  # noqa: E402
from drrel.controls import Ephemeris, test  # noqa: E402
from drrel.nsl import DR4_END, DR4_START  # noqa: E402


def eph_of(r):
    return Ephemeris("box" if r["tier"] == "box" else "sine", r["period"], r["sigma_p"], r["t0"], r["duration"],
                     r["amp"], r["mode"], r["err_group"])


def n_indep(r):
    dt = max(abs(r["t0"] - DR4_START), abs(r["t0"] - DR4_END))
    return 2 * RULES["window_nsigma"] * r["sigma_p"] * dt / r["period"] ** 2


def run(rows, rng, n_pts=None, seed0=0):
    c = collections.Counter()
    for k, r in enumerate(rows):
        t = nsl.transit_times(r["ra"], r["dec"], seed=seed0 + k)
        if n_pts:
            if len(t) < n_pts:
                continue
            t = np.sort(rng.choice(t, n_pts, replace=False))
        s = np.full(len(t), nsl.sigma_g(r["g"]))
        o = test(t, rng.normal(0, s), s, eph_of(r), with_power=False)
        key = o["p_method"] if o["status"] == "OK" else f"{o['status']}:{o.get('p_method')}"
        c[key] += 1
        if o["status"] == "OK" and o["p"] <= RULES["alpha"]:
            c[key + " false"] += 1
    return dict(c)


def main():
    nA, nB, nC = (int(x) for x in sys.argv[1:4])
    rng = np.random.default_rng(20261202)
    pr = pq.read_table(ROOT / "data" / "predictions.parquet", filters=[("kind", "=", "item"), ("primary", "=", True)],
                       columns=["tier", "period", "sigma_p", "t0", "duration", "amp", "ra", "dec", "g", "mode",
                                "err_group"]).to_pylist()
    box = [r for r in pr if r["tier"] == "box"]
    sol = sorted((r for r in box if r["mode"] == "anchored" and .12 <= r["duration"] / r["period"] <= .3),
                 key=lambda r: 6 * r["sigma_p"] * 3000 / r["period"] ** 2)[0]
    out = {"A_sol_setting_40_transits": run([sol] * nA, rng, n_pts=40)}
    narrow = [r for r in box if n_indep(r) < 1]
    out["B_narrow_box_natural"] = dict(n_items=len(narrow),
                                       result=run([narrow[j] for j in rng.permutation(len(narrow))[:nB]] if narrow else [], rng, seed0=10000))
    samp = [pr[j] for j in rng.permutation(len(pr))[:nC]]
    out["C_primary_sample_natural"] = run(samp, rng, seed0=20000)
    (ROOT / "results" / "validation" / "narrow_window_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
