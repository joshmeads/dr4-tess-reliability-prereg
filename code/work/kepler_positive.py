"""Real-data validation on known binaries: Kepler labels of items matching a catalogued (VSX)
eclipsing binary, split into the half held out from the sigma_P calibration and the calibration
half, against the injection power the same code predicted, and against items with no catalogue
match. Output: results/validation/positive_controls_kepler.json."""
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "data"
sys.path.insert(0, str(ROOT / "work"))
from kepmap import plan_to_items  # noqa: E402


def main():
    plan = pq.read_table(D / "kepler_plan.parquet").to_pydict()
    lab = {r["row_id"]: r for r in pq.read_table(D / "kepler_labels.parquet").to_pylist()}
    it = pq.read_table(D / "items.parquet", columns=["gaia_dr3", "tier", "period", "vsx_eb", "vsx_holdout"]).to_pydict()
    idx = plan_to_items(plan, it)
    groups = {"vsx_eclipsing_binary_held_out": [], "vsx_eclipsing_binary_calibration_half": [],
              "no_catalogue_match": [], "quiet_star_borrowed_ephemeris": []}
    for rid, kind, ii in zip(plan["row_id"], plan["kind"], idx):
        r = lab.get(rid)
        if r is None or r["status"] != "OK":
            continue
        if kind == "shuffled":
            groups["quiet_star_borrowed_ephemeris"].append(r)
        elif ii is None:
            continue
        elif it["vsx_holdout"][ii]:
            groups["vsx_eclipsing_binary_held_out"].append(r)
        elif it["vsx_eb"][ii]:
            groups["vsx_eclipsing_binary_calibration_half"].append(r)
        else:
            groups["no_catalogue_match"].append(r)
    out = {}
    for g, rows in groups.items():
        pw = np.array([r["power"] if r["power"] is not None else np.nan for r in rows])
        det = np.array([bool(r["detected_any_amp"]) for r in rows])
        real = np.array([bool(r["real_on"]) for r in rows])
        bins = []
        for lo, hi in ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)):
            m = (pw >= lo) & (pw < hi)
            if m.sum():
                bins.append(dict(power_bin=[lo, hi], n=int(m.sum()), mean_power=float(pw[m].mean()),
                                 detected=float(det[m].mean()), real_on=float(real[m].mean())))
        out[g] = dict(n=len(rows), detected=float(det.mean()) if len(rows) else None,
                      real_on=float(real.mean()) if len(rows) else None,
                      mean_power=float(np.nanmean(pw)) if len(rows) else None, by_power=bins)
    out["note"] = ("Kepler labels were computed with the labelling rules of kepler_calib.py (same-star test with a "
                   "free phase on two Kepler quarters, 100 controls at 0.4-2.5 P, alpha 0.01; real_on also needs "
                   "Kepler amplitude >= 0.3 x TESS amplitude) and the sigma_P of the plan file (first calibration).")
    (ROOT / "results" / "validation" / "positive_controls_kepler.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({g: {k: v for k, v in o.items() if k != "by_power"} for g, o in out.items() if g != "note"}, indent=1))


if __name__ == "__main__":
    main()
