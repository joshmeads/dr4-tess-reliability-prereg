"""Kepler calibration sample (seeded): items on KIC stars, plus quiet KIC stars with borrowed
ephemerides as the false-label control. Output: data/kepler_plan.parquet."""
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

D = pathlib.Path(__file__).resolve().parents[1] / "data"
N_BOX, N_LS, N_SHUF = 2000, 2000, 500


def main():
    rng = np.random.default_rng(20261202)
    km = pq.read_table(D / "kic_match.parquet").to_pydict()
    kic_of = {int(g): int(k) for g, k in zip(km["gaia_dr3"], km["kic"])}
    it = pq.read_table(D / "items.parquet").to_pylist()
    on_kic = [r for r in it if int(r["gaia_dr3"]) in kic_of and r["refined"] and r["sigma_p"] > 0 and r["amp"] > 0]
    box = [r for r in on_kic if r["tier"] == "box"]
    ls = [r for r in on_kic if r["tier"] == "ls"]
    pick = [box[i] for i in rng.permutation(len(box))[:N_BOX]] + [ls[i] for i in rng.permutation(len(ls))[:N_LS]]
    rows = [dict(row_id=f"k{r['item_id']}", kind="signal", item_id=r["item_id"], kic=kic_of[int(r["gaia_dr3"])],
                 gaia_dr3=int(r["gaia_dr3"]), tier="box" if r["tier"] == "box" else "sine", period=r["period"],
                 sigma_p=r["sigma_p"], t0=r["t0"], duration=r["duration"], amp=r["amp"]) for r in pick]
    quiet = [(int(g), int(k)) for g, k, hp in zip(km["gaia_dr3"], km["kic"], km["has_peak"]) if not hp]
    qs = [quiet[i] for i in rng.permutation(len(quiet))[:N_SHUF]]
    for j, (g, k) in enumerate(qs):
        d = pick[rng.integers(len(pick))]
        rows.append(dict(row_id=f"q{j}", kind="shuffled", item_id=-1, kic=k, gaia_dr3=g,
                         tier="box" if d["tier"] == "box" else "sine", period=d["period"], sigma_p=d["sigma_p"],
                         t0=d["t0"], duration=d["duration"], amp=d["amp"]))
    pq.write_table(pa.Table.from_pylist(rows), D / "kepler_plan.parquet")
    print("on KIC: box", len(box), "ls", len(ls), "quiet KIC", len(quiet), "-> plan rows", len(rows))


if __name__ == "__main__":
    main()
