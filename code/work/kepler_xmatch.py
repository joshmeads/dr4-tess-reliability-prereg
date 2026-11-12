"""KIC identifiers for our Kepler-field stars, from TIC v8 at MAST (TAP), looked up by TIC ID
(the table's primary key; our ledger holds the TIC of most stars). Kept only where the KIC is a
Kepler target with light curves (data/kepler_targets.txt, from kepler_targets.py).

Sample: every signal star inside the Kepler field's bounding box, and 10,000 seeded quiet stars
there. Output: data/kic_match.parquet.
"""
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import tap  # noqa: E402

D = pathlib.Path(__file__).resolve().parents[1] / "data"
URL = "https://mast.stsci.edu/vo-tap/api/v0.1/tic/sync"
N_QUIET, B = 10000, 500


def main():
    st = pq.read_table(D / "searched_stars.parquet", columns=["gaia_dr3", "tic", "ra", "dec", "has_signal",
                                                              "has_peak"]).to_pydict()
    ra, dec = np.asarray(st["ra"]), np.asarray(st["dec"])
    gaia = np.asarray(st["gaia_dr3"], np.int64)
    tic = np.asarray([t or 0 for t in st["tic"]], np.int64)
    box = (ra > 279.5) & (ra < 301.8) & (dec > 36.5) & (dec < 52.5) & (tic > 0)
    sig = box & np.asarray(st["has_signal"]); quiet = box & ~np.asarray(st["has_peak"])
    rng = np.random.default_rng(20261202)
    pick = np.concatenate([np.nonzero(sig)[0], rng.permutation(np.nonzero(quiet)[0])[:N_QUIET]])
    print("to look up:", len(pick), "signal:", int(sig.sum()), flush=True)
    parts = D / "kic_parts_tic"; parts.mkdir(exist_ok=True)
    ids = tic[pick]
    for k in range(0, len(ids), B):
        f = parts / f"b{k // B:04d}.parquet"
        if f.exists():
            continue
        t, _ = tap.query_retry(URL, "SELECT id, kic FROM dbo.catalogrecord WHERE id IN (" +
                               ",".join(str(int(i)) for i in ids[k:k + B]) + ")", timeout=300, maxrec=2 * B)
        pq.write_table(tap.to_arrow(t), f)
        if k // B % 20 == 0:
            print(f"batch {k // B}/{len(ids) // B}", flush=True)
    t = pa.concat_tables([pq.read_table(p) for p in sorted(parts.glob("*.parquet"))], promote_options="default").to_pydict()
    kic_of_tic = {int(i): int(k) for i, k in zip(t["id"], t["kic"]) if k is not None and int(k) > 0}
    targets = {int(x) for x in (D / "kepler_targets.txt").read_text().split()}
    sigset = set(np.nonzero(sig)[0].tolist())
    rows = [(int(gaia[i]), kic_of_tic[int(tic[i])], int(tic[i]), i in sigset) for i in pick
            if int(tic[i]) in kic_of_tic and kic_of_tic[int(tic[i])] in targets]
    out = pa.table({"gaia_dr3": [r[0] for r in rows], "kic": [r[1] for r in rows], "tic": [r[2] for r in rows],
                    "has_signal": [r[3] for r in rows], "has_peak": [r[3] for r in rows]})
    pq.write_table(out, D / "kic_match.parquet")
    print("with a KIC:", len(kic_of_tic), "that are Kepler targets:", len(rows),
          "signal:", sum(r[3] for r in rows), "quiet:", sum(not r[3] for r in rows))


if __name__ == "__main__":
    main()
