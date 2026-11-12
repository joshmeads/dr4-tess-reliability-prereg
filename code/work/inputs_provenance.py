"""Public provenance of the inputs that are not in the deposit. Output: results/inputs_provenance.json.

- The VSX table (public; AAVSO International Variable Star Index, as fetched 2026-11-10):
  SHA-256 and row count of the cached copy that build_items.py matched against.
- The survey's private ledger and stored light curves: described, not published. The frozen
  catalogue (data/catalogue.parquet) and refined ephemerides (data/inputs/ephem_refined.parquet)
  are their complete contribution; every later step can be rebuilt from the deposit.
Needs DR4REL_SURVEY_STATE.
"""
import hashlib
import json
import pathlib
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from paths import survey_state  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    vsx = survey_state() / "cache" / "vsx_table.parquet"
    out = dict(
        vsx_table=dict(description="AAVSO VSX: name, ra, dec, type, period for every object; fetched 2026-11-10 "
                                   "in strips and checked against VizieR's row count",
                       rows=pq.ParquetFile(vsx).metadata.num_rows, sha256=sha(vsx)),
        survey_ledger=dict(description="the survey's private SQLite ledger, opened read-only; snapshot 2026-11-11. "
                                       "Its content used here is data/catalogue.parquet (every signal) and the "
                                       "searched-star lists in data/inputs/."),
        stored_light_curves=dict(description="the survey's stored, conditioned TGLC light curves (private, about "
                                             "13 GB). Their content used here is data/inputs/ephem_refined.parquet."),
        gaia_dr3=dict(description="IRSA TAP gaia_dr3_source columns for signal stars and a seeded quiet sample: "
                                  "data/inputs/dr3_columns.parquet"),
        kepler=dict(description="MAST stpubdata bucket: the list of Kepler targets with light curves "
                                "(data/inputs/kepler_targets.txt) and two long-cadence quarters per sampled target, "
                                "streamed and reduced to data/inputs/kepler_labels.parquet; no raw file was kept"),
        rebuild_from_deposit="build_items.py (with the VSX table) -> purity_dr3.py -> build_predictions.py; "
                             "kepler_plan.py and kepler_calib.py are recorded but their outputs are frozen inputs")
    (ROOT / "results" / "inputs_provenance.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["vsx_table"]))


if __name__ == "__main__":
    main()
