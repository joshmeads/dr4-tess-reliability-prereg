"""Assemble the sealed directory and its manifest.

seal/
  PLAN.md                        the preregistered analysis plan                    public
  code/                          drrel (scorer and its sealed tables), work (the code that built
                                 every input and prediction), tests, lock file       public
  results/                       pre-DR4 estimates, predictions, validations,
                                 input provenance                                   public
  data/catalogue.parquet         frozen catalogue of every signal                   embargoed
  data/items.parquet             test items with sealed predictions                 embargoed
  data/predictions.parquet       sealed rows: items, shuffled and positive controls embargoed
  data/inputs/*                  frozen inputs needed to rebuild the predictions    embargoed
  MANIFEST.sha256                SHA-256 of every file above
  SEAL.json                      top-level hash = SHA-256 of MANIFEST.sha256

Everything under data/ carries candidate identifiers and is embargoed (hash only in public).
usage: build_seal.py   (refuses to overwrite an existing seal)
"""
import hashlib
import json
import pathlib
import shutil
import sys
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEAL = ROOT / "seal"
PUBLIC_RESULTS = ("purity_dr3.json", "sigma_calibration.json", "kepler_calibration.json", "identity_counts.json",
                  "endpoint_predictions.json", "sample_summary.json", "inputs_provenance.json",
                  "validation/e2e.json", "validation/positive_controls_kepler.json", "validation/pivot_check.json",
                  "validation/narrow_window_check.json")
DATA = {"data/catalogue.parquet": "data/signals.parquet", "data/items.parquet": "data/items_sealed.parquet",
        "data/predictions.parquet": "data/predictions.parquet",
        "data/inputs/items_built.parquet": "data/items.parquet",
        "data/inputs/ephem_refined.parquet": "data/ephem_refined.parquet",
        "data/inputs/searched_stars.parquet": "data/searched_stars.parquet",
        "data/inputs/quiet_stars.parquet": "data/quiet_stars.parquet",
        "data/inputs/dr3_item_lr.parquet": "data/dr3_item_lr.parquet",
        "data/inputs/kic_match.parquet": "data/kic_match.parquet",
        "data/inputs/kepler_plan.parquet": "data/kepler_plan.parquet",
        "data/inputs/kepler_labels.parquet": "data/kepler_labels.parquet",
        "data/inputs/kepler_targets.txt": "data/kepler_targets.txt"}
CODE_SKIP = {"__pycache__"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dr3_columns(dst):
    """The DR3 snapshot as one table: every fetched batch, one row per source_id."""
    parts = sorted((ROOT / "data" / "dr3_parts").glob("cols_*.parquet"))
    t = pa.concat_tables([pq.read_table(p) for p in parts], promote_options="default")
    ids = t["source_id"].to_numpy()
    _, first = __import__("numpy").unique(ids, return_index=True)
    pq.write_table(t.take(pa.array(sorted(first))), dst, compression="zstd")


def main():
    if SEAL.exists():
        print("seal/ exists; refusing to overwrite. Move it aside to rebuild.")
        return 1
    SEAL.mkdir()
    shutil.copy2(ROOT / "PLAN.md", SEAL / "PLAN.md")
    code = SEAL / "code"
    for sub in ("drrel", "work", "tests"):
        (code / sub).mkdir(parents=True)
        for f in sorted(list((ROOT / sub).glob("*.py")) + list((ROOT / sub).glob("*.json"))):
            shutil.copy2(f, code / sub / f.name)
    shutil.copy2(ROOT / "requirements.lock.txt", code / "requirements.lock.txt")
    for r in PUBLIC_RESULTS:
        src = ROOT / "results" / r
        if not src.exists():
            print("missing result:", r)
            return 1
        (SEAL / "results" / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, SEAL / "results" / r)
    (SEAL / "data" / "inputs").mkdir(parents=True)
    for dst, src in DATA.items():
        shutil.copy2(ROOT / src, SEAL / dst)
    dr3_columns(SEAL / "data" / "inputs" / "dr3_columns.parquet")
    files = sorted(p for p in SEAL.rglob("*") if p.is_file())
    lines = [f"{sha(p)}  {p.relative_to(SEAL).as_posix()}" for p in files]
    (SEAL / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    top = sha(SEAL / "MANIFEST.sha256")
    embargoed = sorted(p.relative_to(SEAL).as_posix() for p in files if p.relative_to(SEAL).parts[0] == "data")
    seal = dict(top_level_sha256=top, rule="SHA-256 of MANIFEST.sha256 (UTF-8, one 'sha256  path' line per file)",
                sealed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                embargoed=embargoed, gaia_dr4_release="2026-12-02T12:00:00+01:00", author="Josh Meads")
    (SEAL / "SEAL.json").write_text(json.dumps(seal, indent=1) + "\n")
    print("top-level hash:", top)
    for p in files:
        print(f"  {p.stat().st_size:>12,d}  {p.relative_to(SEAL)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
