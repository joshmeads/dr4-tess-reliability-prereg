"""Rebuild the sealed predictions from the deposits as published, and compare byte for byte.

usage: python -I code/work/rebuild.py --public PUBLIC_DIR --embargoed EMBARGOED_DIR --work SCRATCH
                                      [--vsx VSX_TABLE.parquet] [--purity]

PUBLIC_DIR is the git repository or Zenodo record 1 (it holds code/ and results/); EMBARGOED_DIR is
Zenodo record 2 (it holds data/). SCRATCH must not exist; it becomes a working copy of code/ with
the deposited inputs under the names the scripts read:

  deposit                              read by the scripts as
  data/catalogue.parquet               data/signals.parquet
  data/inputs/ephem_refined.parquet    data/ephem_refined.parquet
  data/inputs/items_built.parquet      data/items.parquet          (build_items.py output)
  data/inputs/dr3_columns.parquet      data/dr3_parts/cols_0000.parquet
  data/inputs/dr3_item_lr.parquet      data/dr3_item_lr.parquet    (purity_dr3.py output)
  data/inputs/{quiet_stars, searched_stars, kic_match, kepler_plan, kepler_labels}.parquet
                                       data/<same name>
  results/identity_counts.json         results/identity_counts.json

Steps, each compared with the deposit:
  1. with --vsx (the public VSX table; its SHA-256 is in results/inputs_provenance.json):
     build_items.py -> data/items.parquet and drrel/period_errors.json (inside SCRATCH only);
  2. with --purity: purity_dr3.py -> data/dr3_item_lr.parquet and results/purity_dr3.json (slow);
  3. build_predictions.py -> data/predictions.parquet, data/items_sealed.parquet (the deposit's
     data/items.parquet) and results/{endpoint_predictions, sample_summary, kepler_calibration}.json.
Exit code 0 only if every compared file is byte-identical.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

MAP = {"data/signals.parquet": "data/catalogue.parquet",
       "data/ephem_refined.parquet": "data/inputs/ephem_refined.parquet",
       "data/items.parquet": "data/inputs/items_built.parquet",
       "data/dr3_parts/cols_0000.parquet": "data/inputs/dr3_columns.parquet",
       "data/dr3_item_lr.parquet": "data/inputs/dr3_item_lr.parquet",
       "data/quiet_stars.parquet": "data/inputs/quiet_stars.parquet",
       "data/searched_stars.parquet": "data/inputs/searched_stars.parquet",
       "data/kic_match.parquet": "data/inputs/kic_match.parquet",
       "data/kepler_plan.parquet": "data/inputs/kepler_plan.parquet",
       "data/kepler_labels.parquet": "data/inputs/kepler_labels.parquet"}


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def run(work, script, env=None):
    print(f"running {script} ...", flush=True)
    r = subprocess.run([sys.executable, "-I", str(work / "work" / script)], cwd=work, env=env,
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-4000:])
        raise SystemExit(f"{script} failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", required=True)
    ap.add_argument("--embargoed", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--vsx")
    ap.add_argument("--purity", action="store_true")
    a = ap.parse_args()
    pub, emb, work = pathlib.Path(a.public), pathlib.Path(a.embargoed), pathlib.Path(a.work)
    if work.exists():
        raise SystemExit("--work must not exist")
    shutil.copytree(pub / "code", work)
    for dst, src in MAP.items():
        (work / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(emb / src, work / dst)
    (work / "results").mkdir()
    shutil.copy2(pub / "results" / "identity_counts.json", work / "results" / "identity_counts.json")
    compare = []                                  # (rebuilt file in work, deposited file)
    if a.vsx:
        state = work / "_vsx_state"
        (state / "cache").mkdir(parents=True)
        shutil.copy2(a.vsx, state / "cache" / "vsx_table.parquet")
        run(work, "build_items.py", env=dict(os.environ, DR4REL_SURVEY_STATE=str(state)))
        compare += [(work / "data" / "items.parquet", emb / "data" / "inputs" / "items_built.parquet"),
                    (work / "drrel" / "period_errors.json", pub / "code" / "drrel" / "period_errors.json"),
                    (work / "results" / "sigma_calibration.json", pub / "results" / "sigma_calibration.json")]
    if a.purity:
        run(work, "purity_dr3.py")
        compare += [(work / "data" / "dr3_item_lr.parquet", emb / "data" / "inputs" / "dr3_item_lr.parquet"),
                    (work / "results" / "purity_dr3.json", pub / "results" / "purity_dr3.json")]
    run(work, "build_predictions.py")
    compare += [(work / "data" / "predictions.parquet", emb / "data" / "predictions.parquet"),
                (work / "data" / "items_sealed.parquet", emb / "data" / "items.parquet")]
    compare += [(work / "results" / n, pub / "results" / n)
                for n in ("endpoint_predictions.json", "sample_summary.json", "kepler_calibration.json")]
    ok = True
    report = []
    for new, old in compare:
        same = new.exists() and sha(new) == sha(old)
        ok &= same
        report.append(dict(rebuilt=str(new.relative_to(work)), deposited=str(old.name), identical=same))
        print(f"  {'identical' if same else 'DIFFERS  '}  {new.relative_to(work)}  vs  {old.relative_to(old.parents[1])}")
    (work / "rebuild_report.json").write_text(json.dumps(report, indent=1))
    print("ALL REBUILT FILES ARE BYTE-IDENTICAL TO THE DEPOSIT" if ok else "SOME REBUILT FILES DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
