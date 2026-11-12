# Preregistered Gaia DR4 reliability test of faint TESS FFI periodic signals

Author: Josh Meads

This record is a preregistration, deposited before Gaia Data Release 4 (planned 2026-12-02 12:00 CET).
Before any DR4 data are public, it fixes:

- the analysis plan (`PLAN.md`), with five primary endpoints and their predicted values;
- the complete scoring code (`code/drrel/`, with its sealed tables), its tests, and the code that built every input and prediction (`code/work/`);
- the pre-DR4 estimates the predictions rest on (`results/`);
- the SHA-256 of every file (`MANIFEST.sha256`), including the sealed catalogue, predictions and frozen inputs.

## The seal

Top-level hash (SHA-256 of `MANIFEST.sha256`):

    2b3632a3013097e46f0e76c14713a1c300720b29b96b8f40c50dc5f6871ecacc

The catalogue, the per-item predictions and their frozen inputs list about 10^5 periodic signals
that have not yet been published as variable stars. They are withheld here; only their hashes are
in the manifest. There are 13 withheld files, all under `data/`. When they are
released, anyone can check them:

    python -I code/drrel/verify_seal.py <folder> 2b3632a3013097e46f0e76c14713a1c300720b29b96b8f40c50dc5f6871ecacc

## Reproduce

    python -m venv .venv
    .venv/bin/pip install -r code/requirements.lock.txt
    cd code
    ../.venv/bin/python -I -m pytest -q tests

After DR4, follow PLAN.md section 13 and pass the hash above to the scorer:

    ../.venv/bin/python -m drrel.score --seal-hash 2b3632a3013097e46f0e76c14713a1c300720b29b96b8f40c50dc5f6871ecacc --predictions ../data/predictions.parquet --bundles B --out OUT

It checks sha256(MANIFEST.sha256) against that hash, then every rule file and the predictions file
against the manifest, and refuses to score on any mismatch. (The hash cannot be written inside
PLAN.md or the code: they are among the files it covers.)

To rebuild the predictions from this record and the embargoed record (PLAN.md section 14):

    python -I code/work/rebuild.py --public . --embargoed <embargoed record> --work <new scratch dir> [--vsx <VSX table>] [--purity]

## Licence

All files in this record are released under the MIT licence (see LICENSE).
