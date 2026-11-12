"""Publication folders for the user (nothing is uploaded).

publish/zenodo_public/     plan, code, public results, MANIFEST.sha256, SEAL.json, README, CITATION,
                           LICENSE, .zenodo.json. No embargoed data; their hashes are in the manifest.
                           This is the record that fixes the date.
publish/zenodo_embargoed/  the embargoed data (data/), for a second record under Zenodo's embargo.
publish/git/               the same public content as a repository tree.

Embargo guard: every file in the two public folders is scanned (scan_text) for
- any run of 6-19 digits, bounded only by non-digits, that is a Gaia DR3, TIC or KIC ID of the
  embargoed data, in any form: "TIC 123", "TIC123", "tic_123", "TIC-123", "123.json",
  "Gaia DR3 123.", ...;
- any consecutive pair of decimal numbers that equals an embargoed star's (RA, Dec) to 4 decimals;
- any 2MASS designation.
One hit fails the build and removes the public folders.
usage: build_publish.py
"""
import json
import pathlib
import re
import shutil
import sys

import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEAL = ROOT / "seal"
PUB = ROOT / "publish"
AUTHOR = "Josh Meads"
TITLE = "Preregistered Gaia DR4 reliability test of faint TESS FFI periodic signals"
LICENCE_LINE = "All files in this record are released under the MIT licence (see LICENSE)."
ID_COLUMNS = ("gaia_dr3", "source_id", "tic", "kic")
TWO_MASS = re.compile(r"(2MASS\s*)?J\d{8}[+-]\d{7}")
DIGITS = re.compile(r"(?<!\d)\d{6,19}(?!\d)")
DECIMAL = re.compile(r"-?\d{1,3}\.\d{4,}")


def embargoed_ids(files):
    """Every integer in an identifier column (Gaia, TIC, KIC) of the embargoed tables, and every
    (RA, Dec) pair rounded to 4 decimals. kepler_targets.txt, the public list of all Kepler
    targets, is not read."""
    ids, coords = set(), set()
    for p in files:
        if p.suffix != ".parquet":
            continue
        names = pq.ParquetFile(p).schema_arrow.names
        cols = [c for c in names if c in ID_COLUMNS]
        if cols:
            t = pq.read_table(p, columns=cols)
            for c in cols:
                ids.update(int(v) for v in t[c].to_pylist() if v is not None and int(v) > 0)
        if "ra" in names and "dec" in names:
            t = pq.read_table(p, columns=["ra", "dec"])
            coords.update((round(a, 4), round(b, 4)) for a, b in zip(t["ra"].to_pylist(), t["dec"].to_pylist())
                          if a is not None and b is not None)
    return ids, coords


def scan_text(text, ids, coords):
    """Hits of one text: [(kind, matched text)]."""
    hits = [("2MASS designation", m.group(0)) for m in TWO_MASS.finditer(text)]
    hits += [("embargoed identifier", m.group(0)) for m in DIGITS.finditer(text) if int(m.group(0)) in ids]
    vals = [(m.group(0), float(m.group(0))) for m in DECIMAL.finditer(text)]
    hits += [("embargoed coordinates", f"{a} {b}") for (a, x), (b, y) in zip(vals, vals[1:])
             if (round(x, 4), round(y, 4)) in coords]
    return hits


def leak_scan(folders, ids, coords):
    hits = []
    for folder in folders:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(folder.parent))
                hits += [(rel, k, v) for k, v in scan_text(p.read_bytes().decode("utf8", "replace"), ids, coords)]
                hits += [(rel, "file name", k) for k, _ in scan_text(p.name, ids, set())]
    return hits


def readme(top, embargoed):
    return f"""# {TITLE}

Author: {AUTHOR}

This record is a preregistration, deposited before Gaia Data Release 4 (planned 2026-12-02 12:00 CET).
Before any DR4 data are public, it fixes:

- the analysis plan (`PLAN.md`), with five primary endpoints and their predicted values;
- the complete scoring code (`code/drrel/`, with its sealed tables), its tests, and the code that built every input and prediction (`code/work/`);
- the pre-DR4 estimates the predictions rest on (`results/`);
- the SHA-256 of every file (`MANIFEST.sha256`), including the sealed catalogue, predictions and frozen inputs.

## The seal

Top-level hash (SHA-256 of `MANIFEST.sha256`):

    {top}

The catalogue, the per-item predictions and their frozen inputs list about 10^5 periodic signals
that have not yet been published as variable stars. They are withheld here; only their hashes are
in the manifest. There are {len(embargoed)} withheld files, all under `data/`. When they are
released, anyone can check them:

    python -I code/drrel/verify_seal.py <folder> {top}

## Reproduce

    python -m venv .venv
    .venv/bin/pip install -r code/requirements.lock.txt
    cd code
    ../.venv/bin/python -I -m pytest -q tests

After DR4, follow PLAN.md section 13 and pass the hash above to the scorer:

    ../.venv/bin/python -m drrel.score --seal-hash {top} --predictions ../data/predictions.parquet --bundles B --out OUT

It checks sha256(MANIFEST.sha256) against that hash, then every rule file and the predictions file
against the manifest, and refuses to score on any mismatch. (The hash cannot be written inside
PLAN.md or the code: they are among the files it covers.)

To rebuild the predictions from this record and the embargoed record (PLAN.md section 14):

    python -I code/work/rebuild.py --public . --embargoed <embargoed record> --work <new scratch dir> [--vsx <VSX table>] [--purity]

## Licence

{LICENCE_LINE}
"""


def citation(top):
    return f"""cff-version: 1.2.0
message: "If you use this preregistration or its code, please cite it."
title: "{TITLE}"
authors:
  - family-names: Meads
    given-names: Josh
version: "1.1.1"
date-released: "2026-11-12"
license: MIT
keywords: [TESS, Gaia DR4, variable stars, eclipsing binaries, preregistration, reliability]
abstract: >-
  Sealed predictions and frozen scoring code for testing, with Gaia DR4 epoch photometry, what
  fraction of faint (G 13.5-15.5) TESS full-frame-image periodic signals are real and on the named
  star, on a neighbour, or nowhere. Seal SHA-256 {top}.
"""


def zenodo_meta(top):
    return dict(title=TITLE, upload_type="software", creators=[dict(name="Meads, Josh")],
                description=(f"Preregistration deposited before Gaia DR4 (2026-12-02). Plan, scoring code, pre-DR4 "
                             f"estimates and SHA-256 manifest. Seal (SHA-256 of MANIFEST.sha256): {top}. "
                             "The catalogue, prediction and input data files are withheld (hash only) until the "
                             "corresponding variable-star submissions are made. " + LICENCE_LINE),
                access_right="open", license="MIT", version="1.1.1",
                keywords=["TESS", "Gaia DR4", "preregistration", "eclipsing binaries", "variable stars"])


MIT = """MIT License

Copyright (c) 2026 Josh Meads

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def main():
    seal = json.loads((SEAL / "SEAL.json").read_text())
    top, emb = seal["top_level_sha256"], seal["embargoed"]
    if PUB.exists():
        shutil.rmtree(PUB)
    pub = PUB / "zenodo_public"
    for p in SEAL.rglob("*"):
        rel = p.relative_to(SEAL).as_posix()
        if p.is_file() and rel not in emb:
            (pub / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, pub / rel)
    (pub / "README.md").write_text(readme(top, emb))
    (pub / "CITATION.cff").write_text(citation(top))
    (pub / "LICENSE").write_text(MIT)
    (pub / ".zenodo.json").write_text(json.dumps(zenodo_meta(top), indent=1))
    embd = PUB / "zenodo_embargoed"
    for rel in emb:
        (embd / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SEAL / rel, embd / rel)
    shutil.copy2(SEAL / "MANIFEST.sha256", embd / "MANIFEST.sha256")
    (embd / "README.md").write_text(f"# Embargoed data of the DR4 reliability preregistration\n\nAuthor: {AUTHOR}\n\n"
                                    f"Files listed in MANIFEST.sha256 of the public record; seal {top}.\n\n{LICENCE_LINE}\n")
    shutil.copy2(pub / "LICENSE", embd / "LICENSE")
    git = PUB / "git"
    shutil.copytree(pub, git)
    (git / ".gitignore").write_text("data/\n.venv/\n__pycache__/\n")
    ids, coords = embargoed_ids([SEAL / rel for rel in emb])
    hits = leak_scan([pub, git], ids, coords)
    if hits:
        for h in hits[:50]:
            print("LEAK:", *h)
        shutil.rmtree(pub); shutil.rmtree(git)
        print(f"FAILED: {len(hits)} embargoed identifiers or 2MASS names in public files; public folders removed")
        return 1
    print(f"embargo scan: {len(ids):,} embargoed identifiers and {len(coords):,} coordinate pairs checked "
          "against every public file and file name, 0 found")
    print("public files:", sum(1 for p in pub.rglob('*') if p.is_file()), "embargoed:", len(emb), "seal:", top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
