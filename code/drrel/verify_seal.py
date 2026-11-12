"""Verify a sealed directory: every file's SHA-256 against MANIFEST.sha256, and the top-level hash.

usage: python -I path/to/drrel/verify_seal.py SEAL_DIR [EXPECTED_TOP_LEVEL_SHA256]
Files listed in the manifest but absent (for example the embargoed data in a public copy) are
reported as absent, not as failures; a present file with a different hash is a failure.
"""
import hashlib
import pathlib
import sys


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    d = pathlib.Path(argv[0])
    top = sha(d / "MANIFEST.sha256")
    ok, absent, bad = 0, [], []
    for line in (d / "MANIFEST.sha256").read_text().splitlines():
        h, rel = line.split("  ", 1)
        p = d / rel
        if not p.exists():
            absent.append(rel)
        elif sha(p) != h:
            bad.append(rel)
        else:
            ok += 1
    print(f"top-level sha256: {top}")
    if len(argv) > 1:
        print("matches expected" if top == argv[1] else "DOES NOT MATCH expected " + argv[1])
    print(f"{ok} files verified, {len(absent)} absent (hash only), {len(bad)} mismatched")
    for rel in absent:
        print("  absent:", rel)
    for rel in bad:
        print("  MISMATCH:", rel)
    return 1 if bad or (len(argv) > 1 and top != argv[1]) else 0


if __name__ == "__main__":
    sys.exit(main())
