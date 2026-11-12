"""Score the sealed predictions against Gaia epoch photometry. Run after DR4 is public.

usage (from the code directory):
  python -m drrel.score --seal-hash TOP_HASH --predictions ../data/predictions.parquet --bundles DIR --out OUTDIR [--workers N]

A bundle is the Gaia data for one sealed row (an item, a shuffled control or a positive control):
  DIR/<row_id>.sources.json   [{"source_id", "g", "ra", "dec", "role": "target"|"neighbour"}, ...]
  DIR/<row_id>.epochs.parquet epoch photometry rows of those sources, with a source_id column
A row without a bundle is not scored; it is counted in the report as missing, never as a null.

Before scoring (verify_rules), sha256(MANIFEST.sha256) is checked against the published top-level
hash (--seal-hash); every rule file (all of drrel/ except the format adapters gaia_io.py and
fetch_dr4.py) and the predictions file against their manifest lines; the adapters' hashes are
reported. Each result is printed and written to the report, and a mismatch stops the run.

A positive-control row with `subsample` < 1 keeps that fraction of the target's epochs (a
deterministic choice seeded by the row id), so check (b) also sees intermediate power on real
binaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

from . import RULES, VERSION
from .controls import Ephemeris
from .endpoints import all_endpoints
from .gaia_io import series
from .outcome import classify
from .stat import best_epoch, odd_even, robust_beta

KEEP = ("row_id", "kind", "item_id", "gaia_dr3", "tier", "mode", "period", "sigma_p", "t0", "duration", "amp",
        "call_unequal", "p_pred", "primary", "clean_eb", "kepler_cal", "sigma_cal", "err_group", "subsample",
        "g", "f21", "stratum", "dr3_epoch_public", "dr3_variable_flag", "sigma_rel_bin", "power_coarse")
FORMAT_FILES = {"gaia_io.py", "fetch_dr4.py"}


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def verify_rules(expected_top=None, predictions=None):
    """Check this code against the published seal.

    1. sha256(MANIFEST.sha256) against expected_top, the top-level hash published outside the
       manifest (README, Zenodo record). The manifest cannot carry it (it would hash itself), so a
       forged manifest is caught here and only here.
    2. Every rule file (drrel/ except the format adapters) against its manifest line.
    3. The format adapters (gaia_io.py, fetch_dr4.py) against theirs: reported, not required,
       since the plan lets them change for format reasons; any change is visible in the report.
    4. The predictions file against its manifest line (data/predictions.parquet).
    all_match requires 1, 2 and 4."""
    here = pathlib.Path(__file__).resolve().parent
    manifest = here.parent.parent / "MANIFEST.sha256"
    files = sorted(p for p in here.iterdir() if p.is_file() and p.suffix in (".py", ".json"))
    if not manifest.exists():
        return dict(manifest_found=False, all_match=False, top=None, files={p.name: None for p in files})
    top = dict(computed=_sha(manifest), expected=expected_top)
    top["match"] = expected_top is not None and top["computed"] == expected_top
    sealed = {}
    pred_line = None
    for line in manifest.read_text().splitlines():
        h, rel = line.split("  ", 1)
        if rel.startswith("code/drrel/"):
            sealed[rel.split("/")[-1]] = h
        if rel == "data/predictions.parquet":
            pred_line = h
    rules = {p.name: _sha(p) == sealed.get(p.name) for p in files if p.name not in FORMAT_FILES}
    adapters = {p.name: _sha(p) == sealed.get(p.name) for p in files if p.name in FORMAT_FILES}
    missing = sorted(set(sealed) - FORMAT_FILES - set(rules))
    pred = None
    if predictions is not None:
        pred = dict(sha256=_sha(predictions), manifest=pred_line)
        pred["match"] = pred["sha256"] == pred_line
    all_match = bool(top["match"] and all(rules.values()) and not missing and pred is not None and pred["match"])
    return dict(manifest_found=True, top=top, files=rules, format_adapters_unchanged=adapters,
                sealed_rule_files_absent=missing, predictions=pred, all_match=all_match)


def sep_arcsec(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(min(1.0, max(-1.0, c)))) * 3600


def load_bundle(d: pathlib.Path, row_id: str):
    sj, ep = d / f"{row_id}.sources.json", d / f"{row_id}.epochs.parquet"
    if not (sj.exists() and ep.exists()):
        return None
    sources = json.loads(sj.read_text())
    tab = pq.read_table(ep).to_pydict()
    sid = np.asarray(tab["source_id"], dtype=np.int64)
    per = {}
    for s in sources:
        m = sid == int(s["source_id"])
        per[int(s["source_id"])] = series({k: np.asarray(v)[m] for k, v in tab.items()}) if m.any() else None
    return sources, per


def score_row(row: dict, bundle_dir: str):
    out = {k: row.get(k) for k in KEEP}
    b = load_bundle(pathlib.Path(bundle_dir), row["row_id"])
    if b is None:
        out["outcome"] = "MISSING"
        return out
    sources, per = b
    tgt = [s for s in sources if s["role"] == "target"]
    if len(tgt) != 1 or per.get(int(tgt[0]["source_id"])) is None:
        out["outcome"] = "NO_TEST"
        return out
    tgt = tgt[0]
    t, x, s = per[int(tgt["source_id"])]
    frac = row.get("subsample")
    if frac is not None and frac < 1:
        seed = int(hashlib.sha256(str(row["row_id"]).encode()).hexdigest()[:16], 16)
        keep = np.random.default_rng(seed).random(len(t)) < frac
        t, x, s = t[keep], x[keep], s[keep]
    eph = Ephemeris(tier="box" if row["tier"] == "box" else "sine", period=row["period"],
                    sigma_p=row["sigma_p"], t0=row["t0"], duration=row["duration"], amp=row["amp"],
                    mode=row.get("mode") or "anchored", err_group=row.get("err_group") or "")
    nbs = []
    for n in sources:
        if n["role"] != "neighbour":
            continue
        ser = per.get(int(n["source_id"]))
        nb = dict(source_id=int(n["source_id"]), g=float(n["g"]),
                  sep_arcsec=sep_arcsec(tgt["ra"], tgt["dec"], n["ra"], n["dec"]))
        if ser is not None:
            nb.update(t=ser[0], x=ser[1], s=ser[2])
        nbs.append(nb)
    nbs.sort(key=lambda n: n["g"])
    # crowding for endpoint 2: flux of the bundle's neighbours within 21" over the target's flux
    out["f21"] = float(sum(10 ** (-0.4 * (n["g"] - float(tgt["g"]))) for n in nbs if n["sep_arcsec"] < 21.0))
    if row.get("mode") == "untestable":
        out["outcome"] = "NO_TEST"
        return out
    res = classify(dict(t=t, x=x, s=s, g=float(tgt["g"])), eph, nbs)
    st = res["star"]
    out.update(outcome=res["outcome"], source=res.get("source"), power=st.get("power"), p=st.get("p"),
               S=st.get("S"), p_best=st.get("p_best"), amp_gaia=st.get("amp"), amp_sigma=st.get("amp_sigma"),
               n_eff=st.get("n_eff"), n_points=st.get("n_points"), nb_complete=res.get("nb_complete"),
               n_eligible=res.get("n_eligible"), neighbours=res.get("neighbours"))
    if res["outcome"] == "ON_STAR":
        t_ref = eph.t0 if eph.mode == "anchored" else best_epoch(eph.tier, t, x, s, st["p_best"], eph.duration)
        d, sd, a = odd_even(eph.tier, t, x, s, st["p_best"], t_ref, eph.duration, robust_beta(x, s))
        out.update(p2p_diff=d, p2p_sigma=sd, p2p_amp=a)
    return out


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--bundles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seal-hash", help="the published top-level hash (README / Zenodo record); required "
                                         "for a sealed run, which then refuses to score on any mismatch")
    ap.add_argument("--unsealed", action="store_true",
                    help="score without the seal check (simulations only); the report says so")
    a = ap.parse_args(argv)
    rules_check = verify_rules(a.seal_hash, a.predictions)
    tp = rules_check.get("top") or {}
    print("MANIFEST.sha256:", "found" if rules_check["manifest_found"] else "NOT FOUND")
    print(f"  sha256(MANIFEST.sha256) = {tp.get('computed')}")
    print(f"  published seal hash     = {tp.get('expected')}  ->  {'match' if tp.get('match') else 'MISMATCH'}")
    for name, ok in rules_check["files"].items():
        print(f"  rule file {name}: {'match' if ok else ('MISMATCH' if ok is False else 'unchecked')}")
    for name, ok in rules_check.get("format_adapters_unchanged", {}).items():
        print(f"  format adapter {name}: {'unchanged' if ok else 'CHANGED (allowed for format only; list the change)'}")
    pr = rules_check.get("predictions") or {}
    print(f"  predictions {pr.get('sha256')}: {'match' if pr.get('match') else 'MISMATCH'}")
    print("THE CODE AND PREDICTIONS ARE THE PUBLISHED SEAL" if rules_check["all_match"]
          else "THE CODE OR PREDICTIONS DO NOT MATCH THE PUBLISHED SEAL")
    if not rules_check["all_match"] and not a.unsealed:
        print("refusing to score; pass --unsealed only for simulations")
        return 2
    rows = pq.read_table(a.predictions).to_pylist()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(score_row, rows, [a.bundles] * len(rows), chunksize=16))
    results = [_clean(r) for r in results]
    with open(out / "results.jsonl", "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    scored = [r for r in results if r["outcome"] != "MISSING"]
    items = [r for r in scored if r["kind"] == "item" and r.get("primary")]
    ep = all_endpoints(items, [r for r in scored if r["kind"] == "shuffled"],
                       [r for r in scored if r["kind"] == "positive"])
    report = dict(version=VERSION, seal_check=rules_check, sealed_run=bool(rules_check["all_match"]),
                  counts={k: sum(r["outcome"] == k for r in results) for k in
                          ("ON_STAR", "ON_NEIGHBOUR", "NOWHERE", "UNDECIDED", "NO_TEST", "MISSING")},
                  endpoints=_clean(ep))
    (out / "endpoints.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
