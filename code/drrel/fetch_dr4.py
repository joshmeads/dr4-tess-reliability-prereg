"""Build scoring bundles from Gaia DR4 after release (stream and reduce). Format code only.

The DR4 services are not public at sealing time, so the endpoints, table and column names below
are configuration. The plan allows this module, and gaia_io's column lists, to change after
release for format reasons only; every change is logged in the scoring report. The sealed rules
that this module applies, and which may not change:

  target     the DR4 source matched to the sealed DR3 source_id by the official DR3-DR4 crossmatch
             (best match); failing that, the nearest DR4 source within 0.5 arcsec with |dG| < 0.3;
             failing that, no bundle (the row is NO_TEST, never a null).
  neighbours every DR4 source within 63 arcsec of the DR4 target (measured as the scorer measures
             it, before any ranking), other than the target, with
             G <= G_target - 2.5 log10(amp / cap) (cap 1 for box, 0.6 for sine), brightest first.
             All of them are listed in the bundle, so the scorer knows which went untested.
  photometry G-band epoch photometry of the target and of the 10 brightest such neighbours.

usage (from the code directory): python -m drrel.fetch_dr4 --predictions PRED.parquet --out BUNDLES --config dr4.json
config keys: tap_url, source_table, xmatch_table, datalink_url, datalink_params, batch (<= 5000)
"""
from __future__ import annotations

import argparse
import io
import json
import math
import pathlib
import time
import urllib.parse
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import RULES
from .score import sep_arcsec

UA = {"User-Agent": "dr4-reliability/1.0 (python-urllib)"}


def _post(url, fields, timeout=600):
    data = urllib.parse.urlencode(fields).encode()
    for k in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=UA), timeout=timeout) as r:
                return r.read()
        except OSError:
            if k == 5:
                raise
            time.sleep(30 * 2 ** k)
    return b""


def tap(cfg, adql):
    from astropy.io.votable import parse_single_table
    raw = _post(cfg["tap_url"], dict(REQUEST="doQuery", LANG="ADQL", FORMAT="votable", QUERY=adql))
    return parse_single_table(io.BytesIO(raw)).to_table(use_names_over_ids=True)


def max_dg(row):
    cap = RULES["max_implied_box"] if row["tier"] == "box" else RULES["max_implied_sine"]
    return 2.5 * math.log10(cap / max(row["amp"], 1e-6))


def sources_for(cfg, row):
    xm = tap(cfg, f"SELECT dr4_source_id FROM {cfg['xmatch_table']} WHERE dr3_source_id = {int(row['gaia_dr3'])} "
                  "ORDER BY angular_distance")
    st = cfg["source_table"]
    r_deg = RULES["neighbour_radius_arcsec"] / 3600
    cone = tap(cfg, f"SELECT source_id, ra, dec, phot_g_mean_mag FROM {st} WHERE 1 = CONTAINS("
                    f"POINT('ICRS', ra, dec), CIRCLE('ICRS', {row['ra']}, {row['dec']}, {r_deg + 2 / 3600}))")
    ids = [int(x) for x in cone["source_id"]]
    tgt = int(xm["dr4_source_id"][0]) if len(xm) and int(xm["dr4_source_id"][0]) in ids else None
    if tgt is None:
        best = None
        for sid, ra, dec, g in zip(ids, cone["ra"], cone["dec"], cone["phot_g_mean_mag"]):
            sep = math.hypot((ra - row["ra"]) * math.cos(math.radians(row["dec"])), dec - row["dec"]) * 3600
            if sep < 0.5 and abs(g - row["g"]) < 0.3 and (best is None or sep < best[1]):
                best = (sid, sep)
        tgt = best[0] if best else None
    if tgt is None:
        return None
    k = ids.index(tgt)
    out = [dict(source_id=tgt, g=float(cone["phot_g_mean_mag"][k]), ra=float(cone["ra"][k]),
                dec=float(cone["dec"][k]), role="target")]
    lim = out[0]["g"] + max_dg(row)
    # the radius is measured from the DR4 target, with the scorer's own separation, before ranking,
    # so every one of the 10 epoch slots goes to a source that classify will test
    nb = sorted((dict(source_id=s, g=float(g), ra=float(ra), dec=float(dec), role="neighbour")
                 for s, ra, dec, g in zip(ids, cone["ra"], cone["dec"], cone["phot_g_mean_mag"])
                 if s != tgt and np.isfinite(g) and g <= lim
                 and sep_arcsec(out[0]["ra"], out[0]["dec"], float(ra), float(dec)) <= RULES["neighbour_radius_arcsec"]),
                key=lambda d: d["g"])
    return out + nb                       # every eligible neighbour; epochs are fetched for the first 10


def epochs(cfg, ids):
    params = dict(cfg.get("datalink_params", {}), ID=",".join(str(i) for i in ids))
    raw = _post(cfg["datalink_url"], params)
    from astropy.io.votable import parse
    tabs = []
    for res in parse(io.BytesIO(raw)).iter_tables():
        tabs.append(res.to_table(use_names_over_ids=True))
    return tabs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    cfg = json.loads(pathlib.Path(a.config).read_text())
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for row in pq.read_table(a.predictions).to_pylist():
        if (out / f"{row['row_id']}.sources.json").exists() or row.get("mode") == "untestable":
            continue
        srcs = sources_for(cfg, row)
        if srcs is None:
            continue
        tabs = epochs(cfg, [s["source_id"] for s in srcs[: 1 + RULES["max_neighbours"]]])
        if not tabs:
            continue
        cols = {}
        for t in tabs:
            for c in t.colnames:
                cols.setdefault(c, []).append(np.asarray(t[c]))
        tab = pa.table({c: np.concatenate(v) for c, v in cols.items() if len(v) == len(tabs)})
        pq.write_table(tab, out / f"{row['row_id']}.epochs.parquet")
        (out / f"{row['row_id']}.sources.json").write_text(json.dumps(srcs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
