"""Small TAP client (IRSA and MAST) with optional VOTable upload. Standard library + astropy.

A failed or truncated query raises; an empty answer from a failing service is never returned
as a result. Callers retry with backoff.
"""
import http.client
import io
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import numpy as np
from astropy.io.votable import from_table, parse_single_table, writeto
from astropy.table import Table

IRSA = "https://irsa.ipac.caltech.edu/TAP/sync"
MAST_GAIA = "https://mast.stsci.edu/vo-tap/api/v0.1/gaiadr3/sync"
UA = {"User-Agent": "dr4-reliability/1.0 (python-urllib)"}


def _multipart(fields, files):
    b = uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    for k, (fn, data) in files.items():
        out.write(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fn}\"\r\n"
                  "Content-Type: application/x-votable+xml\r\n\r\n".encode())
        out.write(data)
        out.write(b"\r\n")
    out.write(f"--{b}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={b}"


def query(url, adql, upload=None, timeout=600, maxrec=None):
    """Run a sync ADQL query. upload: astropy Table sent as TAP_UPLOAD.up. Returns astropy Table."""
    fields = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": adql}
    if maxrec:
        fields["MAXREC"] = str(maxrec)
    if upload is not None:
        buf = io.BytesIO()
        writeto(from_table(upload), buf)
        fields["UPLOAD"] = "up,param:up"
        body, ctype = _multipart(fields, {"up": ("up.xml", buf.getvalue())})
    else:
        body, ctype = urllib.parse.urlencode(fields).encode(), "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=body, headers={**UA, "Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        vot = parse_single_table(io.BytesIO(raw))
    except IndexError as e:   # an error document, not a result: never treat it as empty
        raise ValueError("service returned no table: " + raw[:600].decode("utf8", "replace")) from e
    t = vot.to_table(use_names_over_ids=True)
    return t, raw


def query_retry(url, adql, upload=None, tries=6, timeout=600, maxrec=None):
    delay = 30
    for k in range(tries):
        try:
            return query(url, adql, upload=upload, timeout=timeout, maxrec=maxrec)
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError,
                http.client.HTTPException) as e:  # noqa: PERF203
            if k == tries - 1:
                raise
            print(f"  retry {k + 1} after {type(e).__name__}: {str(e)[:200]}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 600)
    raise RuntimeError("unreachable")


def to_arrow(t):
    """astropy Table -> pyarrow Table; masked floats become NaN, masked ints -1, text stays text."""
    import pyarrow as pa
    cols = {}
    for c in t.colnames:
        col = t[c]
        kind = col.dtype.kind
        if kind in "US" or kind == "O":
            cols[c] = pa.array([None if (hasattr(col, "mask") and np.ma.is_masked(v)) else str(v) for v in col])
            continue
        mask = np.ma.getmaskarray(col) if hasattr(col, "mask") else np.zeros(len(col), bool)
        a = np.asarray(col.data if hasattr(col, "mask") else col)
        if kind == "f":
            a = np.where(mask, np.nan, a.astype(np.float64))
        elif kind in "iu":
            a = np.where(mask, -1, a.astype(np.int64))
        elif kind == "b":
            a = np.where(mask, False, a.astype(bool))
        cols[c] = pa.array(a)
    return pa.table(cols)


def id_table(ids, **extra):
    t = Table({"source_id": np.asarray(ids, dtype=np.int64)})
    for k, v in extra.items():
        t[k] = np.asarray(v)
    return t
