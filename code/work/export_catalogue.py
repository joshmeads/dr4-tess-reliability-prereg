"""Catalogue snapshot: every periodic signal the survey holds, read-only from the ledger.

A signal is a ledger peak with is_signal=1 from a real-sky source: 'survey' (production),
'screen' and 'census' (imported legacy lanes). 'control' peaks are planted injections and
'audit' peaks are CPU re-runs of production stars; both are excluded.

Also writes the quiet-star pool: production-searched stars with no significant peak of any
kind in any searched sector. Shuffled-ephemeris controls and DR3 comparison stars come from it.

survey.db is opened with mode=ro. Output: data/signals.parquet, data/quiet_stars.parquet,
data/searched_stars.parquet.
"""
import json
import pathlib
import sqlite3
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from paths import survey_state  # noqa: E402

STATE = str(survey_state())    # the survey state directory, from DR4REL_SURVEY_STATE
OUT = pathlib.Path(__file__).resolve().parents[1] / "data"
FEATURES = ("duty", "single_event_share", "gap_fraction", "odd_even_sigma", "odd_even_fraction",
            "secondary_depth", "secondary_sigma", "shape_ratio", "sine_p", "sine_half_p",
            "local_depth", "noise_2h", "spacecraft_ratio", "amplitude", "g")


def main():
    OUT.mkdir(exist_ok=True)
    con = sqlite3.connect(f"file:{STATE}/survey.db?mode=ro", uri=True)
    con.execute("PRAGMA query_only=1")

    pool = {}
    for pid, pool_id, status, reason, ntried, ncoin, pch, spike, flags in con.execute(
            "SELECT peak_id, pool_id, status, reason, n_tried, n_coincident, p_chance, spike_id, flags "
            "FROM pool_verdict ORDER BY peak_id, pool_id"):
        pool[pid] = (pool_id, status, reason, ntried, ncoin, pch, spike, flags)  # latest pool run wins
    phen_of = dict(con.execute("SELECT peak_id, phen_id FROM phen_peak"))
    items = {}
    for phen, queue, track, conf, state, fv, reasons in con.execute(
            "SELECT phen_id, queue, track, confidence, state, final_verdict, reasons_json FROM item"):
        hf = None
        try:
            hf = json.loads(reasons or "{}").get("hyp_factor")
        except ValueError:
            pass
        items.setdefault(phen, {})[queue] = (track, conf, state, fv, hf)

    cols = {k: [] for k in (
        "signal_id", "source", "tier", "method", "star_id", "gaia_dr3", "tic", "ra", "dec", "gmag",
        "sector", "camera", "ccd", "period", "t0_bjd", "duration", "depth", "n_events", "sde", "fap",
        "snr", "track", "lc_ref", "pool_id", "pool_status", "pool_reason", "pool_n_tried",
        "pool_n_coincident", "pool_p_chance", "pool_spike_id", "pool_flags", "phen_id",
        "eb_track", "eb_confidence", "eb_state", "eb_final", "eb_hyp_factor",
        "var_confidence", "var_state", "var_final", "planet_confidence", "planet_final")}
    for f in FEATURES:
        cols["f_" + f] = []

    q = ("SELECT p.peak_id, p.source, p.method, p.star_id, s.gaia_dr3, s.tic, s.ra, s.dec, s.gmag, "
         "p.sector, p.camera, p.ccd, p.period, p.t0, p.duration, p.depth, p.n_events, p.sde, p.fap, "
         "p.snr, p.track, p.lc_ref, p.features_json FROM peak p JOIN star s USING(star_id) "
         "WHERE p.is_signal=1 AND p.source IN ('survey','screen','census') ORDER BY p.peak_id")
    for r in con.execute(q):
        (pid, src, meth, sid, gaia, tic, ra, dec, g, sec, cam, ccd, per, t0, dur, dep, nev, sde, fap,
         snr, track, lcref, fj) = r
        cols["signal_id"].append(pid); cols["source"].append(src)
        cols["tier"].append("box" if meth == "bls" else "ls"); cols["method"].append(meth)
        for k, v in (("star_id", sid), ("gaia_dr3", gaia), ("tic", tic), ("ra", ra), ("dec", dec),
                     ("gmag", g), ("sector", sec), ("camera", cam), ("ccd", ccd), ("period", per),
                     ("t0_bjd", t0), ("duration", dur), ("depth", dep), ("n_events", nev), ("sde", sde),
                     ("fap", fap), ("snr", snr), ("track", track), ("lc_ref", lcref)):
            cols[k].append(v)
        pv = pool.get(pid, (None,) * 8)
        for k, v in zip(("pool_id", "pool_status", "pool_reason", "pool_n_tried", "pool_n_coincident",
                         "pool_p_chance", "pool_spike_id", "pool_flags"), pv):
            cols[k].append(v)
        phen = phen_of.get(pid)
        cols["phen_id"].append(phen)
        it = items.get(phen, {})
        eb = it.get("eb", (None,) * 5); var = it.get("variable", (None,) * 5); pl = it.get("planet", (None,) * 5)
        cols["eb_track"].append(eb[0]); cols["eb_confidence"].append(eb[1]); cols["eb_state"].append(eb[2])
        cols["eb_final"].append(eb[3]); cols["eb_hyp_factor"].append(eb[4])
        cols["var_confidence"].append(var[1]); cols["var_state"].append(var[2]); cols["var_final"].append(var[3])
        cols["planet_confidence"].append(pl[1]); cols["planet_final"].append(pl[3])
        feat = json.loads(fj) if fj else {}
        for f in FEATURES:
            v = feat.get(f)
            cols["f_" + f].append(float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None)
    tab = pa.table(cols)
    pq.write_table(tab, OUT / "signals.parquet", compression="zstd")
    print("signals:", tab.num_rows)

    # Searched production stars, their sectors, and whether they carry any significant peak.
    searched = {}
    for sid, sec, cam, ccd in con.execute(
            "SELECT i.star_id, i.sector, i.camera, i.ccd FROM job j JOIN input i USING(input_id) "
            "WHERE j.state='DONE' AND j.search='screen' AND j.kind='production'"):
        searched.setdefault(sid, []).append((sec, cam, ccd))
    with_peak = {sid for (sid,) in con.execute("SELECT DISTINCT star_id FROM peak WHERE source IN ('survey','screen','census')")}
    with_signal = {sid for (sid,) in con.execute("SELECT DISTINCT star_id FROM peak WHERE is_signal=1 AND source IN ('survey','screen','census')")}
    star = {r[0]: r[1:] for r in con.execute("SELECT star_id, gaia_dr3, tic, ra, dec, gmag FROM star")}
    ids = sorted(searched)
    st = pa.table({
        "star_id": ids,
        "gaia_dr3": [star[i][0] for i in ids], "tic": [star[i][1] for i in ids],
        "ra": [star[i][2] for i in ids], "dec": [star[i][3] for i in ids], "gmag": [star[i][4] for i in ids],
        "sectors": [",".join(str(s[0]) for s in sorted(set(searched[i]))) for i in ids],
        "n_sectors": [len({s[0] for s in searched[i]}) for i in ids],
        "has_peak": [i in with_peak for i in ids], "has_signal": [i in with_signal for i in ids],
    })
    pq.write_table(st, OUT / "searched_stars.parquet", compression="zstd")
    quiet = st.filter(pc.invert(st["has_peak"]))
    pq.write_table(quiet, OUT / "quiet_stars.parquet", compression="zstd")
    print("searched production stars:", st.num_rows, "with signal:", sum(st["has_signal"].to_pylist()),
          "quiet (no peak at all):", quiet.num_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
