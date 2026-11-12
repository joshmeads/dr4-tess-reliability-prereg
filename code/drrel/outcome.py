"""The three outcomes per item, and the rule for each.

ON_STAR       the named star shows the signal: same-star p <= alpha, and no detected neighbour
              carries more variable flux (amplitude x flux) than the star.
ON_NEIGHBOUR  a Gaia source within 63 arcsec (3 TESS pixels) shows the signal at p <= alpha, with
              an amplitude at least half the minimum it needs to make the TESS signal, and it
              carries the most variable flux of all detected sources.
NOWHERE       neither: the star's power is >= 0.5, and nb_complete says whether every eligible
              neighbour was also tested with power >= 0.5.
UNDECIDED     not detected, and the star's power is below 0.5.
NO_TEST       the star has no usable Gaia series (too few points or valid control windows).

A neighbour is eligible when a full-depth signal on it could make the TESS amplitude:
implied minimum amplitude = TESS amplitude x 10^(0.4 (G_neighbour - G_star)) <= 1 (box) or 0.6 (sine).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import RULES
from .controls import Ephemeris, test


def implied_amp(amp_tess, g_star, g_nb):
    return amp_tess * 10 ** (0.4 * (g_nb - g_star))


def eligible(eph, g_star, g_nb, rules=RULES):
    cap = rules["max_implied_box"] if eph.tier == "box" else rules["max_implied_sine"]
    return implied_amp(eph.amp, g_star, g_nb) <= cap


def classify(star, eph: Ephemeris, neighbours, rules=RULES, with_power=True):
    """star: dict(t, x, s, g). neighbours: list of dict(source_id, g, sep_arcsec, t, x, s) or with
    t None when the neighbour has no epoch photometry. Returns a result dict."""
    res = dict(outcome="NO_TEST")
    st = test(star["t"], star["x"], star["s"], eph, rules=rules, with_power=with_power)
    res["star"] = {k: v for k, v in st.items() if k != "S_controls"}
    if st["status"] == "NO_TEST":
        return res
    detections = []
    if st["status"] == "OK" and st["p"] <= rules["alpha"]:
        detections.append(("star", st["amp"] * 10 ** (-0.4 * star["g"]), None))
    nb_rows, nb_complete = [], True
    elig = sorted((nb for nb in neighbours if nb["sep_arcsec"] <= rules["neighbour_radius_arcsec"]
                   and eligible(eph, star["g"], nb["g"], rules)), key=lambda nb: nb["g"])
    n_elig = len(elig)
    cap = rules["max_implied_box"] if eph.tier == "box" else rules["max_implied_sine"]
    for nb in elig[rules["max_neighbours"]:]:
        if implied_amp(eph.amp, star["g"], nb["g"]) <= rules["untested_plausible"] * cap:
            nb_complete = False
    for nb in elig[: rules["max_neighbours"]]:
        a_min = implied_amp(eph.amp, star["g"], nb["g"])
        row = dict(source_id=nb["source_id"], g=nb["g"], sep_arcsec=nb["sep_arcsec"], amp_min=a_min)
        if nb.get("t") is None or len(nb["t"]) < 20:
            row["status"] = "NO_DATA"; nb_complete = False; nb_rows.append(row); continue
        r = test(nb["t"], nb["x"], nb["s"], replace(eph, amp=a_min), rules=rules, with_power=with_power)
        row.update({k: v for k, v in r.items() if k != "S_controls"})
        if r["status"] == "OK" and r["p"] <= rules["alpha"] and r["amp"] >= rules["neighbour_amp_consistency"] * a_min:
            detections.append(("nb", r["amp"] * 10 ** (-0.4 * nb["g"]), nb["source_id"]))
        if not (r.get("power", 0) >= rules["power_decided"]):
            nb_complete = False
        nb_rows.append(row)
    res.update(neighbours=nb_rows, n_eligible=n_elig, nb_complete=nb_complete)
    if detections:
        who = max(detections, key=lambda d: d[1])
        res["outcome"] = "ON_STAR" if who[0] == "star" else "ON_NEIGHBOUR"
        res["source"] = who[2]
        return res
    pw = st.get("power", np.nan)
    res["outcome"] = "NOWHERE" if (with_power and pw >= rules["power_decided"]) else "UNDECIDED"
    return res
