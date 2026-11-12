"""One focused suite for the sealed scorer: the null is calibrated, power predicts recovery,
the outcome rule places signals on the right source, and the adapter reads what it should."""
import json
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drrel import RULES, nsl  # noqa: E402
from drrel.controls import Ephemeris, control_ratios, method_for, pvalue  # noqa: E402
from drrel.controls import testable as is_testable  # noqa: E402
from drrel.controls import test as run_test  # noqa: E402
from drrel.endpoints import all_endpoints, e2_off_target, e5_period, false_rates, gaia_unequal, purity  # noqa: E402
from drrel.gaia_io import series, tcb_to_tdb  # noqa: E402
from drrel.outcome import classify  # noqa: E402
from drrel.score import score_row  # noqa: E402
from drrel.simulate import _tcb_from_tdb, make_bundle  # noqa: E402

T0 = 2459420.3


def star_series(seed, ra=None, dec=None, g=14.8):
    """Gaia-like noise series. Each star gets its own sky position (and so its own scan times),
    so that the claimed period's coverage is not one coincidence shared by every star."""
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360) if ra is None else ra
    dec = float(np.degrees(np.arcsin(rng.uniform(-0.4, 1.0)))) if dec is None else dec
    t = nsl.transit_times(ra, dec, seed=seed)
    s = np.full(len(t), nsl.sigma_g(g))
    return t, rng.normal(0, s), s, rng


def box(P=0.83, sp=2e-5, D=0.08, a=0.02):
    return Ephemeris("box", P, sp, T0, D, a)


def test_controls_are_disjoint_and_off_alias():
    eph = box()
    r = control_ratios(eph)
    assert len(r) == RULES["n_control"]
    f = 1 / (eph.period * r)
    w = RULES["window_nsigma"] * eph.sigma_p / eph.period ** 2
    assert np.all(np.diff(np.sort(f)) >= 2 * w)
    for a, b in ((1, 1), (2, 3), (3, 2), (4, 3), (3, 4)):
        assert np.min(np.abs(r - a / b)) >= RULES["exclude_ratio_tol"]
    # a window too wide for 41 disjoint controls is untestable, not tested with a broken null
    assert not is_testable(Ephemeris("sine", 1.03, 1e-2, T0, 0.1, 0.01, mode="free"))


def test_method_boundaries_let_every_testable_count_reach_alpha():
    """The method is fixed by the claim's control count and kept for leave-one-out injections,
    and at every testable count both the claim and an injection can reach p <= alpha."""
    assert method_for(200) == "rank" and method_for(199) == "pivot"
    assert method_for(41) == "pivot" and method_for(40) is None
    big = 1e6
    for n in (41, 120, 199, 200):
        m = method_for(n)
        null = np.random.default_rng(n).gumbel(size=n)
        assert pvalue(big, null, m) <= RULES["alpha"]               # the claim, n controls
        assert pvalue(big, null[:-1], m) <= RULES["alpha"]          # an injection, n - 1 controls


@pytest.mark.parametrize("eph", [box(), Ephemeris("sine", 0.41, 1e-5, T0, 0.041, 0.01)])
def test_null_false_recovery_rate_is_near_alpha(eph):
    """On pure noise the claimed window behaves like any control window: p is uniform, so
    recoveries at p <= 0.005 are rare (expect 1 in 200; allow up to 4 in 240)."""
    hits, ps = 0, []
    for k in range(240):
        t, x, s, _ = star_series(k)
        r = run_test(t, x, s, eph, with_power=False)
        assert r["status"] == "OK"
        ps.append(r["p"])
        hits += r["p"] <= RULES["alpha"]
    assert hits <= 4
    assert 0.35 < np.median(ps) < 0.65


def test_gumbel_pivot_is_calibrated_on_wide_windows():
    """Wide LS windows (40-189 disjoint controls) use the studentised Gumbel pivot. On pure noise
    its false recoveries stay near alpha (expect 1 in 200; allow up to 4 in 240)."""
    from drrel.gumbel import fit
    mu, b = fit(np.random.default_rng(0).gumbel(3.0, 2.0, size=20000))
    assert abs(mu - 3.0) < 0.05 and abs(b - 2.0) < 0.05
    eph = Ephemeris("sine", 0.83, 0.0015 * 0.83, T0, 0.083, 0.005, mode="free")
    assert method_for(len(control_ratios(eph))) == "pivot"
    hits = 0
    for k in range(240):
        t, x, s, _ = star_series(5000 + k)
        r = run_test(t, x, s, eph, with_power=False)
        assert r["status"] == "OK" and r["p_method"] == "pivot"
        hits += r["p"] <= RULES["alpha"]
    assert hits <= 4


def test_power_predicts_recovery():
    """Items injected at the claimed period are recovered about as often as the injection power
    (measured at control periods) predicts, including the ephemeris error."""
    pw, hit = [], []
    eph = box(P=1.37, sp=4e-5, D=0.08, a=0.006)
    for k in range(80):
        t, x, s, rng = star_series(1000 + k)
        p_true = eph.period + rng.normal(0, eph.sigma_p)
        from drrel.stat import model
        x = x + model("box", t, p_true, eph.t0, eph.duration, eph.amp)
        r = run_test(t, x, s, eph)
        if r["status"] == "NO_TEST":
            continue
        pw.append(r["power"]); hit.append(r["status"] == "OK" and r["p"] <= RULES["alpha"])
    assert 0.2 < np.mean(pw) < 0.95                  # a regime where the check means something
    assert abs(np.mean(hit) - np.mean(pw)) < 0.15


def _nb(sid, g, sep, t, x, s):
    return dict(source_id=sid, g=g, sep_arcsec=sep, t=t, x=x, s=s)


def test_outcome_places_the_signal_on_the_source_with_most_variable_flux():
    from drrel.stat import model
    eph = box(a=0.02)
    t, x, s, rng = star_series(7)
    tn, xn, sn, _ = star_series(8)
    # the neighbour, 1.5 mag fainter, carries the eclipse at its implied depth (8%); half of its
    # flux leaks into the target's photometry, so the target also shows a weaker copy
    a_n = 0.02 * 10 ** (0.4 * 1.5)
    leak = 0.5 * 10 ** (-0.4 * 1.5) * a_n
    res = classify(dict(t=t, x=x + model("box", t, eph.period, eph.t0, eph.duration, leak), s=s, g=14.0),
                   eph, [_nb(2, 15.5, 15.0, tn, xn + model("box", tn, eph.period, eph.t0, eph.duration, a_n), sn)])
    assert res["outcome"] == "ON_NEIGHBOUR"
    # the same eclipse on the target itself
    res = classify(dict(t=t, x=x + model("box", t, eph.period, eph.t0, eph.duration, 0.02), s=s, g=14.0),
                   eph, [_nb(2, 15.5, 15.0, tn, xn, sn)])
    assert res["outcome"] == "ON_STAR"
    # nothing anywhere, with power
    res = classify(dict(t=t, x=x, s=s, g=14.0), eph, [_nb(2, 15.5, 15.0, tn, xn, sn)])
    assert res["outcome"] == "NOWHERE" and res["nb_complete"]
    # a neighbour too faint to make the TESS depth is not eligible
    res = classify(dict(t=t, x=x, s=s, g=14.0), eph, [_nb(3, 19.0, 15.0, tn, xn, sn)])
    assert res["n_eligible"] == 0
    # 12 plausible neighbours listed, epochs for the 10 brightest only (as fetch_dr4 builds a
    # bundle): the untested two make the neighbour test incomplete
    ten = [_nb(10 + k, 14.2 + 0.01 * k, 20.0, tn, xn, sn) for k in range(10)]
    res = classify(dict(t=t, x=x, s=s, g=14.0), eph, ten)
    assert res["n_eligible"] == 10 and res["nb_complete"]
    extra = [dict(source_id=30 + k, g=14.4 + 0.01 * k, sep_arcsec=30.0) for k in range(2)]
    res = classify(dict(t=t, x=x, s=s, g=14.0), eph, ten + extra)
    assert res["n_eligible"] == 12 and len(res["neighbours"]) == 10 and not res["nb_complete"]


def test_adapter_drops_rejected_rows_combines_ccds_and_converts_time():
    t_tdb = np.linspace(2457000, 2458800, 40)
    tc = _tcb_from_tdb(t_tdb) - RULES["gaia_time_zero_jd"]
    assert np.allclose(tcb_to_tdb(tc + RULES["gaia_time_zero_jd"]), t_tdb, atol=1e-9)
    tab = dict(g_transit_time=np.repeat(tc, 2), g_transit_flux=np.repeat(np.full(40, 1000.0), 2),
               g_transit_flux_error=np.full(80, 2.0), transit_id=np.repeat(np.arange(40), 2),
               rejected_by_variability=np.zeros(80, bool))
    tab["g_transit_flux"][::2] += 10; tab["g_transit_flux"][1::2] -= 10   # two CCD rows per transit
    tab["rejected_by_variability"][-2:] = True
    t, x, s = series(tab)
    assert len(t) == 39 and np.allclose(x, 0, atol=1e-12)
    assert np.all(np.diff(t) > 0) and abs((t_tdb[0] - t[0]) * 86400) < 1e-3


def test_purity_estimator_and_false_rates():
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(4000):                           # true purity 0.4, power 0.7, false rate 0.005
        real = rng.random() < 0.4
        rows.append(dict(outcome="ON_STAR" if (real and rng.random() < 0.7) or rng.random() < 0.005 else "NOWHERE",
                         power=0.7))
    assert abs(purity(rows, 0.005) - 0.4) < 0.03
    sh = [dict(outcome="ON_STAR")] * 5 + [dict(outcome="NOWHERE")] * 995 + [dict(outcome="NO_TEST")] * 50
    assert false_rates(sh)["on"] == pytest.approx(0.005)
    assert gaia_unequal(dict(p2p_diff=0.01, p2p_sigma=0.001, p2p_amp=0.02)) is True
    assert gaia_unequal(dict(p2p_diff=0.0, p2p_sigma=0.01, p2p_amp=0.02)) is None


def test_endpoint_rules():
    """E2 has one population (star power >= 0.5, nb_complete not a condition); E5's spread is
    centred; a withheld E1 carries no number anywhere."""
    on = [dict(outcome="ON_STAR", power=0.9, nb_complete=True, f21=0.1)] * 60
    nb = [dict(outcome="ON_NEIGHBOUR", power=0.9, nb_complete=False, f21=0.1)] * 20
    weak = [dict(outcome="ON_NEIGHBOUR", power=0.3, nb_complete=True, f21=0.1)] * 50
    assert e2_off_target(on + nb + weak, [])["all"]["value"] == pytest.approx(0.25)
    z = np.tile([-0.6, 0.0, 0.6], 30)
    for shift in (0.0, 2.0):
        rows = [dict(outcome="ON_STAR", period=1.0, sigma_p=0.01, p_best=1.0 + 0.01 * (v + shift)) for v in z]
        r = e5_period(rows, [])
        assert r["robust_std"]["value"] == pytest.approx(1.4826 * 0.6, rel=1e-6)
        assert r["median"]["value"] == pytest.approx(shift)
    res = all_endpoints([dict(outcome="ON_STAR", power=1.0, clean_eb=True, g=14.2)], [],
                        [dict(outcome="NOWHERE", power=1.0)] * 30)
    assert res["E1_purity_clean_eb"] == dict(withheld=True, reason=res["E1_purity_clean_eb"]["reason"])
    assert all(v["purity"] == "withheld" for g in res["secondary_descriptive"].values() for v in g.values())


def test_scorer_checks_the_published_seal(tmp_path):
    """The scorer anchors the manifest to the published top hash, then every rule file and the
    predictions file to the manifest. Format adapters may change (reported); a rule edit, a
    predictions edit, or a rule edit with a forged manifest all fail."""
    import hashlib
    import shutil
    import subprocess
    code = tmp_path / "code"
    shutil.copytree(ROOT / "drrel", code / "drrel", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "data").mkdir()
    pred = tmp_path / "data" / "predictions.parquet"
    pq.write_table(pa.table({"row_id": ["i1"]}), pred)

    def manifest():
        files = sorted((code / "drrel").iterdir()) + [pred]
        lines = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(tmp_path).as_posix()}"
                 for f in files if f.is_file()]
        (tmp_path / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
        return hashlib.sha256((tmp_path / "MANIFEST.sha256").read_bytes()).hexdigest()
    top = manifest()
    probe = ("import sys; sys.path.insert(0, sys.argv[1]); from drrel.score import verify_rules; "
             "r = verify_rules(sys.argv[2], sys.argv[3]); "
             "print(r['top']['match'], r['all_match'], all(r['format_adapters_unchanged'].values()))")
    run = lambda: subprocess.run([sys.executable, "-I", "-B", "-c", probe, str(code), top, str(pred)],  # noqa: E731
                                 capture_output=True, text=True, check=True).stdout.split()
    assert run() == ["True", "True", "True"]
    g = code / "drrel" / "gaia_io.py"
    g_orig = g.read_text()
    g.write_text(g_orig + "\n# a format adapter may change after release\n")
    assert run() == ["True", "True", "False"]                 # allowed, and visible
    t = code / "drrel" / "gumbel_table.json"
    t.write_text(t.read_text().replace("\"0.005\": 5.", "\"0.005\": 9.", 1))
    assert run()[:2] == ["True", "False"]                      # a rule file edited
    manifest()                                                 # ... and the manifest forged to match
    assert run()[:2] == ["False", "False"]                     # the published top hash catches it
    t.write_text(t.read_text().replace("\"0.005\": 9.", "\"0.005\": 5.", 1))
    g.write_text(g_orig)
    manifest()
    assert run()[:2] == ["True", "True"]
    pq.write_table(pa.table({"row_id": ["i2"]}), pred)          # the predictions edited
    assert run()[:2] == ["True", "False"]


def test_leak_scanner_flags_every_form_of_an_embargoed_id():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bp", ROOT / "work" / "build_publish.py")
    bp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bp)
    tic, gaia = 123456789, 1234567890123456789          # synthetic, in no catalogue
    ids, coords = {tic, gaia}, {(290.1234, 45.5678)}
    for txt in (f"see TIC {tic} for the dip", f"TIC{tic}", f"tic_{tic}", f"lightcurve of TIC {tic}.",
                f"{tic}.json", f"(TIC-{tic})", f"Gaia DR3 {gaia}.", f"gaia_dr3={gaia}", "ra 290.12341, dec 45.56779",
                "2MASS J" + "0" * 8 + "+" + "0" * 7):    # built at run time: no literal designation in public files
        assert bp.scan_text(txt, ids, coords), txt
    for txt in ("TIC 123456780", "0.4490834434666446", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "ra 290.1300 dec 45.5678"):
        assert not bp.scan_text(txt, ids, coords), txt


def test_fetch_fills_epoch_slots_only_inside_the_neighbour_radius():
    from astropy.table import Table
    from drrel import fetch_dr4
    from drrel.score import sep_arcsec
    seps = [64.0] + [10.0 + 4 * k for k in range(11)]          # one bright source just outside 63"
    cone = Table(dict(source_id=list(range(1, 14)), ra=[290.0] * 13,
                      dec=[45.0] + [45.0 + sp / 3600 for sp in seps],
                      phot_g_mean_mag=[14.0, 13.5] + [14.5 + 0.2 * k for k in range(11)]))
    old = fetch_dr4.tap
    fetch_dr4.tap = lambda cfg, q: Table(dict(dr4_source_id=[1])) if "dr3_source_id" in q else cone
    try:
        src = fetch_dr4.sources_for(dict(xmatch_table="x", source_table="y"),
                                    dict(gaia_dr3=1, ra=290.0, dec=45.0, g=14.0, tier="box", amp=0.02))
    finally:
        fetch_dr4.tap = old
    nb = [n for n in src if n["role"] == "neighbour"]
    assert len(nb) == 11 and all(sep_arcsec(290.0, 45.0, n["ra"], n["dec"]) <= 63.0 for n in nb)
    assert 2 not in [n["source_id"] for n in src[: 1 + RULES["max_neighbours"]]]


def test_pivot_refuses_zero_inflated_controls():
    """A window spanning less than one independent period leaves many control maxima at S <= 0;
    the pivot must not run on them (a calibrated rank test, or NO_TEST)."""
    eph = Ephemeris("box", 1.3, 1e-7, T0, 0.26, 0.01)
    seen = set()
    for k in range(30):
        rng = np.random.default_rng(k)
        t = np.sort(rng.choice(nsl.transit_times(290.0, 45.0, seed=k), 40, replace=False))
        s = np.full(40, nsl.sigma_g(14.5))
        r = run_test(t, rng.normal(0, s), s, eph, with_power=False)
        seen.add(r["p_method"])
        if r["p_method"] == "pivot":
            sc = r["S_controls"]; sc = sc[np.isfinite(sc)]
            assert np.mean(sc <= 0) <= RULES["pivot_max_nonpositive"]
    assert seen & {"rank_extended", "pivot_inapplicable"}


def test_score_row_end_to_end_on_a_bundle(tmp_path):
    rng = np.random.default_rng(11)
    row = dict(row_id="i1", kind="item", item_id=1, gaia_dr3=4242, ra=290.0, dec=45.0, g=14.6, tier="box",
               mode="anchored", period=0.91, sigma_p=2e-5, t0=T0, duration=0.08, amp=0.02,
               call_unequal=False, p_pred=0.6, primary=True, clean_eb=True, kepler_cal=False, f21=0.1)
    make_bundle(tmp_path, row, "on_star", rng, neighbours=[(1.0, 20.0)])
    r = score_row(row, str(tmp_path))
    assert r["outcome"] == "ON_STAR" and r["power"] >= 0.8
    assert abs(r["p_best"] - row["period"]) <= 3 * row["sigma_p"]
    assert score_row(dict(row, row_id="absent"), str(tmp_path))["outcome"] == "MISSING"
