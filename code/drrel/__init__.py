"""drrel: sealed scoring code for the preregistered Gaia DR4 reliability test of TESS FFI signals.

Every number that decides an outcome lives in RULES below. RULES is part of the sealed code;
its SHA-256 is in the seal manifest. Changing it after 2026-12-02 makes the result post hoc.
"""

VERSION = "1.1.1"

RULES = {
    # significance
    "alpha": 0.005,                 # recovery: same-star empirical p <= alpha
    "n_control": 200,               # same-star control windows per source
    # p-value method, fixed by the claim's count n of valid disjoint controls (controls.method_for):
    #   n >= 200: rank p, (1 + #controls >= S) / (1 + n); leave-one-out injections use n - 1 = 199,
    #             whose smallest p, 1/200, still reaches alpha
    #   41 <= n <= 199: studentised Gumbel pivot on S^2 (gumbel.py); leave-one-out keeps >= 40
    #   n < 41: untestable
    "n_rank_min": 200,
    "min_controls_pivot": 41,
    # the pivot models Y = max(S,0)^2 as Gumbel; a window spanning less than a few independent
    # periods leaves many control maxima at S <= 0 (Y = 0), which no Gumbel fits. When more than this
    # share of the claim's valid controls has S <= 0, the pivot does not apply: the controls are
    # rebuilt with up to n_control_extended disjoint windows (narrow windows have room for many), and
    # the exact rank test is used if >= n_rank_min of them are valid; otherwise the row is NO_TEST.
    # The decision uses the controls only, never the claim.
    "pivot_max_nonpositive": 0.02,
    "n_control_extended": 600,
    "control_ratio": (0.6, 1.7),    # control periods as a multiple of the claimed one
    "control_candidates": 4000,     # log-uniform candidate ratios before exclusions
    "exclude_ratios_max_int": 5,    # exclude r near a/b with a, b <= this
    "exclude_ratio_tol": 0.005,     # ... within this plus 2 window half-widths (windows must not touch)
    "alias_harmonics": (0.5, 1.0, 2.0),
    "alias_spin_offsets": (-2, -1, 0, 1, 2),  # multiples of the 4 /d spin frequency
    "alias_prec_k": 10,             # |k| <= this, multiples of the precession frequency
    "f_spin": 4.0,                  # 1/d, Gaia 6 h spin
    "p_prec": 63.12,                # d, Gaia precession period
    "gaia_span_days": 2005.0,       # DR4 time base (2014-07-25 to 2020-01-20)
    # ephemeris window
    "window_nsigma": 3.0,           # period scanned over P +- 3 sigma_P, epoch anchored at TESS t0
    "grid_phase_step_box": 0.25,    # max phase drift between grid periods, in units of the duty cycle
    "grid_phase_step_sine": 0.05,   # ... in cycles, for the sinusoid
    "grid_min": 3,
    "grid_max": 512,
    # mode: the epoch is anchored at the TESS value wherever the anchored period grid fits in
    # grid_max periods (an anchored scan is a subset of the free scan, so it has fewer trials);
    # otherwise the phase is free. Trials are counted as independent periods, 2 w_f dt_max.
    "free_oversample": 5,           # free mode: frequency step = 1 / (5 x data span)
    "free_grid_max": 512,
    "free_phase_per_duty": 4,       # free box mode: epoch positions per duty cycle
    "free_min_in_points": 5,
    # box statistic
    "min_in_points": 3,
    "min_in_cycles": 3,             # in-eclipse points must come from >= 3 distinct cycles
    # power by injection
    "n_inject": 40,                 # control windows used for injections (every 5th)
    "power_decided": 0.5,           # below: UNDECIDED
    "power_strong": 0.8,            # endpoint 1 uses items at or above this
    # neighbours
    "neighbour_radius_arcsec": 63.0,  # 3 TESS pixels
    "neighbour_amp_consistency": 0.5,  # measured amplitude on a neighbour >= 0.5 x implied minimum
    "max_neighbours": 10,           # eligible neighbours tested, smallest implied amplitude first
    "untested_plausible": 0.5,      # an untested eligible neighbour with implied amp <= 0.5 x cap
                                    # leaves the item's neighbour test incomplete
    "max_implied_box": 1.0,         # a neighbour is eligible if its implied minimum depth is <= this
    "max_implied_sine": 0.6,        # ... or semi-amplitude
    # P versus 2P
    "p2p_decidable_rel_sigma": 0.2, # Gaia decides when sigma(odd-even) <= 0.2 x measured depth
    "p2p_unequal_nsigma": 3.0,
    # time
    "gaia_time_zero_jd": 2455197.5, # DR3 epoch photometry times: TCB days since J2010.0
}
