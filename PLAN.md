# Preregistered analysis plan: the reliability of faint TESS FFI periodic signals, tested with Gaia DR4

Author: Josh Meads. Version 1.1.1, 2026-11-12. Written before the Gaia DR4 release (planned 2026-12-02 12:00 CET).

Every rule below is implemented in `code/drrel/` and fixed by the SHA-256 manifest of this record.
Every number that decides an outcome is in `drrel/__init__.py` (`RULES`) or in the two sealed
tables next to it (`gumbel_table.json`, `period_errors.json`). A change to any of them, to any
statistic or to any selection after 2026-12-02 makes the affected result post hoc, and the report
will say so. Before scoring, the scorer checks every rule file against `MANIFEST.sha256` and
prints the result.

**Revision history.** Version 1.0.0 was sealed locally on 2026-11-11 and never published. An
internal adversarial review then found faults, and version 1.1.0 corrects them before any DR4
data exist:

- one definition of E2;
- control-count boundaries that let every testable item reach α;
- neighbour bookkeeping;
- reproducibility inputs;
- an embargo leak in an exploratory script, which is now removed and guarded against;
- a heavy-tailed period-error model with a calibration/hold-out split;
- the E1 prediction population;
- withheld results;
- the robust-spread definition;
- rule-file verification;
- text.

Version 1.1.0 was then reviewed again and corrected, as version 1.1.1, before publication:

- a calibrated test for very narrow box windows, whose zero-inflated controls broke the Gumbel pivot;
- a scorer check anchored to the published top-level hash;
- an executable rebuild script;
- the neighbour radius applied before epoch slots are chosen;
- a stricter embargo scan;
- text.

No prediction was changed to fit any result; there are no DR4 results.

## 1. Question

What fraction of faint (G 13.5–15.5) periodic signals found in single TESS full-frame-image sectors
are real and on the named star? What fraction come from a neighbour? And what fraction are nowhere,
meaning instrumental or noise?

A TESS pixel is 21″ across. Gaia resolves the sky at about 0.4″ and takes 50–150 photometric epochs
per source over 66 months. So DR4 can decide, source by source, where a TESS signal lives.

## 2. The catalogue (frozen, embargoed, hashed)

`data/catalogue.parquet` holds every periodic signal of the survey's ledger at the snapshot of 2026-11-11. The ledger is the survey's private SQLite database, opened read-only. The survey searched TGLC light curves of TESS sectors 14, 15, 40, 41, 54 and 55, plus imported lanes, with box least squares (BLS) and Lomb–Scargle (LS).

- A signal is a ledger peak with `is_signal = 1` from a real-sky source: `survey` (production), `screen` or `census` (imported lanes).
- Planted injections (`control`) and CPU re-runs (`audit`) are excluded.
- Snapshot totals: 574,193 signals on 479,356 stars. There are 1,634,785 searched stars, and 475,354 of them have no significant peak at all.
- Each row holds:
  - the tier: box (BLS) or LS;
  - the period, epoch, duration, depth or amplitude, and the significance;
  - the star (Gaia DR3 ID, position, G);
  - the pool verdict and flags, the queue routing, and the stored vetting features.

The ledger stores one peak per method per light curve. The survey's design document says five per search, but the stored data hold one, so second periods are not in the catalogue.

`data/items.parquet` groups signals into test items: one per star, tier and period. Two signals are the same item when their periods agree within 3 combined σ. There are 541,609 items. Each carries the following.

- **Refined ephemeris.** Each signal's period and epoch are re-fitted on its stored TESS light curve (`work/refine_ephem.py`; the results are frozen in `data/inputs/ephem_refined.parquet`):
  - box: the box SNR profiled over a fine (P, t₀) grid;
  - LS: a free-phase sinusoid fit over a fine frequency grid.

  σ_P is the half-width of the region where the χ² improvement is within β² of its maximum, where β is the 2-hour red-noise factor. The epoch is moved to the event nearest the sector's middle.
- **Two-sector combination.** Where two sectors give an unambiguous cycle count (σ of the count < 0.1), the period is the epoch difference divided by the count. This applies to 7,392 items. The anchor epoch is the member sector nearest Gaia's time span.
- **Calibration of σ_P against VSX, with a held-out half.** Catalogued eclipsing binaries are those with a VSX object within 2″ whose period matches at P, P/2 or 2P within 1%. They are split in two by a sealed hash: SHA-256 of `"sigma-split-20261202:" + Gaia DR3 ID`, with an even first byte meaning the calibration half.
  - On the calibration half only, σ_P is multiplied by the robust scatter (1.4826 × MAD about the median) of (P_TESS − h P_VSX)/σ_P when that exceeds 1. This is done per tier, separately for single-sector and combined items.
  - A fit that hit its grid edge gets a further factor of 3.

  | Group | Calibration n | Factor | Held-out robust scatter after | Held-out share beyond ±3σ |
  |---|---:|---:|---:|---:|
  | box, single sector | 1,674 | 7.71 | 0.99 | 8.9% |
  | LS, single sector | 1,848 | 1.36 | 1.00 | 9.5% |
  | box, combined | 97 | 1.22 | 0.82 | 11.7% |
  | LS, combined | 122 | 1.94 | 1.05 | 15.7% |

  The errors are heavy-tailed. Their empirical distribution on the calibration half, per group, is sealed as quantiles in `drrel/period_errors.json` and drives the power model (section 8.4). The held-out half is used only to test: the positive controls (section 8.6) and endpoint 5. See `results/sigma_calibration.json`.

## 3. Information used before sealing (the test is not blind to Gaia DR3)

The predictions use Gaia DR3 catalogue columns:

- `phot_g_n_obs`, `phot_g_mean_flux_over_error`, `ipd_frac_multi_peak`, `ruwe` and `matched_transits`;
- `has_epoch_photometry`, `phot_variable_flag`, `ecl_lat`, `l` and `b`.

These come from IRSA `gaia_dr3_source` for the signal stars and for 150,000 seeded quiet stars; they are frozen in `data/inputs/dr3_columns.parquet`. The DR3 excess-scatter purity estimate (`results/purity_dr3.json`, section 7) enters each item's probability.

DR3 epoch photometry was not used. Stars with public DR3 epoch photometry are flagged, so that the results can also be reported without them.

## 4. Kepler calibration (information used)

For a stratified, seeded sample of items on Kepler targets, two long-cadence quarters per target were streamed from the public MAST bucket, reduced to a few numbers and discarded (`work/kepler_calib.py`). The labels are frozen in `data/inputs/kepler_labels.parquet`.

- **Data.** PDCSAP flux with quality 0, conditioned as the TESS search was.
- **Test.** drrel's same-star test with a free phase across the TESS period window. Labelling uses 100 controls at 0.4–2.5 P and α = 0.01.
- **Label.** `real_on` = detected, with a Kepler amplitude of at least 0.3 × the TESS amplitude.

Kepler's 4″ pixels make that label "on this star, or within a few arcsec of it". Quiet Kepler stars given borrowed ephemerides measure the false-label rate. Items in the Kepler sample are flagged `kepler_cal` and are excluded from endpoint 4. Labels map to items by star, tier and period.

**Known limitation of the labels.** The labels were computed with the labelling rule's rank test from 95 valid controls. Its smallest p, 1/(n + 1), is above α = 0.01 at 95–98 controls.
- So the 137 labelled rows with 95–98 valid controls could never be labelled detected.
- 7 of them sat at that floor.
- One, a box item with a Kepler amplitude of 0.084 against a threshold of 0.022, would otherwise have been real and on the star.

That is about one positive label of 477. The labels are frozen inputs and were not recomputed.

## 5. Sealed per-item predictions (`data/predictions.parquet`, embargoed)

Each row holds:

- `row_id` and `kind`:
  - `item`: a sealed prediction;
  - `shuffled`: a quiet star given an item's ephemeris;
  - `positive`: a held-out catalogued eclipsing binary, with `subsample` 1, 0.5 or 0.3;
- the Gaia DR3 ID, position and G;
- the tier;
- the mode: `anchored` or `free` (section 8.2);
- P, σ_P, t₀ (BJD_TDB), duration, and the predicted amplitude (box depth, or sinusoid semi-amplitude, as measured by TESS);
- `err_group`, the period-error group;
- `call_unequal`: the sealed P-versus-2P call, true when TESS odd/even ≥ 3σ;
- `p_pred`: the sealed probability that the signal is real and on the star;
- `primary`, `clean_eb`, `kepler_cal` and `sigma_cal`;
- strata: `dr3_epoch_public` and `sigma_rel_bin`;
- `power_coarse`: planning only, never used to score.

Untestable items get no row.

`p_pred` comes from an L2-regularised logistic model fitted to the Kepler labels. Its features are:

- tier, log amplitude, log refined SNR, G and log period;
- pool released or vetoed, and the common-mode suspect flag;
- whether the period is within 3% of an alias centre;
- whether the item spans more than one sector;
- the DR3 log likelihood ratio;
- the resolved-companion flag.

The coefficients and their cross-validated skill are in `results/kepler_calibration.json`.

**Primary sample (seeded, 20261202).** It contains:

- every testable box item;
- every testable item routed to the eb or variable queue;
- 30,000 other testable LS items.

The primary sample is the declared scoring set. Non-primary testable items are sealed too, and may be scored as secondary.

**Clean eb tier.** An item with an eb queue entry, a pool verdict of RELEASED and no `cm_suspect` flag.

## 6. Three outcomes per item, and the rule for each

All outcomes come from `drrel/outcome.py`. All p-values are same-star empirical p-values (section 8).

- **ON_STAR.** The named star's p ≤ α = 0.005, and no detected neighbour carries more variable flux (amplitude × flux) than the star.
- **ON_NEIGHBOUR.** Three conditions hold:
  - a Gaia DR4 source within 63″ (3 TESS pixels) has p ≤ 0.005;
  - its amplitude is at least 0.5 × the minimum it would need to make the TESS signal;
  - it carries the most variable flux of all detected sources.

  The minimum amplitude is the TESS amplitude × 10^(0.4 (G_nb − G_star)). A neighbour is eligible only if that minimum is ≤ 1 (box) or ≤ 0.6 (sinusoid).

  Every eligible neighbour is listed in the bundle, but epochs are fetched for at most 10, smallest implied amplitude (brightest) first. The 63″ radius is measured from the DR4 target, with the scorer's own separation, before this ranking. So no epoch slot goes to a source the scorer would drop. An untested eligible neighbour whose implied amplitude is ≤ 0.5 × the cap makes the neighbour test incomplete.
- **NOWHERE.** Not detected anywhere, and the star's power is ≥ 0.5. `nb_complete` records whether every eligible neighbour was tested with power ≥ 0.5, with no untested plausible neighbour.
- **UNDECIDED.** Not detected, and power < 0.5.
- **NO_TEST.** No usable Gaia series, or too few valid control windows.

DR4 identities are fixed by rule (`drrel/fetch_dr4.py`):

- the target is the official DR3–DR4 crossmatch best match; failing that, the nearest DR4 source within 0.5″ with |ΔG| < 0.3;
- neighbours are taken from DR4's own source table.

## 7. Pre-DR4 purity estimate from DR3 (results/purity_dr3.json)

For each star, A = √(phot_g_n_obs) / phot_g_mean_flux_over_error. This is the per-CCD fractional scatter.

Quiet stars, stratified by G (0.1 mag) and by the resolved-companion flag, give the reference distribution Q. There are 197,299 of them: the 150,000-star seeded sample plus the quiet stars that fell in the first, all-star DR3 batches.

A real on-star signal adds its rms in quadrature:

- box: depth · √(q(1 − q)), where q is the duty cycle;
- sinusoid: semi-amplitude / √2.

That gives V_i. Each tier is fitted as (1 − f) Q + f V by maximum likelihood, with a bootstrap interval.

| Tier | f (DR3) | 95% | n |
|---|---:|---|---:|
| Box, all | 0.211 | 0.204–0.216 | 22,438 |
| Box, pool released | 0.236 | 0.228–0.244 | 17,226 |
| Box, pool vetoed | 0.066 | 0.050–0.082 | 1,969 |
| LS, all | 0.019 | 0.019–0.020 | 513,076 |
| LS, pool released | 0.039 | 0.037–0.040 | 181,909 |
| LS, pool vetoed | 0.003 | 0.002–0.004 | 262,323 |
| Clean eb tier | 0.126 | 0.121–0.131 | 37,416 |
| Variable queue | 0.086 | 0.083–0.090 | 48,401 |
| Alias-window track (periods near 1/3, 1/2, 2/3, 1, 2, 3 d) | 0.007 | 0.006–0.008 | 335,065 |
| Off-window track | 0.065 | 0.063–0.067 | 200,449 |

Two validations run inside the same fit:

- Catalogued eclipsing binaries give f = 0.836 (0.825–0.846, n = 6,090). This is the method's sensitivity.
- Quiet stars given borrowed signals give f = 0.000 (n = 20,000).

Dividing f by 0.836 gives the sensitivity-corrected purity. f is the fraction "real and on the star"; a signal on a neighbour counts as not on the star.

## 8. The test

### 8.1 Statistic (drrel/stat.py)

Both statistics share one setup. The epoch is anchored at the TESS t₀, and the period is scanned over P ± 3σ_P only.

- **Box tier.** A matched-filter box of the TESS duration. S = (weighted mean outside − weighted mean inside) / σ, where σ is inflated by the series' robust scatter. The secondary box at phase 0.5 is left out of the baseline. A trial needs ≥ 3 in-box points from ≥ 3 distinct cycles.
- **Sinusoid tier.** x = c − A cos(2π (t − t₀)/P), and S = A/σ_A, one-sided.
- **Grid spacing.** The grid keeps the phase drift between neighbouring trial periods at the farthest epoch below 0.25 duty cycles (box) or 0.05 cycles (sinusoid).
- **Trials.** The count is reported as independent periods across the window at the farthest DR4 epoch, 2 w_f Δt_max. The median is 7.6 for box items (IQR 5.1–11.1) and 29 for LS items (IQR 22–37).

### 8.2 Mode (sealed per item)

- **anchored**: when the anchored grid needs ≤ 512 periods.
- **free**: otherwise, with the phase left free:
  - free-phase sinusoid S = √(Δχ²);
  - box at 4 epoch positions per duty cycle;
  - frequency step 1/(5 × data span).
- **untestable**: when the window is too wide for 41 disjoint control windows (section 8.3). An untestable item has no prediction row.

### 8.3 Same-star controls and p-value (drrel/controls.py)

Control windows sit at periods r·P, with r between 0.6 and 1.7.

- **Candidates.** 4,000 log-uniform ratios.
- **Exclusions.** Ratios within 0.005 + 2 w/P of any a/b with a, b ≤ 5 are excluded, so that no control window touches a harmonic window. So are control frequencies within 2/T + 2 w_f of the Gaia alias comb of the claimed frequency:
  - harmonics 1/2, 1 and 2;
  - spin offsets of 0, ±4 and ±8 per day;
  - precession offsets k/63.12 per day with |k| ≤ 10. The precession comb is dropped when the windows are wider than 0.8 of its spacing, because it cannot then be avoided.
- **Disjointness.** Every control window has the claim's width in frequency and the same grid. Windows must not overlap one another: overlapping windows share maxima, and in testing on pure noise they produced false recoveries.
- **Selection.** Up to 200 disjoint windows are kept, evenly spaced.

The p-value method is fixed by n, the claim's count of valid controls. The same method is used for the claim (n controls) and for every leave-one-out power injection (n − 1 controls), so at every testable n both can reach α:

- **n = 200.** The rank p-value: p = (1 + #controls with S ≥ S_claim)/(1 + n). Leave-one-out uses 199 controls, whose smallest p, 1/200, still reaches α.
- **41 ≤ n ≤ 199.** A studentised Gumbel pivot on Y = max(S, 0)² (`drrel/gumbel.py`). Fit (μ, β) by maximum likelihood to the control values, form T = (Y_claim − μ)/β, and recover when T ≥ c(n, 0.005).
  - c is read from a sealed Monte Carlo table: `drrel/gumbel_table.json`, seed 20261202, 10⁵ simulations per n from 40 to 190, smoothed as c∞ + a/n and used up to n = 200.
  - Because Gumbel is a location-scale family, T's null distribution depends on n only, so the table removes the small-sample bias.
  - No free-shape extreme-value fit is used.
  - Leave-one-out keeps at least 40 controls, the table's smallest n.
- **Zero-inflated controls.** If the pivot would apply but more than 2% of the claim's valid controls have S ≤ 0, Y is zero-inflated and no Gumbel fits it. This happens when the window spans less than a few independent periods.
  - The controls are rebuilt with up to 600 disjoint windows, which narrow windows have room for.
  - If at least 200 of them are valid, the exact rank test is used on all of them (`rank_extended`); otherwise the row is NO_TEST (`pivot_inapplicable`).
  - The decision uses the controls only, never the claim, and the injections keep the claim's method and control set.
- **n < 41.** Untestable. If the scorer finds fewer valid controls in the data, the row is NO_TEST.

The pivot follows a recommendation made on 2026-11-11 by an AI model (Claude Fable 5.2, consulted on this one question). It was made after the measured LS period errors showed that only 7% of LS items could fit 190 disjoint controls. The model preferred it to widening the control range, because a wider range stretches across the ratios where exclusions are widest and noise is least exchangeable.

Pre-seal checks, on simulated Gaia sampling with the final code:

- **Wide LS windows** (`results/validation/pivot_check.json`). The pivot falsely recovered 3 of 712 with white noise (0.0042) and 0 of 667 with red noise plus a 63-day term, against α = 0.005.
- **Stress test.** Narrow box windows reduced to 40 controls, the pivot's smallest n:
  - the pivot falsely recovered 9 of 731 (0.012), against 4 of 731 for the rank test on the full set;
  - on real signals the pivot recovered 526 of 731 and the rank test 516.

  So with only 40 controls of the box statistic, the pivot runs above α.
- **Box items in their natural setting.** 1,200 sealed box ephemerides on pure noise:
  - 86% used the rank test (3 false of 1,026);
  - 7% used the pivot, with more than 40 controls (0 false of 85);
  - 7% were NO_TEST.

- **Very narrow box windows** (`results/validation/narrow_window_check.json`), the case found in review: the narrowest-window box ephemeris, with 40 sampled transits.
  - Before the zero-inflation rule, the pivot falsely recovered 38 of 879 (4.3%).
  - After it, the extended rank test falsely recovered 4 of 872 (0.0046).
  - On all 351 primary box items whose window spans less than one independent period, with natural sampling, there were 0 false recoveries:

    | Method | Items |
    |---|---:|
    | Rank | 269 |
    | Pivot | 19 |
    | Extended rank | 54 |
    | NO_TEST | 6 (1.7%) |

  - On 1,500 random primary items the rule fired once, so the endpoint populations are otherwise unchanged.

Check (a) measures the false rate of the whole pipeline on real DR4 data. That measured rate, not α, is used in every correction.

### 8.4 Power by injection

Power is the probability of recovery if the claimed signal is real, computed the same way for every source:

1. The fitted claimed signal is subtracted from the series. For the box tier, the fitted depth at the best period is added back inside the box; for the sinusoid tier, the fitted sinusoid is subtracted.
2. The predicted signal at the claimed period is added to that series, and the control maxima are computed on the result. This is the null for the injections: it carries the leakage a real signal would put into the control windows. For a real source it is close to the claim's own null; for a non-real source it supplies the same handicap.
3. The predicted signal is injected into the windows at positions 0, k, 2k, … (k = ⌊n/40⌋, at most 40 injections). Here n is the number of control windows, and windows whose maxima are not finite are skipped. With 200 controls these are every 5th window across the whole ratio range; with fewer than 80 they are the first 40, at the lowest ratios. Each injection goes into the subtracted series, at the window's central period plus z_k σ_P (scaled to the window). The z_k are the (k + ½)/m quantiles of the sealed empirical period-error distribution of the item's group, where m is the number of injections made. Power therefore includes the chance that the true period lies outside the ±3σ window.
4. Each injection is judged against the step-2 null without its own window (leave-one-out), by the claim's method.

Earlier variants failed in simulation:

- **Injections judged against a null without signal leakage** overestimated power for real signals.
- **Injections judged against the observed data's null** made power differ between real and non-real sources, and biased purity low by 6%.
- **Gaussian period offsets** ignored the 9–16% of real errors beyond 3σ.

### 8.5 Shuffled-ephemeris controls

For a seeded quarter of the primary items, capped at 15,000 (14,741 were matched; 17.7% of the primary sample), a quiet star takes that item's ephemeris and amplitude. A quiet star has no significant peak in any searched sector. It is matched on:

- G, in 0.25-mag bins;
- ecliptic latitude, in 10° bins;
- |b|, in 5° bins;
- the resolved-companion flag.

It goes through the identical pipeline, neighbours included. Its outcome rates are the false ON_STAR and false ON_NEIGHBOUR rates.

### 8.6 Positive controls

Testable items that match a catalogued eclipsing binary (EA, EB, EW or E) in the held-out half are scored as `positive` rows: 3,038 binaries.

Each is scored three times: on all its Gaia epochs, and on a deterministic 50% and a 30% subsample (seeded by the row id). The subsamples put real binaries at intermediate power, which check (b) needs to test the power model across its range.

## 9. Endpoints: five primary, and no others

All five come from `drrel/endpoints.py`, with bootstrap 95% intervals (2,000 resamples, seed 20261202).

1. **E1. Purity of the clean eb tier at power ≥ 0.8, by G bin** (13.5–14.0–14.5–15.0–15.6).
   - Population: primary clean-eb items, Kepler-labelled ones included, with measured power ≥ 0.8.
   - Estimate: (R − n α̂) / (Σ power × (1 − α̂)), where R is the count of ON_STAR outcomes.
2. **E2. Off-target fraction by crowding.** This is ON_NEIGHBOUR / (ON_STAR + ON_NEIGHBOUR), each count corrected by n × its false rate on shuffled controls.
   - Population: tested items whose star has power ≥ 0.5. This is the only population.
   - `nb_complete` is deliberately not a condition. It turns false when any neighbour's power at its minimum implied amplitude is below 0.5, and a signal that really sits on a neighbour usually has more than that minimum. Filtering on it removes true off-target detections: in the end-to-end simulation it gave 0.175 against a true 0.266.
   - Crowding is f21, the DR4 flux of the bundle's neighbours within 21″ relative to the star, in bins [0, 0.05), [0.05, 0.2), [0.2, 1), [1, ∞).
3. **E3. Accuracy of the P-versus-2P call** where Gaia decides. Gaia decides when σ(odd − even) ≤ 0.2 × the measured amplitude; Gaia calls "unequal" at ≥ 3σ.
4. **E4. Calibration of `p_pred`.** The forecast of the observed outcome is q = p_pred · power + (1 − p_pred · power) α̂. It is scored with the Brier score, the Brier skill score against climatology, the log score and a 10-bin reliability diagram. Items: primary, not `kepler_cal`, power ≥ 0.5.
5. **E5. Period accuracy.** The distribution of (P_Gaia − P_TESS)/σ_P for ON_STAR items that were not in the σ_P calibration half.
   - Spread: the robust standard deviation, 1.4826 × the MAD about the median.
   - Shift: the median, reported separately.
   - Also the fractions within 1, 2 and 3, and the fraction at the window edge.

**Validity checks**, fixed in advance:

- **(a)** The false ON_STAR rate on shuffled controls must be ≤ 2α = 0.01. Otherwise the result is reported as miscalibrated, with the measured rate used in the corrections.
- **(b)** On positive controls with power ≥ 0.5, observed/predicted recovery must lie in [0.8, 1.15], and in every power bin ([0.5, 0.8), [0.8, 0.95), [0.95, 1]) with n ≥ 30 the observed rate must be ≥ power − 0.2.

  If (b) fails, E1 is withheld: only the flag and the reason are reported, with no number, and the descriptive purity numbers are withheld too.

**Secondary, descriptive only (not endpoints):**

- purity and off-target fractions per tier stratum, σ_P/P bin, and DR3-epoch-public flag;
- the diurnal blind spot of ground catalogues;
- the false-veto rate of the common-mode rule.

## 10. Predicted values (results/endpoint_predictions.json)

The sample (`results/sample_summary.json`):

- 541,609 items;
- 349,314 testable: 143,682 anchored and 205,632 free;
  - box: 22,326 testable of 25,116;
  - LS: 326,988 testable of 516,493;
- primary: 83,059 (box 22,326; LS 60,733), of which 27,321 are clean eb;
- 14,741 shuffled controls;
- 3,038 held-out positive binaries, in 9,114 rows.

Predictions:

- **E1. Clean eb tier purity at power ≥ 0.8**, over the E1 population. The basis is the power-weighted mean of p_pred, Σ p_pred·power / Σ power, which is the expectation of the estimator. Before DR4 the coarse power stands in for the measured power.

  | G | Predicted | 95% | n (coarse power ≥ 0.8) |
  |---|---:|---|---:|
  | all | 0.151 | 0.132–0.169 | 27,256 |
  | 13.5–14.0 | 0.294 | 0.254–0.330 | 2,310 |
  | 14.0–14.5 | 0.209 | 0.183–0.235 | 4,222 |
  | 14.5–15.0 | 0.152 | 0.133–0.170 | 7,738 |
  | 15.0–15.6 | 0.106 | 0.088–0.124 | 12,986 |

  The intervals cover only the calibration model's sampling error, not the shift from the Kepler field to the rest of the sky. An independent cross-check is the DR3 estimate for the same tier: 0.126 / 0.836 = 0.151 after the sensitivity correction.
- **E2. Off-target fraction.** Predicted 0.23, with a 95% interval of 0.10–0.40, rising monotonically across the four crowding bins. The basis is weak: the survey's identity stage, which matches each eb item against catalogued variables nearby, closed 393 of 1,707 examined items (23%) as off-target (`results/identity_counts.json`).
- **E3. P-versus-2P accuracy.** Predicted 0.86, from the agreement of the sealed call with Kepler's own odd/even on 641 Kepler-decided items.
- **E4. Calibration.**
  - A Brier skill score above 0, with a 95% interval excluding 0.
  - If p_pred is calibrated, the expected Brier score is 0.043.
  - On the Kepler labels, 5-fold cross-validation gives Brier 0.044 against a climatology of 0.122 (skill 0.64).
- **E5. Period accuracy** (held-out items only).
  - The robust standard deviation of (P_Gaia − P_TESS)/σ_P lies in [0.8, 1.4].
  - The median lies in [−0.5, 0.5].
  - At most 15% of recovered items sit at the window edge.
- **Validity checks.** The shuffled-control false ON_STAR rate is ≤ 0.01. Positive controls pass rule (b).

**Kepler calibration.** 3,957 planned rows on 3,745 Kepler targets (two long-cadence quarters each, streamed and discarded) gave 3,369 labelled items and 484 quiet controls.

- 477 labelled items are real and on the star: 30% of box items and 2.4% of LS items. The DR3 sensitivity-corrected purity is 25% for box and 2.3% for LS.
- Catalogued eclipsing binaries in the held-out half: 96.6% are real and on the star (n = 145, mean predicted power 0.98). In the calibration half: 96.8% (n = 154).
- Uncatalogued signals: 6.1% (n = 3,070).
- Quiet stars with borrowed ephemerides: 0.0% (n = 484).

See `results/validation/positive_controls_kepler.json`.

## 11. Validation before sealing

**End-to-end simulation** (`results/validation/e2e.json`, run with `work/validate_e2e.py 2000 3`). The DR4-like data are:

- 2,000 items with known truth: 45% on the star, 15% on a neighbour, 40% nowhere;
- shuffled controls;
- positive controls, at full, 50% and 30% of their epochs;
- scanning-law time stamps, G-dependent errors and 1% outliers.

Each true period is P + z σ_P, with z drawn from the held-out half's empirical errors, which are heavy-tailed. The sealed injections use the calibration half's quantiles, so the check is not circular. Everything goes through the sealed CLI (`drrel.score`).

| Quantity | Estimated | True |
|---|---|---|
| Purity at power ≥ 0.8 | 0.449 | 0.434 |
| E2 off-target fraction | 0.282 (95% CI 0.247–0.315) | 0.276 (in its population) |
| Shuffled false ON_STAR rate | 0.0012 (n = 852) | nominal 0.005 |

- Positive controls (check b, passed): observed/predicted 1.04. By power bin, observed against predicted: 0.68 against 0.65 (n = 186), 0.93 against 0.89 (n = 456), 0.92 against 0.96 (n = 52).
- E5: robust standard deviation 0.85 (0.75–0.95), median 0.05, with 1.1% of recovered items at the window edge.

**Gumbel pivot.** See section 8.3.

**Real data on known binaries.** See the Kepler results in section 10.

**Test suite** (`tests/test_drrel.py`, 15 tests). It checks:

- control disjointness and alias exclusion;
- the method boundaries at every testable control count;
- the null false-recovery rate, for the rank test and for the pivot;
- that the pivot refuses zero-inflated controls;
- that power predicts recovery;
- the outcome rules, including untested neighbours;
- that epoch slots are filled only inside the neighbour radius;
- the single E2 rule, the centred E5 spread, and a withheld E1;
- the scorer's seal check:
  - a rule edit fails;
  - a forged manifest fails against the published hash;
  - a predictions edit fails;
  - an adapter edit is reported;
- every written form of an embargoed ID (`TIC123…`, `tic_123…`, `123….json`, `Gaia DR3 123….`, …) and embargoed coordinates;
- the format adapter;
- the estimators;
- one bundle scored end to end.

## 12. What can make the test non-decisive

- **About 35% of items cannot be tested at all** (192,295 of 541,609): 37% of the LS tier and 11% of the box tier. Their windows are too wide for 41 disjoint control windows. The median σ_P/P is 2.4×10⁻³ for LS against 1.5×10⁻⁴ for box. These are the worst-determined, mostly weakest signals, so LS purity is measured on a subset biased toward better periods. Results are also reported by σ_P/P bin.
- **The anchoring promise holds for the box tier only.** Box items have a median of 7.6 independent periods per window. LS windows span 15–50, and about 205,000 LS items run with a free phase.
- **Heavy-tailed period errors.** 9–16% of real periods lie beyond ±3σ_P. Those signals fall outside the window and are mostly not recovered; the power model includes this from the calibration half's distribution. If DR4 objects differ from the catalogued binaries in this respect, check (b) on the held-out binaries is the guard.
- **Purity is low**, so the absolute number of real signals is small. The predicted purity of the clean eb tier is about 15%, and the LS tier's about 2–5%. E1 rests on about 27,000 clean-eb items, which is ample for a fraction but leaves few real objects in the faintest bins.
- **Period alias windows.** The survey's chosen niche (periods near 1/3–3 d) is 0.7% real and on the star in DR3, against 6.5% off-window. These items will mostly come out NOWHERE, as predicted. That is a decisive result, not a failure.
- **Correlated controls.** Where windows are wider than the precession comb spacing, neighbouring controls can be aliases of one another. The shuffled-control false rate (check a) measures this on real DR4 data.
- **Domain shift.** p_pred is fitted in the Kepler field (high ecliptic latitude, |b| about 5–20°). The rest of the footprint is more crowded. E4 tests this directly.
- **DR4 format and access.** Bundle fetching (`drrel/fetch_dr4.py`) must be adapted to the real services, and DataLink limits will slow it. A reported archive downtime of 5–10 December is unverified. A row without data is MISSING, not null, and the counts are reported.
- **Neighbours.** At most 10 eligible neighbours per item are tested; the rest are listed and `nb_complete` records them.
- **Blindness.** Stars with public DR3 epoch photometry are flagged (`dr3_epoch_public`), and results are also reported without them.

## 13. After release

1. Fetch bundles with `drrel/fetch_dr4.py`. It streams and reduces, and is format code only: endpoints and column names may be adapted to the DR4 services, and each change is logged. Only `fetch_dr4.py` and `gaia_io.py` may change.
2. From `code/`, run `python -m drrel.score --seal-hash TOP --predictions ../data/predictions.parquet --bundles B --out OUT`.
   - TOP is the published top-level hash, from the README and the Zenodo record. It cannot be written in this plan, because the plan is one of the files the hash covers.
   - The scorer first checks that sha256(`MANIFEST.sha256`) equals TOP, so a forged manifest fails.
   - It then checks every rule file of `drrel` and `data/predictions.parquet` against the manifest, and reports the hashes of the two format adapters.
   - It prints each result, writes them into the report, and refuses to score on any mismatch.
3. Report:
   - every endpoint and both validity checks;
   - the counts of MISSING and NO_TEST rows;
   - the rule-file verification printed by the scorer.

   A row without a bundle is MISSING, never a null.

## 14. Reproducing the predictions

The deposit holds every frozen input under `data/inputs/` (embargoed with the catalogue), plus `results/inputs_provenance.json` for the inputs that are public or private elsewhere:

- the VSX table, by SHA-256;
- the survey's ledger and stored light curves, described.

`code/work/rebuild.py` runs on the two deposits as published:

    python -I code/work/rebuild.py --public <git or Zenodo record 1> --embargoed <Zenodo record 2> \
        --work <new scratch dir> [--vsx <VSX table>] [--purity]

It copies `code/` into the scratch directory and maps every deposited input to the name the scripts read (the table is in its docstring). It then runs, in turn:

1. `build_items.py`, with `--vsx`. Its output `period_errors.json` stays inside the scratch copy.
2. `purity_dr3.py`, with `--purity`. This step is slow.
3. `build_predictions.py`.

It compares every rebuilt file byte for byte with the deposit:

- `data/predictions.parquet` and `data/items.parquet`;
- the items, period-error table and σ calibration;
- the DR3 likelihood ratios;
- the five results files.

It exits with 0 only if all of them are identical.

The scripts that read the survey's private stores (the catalogue export and the ephemeris refinement) take their location from the environment variable `DR4REL_SURVEY_STATE`. Their outputs are the frozen inputs.
