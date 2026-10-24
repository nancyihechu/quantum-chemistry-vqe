# Pre-registration: lattice connectivity and SQD sample quality on IBM Nighthawk

Frozen by the git commit that adds this version, before any hardware job (pilot included) is submitted. Later changes go under Deviations with date and reason.

## Questions

1. At a fixed subspace size, do LUCJ samples from a square-lattice interaction pattern give lower SQD energy errors than the heavy-hex pattern?
2. For the same QPU time, does ibm_phoenix (Nighthawk, fast repetition) give lower SQD errors than ibm_kingston (Heron)?
3. Do hardware samples carry more information than uniform random bitstrings?

## Common methods

- One-layer LUCJ seeded from CCSD amplitudes, plain truncation unless stated. Gate and measurement twirling, XpXm dynamical decoupling, no other error mitigation.
- Logical designs (pair lists per pattern, optimized parameters) are frozen in `results/hardware/design_frozen.json` and `results/hardware/design_ccpvdz_frozen.json`. Circuits sharing phoenix qubits are transpiled with routing disabled, so no arm contains SWAPs. Physical layouts are chosen from each day's calibration and recorded.
- Five repeats on at least two calendar days: repeats 1 to 3 on day 1, repeats 4 and 5 on a later day.
- Saved for every job: job ID, raw shots, ISA and logical circuits, calibration at submission and at execution, IBM-reported usage.
- Effective alpha-beta pairs (pairs with a non-zero coupling in the operator) are reported for every arm as a result, next to the pairs the layout allows. Simulation check of a second layer (`results/checks/nreps2.json`): n_reps = 2 uses the same effective pairs as n_reps = 1 for every arm and geometry, stays below RHF at every 32-qubit geometry, and costs about 1.8 times the two-qubit gates; it is therefore not used.

## Analysis settings (identical for every sample source, hardware or baseline)

| Setting | 32 qubits | 52 qubits |
|---|---|---|
| Software | qiskit-addon-sqd 0.14.0, pyscf 2.14.0, ffsim 0.0.84, qiskit 2.5.2 | same |
| Entry point | `diagonalize_fermionic_hamiltonian` via `qchem_sqd.sqd.run_sqd` | same |
| samples_per_batch | 300 | 1500 |
| num_batches | 3 | 3 |
| max_iterations | 5 (iteration 1 = postselection only, then 4 recovery rounds) | 5 |
| max_dim (strings per spin) | 30, 60, 100, 150, 200 | 1200 (d = 1.44M), 2000 (d = 4M) |
| symmetrize_spin | True | True |
| energy_tol, occupancies_tol | 1e-8, 1e-5 | 1e-8, 1e-5 |
| subspace policy | StandardPolicy, carryover_threshold 1e-4 | same |
| eigensolver | solve_sci_batch (pyscf), spin_sq unset | same |
| SQD seed | 24 | 24 |
| Uniform-random baseline | 3 sets, bitstring seeds 101 to 103, SQD seeds 24 to 26, same shot count as the arm | 1 set per (grid, d), seed 101, same shot count as T unless stated |
| Matched-shot subsampling | n/a | numpy default_rng(7), without replacement |
| HCI | pyscf selected_ci, select_cutoff 1e-4, ci_coeff_cutoff 1e-4, conv_tol 1e-8 | the repository's HCI energy |
| Exact reference | pyscf direct_spin1 FCI, conv_tol 1e-10 | the repository's HCI energy |
| Threads | circuit builds and the frozen design use 1 thread (optimize=True angles and the 2.4 A CCSD convergence path are thread-dependent; see Deviation 1) | same |

## System 1: N2/6-31G, 16 active orbitals, 32 qubits

- Symmetry-adapted RHF orbitals, 2 frozen core orbitals, (5, 5) electrons. CCSD continued from 1.0 A where a cold start fails (only 2.4 A). Bond lengths 1.0, 1.2, 1.8, 2.4 A. 1e5 shots per circuit.
- Arms: A ibm_phoenix, square pattern (12 allowed pairs, 6 effective); B ibm_phoenix, heavy-hex pattern on A's qubits (pairs 0, 4, 8 allowed, 0 and 4 effective); C ibm_phoenix, square pattern with ffsim optimize=True (regularization 1e-2, L-BFGS-B, 200 iterations, 1 thread) on A's qubits, at 1.0 and 1.2 A only (pre-specified option B: with 1000 and 3000 iterations the converged state is still above RHF, by 48.1 mHa at 1.8 A and 385.6 mHa at 2.4 A); D ibm_kingston, the logical circuit of B.
- Baselines at each per-spin cap d = 30, 60, 100, 150, 200: uniform random bitstrings through the same SQD loop; HCI (pyscf selected CI from the integrals only, select_cutoff 1e-4; it never sees the FCI vector); best FCI subspace (top-d strings of the exact wavefunction). In noiseless tests HCI shares 98 to 100% of its strings with best-FCI, so HCI here is effectively the best possible subspace, not a practical competitor.
- H1 (primary): at 1.0 and 1.2 A, A has a lower final SQD error than B.
- H2: every arm's valid fraction exceeds 0.444% (uniform random) by at least 10 times, and every arm's before-recovery error is below random's at d = 30 and 60.
- Exploratory: B versus D (device), A versus B at 1.8 and 2.4 A, C versus A (optimized versus plain angles).

## System 2 (flagship): N2/cc-pVDZ (10e, 26o), 52 qubits

- FCIDUMPs, ibm_torino shots and the HCI reference from the data repository of arXiv:2405.05068. Grid values 1.0, 1.5, 2.2 in units of 1.0975135 A (1.098, 1.646, 2.415 A). Our RHF and CCSD reproduce the repository's values to under 0.001 mHa.
- Arms: E ibm_phoenix, square pattern (12 of 26 pairs allowed, 0 to 4 effective); F ibm_phoenix, heavy-hex pattern on E's qubits (0, 4, 8 allowed; 0 and 4 effective); G ibm_kingston, heavy-hex pattern with 2 auxiliary qubits; T the published ibm_torino shots.
- Shots per circuit: 1e5 on kingston. Matched QPU time is a rule, not an estimate: phoenix study shots N_P are set so that IBM-reported usage per circuit is equal on both devices, calibrated from the pilot. The pilot runs one 52-qubit circuit per device (E on phoenix, G on kingston, grid 1.0); with U the IBM-reported job usage and N the pilot shots, N_P = 1e5 x (U_kingston / N_kingston) / (U_phoenix / N_phoenix), rounded to the nearest 1e4, computed by `run_hardware.py calibrate` and saved in `results/hardware/phoenix52_shots_calibrated.json` before the study is submitted. Pilot phoenix shots use the duration-model value 1.61e6 shots (about 26.5 s, against 26.4 s for the kingston pilot circuit). In the study, achieved usage per circuit (job usage divided by its number of circuits) is reported for both devices; a ratio outside 0.8 to 1.25 is reported as a deviation but does not change the analysis.
- All sample sources, T included, go through our pipeline with identical settings. The published SQD energies of arXiv:2405.05068 are never used for comparison: our pipeline does not reproduce them (on the same torino shots at d = 1.44M we obtain +21.1 and +75.9 mHa versus HCI at grid 1.0 and 2.2, against published +45.5 and +201.8 mHa).
- Random baselines at 52 qubits, same shot count as T, same settings: random_hfinit, uniform 52-bit strings (expected valid fraction 9.6e-7, so essentially none are valid) with SQD started from RHF orbital occupations; random_particle, uniform strings with the correct alpha and beta electron counts. The 32-qubit random baseline cannot be used unchanged at 52 qubits because the loop needs at least one valid string.
- Outcome: error versus the repository's HCI energy at d = 1.44M and 4M (1200 and 2000 strings per spin), before recovery and after each round. Also valid fraction (against 9.6e-7 for uniform random) and valid samples per QPU second.
- H3 (primary, matched QPU time): at each grid value, E has a lower final error than G when both use their full, QPU-time-matched shot counts.
- H4 (primary, matched shots): the same comparison after subsampling E to G's 1e5 shots.
- H5: E has a lower final error than F at grid 1.0 (connectivity on one chip).
- H6: E, F and G each beat both random baselines at the same d and shot count; each is compared with T after subsampling to T's number of valid shots.

## Decision rule and reporting

- For each primary or secondary comparison: per repeat, the paired difference of final errors. A hypothesis is supported at a geometry if the mean over the 5 repeats has the predicted sign and exceeds twice its standard error, at a majority of caps (32 qubits: 3 of 5; 52 qubits: both of d = 1.44M and 4M).
- A day effect is reported as the difference between day-1 and day-2 means.
- Every run is reported. A failed job is resubmitted once; nothing is excluded after results are seen. Pilot data are excluded from every hypothesis test.

## QPU budget (estimates from scheduled durations, default repetition delays, 5 s per job)

- Study, "Ligand Based Drug Discovery" (5100 s): 32 qubits 5 x (17 s phoenix + 109 s kingston) = 630 s; 52 qubits 5 x (164 s phoenix + 84 s kingston) = 1240 s; total 1870 s (37%). Phoenix 52-qubit shots are recalibrated from the pilot, which can change the phoenix share.
- Pilot, "Quantum Computation and Cryptography" (600 s): 131 s (22%).

## Pilot

32 qubits: bond lengths 1.0 and 2.4 A, all 32-qubit arms, one repeat. 52 qubits: one circuit per device at grid 1.0 (E on phoenix, G on kingston), QPU-time-matched shots. Run on the "Quantum Computation and Cryptography" instance. Pass criteria: every job finishes and its data are saved; valid fractions are at least 10 times random; IBM-reported usage is at most 1.5 times the estimate; the analysis runs end to end.

## Predictions from noiseless simulation

System 1, noiseless samples (1e5 shots), final error versus FCI in mHa at d = 60 / d = 200. B and D share one logical circuit, so their noiseless values are identical.

| Bond length | A square | B (= D) heavy-hex pattern | C optimized | uniform random | HCI = best FCI |
|---|---|---|---|---|---|
| 1.0 A | 80.2 / 80.2 | 124.9 / 124.9 | 33.2 / 10.7 | 104.0 / 30.3 | 24.7 / 4.5 |
| 1.2 A | 87.6 / 87.6 | 106.1 / 106.1 | 38.1 / 22.4 | 143.2 / 49.2 | 30.4 / 6.8 |
| 1.8 A | 109.7 / 109.7 | 154.9 / 154.9 | not run | 174.6 / 91.1 | 56.6 / 19.1 |
| 2.4 A | 79.8 / 76.1 | 89.6 / 81.9 | not run | 270.5 / 93.4 | 74.8 / 31.1 |

Predicted direction: A below B at every geometry (H1), every arm far below random before recovery (H2). Noise will add errors and valid-fraction losses that this table does not include.

System 2: noiseless sampling of the 52-qubit state is not feasible (4.3e9 amplitudes). Predictions are the device noise model and the torino reference T through our pipeline.

| Grid | E valid fraction (model) | F | G | phoenix valid shots per QPU s | kingston | T final error, d = 1.44M / 4M (mHa) |
|---|---|---|---|---|---|---|
| 1.0 | 0.106% | 0.109% | 0.147% | 64 | 5.5 | 21.1 / 10.8 |
| 1.5 | 0.125% | 0.128% | 0.176% | 76 | 6.7 | 36.4 / 34.9 |
| 2.2 | 0.128% | 0.132% | 0.175% | 78 | 6.6 | 75.9 / 43.0 (a) |

(a) Computed from the published torino shots by job 19206070, still running at freeze time; it uses no hardware data of ours. Value filled in on 2026-10-23 (Addendum A), with no other change.

Noise-model check: at 16 qubits the gate-and-readout model agrees with exact Aer noisy sampling within 2 to 6% on phoenix, kingston and fez (`results/checks/noise_model_validation.json`). The idle-decay term cannot be checked against Aer and is the largest source of uncertainty in these valid fractions.

## Deviations

1. 2026-10-22, before any study job was submitted (no study data existed). The first study-day-1 build stopped on the built-in guard "CCSD route changed since the design was frozen" at 2.4 A. Cause: with more than one thread the failed cold-start CCSD at 2.4 A ends in a thread-dependent state (three runs at 2 threads: -108.536, -108.502, -108.490 Ha), so the continuation can occasionally fail and fall back to a level-shifted route. With 1 thread the path is identical on every run and the continued solution equals the frozen one (-108.955193504 Ha). Change: all circuit builds (`run_hardware.py`) now pin threads to 1, the setting used to freeze the design. The frozen design, circuits and analysis are unchanged; the statement under Analysis settings that results are thread-count independent is corrected accordingly.

2. 2026-10-23. Study day 1 was submitted as one session on 2026-10-22 (21:26 UTC). The ibm_kingston jobs ran immediately (21:26 to 21:33 UTC, 2026-10-22), but the ibm_phoenix jobs waited in the device queue and ran at 05:33 UTC on 2026-10-23, about 8 hours later and on the next calendar day. Consequences: within-phoenix comparisons (H1, H5; arms interleaved in each job) are unaffected; phoenix versus kingston comparisons (H3, H4, B versus D) use data from different calibration windows, which is reported alongside those results. Repeats 4 and 5 will be submitted no earlier than 2026-10-24 so that they fall on a day distinct from both day-1 execution dates. No analysis setting changes. Usage matching held: 52-qubit usage per circuit was 28.0 s on phoenix and 28.3 s on kingston (ratio 0.99).

3. 2026-10-24. Order change for the Addendum B2 shot-scaling run. B2 was planned to be submitted after the day-2 data (repeats 4 and 5) had been fetched. Because ibm_phoenix has scheduled maintenance on 2026-10-27, and the shot-scaling run does not use or depend on day-2 data, it is submitted now, while the day-2 phoenix jobs are still queued. Effects: none on any analysis or setting; the shot-scaling jobs and the day-2 phoenix jobs may run in either order and in different calibration windows, which is recorded from the execution timestamps. Day-2 data had still not been fetched or viewed when this was committed.

Addendum A (2026-10-23): torino reference through our pipeline at grid 2.2, d = 4M = 43.0 mHa above the repository HCI energy (job 19206070). Fills footnote (a); no other change.

## Addendum B (2026-10-24, written before any data from repeats 4 and 5 were fetched or viewed)

Disclosure: this addendum was drafted after the day-1 interim results (repeats 1 to 3) had been seen. Those results showed small paired differences for H1, H3, H4 and H5, which is why equivalence tests are added here. No data from repeats 4 and 5 had been fetched or analyzed when this addendum was committed; its git commit time is the reference. The existing hypotheses, settings and decision rules are unchanged; this adds tests and one experiment.

### B1. Equivalence tests (in addition to the existing superiority rule)

- Applies to H1 (A versus B, 32 qubits, 1.0 and 1.2 A), H3 (E versus G at matched QPU time, 52 qubits), H4 (E subsampled to 1e5 shots versus G) and H5 (E versus F), using all 5 repeats.
- Test: two one-sided tests (TOST) on the per-repeat paired difference of final errors, alpha = 0.05, t distribution with 4 degrees of freedom. Equivalence is concluded at a geometry and cap when the 90% confidence interval of the mean paired difference lies entirely inside (-delta, +delta).
- Margins: delta = 5 mHa for the 52-qubit comparisons (H3, H4, H5); delta = 10 mHa for the 32-qubit comparison (H1). Rationale: 5 mHa is about three times chemical accuracy (1.6 mHa) and well below the 10 to 50 mHa errors of every arm at these subspace sizes; the 32-qubit margin is wider because the 32-qubit errors and their repeat-to-repeat spread are about twice as large.
- Aggregation, mirroring the existing rule: equivalence for a hypothesis at a geometry requires equivalence at a majority of caps (32 qubits: 3 of 5; 52 qubits: both d = 1.44M and 4M).
- Each comparison is reported as one of: superiority (existing rule), equivalence (this rule), or inconclusive. Both rules are always reported; neither replaces the other.

### B2. Phoenix shot-scaling run (52 qubits, exploratory)

- Circuit: arm E (ibm_phoenix, square pattern, frozen design in `results/hardware/design_ccpvdz_frozen.json`) at grid 1.0 and 2.2, rebuilt with the same frozen procedure (1 thread), same twirling and dynamical decoupling.
- Shots: 10,000,000 per grid value, submitted as 5 jobs of 2,000,000 shots per grid value, each job holding both grid values interleaved (4,000,000 executions per job, comparable to the 6,240,000 of the day-1 phoenix jobs). Instance: "Ligand Based Drug Discovery".
- Cost: 20,000,000 shots at the IBM-reported day-1 rate of 26.92 microseconds per shot (168 s per 6,240,000 shots) = 538 s; at the pilot rate of 28.93 microseconds per shot it would be 579 s. Expected 538 s, upper estimate 579 s.
- Analysis: pool the 5 jobs per grid value, then subsample without replacement to 0.1M, 0.3M, 1M, 3M and 10M shots (10M = all shots). For each size below 10M, 3 independent subsamples with numpy default_rng seeds 7, 8 and 9. SQD with the frozen 52-qubit settings at d = 1.44M and 4M. Reported: error versus the repository HCI energy before recovery and after each round, valid shots, mean and spread over the 3 subsamples, plotted against shot count. No hypothesis test; the result is described as exploratory.

### B3. Reproduction gap (reported; no change to any analysis)

qiskit-addon-sqd 0.1.0, used for the published results, is not publicly available; 0.3.0 (earliest public release) is used as a proxy. Errors versus HCI in mHa at d up to 1.44M on the published ibm_torino shots:

| Grid | Published | 0.3.0, 600 per batch, 3 batches | 0.3.0, 600 per batch, 10 batches | 0.14.0, 600 per batch | 0.14.0, 1500 per batch |
|---|---|---|---|---|---|
| 1.0 | 45.5 | 83.9 | 81.2 | 36.4 | 21.1 |
| 2.2 | 201.8 | 272.8 | 213.8 | 88.6 | 75.9 |

Breakdown of the gap between the published value and our pipeline (0.14.0, 1500 per batch): version effect (0.3.0 minus 0.14.0 at 600 per batch) +45 to +48 mHa at grid 1.0 and +125 to +184 mHa at grid 2.2; samples-per-batch effect (600 minus 1500 within 0.14.0) +15.3 and +12.8 mHa; residual (published minus 0.3.0) -36 to -39 and -12 to -71 mHa. Caveat: 0.3.0 at 600 samples per batch reached subspaces of about 0.9M to 1.09M, not 1.44M, which accounts for part of the negative residual. Most of the gap is attributable to algorithm changes between versions (carry-over and subspace caps), not to samples per batch.

## Addendum C (2026-10-24, written before any data from repeats 4 and 5 were fetched or viewed)

- One additional exploratory kingston run: arm G (ibm_kingston, heavy-hex pattern, frozen 52-qubit design in `results/hardware/design_ccpvdz_frozen.json`) at grid 2.2, 1,000,000 shots in one job, rebuilt with the frozen procedure (1 thread), same twirling and dynamical decoupling. Instance: "Quantum Computation and Cryptography".
- Purpose: a kingston 52-qubit data set with a shot count comparable to the phoenix runs, at the stretched geometry, to place next to the B2 phoenix shot-scaling curve.
- Cost: 1,000,000 shots at the IBM-reported day-1 kingston rate of 283.3 microseconds per shot (85 s per 300,000 shots) = 283 s; at the pilot rate of 300.0 microseconds per shot, 300 s. The instance had 454 s remaining.
- Analysis: subsample without replacement to 0.1M and 0.3M shots (numpy default_rng seeds 7, 8, 9) and use all 1M, at d = 1.44M and 4M, with the frozen 52-qubit settings. Reported: error versus the repository HCI energy before recovery and after each round, valid shots, mean and spread over the 3 subsamples. No hypothesis test; exploratory.

