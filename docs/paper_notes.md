# Paper notes

Working notes for the manuscript: caveats and wording decisions that the text must reflect. Not part of the pre-registration.

## Reproduction of the published ibm_torino SQD energies (arXiv:2405.05068)

- qiskit-addon-sqd 0.1.0, used for the published results, is not publicly available (PyPI and the official repository start at 0.3.0, 2024-09-06). We used 0.3.0, the earliest public release, as a proxy.
- Caveat (2026-10-24): with 600 samples per batch, the 0.3.0 runs reached subspaces of only about 0.9M to 1.09M configurations, not the published d = 1.44M, because the batches contained about 950 to 1050 distinct strings rather than 1200. The version gap we report (0.3.0 minus 0.14.0 at 600 per batch: +45 to +48 mHa at grid 1.0, +125 to +184 mHa at grid 2.2) therefore partly reflects this smaller subspace, not only algorithm changes. The text must say "partly attributable to subspace size" and must not present the version effect as a pure algorithm effect.
- At grid 2.2 with 10 batches, 0.3.0 comes within 12 mHa of the published value (213.8 vs 201.8 mHa versus HCI); at grid 1.0 it is worse than published (81 to 84 vs 45.5 mHa), consistent with its smaller subspace.
- Published SQD energies are never used as a comparison point for our hardware results; the torino shots are always processed with our pipeline and settings (pre-registered).

## Interim observations to handle carefully

- Arm B at 2.4 A, d = 30 (32 qubits, day 1): about 75 mHa better than A and D in all three repeats. Exploratory analysis (`results/exploratory/b24_d30.json`): a cap-boundary effect. The decisive multiply excited strings (into orbitals 5 to 7) are sampled at similar rates by all arms (frequency rank 16 to 30, at the edge of the 30-string cap); B's recovery keeps them. Not a device or connectivity effect; gone from d = 100 upward. Report d = 30 results at stretched bonds as cap-sensitive.
- 52 qubits: the random_hfinit baseline (uniform strings, recovery started from RHF occupations) is close to or better than hardware at grid 1.0 and at d = 4M. Discuss how much of SQD's 52-qubit accuracy comes from configuration recovery seeded with Hartree-Fock information rather than from device samples.
- Valid-fraction noise model: the idle-decay (T1) term was far too pessimistic at 52 qubits (predicted 0.075% and 0.21%, measured 1.5% and 1.06%); the gate-and-readout part agreed with Aer within 2 to 6% at 16 qubits. Report the model as validated only without the idle term.
- Day-1 phoenix jobs ran about 8 hours after the kingston jobs (Deviation 2); phoenix versus kingston comparisons span calibration windows.

## Five-repeat results: wording for the manuscript (2026-10-27)

Numbers are in the Results section of `docs/preregistration.md` and in `results/hardware/analysis_preregistered.json`.

- Headline: a null result. The square pattern did not beat the heavy-hex pattern on the same Nighthawk chip (H1 at 32 qubits, H5 at 52 qubits), and phoenix did not beat kingston at matched QPU time or matched shots (H3, H4). Most of these comparisons are formally equivalent within the Addendum B margins (10 mHa at 32 qubits, 5 mHa at 52 qubits). Write "no detectable advantage, equivalent within 5 mHa", not "no difference".
- The equivalence margins were added after the day-1 interim results were seen (Addendum B disclosure). The text must say so wherever an equivalence verdict is used.
- The noiseless prediction for H1 (A about 45 mHa below B at 1.0 A, d = 60) vanished on hardware. Candidate explanations to state as hypotheses, not findings: noise flattens the sampled distribution, and configuration recovery removes the differences between the sample sets.
- H1 at 1.2 A, d = 30 shows A 7.2 +/- 2.1 mHa better than B. It is one cap of five and does not change the verdict; do not report it as a finding.
- The only "supported" 32-qubit comparison is exploratory: C (optimize=True) beats A at 1.0 A by 7.6, 5.3 and 1.3 mHa at d = 60, 100 and 150. The noiseless gain was about 47 mHa at d = 60, so hardware kept roughly a sixth of it. At 1.2 A it is inconclusive, and C is worse by 7.3 mHa at d = 30. Label it exploratory and do not generalize to "optimized angles help on hardware".
- Phoenix collects about 11 times more valid shots per QPU second than kingston, but this does not lower the final error: the shot-scaling curves are flat from 1e5 shots. The throughput advantage is real and should be reported, but the paper must not imply it buys accuracy at these subspace sizes. Possible exception, for discussion only: H3 at grid 2.2, d = 4M (E better by 3.1 +/- 1.4 mHa).
- Against random baselines, keep the two baselines apart. Particle-conserving random strings are far worse (216 to 438 mHa), but uniform strings with HF-seeded recovery (random_hfinit) match or beat hardware at grid 1.0 (E and F are 4.3 and 4.2 mHa worse at d = 1.44M). Hardware wins only at stretched bonds. This is in line with van Rossum et al. and Vaquero-Sabater et al. and should be framed as agreeing with them.
- The H6 comparisons with T and with the random baselines use a single baseline run, so the SE reflects only the arm's repeat-to-repeat spread. State that baseline variance is not included. Both ways of matching to T (Deviation 5) agree at grids 1.5 and 2.2; at grid 1.0 the verdict depends on the matching (equivalent for E either way, supported for F and G only at matched total shots).
- The measured 52-qubit valid fractions (0.96% to 1.09%) are 5 to 10 times the model's predictions; see the idle-term note above.
