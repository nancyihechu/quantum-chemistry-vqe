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
