# Superseded: results made with unsymmetrized orbitals

Everything in this folder was computed before 2026-10-21 with `symmetry=False` RHF orbitals. Do not cite these numbers. They are kept only so earlier commits and messages can be traced.

## Why they are superseded

For N2 from 1.6 A onward, the unsymmetrized RHF followed a stability instability into a symmetry-broken solution. Its density mixes irreducible representations (cross-irrep elements 0.6 to 0.8, against about 1e-16 for the symmetric solution), and it lies 20 mHa (1.6 A) to 234 mHa (2.4 A) below the symmetry-adapted RHF. Reference orbitals, CCSD amplitudes, LUCJ circuits and every SQD result at stretched bonds therefore changed. Results at 0.9 to 1.3 A were numerically unaffected, but are superseded too so that one consistent orbital convention is used throughout.

Two further caveats apply to these files:

- `sqd_sim/` (both files) used FakeMiami or `optimize=True` LUCJ parameters without regularization, both since dropped.
- `sqd_sim/n2_631g_8o_noisy.json` (16 qubits) and all 16-qubit SQD results are pipeline checks only: uniform random bitstrings already give the exact energy in that space.

## Replacements

- References: `results/references/<system>_sym.json`
- Software checks: `results/checks/`
- Hardware design and predictions: `results/hardware/design_frozen.json`, `docs/preregistration.md`

| File | Orbitals | Other issues |
|---|---|---|
| references/h2_sto6g.json, lih_sto3g.json | unsymmetrized | none (H2/LiH not used in the current study) |
| references/n2_631g_8o.json, n2_631g_16o.json | unsymmetrized, symmetry-broken from 1.6 A | none |
| sqd_sim/n2_631g_16o.json | unsymmetrized | FakeMiami layout, optimize=True without regularization |
| sqd_sim/n2_631g_8o_noisy.json | unsymmetrized | 16 qubits: random bitstrings are exact |
| sqd_dimsweep/n2_631g_16o.json, n2_631g_8o.json | unsymmetrized | FakeMiami layout |
