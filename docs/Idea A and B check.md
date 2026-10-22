# Idea A (flagship 52-qubit N2) and Idea B (pair selection) check, 22 Oct 2026

## Idea B: amplitude-guided alpha-beta pair selection / orbital ordering
- Simple version already done: Motta et al. 2023 Chem. Sci. SI section S4 permutes orbitals so the kept alpha-beta pairs have the largest |J_pp| amplitudes. https://www.rsc.org/suppdata/d3/sc/d3sc02516k/d3sc02516k1.pdf
- Relabeling orbitals does not change the LUCJ family (orbital rotations absorb it); it only helps the optimization-free CCSD seed. Compressed double factorization (optimize=True, Lin et al. arXiv:2511.22476) may already recover that gain.
- VQE prior art: Tkachenko et al. PRX Quantum 2021 (mutual-information qubit ordering), Romero et al. 2019 (MP2 amplitude screening), Zhang et al. QST 2021, Tkachenko/Head-Gordon Chem. Sci. 2025.
- Still open: non-standard interaction graphs (off-diagonal alpha-beta pairs, per-layer graphs) fitted to the chip, plus hardware evidence. Bigger method project; not for the March to April 2027 paper.

## Idea A: N2 cc-pVDZ (10e,26o) flagship system
- Source: Robledo-Moreno et al., Sci. Adv. 11, eadu9991 (arXiv:2405.05068), title now "Chemistry beyond the scale of exact diagonalization on a quantum-centric supercomputer".
- Public data: https://github.com/jrm874/sqd_data_repository (Zenodo 10.5281/zenodo.15324153): FCIDUMPs, raw hardware counts, HCI, RHF, MP2, CISD, CCSD, SQD, OO-SQD and random-sample SQD energies.
- Device: ibm_torino (Heron r1), Dec 2023. 58 qubits (52 + 6 auxiliary), LUCJ n_reps=1 + final orbital rotation, CCSD seed, alpha-beta pairs (p,p) for p = 0,4,8,..., about 98,304 shots per bond length, only 0.13 to 0.29% valid shots.
- Random valid fraction for (10e,26o) is about 0.0001% (9.6e-7), so 0.13% is about 1,350 times random. For (10e,16o) it is 0.444%.
- Bond lengths in the repo are multiples of 1.0975135 A (not A). SQD energies at 23 points, 0.768 to 3.183 A. SQD settings: 10 batches, 10 recovery rounds, d = 1.44M, 4M, 9M, 16M.
- Published SQD error vs HCI at d = 16M: 13 to 22 mEh near equilibrium (0.77 to 1.10 A), 34 to 49 mEh at 1.54 to 1.76 A, 73 to 90 mEh at 2.0 to 3.2 A. Random-sample SQD is worse by about 10 to 90 mEh.
- Later work on this system: Barison et al. 2411.00468 (ext-SQD, same data), Danilov et al. 2503.05967 (SQD + AFQMC, same data), Lin et al. 2511.22476 (ibm_kingston, 1.2 A, data Zenodo 17704385), Park et al. 2603.09346 (ibm_fez), McFarthing et al. 2602.01165 (half-qubit SQD), Vaquero-Sabater et al. 2605.23697 (simulation vs torino data). IBM SQD tutorial runs this system at 1.0 A on ibm_boston (31.9 mEh at d about 0.96M).
- Classical cost (qiskit-addon-sqd 0.14 solve_sci, 2 threads): d = 1M 172 s / 0.7 GB; d = 4M 524 s / 1.4 GB. d = 16M estimated 25 to 30 min and 4 to 6 GB per diagonalization. Use d up to 4M on the cluster.
