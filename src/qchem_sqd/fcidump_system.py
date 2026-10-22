"""N2/cc-pVDZ (10e, 26o) from the published FCIDUMPs of arXiv:2405.05068.

The FCIDUMPs are already expressed in the authors' RHF orbitals (no symmetry
labels: ORBSYM is all 1). We never re-solve SCF in them (re-solving from a
default guess can land on another state, as found earlier for ffsim's
MolecularData.scf); instead an RHF object is built with identity orbitals and
aufbau occupations, and its energy is checked against the published RHF value.

Bond lengths in the repository are in units of 1.0975135 A.
"""

from pathlib import Path

import ffsim
import numpy as np
import pyscf.cc
from pyscf import ao2mo, gto, scf

REPO = Path("/cluster/pixstor/dkhf3-lab/users/nancy/GITHUB/external/sqd_data_repository")
FCIDUMP_DIR = REPO / "integrals" / "N2_cc-pVDZ"
REFERENCE_FILE = REPO / "classical_reference_energies" / "N2_cc-pVDZ" / "energies_n2_cc-pvdz.txt"
PUBLISHED_SQD = REPO / "experiments" / "N2_cc-pVDZ" / "sqd_hardware_energetics" / "sqd_energies_vs_bond_length.txt"
PUBLISHED_UNIFORM = REPO / "experiments" / "N2_cc-pVDZ" / "sqd_on_uniform_distribution" / "sqd_energies_vs_bond_length_uniform.txt"
TORINO_COUNTS = Path("/cluster/pixstor/dkhf3-lab/users/nancy/GITHUB/external/torino_counts_npz")
UNIT_ANGSTROM = 1.0975135
STUDY_GRID = (1.0, 1.5, 2.2)  # 1.098, 1.646, 2.415 A


def angstrom(grid_value: float) -> float:
    return grid_value * UNIT_ANGSTROM


def load(grid_value: float) -> ffsim.MolecularData:
    return ffsim.MolecularData.from_fcidump(FCIDUMP_DIR / f"R_26_{grid_value:.2f}_fcidump.txt")


def rhf_from_integrals(data: ffsim.MolecularData) -> scf.hf.RHF:
    """RHF object in the FCIDUMP orbital basis, with no SCF iterations."""
    norb, (na, nb) = data.norb, data.nelec
    mol = gto.M(verbose=0)
    mol.nelectron = na + nb
    mol.incore_anyway = True
    mol.energy_nuc = lambda *args: data.core_energy
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: data.one_body_integrals
    mf.get_ovlp = lambda *args: np.eye(norb)
    mf._eri = ao2mo.restore(8, data.two_body_integrals, norb)
    mf.mo_coeff = np.eye(norb)
    mf.mo_occ = np.array([2.0] * na + [0.0] * (norb - na))
    dm = mf.make_rdm1()
    fock = mf.get_fock(dm=dm)
    mf.mo_energy = np.diag(fock).copy()
    mf.e_tot = mf.energy_tot(dm=dm)
    mf.converged = True
    mf.fock_offdiag_occ_virt = float(np.abs(fock[:na, na:]).max())  # 0 for a converged RHF
    return mf


def ccsd_amplitudes(data: ffsim.MolecularData, mf, guess=None) -> dict:
    """CCSD on the FCIDUMP RHF; stores t1/t2 on ``data``."""
    cc = pyscf.cc.CCSD(mf)
    cc.max_cycle, cc.conv_tol, cc.diis_space, cc.verbose = 500, 1e-8, 12, 0
    cc.kernel(t1=None if guess is None else guess[0], t2=None if guess is None else guess[1])
    data.ccsd_t1, data.ccsd_t2 = cc.t1, cc.t2
    data.ccsd_energy = cc.e_tot if cc.converged else None
    return {"e_ccsd": float(cc.e_tot) if cc.converged else None, "ccsd_converged": bool(cc.converged)}


def continued_ccsd(grid_value: float, start: float = 1.0, step: float = 0.1):
    """CCSD at grid_value, continued from ``start`` along the published grid when a cold start fails."""
    data = load(grid_value)
    mf = rhf_from_integrals(data)
    record = ccsd_amplitudes(data, mf)
    record["ccsd_route"] = "cold"
    if not record["ccsd_converged"]:
        guess = None
        for g in np.round(np.arange(start, grid_value + 1e-9, step), 2):
            d = load(float(g))
            r = ccsd_amplitudes(d, rhf_from_integrals(d), guess)
            if not r["ccsd_converged"]:
                break
            guess = (d.ccsd_t1, d.ccsd_t2)
        if guess is not None:
            record = ccsd_amplitudes(data, mf, guess)
            record["ccsd_route"] = "continued"
    record.update({"e_rhf": float(mf.e_tot), "fock_offdiag_occ_virt": mf.fock_offdiag_occ_virt})
    return data, mf, record


def reference_energies() -> dict:
    """{grid value: {rhf, mp2, cisd, ccsd, hci}} from the repository's classical reference file."""
    out = {}
    for line in REFERENCE_FILE.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) == 6:
            r, rhf, mp2, cisd, ccsd, hci = map(float, parts)
            out[round(r, 2)] = {"rhf": rhf, "mp2": mp2, "cisd": cisd, "ccsd": ccsd, "hci": hci}
    return out


def published_sqd() -> dict:
    """{grid value: {"1.44M": E, "4M": E, "9M": E, "16M": E, "uniform_1.44M": E}}."""
    out = {}
    for line in PUBLISHED_SQD.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) == 5:
            g = round(float(parts[0]) / UNIT_ANGSTROM, 2)
            out[g] = dict(zip(("1.44M", "4M", "9M", "16M"), map(float, parts[1:])))
    for line in PUBLISHED_UNIFORM.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            g = round(float(parts[0]) / UNIT_ANGSTROM, 2)
            out.setdefault(g, {})["uniform_1.44M"] = float(parts[1])
    return out


def torino_bit_array(grid_value: float):
    """Published ibm_torino shots as a BitArray (strings are Qiskit order: leftmost = highest qubit)."""
    from qiskit.primitives import BitArray

    z = np.load(TORINO_COUNTS / f"torino_R{grid_value:.1f}.npz")
    return BitArray.from_bool_array(z["bitstrings"].astype(bool), order="big")
