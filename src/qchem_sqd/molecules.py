"""Benchmark systems: diatomic molecules with fixed basis sets and active spaces."""

from dataclasses import dataclass

import ffsim
from pyscf import gto, scf


@dataclass(frozen=True)
class System:
    """A diatomic molecule scanned over bond length.

    The active space is ``norb`` spatial orbitals directly above ``n_frozen``
    doubly occupied core orbitals. ``norb=None`` uses every non-frozen orbital.
    """

    name: str
    atom_a: str
    atom_b: str
    basis: str
    n_frozen: int
    norb: int | None
    bond_lengths: tuple[float, ...]  # Angstrom


N2_BOND_LENGTHS = (0.9, 1.0, 1.1, 1.3, 1.6, 2.0, 2.4)

SYSTEMS = {
    s.name: s
    for s in [
        # Small systems for the error-mitigation benchmark (estimator-based).
        System("h2_sto6g", "H", "H", "sto-6g", 0, None, (0.5, 0.6, 0.735, 0.9, 1.2, 1.6, 2.0)),
        System("lih_sto3g", "Li", "H", "sto-3g", 1, None, (1.2, 1.4, 1.6, 1.8, 2.2, 2.6, 3.0)),
        # N2 for SQD: 16-qubit validation tier and 32-qubit hardware tier.
        System("n2_631g_8o", "N", "N", "6-31g", 2, 8, N2_BOND_LENGTHS),
        System("n2_631g_16o", "N", "N", "6-31g", 2, None, N2_BOND_LENGTHS),
    ]
}


def build_molecule(system: System, bond_length: float) -> gto.Mole:
    return gto.M(
        atom=[[system.atom_a, (0.0, 0.0, 0.0)], [system.atom_b, (0.0, 0.0, bond_length)]],
        basis=system.basis,
        unit="Angstrom",
        symmetry=False,
        verbose=0,
    )


def run_rhf(mol: gto.Mole) -> scf.hf.RHF:
    """RHF with a stability check, since stretched N2 can land on a saddle point."""
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    mo, _, stable, _ = mf.stability(return_status=True)
    if not stable:
        mf.kernel(mf.make_rdm1(mo, mf.mo_occ))
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge for {mol.atom}")
    return mf


def active_orbitals(system: System, mol: gto.Mole) -> list[int]:
    stop = mol.nao if system.norb is None else system.n_frozen + system.norb
    return list(range(system.n_frozen, stop))


def molecular_data(system: System, bond_length: float) -> tuple[ffsim.MolecularData, scf.hf.RHF]:
    """Active-space Hamiltonian (one- and two-body tensors plus core energy)."""
    mol = build_molecule(system, bond_length)
    mf = run_rhf(mol)
    data = ffsim.MolecularData.from_scf(mf, active_space=active_orbitals(system, mol))
    return data, mf
