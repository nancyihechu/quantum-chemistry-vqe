"""Classical reference energies within each active space (RHF, CCSD, FCI = exact).

Everything here works from the full-molecule RHF solution. ffsim's
MolecularData.run_ccsd/run_fci re-solve SCF inside the active space from a
default guess, which for stretched N2 in the 16-orbital space converges to a
state ~0.5 Ha above the true RHF and corrupts CCSD (and the LUCJ seed).
"""

import time

import ffsim
import numpy as np
import pyscf.cc
import pyscf.fci
from pyscf import scf

from .molecules import System, active_orbitals, molecular_data

# Stretched N2 can need more iterations or a level shift for CCSD to converge.
CCSD_LEVEL_SHIFTS = (0.0, 0.3, 1.0)


def _ccsd(mf: scf.hf.RHF, frozen: list[int], level_shift: float = 0.0, guess=None):
    ccsd = pyscf.cc.CCSD(mf, frozen=frozen or None)
    ccsd.max_cycle = 500
    ccsd.conv_tol = 1e-8
    ccsd.diis_space = 12
    ccsd.level_shift = level_shift
    ccsd.verbose = 0
    ccsd.kernel(t1=None if guess is None else guess[0], t2=None if guess is None else guess[1])
    return ccsd


def continued_amplitudes(system: System, bond_length: float, start: float = 1.0, step: float = 0.1):
    """Follow converged CCSD from ``start`` to ``bond_length``, seeding each step with the last.

    With symmetry-adapted RHF orbitals, cold-started CCSD fails for stretched N2
    (2.4 A); the continued solution converges along the whole path to 2.4 A.
    """
    guess, r = None, start
    path = list(np.round(np.arange(start, bond_length, step), 6)) + [bond_length]
    for r in path:
        data, mf = molecular_data(system, float(r))
        frozen = [p for p in range(mf.mo_coeff.shape[1]) if p not in active_orbitals(system, mf.mol)]
        ccsd = _ccsd(mf, frozen, guess=guess)
        if not ccsd.converged:
            return None
        guess = (ccsd.t1, ccsd.t2)
    return guess


def run_ccsd(data: ffsim.MolecularData, mf: scf.hf.RHF, active: list[int], system: System | None = None,
             bond_length: float | None = None) -> dict:
    """Frozen-orbital CCSD on the full RHF, storing amplitudes on ``data``.

    Tries a cold start, then continuation along the bond-length path (when the
    system is given), then level shifts. Amplitudes from an unconverged run are
    still stored (they seed LUCJ), but the energy is reported as None so it
    never enters a comparison table.
    """
    frozen = [p for p in range(mf.mo_coeff.shape[1]) if p not in active]
    route, ccsd = "cold", _ccsd(mf, frozen)
    if not ccsd.converged and system is not None and bond_length is not None:
        guess = continued_amplitudes(system, bond_length)
        if guess is not None:
            route, ccsd = "continued", _ccsd(mf, frozen, guess=guess)
    for level_shift in CCSD_LEVEL_SHIFTS[1:]:
        if ccsd.converged:
            break
        route, ccsd = f"level_shift_{level_shift}", _ccsd(mf, frozen, level_shift=level_shift)
    data.ccsd_t1, data.ccsd_t2 = ccsd.t1, ccsd.t2
    data.ccsd_energy = ccsd.e_tot if ccsd.converged else None
    return {
        "e_ccsd": float(ccsd.e_tot) if ccsd.converged else None,
        "ccsd_converged": bool(ccsd.converged),
        "ccsd_route": route,
    }


def compute_fci(data: ffsim.MolecularData) -> float:
    """Exact ground-state energy in the active space, directly from the integrals."""
    energy, _ = pyscf.fci.direct_spin1.kernel(
        data.one_body_integrals,
        data.two_body_integrals,
        data.norb,
        data.nelec,
        ecore=data.core_energy,
        conv_tol=1e-10,
        max_cycle=200,
    )
    data.fci_energy = energy
    return float(energy)


def reference_energies(system: System, bond_length: float, run_fci: bool = True) -> tuple[dict, ffsim.MolecularData]:
    """Compute RHF, CCSD and (optionally) active-space FCI energies.

    Returns a JSON-serializable record and the MolecularData, whose CCSD
    amplitudes seed the LUCJ ansatz.
    """
    data, mf = molecular_data(system, bond_length)
    record = {
        "system": system.name,
        "bond_length": bond_length,
        "norb": data.norb,
        "nelec": list(data.nelec),
        "n_qubits": 2 * data.norb,
        "e_rhf": float(mf.e_tot),
    }
    record.update(run_ccsd(data, mf, active_orbitals(system, mf.mol), system, bond_length))

    if run_fci:
        t0 = time.perf_counter()
        record["e_fci"] = compute_fci(data)
        record["fci_seconds"] = time.perf_counter() - t0

    return record, data
