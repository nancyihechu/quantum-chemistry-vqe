"""Classical reference energies within each active space (RHF, CCSD, CASCI = exact)."""

import time

import ffsim
from pyscf import scf

from .molecules import System, molecular_data


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

    try:
        data.run_ccsd(store_t1=True, store_t2=True)
        record["e_ccsd"] = float(data.ccsd_energy)
    except Exception as err:  # CCSD can fail to converge at stretched geometries
        record["e_ccsd"] = None
        record["ccsd_error"] = repr(err)

    if run_fci:
        t0 = time.perf_counter()
        data.run_fci()
        record["e_fci"] = float(data.fci_energy)
        record["fci_seconds"] = time.perf_counter() - t0

    return record, data
