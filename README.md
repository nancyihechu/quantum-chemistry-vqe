# Variational Quantum Eigensolver for Quantum Chemistry

This repository documents a structured, hands-on implementation of the Variational Quantum Eigensolver (VQE) for quantum chemistry, developed while working through the IBM Quantum learning materials in preparation for—and successful completion of—the **Quantum Chemistry with VQE** certification exam.

The notebooks follow the instructional progression of the learning tutorial, but are implemented independently with an emphasis on understanding, reproducibility, and hardware-aware execution. All circuits are transpiled and optimized for the **IBM Quantum Nighthawk R1 processor**, and simulations are performed using noise models derived from this backend to reflect realistic device behavior.

The workflow is implemented using Qiskit and PySCF and is organized as a sequence of Jupyter notebooks that mirror the key stages of a quantum chemistry calculation on near-term quantum hardware.

## Notebook Overview

### 1. Hamiltonian Preparation
**`01_Hamiltonian_Preparation.ipynb`**  
Constructs an electronic structure Hamiltonian for H₂ using PySCF, applies a fermion-to-qubit mapping (Jordan–Wigner), and produces a qubit Hamiltonian expressed as a sum of Pauli operators.

### 2. Ansatz Construction
**`02_Ansatz_Construction.ipynb`**  
Explores variational circuit (ansatz) design, including hardware-efficient and custom ansätze. Discusses expressibility, entanglement patterns, circuit depth, and practical considerations for near-term devices.

### 3. Ground-State Energy Estimation
**`03_Ground_State_Energies_VQE.ipynb`**  
Implements the VQE algorithm to estimate the ground-state energy of H₂ using the prepared Hamiltonian and selected ansatz. Includes circuit transpilation, backend-aware execution, and classical optimization.

### 4. Molecular Geometry Optimization
**`04_Molecular_Geometry_Optimization.ipynb`**  
Extends the VQE workflow to compute energies across varying bond lengths and constructs a potential energy curve to estimate the equilibrium geometry of the molecule.

## Notes

- Simulations use hardware-aware noise models derived from IBM Quantum backends.
- Optimization parameters are chosen to balance runtime and clarity of demonstration.
- Results reflect the limitations of basis sets, ansatz expressibility, and near-term hardware constraints.

This repository documents the full workflow completed as part of the IBM Quantum learning program.
