"""Save today's configuration and calibration of the project devices (read-only, no jobs).

Usage: python scripts/snapshot_backends.py [--instance "Ligand Based Drug Discovery"]
Writes results/backends/<name>_<UTC time>/{conf,props}.json
"""

import argparse

from qchem_sqd.backends import HEAVY_HEX_DEVICES, SQUARE_DEVICE, connectivity_of, save_snapshot, service_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="Ligand Based Drug Discovery")
    args = parser.parse_args()

    service = service_for(args.instance)
    for name in (SQUARE_DEVICE, *HEAVY_HEX_DEVICES):
        backend = service.backend(name)
        folder = save_snapshot(backend)
        print(f"{name}: {connectivity_of(backend)}, {backend.num_qubits} qubits -> {folder}")


if __name__ == "__main__":
    main()
