"""Real IBM backends: account access, calibration snapshots, and offline copies.

Project devices: ibm_phoenix (Nighthawk r2, square lattice) and ibm_fez /
ibm_kingston (Heron r2, heavy-hex). ibm_miami is not used in this project.

A snapshot stores a backend's configuration and calibration properties as the
same JSON files qiskit's fake backends use, so circuits and noise models can
be built offline (e.g. on Slurm nodes) from the real device data of that day.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "results" / "backends"
ACCOUNT = "QUASAR"  # saved account in ~/.qiskit/qiskit-ibm.json
SQUARE_DEVICE = "ibm_phoenix"
HEAVY_HEX_DEVICES = ("ibm_fez", "ibm_kingston")
FORBIDDEN_DEVICES = ("ibm_miami", "fake_miami")


def connectivity_of(backend) -> str:
    coupling = backend.coupling_map
    max_degree = max(len(set(coupling.neighbors(q))) for q in range(backend.num_qubits))
    return "square" if max_degree >= 4 else "heavy-hex"


def check_allowed(backend) -> None:
    if backend.name.lower() in FORBIDDEN_DEVICES:
        raise ValueError(f"{backend.name} must not be used in this project")


def service_for(instance_name: str, account: str = ACCOUNT) -> QiskitRuntimeService:
    """Runtime service bound to an instance, looked up by its display name."""
    base = QiskitRuntimeService(name=account)
    crns = [inst["crn"] for inst in base.instances() if inst["name"] == instance_name]
    if not crns:
        raise ValueError(f"No instance named {instance_name!r} for account {account!r}")
    return QiskitRuntimeService(name=account, instance=crns[0])


def save_snapshot(backend, out_dir: Path = SNAPSHOT_DIR) -> Path:
    """Write conf/props JSON for a live backend into out_dir/<name>_<UTC time>/."""
    check_allowed(backend)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    folder = out_dir / f"{backend.name}_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    conf = backend.configuration().to_dict()
    props = backend.properties().to_dict()
    (folder / "conf.json").write_text(json.dumps(conf, default=str))
    (folder / "props.json").write_text(json.dumps(props, default=str))
    return folder


def latest_snapshot(name: str, out_dir: Path = SNAPSHOT_DIR) -> Path:
    folders = sorted(out_dir.glob(f"{name}_*"))
    if not folders:
        raise FileNotFoundError(f"No snapshot for {name} in {out_dir}; run scripts/snapshot_backends.py")
    return folders[-1]


def load_snapshot(folder: Path) -> FakeBackendV2:
    """Offline backend with the real device's coupling map, gate set and calibration."""
    folder = Path(folder)
    name = folder.name.rsplit("_", 1)[0]

    class SnapshotBackend(FakeBackendV2):
        dirname = str(folder)
        conf_filename = "conf.json"
        props_filename = "props.json"
        backend_name = name  # real device name, so job specs match the live backend

    backend = SnapshotBackend()
    backend.snapshot_folder = folder  # marks it as offline
    return backend


def offline_backend(name: str) -> FakeBackendV2:
    """Latest saved snapshot of a project device, by real backend name."""
    if name.lower() in FORBIDDEN_DEVICES:
        raise ValueError(f"{name} must not be used in this project")
    return load_snapshot(latest_snapshot(name))
