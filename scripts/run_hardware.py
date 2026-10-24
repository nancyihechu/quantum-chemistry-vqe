"""Plan, test offline, submit, and fetch the SQD hardware study (design in qchem_sqd.hardware).

Commands:
  plan    build every job on the saved device snapshots and print QPU estimates (offline)
  test    offline: ISA validity, plus noiseless SQD through the full analysis on one PUB per arm
  submit  rebuild against the live devices and submit; refuses without --approved
  fetch   download raw shots and execution-time calibrations for a submitted manifest
  calibrate  pilot manifest -> phoenix 52-qubit study shots at matched IBM-reported usage per circuit

  --what pilot   32 qubits: bond lengths 1.0 and 2.4 A, one repeat, all arms; plus one 52-qubit
                 N2/cc-pVDZ circuit per device (grid 1.0, matched QPU time). Excluded from the main analysis.
  --what study   32 qubits: four bond lengths, all arms; 52 qubits: grid 1.0/1.5/2.2, three arms
                 (kingston 1e5 shots, phoenix shots matched to the same QPU time); --repeats selects which of the 5 repeats to
                 build or submit (the pre-registration splits them over at least 2 days)

  --what shotscale  Addendum B2: arm E (phoenix square, 52 qubits) at grid 1.0 and 2.2, 5 jobs x 2M shots

Circuits are rebuilt from results/hardware/design_frozen.json (scripts/freeze_design.py).

Usage: python scripts/run_hardware.py plan --what study
       python scripts/run_hardware.py submit --what pilot --approved
       python scripts/run_hardware.py submit --what study --repeats 1 2 3 --approved   (day 1)
       python scripts/run_hardware.py fetch results/hardware/pilot_<time>/manifest.json
"""

from qchem_sqd.repro import pin_threads

pin_threads(1)  # builds must match the frozen design: CCSD convergence paths are thread-dependent

import argparse  # noqa: E402
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import qpy

from qchem_sqd.backends import offline_backend, save_snapshot, service_for
from qchem_sqd.ccpvdz_arms import DESIGN_FILE as CCPVDZ_DESIGN_FILE
from qchem_sqd.ccpvdz_arms import build_arms as build_ccpvdz_arms
from qchem_sqd.ccpvdz_arms import SHOTSCALE_GRIDS, CALIBRATION_FILE, calibrated_phoenix_shots, ccpvdz_jobs, shotscale_jobs
from qchem_sqd.ccpvdz_arms import pilot_jobs as ccpvdz_pilot_jobs
from qchem_sqd.hardware import (
    BOND_LENGTHS, DESIGN_FILE, PILOT_BOND_LENGTHS, REPEATS, JobSpec, build_arms, estimate_qpu_seconds, study_jobs,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
INSTANCES = {"pilot": "Quantum Computation and Cryptography", "study": "Ligand Based Drug Discovery",
             "shotscale": "Ligand Based Drug Discovery"}


def load_fci() -> dict:
    path = RESULTS / "references" / "n2_631g_16o_sym.json"
    return {rec["bond_length"]: rec["e_fci"] for rec in json.loads(path.read_text())} if path.exists() else {}


def build_jobs(what: str, instance: str, get_backend, shots: int, repeats, require_calibration: bool = False) -> list[JobSpec]:
    if not DESIGN_FILE.exists():
        raise SystemExit(f"Missing {DESIGN_FILE}; run scripts/freeze_design.py first.")
    if not CCPVDZ_DESIGN_FILE.exists():
        raise SystemExit(f"Missing {CCPVDZ_DESIGN_FILE}; run scripts/freeze_ccpvdz_design.py first.")
    design = json.loads(DESIGN_FILE.read_text())
    design52 = json.loads(CCPVDZ_DESIGN_FILE.read_text())
    phoenix, kingston = get_backend("ibm_phoenix"), get_backend("ibm_kingston")
    if what == "shotscale":  # Addendum B2: arm E only, grid 1.0 and 2.2, 5 jobs x 2M shots
        return shotscale_jobs(build_ccpvdz_arms(phoenix, kingston, design52, grids=SHOTSCALE_GRIDS), instance)
    bond_lengths = PILOT_BOND_LENGTHS if what == "pilot" else BOND_LENGTHS
    arms = build_arms(phoenix, kingston, design, bond_lengths, fci=load_fci())
    if what == "pilot":
        arms52 = build_ccpvdz_arms(phoenix, kingston, design52, grids=(1.0,))
        return (study_jobs(arms, instance, repeats=[1], shots=shots, prefix="pilot_")
                + ccpvdz_pilot_jobs(arms52, instance, phoenix, kingston))
    arms52 = build_ccpvdz_arms(phoenix, kingston, design52)
    phoenix_shots = None
    if CALIBRATION_FILE.exists():
        phoenix_shots = json.loads(CALIBRATION_FILE.read_text())["phoenix_shots"]
    elif require_calibration:
        raise SystemExit(f"Missing {CALIBRATION_FILE}: run 'calibrate' on the pilot manifest first (pre-registered rule).")
    else:
        print("note: phoenix 52-qubit shots from the duration model; the study uses the pilot calibration")
    return (study_jobs(arms, instance, repeats=repeats, shots=shots)
            + ccpvdz_jobs(arms52, instance, repeats, phoenix, kingston, phoenix_shots=phoenix_shots))


def print_estimates(jobs: list[JobSpec], get_backend) -> float:
    total = 0.0
    print(f"{'job':16s} {'backend':13s} {'pubs':>4s} {'executions':>11s} {'rep delay':>10s} {'longest circ':>12s} {'QPU est':>8s}")
    for job in jobs:
        est = estimate_qpu_seconds(job, get_backend(job.backend_name))
        total += est["qpu_seconds"]
        print(f"{job.name:16s} {job.backend_name:13s} {est['pubs']:4d} {est['executions']:11,d} "
              f"{est['rep_delay_us']:8.0f}us {est['longest_circuit_us']:10.1f}us {est['qpu_seconds']:7.0f}s")
    return total


def check_isa(pub, backend) -> int:
    target, bad = backend.target, 0
    for inst in pub.circuit.data:
        if inst.operation.name in ("barrier", "measure"):
            continue
        qargs = tuple(pub.circuit.find_bit(q).index for q in inst.qubits)
        bad += not target.instruction_supported(inst.operation.name, qargs)
    return bad


def submit(args, get_offline) -> None:
    if not args.approved:
        raise SystemExit("Refusing to submit: pass --approved only after the user has approved this exact plan.")
    instance = args.instance or INSTANCES[args.what]
    service = service_for(instance)
    live = {}
    get_live = lambda name: live.setdefault(name, service.backend(name))
    if args.what == "study" and not args.repeats:
        raise SystemExit("For the study, pass --repeats explicitly (e.g. 1 2 3 on day 1, 4 5 on day 2).")
    jobs = build_jobs(args.what, instance, get_live, args.shots, args.repeats,
                      require_calibration=args.what == "study")  # rebuilt on today's calibration
    estimate = sum(estimate_qpu_seconds(j, get_live(j.backend_name))["qpu_seconds"] for j in jobs)
    remaining = service.usage().get("usage_remaining_seconds")
    print(f"Estimated {estimate:.0f}s; {instance!r} has {remaining}s remaining")
    if remaining is not None and estimate > remaining:
        raise SystemExit("Estimate exceeds remaining usage; not submitting.")

    from qiskit_ibm_runtime import Batch

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    run_dir = RESULTS / "hardware" / f"{args.what}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"what": args.what, "instance": instance, "submitted_utc": stamp, "estimate_s": estimate,
                "repeats": args.repeats if args.what == "study" else [args.what], "design_file": str(DESIGN_FILE),
                "snapshots_at_submit": {}, "jobs": []}
    for name in dict.fromkeys(j.backend_name for j in jobs):
        manifest["snapshots_at_submit"][name] = str(save_snapshot(get_live(name), run_dir / "calibration"))
    # Alternate devices job by job (phoenix rep1, kingston rep1, ...) inside one batch per device.
    batches = {name: Batch(backend=get_live(name)) for name in manifest["snapshots_at_submit"]}
    try:
        for job in jobs:
            runtime_job = job.sampler(batches[job.backend_name]).run([p.circuit for p in job.pubs])
            with open(run_dir / f"{job.name}_isa.qpy", "wb") as fh:
                qpy.dump([p.circuit for p in job.pubs], fh)
            with open(run_dir / f"{job.name}_logical.qpy", "wb") as fh:
                qpy.dump([p.logical for p in job.pubs], fh)
            manifest["jobs"].append({"name": job.name, "job_id": runtime_job.job_id(), "backend": job.backend_name,
                                     "options": job.options, "pubs": [{"label": p.label, **p.meta} for p in job.pubs]})
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
            print(f"submitted {job.name} -> {runtime_job.job_id()}")
    finally:
        for batch in batches.values():
            batch.close()  # no new jobs; submitted ones still run
    print(f"manifest: {run_dir / 'manifest.json'}")


def fetch(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    service = service_for(manifest["instance"])
    out_dir = manifest_path.parent
    for entry in manifest["jobs"]:
        job = service.job(entry["job_id"])
        status = str(job.status())
        print(f"{entry['name']:16s} {entry['job_id']}  {status}")
        if "DONE" not in status.upper():
            continue
        metrics = job.metrics()
        usage = metrics.get("usage") or {}
        entry["usage_seconds"] = float(job.usage())  # IBM-reported usage, the quantity the budget is charged
        entry["qpu_charge_time_seconds"] = usage.get("qpu_charge_time_seconds")
        entry["circuits_execution_time_ns"] = metrics.get("circuits_execution_time_ns")
        entry["timestamps"] = metrics.get("timestamps")
        ran_at = (metrics.get("timestamps") or {}).get("running")
        backend = service.backend(entry["backend"])
        props = backend.properties(datetime=datetime.fromisoformat(ran_at.replace("Z", "+00:00"))) if ran_at else backend.properties()
        (out_dir / "calibration").mkdir(exist_ok=True)
        (out_dir / "calibration" / f"{entry['name']}_props_at_run.json").write_text(json.dumps(props.to_dict(), default=str))
        for pub_meta, pub_result in zip(entry["pubs"], job.result()):
            bits = pub_result.data.meas
            np.savez_compressed(out_dir / f"{entry['name']}__{pub_meta['label']}.npz", array=bits.array, num_bits=bits.num_bits)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


def calibrate(manifest_path: Path) -> None:
    """Matched-QPU-time rule: phoenix study shots from the pilot's measured usage per shot."""
    manifest = json.loads(manifest_path.read_text())
    usage = {}
    for name, backend in (("pilot_phoenix52", "ibm_phoenix"), ("pilot_kingston52", "ibm_kingston")):
        entry = next(j for j in manifest["jobs"] if j["name"] == name)
        if entry.get("usage_seconds") is None:
            raise SystemExit(f"{name} has no usage yet; run fetch first")
        usage[backend] = (float(entry["usage_seconds"]), int(entry["options"]["default_shots"]))
    result = calibrated_phoenix_shots(usage)
    result["pilot_manifest"] = str(manifest_path)
    CALIBRATION_FILE.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["plan", "test", "submit", "fetch", "calibrate"])
    parser.add_argument("target", nargs="?", help="manifest.json for fetch or calibrate")
    parser.add_argument("--what", choices=["pilot", "study", "shotscale"], default="study")
    parser.add_argument("--instance", help="override the default instance for --what")
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, nargs="+", help="study repeats to build/submit (plan: all 5)")
    parser.add_argument("--approved", action="store_true", help="the user approved this exact submission")
    parser.add_argument("--full-baselines", action="store_true", help="test: also HCI and best-FCI baselines")
    parser.add_argument("--all-bond-lengths", action="store_true", help="test: analyze every bond length, not just the first")
    args = parser.parse_args()

    if args.command == "fetch":
        return fetch(Path(args.target))
    if args.command == "calibrate":
        return calibrate(Path(args.target))
    snapshots = {}
    get_offline = lambda name: snapshots.setdefault(name, offline_backend(name))
    if args.command == "submit":
        return submit(args, get_offline)
    jobs = build_jobs(args.what, args.instance or INSTANCES[args.what], get_offline, args.shots,
                      args.repeats or list(range(1, REPEATS + 1)))
    total = print_estimates(jobs, get_offline)
    print(f"\n{args.what}: estimated {total:.0f}s QPU on {args.instance or INSTANCES[args.what]!r}")
    if args.command == "test":
        print("\n== ISA validity ==")
        for job in jobs:
            bad = sum(check_isa(p, get_offline(job.backend_name)) for p in job.pubs)
            print(f"  {job.name:16s} {'OK' if not bad else f'{bad} unsupported operations'}")
        from analyze_sqd import analyze_offline  # noqa: E402  (scripts/ is on sys.path)
        if args.what == "shotscale":
            print("(52-qubit circuits cannot be simulated noiselessly; analysis is tested on hardware data instead)")
        else:
            analyze_offline(jobs, full=args.full_baselines, all_bond_lengths=args.all_bond_lengths)


if __name__ == "__main__":
    main()
