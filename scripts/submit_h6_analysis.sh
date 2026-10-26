#!/bin/bash
# H6 analyses (52 qubits) not covered by the per-repeat runs:
#  - each arm (E, F, G; every repeat) subsampled to T's number of valid shots, for the comparison with T;
#  - both random baselines at the arms' shot counts (1.04M for E and F, 1e5 for G).
# Usage: scripts/submit_h6_analysis.sh "<study run dir> ..." [partition]
set -euo pipefail
RUNS=$1; PART=${2:-general}
PY=/cluster/pixstor/dkhf3-lab/users/nancy/envs/qchem-sqd/bin/python
for grid in 1.0 1.5 2.2; do
  nvalid=$($PY -c "import json; r = json.load(open('results/ccpvdz/sqd_torino_R${grid}_d1200.json')); print(round(r['shots'] * r['valid_fraction']))")
  for d in 1200 2000; do
    for run in $RUNS; do
      for src in $run/{kingston52,phoenix52}_rep*__*_G$grid.npz; do
        sbatch --parsable --partition=$PART --cpus-per-task=32 --mem=200G --time=24:00:00 --job-name=h6-vsub-G$grid-d$d \
          scripts/slurm_job.sh scripts/analyze_ccpvdz.py --grid $grid --source $src --max-dim $d --subsample-valid $nvalid
      done
    done
    for rnd in random_hfinit random_particle; do
      for n in 100000 1040000; do
        sbatch --parsable --partition=$PART --cpus-per-task=32 --mem=200G --time=48:00:00 --job-name=h6-$rnd-n$n-G$grid-d$d \
          scripts/slurm_job.sh scripts/analyze_ccpvdz.py --grid $grid --source $rnd --shots $n --max-dim $d
      done
    done
  done
done
