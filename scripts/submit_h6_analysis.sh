#!/bin/bash
# H6 analyses (52 qubits) not covered by the per-repeat runs (Deviation 5):
#  vsub    each arm (E, F, G; every repeat) subsampled to T's number of valid shots, compared with T;
#  sub     each arm subsampled to T's total shot count (98,304), compared with T;
#  random  both random baselines at the arms' shot counts (1.04M for E and F, 1e5 for G).
# Usage: scripts/submit_h6_analysis.sh "<study run dir> ..." [partition] [parts, default "vsub sub random"]
set -euo pipefail
RUNS=$1; PART=${2:-general}; PARTS=${3:-vsub sub random}
PY=/cluster/pixstor/dkhf3-lab/users/nancy/envs/qchem-sqd/bin/python
for grid in 1.0 1.5 2.2; do
  read nvalid nshots < <($PY -c "import json; r = json.load(open('results/ccpvdz/sqd_torino_R${grid}_d1200.json')); print(round(r['shots'] * r['valid_fraction']), r['shots'])")
  for d in 1200 2000; do
    for run in $RUNS; do
      for src in $run/{kingston52,phoenix52}_rep*__*_G$grid.npz; do
        if [[ " $PARTS " == *" vsub "* ]]; then
          sbatch --parsable --partition=$PART --cpus-per-task=32 --mem=200G --time=24:00:00 --job-name=h6-vsub-G$grid-d$d \
            scripts/slurm_job.sh scripts/analyze_ccpvdz.py --grid $grid --source $src --max-dim $d --subsample-valid $nvalid
        fi
        if [[ " $PARTS " == *" sub "* ]]; then
          sbatch --parsable --partition=$PART --cpus-per-task=32 --mem=200G --time=24:00:00 --job-name=h6-sub-G$grid-d$d \
            scripts/slurm_job.sh scripts/analyze_ccpvdz.py --grid $grid --source $src --max-dim $d --subsample $nshots
        fi
      done
    done
    if [[ " $PARTS " == *" random "* ]]; then
      for rnd in random_hfinit random_particle; do
        for n in 100000 1040000; do
          sbatch --parsable --partition=$PART --cpus-per-task=32 --mem=200G --time=48:00:00 --job-name=h6-$rnd-n$n-G$grid-d$d \
            scripts/slurm_job.sh scripts/analyze_ccpvdz.py --grid $grid --source $rnd --shots $n --max-dim $d
        done
      done
    fi
  done
done
