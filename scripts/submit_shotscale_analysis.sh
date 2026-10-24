#!/bin/bash
# Addendum B2 analysis: subsample pooled arm-E shots to 0.1M, 0.3M, 1M, 3M (seeds 7, 8, 9) and use all
# 10M, at d = 1.44M and 4M, with the frozen 52-qubit settings. One Slurm job per run.
# Usage: bash scripts/submit_shotscale_analysis.sh results/hardware/shotscale_<time> [partition]
set -euo pipefail
RUN=$1; PART=${2:-general}
for grid in 1.0 2.2; do
  src=$RUN/pooled_phoenix52_square_G$grid.npz
  for d in 1200 2000; do
    for n in 100000 300000 1000000 3000000; do
      for seed in 7 8 9; do
        sbatch --parsable --partition=$PART --cpus-per-task=32 --mem=200G --time=48:00:00 --job-name=ss-G$grid-n$n-s$seed-d$d \
          scripts/slurm_job.sh scripts/analyze_ccpvdz.py --grid $grid --source $src --max-dim $d --subsample $n --subsample-seed $seed
      done
    done
    sbatch --parsable --partition=$PART --cpus-per-task=64 --mem=400G --time=48:00:00 --job-name=ss-G$grid-full-d$d \
      scripts/slurm_job.sh scripts/analyze_ccpvdz.py --grid $grid --source $src --max-dim $d
  done
done
