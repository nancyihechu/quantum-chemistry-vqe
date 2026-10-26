#!/bin/bash
# Addendum C analysis: kingston arm G at grid 2.2, 1M shots, subsampled to 0.1M and 0.3M
# (seeds 7, 8, 9) and in full, at d = 1.44M and 4M.
# Usage: scripts/submit_kingstonscale_analysis.sh results/hardware/kingstonscale_<time> [partition]
set -euo pipefail
RUN=$1; PART=${2:-general}
src=$RUN/kingstonscale_kingston52__kingston52_heavyhex_G2.2.npz
for d in 1200 2000; do
  for n in 100000 300000; do
    for seed in 7 8 9; do
      sbatch --parsable --partition=$PART --cpus-per-task=32 --mem=200G --time=48:00:00 --job-name=ks-n$n-s$seed-d$d \
        scripts/slurm_job.sh scripts/analyze_ccpvdz.py --grid 2.2 --source $src --max-dim $d --subsample $n --subsample-seed $seed
    done
  done
  sbatch --parsable --partition=$PART --cpus-per-task=32 --mem=200G --time=48:00:00 --job-name=ks-full-d$d \
    scripts/slurm_job.sh scripts/analyze_ccpvdz.py --grid 2.2 --source $src --max-dim $d
done
