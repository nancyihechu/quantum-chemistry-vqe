#!/bin/bash
# Day-2 analyses (repeats 4-5), the same settings as day 1: 32q all arms, 52q every file at
# d = 1.44M and 4M, plus arm E subsampled to 100k shots (matched shots, H4).
# Usage: scripts/submit_day2_analysis.sh results/hardware/study_<time> [partition]
set -euo pipefail
RUN=$1; PART=${2:-general}
sbatch --parsable --partition=$PART --cpus-per-task=32 --mem=200G --time=48:00:00 --job-name=day2-all-32q \
  scripts/slurm_job.sh scripts/analyze_sqd.py $RUN/manifest.json --full-baselines --out analysis_day2.json
for src in $RUN/{kingston52,phoenix52}_rep*__*_G*.npz; do
  grid=${src##*_G}; grid=${grid%.npz}
  for d in 1200 2000; do
    sbatch --parsable --partition=$PART --cpus-per-task=32 --mem=200G --time=24:00:00 --job-name=day2-52q-G$grid-d$d \
      scripts/slurm_job.sh scripts/analyze_ccpvdz.py --grid $grid --source $src --max-dim $d
    if [[ $src == *phoenix52_square* ]]; then
      sbatch --parsable --partition=$PART --cpus-per-task=32 --mem=200G --time=24:00:00 --job-name=day2-52qsub-G$grid-d$d \
        scripts/slurm_job.sh scripts/analyze_ccpvdz.py --grid $grid --source $src --max-dim $d --subsample 100000
    fi
  done
done
