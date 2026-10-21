#!/bin/bash
# Reference data for test and OOD sets, then the coarse-solver baselines.
set -e
cd "$(dirname "$0")/.."
OUT=${OUT:-data}
for split in test ood; do
  python3 scripts/generate.py --split $split --batch 6 --threads 2 --out $OUT
done
for d in 8 10 12 16; do
  for split in test ood; do
    python3 scripts/generate.py --split $split --batch 6 --threads 2 --d $d --out $OUT
  done
done
