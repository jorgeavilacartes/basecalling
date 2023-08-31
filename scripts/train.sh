#!/bin/bash
OUTDIR="output-rodan-smoothctc-momentum"

python feito/train.py \
--path-train data/RODAN/train/rna-train.hdf5 \
--path-val data/RODAN/train/rna-valid.hdf5 \
--epochs 30 \
--batch-size 32 \
--num-workers 4 \
--model Rodan \
--device cuda \
--path-weights output-rodan-smoothctc/training/checkpoints/Rodan-epoch23.pt \
--outfile-train-logger $OUTDIR/training/metrics.csv \
--dirpath-checkpoint $OUTDIR/training/checkpoints
