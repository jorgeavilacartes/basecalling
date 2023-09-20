#!/bin/bash
OUTDIR="output-rodan-smoothctc-ranger"

python feito/train.py \
--path-train data/RODAN/train/rna-train.hdf5 \
--path-val data/RODAN/train/rna-valid.hdf5 \
--epochs 30 \
--batch-size 32 \
--num-workers 4 \
--model Rodan \
--device cuda \
--path-weights output-rodan-smoothctc-momentum/training/checkpoints/Rodan-epoch30.pt \
--outfile-train-logger $OUTDIR/training/metrics.csv \
--dirpath-checkpoint $OUTDIR/training/checkpoints