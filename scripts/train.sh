#!/bin/bash
OUTDIR="output-rodan-smoothctc"

python feito/train.py --path-train data/RODAN/train/rna-train.hdf5 \
--path-val data/RODAN/train/rna-valid.hdf5 \
--model Rodan --epochs 30 \
--batch-size 32 \
--num-workers 4 \
--outfile-train-logger $OUTDIR/training/metrics.csv \
--dirpath-checkpoint $OUTDIR/training/checkpoints
