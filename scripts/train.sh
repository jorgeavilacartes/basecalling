#!/bin/bash
OUTDIR="output-rodan-smoothctc-ranger-gnoise"

mkdir -p $OUTDIR/logs

# --path-train data/subsample_train.hdf5 \
# --path-val data/subsample_val.hdf5 \

/usr/bin/time -v python feito/train.py \
--path-train data/RODAN/train/rna-train.hdf5 \
--path-val data/RODAN/train/rna-valid.hdf5 \
--epochs 15 \
--batch-size 32 \
--num-workers 4 \
--augmentation gaussian_noise \
--model Rodan \
--device cuda \
--path-weights output-rodan-smoothctc-momentum/training/checkpoints/Rodan-epoch30.pt \
--outfile-train-logger $OUTDIR/training/metrics.csv \
--dirpath-checkpoint $OUTDIR/training/checkpoints > $OUTDIR/logs/train.std.log 2> $OUTDIR/logs/train.err.log