#!/bin/bash
OUTDIR="output-rodan-smoothctc-ranger-gnoise"

# --path-train data/RODAN/train/rna-train.hdf5 \
# --path-val data/RODAN/train/rna-valid.hdf5 \
python feito/train.py \
--path-train data/subsample_train.hdf5 \
--path-val data/subsample_val.hdf5 \
--epochs 15 \
--batch-size 32 \
--num-workers 4 \
--augmentation gaussian_noise flip \
--model Rodan \
--device cpu \
--path-weights output-rodan-smoothctc-momentum/training/checkpoints/Rodan-epoch30.pt \
--outfile-train-logger $OUTDIR/training/metrics.csv \
--dirpath-checkpoint $OUTDIR/training/checkpoints