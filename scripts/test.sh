#!/bin/bash

PATH_TEST="data/RODAN/train/rna-valid.hdf5"
MODEL="Rodan"
NUM_WORKERS=4
BATCH_SIZE=16
DEVICE="cuda"
PATH_CHECKPOINT="output-rodan/training/checkpoints/Rodan-epoch29.pt"
PATH_FASTA="output-rodan/test/basecalled_reads.fa"

python feito/test.py \
--path-test $PATH_TEST \
--num-workers $NUM_WORKERS \
--batch-size $BATCH_SIZE \
--model $MODEL \
--device $DEVICE \
--path-checkpoint $PATH_CHECKPOINT \
--path-fasta $PATH_FASTA 
# --rna 
# --use-viterbi
