#!/bin/bash

# python feito/accuracy.py output-rodan/basecalling/test/yeast.sam data/RODAN/test/transcriptomes/yeast_reference.fasta > acc-yeast.out.log 2> acc-yeast.err.log

python feito/debug.py \
--path-set data/RODAN/train/rna-train.hdf5 \
--model Rodan \
--device cuda \
--checkpoint output-rodan-orig/training/checkpoints/rodan-checkpoint.pt