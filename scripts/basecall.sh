#!/bin/bash
specie="mouse" # poplar
python feito/basecall.py \
--path-fast5 /projects5/basecalling-jorge/RODAN/data \
--len-subsignals 4096 \
--batch-size 32 \
--model Rodan \
--device cuda \
--path-checkpoint /projects5/basecalling-jorge/RODAN/rodan-checkpoint.pt \
--path-fasta output-debug/basecalling/$specie-basecalled_reads.fa \
--path-reads output-debug/basecalling/$specie-full_reads.fa \
--path-index output-debug/basecalling/$specie-index.csv 2> basecall2.log

# --path-fast5 data/RODAN/test/$specie-dataset/0 \
#--path-checkpoint output-rodan/training/checkpoints/Rodan-epoch29.pt \