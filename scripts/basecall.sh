#!/bin/bash
specie=$1
echo $specie
# specie="mouse" # poplar
/usr/bin/time -v python feito/basecall.py \
--path-fast5 data/RODAN/test/$specie-dataset \
--len-subsignals 4096 \
--batch-size 32 \
--model Rodan \
--device cuda \
--path-checkpoint /projects5/basecalling-jorge/RODAN/rodan-checkpoint.pt \
--path-fasta output-rodan-orig/basecalling/$specie-basecalled_reads.fa \
--path-reads output-rodan-orig/basecalling/$specie-full_reads.fa \
--path-index output-rodan-orig/basecalling/$specie-index.csv 2> logs/basecall-rodan-orig_$specie.err.log > logs/basecall-rodan-orig_$specie.out.log

# --path-fast5 data/RODAN/test/$specie-dataset/0 \
# --path-checkpoint /projects5/basecalling-jorge/RODAN/rodan-checkpoint.pt \
# /projects5/basecalling-jorge/basecalling/output-rodan/basecalling/test
# --path-checkpoint output-rodan/training/checkpoints/Rodan-epoch29.pt \