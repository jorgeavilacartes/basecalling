#!/bin/bash
specie=$1
# specie="mouse" # poplar
echo $specie

dirout="output-rodan-smoothctc-ranger"
mkdir -p $dirout/basecalling/logs

/usr/bin/time -v python feito/basecall.py \
--path-fast5 data/RODAN/test/$specie-dataset \
--len-subsignals 4096 \
--batch-size 32 \
--model Rodan \
--path-checkpoint $dirout/training/checkpoints/Rodan-epoch28.pt \
--device cuda \
--path-fasta $dirout/basecalling/$specie-basecalled_reads.fa \
--path-reads $dirout/basecalling/$specie-full_reads.fa \
--path-index $dirout/basecalling/$specie-index.csv \
> $dirout/basecalling/logs/basecall-$specie.out.log \
2> $dirout/basecalling/logs/basecall-$specie.err.log 

# --path-fast5 data/RODAN/test/$specie-dataset/0 \
# --path-checkpoint /projects5/basecalling-jorge/RODAN/rodan-checkpoint.pt \
# /projects5/basecalling-jorge/basecalling/output-rodan/basecalling/test
# --path-checkpoint output-rodan/training/checkpoints/Rodan-epoch29.pt \
# --path-checkpoint /projects5/basecalling-jorge/RODAN/rodan-checkpoint.pt \