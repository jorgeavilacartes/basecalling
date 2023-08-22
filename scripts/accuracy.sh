#!/bin/bash
output="output-rodan-orig"
specie=$1
echo $specie
# specie="mouse" # poplar

samfile="$output/basecalling/test/$specie.sam"
genomefile="data/RODAN/test/transcriptomes/${specie}_reference.fasta"
/usr/bin/time -v python feito/accuracy.py $samfile $genomefile \
--path-save $output/basecalling/test/$specie-accuracy.json > $output/logs/accuracy_$specie.out.log 2> $output/logs/accuracy_$specie.err.log