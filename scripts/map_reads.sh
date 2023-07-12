# !/usr/bin/bash

transcriptome="/projects5/basecalling-jorge/basecalling/data/RODAN/test/transcriptomes/mouse_reference.fasta"
reads="/projects5/basecalling-jorge/basecalling/output-old/basecalling/simplenet-basecalled_reads.fa"
outputsam="output-old/basecalling/mapped_reads.sam"
minimap2 --secondary=no -ax map-ont -t 32 --cs $transcriptome $reads > $outputsam
