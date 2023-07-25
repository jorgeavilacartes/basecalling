# !/usr/bin/bash

transcriptome="/projects5/basecalling-jorge/basecalling/data/RODAN/test/transcriptomes/mouse_reference.fasta"
reads="/projects5/basecalling-jorge/output-rodan/basecalling/basecalling/mouse-full_reads.fa"
outputsam="output-rodan/basecalling/mapped_reads.sam"
minimap2 --secondary=no -ax map-ont -t 32 --cs $transcriptome $reads > $outputsam
