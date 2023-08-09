configfile: "rules/params/test_basecaller.yml"
from os.path import join as pjoin

SPECIES=config["SPECIES"]
PATH_TRANSCRIPTOMES=config["PATH_TRANSCRIPTOMES"]
PATH_RAW_READS=config["PATH_RAW_READS"]
PATH_OUTPUT=config["PATH_OUTPUT"]

rule all: 
    input: 
        # expand(pjoin( PATH_OUTPUT, "test", "{specie}.sam"), specie=SPECIES),
        expand(pjoin( PATH_OUTPUT, "test", "{specie}.bam.bai"), specie=SPECIES),
        expand(pjoin( PATH_OUTPUT, "test", "{specie}-stats_mapping.txt"), specie=SPECIES)

rule minimap:
    input: 
        transcriptome=pjoin( PATH_TRANSCRIPTOMES, "{specie}_reference.fasta"),
        # reads=pjoin( PATH_OUTPUT, "{specie}-full_reads.fa"),
        reads=pjoin( PATH_OUTPUT, "basecalled_reads.fa")
    output:
        sam=pjoin( PATH_OUTPUT, "test", "{specie}.sam")
    conda:
        "../envs/minimap2.yml"
    shell:
        "minimap2 --secondary=no -ax map-ont -t 32 --cs {input.transcriptome} {input.reads} > {output.sam}"


rule sort:
    input:
        sam=pjoin( PATH_OUTPUT, "test", "{specie}.sam")
    output:
        bam=pjoin( PATH_OUTPUT, "test", "{specie}.bam")
    conda:
        "../envs/minimap2.yml"
    shell:
        "samtools view -bS {input.sam} | samtools sort > {output.bam}"

rule index:
    input:
        bam=pjoin( PATH_OUTPUT, "test", "{specie}.bam")
    output:
        bai=pjoin( PATH_OUTPUT, "test", "{specie}.bam.bai")
    conda:
        "../envs/minimap2.yml"
    shell:
        "samtools index {input.bam}" 

rule stats_mapping:
    input:
        bam=pjoin( PATH_OUTPUT, "test", "{specie}.bam"),
        bai=pjoin( PATH_OUTPUT, "test", "{specie}.bam.bai")
    output:
        stats=pjoin( PATH_OUTPUT, "test", "{specie}-stats_mapping.txt")
    shell:
        "samtools flagstat {input.bam} > {output.stats}"
# visualize alignment
# ```bash
# samtools view $bamfile | less -S
# ```