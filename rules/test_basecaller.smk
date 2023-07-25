configfile: "rules/params/test_basecaller.yml"
from os.path import join as pjoin

SPECIES = [
    "mouse",
    "yeast"
]

PATH_TRANSCRIPTOMES=config["PATH_TRANSCRIPTOMES"]
PATH_RAW_READS=config["PATH_RAW_READS"]
# PATH_READS=config["PATH_READS"]
PATH_OUTPUT=config["PATH_OUTPUT"]

rule all: 
    input: 
        # expand(pjoin( PATH_OUTPUT, "test", "{specie}.sam"), specie=SPECIES),
        expand(pjoin( PATH_OUTPUT, "test", "{specie}.bam.bai"), specie=SPECIES)

rule basecalling:
    input:
        pjoin( PATH_RAW_READS, "{specie}-dataset")
    output:
        pjoin( PATH_OUTPUT, "{specie}-full_reads.fa")
    params:
        batch_size=config["BATCH_SIZE"],
        model=config["MODEL"],
        device=config["DEVICE"],
        input_len=config["INPUT_LEN"],
        checkpoint=config["CHECKPOINT"],
        index=pjoin(PATH_OUTPUT, "{specie}-index.csv"),
        basecalled_reads=pjoin(PATH_OUTPUT, "{specie}-basecalled_reads.fa"),
        full_reads=pjoin(PATH_OUTPUT, "{specie}-full_reads.fa"),
    conda:
        "/home/javila/micromamba/envs/basecalling-cuda117"
    shell:
        """
        python feito/basecall.py \
        --path-fast5 {input} \
        --batch-size {params.batch_size} \
        --model {params.model} \
        --device {params.device} \
        --len-subsignals {params.input_len} \
        --path-checkpoint {params.checkpoint} \
        --path-index {params.index} \
        --path-fasta {params.basecalled_reads} \
        --path-reads {params.full_reads} 
        """

rule minimap:
    input: 
        transcriptome=pjoin( PATH_TRANSCRIPTOMES, "{specie}_reference.fasta"),
        reads=pjoin( PATH_OUTPUT, "{specie}-full_reads.fa"),
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

# visualize alignment
# ```bash
# samtools view $bamfile | less -S
# ```