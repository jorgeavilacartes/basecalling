configfile: "rules/params/test_basecaller.yml"
from os.path import join as pjoin

SPECIES=config["SPECIES"]
PATH_TRANSCRIPTOMES=config["PATH_TRANSCRIPTOMES"]
PATH_RAW_READS=config["PATH_RAW_READS"]
PATH_OUTPUT=config["PATH_OUTPUT"]

rule all: 
    input: 
        expand(pjoin( PATH_OUTPUT, "{specie}-full_reads.fa"), specie=SPECIES)

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
        rna=config["RNA"],
        index=pjoin(PATH_OUTPUT, "{specie}-index.csv"),
        basecalled_reads=pjoin(PATH_OUTPUT, "{specie}-basecalled_reads.fa"),
        full_reads=pjoin(PATH_OUTPUT, "{specie}-full_reads.fa"),
    conda:
        "../envs/basecalling.yml"
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
        --path-reads {params.full_reads} \
        --rna {params.rna}
        """