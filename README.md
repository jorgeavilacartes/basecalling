# RNA Basecaller for ONT data

### Create and activate environmnet
```bash
micromamba env create -n basecalling-cuda117 -f envs/basecalling_cuda11.7_pytorch2.yml
micromamba activate basecalling-cuda117
```
use `conda`/`miniconda`/`mamba`/`micromamba`

### Training
for testing with small datasets
```bash
python feito/train.py --path-train data/subsample_train.hdf5 --path-val data/subsample_val.hdf5 --model Rodan --epochs 5 --batch-size 16
```

```bash
python3 feito/train.py --path-train data/RODAN/train/rna-train.hdf5 --path-val data/RODAN/train/rna-valid.hdf5 --epochs 30 --batch-size 16 --num-workers 4 --model SimpleNet --device cuda
```

with RODAN's dataset
```bash
python feito/train.py --path-train data/RODAN/train/rna-train.hdf5 --path-val data/RODAN/train/rna-valid.hdf5 --model Rodan --epochs 20 --batch-size 64 --device cuda
```

### Testing
- This test assumes that testing dataset is in the same format than training and validations (`hdf5`` format), i.e. you have split reads with their ground truths.
- For experimental purposes use `/extdata/RODAN/train/rna-test.hdf5`.

RODAN with small dataset
```bash
python feito/test.py --path-test data/subsample_val.hdf5 --batch-size 16 --model Rodan --device cpu --path-checkpoint output/training/checkpoints/Rodan-epoch5.pt --path-fasta output/test/basecalled_signals.fa --rna true --use-viterbi true
```

SimpleNet with small dataset
```bash
python feito/test.py --path-test data/subsample_val.hdf5 --batch-size 16 --model SimpleNet --device cpu --path-checkpoint output/training/checkpoints/SimpleNet-epoch1.pt --path-fasta output/test/basecalled_signals_SimpleNet.fa --rna true --use-viterbi true
```

### **Basecalling** 
- This assumes you have a trained model, and a set of reads in fast5 format. 
- Reads will be split by the dataloader in non-overlapping signals with length equal to the input of the model (this must be provided as parameter, but it shouldn't (FIXME:)), and an index will be created, to refer each portion of the basecalled signal to its portion of read.

```bash
python feito/feito.py --path-fast5 data/RODAN/test/mouse-dataset/0 --len-subsignals 4096 --path-index output/basecalling/simplenet-index.csv --batch-size 16 --model SimpleNet --device cpu --path-checkpoint output/training/checkpoints/SimpleNet-epoch30.pt --path-fasta output/basecalling/simplenet-basecalled_reads.fa --path-reads output/basecalling/simplenet-basecalled_reads.fa
```

### **Reconstruct full reads from basecalled signals**
Since raw signals need to be split into chunks of a fix length, 
we need to . For this reason, an index for portion of basecalled reads is built
during the previous step. 
Now we need to take those portions of reads plus the index and reconstruct each read
by concatenating the portions in the right order. 



___

# Mapping reads with minimap2

### OPTIONAL
Install minimap2 in a conda environment
```bash
micromamba env create -n map-reads -f envs/minimap2.yml
micromamba activate map-reads
```

map reads to transcriptome
```bash
transcriptome="/projects5/basecalling-jorge/basecalling/data/RODAN/test/transcriptomes/mouse_reference.fasta"
reads="/projects5/basecalling-jorge/basecalling/output-old/basecalling/simplenet-basecalled_reads.fa"
samfile="output-old/basecalling/mapped_reads.sam"
minimap2 --secondary=no -ax map-ont -t 32 --cs $transcriptome $reads > $outputsam
```

### `samtools`

sort mapped reads 
```bash
bamfile="output-old/basecalling/mapped_reads.bam"
samtools view -bS $samfile | samtools sort > $bamfile
```

indexing
```bash
samtools index $bamfile 
```

visualize alignment
```bash
samtools view $bamfile | less -S
```

check statistics of mapped/unnmaped reads
```
samtools flagstat $bamfile
```

___
# TODO list
- [X] Callbacks:
    - [X] Checkpoint: save best model
    - [X] Early stopping
- [X] Test model: compute accuracy of basecalled reads
    - [X] use viterbi (and or beam search) to generate reads from output model
    - [X] align basecalled read against ground truth with smith waterman
- [ ] Create own datasets from raw signals and a reference
- [ ] New architecture for RNA, consider sampling rate


___

### Info

**Basecalling**

To map the output of the model to an RNA sequence, use beam search to decode the output of the neural network https://github.com/nanoporetech/fast-ctc-decode

**Computation of accuracy**

To compare the basecalled read against the ground truth read, use Smith Waterman 

**Connect to a GPU in the server**
Avoid the usage of CPU10
```bash
qrsh -l gpu_mem=8G
```
___ 
Steps for basecall signals for ONT

1. Generate dataset for training a supervised model
    - Split raw signals in chunks of a fixed size (RODAN uses 4096-long signals)
    - Basecall the to obtain a Ground Truth (RODAN basecalle)

___
# TO CONSIDER

How do they influence the architectures?  


|          | sampling rate [samples/sec] |  [bp/sec]  |  [samples/bp]  |
|----------|:---------------------------:|-----------:|---------------:|
| DNA      |            4000             |     450    |       8.89     |
| RNA      |            3012             |      70    |      43.03     | 


**Path to datasets in the server** compbio
RODAN's dataset
- `/extdata/RODAN/train/rna-train.hdf5`
- `/extdata/RODAN/train/rna-test.hdf5`
- `/extdata/RODAN/test`

**Directory where I am working on**
`compbio:/projects5/basecalling-jorge/basecalling`
___ 
### Folder structure
```
basecalling
├── feito: source code
├── envs: yaml files with different environments that can be installed with conda/miniconda/mamba/micromamba (different versions of pytorch and cuda)
├── data: store data here
├── notebooks: jupyter notebooks to test code
├── output: results of training
├── params.yml: input parameters for train with DVC
└── README.md
```

**Source code** 
```
feito
├── callbacks: functions to be run after each epoch in the training
├── loss_functions: variants of CTCLoss
├── models: architectures and custom layers
├── utils: accuracy and others
├── basecaller_trainer.py: custom pipeline to train a network
├── dataloader.py: custom dataloader from hdf5 files
└── train.py: script to train a neural network
```