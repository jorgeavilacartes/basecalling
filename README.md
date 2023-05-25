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

with RODAN's dataset
```bash
python feito/train.py --path-train data/RODAN/train/rna-train.hdf5 --path-val data/RODAN/train/rna-valid.hdf5 --model Rodan --epochs 5 --batch-size 64 --device cuda
```

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


___
___
# TODO list
- [X] Callbacks:
    - [X] Checkpoint: save best model
    - [X] Early stopping
- [ ] Test model: compute accuracy of basecalled reads
    - [ ] use viterbi (and or beam search) to generate reads from output model
    - [ ] align basecalled read against ground truth with smith waterman
- [ ] Add viterbi and smith waterman to validation step
- [ ] Create own datasets from raw signals and a reference
- [ ] New architecture for RNA, consider sampling rate
