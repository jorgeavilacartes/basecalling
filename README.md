# RNA Basecaller for ONT data

### Create and activate environmnet
```bash
micromamba env create -n basecalling -f envs/basecalling.yml
micromamba activate basecalling
```

### Training
```bash
python feito/train.py --path-train data/subsample_train.hdf5 --path-val data/subsample_val.hdf5 --model Rodan --epochs 5 --batch-size 16
```
___ 
### Code structure
```
basecalling
├── feito: source code
├── envs: yaml files with different environments (different versions of pytorch and cuda)
├── data: store data here
├── notebooks: jupyter notebooks to test code
├── output: results of training
├── params.yml: input parameters for train with DVC
└── README.md
```


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
#### Basecalling
To map the output of the model to an RNA sequence, use beam search to decode the output of the neural network https://github.com/nanoporetech/fast-ctc-decode

#### Computation of accuracy
To compare the basecalled read against the ground truth read, use Smith Waterman 

#### Connect to a GPU in the server
```
qrsh -l gpu_mem=8G
```
___ 
Steps for basecall signals for ONT

1. Generate dataset for training a supervised model
    - Split raw signals in chunks of a fixed size (RODAN uses 4096-long signals)
    - Basecall the to obtain a Ground Truth (RODAN basecalle)