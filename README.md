# RNA Basecaller for ONT data

### Create and activate environmnet
```bash
micromamba env create -n basecalling -f envs/basecalling.yml
micromamba activate basecalling
```

### Training
```bash
python feito/train.py --path-train <path-to-hdf5> --path-val <path-to-hdf5> --model Rodan --epochs 5 --batch-size 16 
```

#### Basecalling
use beam search to decode the output of the neural network https://github.com/nanoporetech/fast-ctc-decode

___ 
Steps for basecall signals for ONT

1. Generate dataset for training a supervised model
    - Split raw signals in chunks of a fixed size (RODAN uses 4096-long signals)
    - Basecall the to obtain a Ground Truth (RODAN basecalle)