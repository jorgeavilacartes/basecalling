# RNA Basecaller for ONT data

### environmnet
```bash
micromamba env create -n basecalling -f envs/basecalling.yml
micromamba activate basecalling
```

#### Basecalling
use beam search to decode the output of the neural network https://github.com/nanoporetech/fast-ctc-decode