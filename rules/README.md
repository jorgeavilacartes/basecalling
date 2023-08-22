### `test_basecaller.smk`
modify `rules/params/test_basecaller.yml`:

```yaml

# Model
MODEL: "Rodan" # or "SimpleNet"
DEVICE: "cuda" # or "cpu"
CHECKPOINT: "path/to/pytorch_checkpoint.pt"
INPUT_LEN: 4096
BATCH_SIZE: 32
PATH_OUTPUT: "output-rodan-orig/basecalling" #"output-rodan/basecalling" #"output/basecalling"
RNA: False # alphabet. If False, then DNA alphabet is used
USE_VITERBI: False # signal to alphabet. If False, beam_search is used 

# Datasets
SPECIES: 
  - "mouse"
  - "yeast"
  - "human"
  - "poplar"
  - "arabidopsis"
PATH_TRANSCRIPTOMES: "data/RODAN/test/transcriptomes" # path where <specie>_reference.fasta are stored
PATH_RAW_READS: "data/RODAN/test" # path where <specie>-dataset/ are stored
```

### `basecall.smk`
# TODO: not working yet