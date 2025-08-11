# tkgc_trans
Temporal Knowledge Graph Completion (TKGC)
## Dataset
### YAGO11k
### Wikidata12k
### ICEWS14 
## Temporal Entity Link Prediction
### TE-TransT 

### TE-TransR


## Temporal Relation Prediction

### TE-TransT 

### TE-TransR

## Usage

```bash
python train_transr_r_addTime_tp2id.py -data_type  wiki_data -dataset_path path  -neg_sample 20 -gpu 3 -lr 0.01 -dim 128 -margin 10 -epoch 300 -batch 2000  -data_process tp_point1 -filter 1 
```
