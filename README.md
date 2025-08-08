# tkgc_trans
时序知识图谱补全
## 数据集
### YAGO11k
### Wikidata12k
### ICEWS14 
## 实体链接预测
### TE-TransT 

### TE-TransR


## 关系预测

### TE-TransT 

### TE-TransR

## 使用
启动命令,例如
```bash
python train_transr_r_addTime_tp2id.py -data_type  wiki_data -dataset_path path  -neg_sample 20 -gpu 3 -lr 0.01 -dim 128 -margin 10 -epoch 300 -batch 2000  -data_process tp_point1 -filter 1 
```
