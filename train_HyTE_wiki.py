'''
Author: cmp
Date: 2024-05-11 11:49:52
Description: 
1、修改了数据集的处理方式，之前划分时间段时，多了（max(time),3000）这段，现在去掉了


'''
import time
import trans
from trans.config import Trainer_AddTime, Tester_AddTime
from trans.module.model import TransH2HyTE
from trans.module.loss import MarginLoss
from trans.module.strategy import NegativeSampling
from trans.data import TrainDataLoader, TestDataLoader
from trans.data.PyTorchTrainDataLoader_AddTime_tp2id import PyTorchTrainDataLoader
from trans.data.PyTorchTestDataLoader_AddTime_tp2id import PyTorchTestDataLoader

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
start_time = time.time()
# dataloader for training
train_dataloader = PyTorchTrainDataLoader(
	# in_path = "./benchmarks/data/wiki_data_addTime_year_tp2id_HyTE_process/", 
    in_path = "./benchmarks/data/yago_data_addTime_year_tp2id_HyTE_process/", 
	batch_size = 50000,
	threads = 8, 
	sampling_mode = "normal", 
	sampling_choice="1", # 为None是表示用TransH的负取样方法
	bern_flag = 1, 
	filter_flag = 1, 

	neg_ent = 10,
	neg_rel = 0)


# define the model
Hyte = TransH2HyTE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	tp_tot=train_dataloader.get_tp_tot(),
	dim = 128, 	
	p_norm = 1, 
	norm_flag = True)

# define the loss function
model = NegativeSampling(
	model = Hyte, 
	loss = MarginLoss(margin = 10),
	batch_size = train_dataloader.get_batch_size()
)


# train the model
trainer = Trainer_AddTime(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 0.1, use_gpu = True)
trainer.run()
# Hyte.save_checkpoint('./checkpoint_20250801_hyte/Hyte_year_tp2id_wiki_HyTE_process_data_128_10_0.1.ckpt')
Hyte.save_checkpoint('./checkpoint_20250801_hyte/Hyte_year_tp2id_yago_HyTE_process_data_128_10_0.1.ckpt')
# test the model
# Hyte.load_checkpoint('./checkpoint/Hyte_year_tp2id_wiki_HyTE_process_data_128_10_0.1_20240813.ckpt')

end_time = time.time()
elapsed_time = end_time - start_time

print(f"程序训练完毕运行耗时：{elapsed_time:.4f} 秒")


test_dataloader = PyTorchTestDataLoader(
	# in_path = "./benchmarks/data/wiki_data_addTime_year_tp2id_HyTE_process/", 
	in_path = "./benchmarks/data/yago_data_addTime_year_tp2id_HyTE_process/", 
	threads = 8, 
	sampling_mode = "normal", 
	filter_flag = 1, 

	ent_tot = Hyte.ent_tot,
	rel_tot = Hyte.rel_tot,
	tp_tot= Hyte.tp_tot
	
 )
tester = Tester_AddTime(model = Hyte, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction()

end_time1 = time.time()
elapsed_time = end_time1 - start_time

print(f"程序总共运行耗时：{elapsed_time:.4f} 秒")