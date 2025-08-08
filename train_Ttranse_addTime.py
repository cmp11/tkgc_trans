
import argparse
import time
from utils.file_util import ensure_directory_exists
from trans.module.model import TransE_AddTime
import trans
# from trans.config import Trainer, Tester
from trans.config import Trainer_AddTime,Tester_AddTime
from trans.config.Tester_AddTime_tp_point import Tester_AddTime as Tester_AddTime_mean
from trans.module.model import  TransR_AddTime
from trans.module.loss import MarginLoss
from trans.module.strategy import NegativeSampling
from trans.data import TrainDataLoader, TestDataLoader

from trans.data.PyTorchTrainDataLoader_AddTime_tp2id import PyTorchTrainDataLoader
from trans.data.PyTorchTestDataLoader_AddTime_tp2id import PyTorchTestDataLoader
from trans.data.PyTorchTestDataLoader_AddTime_tp2id_tp_point import PyTorchTestDataLoader as PyTorchTestDataLoader_quintuple

import os 
'''
往关系空间映射 在测试时，都只关心起始时间，想要测试所有时间上的平均得分的方法去对应的测试方法中测试模型
'''

parser = argparse.ArgumentParser(description='Ttrane')

parser.add_argument('-data_type', dest= "data_type", default ='mmbk_yago15k', choices = ['yago','wiki_data',"icews05-15","icews14","mmbk_wikidata","mmbk_yago15k","gdelt"], help ='dataset to choose')
parser.add_argument('-dataset_path',dest = 'dataset_path', default = '/u01/cmp/exp_code/OpenKE/benchmarks/data/mmbk_yago15k_tp_point2/',help = 'dataset_path')
parser.add_argument('-neg_sample', 	 dest="M", 		default = 20,   	type=int, 	help='Batch size')
parser.add_argument('-gpu', 	 dest="gpu", 		default='2',			help='GPU to use')
parser.add_argument('-lr',	 dest="lr", 		default=0.01,  type=float,	help='Learning rate')
parser.add_argument('-dim',  dest="dim", 	default = 200,   	type=int, 	help='dimension')

parser.add_argument('-margin', 	 dest="margin", 	default=10,   	type=float, 	help='margin')
parser.add_argument('-batch', 	 dest="batch_size", 	default= 1000,   	type=int, 	help='Batch size')
parser.add_argument('-epoch', 	 dest="max_epochs", 	default= 200,   	type=int, 	help='Max epochs')
parser.add_argument('-data_process', 	 dest="data_process", choices = ["tp_point","tp_point1","tp_point2","filter"],   	help='data process means')
parser.add_argument('-filter', 	 dest="filter", 	default= 1,   type=int,	help='test measure filter')
parser.add_argument('-save_steps', 	 dest="save_steps", 	default= 25,   type=int,	help='save_steps')

args = parser.parse_args()
# args.checkpoint = "./checkpoint_20250801"+args.data_process + "/Ttranse/"+args.data_type+"/"+str(args.dim)+"_"+str(args.margin)+"_"+str(args.lr)+"_"+str(args.M)+"/train"+".ckpt"
args.checkpoint = "./checkpoint_20250801_"+args.data_process + "/transe/"+args.data_type+"/"+str(args.dim)+".ckpt"


print("param",args,"\n")
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) 
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 检验checkpoint和transe_checkpoint所在的文件夹是否存在，如果不存在则创建
# ensure_directory_exists(args.transe_checkpoint)
ensure_directory_exists(args.checkpoint)

start_time = time.time()

# dataloader for training
train_dataloader = PyTorchTrainDataLoader(
    in_path = str(args.dataset_path), 
	# nbatches = 300,
	batch_size = args.batch_size,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1,
	sampling_choice= 1,
	neg_ent = int(args.M),
	neg_rel = 0)



# dataloader for test
test_dataloader = PyTorchTestDataLoader_quintuple(
    # in_path = "./benchmarks/data/wiki_data_addTime_year_tp2id_HyTE_process/", 
    in_path = str(args.dataset_path), 
	 
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = args.filter, 
	# neg_ent = transe.ent_tot*2,
	# neg_rel = 0,
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot()
	
 )

# define the model
transe = TransE_AddTime(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	tp_tot = train_dataloader.get_tp_tot(),

	dim = int(args.dim), 
	p_norm = 1, 
	norm_flag = True)

model_e = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = float(args.margin)),
	batch_size = train_dataloader.get_batch_size())



# pretrain transe
# transe.load_parameters(str(args.transe_checkpoint))
traine = Trainer_AddTime(model = model_e, data_loader = train_dataloader,train_times = 300, alpha = 0.05, use_gpu = True)
traine.run()


traine.save_checkpoint(str(args.checkpoint))

end_time = time.time()
elapsed_time = end_time - start_time

print(f"程序训练完毕运行耗时：{elapsed_time:.4f} 秒")

print("test……")
# # test the model
transe.load_checkpoint(str(args.checkpoint))
# transr.load_checkpoint("./checkpoint_transr_r_icews05-15_256/train_transr_r_icews05_15_256_6_0.05-24.ckpt")

tester = Tester_AddTime_mean(model = transe, data_loader = test_dataloader, use_gpu = True)

tester.run_link_prediction()

end_time1 = time.time()
elapsed_time = end_time1 - start_time

print(f"程序总共运行耗时：{elapsed_time:.4f} 秒")
