'''
预测时间，训练时，负样本生成应该是时间的负样本，跟头尾实体没关系，因而在进行数据加载时，实体、关系的负样本生成个数为0,,设置时间的生成样本。
测试时，调用的测试方法和按年查分 计算实体在时间上平均值的函数一样，里面根据预测任务字段 predict_task=time  调用时间预测的计算方法
'''
import argparse
from utils.file_util import ensure_directory_exists
from trans.module.model import TransE_AddTime
import trans
from trans.config import Trainer_AddTime
from trans.config.Tester_AddTime_tp_point import Tester_AddTime
from trans.module.model.TransR_AddTime import  TransR_AddTime
from trans.module.loss import MarginLoss
from trans.module.strategy import NegativeSampling
from trans.data import TrainDataLoader, TestDataLoader

from trans.data.PyTorchTrainDataLoader_AddTime_tp2id import PyTorchTrainDataLoader
from trans.data.PyTorchTestDataLoader_AddTime_tp2id_tp_point import PyTorchTestDataLoader
import os 

parser = argparse.ArgumentParser(description='tranr_r_time_predict')

parser.add_argument('-data_type', dest= "data_type", default ='wiki_data', choices = ['yago','wiki_data',"icews05-15","icews14"], help ='dataset to choose')
parser.add_argument('-dataset_path',dest = 'dataset_path', default = './benchmarks/data/wiki_data_addTime_year_tp2id_filter1/',help = 'dataset_path')
parser.add_argument('-neg_sample', 	 dest="M", 		default = 10,   	type=int, 	help='Batch size')
parser.add_argument('-gpu', 	 dest="gpu", 		default='0',			help='GPU to use')
parser.add_argument('-lr',	 dest="lr", 		default=0.0001,  type=float,	help='Learning rate')
parser.add_argument('-dim',  dest="dim", 	default = 128,   	type=int, 	help='dimension')

parser.add_argument('-margin', 	 dest="margin", 	default=4,   	type=float, 	help='margin')
parser.add_argument('-batch', 	 dest="batch_size", 	default= 2000,   	type=int, 	help='Batch size')
parser.add_argument('-epoch', 	 dest="max_epochs", 	default= 100,   	type=int, 	help='Max epochs')
parser.add_argument('-data_process', 	 dest="data_process", 	default= 'time_predict',choices = ["HyTE","tp_point","tp_point1","tp_point2","filter"],   	help='data process means')
parser.add_argument('-save_steps', 	 dest="save_steps", 	default= 25,   type=int,	help='save_steps')

# parser.add_argument('-checkpoint', 	 dest="checkpoint", 	default= "./checkpoint/current.ckpt",  	help='Max epochs')
# parser.add_argument('-transe_checkpoint', 	 dest="transe_checkpoint", default= "./checkpoint_transe_icews05-15_512/train_transr_r_icews05_15_512_6_0.05.ckpt",	help='checkpoint')
# parser.add_argument('-filter', 	 dest="filter", 	default= 1,   type=int,	help='test measure filter')
args = parser.parse_args()
args.checkpoint = "./checkpoint_"+args.data_process + "/transr_r_time_predict/"+args.data_type+"/"+str(args.dim)+"_"+str(args.margin)+"_"+str(args.lr)+"/train"+".ckpt"
# args.transe_checkpoint = "./checkpoint_"+args.data_process + "/transe_"+args.data_type+"/"+str(args.dim)+".ckpt"
# train_dataset_path = args.dataset_path+"_test_tp_point"

print("param",args,"\n")
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 检验checkpoint和transe_checkpoint所在的文件夹是否存在，如果不存在则创建
# ensure_directory_exists(args.transe_checkpoint)
ensure_directory_exists(args.checkpoint)

# dataloader for training
train_dataloader = PyTorchTrainDataLoader(
    in_path = args.dataset_path, 
	batch_size = args.batch_size,
	threads = 1, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	sampling_choice= 1,
	neg_ent = 0,
	neg_rel = 0,
 	neg_time=args.M)

# dataloader for valid
valid_dataloader = PyTorchTrainDataLoader(
    # in_path = str(args.dataset_path), 
    train_file=args.dataset_path+"valid.txt",
    ent_file=args.dataset_path+"entity2id.txt",
    rel_file=args.dataset_path+"relation2id.txt",
    tp_file=args.dataset_path+"tp2id.txt",
	# nbatches = 300,
	batch_size = args.batch_size,
	threads = 6, 
	sampling_mode = "normal", 
 	sampling_choice= 1,
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 0,
	neg_rel = 0,
 	neg_time=args.M)

# dataloader for test for valid 
valid_dataloader_for_test = PyTorchTestDataLoader(

	tri_file=args.dataset_path+"triple2id.txt",
    test_tri_file=args.dataset_path+"valid.txt",
    
	threads = 6, 
	sampling_mode = "normal", 
	predict_task="time",
	bern_flag = 0, 
 
 	# 时间测试时，不需要过滤
	# filter_flag = args.filter, 

	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	tp_tot = train_dataloader.get_tp_tot()
	
 )


transr = TransR_AddTime(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
 	tp_tot = train_dataloader.get_tp_tot(),
	
	dim_e = int(args.dim),
	dim_r = int(args.dim),
	dim_tp = int(args.dim),
	p_norm = 1, 
	norm_flag = True,
	rand_init = False)

model_r = NegativeSampling(
	model = transr,
	loss = MarginLoss(margin = float(args.margin)),
	batch_size = train_dataloader.get_batch_size()
)

# pretrain transe
trainer = Trainer_AddTime(model = model_r, data_loader = train_dataloader, train_times = int(args.max_epochs), alpha = float(args.lr),
                          use_gpu = True
                        #   ,save_steps=25, checkpoint_dir = str(args.checkpoint).split(".ckpt")[0], valid_data_loader_for_loss = valid_dataloader,valid_data_loader_for_test=valid_dataloader_for_test
                          )
train_tester = Tester_AddTime(model = transr, data_loader = valid_dataloader_for_test, use_gpu = True)

trainer.run(tester = train_tester,save_steps=25, checkpoint_dir = str(args.checkpoint).split(".ckpt")[0],valid_data_loader_for_loss = valid_dataloader)


# trainer.run()
transr.save_checkpoint(str(args.checkpoint))
# transr.save_checkpoint('./checkpoint/train_transr_r_wiki_data_addTime_tp2id_year_180_10_0.05.ckpt')


# dataloader for test
test_dataloader = PyTorchTestDataLoader(
    # in_path = "./benchmarks/data/wiki_data_addTime_year_tp2id_HyTE_process/", 
    
	in_path = args.dataset_path,
	threads = 6, 
	sampling_mode = "normal", 
 	predict_task="time",
	bern_flag = 0, 
 	# 时间测试时，这里一定为0 
	# filter_flag = args.filter, 
	
	ent_tot = transr.ent_tot,
	rel_tot = transr.rel_tot,
 	tp_tot = train_dataloader.get_tp_tot()
 )
print("test……")
# test the model
transr.load_checkpoint(str(args.checkpoint))
tester = Tester_AddTime(model = transr, data_loader = test_dataloader, use_gpu = True,constraint_file = args.dataset_path+"r_time_static.txt")

tester.run_link_prediction()
